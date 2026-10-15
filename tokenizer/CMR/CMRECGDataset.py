import torch
import numpy as np
import os
import warnings
from monai import data, transforms
from monai.data import *
import glob
import math
from pathlib import Path
import json
import tempfile
from sklearn.model_selection import train_test_split
from monai.transforms import *
import pickle
from typing import List, Optional
import ecgmentations as E


warnings.filterwarnings("ignore")


class Ecgmentationd(MapTransform):
    def __init__(self, keys, aug, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.aug = aug

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            ecg = d[key]

            if isinstance(ecg, torch.Tensor):
                ecg_np = ecg.detach().cpu().numpy()
            else:
                ecg_np = np.asarray(ecg)

            if ecg_np.ndim != 2:
                raise ValueError(f"Ecgmentationd expects shape [n_leads, T], got {ecg_np.shape}")

            ecg_len_first = ecg_np.T

            auged = self.aug(ecg=ecg_len_first)
            ecg_aug = np.asarray(auged["ecg"])

            d[key] = ecg_aug.T

        return d


class ECGResizeToFixedLengthd(MapTransform):
    def __init__(
        self,
        keys,
        target_length: int = 500,
        random_crop: bool = True,
        fill_value: float = 0.0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.target_length = int(target_length)
        self.random_crop = random_crop
        self.fill_value = float(fill_value)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            ecg = d[key]

            if isinstance(ecg, torch.Tensor):
                ecg_np = ecg.detach().cpu().numpy()
            else:
                ecg_np = np.asarray(ecg)

            if ecg_np.ndim != 2:
                raise ValueError(f"ECGResizeToFixedLengthd expects shape [n_leads, T], got {ecg_np.shape}")

            n_leads, T = ecg_np.shape

            if T == self.target_length:
                fixed = ecg_np
            elif T > self.target_length:
                if self.random_crop:
                    max_start = T - self.target_length
                    start = int(np.random.randint(0, max_start + 1))
                else:
                    start = (T - self.target_length) // 2
                fixed = ecg_np[:, start:start + self.target_length]
            else:
                pad_width = self.target_length - T
                pad = np.full((n_leads, pad_width), self.fill_value, dtype=ecg_np.dtype)
                fixed = np.concatenate([ecg_np, pad], axis=1)
            d[key] = fixed

        return d


class ECGZNormd(MapTransform):
    def __init__(
        self,
        keys,
        per_lead: bool = True,
        eps: float = 1e-8,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.per_lead = per_lead
        self.eps = float(eps)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            ecg = d[key]

            if isinstance(ecg, torch.Tensor):
                ecg_np = ecg.detach().cpu().numpy()
            else:
                ecg_np = np.asarray(ecg, dtype=np.float32)

            if ecg_np.ndim != 2:
                raise ValueError(f"ECGZNormd expects shape [n_leads, T], got {ecg_np.shape}")

            if self.per_lead:
                mean = ecg_np.mean(axis=1, keepdims=True)
                std = ecg_np.std(axis=1, keepdims=True)
            else:
                mean = ecg_np.mean()
                std = ecg_np.std()

            std_safe = np.where(std < self.eps, 1.0, std)
            ecg_norm = (ecg_np - mean) / std_safe
            d[key] = ecg_norm

        return d


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_CMRECG(args):
    data_list = get_volume_UKBB(args.data_dir)
    split_file = getattr(args, "split_file", None) or Path(args.output_dir) / "ukbb_split.json"
    train_files, val_files, test_files = dataset_split(data_list, args.seed, split_file)
    print(f"UKBB train/val/test: {len(train_files)}/{len(val_files)}/{len(test_files)}; split: {split_file}")
    return {
        "data_name": "UKBB",
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "modality": "MRI+ECG",
    }


def _subject_id(record):
    for key in ("image", "latent_mri", "x_mri"):
        if key in record:
            return str(record[key]).replace("\\", "/").rsplit("/", 1)[-1].split("_")[0]
    raise ValueError("A UKBB record must contain image, latent_mri, or x_mri.")


def dataset_split(data_list, seed, split_file=None):
    """Split subjects 80/10/10, giving 3503/438/438 for the 4379-subject cohort."""
    grouped = {}
    for record in data_list:
        grouped.setdefault(_subject_id(record), []).append(record)
    subjects = sorted(grouped)
    if len(subjects) < 3:
        raise ValueError("UKBB train/validation/test splitting requires at least three matched subjects.")

    path = Path(split_file) if split_file else None
    if path is not None and path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("seed") != seed:
            raise ValueError(f"UKBB split seed differs from the saved manifest: {path}")
        partitions = [manifest.get(name, []) for name in ("train", "val", "test")]
        if any(not isinstance(part, list) or not part for part in partitions):
            raise ValueError(f"UKBB split manifest must contain nonempty train/val/test lists: {path}")
        assigned = [ident for part in partitions for ident in part]
        if len(set(assigned)) != len(assigned):
            raise ValueError(f"UKBB split manifest contains duplicate or overlapping subjects: {path}")
        if set(assigned) != set(subjects):
            missing = len(set(assigned) - set(subjects))
            extra = len(set(subjects) - set(assigned))
            raise ValueError(f"UKBB cohort differs from the saved split ({missing} missing, {extra} extra subjects). "
                             "Restore the matching data/latent files or use a separate split manifest.")
    else:
        holdout_size = max(2, math.ceil(0.2 * len(subjects)))
        train, holdout = train_test_split(subjects, test_size=holdout_size, random_state=seed)
        val, test = train_test_split(holdout, test_size=0.5, random_state=seed)
        partitions = [train, val, test]
        manifest = dict(seed=seed, train=train, val=val, test=test)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
                json.dump(manifest, stream, indent=2)
                temporary_path = stream.name
            os.replace(temporary_path, path)
    return tuple([record for ident in part for record in grouped[ident]] for part in partitions)


def get_volume_UKBB(data_dir):
    mr_pattern = os.path.join(data_dir, "MR", "*", "*_4D.nii.gz")
    ecg_pattern = os.path.join(data_dir, "ECG", "*", "*.npy")

    mr_paths = sorted(glob.glob(mr_pattern))
    ecg_paths = sorted(glob.glob(ecg_pattern))

    ecg_map = {}
    for ep in ecg_paths:
        ecg_fname = os.path.basename(ep)
        ecg_id = ecg_fname.split("_")[0]
        if ecg_id in ecg_map:
            warnings.warn(f"Duplicate ECG id {ecg_id}, keeping first one only.")
            continue
        ecg_map[ecg_id] = ep

    paired = []
    missing_ecg = 0

    for mp in mr_paths:
        mr_fname = os.path.basename(mp)
        mr_id = mr_fname.split("_")[0]
        ecg_path = ecg_map.get(mr_id, None)
        if ecg_path is None:
            missing_ecg += 1
            continue
        paired.append(
            {
                "image": mp,
                "ecg": ecg_path
            }
        )

    print(f"Dataset UKBB CMR+ECG (by subject id prefix): paired={len(paired)}, CMR-only (no ECG)={missing_ecg}")
    return paired


class TimeToSamplesd(MapTransform):
    def __init__(
        self,
        keys,
        num_samples: int = 4,
        strategy: str = "random",
        window_size: Optional[int] = None,
        with_replacement: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        assert strategy in {"random", "window", "all"}
        self.num_samples = num_samples
        self.strategy = strategy
        self.window_size = window_size
        self.with_replacement = with_replacement

    def _select_indices(self, T: int) -> List[int]:
        if self.strategy == "all":
            return list(range(T))
        if self.strategy == "random":
            k = min(self.num_samples, T)
            if self.with_replacement:
                return [int(torch.randint(0, T, (1,))) for _ in range(k)]
            return torch.randperm(T)[:k].tolist()
        w = self.window_size or self.num_samples
        w = min(w, T)
        if w == T:
            return list(range(T))
        start = int(torch.randint(0, T - w + 1, (1,)))
        return list(range(start, start + w))

    def __call__(self, data):
        d = dict(data)
        first_key = next(self.key_iterator(d))
        img0 = torch.as_tensor(d[first_key])
        if img0.ndim == 5:
            T = img0.shape[0]
        elif img0.ndim == 4:
            T = img0.shape[0]
        else:
            raise ValueError(f"Expect [T,H,W,D] or [T,1,H,W,D], got shape {tuple(img0.shape)}")

        idxs = self._select_indices(T)
        out_samples = []
        temporal_keys = list(self.key_iterator(d))
        other_keys = [k for k in d.keys() if k not in temporal_keys]

        for t in idxs:
            dd = {}
            for key in temporal_keys:
                x = torch.as_tensor(d[key])
                if x.ndim == 5:
                    x = x[t].contiguous()
                elif x.ndim == 4:
                    x = x[t:t+1].contiguous()
                dd[key] = x
            for key in other_keys:
                dd[key] = d[key]
            dd["t_index"] = torch.tensor(t, dtype=torch.int64)
            out_samples.append(dd)

        return out_samples


class LoadECGd(MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False, dtype=np.float32):
        super().__init__(keys, allow_missing_keys)
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            path = d[key]
            if isinstance(path, str) and path.endswith(".npy"):
                ecg = np.load(path).astype(self.dtype)
            else:
                ecg = np.asarray(path, dtype=self.dtype)
            d[key] = ecg
        return d


def get_transforms(args):
    ecg_augs = E.Sequential([
        E.RandomTimeWrap(p=0.2),
        E.TimeCutout(length_range=(0, min(50, args.ECG_length)), p=0.2),
        E.ChannelShuffle(p=0.1),
        E.AmplitudeScale(p=0.3),
        E.GaussNoise(p=0.3),
    ])

    common_transform = [
        LoadImaged(keys=['image'], image_only=True, allow_missing_keys=True),
        LoadECGd(keys=['ecg'], allow_missing_keys=True),
        EnsureChannelFirstd(keys=['image'], allow_missing_keys=True),
        ScaleIntensityRangePercentilesd(keys=['image'], lower=0.5, upper=99.5, b_min=0, b_max=1),
        SpatialPadd(keys=['image'], spatial_size=args.cardiac_pad, mode='edge', allow_missing_keys=True),
        CenterSpatialCropd(keys=['image'], roi_size=args.cardiac_size, allow_missing_keys=True),
    ]

    train_transform = Compose(
        common_transform
        + [
            RandFlipd(keys=["image"], prob=0.3, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=["image"], prob=0.3, spatial_axis=1, allow_missing_keys=True),
            RandFlipd(keys=["image"], prob=0.3, spatial_axis=2, allow_missing_keys=True),
            RandRotate90d(keys=["image"], prob=0.3, max_k=3, allow_missing_keys=True),
            RandScaleIntensityd(keys=["image"], prob=0.2, factors=(0.9, 1.1), allow_missing_keys=True),
            RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.05, allow_missing_keys=True),

            ECGResizeToFixedLengthd(keys=["ecg"], target_length=args.ECG_length, random_crop=True, allow_missing_keys=True),
            Ecgmentationd(keys=["ecg"], aug=ecg_augs, allow_missing_keys=True),
            ECGZNormd(keys=["ecg"], per_lead=True, allow_missing_keys=True),

            ToTensord(keys=["image", "ecg"], allow_missing_keys=True),
            EnsureTyped(keys=["image", "ecg"], allow_missing_keys=True, dtype=torch.float32),

            TimeToSamplesd(keys=["image"], num_samples=args.time_samples, strategy="all", allow_missing_keys=True),
        ])

    val_transform = Compose(
        common_transform
        + [
            ECGResizeToFixedLengthd(keys=["ecg"], target_length=args.ECG_length, random_crop=False, allow_missing_keys=True),
            ToTensord(keys=["image", "ecg"], allow_missing_keys=True),
            EnsureTyped(keys=["image", "ecg"], allow_missing_keys=True, dtype=torch.float32),
            TimeToSamplesd(keys=["image"], num_samples=args.time_samples, strategy="all", allow_missing_keys=True),
        ])

    return train_transform, val_transform


def cmr_ecg_collate(batch):
    images = []
    ecgs = []

    for sample_list in batch:
        ecg = sample_list[0]["ecg"]
        ecgs.append(ecg)

        for dd in sample_list:
            images.append(dd["image"])

    images = torch.stack(images, dim=0)
    ecgs   = torch.stack(ecgs, dim=0)

    return {"image": images, "ecg": ecgs}


def get_loader(args, train_list, val_list, train_transform, val_transform):
    if args.use_persistent_dataset:
        print('use persistent')
        train_ds = PersistentDataset(
            data=train_list,
            transform=train_transform,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            cache_dir=args.cache_dir
        )
    else:
        train_ds = data.Dataset(data=train_list, transform=train_transform)

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=cmr_ecg_collate,
    )

    if args.use_persistent_dataset:
        val_ds = PersistentDataset(
            data=val_list,
            transform=val_transform,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            cache_dir=args.cache_dir
        )
    else:
        val_ds = data.Dataset(data=val_list, transform=val_transform)

    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=cmr_ecg_collate,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return (train_loader, val_loader), (train_ds, val_ds)
