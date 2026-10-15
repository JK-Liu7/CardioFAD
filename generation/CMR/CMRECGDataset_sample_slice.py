from runtime import load_latent_records
import argparse
import pandas as pd
import torch
import numpy as np
import collections
from collections.abc import Sequence
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


warnings.filterwarnings("ignore")


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
    data_list = get_latent_UKBB(args.data_dir, args.latent_dir)
    split_file = getattr(args, "split_file", None) or Path(args.latent_dir) / "ukbb_split.json"
    train_files, val_files, test_files = dataset_split(data_list, args.global_seed, split_file)
    print(f"UKBB train/val/test: {len(train_files)}/{len(val_files)}/{len(test_files)}; split: {split_file}")
    return {
        "data_name": "UKBB",
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "modality": "MRI+ECG + keyframes",
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


def get_latent_UKBB(data_dir, latent_dir):
    return load_latent_records(latent_dir, "slice", data_dir=data_dir)


class SaveImageFilenamed(MapTransform):
    def __init__(
        self,
        keys,
        out_key: str = "image_filename",
        basename: bool = True,
        strip_nii_gz: bool = True,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.out_key = out_key
        self.basename = basename
        self.strip_nii_gz = strip_nii_gz

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            v = d[key]
            if isinstance(v, str):
                path = v
            else:
                continue

            fname = os.path.basename(path) if self.basename else path


            if self.strip_nii_gz and fname.endswith(".nii.gz"):
                fname = fname[:-7]

            d[self.out_key] = fname

        return d


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


class LoadKeyframeqd(MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False, dtype=np.float32):
        super().__init__(keys, allow_missing_keys)
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            path = d[key]
            if isinstance(path, str) and path.endswith(".npy"):
                keyframe_q = np.load(path).astype(self.dtype)
            else:
                keyframe_q = np.asarray(path, dtype=self.dtype)
            d[key] = keyframe_q
        return d


class LoadKeyframeidxd(MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False, dtype=np.int64):
        super().__init__(keys, allow_missing_keys)
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            path = d[key]
            if isinstance(path, str) and path.endswith(".npy"):
                keyframe_idx = np.load(path).astype(self.dtype)
            else:
                keyframe_idx = np.asarray(path, dtype=self.dtype)
            d[key] = keyframe_idx
        return d


class LoadTidxd(MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False, dtype=np.int64):
        super().__init__(keys, allow_missing_keys)
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            path = d[key]
            if isinstance(path, str) and path.endswith(".npy"):
                t_idx = np.load(path).astype(self.dtype)
            else:
                t_idx = np.asarray(path, dtype=self.dtype)
            d[key] = t_idx
        return d


class LoadSliceidxd(MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False, dtype=np.int64):
        super().__init__(keys, allow_missing_keys)
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            path = d[key]
            if isinstance(path, str) and path.endswith(".npy"):
                slice_idx = np.load(path).astype(self.dtype)
            else:
                slice_idx = np.asarray(path, dtype=self.dtype)
            d[key] = slice_idx
        return d


class LoadTimemaskd(MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False, dtype=np.int64):
        super().__init__(keys, allow_missing_keys)
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            path = d[key]
            if isinstance(path, str) and path.endswith(".npy"):
                time_mask = np.load(path).astype(self.dtype)
            else:
                time_mask = np.asarray(path, dtype=self.dtype)
            d[key] = time_mask
        return d


def get_transforms(args):

    common_transform = [
        SaveImageFilenamed(keys=["x_mri"], out_key="image_filename"),
        LoadImaged(keys=["x_mri", 'latent_mri', 'latent_slice'], image_only=True, allow_missing_keys=True),
        LoadECGd(keys=['latent_ecg'], allow_missing_keys=True),
        LoadKeyframeqd(keys=['keyframe_q'], allow_missing_keys=True),
        LoadKeyframeidxd(keys=['keyframe_idx'], allow_missing_keys=True),
        LoadTimemaskd(keys=['time_mask'], allow_missing_keys=True),
        LoadTidxd(keys=['t_given'], allow_missing_keys=True),
        LoadSliceidxd(keys=['slice_idx'], allow_missing_keys=True),

        EnsureChannelFirstd(keys=['x_mri'], allow_missing_keys=True),
        ScaleIntensityRangePercentilesd(keys=['x_mri'], lower=0.5, upper=99.5, b_min=0, b_max=1),
        SpatialPadd(keys=['x_mri'], spatial_size=args.cardiac_pad, mode='edge', allow_missing_keys=True),
        CenterSpatialCropd(keys=['x_mri'], roi_size=args.cardiac_size, allow_missing_keys=True),
    ]

    test_transform = Compose(
        common_transform
        + [
            ToTensord(
                keys=[
                    "x_mri",
                    "latent_mri",
                    "latent_ecg",
                    "keyframe_q",
                    "keyframe_idx",
                    "latent_slice",
                    "time_mask",
                    "t_given",
                    "slice_idx",
                ],
                allow_missing_keys=True,
            ),

            EnsureTyped(
                keys=["x_mri", "latent_mri", "latent_ecg", "keyframe_q", "latent_slice", "time_mask"],
                allow_missing_keys=True,
                dtype=torch.float32,
            ),

        ])

    return test_transform


def cmr_ecg_collate(batch):
    latent_mri_list = []
    latent_ecg_list = []
    q_list = []
    idx_list = []
    z_given_list = []
    time_mask_list = []
    t_given_list = []
    slice_idx_list = []

    for d in batch:
        if "latent_mri" in d:
            latent_mri_list.append(d["latent_mri"])
        elif "image" in d:
            latent_mri_list.append(d["image"])
        else:
            raise KeyError("No 'latent_mri' or 'image' key in sample.")


        if "latent_ecg" in d:
            latent_ecg_list.append(d["latent_ecg"])
        elif "ecg" in d:
            latent_ecg_list.append(d["ecg"])
        else:
            raise KeyError("No 'latent_ecg' or 'ecg' key in sample.")

        if "keyframe_q" in d and d["keyframe_q"] is not None:
            q_list.append(d["keyframe_q"])
        if "keyframe_idx" in d and d["keyframe_idx"] is not None:
            idx_list.append(d["keyframe_idx"])

        if "z_given_latent" in d:
            z_given_list.append(d["z_given_latent"])
        if "time_mask" in d:
            time_mask_list.append(d["time_mask"])
        if "t_given" in d:

            t_given_list.append(int(d["t_given"]))
        if "slice_idx" in d:
            slice_idx_list.append(int(d["slice_idx"]))


    latent_mri = torch.stack(latent_mri_list, dim=0)
    latent_ecg = torch.stack(latent_ecg_list, dim=0)

    q = torch.stack(q_list, dim=0) if len(q_list) > 0 else None
    idx = torch.stack(idx_list, dim=0) if len(idx_list) > 0 else None

    z_given_latent = torch.stack(z_given_list, dim=0) if len(z_given_list) > 0 else None
    time_mask = torch.stack(time_mask_list, dim=0) if len(time_mask_list) > 0 else None

    t_given = torch.tensor(t_given_list, dtype=torch.long).unsqueeze(-1) if len(t_given_list) > 0 else None
    slice_idx = torch.tensor(slice_idx_list, dtype=torch.long).unsqueeze(-1)  if len(slice_idx_list) > 0 else None

    return {
        "latent_mri": latent_mri,
        "latent_ecg": latent_ecg,
        "keyframe_q": q,
        "keyframe_idx": idx,
        "z_given_latent": z_given_latent,
        "time_mask": time_mask,
        "t_given": t_given,
        "slice_idx": slice_idx,
    }


def get_loader(args, test_list, test_transform):
    if args.use_persistent_dataset:
        test_ds = PersistentDataset(
            data=test_list,
            transform=test_transform,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            cache_dir=args.cache_dir
        )
    else:
        test_ds = data.Dataset(data=test_list, transform=test_transform)

    test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return test_loader, test_ds
