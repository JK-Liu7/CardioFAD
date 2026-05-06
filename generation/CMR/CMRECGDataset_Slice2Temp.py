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
from sklearn.model_selection import train_test_split
from monai.transforms import *
import pickle
from typing import List, Optional
import random


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
    data_list = get_latent_UKBB(args.latent_dir)
    train_files, test_files = dataset_split(data_list, args.global_seed)

    train_dicts = train_files
    test_dicts = test_files

    print("Dataset all pretraining (CMR+ECG paired): number of data: {}".format(len(data_list)))

    datasets = {
        "data_name": "UKBB",
        "train_files": train_dicts,
        "test_files": test_dicts,
        "modality": "MRI+ECG+keyframes",
    }

    return datasets


def dataset_split(data_list, seed):
    train_files, test_files = train_test_split(data_list, test_size=0.2, random_state=seed)
    return train_files, test_files


def get_latent_UKBB(latent_dir):
    mr_pattern = os.path.join(latent_dir, "latent_mri", "*_4D.nii.gz")
    ecg_pattern = os.path.join(latent_dir, "latent_ecg", "*_4D.npy")
    keyframe_q_pattern = os.path.join(latent_dir, "keyframes", "*_q.npy")
    keyframe_idx_pattern = os.path.join(latent_dir, "keyframes", "*_idx.npy")
    time_mask_pattern = os.path.join(latent_dir, "keyframes", "*_mask.npy")
    slice_pattern = os.path.join(latent_dir, "latent_slice", "*_4D.nii.gz")
    t_idx_pattern = os.path.join(latent_dir, "latent_slice", "*_t_idx.npy")
    slice_idx_pattern = os.path.join(latent_dir, "latent_slice", "*_slice_idx.npy")

    mr_paths = sorted(glob.glob(mr_pattern))
    ecg_paths = sorted(glob.glob(ecg_pattern))

    keyframe_q_paths = sorted(glob.glob(keyframe_q_pattern))
    keyframe_idx_paths = sorted(glob.glob(keyframe_idx_pattern))
    time_mask_paths = sorted(glob.glob(time_mask_pattern))

    slice_paths = sorted(glob.glob(slice_pattern))
    t_idx_paths = sorted(glob.glob(t_idx_pattern))
    slice_idx_paths = sorted(glob.glob(slice_idx_pattern))

    num = len(mr_paths)
    paired = []
    for i in range(num):
        paired.append({
            "latent_mri": mr_paths[i],
            "latent_ecg": ecg_paths[i],
            "keyframe_q": keyframe_q_paths[i],
            "keyframe_idx": keyframe_idx_paths[i],
            "time_mask": time_mask_paths[i],
            "latent_slice": slice_paths[i],
            "t_given": t_idx_paths[i],
            "slice_idx": slice_idx_paths[i],
        })

    return paired



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


class RandFlipNoTime(MapTransform, Randomizable):
    def __init__(
        self,
        keys,
        prob: float = 0.3,
        spatial_axes=(2, 3, 4),
        slice_idx_key: str = "slice_idx",
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        Randomizable.__init__(self)
        self.prob = prob
        self.spatial_axes = tuple(spatial_axes)
        self.slice_idx_key = slice_idx_key
        self._flip_axes = None

    def randomize(self) -> None:
        self._flip_axes = [
            ax for ax in self.spatial_axes if self.R.random() < self.prob
        ]

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._flip_axes:
            return d

        if self.slice_idx_key in d and 4 in self._flip_axes:
            ref_key = None
            for key in self.key_iterator(d):
                ref_key = key
                break

            if ref_key is not None:
                img_ref = d[ref_key]
                arr_ref = np.asarray(img_ref)
                if arr_ref.ndim != 5:
                    raise ValueError(
                        f"RandFlipNoTime expects a 5D tensor/array [T,C,H,W,D], "
                        f"but got shape {arr_ref.shape} for key '{ref_key}'."
                    )
                D = arr_ref.shape[4]
                old_slice_idx = d[self.slice_idx_key]
                old_slice_idx = int(np.asarray(old_slice_idx).item())
                new_slice_idx = D - 1 - old_slice_idx
                d[self.slice_idx_key] = np.asarray([new_slice_idx], dtype=np.int64)

        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim != 5:
                raise ValueError(
                    f"RandFlipNoTime expects a 5D tensor/array [T, C, H, W, D], "
                    f"but got shape {img.shape} for key '{key}'."
                )

            if torch.is_tensor(img):
                for ax in self._flip_axes:
                    img = torch.flip(img, dims=(ax,))
            else:
                arr = np.asarray(img)
                for ax in self._flip_axes:
                    arr = np.flip(arr, axis=ax)
                img = arr

            d[key] = img

        return d

class RandRotate90TimeFramed(MapTransform, Randomizable):
    def __init__(
        self,
        keys,
        prob: float = 0.3,
        max_k: int = 3,
        spatial_axes=(2, 3),
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        Randomizable.__init__(self)
        self.prob = prob
        self.max_k = max_k
        self.spatial_axes = tuple(spatial_axes)
        self._do_transform = False
        self._k = 0

    def randomize(self) -> None:
        if self.R.random() < self.prob:
            self._do_transform = True
            self._k = int(self.R.randint(self.max_k + 1))
        else:
            self._do_transform = False
            self._k = 0

    def _rotate_once(self, x):
        if (not self._do_transform) or self._k == 0:
            return x
        return np.rot90(x, k=self._k, axes=(1, 2))

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if (not self._do_transform) or self._k == 0:
            return d

        for key in self.key_iterator(d):
            x = np.asarray(d[key])
            assert x.ndim == 5, f"Expect 5D [T, C, H, W, D], got {x.shape} for key {key}"
            T, C, H, W, D = x.shape
            for t in range(T):
                x[t] = self._rotate_once(x[t])
            d[key] = x

        return d


# ===================== Transforms =====================
def get_transforms(args):
    common_transform = [
        LoadImaged(keys=['latent_mri', 'latent_slice'], image_only=True, allow_missing_keys=False),
        LoadECGd(keys=['latent_ecg'], allow_missing_keys=True),
        LoadKeyframeqd(keys=['keyframe_q'], allow_missing_keys=True),
        LoadKeyframeidxd(keys=['keyframe_idx'], allow_missing_keys=True),
        LoadTimemaskd(keys=['time_mask'], allow_missing_keys=True),
        LoadTidxd(keys=['t_given'], allow_missing_keys=True),
        LoadSliceidxd(keys=['slice_idx'], allow_missing_keys=True),
    ]

    train_transform = Compose(
        common_transform
        + [
            RandFlipNoTime(keys=["latent_mri", 'latent_slice'], prob=0.3, allow_missing_keys=True),
            RandRotate90TimeFramed(keys=["latent_mri", 'latent_slice'], prob=0.3, max_k=3, allow_missing_keys=True),

            ToTensord(
                keys=[
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
                keys=["latent_mri", "latent_ecg", "keyframe_q", "latent_slice", "time_mask"],
                allow_missing_keys=True,
                dtype=torch.float32,
            ),
        ])

    test_transform = Compose(
        common_transform
        + [
            ToTensord(
                keys=[
                    "latent_mri",
                    "latent_ecg",
                    "keyframe_q",
                    "keyframe_idx",
                    "z_given_latent",
                    "time_mask",
                    "t_given",
                    "slice_idx",
                ],
                allow_missing_keys=True,
            ),
            EnsureTyped(
                keys=["latent_mri", "latent_ecg", "keyframe_q", "z_given_latent", "time_mask"],
                allow_missing_keys=True,
                dtype=torch.float32,
            ),
        ])

    return train_transform, test_transform


def get_loader(args, train_list, test_list, train_transform, test_transform):
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
    )

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
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    return (train_loader, test_loader), (train_ds, test_ds)
