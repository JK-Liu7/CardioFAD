import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T


class EchoPseudoDataset(Dataset):


    def __init__(
        self,
        file_paths: List[Path],
        split: str = "train",
        augment: bool = True,
    ):
        self.file_paths = list(file_paths)
        self.split = split
        self.augment = augment and (split == "train")


        if self.augment:
            self.frame_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.0,
                    hue=0.0,
                ),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=0.2,
                ),


            ])
        else:
            self.frame_transform = None

    def __len__(self) -> int:
        return len(self.file_paths)

    def _augment_video(self, video: torch.Tensor) -> torch.Tensor:


        if self.frame_transform is None:
            return video

        C, T_len, H, W = video.shape

        frames = video.permute(1, 0, 2, 3)
        aug_frames = []
        for t in range(T_len):
            frame = frames[t]
            frame = self.frame_transform(frame)
            aug_frames.append(frame)
        aug_frames = torch.stack(aug_frames, dim=0)

        video = aug_frames.permute(1, 0, 2, 3).contiguous()
        return video

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        npy_path = self.file_paths[idx]
        sample = np.load(npy_path, allow_pickle=True).item()


        video = sample["video"]
        assert video.ndim == 4, f"video shape should be (T, H, W, C), got {video.shape}"
        T_len, H, W, C = video.shape
        video = torch.from_numpy(video)
        video = video.permute(3, 0, 1, 2).contiguous()

        pseudo_ecg = sample["pseudo_ecg"]
        pseudo_ecg = torch.from_numpy(pseudo_ecg)

        if self.augment:
            video = self._augment_video(video)

        return {
            "video": video,
            "pseudo_ecg": pseudo_ecg,
            "name": npy_path.stem,
        }


def _build_splits_from_single_csv(
    csv_path: Path,
    cache_dir: Path,
) -> Tuple[List[Path], List[Path], List[Path]]:


    df = pd.read_csv(csv_path)

    if "FileName" not in df.columns or "Split" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'FileName' and 'Split' columns.")

    train_files: List[Path] = []
    val_files: List[Path] = []
    test_files: List[Path] = []


    split_values = df["Split"].astype(str).str.upper().unique().tolist()
    is_explicit = set(split_values) <= {"TRAIN", "VAL", "TEST"}

    for _, row in df.iterrows():
        fname = str(row["FileName"])

        sample_id = Path(fname).stem
        npy_path = cache_dir / f"{sample_id}.npy"


        if not npy_path.exists():
            continue

        if is_explicit:
            split = str(row["Split"]).upper()
            if split == "TRAIN":
                train_files.append(npy_path)
            elif split == "VAL":
                val_files.append(npy_path)
            elif split == "TEST":
                test_files.append(npy_path)
        else:

            try:
                fold_raw = int(row["Split"])
            except ValueError:

                continue
            fold_id = fold_raw + 1

            if 1 <= fold_id <= 8:
                train_files.append(npy_path)
            elif fold_id == 9:
                val_files.append(npy_path)
            elif fold_id == 10:
                test_files.append(npy_path)
            else:

                continue

    return train_files, val_files, test_files


def build_combined_splits_from_datasets(
    data_root: str,
) -> Tuple[List[Path], List[Path], List[Path]]:


    root = Path(data_root)

    dataset_dirs = [
        root / "EchoNet-Dynamic",
        root / "EchoNet-Pediatric" / "A4C",
        root / "EchoNet-Pediatric" / "PSAX",
    ]

    all_train: List[Path] = []
    all_val: List[Path] = []
    all_test: List[Path] = []

    for ds_dir in dataset_dirs:
        if not ds_dir.exists():
            raise RuntimeError(f"Dataset directory not found: {ds_dir}")

        cache_dir = ds_dir / "preprocessed"
        if not cache_dir.exists():
            raise RuntimeError(f"preprocessed directory not found: {cache_dir}")


        csv_paths = sorted(ds_dir.glob("FileList.csv"))
        if len(csv_paths) == 0:
            raise RuntimeError(f"No csv files found in {ds_dir}")

        csv_path = csv_paths[0]


        t_files, v_files, te_files = _build_splits_from_single_csv(csv_path, cache_dir)
        all_train.extend(t_files)
        all_val.extend(v_files)
        all_test.extend(te_files)


    all_train = sorted(set(all_train))
    all_val = sorted(set(all_val))
    all_test = sorted(set(all_test))

    return all_train, all_val, all_test


def create_train_val_test_datasets_from_root(
    data_root: str,
    augment_train: bool = True,
) -> Tuple[EchoPseudoDataset, EchoPseudoDataset, EchoPseudoDataset]:


    train_files, val_files, test_files = build_combined_splits_from_datasets(
        data_root=data_root,
    )

    train_set = EchoPseudoDataset(
        file_paths=train_files,
        split="train",
        augment=augment_train,
    )
    val_set = EchoPseudoDataset(
        file_paths=val_files,
        split="val",
        augment=False,
    )
    test_set = EchoPseudoDataset(
        file_paths=test_files,
        split="test",
        augment=False,
    )

    return train_set, val_set, test_set


def create_dataloaders_from_root(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 4,
):


    train_set, val_set, test_set = create_train_val_test_datasets_from_root(
        data_root=data_root,
        augment_train=True,
    )

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    test_sampler = DistributedSampler(
        test_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=False,
    )

    return (train_loader, train_sampler), (val_loader, val_sampler), (test_loader, test_sampler)
