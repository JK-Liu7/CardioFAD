from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import rotate


class EchoDataset(Dataset):
    def __init__(
        self,
        items: List[Dict],
        split: str,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.2,
    ):

        self.items = items
        self.split = split
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.items[idx]
        name = info["name"]
        dataset_name = info["dataset_name"]
        video_path = info["video_path"]
        latent_echo_path = info["latent_echo_path"]
        latent_ecg_path = info["latent_ecg_path"]
        keyframe_q_path = info["keyframe_q_path"]
        keyframe_mask_path = info["keyframe_mask_path"]
        latent_frame_path = info["latent_frame_path"]
        frame_idx_path = info["frame_idx_path"]

        video_sample = np.load(video_path, allow_pickle=True).item()
        video = video_sample["video"]
        assert video.ndim == 4, f"video should be (T,H,W,C), got {video.shape}"

        latent_echo_np = np.load(latent_echo_path)
        latent_ecg_np = np.load(latent_ecg_path)
        keyframe_q_np = np.load(keyframe_q_path)
        keyframe_mask_np = np.load(keyframe_mask_path)
        latent_frame_np = np.load(latent_frame_path)
        frame_idx_np = np.load(frame_idx_path)

        latent_echo = torch.from_numpy(latent_echo_np).float()
        latent_ecg = torch.from_numpy(latent_ecg_np).float()
        keyframe_q = torch.from_numpy(keyframe_q_np)
        keyframe_mask = torch.from_numpy(keyframe_mask_np)
        latent_frame = torch.from_numpy(latent_frame_np).float()
        frame_idx = torch.from_numpy(frame_idx_np).long().reshape(1)

        do_flip = (self.split == "train") and (torch.rand(1).item() < self.flip_prob)

        if do_flip:

            latent_echo = torch.flip(latent_echo, dims=[-1])

            latent_frame = torch.flip(latent_frame, dims=[-1])

        do_rotate = (self.split == "train") and (torch.rand(1).item() < self.rotate_prob)
        if do_rotate:
            angle = torch.empty(1).uniform_(-5, 5).item()

            latent_frame = torch.stack(
                [rotate(latent_frame[:, t], angle) for t in range(latent_frame.shape[1])],
                dim=1
            )

            latent_echo = torch.stack(
                [rotate(latent_echo[:, t], angle) for t in range(latent_echo.shape[1])],
                dim=1
            )

        return {
            "name": name,
            "dataset_name": dataset_name,
            "latent_echo": latent_echo,
            "latent_ecg": latent_ecg,
            "z_frame": latent_frame,
            "frame_idx": frame_idx,
            "keyframe_q": keyframe_q,
            "time_mask": keyframe_mask,
        }

def _build_split_items_from_csv(
    csv_path: str,
    video_dir: str,
    latent_dir: str,
    dataset_name: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:

    csv_path = Path(csv_path)
    video_dir = Path(video_dir)
    latent_dir = Path(latent_dir)

    latent_echo_dir = latent_dir / "latent_echo" / dataset_name
    latent_ecg_dir = latent_dir / "latent_ecg" / dataset_name
    keyframe_dir = latent_dir / "keyframes" / dataset_name
    latent_frame_dir = latent_dir / "latent_frame" / dataset_name

    df = pd.read_csv(csv_path)

    if "FileName" not in df.columns or "Split" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'FileName' and 'Split' columns.")

    train_items: List[Dict] = []
    val_items: List[Dict] = []
    test_items: List[Dict] = []

    split_values = df["Split"].astype(str).str.upper().unique().tolist()
    is_explicit = set(split_values) <= {"TRAIN", "VAL", "TEST"}

    for _, row in df.iterrows():
        fname = str(row["FileName"])
        sample_id = Path(fname).stem

        video_path = video_dir / f"{sample_id}.npy"
        latent_echo_path = latent_echo_dir / f"{sample_id}.npy"
        latent_ecg_path = latent_ecg_dir / f"{sample_id}_proj.npy"

        keyframe_q_path = keyframe_dir / f"{sample_id}_q.npy"
        keyframe_mask_path = keyframe_dir / f"{sample_id}_mask.npy"

        latent_frame_path = latent_frame_dir / f"{sample_id}.npy"
        frame_idx_path = latent_frame_dir / f"{sample_id}_frame_idx.npy"

        if not (video_path.exists() and latent_echo_path.exists() and latent_ecg_path.exists()):
            continue

        if is_explicit:
            split = str(row["Split"]).upper()
            if split == "TRAIN":
                target = "train"
            elif split == "VAL":
                target = "val"
            elif split == "TEST":
                target = "test"
            else:
                continue
        else:

            split_raw = str(row["Split"]).strip()
            try:
                fold_raw = int(split_raw)
            except ValueError:

                continue

            fold_id = fold_raw + 1

            if 1 <= fold_id <= 8:
                target = "train"
            elif fold_id == 9:
                target = "val"
            elif fold_id == 10:
                target = "test"
            else:
                continue

        item = {
            "name": sample_id,
            "dataset_name": dataset_name,
            "video_path": video_path,
            "latent_echo_path": latent_echo_path,
            "latent_ecg_path": latent_ecg_path,
            "keyframe_q_path": keyframe_q_path,
            "keyframe_mask_path": keyframe_mask_path,
            "latent_frame_path": latent_frame_path,
            "frame_idx_path": frame_idx_path,
        }

        if target == "train":
            train_items.append(item)
        elif target == "val":
            val_items.append(item)
        elif target == "test":
            test_items.append(item)

    return train_items, val_items, test_items

def create_datasets(
    csv_path: str,
    video_dir: str,
    latent_dir: str,
    dataset_name: str,
    flip_prob: float = 0.5,
    rotate_prob: float = 0.2,
) -> Tuple[EchoDataset, EchoDataset, EchoDataset]:

    train_items, val_items, test_items = _build_split_items_from_csv(
        csv_path=csv_path,
        video_dir=video_dir,
        latent_dir=latent_dir,
        dataset_name=dataset_name,
    )

    train_set = EchoDataset(
        items=train_items,
        split="train",
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
    )
    val_set = EchoDataset(
        items=val_items,
        split="val",
        flip_prob=0.0,
        rotate_prob=0.0,
    )
    test_set = EchoDataset(
        items=test_items,
        split="test",
        flip_prob=0.0,
        rotate_prob=0.0,
    )

    return train_set, val_set, test_set

def create_dataloaders(
    data_dir: str,
    latent_dir: str,
    dataset_name: str,
    batch_size: int = 4,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 4,
    flip_prob: float = 0.5,
    rotate_prob: float = 0.2,
):

    data_dir = data_dir
    csv_path = data_dir + "/FileList.csv"
    video_dir = data_dir + "/preprocessed/"
    train_set, val_set, test_set = create_datasets(
        csv_path=csv_path,
        video_dir=video_dir,
        latent_dir=latent_dir,
        dataset_name=dataset_name,
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
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
