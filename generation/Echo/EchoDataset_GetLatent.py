from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from einops import repeat
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class EchoPseudoDataset(Dataset):

    def __init__(
        self,
        file_paths: List[Path],
        dataset_names: List[str],
        split: str = "all",
        seed: int = 2025,

    ):
        assert len(file_paths) == len(dataset_names), \
            "file_paths and dataset_names must have equal length"

        self.file_paths = list(file_paths)
        self.dataset_names = list(dataset_names)
        self.split = split
        self.seed = seed

        if not self.file_paths:
            raise ValueError('No preprocessed Echo samples found.')
        first_video_path = self.file_paths[0]
        v0 = np.load(first_video_path, allow_pickle=True).item()
        v0_video = v0["video"]
        assert v0_video.ndim == 4, f"video should be (T,H,W,C), got {v0_video.shape}"
        self.video_length = v0_video.shape[0]

        rng = np.random.RandomState(self.seed)
        self.frame_indices = rng.randint(0, self.video_length, size=len(self.file_paths))

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
        dataset_name = self.dataset_names[idx]

        sample = np.load(npy_path, allow_pickle=True).item()

        video = sample["video"]
        assert video.ndim == 4, f"video shape should be (T, H, W, C), got {video.shape}"
        T_len, H, W, C = video.shape
        video = torch.from_numpy(video)
        video = video.permute(3, 0, 1, 2).contiguous()

        pseudo_ecg = sample["pseudo_ecg"]
        pseudo_ecg = torch.from_numpy(pseudo_ecg)

        frame_idx = self.frame_indices[idx]
        frame = video[:, frame_idx]

        frame_idx = torch.tensor(frame_idx, dtype=torch.long).unsqueeze(0)

        frame = repeat(frame, 'c h w -> c t h w', t=T_len)

        return {
            "video": video,
            "pseudo_ecg": pseudo_ecg,
            "name": npy_path.stem,
            "dataset_name": dataset_name,
            "x_frame": frame,
            "frame_idx": frame_idx,
        }

def _build_samples_from_single_csv(
    csv_path: Path,
    cache_dir: Path,
    dataset_name: str,
) -> Tuple[List[Path], List[str]]:

    df = pd.read_csv(csv_path)

    if "FileName" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'FileName' column.")

    file_paths: List[Path] = []
    dataset_names: List[str] = []

    for _, row in df.iterrows():
        fname = str(row["FileName"])

        sample_id = Path(fname).stem
        npy_path = cache_dir / f"{sample_id}.npy"

        if not npy_path.exists():
            continue

        file_paths.append(npy_path)
        dataset_names.append(dataset_name)

    return file_paths, dataset_names

def build_all_samples_from_datasets(
    data_root: str,
) -> Tuple[List[Path], List[str]]:

    root = Path(data_root)

    dataset_specs = [
        (root / "EchoNet-Dynamic",              "EchoNet-Dynamic"),
        (root / "EchoNet-Pediatric" / "A4C",    "EchoNet-Pediatric-A4C"),
        (root / "EchoNet-Pediatric" / "PSAX",   "EchoNet-Pediatric-PSAX"),
    ]

    all_files: List[Path] = []
    all_dataset_names: List[str] = []

    for ds_dir, ds_name in dataset_specs:
        if not ds_dir.exists():
            continue

        cache_dir = ds_dir / "preprocessed"
        if not cache_dir.exists():
            raise RuntimeError(f"preprocessed directory not found: {cache_dir}")

        csv_paths = sorted(ds_dir.glob("*.csv"))
        if len(csv_paths) == 0:
            raise RuntimeError(f"No csv files found in {ds_dir}")

        csv_path = csv_paths[0]

        file_paths, dataset_names = _build_samples_from_single_csv(
            csv_path=csv_path,
            cache_dir=cache_dir,
            dataset_name=ds_name,
        )

        all_files.extend(file_paths)
        all_dataset_names.extend(dataset_names)

    return all_files, all_dataset_names

def create_dataset_from_root(
    data_root: str,
    seed: int,
) -> EchoPseudoDataset:

    file_paths, dataset_names = build_all_samples_from_datasets(
        data_root=data_root
    )

    dataset = EchoPseudoDataset(
        file_paths=file_paths,
        dataset_names=dataset_names,
        split="all",
        seed=seed,
    )

    return dataset

def create_dataloader_from_root(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = False,
    seed: int = 2025,
    rank: int = 0,
    world_size: int = 4,
):

    dataset = create_dataset_from_root(
        data_root=data_root,
        seed=seed,
    )

    data_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=data_sampler,
    )

    return dataloader, data_sampler
