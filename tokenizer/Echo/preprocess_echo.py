import argparse
import os
from pathlib import Path

import numpy as np
import cv2
from decord import VideoReader, cpu


def read_and_resample_video(
    video_path: Path,
    target_length: int,
    target_size: tuple[int, int] = (112, 112),
) -> np.ndarray:
    if target_length <= 0 or min(target_size) <= 0:
        raise ValueError('Target length and spatial dimensions must be positive.')
    vr = VideoReader(str(video_path), ctx=cpu(0))
    num_frames = len(vr)

    if num_frames == 0:
        raise ValueError(f"Video {video_path} has 0 frames.")


    frames = vr.get_batch(np.arange(num_frames)).asnumpy()
    height, width = target_size
    if frames.shape[1:3] != (height, width):
        interpolation = cv2.INTER_AREA if height <= frames.shape[1] and width <= frames.shape[2] else cv2.INTER_LINEAR
        frames = np.stack([cv2.resize(frame, (width, height), interpolation=interpolation) for frame in frames])
    frames = frames.astype("float32") / 255.0

    if num_frames == target_length:

        return frames


    T, H, W, C = frames.shape


    t_new = np.linspace(0, T - 1, target_length, dtype=np.float32)
    t0 = np.floor(t_new).astype(np.int64)
    t1 = np.clip(t0 + 1, 0, T - 1)
    alpha = (t_new - t0).reshape(-1, 1, 1, 1)

    frames0 = frames[t0]
    frames1 = frames[t1]


    resampled = (1.0 - alpha) * frames0 + alpha * frames1

    return resampled.astype("float32")


def main(args):
    video_dir = Path(args.video_dir)
    pseudo_dir = Path(args.pseudo_ecg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    pseudo_paths = sorted(pseudo_dir.glob("*.npy"))

    if len(pseudo_paths) == 0:
        print(f"[WARN] No .npy files found in {pseudo_dir}")
        return

    print(f"Found {len(pseudo_paths)} pseudo ECG files in {pseudo_dir}")
    print(f"Will save processed samples to {out_dir}")
    print(f"Target video length: {args.target_length} frames\n")

    num_processed = 0
    num_skipped = 0

    for pseudo_path in pseudo_paths:
        stem = pseudo_path.stem
        video_path = video_dir / f"{stem}.avi"

        if not video_path.exists():
            print(f"[SKIP] Video not found for pseudo ECG: {pseudo_path.name}")
            num_skipped += 1
            continue

        try:

            video = read_and_resample_video(
                video_path=video_path,
                target_length=args.target_length,
                target_size=tuple(args.target_size),
            )


            pseudo_ecg = np.load(pseudo_path)
            pseudo_ecg = pseudo_ecg.astype("float32")


            sample = {
                "video": video,
                "pseudo_ecg": pseudo_ecg,
            }

            out_path = out_dir / f"{stem}.npy"
            np.save(out_path, sample)

            num_processed += 1
            if num_processed % 50 == 0:
                print(f"[INFO] Processed {num_processed} samples...")

        except Exception as e:
            print(f"[ERROR] Failed on {stem}: {e}")
            num_skipped += 1
            continue

    print("\nDone.")
    print(f"Processed: {num_processed}")
    print(f"Skipped:   {num_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Echo videos and matching pseudo ECG into cached .npy files."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to Videos folder containing .avi files.",
    )
    parser.add_argument(
        "--pseudo_ecg_dir",
        type=str,
        required=True,
        help="Path to pseudo_ecg folder containing .npy files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to save merged .npy samples.",
    )
    parser.add_argument(
        '--target-size', '--target_size', dest='target_size',
        type=int, nargs=2, default=(112, 112), metavar=('HEIGHT', 'WIDTH'),
        help='Spatial dimensions of cached videos; defaults match the tokenizer and DiT.',
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=125,
        help="Number of frames to resample each video to (temporal length).",
    )

    args = parser.parse_args()

    main(args)
