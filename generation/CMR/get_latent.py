import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from einops import rearrange
from torch.cuda.amp import autocast
from tqdm import tqdm

from CMRECGDataset_GetLatent import get_CMRECG, get_transforms, get_loader
from model.Tokenizer_CMR import Tokenizer_CMR
from runtime import add_runtime_args, prepare_args, init_distributed_mode, cleanup, create_logger


@torch.no_grad()
def get_latent(args, model):
    init_distributed_mode(args)
    logger = create_logger(args.log_dir, args.distributed)
    data_list = get_CMRECG(args)["data_files"]
    if not data_list:
        raise ValueError("No paired CMR/ECG files found under --data-dir")
    loader, _ = get_loader(args, data_list, get_transforms(args))
    model = model.to(args.device).eval().requires_grad_(False)
    logger.info(f"Tokenizer_CMR parameters: {sum(p.numel() for p in model.parameters()):,}")
    output = Path(args.latent_dir)
    for folder in ("latent_mri", "latent_ecg", "keyframes", "latent_slice", "latent_volume"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader):
        x_mri = batch["image"].to(args.device)
        if x_mri.shape[1] != args.num_frames:
            raise ValueError(f"Expected {args.num_frames} CMR frames, got {x_mri.shape[1]}")
        with autocast(enabled=args.amp):
            z = model.MRI_tok.encode(rearrange(x_mri, "b t h w d -> (b t) 1 d h w"))
            z_mri = rearrange(z, "(b t) c d h w -> b t c h w d", t=args.num_frames)
            z_traj = rearrange(z.mean(dim=(2, 3, 4)), "(b t) c -> b t c", t=args.num_frames)
            z_ecg = model.ECG_encoder.forward_features(batch["ecg"].to(args.device).unsqueeze(1))[:, 1:]
            z_ecg = model.ECG_proj(z_ecg)
            _, q = model.keyframe_selector(z_ecg, model.MRI_proj(z_traj))
            indices = model.get_keyframe_indices(q)
            z_slice = model.MRI_tok.encode(rearrange(batch["x_given_slice"].to(args.device), "b c h w d -> b c d h w"))
            z_volume = model.MRI_tok.encode(rearrange(batch["x_given_volume"].to(args.device), "b c h w d -> b c d h w"))
            z_slice = rearrange(z_slice, "b c d h w -> b c h w d")
            z_volume = rearrange(z_volume, "b c d h w -> b c h w d")
        for j, name in enumerate(batch["image_filename"]):
            for folder, value in (("latent_mri", z_mri[j]), ("latent_slice", z_slice[j:j+1]), ("latent_volume", z_volume[j:j+1])):
                image = nib.Nifti1Image(value.float().cpu().numpy(), np.eye(4))
                nib.save(image, output / folder / f"{name}.nii.gz")
            mask = np.zeros(args.num_frames, dtype=np.float32)
            mask[indices[j].cpu().numpy()] = 1
            for folder, suffix, value in (
                ("latent_ecg", "", z_ecg[j].float().cpu().numpy()),
                ("keyframes", "_q", q[j].float().cpu().numpy()),
                ("keyframes", "_idx", indices[j].cpu().numpy()),
                ("keyframes", "_mask", mask),
                ("latent_slice", "_t_idx", np.asarray([int(batch["t_given"][j])])),
                ("latent_slice", "_slice_idx", np.asarray([int(batch["slice_idx"][j])])),
            ):
                np.save(output / folder / f"{name}{suffix}.npy", value)
            logger.info(f"Saved {name}: CMR {tuple(z_mri[j].shape)}, ECG {tuple(z_ecg[j].shape)}")
    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1000, type=int, help="number of training epochs")
    parser.add_argument('--warmup_epochs', type=int, default=10, help='epochs to warmup LR')
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument("--val_interval", default=5, type=int)
    parser.add_argument("--ckpt_interval", default=10, type=int)

    parser.add_argument("--global_seed", type=int, default=2025)


    parser.add_argument('--distributed', default=False, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")


    parser.add_argument('--amp', action=argparse.BooleanOptionalAction)
    parser.set_defaults(amp=True)


    parser.add_argument('--cache', default=0.0, type=float)
    parser.add_argument('--replace_rate', default=0.2, type=float)
    parser.add_argument("--use_persistent_dataset", default=True, help="use monai Dataset class")
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", default=False, help="use monai cache Dataset")
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument("--time_samples", default=50, type=int)
    parser.add_argument("--cardiac_pad", default=(128, 128, 13))
    parser.add_argument("--cardiac_size", default=(128, 128, 13))

    parser.add_argument("--ECG_size", default=(12, 500))
    parser.add_argument("--ECG_enc_patch", default=10, type=int)
    parser.add_argument("--ECG_enc_dim", default=384, type=int)
    parser.add_argument("--ECG_enc_depth", default=4, type=int)


    parser.add_argument("--proj_dim", default=256, type=int)

    parser.add_argument("--selector_hidden", default=256, type=int)
    parser.add_argument("--selector_layers", default=2, type=int)
    parser.add_argument("--selector_heads", default=4, type=int)

    parser.add_argument("--unet_channels", default=(16, 32, 64, 128))
    parser.add_argument("--unet_strides", default=(2, 1, 1))

    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--num_keyframes", type=int, default=10)
    parser.add_argument("--num_bins", type=int, default=4)

    parser.add_argument("--sigma_align", type=float, default=0.1)
    parser.add_argument("--lambda_q", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.2)


    parser.add_argument("--embedding_dim", type=int, default=512, help="Dimension of the embedding space.")
    parser.add_argument("--latent_dim", type=int, default=16, help="Dimension of the latent channel.")
    parser.add_argument("--ista_iter_num", type=int, default=2, help="Number of iterations in ISTA latent bottleneck.")
    parser.add_argument("--ista_layer_num", type=int, default=2, help="Number of layers in ISTA latent bottleneck.")

    parser.add_argument("--l_dim", type=int, default=128)
    parser.add_argument("--h_dim", type=int, default=384)
    parser.add_argument("--sep_num_layer", type=int, default=2,
                        help="Number of separate processing layers in encoder/decoder.")
    parser.add_argument("--fusion_num_layer", type=int, default=4, help="Number of fusion layers in encoder/decoder.")


    parser.add_argument("--use_tile_inference", action="store_true",
                        help="Enable tiling inference to process video in chunks.")
    parser.add_argument("--chunksize_enc", type=int, default=9,
                        help="Number of frames per chunk during tiled encoding.")
    parser.add_argument("--chunksize_dec", type=int, default=5,
                        help="Number of frames per chunk during tiled decoding.")


    parser.add_argument("--recon_loss", default='l1', type=str)

    parser.add_argument('--kl_weight', default=1e-7, type=float)
    parser.add_argument('--per_weight', default=0.01, type=float)
    parser.add_argument('--adv_weight', default=0.01, type=float)

    parser.add_argument('--conv_weight', default=0.1, type=float)
    parser.add_argument('--k2t_weight', default=0.1, type=float)
    parser.add_argument('--cl_weight', default=0.05, type=float)

    add_runtime_args(parser, "latent")
    args = prepare_args(parser.parse_args(), "latent")

    args.mri_latent_dim = args.latent_dim
    args.volume_depth = args.cardiac_size[-1]
    args.ECG_length = args.ECG_size[-1]


    model = Tokenizer_CMR(args)
    state = torch.load(args.tokenizer_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    get_latent(args, model)
