# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample Echo videos on one GPU or across multiple GPUs with torchrun.

Single GPU: python sample_frame2video.py --batch_size 1 [options]
Multiple GPUs: torchrun --nproc_per_node=4 sample_frame2video.py --distributed [options]

Per-video PSNR/SSIM/LPIPS are written to <sample_dir>/metrics_per_sample.csv.
"""
import csv
from einops import rearrange
from torch.cuda.amp import GradScaler, autocast
from model.models_Frame2Video import DiT_models
from EchoDataset_sample import *

from transport import create_transport, Sampler
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from utils import *
from tqdm import tqdm
import os
import numpy as np
import argparse
from model.Tokenizer_ECHO import Tokenizer_ECHO

import cv2
from monai.losses.perceptual import PerceptualLoss
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

import warnings
warnings.filterwarnings("ignore")


def save_video_opencv(video, save_path, fps=25):
    """
    video: np.ndarray or torch.Tensor, shape [T, H, W, C]
    save_path: str, e.g. 'output.avi'
    fps: int
    """

    if 'torch' in str(type(video)):
        video = video.detach().cpu().numpy()

    assert video.ndim == 4, "video must be [T, H, W, C]"
    T, H, W, C = video.shape
    assert C == 3, "Only support 3-channel video"

    if video.dtype != np.uint8:
        video = np.clip(video, 0, 1) if video.max() <= 1.0 else np.clip(video, 0, 255)
        video = (video * 255).astype(np.uint8) if video.max() <= 1 else video.astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer: {save_path}')

    for t in range(T):
        frame = video[t]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        writer.write(frame)

    writer.release()

def main(args):
    """
    Run sampling.
    """

    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    logger = create_logger(args.log_dir, args.distributed)

    print(f"Rank {args.rank}: Starting to load data")
    _, _, test_ = create_dataloaders(
        data_dir=args.data_dir,
        latent_dir=args.latent_dir,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.global_seed
    )
    dataloader_test, test_sampler = test_
    if len(dataloader_test) == 0:
        raise ValueError('No test samples; check data paths and split labels.')

    print(f"Rank {args.rank}: Finished get dataloaders")

    scale_factor = args.scale_factor

    loss_perceptual = (
        PerceptualLoss(spatial_dims=2, network_type="squeeze").eval().to(args.device)
    )

    tokenizer = Tokenizer_ECHO(args)
    state_dict = torch.load(args.tokenizer_ckpt, map_location='cpu', weights_only=False)
    tokenizer.load_state_dict(state_dict['model'])
    tokenizer = tokenizer.to(args.device)
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    logger.info(f"Tokenizer_ECHO Parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")

    model = DiT_models[args.model](
        input_size=args.latent_size[-1],
        input_frames=args.latent_size[1],
        in_channels=args.latent_size[0],
        cond_channels=args.latent_size[0],
        time_dims=args.proj_dim,
    )

    ckpt_path = args.ckpt

    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict["ema"])
    if scale_factor is None:
        scale_factor = state_dict.get('scale_factor')
    if scale_factor is None or not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError('Supply --scale-factor for a checkpoint without a valid saved scale factor.')

    model = model.to(args.device)
    model.eval()
    model.requires_grad_(False)

    ckpt_path_ = '/'.join(ckpt_path.rsplit('/', 2)[-2:])
    if args.rank == 0:
        print('DiT weighted loaded!')
        print(f"Inference from checkpoint: {ckpt_path_}")
    logger.info(f"Inference from checkpoint: {ckpt_path_}")

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if args.mode == "ODE":
        if args.likelihood:
            raise ValueError('Likelihood evaluation is not supported by two-stage conditional sampling.')
        sample_fn = sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_sampling_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=args.reverse,
        )
    elif args.mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    metric_sum = {k: torch.tensor(0.0, device=args.device) for k in ["psnr", "ssim", "lpips"]}
    count = torch.tensor(0.0, device=args.device)
    local_sample_offset = 0

    metrics_csv_path = os.path.join(args.sample_dir, "metrics_per_sample.csv")
    metrics_csv_fields = ["sample_name", "psnr", "ssim", "lpips"]
    if args.rank == 0:
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=metrics_csv_fields).writeheader()
        logger.info(f"Per-sample metrics CSV: {metrics_csv_path}")

    if args.distributed:
        test_sampler.set_epoch(0)

    progress_bar = tqdm(enumerate(dataloader_test), total=len(dataloader_test), ncols=100)

    for i, batch in progress_bar:

        with torch.no_grad():
            with autocast(enabled=args.amp):

                x_echo = batch["video"].to(args.device)
                z_ecg = batch["latent_ecg"].to(args.device)
                keyframe_q = batch["keyframe_q"].to(args.device)

                time_mask = batch["time_mask"].to(args.device)
                frame_idx = batch["frame_idx"].to(args.device)
                name = batch["name"]

                x_frame = batch["x_frame"].to(args.device)

                z_frame = tokenizer.echo_tok.encode(x_frame)

                z_frame = z_frame * scale_factor
                c = (z_frame, frame_idx, z_ecg, keyframe_q)

                x_echo_bt = rearrange(x_echo, "b t h w c -> b t c h w").contiguous()

                x_echo = rearrange(x_echo, "b t h w c -> b c h w t")

                B = z_ecg.shape[0]
                T = args.latent_size[1]
                C = model.in_channels
                H_lat, W_lat = args.latent_size[-2], args.latent_size[-1]

                def broadcast_time_mask(mask_bt, z_like):

                    m = mask_bt
                    m = m.unsqueeze(1)
                    while m.dim() < z_like.dim():
                        m = m.unsqueeze(-1)
                    return m

                z1 = torch.randn(B, C, T, H_lat, W_lat, device=args.device)

                M_key = time_mask
                M_key_b = broadcast_time_mask(M_key, z1)

                def model_stage1(x, t, **model_kwargs):
                    v = model(x, t, **model_kwargs)
                    return v * M_key_b

                stage1_kwargs = {'update_mask': M_key_b} if args.mode == 'SDE' else {}
                xs1 = sample_fn(z1, model_stage1, context=c, **stage1_kwargs)
                latent_stage1 = xs1[-1]
                del xs1

                z2_noise = torch.randn_like(latent_stage1)
                M_nonkey = 1.0 - M_key
                M_nonkey_b = broadcast_time_mask(M_nonkey, z2_noise)

                z2_init = z2_noise * M_nonkey_b + latent_stage1 * M_key_b

                def model_stage2(x, t, **model_kwargs):
                    v = model(x, t, **model_kwargs)
                    return v * M_nonkey_b

                stage2_kwargs = {'update_mask': M_nonkey_b} if args.mode == 'SDE' else {}
                xs2 = sample_fn(z2_init, model_stage2, context=c, **stage2_kwargs)

                latent_final = xs2[-1]
                del xs2

                samples = tokenizer.echo_tok.decode(latent_final / scale_factor)

                samples.clamp_(0.0, 1.0)

                del z1, z2_noise, z2_init

        bs = samples.shape[0]

        valid_sample_indices = [
            j for j in range(bs)
            if (local_sample_offset + j) * args.world_size + args.rank < len(dataloader_test.dataset)
        ]
        batch_metric_rows = []
        with torch.no_grad(), autocast(enabled=False):
            for sample_idx in valid_sample_indices:

                sample_frames = samples[sample_idx].permute(1, 0, 2, 3).contiguous().float()
                target_frames = x_echo_bt[sample_idx].float()
                sample_metrics = {
                    "psnr": psnr(sample_frames, target_frames, data_range=1.0),
                    "ssim": ssim(sample_frames, target_frames, data_range=1.0),
                    "lpips": loss_perceptual(sample_frames * 2.0 - 1.0, target_frames * 2.0 - 1.0).mean(),
                }
                batch_metric_rows.append({
                    "sample_name": str(name[sample_idx]),
                    **{k: value.item() for k, value in sample_metrics.items()},
                })
                for k, value in sample_metrics.items():
                    metric_sum[k] += value
                count += 1
        local_sample_offset += bs

        samples = rearrange(samples, 'b c t h w -> b t h w c')
        samples = samples.detach().float().cpu().numpy()

        GTs = rearrange(x_echo, 'b c h w t -> b t h w c')
        GTs = GTs.detach().cpu().numpy()

        for sample_idx in valid_sample_indices:
            out = np.float32(samples[sample_idx])
            out_filename = os.path.join(args.sample_dir, str(name[sample_idx]) + ".avi")
            save_video_opencv(out, out_filename, fps=50)

        for sample_idx in valid_sample_indices:
            out = np.float32(GTs[sample_idx])
            out_filename = os.path.join(args.sample_dir, str(name[sample_idx]) + "_GT.avi")
            save_video_opencv(out, out_filename, fps=50)

        if args.distributed:
            gathered_rows = [None] * args.world_size if args.rank == 0 else None
            dist.gather_object(batch_metric_rows, gathered_rows, dst=0)
        else:
            gathered_rows = [batch_metric_rows]
        if args.rank == 0:
            with open(metrics_csv_path, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=metrics_csv_fields)
                for rank_rows in gathered_rows:
                    writer.writerows(rank_rows)

    if args.distributed:
        for k in metric_sum:
            dist.all_reduce(metric_sum[k], op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    if args.rank == 0 and count.item() > 0:
        metric_global = {k: (metric_sum[k] / count).item() for k in metric_sum}

        print(
            "[PSNR: %.4f][SSIM: %.4f][LPIPS: %.4f]"
            % (metric_global['psnr'], metric_global['ssim'],
               metric_global['lpips'])
        )
        logger.info(
            "[PSNR: %.4f][SSIM: %.4f][LPIPS: %.4f]"
            % (metric_global['psnr'], metric_global['ssim'],
               metric_global['lpips'])
        )
    elif args.rank == 0:
        print("[PSNR/SSIM/LPIPS] No metric samples collected, skip computation.")
        logger.info("[PSNR/SSIM/LPIPS] No metric samples collected, skip computation.")

    if args.rank == 0:
        print(f"Per-sample metrics saved to: {metrics_csv_path}")
        logger.info(f"Per-sample metrics saved to: {metrics_csv_path}")

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='EchoNet-Dynamic')

    parser.add_argument("--mode", type=str, default="ODE", choices=["ODE", "SDE"])
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--per-proc-batch-size", type=int, default=1)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', default=1, type=int, help='Batch size per GPU')
    parser.add_argument('--global-batch-size', '--global_batch_size', dest='global_batch_size', type=int, default=4)

    parser.add_argument('--scale-factor', '--scale_factor', dest='scale_factor', type=float, default=None)

    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=2025)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, required=True, help="DiT checkpoint to sample from.")

    parser.add_argument('--distributed', default=False, action='store_true',
                        help='Enable distributed sampling with torchrun; plain python defaults to one GPU.')
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=0, help="local rank")

    parser.add_argument('--amp', action=argparse.BooleanOptionalAction)
    parser.set_defaults(amp=True)

    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', default=2, type=int)

    parser.add_argument('--latent-size', '--latent_size', dest='latent_size', type=int, nargs=4, default=[16, 32, 14, 14], help='images input size')

    parser.add_argument('--echo-length', '--echo_length', dest='echo_length', type=int, default=125)
    parser.add_argument('--video-size', '--video_size', dest='video_size', type=int, nargs=3, default=(112, 112, 125))

    parser.add_argument('--ECG-size', '--ECG_size', dest='ECG_size', type=int, nargs=2, default=(1, 128))
    parser.add_argument('--ECG-enc-patch', '--ECG_enc_patch', dest='ECG_enc_patch', type=int, nargs=2, default=(1, 4))
    parser.add_argument('--ECG-enc-dim', '--ECG_enc_dim', dest='ECG_enc_dim', default=256, type=int)
    parser.add_argument('--ECG-enc-depth', '--ECG_enc_depth', dest='ECG_enc_depth', default=4, type=int)

    parser.add_argument('--proj-dim', '--proj_dim', dest='proj_dim', default=256, type=int)

    parser.add_argument('--selector-hidden', '--selector_hidden', dest='selector_hidden', default=256, type=int)
    parser.add_argument('--selector-layers', '--selector_layers', dest='selector_layers', default=2, type=int)
    parser.add_argument('--selector-heads', '--selector_heads', dest='selector_heads', default=4, type=int)

    parser.add_argument('--unet-channels', '--unet_channels', dest='unet_channels', type=int, nargs='+', default=(16, 32, 64, 128))
    parser.add_argument('--unet-strides', '--unet_strides', dest='unet_strides', type=int, nargs='+', default=(2, 1, 1))

    parser.add_argument('--num-frames', '--num_frames', dest='num_frames', type=int, default=125)
    parser.add_argument('--num-keyframes', '--num_keyframes', dest='num_keyframes', type=int, default=4)
    parser.add_argument('--num-bins', '--num_bins', dest='num_bins', type=int, default=4)

    parser.add_argument('--sigma-align', '--sigma_align', dest='sigma_align', type=float, default=0.2)
    parser.add_argument('--lambda-q', '--lambda_q', dest='lambda_q', type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.2)

    parser.add_argument('--embedding-dim', '--embedding_dim', dest='embedding_dim', type=int, default=512, help="Dimension of the embedding space.")
    parser.add_argument('--latent-dim', '--latent_dim', dest='latent_dim', type=int, default=16, help="Dimension of the latent channel.")
    parser.add_argument('--ista-iter-num', '--ista_iter_num', dest='ista_iter_num', type=int, default=2, help="Number of iterations in ISTA latent bottleneck.")
    parser.add_argument('--ista-layer-num', '--ista_layer_num', dest='ista_layer_num', type=int, default=2, help="Number of layers in ISTA latent bottleneck.")

    parser.add_argument('--l-dim', '--l_dim', dest='l_dim', type=int, default=128)
    parser.add_argument('--h-dim', '--h_dim', dest='h_dim', type=int, default=384)
    parser.add_argument('--sep-num-layer', '--sep_num_layer', dest='sep_num_layer', type=int, default=2,
                        help="Number of separate processing layers in encoder/decoder.")
    parser.add_argument('--fusion-num-layer', '--fusion_num_layer', dest='fusion_num_layer', type=int, default=4, help="Number of fusion layers in encoder/decoder.")

    parser.add_argument('--use-tile-inference', '--use_tile_inference', dest='use_tile_inference', action="store_true",
                        help="Enable tiling inference to process video in chunks.")
    parser.add_argument('--chunksize-enc', '--chunksize_enc', dest='chunksize_enc', type=int, default=9,
                        help="Number of frames per chunk during tiled encoding.")
    parser.add_argument('--chunksize-dec', '--chunksize_dec', dest='chunksize_dec', type=int, default=5,
                        help="Number of frames per chunk during tiled decoding.")

    parser.add_argument('--recon-loss', '--recon_loss', dest='recon_loss', default='l1', type=str)

    parser.add_argument('--kl-weight', '--kl_weight', dest='kl_weight', default=1e-7, type=float)
    parser.add_argument('--per-weight', '--per_weight', dest='per_weight', default=0.01, type=float)
    parser.add_argument('--adv-weight', '--adv_weight', dest='adv_weight', default=0.01, type=float)

    parser.add_argument('--conv-weight', '--conv_weight', dest='conv_weight', default=0.1, type=float)
    parser.add_argument('--k2t-weight', '--k2t_weight', dest='k2t_weight', default=0.1, type=float)
    parser.add_argument('--cl-weight', '--cl_weight', dest='cl_weight', default=0.05, type=float)

    parse_transport_args(parser)

    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument('--mode', default='ODE')
    mode = mode_parser.parse_known_args()[0].mode
    if mode == "ODE":
        parse_ode_args(parser)
    else:
        parse_sde_args(parser)

    parser.add_argument('--data-dir', '--data_dir', required=True)
    parser.add_argument('--latent-dir', '--latent_dir', required=True)
    parser.add_argument('--output-dir', '--output_dir', required=True)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--tokenizer-ckpt', '--tokenizer_ckpt', required=True)

    args = parser.parse_args()

    args.echo_latent_dim = args.latent_dim
    args.ECG_length = args.ECG_size[-1]

    args.log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    args.sample_dir = args.output_dir

    main(args)
