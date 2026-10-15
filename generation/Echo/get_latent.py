import argparse
import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from EchoDataset_GetLatent import create_dataloader_from_root
from utils import init_distributed_mode, create_logger
from utils_getlatent import cleanup
from model.Tokenizer_ECHO import Tokenizer_ECHO
from torch.cuda.amp import autocast


@torch.no_grad()
def get_latent(args, model):
    init_distributed_mode(args)
    logger = create_logger(args.log_dir, args.distributed)
    loader, sampler = create_dataloader_from_root(
        data_root=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, seed=args.seed,
        rank=args.rank, world_size=args.world_size,
    )
    model = model.to(args.device).eval().requires_grad_(False)
    root = Path(args.latent_dir)
    local_offset = 0
    for batch in tqdm(loader, ncols=100):
        x_echo = batch['video'].to(args.device)
        x_ecg = rearrange(batch['pseudo_ecg'].to(args.device), 'b t d -> b d 1 t')
        with autocast(enabled=args.amp):
            z_echo = model.echo_tok.encode(x_echo)
            z_frame = model.echo_tok.encode(batch['x_frame'].to(args.device))
            z_ecg = model.ECG_encoder.forward_features(x_ecg)[:, 1:, :]
            z_ecg_proj = model.ECG_proj(z_ecg)
            _, q = model.keyframe_selector(z_ecg_proj)
            indices = model.get_keyframe_indices(q)
        mask = torch.zeros_like(q).scatter_(1, indices, 1.0)
        outputs = {
            ('latent_echo', ''): z_echo,
            ('latent_ecg', ''): z_ecg,
            ('latent_ecg', '_proj'): z_ecg_proj,
            ('keyframes', '_q'): q,
            ('keyframes', '_idx'): indices,
            ('keyframes', '_mask'): mask,
            ('latent_frame', ''): z_frame,
            ('latent_frame', '_frame_idx'): batch['frame_idx'],
        }
        for j, name in enumerate(batch['name']):
            if (local_offset + j) * args.world_size + args.rank >= len(loader.dataset):
                continue
            for (folder, suffix), value in outputs.items():
                path = root / folder / batch['dataset_name'][j] / f'{name}{suffix}.npy'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, value[j].detach().cpu().numpy())
        local_offset += len(batch['name'])
        logger.info(f'Encoded {local_offset} local samples; latent shape: {tuple(z_echo.shape)}')
    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', default=1, type=int, help="number of batch size")
    parser.add_argument('--seed', default=2025, type=int)

    parser.add_argument('--distributed', default=False, action='store_true', help='Enable distributed execution with torchrun')
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=0, help="local rank")

    parser.add_argument('--amp', action=argparse.BooleanOptionalAction)
    parser.set_defaults(amp=True)

    parser.add_argument('--cache', default=0.0, type=float)
    parser.add_argument('--replace-rate', '--replace_rate', dest='replace_rate', default=0.2, type=float)
    parser.add_argument('--use-persistent-dataset', '--use_persistent_dataset', dest='use_persistent_dataset', default=True, help="use monai Dataset class")
    parser.add_argument('--smartcache-dataset', '--smartcache_dataset', dest='smartcache_dataset', default=False, help="use monai smartcache Dataset")
    parser.add_argument('--cache-dataset', '--cache_dataset', dest='cache_dataset', default=False, help="use monai cache Dataset")
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', default=8, type=int)

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
    parser.add_argument('--latent-frames', '--latent_frames', dest='latent_frames', type=int, default=32)
    parser.add_argument('--num-keyframes', '--num_keyframes', dest='num_keyframes', type=int, default=8)
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
    os.makedirs(args.latent_dir, exist_ok=True)

    model = Tokenizer_ECHO(args)
    state_dict = torch.load(args.tokenizer_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict['model'])

    get_latent(args, model)
