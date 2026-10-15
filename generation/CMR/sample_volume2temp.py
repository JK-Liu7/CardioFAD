# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import CMRECGDataset_sample_volume as dataset_module
from model.models_Volume2Temp import DiT_models
from runtime import add_runtime_args, prepare_args
from sampling import run_sampling, gather_first_dim_variable
from train_utils import parse_transport_args, parse_ode_args, parse_sde_args


def main(args):
    run_sampling(args, DiT_models, dataset_module, "volume")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("--mode", type=str, default="ODE", choices=["ODE", "SDE"])
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--per-proc-batch-size", type=int, default=1)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU')
    parser.add_argument("--global_batch_size", type=int, default=16)

    parser.add_argument("--fid_micro_bs", type=int, default=512)

    parser.add_argument("--scale-factor", "--scale_factor", type=float, default=None)

    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=2025)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", required=True, help="Path to the DiT checkpoint.")

    parser.add_argument('--distributed', default=False, action='store_true',
                        help='Enable distributed sampling with torchrun; plain python defaults to one GPU.')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")

    parser.add_argument('--amp', action=argparse.BooleanOptionalAction)
    parser.set_defaults(amp=True)

    parser.add_argument('--cache', default=1.0, type=float)
    parser.add_argument('--replace_rate', default=0.2, type=float)
    parser.add_argument("--use_persistent_dataset", default=False, help="use monai Dataset class")
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", default=False, help="use monai cache Dataset")
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--latent_size', default=[50, 16, 16, 16, 4], help='images input size')
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--num_keyframes", type=int, default=10)

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

    parse_transport_args(parser)

    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", default="ODE", choices=["ODE", "SDE"])
    mode = mode_parser.parse_known_args()[0].mode
    if mode == "ODE":
        parse_ode_args(parser)

    elif mode == "SDE":
        parse_sde_args(parser)

    add_runtime_args(parser, "sample")
    args = prepare_args(parser.parse_args(), "sample")

    args.mri_latent_dim = args.latent_dim
    args.volume_depth = args.cardiac_size[-1]
    args.ECG_length = args.ECG_size[-1]
    args.volume_mid = args.volume_depth // 2

    main(args)
