# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import datetime
from datetime import datetime
from timm.scheduler.cosine_lr import CosineLRScheduler
from model.models_Volume2Temp import DiT_models
from CMRECGDataset_Volume2Temp import *
from utils_volume2temp import *
from transport import create_transport
from train_utils import parse_transport_args, parse_ode_args, parse_sde_args
from tqdm import tqdm



#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    logger = create_logger(args.log_dir, args.distributed)

    # Create model:
    model = DiT_models[args.model](
        input_size=args.latent_size[2],
        input_depths=args.latent_size[-1],
        input_frames=args.num_frames,
        in_channels=args.latent_size[1],
        cond_channels=args.latent_size[1],
    )

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(args.device)  # Create an EMA of the model for use after training

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        print("Load resume checkpoint from: %s" % args.ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        model.load_state_dict(checkpoint_model, strict=True)
        ema.load_state_dict(checkpoint["ema"], strict=True)
        print("Succesfully load EMA model from: %s" % args.ckpt)

    requires_grad(ema, False)

    model = DDP(model.to(args.device), device_ids=[args.rank], find_unused_parameters=True)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity;

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    lr_scheduler = CosineLRScheduler(opt, warmup_t=args.warmup_epochs, warmup_lr_init=1e-6, t_initial=args.epochs,
                                  lr_min=args.min_lr, cycle_limit=1)

    if args.ckpt:
        opt.load_state_dict(checkpoint["opt"])
        print("Succesfully load opt from: %s" % args.ckpt)

    # Setup data:
    logger.info(f"Rank {args.rank}: Starting to load data")
    datasets = get_CMRECG(args)
    train_list, test_list = datasets["train_files"], datasets["test_files"]

    train_transform, test_transform = get_transforms(args)
    loader, _ = get_loader(args, train_list, test_list, train_transform, test_transform)
    dataloader_train, dataloader_test = loader
    logger.info(f"Rank {args.rank}: Finished get dataloaders")

    scale_factor = calculate_scale_factor(dataloader_train, args.device, logger)

    # Prepare models for training:
    if args.distributed:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0 if not args.ckpt else int(args.ckpt.split('/')[-1].split('.')[0]) # xxx/0300000.pt
    log_steps = 0
    running_loss = 0
    start_time = time()

    scaler = GradScaler()

    update_ratio = torch.tensor(0.0, device=args.device)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if args.distributed:
            dataloader_train.sampler.set_epoch(epoch)

        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), ncols=100)
        progress_bar.set_description(f"Epoch {epoch}")

        for i, batch in progress_bar:

            is_update = ((i + 1) % args.accum_iter == 0) or ((i + 1) == len(dataloader_train))

            x_mri = batch["latent_mri"].to(args.device)
            z_ecg = batch["latent_ecg"].to(args.device)
            keyframe_q = batch["keyframe_q"].to(args.device)
            keyframe_idx = batch["keyframe_idx"].to(args.device)
            z_volume = batch["latent_volume"].to(args.device)
            time_mask = batch["time_mask"].to(args.device)
            t_idx = batch["t_given"].to(args.device)

            x_mri = x_mri * scale_factor
            z_volume = z_volume * scale_factor

            #     Mode A: keyframe generation
            #     Mode B: interpolation
            # ----------------------
            if np.random.rand() < args.prob_modeA:
                mask_used = time_mask
                loss_weight = args.loss_weight_modeA
            else:
                mask_used = 1.0 - time_mask
                loss_weight = args.loss_weight_modeB

            mask_used = mask_used.float()

            c = (z_volume, t_idx, z_ecg, keyframe_q)

            model_kwargs = dict(
                context=c,
                mask=mask_used,
            )

            with autocast(enabled=args.amp):
                loss_dict = transport.training_losses(model, x_mri, model_kwargs)

            loss = loss_dict["loss"].mean()
            loss = loss * loss_weight / args.accum_iter

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if is_update:
                if args.amp:
                    scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                with torch.no_grad():
                    param_norm_sq = torch.zeros(1, device=args.device)
                    update_norm_sq = torch.zeros(1, device=args.device)

                    for group in opt.param_groups:
                        lr_group = group['lr']
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            param_norm_sq += p.data.norm(2).pow(2)
                            update = -lr_group * p.grad
                            update_norm_sq += update.norm(2).pow(2)

                    update_ratio = torch.zeros(1, device=args.device)
                    valid_mask = param_norm_sq > 0
                    if valid_mask.item():
                        update_ratio = torch.sqrt(update_norm_sq) / torch.sqrt(param_norm_sq)

                if args.amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none=True)

                update_ema(ema, model.module if args.distributed else model)
                train_steps += 1

            running_loss += loss.item()
            log_steps += 1
            if is_update and (train_steps % args.log_every == 0):
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=args.device)

                if args.distributed:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                else:
                    avg_loss = avg_loss.item()

                avg_grad_norm = grad_norm.detach().clone() if torch.is_tensor(grad_norm) else torch.tensor(grad_norm,device=args.device)
                if args.distributed:
                    dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.SUM)
                    avg_grad_norm = avg_grad_norm.item() / dist.get_world_size()
                else:
                    avg_grad_norm = avg_grad_norm.item()

                avg_update_ratio = update_ratio.detach().clone() if torch.is_tensor(update_ratio) else torch.tensor(update_ratio, device=args.device)
                if args.distributed:
                    dist.all_reduce(avg_update_ratio, op=dist.ReduceOp.SUM)
                    avg_update_ratio = avg_update_ratio.item() / dist.get_world_size()
                else:
                    avg_update_ratio = avg_update_ratio.item()

                logger.info(
                    f"(Step={train_steps:08d}) Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"Grad Norm: {avg_grad_norm:.4f}, "
                    f"Update Ratio: {avg_update_ratio:.4e}, "
                    f"Lr: {opt.param_groups[0]['lr']:.6f}"
                )
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if is_update and (train_steps % args.ckpt_every == 0) and train_steps > 0:
                checkpoint_path = f"{args.model_dir}/{train_steps:07d}.pt"
                if args.rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

        lr_scheduler.step(epoch)

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DiT training', add_help=False)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument("--global_batch_size", type=int, default=16)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    parser.add_argument("--global_seed", type=int, default=2025)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom DiT checkpoint")

    # distributed training parameters
    parser.add_argument('--distributed', default=True, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")

    # enable amp
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='epochs to warmup LR')
    parser.add_argument('--betas', default=(0.9, 0.95))

    # DiT Model parameters
    parser.add_argument('--latent_size', default=[50, 16, 16, 16, 4], help='images input size')
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--num_keyframes", type=int, default=10)

    # Two-stage (keyframe / interpolation) training parameters
    parser.add_argument('--prob_modeA', type=float, default=0.5, help='Probability of using Mode A (keyframe generation) for each training batch')
    parser.add_argument('--loss_weight_modeA', type=float, default=1.5, help='Loss weight multiplier for Mode A (keyframe generation)')
    parser.add_argument('--loss_weight_modeB', type=float, default=0.5, help='Loss weight multiplier for Mode B (interpolation)')

    # Dataset parameters
    parser.add_argument('--cache', default=1.0, type=float)
    parser.add_argument('--replace_rate', default=0.2, type=float)
    parser.add_argument("--use_persistent_dataset", default=True, help="use monai Dataset class")
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", default=False, help="use monai cache Dataset")
    parser.add_argument('--num_workers', default=8, type=int)

    # CMR parameters
    parser.add_argument("--time_samples", default=50, type=int)
    parser.add_argument("--cardiac_pad", default=(128, 128, 13))
    parser.add_argument("--cardiac_size", default=(128, 128, 13))

    # evaluation
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=100000)

    parse_transport_args(parser)

    mode = "ODE"
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_args()

    main(args)
