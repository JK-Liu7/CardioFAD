# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import torch.distributed as dist
from einops import rearrange
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

from model.models_Frame2Video import DiT_models

from EchoDataset import *
from utils import *
from transport import create_transport, Sampler

from train_utils import parse_transport_args, parse_ode_args, parse_sde_args

import math
from tqdm import tqdm

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():

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
    if dist.is_initialized():
        dist.destroy_process_group()

def save_checkpoint(args, model, ema, optimizer, scheduler, scaler, train_steps,
                    epoch, next_batch_idx, scale_factor):
    if args.rank != 0:
        return
    checkpoint_path = os.path.join(args.model_dir, f'{train_steps:07d}.pt')
    checkpoint = {
        'model': (model.module if args.distributed else model).state_dict(),
        'ema': ema.state_dict(), 'opt': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
        'args': vars(args), 'train_steps': train_steps,
        'epoch': max(0, epoch - 1) if next_batch_idx == 0 else epoch,
        'next_epoch': epoch, 'next_batch_idx': next_batch_idx,
        'scale_factor': float(scale_factor),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def main(args):
    """
    Trains a new DiT model.
    """
    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    logger = create_logger(args.log_dir, args.distributed)


    model = DiT_models[args.model](
        input_size=args.latent_size[-1],
        input_frames=args.latent_size[1],
        in_channels=args.latent_size[0],
        cond_channels=args.latent_size[0],
        time_dims=args.condition_dim,
    )

    ema = deepcopy(model).to(args.device)

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        print("Load resume checkpoint from: %s" % args.ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        model.load_state_dict(checkpoint_model, strict=True)

        ema.load_state_dict(checkpoint["ema"], strict=True)
        print("Succesfully load EMA model from: %s" % args.ckpt)

    requires_grad(ema, False)

    model = model.to(args.device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank] if args.device.type == 'cuda' else None,
                    find_unused_parameters=True)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    lr_scheduler = CosineLRScheduler(opt, warmup_t=args.warmup_epochs, warmup_lr_init=1e-6, t_initial=args.epochs,
                                  lr_min=args.min_lr, cycle_limit=1)

    if args.ckpt:
        opt.load_state_dict(checkpoint["opt"])
        if 'scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
        print("Succesfully load opt from: %s" % args.ckpt)

    logger.info(f"Rank {args.rank}: Starting to load data")

    train_, _, _ = create_dataloaders(
        data_dir=args.data_dir,
        latent_dir=args.latent_dir,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=args.rank,
        world_size=args.world_size,
    )
    dataloader_train, train_sampler = train_

    logger.info(f"Rank {args.rank}: Finished get dataloaders")

    if len(dataloader_train) == 0:
        raise ValueError('No training batches; check data paths, split labels and batch size.')
    scale_factor = (torch.tensor(checkpoint['scale_factor'], device=args.device)
                    if args.ckpt and 'scale_factor' in checkpoint
                    else calculate_scale_factor(dataloader_train, args.device, logger))

    if not args.ckpt:
        update_ema(ema, model.module if args.distributed else model, decay=0)
    model.train()
    ema.eval()

    train_steps = int(checkpoint.get('train_steps', 0)) if args.ckpt else 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    scaler = GradScaler(enabled=args.amp)
    if args.ckpt and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    start_epoch, start_batch = 0, 0
    if args.ckpt:
        if 'next_epoch' in checkpoint:
            start_epoch = int(checkpoint['next_epoch'])
            start_batch = int(checkpoint.get('next_batch_idx', 0))
        elif 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch']) + 1
        else:
            updates_per_epoch = math.ceil(len(dataloader_train) / args.accum_iter)
            start_epoch, completed_updates = divmod(train_steps, updates_per_epoch)
            start_batch = completed_updates * args.accum_iter
    if start_batch > len(dataloader_train):
        raise ValueError('Checkpoint batch progress exceeds the current loader length; keep the original data and batch size.')
    next_epoch, next_batch = start_epoch, start_batch

    update_ratio = torch.tensor(0.0, device=args.device)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), ncols=100)
        progress_bar.set_description(f"Epoch {epoch}")

        for i, batch in progress_bar:
            if epoch == start_epoch and i < start_batch:
                continue

            is_update = ((i + 1) % args.accum_iter == 0) or ((i + 1) == len(dataloader_train))
            accumulation_start = (i // args.accum_iter) * args.accum_iter
            accumulation_size = min(args.accum_iter, len(dataloader_train) - accumulation_start)

            x_echo = batch["latent_echo"].to(args.device)
            z_ecg = batch["latent_ecg"].to(args.device)
            keyframe_q = batch["keyframe_q"].to(args.device)

            time_mask = batch["time_mask"].to(args.device)
            frame_idx = batch["frame_idx"].to(args.device)

            z_frame = batch["z_frame"].to(args.device)

            x_echo = x_echo * scale_factor
            z_frame = z_frame * scale_factor

            if np.random.rand() < args.prob_modeA:
                mask_used = time_mask
                loss_weight = args.loss_weight_modeA
            else:
                mask_used = 1.0 - time_mask
                loss_weight = args.loss_weight_modeB

            mask_used = mask_used.float()

            c = (z_frame, frame_idx, z_ecg, keyframe_q)

            model_kwargs = dict(
                context=c,
                mask=mask_used,
            )

            with autocast(enabled=args.amp):
                loss_dict = transport.training_losses(model, x_echo, model_kwargs)

            loss = loss_dict["loss"].mean()
            loss = loss * loss_weight / accumulation_size

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
                next_epoch, next_batch = epoch, i + 1

            running_loss += loss.item()
            log_steps += 1

            if is_update and (train_steps % args.log_every == 0):

                if args.device.type == 'cuda':
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

            if is_update and (train_steps % args.ckpt_every == 0) and train_steps > 0:
                checkpoint_path = save_checkpoint(args, model, ema, opt, lr_scheduler, scaler,
                                                  train_steps, next_epoch, next_batch, scale_factor)
                if checkpoint_path:
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

        lr_scheduler.step(epoch)
        next_epoch, next_batch = epoch + 1, 0

    if train_steps:
        checkpoint_path = save_checkpoint(args, model, ema, opt, lr_scheduler, scaler,
                                          train_steps, next_epoch, next_batch, scale_factor)
        if checkpoint_path:
            logger.info(f'Saved final checkpoint to {checkpoint_path}')
    model.eval()

    logger.info("Done!")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DiT training')

    parser.add_argument("--dataset", type=str, default='EchoNet-Pediatric-A4C')

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--global-batch-size', '--global_batch_size', dest='global_batch_size', type=int, default=64)
    parser.add_argument('--accum-iter', '--accum_iter', dest='accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    parser.add_argument('--global-seed', '--global_seed', dest='global_seed', type=int, default=2025)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=10000)

    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom DiT checkpoint")

    parser.add_argument('--distributed', default=False, action='store_true', help='Enable distributed execution with torchrun')
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=0, help="local rank")

    parser.add_argument('--amp', action=argparse.BooleanOptionalAction)
    parser.set_defaults(amp=False)

    parser.add_argument('--weight-decay', '--weight_decay', dest='weight_decay', type=float, default=0, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (absolute lr)')
    parser.add_argument('--min-lr', '--min_lr', dest='min_lr', type=float, default=1e-5, help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup-epochs', '--warmup_epochs', dest='warmup_epochs', type=int, default=20, help='epochs to warmup LR')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95))

    parser.add_argument('--latent-size', '--latent_size', dest='latent_size', type=int, nargs=4, default=[16, 32, 14, 14], help='images input size')

    parser.add_argument('--prob-modeA', '--prob_modeA', dest='prob_modeA', type=float, default=0.5, help='Probability of using Mode A (keyframe generation) for each training batch')
    parser.add_argument('--loss-weight-modeA', '--loss_weight_modeA', dest='loss_weight_modeA', type=float, default=1.5, help='Loss weight multiplier for Mode A (keyframe generation)')
    parser.add_argument('--loss-weight-modeB', '--loss_weight_modeB', dest='loss_weight_modeB', type=float, default=0.5, help='Loss weight multiplier for Mode B (interpolation)')

    parser.add_argument('--cache', default=1.0, type=float)
    parser.add_argument('--replace-rate', '--replace_rate', dest='replace_rate', default=0.2, type=float)
    parser.add_argument('--use-persistent-dataset', '--use_persistent_dataset', dest='use_persistent_dataset', default=True, help="use monai Dataset class")
    parser.add_argument('--smartcache-dataset', '--smartcache_dataset', dest='smartcache_dataset', default=False, help="use monai smartcache Dataset")
    parser.add_argument('--cache-dataset', '--cache_dataset', dest='cache_dataset', default=False, help="use monai cache Dataset")
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', default=8, type=int)

    parser.add_argument('--condition-dim', type=int, default=256)

    parse_transport_args(parser)

    mode = "ODE"
    if mode == "ODE":
        parse_ode_args(parser)

    elif mode == "SDE":
        parse_sde_args(parser)

    parser.add_argument('--data-dir', '--data_dir', required=True)
    parser.add_argument('--latent-dir', '--latent_dir', required=True)
    parser.add_argument('--output-dir', '--output_dir', required=True)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')

    args = parser.parse_args()


    args.log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    args.model_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(args.model_dir, exist_ok=True)

    main(args)
