import argparse
import os
import timeit
from datetime import datetime
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from torch.nn import L1Loss, MSELoss
import timm.optim.optim_factory as optim_factory
from monai.networks.nets import PatchDiscriminator
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from CMRECGDataset import *
from utils import *
from model.Tokenizer_CMR import Tokenizer_CMR

from timm.scheduler.cosine_lr import CosineLRScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from torch.cuda.amp import GradScaler, autocast

def train(args, model, discriminator):

    init_distributed_mode(args)
    args.amp = args.amp and args.device.type == "cuda"
    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    logger = create_logger(args.log_dir, args.distributed)

    logger.info(f"Rank {args.rank}: Starting to load data")
    datasets = get_CMRECG(args)
    train_list, val_list = datasets["train_files"], datasets["val_files"]

    train_transform, val_transform = get_transforms(args)

    loader, _ = get_loader(args, train_list, val_list, train_transform, val_transform)
    dataloader_train, dataloader_val = loader
    logger.info(f"Rank {args.rank}: Finished get dataloaders")

    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
        print("Use l2 loss")
    else:
        intensity_loss = L1Loss(reduction="mean")
        print("Use l1 loss")

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = (
        PerceptualLoss(spatial_dims=2, network_type="squeeze").eval().to(args.device)
    )

    if len(dataloader_train) == 0 or len(dataloader_val) == 0:
        raise ValueError("Training and validation loaders must contain at least one batch.")

    accumulation_steps = args.gradient_accumulation_steps

    model = model.to(args.device)
    discriminator = discriminator.to(args.device)

    model_without_ddp = model
    discriminator_without_ddp = discriminator

    if args.distributed:
        if args.device.type == "cuda":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank] if args.device.type == "cuda" else None, find_unused_parameters=True)
        model_without_ddp = model.module
        if args.device.type == "cuda":
            discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        discriminator = DistributedDataParallel(discriminator, device_ids=[args.local_rank] if args.device.type == "cuda" else None)
        discriminator_without_ddp = discriminator.module

    logger.info(f"AE Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ae_param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    dis_param_groups = optim_factory.param_groups_weight_decay(discriminator_without_ddp, args.weight_decay)

    if args.opt == "adam":
        optimizer_g = optim.Adam(params=ae_param_groups, lr=args.lr, weight_decay=0)
        optimizer_d = optim.Adam(params=dis_param_groups, lr=args.lr, weight_decay=0)
    elif args.opt == "adamw":
        optimizer_g = optim.AdamW(params=ae_param_groups, lr=args.lr, weight_decay=0)
        optimizer_d = optim.AdamW(params=dis_param_groups, lr=args.lr, weight_decay=0)

    scheduler_g = CosineLRScheduler(optimizer_g, warmup_t=args.warmup_epochs, warmup_lr_init=1e-6, t_initial=args.epochs,
                          lr_min=args.min_lr, cycle_limit=1)
    scheduler_d = CosineLRScheduler(optimizer_d, warmup_t=args.warmup_epochs, warmup_lr_init=1e-6, t_initial=args.epochs,
                          lr_min=args.min_lr, cycle_limit=1)

    scaler_g = GradScaler(enabled=args.amp)
    scaler_d = GradScaler(enabled=args.amp)

    val_interval = args.val_interval
    best_train_loss = 1000
    start_epoch = 0
    max_epochs = args.epochs
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        discriminator_without_ddp.load_state_dict(checkpoint["discriminator"], strict=True)
        optimizer_g.load_state_dict(checkpoint["opt"])
        if "opt_d" in checkpoint:
            optimizer_d.load_state_dict(checkpoint["opt_d"])
        if "scheduler_g" in checkpoint:
            scheduler_g.load_state_dict(checkpoint["scheduler_g"])
            scheduler_d.load_state_dict(checkpoint["scheduler_d"])
        if "scaler_g" in checkpoint:
            scaler_g.load_state_dict(checkpoint["scaler_g"])
            scaler_d.load_state_dict(checkpoint["scaler_d"])
        start_epoch = checkpoint["epoch"] + 1
        best_train_loss = checkpoint.get("best_train_loss", best_train_loss)
    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)

    logger.info(f"Rank {args.rank}: Start Training")

    for epoch in range(start_epoch, max_epochs):

        if args.distributed:
            dataloader_train.sampler.set_epoch(epoch)
            dataloader_val.sampler.set_epoch(epoch)

        torch.cuda.empty_cache()

        model.train()
        discriminator.train()

        train_epoch_losses = {"rec_loss": 0, "kl_loss": 0, "per_loss": 0, "disc_loss": 0, "adv_loss": 0,
                              "conv_loss": 0, "k2t_loss": 0, "cl_loss": 0}

        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), ncols=100)
        progress_bar.set_description(f"Epoch {epoch}")

        for i, batch in progress_bar:
            window_start = (i // accumulation_steps) * accumulation_steps
            window_size = min(accumulation_steps, len(dataloader_train) - window_start)
            step_now = (i + 1) % accumulation_steps == 0 or i + 1 == len(dataloader_train)
            discriminator.requires_grad_(False)

            x_mri = batch["image"].to(args.device)
            x_mri = rearrange(x_mri, "b c h w d -> b c d h w")
            x_ecg = batch["ecg"].to(args.device)
            x_ecg = x_ecg.unsqueeze(1)

            with autocast(enabled=args.amp):

                if epoch < args.warmup_epochs:
                    out = model(x_mri, x_ecg=None, stage="mri_pretrain")
                else:
                    out = model(x_mri, x_ecg, stage="full")

                x_rec = out['rec_mri']

                generator_loss = get_adv_loss(x_rec, discriminator, adv_loss)
                rec_loss, per_loss = get_train_loss(x_mri, x_rec, intensity_loss, loss_perceptual)

                if epoch < args.warmup_epochs:
                    losses = {
                        "rec_loss": rec_loss,
                        "kl_loss": out["loss_kl"],
                        "per_loss": per_loss,
                        "adv_loss": generator_loss,
                    }
                else:
                    losses = {
                        "rec_loss": rec_loss,
                        "kl_loss": out["loss_kl"],
                        "per_loss": per_loss,
                        "adv_loss": generator_loss,
                        "conv_loss": out["loss_conv"],
                        "k2t_loss": out["loss_k2t"],
                        "cl_loss": out["loss_cl"],
                    }

                for key, val in losses.items():
                    if not torch.isfinite(val).all():
                        raise FloatingPointError(f"Non-finite {key} at epoch {epoch}, batch {i}")

                if epoch < args.warmup_epochs:
                    loss_g = train_loss_weighted_sum_stage1(args, losses) / window_size
                else:
                    loss_g = train_loss_weighted_sum_stage2(args, losses) / window_size

            for loss_name, loss_value in losses.items():
                train_epoch_losses[loss_name] += loss_value.item()

            if args.amp:
                scaler_g.scale(loss_g).backward()

                if step_now:
                    scaler_g.unscale_(optimizer_g)
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                    optimizer_g.zero_grad(set_to_none=True)
            else:
                loss_g.backward()
                if step_now:
                    optimizer_g.step()
                    optimizer_g.zero_grad(set_to_none=True)

            discriminator.requires_grad_(True)
            with autocast(enabled=args.amp):
                disc_loss = get_discriminator_loss(x_mri, x_rec, discriminator, adv_loss)
                loss_d = args.adv_weight * disc_loss / window_size

                if not torch.isfinite(loss_d).all():
                    raise FloatingPointError(f"Non-finite discriminator loss at epoch {epoch}, batch {i}")

                train_epoch_losses['disc_loss'] += disc_loss.item()
                losses.update({'disc_loss': disc_loss})

            if args.amp:
                scaler_d.scale(loss_d).backward()
                if step_now:
                    scaler_d.unscale_(optimizer_d)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad(set_to_none=True)
            else:
                loss_d.backward()
                if step_now:
                    optimizer_d.step()
                    optimizer_d.zero_grad(set_to_none=True)

            if (i != 0 and i % (max(1, len(dataloader_train) // 4)) == 0) or (i == len(dataloader_train) - 1):
                if args.rank == 0:
                    print(
                        "Epoch: {}, L_rec: {:.4f}, L_kl: {:.1f}, L_per: {:.4f}, L_disc: {:.3f}, L_conv: {:.4f}, "
                        "L_k2t: {:.4f}, L_cl: {:.6f}, lr: {:.6f}".format(
                            epoch, train_epoch_losses["rec_loss"]/(i+1), train_epoch_losses["kl_loss"]/(i+1),
                            train_epoch_losses["per_loss"]/(i+1), train_epoch_losses["disc_loss"]/(i+1),
                            train_epoch_losses["conv_loss"]/(i+1), train_epoch_losses["k2t_loss"]/(i+1),
                            train_epoch_losses["cl_loss"]/(i+1), optimizer_g.param_groups[0]['lr'])
                    )

                logger.info(
                        "Epoch: {}, L_rec: {:.4f}, L_kl: {:.1f}, L_per: {:.4f}, L_disc: {:.3f}, L_conv: {:.4f}, "
                        "L_k2t: {:.4f}, L_cl: {:.6f}, lr: {:.6f}".format(
                            epoch, train_epoch_losses["rec_loss"]/(i+1), train_epoch_losses["kl_loss"]/(i+1),
                            train_epoch_losses["per_loss"]/(i+1), train_epoch_losses["disc_loss"]/(i+1),
                            train_epoch_losses["conv_loss"]/(i+1), train_epoch_losses["k2t_loss"]/(i+1),
                            train_epoch_losses["cl_loss"]/(i+1), optimizer_g.param_groups[0]['lr'])
                    )

        scheduler_g.step(epoch)
        scheduler_d.step(epoch)

        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(dataloader_train)

        if epoch < args.warmup_epochs:
            loss_g_total = train_loss_weighted_sum_stage1(args, train_epoch_losses)
        else:
            loss_g_total = train_loss_weighted_sum_stage2(args, train_epoch_losses)

        if epoch % args.ckpt_interval == 0 or epoch + 1 == args.epochs or loss_g_total < best_train_loss:
            best_train_loss = loss_g_total
            if args.rank == 0:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "opt": optimizer_g.state_dict(),
                    "opt_d": optimizer_d.state_dict(),
                    "scheduler_g": scheduler_g.state_dict(),
                    "scheduler_d": scheduler_d.state_dict(),
                    "scaler_g": scaler_g.state_dict(),
                    "scaler_d": scaler_d.state_dict(),
                    "best_train_loss": best_train_loss,
                    "discriminator": discriminator_without_ddp.state_dict(),
                    "epoch": epoch,
                    "args": vars(args)
                }
                checkpoint_path = f"{args.model_dir}/Tokenizer_CMR_epoch{epoch}.pth"
                torch.save(checkpoint, checkpoint_path)

        if epoch % val_interval == 0:
            model.eval()

            metric_epoch = {"psnr": 0, "ssim": 0, "lpips": 0}

            for batch in dataloader_val:
                with torch.no_grad():
                    with autocast(enabled=args.amp):

                        x = batch["image"].to(args.device)
                        x = rearrange(x, "b c h w d -> b c d h w")
                        latent = model_without_ddp.MRI_tok.encode(x)
                        x_rec = model_without_ddp.MRI_tok.decode(latent)

                        x_ = rearrange(x, 'b c d h w -> (b d) c h w').contiguous()
                        rec_ = rearrange(x_rec, 'b c d h w -> (b d) c h w').contiguous()
                        lpips_val = loss_perceptual(rec_, x_)
                        lpips_val = lpips_val.mean()

                        x = rearrange(x, "b c d h w -> b c h w d")
                        x_rec = rearrange(x_rec, "b c d h w -> b c h w d")

                        metrics = {
                            "psnr": psnr(x_rec, x),
                            "ssim": ssim(x_rec, x),
                            "lpips": lpips_val
                        }

                    for metric_name, metric_value in metrics.items():
                        metric_epoch[metric_name] += metric_value

            for key in metric_epoch:
                metric_epoch[key] /= len(dataloader_val)

            if args.rank == 0:
                print(
                    "Validation Epoch: {}, PSNR: {:.3f}, SSIM: {:.4f}, LPIPS: {:.4f}".format
                     (epoch,  metric_epoch['psnr'], metric_epoch['ssim'],
                        metric_epoch['lpips'])
                )

            logger.info(
                    "Validation Epoch: {}, PSNR: {:.3f}, SSIM: {:.4f}, LPIPS: {:.4f}".format
                     (epoch, metric_epoch['psnr'], metric_epoch['ssim'],
                        metric_epoch['lpips'])
                )

    if args.distributed:
        cleanup()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True)
    parser.add_argument("--split-file", "--split_file", default=None,
                        help="Reuse or create a UKBB subject split manifest.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir")
    parser.add_argument("--resume", help="Path to a trusted tokenizer training checkpoint.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", default=800, type=int, help="number of training epochs")
    parser.add_argument('--warmup_epochs', type=int, default=10, help='epochs to warmup LR')
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument("--val_interval", default=5, type=int)
    parser.add_argument("--ckpt_interval", default=10, type=int)

    parser.add_argument('--distributed', default=False, action='store_true', help='Use torchrun distributed training')
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0, help="local rank")

    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(amp=True)

    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument("--min_lr", default=2e-5, type=float)
    parser.add_argument("--opt", default="adam", choices=("adam", "adamw"), help="optimization algorithm")
    parser.add_argument("--weight_decay", default=0, type=float, help="regularization weight")

    parser.add_argument('--cache', default=0.0, type=float)
    parser.add_argument('--replace_rate', default=0.2, type=float)
    parser.add_argument("--use_persistent_dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", default=False, help="use monai cache Dataset")
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument("--time_samples", default=50, type=int)
    parser.add_argument("--cardiac_pad", default=(128, 128, 13), type=int, nargs=3)
    parser.add_argument("--cardiac_size", default=(128, 128, 13), type=int, nargs=3)

    parser.add_argument("--ECG_size", default=(12, 500), type=int, nargs=2)
    parser.add_argument("--ECG_enc_patch", default=10, type=int)
    parser.add_argument("--ECG_enc_dim", default=384, type=int)
    parser.add_argument("--ECG_enc_depth", default=4, type=int)

    parser.add_argument("--proj_dim", default=256, type=int)
    parser.add_argument("--selector_hidden", default=256, type=int)
    parser.add_argument("--selector_layers", default=2, type=int)
    parser.add_argument("--selector_heads", default=4, type=int)

    parser.add_argument("--unet_channels", default=(16, 32, 64, 128), type=int, nargs='+')
    parser.add_argument("--unet_strides", default=(2, 1, 1), type=int, nargs='+')

    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--num_keyframes", type=int, default=10)
    parser.add_argument("--num_bins", type=int, default=4)

    parser.add_argument("--sigma_align", type=float, default=0.2)
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

    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("epochs, batch_size and gradient_accumulation_steps must be positive")
    if args.val_interval < 1 or args.ckpt_interval < 1:
        parser.error("val_interval and ckpt_interval must be positive")
    if args.embedding_dim != args.l_dim + args.h_dim:
        parser.error("embedding_dim must equal l_dim + h_dim")
    if not os.path.isdir(args.data_dir):
        parser.error("data_dir must be an existing directory")
    if args.resume and not os.path.isfile(args.resume):
        parser.error("resume must point to an existing checkpoint")
    args.log_dir = os.path.join(args.output_dir, "logs")
    args.model_dir = os.path.join(args.output_dir, "checkpoints")
    args.cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    args.mri_latent_dim = args.latent_dim
    args.volume_depth = args.cardiac_size[-1]
    args.ECG_length = args.ECG_size[-1]

    model = Tokenizer_CMR(args)

    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=2,
        channels=32,
        in_channels=1,
        out_channels=1)

    train(args, model, discriminator)

