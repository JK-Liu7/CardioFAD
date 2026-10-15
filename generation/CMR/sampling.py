# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torch.cuda.amp import autocast
from tqdm import tqdm

from model.Tokenizer_CMR import Tokenizer_CMR
from runtime import init_distributed_mode, create_logger, cleanup
from transport import create_transport, Sampler


def gather_first_dim_variable(x, gather_to_rank0=True):
    if not dist.is_initialized():
        return x
    sizes = [torch.zeros(1, dtype=torch.long, device=x.device) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, torch.tensor([len(x)], device=x.device))
    sizes = [int(n.item()) for n in sizes]
    padded = torch.zeros((max(sizes), *x.shape[1:]), device=x.device, dtype=x.dtype)
    padded[:len(x)] = x
    parts = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(parts, padded)
    if gather_to_rank0 and dist.get_rank() != 0:
        return None
    return torch.cat([part[:size] for part, size in zip(parts, sizes)])


@torch.no_grad()
def run_sampling(args, models, dataset_module, task):
    init_distributed_mode(args)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    logger = create_logger(args.log_dir, args.distributed)
    datasets = dataset_module.get_CMRECG(args)
    loader, _ = dataset_module.get_loader(args, datasets['test_files'], dataset_module.get_transforms(args))
    if not len(loader):
        raise ValueError('No sampling batches found; check the paired data and latent directories.')
    tokenizer = Tokenizer_CMR(args)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location='cpu', weights_only=False)['model'])
    tokenizer.to(args.device).eval().requires_grad_(False)
    model = models[args.model](input_size=args.latent_size[2], input_depths=args.latent_size[-1],
                              input_frames=args.num_frames, in_channels=args.latent_size[1],
                              cond_channels=args.latent_size[1], time_dims=args.proj_dim)
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['ema'])
    model.to(args.device).eval().requires_grad_(False)
    scale_factor = args.scale_factor if args.scale_factor is not None else checkpoint.get('scale_factor')
    if scale_factor is None or not np.isfinite(float(scale_factor)) or float(scale_factor) <= 0:
        raise ValueError('Pass --scale_factor for a checkpoint without a positive saved scale_factor.')
    scale_factor = float(scale_factor)
    logger.info(f'Loaded {args.ckpt}; latent scaling factor {scale_factor}')
    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    sampler = Sampler(transport)
    if args.mode == 'ODE':
        if args.likelihood:
            raise ValueError('Likelihood evaluation is not supported by two-stage conditional sampling.')
        sample_fn = sampler.sample_ode(sampling_method=args.sampling_method, num_steps=args.num_sampling_steps,
                                       atol=args.atol, rtol=args.rtol, reverse=args.reverse)
    else:
        sample_fn = sampler.sample_sde(sampling_method=args.sampling_method, diffusion_form=args.diffusion_form,
                                       diffusion_norm=args.diffusion_norm, last_step=args.last_step,
                                       last_step_size=args.last_step_size, num_steps=args.num_sampling_steps)
    if not args.skip_metrics:
        from monai.losses.perceptual import PerceptualLoss
        from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
        from torchmetrics.image.fid import FrechetInceptionDistance
        from calculate_fvd import calculate_fvd
        perceptual = PerceptualLoss(spatial_dims=2, network_type='squeeze').eval().to(args.device)
        fid = FrechetInceptionDistance(feature=2048, normalize=True, antialias=False,
                                      dist_sync_on_step=False).to(args.device)
        totals = torch.zeros(4, device=args.device)
        all_real, all_generated = [], []
    generated_count = 0
    for batch_idx, batch in enumerate(tqdm(loader)):
        ecg, q = batch['latent_ecg'].to(args.device), batch['keyframe_q'].to(args.device)
        t_idx = batch['t_given'].to(args.device)
        condition = batch[f'latent_{task}'].to(args.device) * scale_factor
        if task == 'slice':
            context = (condition, batch['slice_idx'].to(args.device), t_idx, ecg, q)
        else:
            context = (condition, t_idx, ecg, q)
        mask = batch['time_mask'].to(args.device).float()[:, :, None, None, None, None]
        z = torch.randn(len(ecg), args.num_frames, *args.latent_size[1:], device=args.device)
        with autocast(enabled=args.amp):
            def keyframe_velocity(x, t, **kwargs):
                return model(x, t, **kwargs) * mask
            stage1_options = {'update_mask': mask} if args.mode == 'SDE' else {}
            keyframes = sample_fn(z, keyframe_velocity, context=context, **stage1_options)[-1]
            initial = torch.randn_like(keyframes) * (1 - mask) + keyframes * mask
            def interpolation_velocity(x, t, **kwargs):
                return model(x, t, **kwargs) * (1 - mask)
            stage2_options = {'update_mask': 1 - mask} if args.mode == 'SDE' else {}
            latent = sample_fn(initial, interpolation_velocity, context=context, **stage2_options)[-1]
            latent = rearrange(latent, 'b t c h w d -> (b t) c d h w')
            samples = tokenizer.MRI_tok.decode(latent / scale_factor)
            samples = rearrange(samples, '(b t) c d h w -> b t c h w d', b=len(ecg), t=args.num_frames)
            samples = samples.float().clamp(0, 1)
        if not torch.isfinite(samples).all():
            raise FloatingPointError('Sampling produced non-finite voxels.')
        for index, name in enumerate(batch['image_filename']):
            volume = rearrange(samples[index, :, 0], 't h w d -> h w d t').cpu().numpy()
            nib.save(nib.Nifti1Image(volume, np.eye(4)), Path(args.sample_dir) / f'{name}.nii.gz')
            if task == 'volume':
                reference = rearrange(batch['x_mri'][index], 't h w d -> h w d t').float().cpu().numpy()
                nib.save(nib.Nifti1Image(reference, np.eye(4)), Path(args.sample_dir) / f'{name}_GT.nii.gz')
            generated_count += 1
        if not args.skip_metrics:
            real = batch['x_mri'].to(args.device).float().unsqueeze(2)
            real_slices = rearrange(real, 'b t c h w d -> (b t d) c h w').as_subclass(torch.Tensor)
            fake_slices = rearrange(samples, 'b t c h w d -> (b t d) c h w').as_subclass(torch.Tensor)
            perceptual_total = torch.zeros((), device=args.device)
            for start in range(0, len(real_slices), args.fid_micro_bs):
                r, f = real_slices[start:start + args.fid_micro_bs], fake_slices[start:start + args.fid_micro_bs]
                fid.update(r.expand(-1, 3, -1, -1), real=True)
                fid.update(f.expand(-1, 3, -1, -1), real=False)
                perceptual_total += perceptual(f, r).mean() * len(r)
            batch_size = len(ecg)
            real_volumes = rearrange(real, 'b t c h w d -> (b t) c h w d')
            fake_volumes = rearrange(samples, 'b t c h w d -> (b t) c h w d')
            totals[0] += peak_signal_noise_ratio(fake_volumes, real_volumes) * batch_size
            totals[1] += structural_similarity_index_measure(fake_volumes, real_volumes) * batch_size
            totals[2] += perceptual_total / len(real_slices) * batch_size
            totals[3] += batch_size
            real_all = gather_first_dim_variable(real[..., args.volume_mid].as_subclass(torch.Tensor))
            fake_all = gather_first_dim_variable(samples[..., args.volume_mid].as_subclass(torch.Tensor))
            if args.rank == 0:
                all_real.append(real_all.cpu())
                all_generated.append(fake_all.cpu())
        if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
            break
    logger.info(f'Rank {args.rank}: saved {generated_count} generated CMR sequences to {args.sample_dir}')
    if not args.skip_metrics:
        if args.distributed:
            dist.all_reduce(totals)
        values = (totals[:3] / totals[3]).tolist()
        fid_value = float(fid.compute())
        if args.rank == 0:
            logger.info(f'PSNR={values[0]:.4f}, SSIM={values[1]:.4f}, LPIPS={values[2]:.4f}, FID={fid_value:.4f}')
            del model, tokenizer, perceptual, fid
            if args.device.type == 'cuda':
                torch.cuda.empty_cache()
            result = calculate_fvd(torch.cat(all_real).to(args.device), torch.cat(all_generated).to(args.device),
                                   args.device, method='styleganv', only_final=True)
            logger.info(f'FVD={float(result["value"][0]):.4f}')
    cleanup()
