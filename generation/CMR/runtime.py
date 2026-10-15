import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def add_runtime_args(parser, task):
    parser.add_argument('--split-file', '--split_file', default=None,
                        help='Reuse or create a UKBB subject split manifest.')
    parser.add_argument('--output-dir', '--output_dir', required=True)
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:N')
    parser.add_argument('--cache-dir', '--cache_dir', default=None)
    if task != 'train':
        parser.add_argument('--data-dir', '--data_dir', required=True)
        parser.add_argument('--tokenizer-ckpt', '--tokenizer_ckpt', required=True)
    parser.add_argument('--latent-dir', '--latent_dir', required=True)
    if task == 'train':
        parser.add_argument('--ecg-dim', type=int, default=256,
                            help='Dimension of the projected ECG latents.')
        parser.add_argument('--max-steps', type=int, default=None)
    if task == 'sample':
        parser.add_argument('--skip-metrics', action='store_true',
                            help='Generate samples without loading evaluation networks.')
        parser.add_argument('--max-batches', type=int, default=None)
    for action in parser._actions:
        if action.dest in {'cardiac_pad', 'cardiac_size'}:
            action.nargs, action.type = 3, int
        elif action.dest in {'ECG_size', 'betas'}:
            action.nargs, action.type = 2, (float if action.dest == 'betas' else int)
        elif action.dest in {'unet_channels', 'unet_strides'}:
            action.nargs, action.type = '+', int
        elif action.dest == 'latent_size':
            action.nargs, action.type = 5, int
        elif action.dest in {'use_persistent_dataset', 'smartcache_dataset', 'cache_dataset'}:
            action.type = lambda value: str(value).lower() in {'1', 'true', 'yes'}


def prepare_args(args, task):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args.log_dir = str(output / 'logs') + os.sep
    args.model_dir = str(output / 'checkpoints') + os.sep
    args.sample_dir = str(output / 'samples') + os.sep
    args.cache_dir = args.cache_dir or str(output / 'cache')
    if task == 'latent':
        Path(args.latent_dir).mkdir(parents=True, exist_ok=True)
    for attr in ('log_dir', 'model_dir', 'sample_dir'):
        if attr == 'log_dir' or (task == 'train' and attr == 'model_dir') or (task == 'sample' and attr == 'sample_dir'):
            Path(getattr(args, attr)).mkdir(parents=True, exist_ok=True)
    for attr in ('data_dir', 'latent_dir'):
        if hasattr(args, attr) and not Path(getattr(args, attr)).is_dir():
            raise FileNotFoundError(f'{attr}: {getattr(args, attr)}')
    for attr in ('tokenizer_ckpt', 'ckpt'):
        value = getattr(args, attr, None)
        if value and not Path(value).is_file():
            raise FileNotFoundError(f'{attr}: {value}')
    if hasattr(args, 'latent_dim'):
        args.mri_latent_dim = args.latent_dim
        args.volume_depth = args.cardiac_size[-1]
        args.volume_mid = args.volume_depth // 2
        args.ECG_length = args.ECG_size[-1]
    return args


def init_distributed_mode(args):
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.rank = int(os.environ.get('RANK', '0'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    args.distributed = args.world_size > 1
    requested = getattr(args, 'device', 'auto')
    if requested == 'auto':
        requested = f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(requested)
    if args.device.type == 'cuda':
        if args.distributed or args.device.index is None:
            args.device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(args.device)
    if args.distributed:
        backend = 'nccl' if args.device.type == 'cuda' and dist.is_nccl_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method=args.dist_url)
    args.amp = args.amp and args.device.type == 'cuda'
    seed = getattr(args, 'global_seed', 2025) + args.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'Running rank {args.rank}/{args.world_size} on {args.device}')


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_logger(log_dir, distributed=False):
    logger = logging.getLogger('CardioFAD.CMR')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        if not distributed or dist.get_rank() == 0:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            formatter = logging.Formatter('[%(asctime)s] %(message)s')
            for handler in (logging.FileHandler(Path(log_dir) / 'run.log'), logging.StreamHandler()):
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        else:
            logger.addHandler(logging.NullHandler())
    return logger


def load_latent_records(latent_dir, task, data_dir=None):
    root = Path(latent_dir)
    specifications = {
        'latent_mri': (root / 'latent_mri', '*_4D.nii.gz', '.nii.gz'),
        'latent_ecg': (root / 'latent_ecg', '*_4D.npy', '.npy'),
        'keyframe_q': (root / 'keyframes', '*_q.npy', '_q.npy'),
        'keyframe_idx': (root / 'keyframes', '*_idx.npy', '_idx.npy'),
        'time_mask': (root / 'keyframes', '*_mask.npy', '_mask.npy'),
        't_given': (root / 'latent_slice', '*_t_idx.npy', '_t_idx.npy'),
        f'latent_{task}': (root / f'latent_{task}', '*_4D.nii.gz', '.nii.gz'),
    }
    if task == 'slice':
        specifications['slice_idx'] = (root / 'latent_slice', '*_slice_idx.npy', '_slice_idx.npy')
    if data_dir is not None:
        specifications['x_mri'] = (Path(data_dir) / 'MR', '*/*_4D.nii.gz', '.nii.gz')
    maps = {}
    for key, (directory, pattern, suffix) in specifications.items():
        mapping = {}
        for path in sorted(directory.glob(pattern)):
            ident = path.name[:-len(suffix)]
            if ident in mapping:
                raise ValueError(f'Duplicate {key} identifier: {ident}')
            mapping[ident] = str(path)
        maps[key] = mapping
    subjects = set(maps['latent_mri'])
    if not subjects:
        raise ValueError(f'No CMR latent files found in {root / "latent_mri"}')
    for key, mapping in maps.items():
        missing = subjects - set(mapping)
        extra = set(mapping) - subjects if key != 'x_mri' else set()
        if missing or extra:
            raise ValueError(f'Incompatible {key} files: missing {sorted(missing)[:3]}, '
                             f'extra {sorted(extra)[:3]}. All latent fields must match by filename.')
    return [{key: mapping[ident] for key, mapping in maps.items()} for ident in sorted(subjects)]


def resume_position(checkpoint, batches_per_epoch, accumulation):
    if 'next_epoch' in checkpoint:
        epoch = int(checkpoint['next_epoch'])
        batch = int(checkpoint.get('next_batch_idx', 0))
    elif 'epoch' in checkpoint:
        epoch, batch = int(checkpoint['epoch']) + 1, 0
    else:
        updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
        epoch, updates = divmod(int(checkpoint.get('train_steps', 0)), updates_per_epoch)
        batch = min(updates * accumulation, batches_per_epoch)
    if epoch < 0 or not 0 <= batch <= batches_per_epoch:
        raise ValueError('Invalid epoch or batch position in the checkpoint.')
    if batch == batches_per_epoch:
        epoch, batch = epoch + 1, 0
    return epoch, batch


def save_checkpoint(args, model, ema, opt, train_steps, epoch, scale_factor,
                    scheduler=None, scaler=None, next_epoch=None, next_batch_idx=0):
    if args.rank == 0:
        path = Path(args.model_dir) / f'{train_steps:07d}.pt'
        torch.save({'model': (model.module if args.distributed else model).state_dict(),
                    'ema': ema.state_dict(), 'opt': opt.state_dict(), 'args': args,
                    'train_steps': train_steps, 'epoch': epoch,
                    'next_epoch': epoch + 1 if next_epoch is None else next_epoch,
                    'next_batch_idx': next_batch_idx,
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'scale_factor': float(scale_factor)}, path)
        return path
