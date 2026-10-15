import os
import numpy as np
import torch
import logging
import torch.distributed as dist
from monai.data import DataLoader
from monai.utils import first


def init_distributed_mode(args):
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.rank = int(os.environ.get('RANK', '0'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', getattr(args, 'local_rank', 0)))
    args.distributed = args.world_size > 1
    requested = getattr(args, 'device', 'auto')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available.')
        device = torch.device('cuda', args.local_rank if args.distributed else (device.index or 0))
        torch.cuda.set_device(device)
    args.device = device
    args.amp = bool(getattr(args, 'amp', False) and device.type == 'cuda')
    if args.distributed:
        backend = 'nccl' if device.type == 'cuda' and dist.is_nccl_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method=args.dist_url)
    seed = getattr(args, 'global_seed', getattr(args, 'seed', 2025)) + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'Running rank {args.rank}/{args.world_size} on {args.device}.')


def create_logger(log_dir, distributed):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        if not distributed or dist.get_rank() == 0:
            formatter = logging.Formatter('[%(asctime)s] %(message)s')
            for handler in [logging.FileHandler(os.path.join(log_dir, 'run.log')), logging.StreamHandler()]:
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        else:
            logger.addHandler(logging.NullHandler())
    return logger


def calculate_scale_factor(train_loader: DataLoader, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (DataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["latent_echo"].to(device)

    std = torch.std(z)
    if not torch.isfinite(std) or std <= 0:
        raise ValueError('Latent standard deviation must be finite and nonzero.')
    scale_factor = 1 / std
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.SUM)
        scale_factor /= dist.get_world_size()

    return scale_factor
