import torch
import logging
import torch.distributed as dist
from monai.data import DataLoader
from monai.utils import first



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
    z = check_data["latent_mri"].to(device)

    scale_factor = 1 / torch.std(z).clamp_min(1e-8)
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.SUM)
        scale_factor /= dist.get_world_size()

    return scale_factor
