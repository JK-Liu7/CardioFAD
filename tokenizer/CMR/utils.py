import os
from datetime import datetime
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
from collections import OrderedDict
from monai.data import DataLoader
from monai.utils import first
from einops import rearrange


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def init_distributed_mode(args):
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.world_size = 1
    args.rank = 0

    if args.distributed:
        dist.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.device = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.device)
        num_gpus = torch.cuda.device_count()
        print(f"Setting up distributed training with {num_gpus} GPUs available")
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        args.device = torch.device("cuda:0")
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0


def create_logger(log_dir, distributed):
    """
    Create a logger that writes to a log file and stdout.
    """
    today_date = datetime.today().strftime('%Y.%m.%d')
    if distributed:
        if dist.get_rank() == 0:  # real logger
            logging.basicConfig(filename=log_dir + f"{today_date}_c16.log",
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
            logger = logging.getLogger(__name__)
        else:  # dummy logger (does nothing)
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
    else:
        logging.basicConfig(filename=log_dir + f"{today_date}.log",
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        logger = logging.getLogger(__name__)

    return logger


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


def get_train_loss(x, x_rec, intensity_loss, loss_perceptual):
    loss_rec = intensity_loss(x_rec, x)
    x_ = rearrange(x, 'b c d h w -> (b d) c h w').contiguous()
    rec_ = rearrange(x_rec, 'b c d h w -> (b d) c h w').contiguous()
    loss_per = loss_perceptual(rec_, x_)

    return loss_rec, loss_per

def get_adv_loss(x_rec, discriminator, adv_loss):
    logits_fake = discriminator(x_rec.contiguous().float())[-1]
    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
    return generator_loss

def get_discriminator_loss(x, x_rec, discriminator, adv_loss):
    logits_fake = discriminator(x_rec.contiguous().detach())[-1]
    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
    logits_real = discriminator(x.contiguous().detach())[-1]
    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
    loss_d = (loss_d_fake + loss_d_real) * 0.5
    return loss_d

def train_loss_weighted_sum_stage1(args, losses):
    return (losses["rec_loss"] + args.kl_weight * losses["kl_loss"] + args.per_weight * losses["per_loss"] + args.adv_weight * losses["adv_loss"])

def train_loss_weighted_sum_stage2(args, losses):
    return (losses["rec_loss"] + args.kl_weight * losses["kl_loss"] + args.per_weight * losses["per_loss"] + args.adv_weight * losses["adv_loss"]
             + args.k2t_weight * losses["k2t_loss"] + args.cl_weight * losses["cl_loss"])

def train_loss_weighted_sum(args, losses):
    return (args.conv_weight * losses["conv_loss"] + args.k2t_weight * losses["k2t_loss"] + args.cl_weight * losses["cl_loss"])

def test_loss_weighted_sum(args, losses):
    return losses["rec_loss"] + args.per_weight * losses["per_loss"]
