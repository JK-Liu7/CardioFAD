import torch.distributed as dist


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
