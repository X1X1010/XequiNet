from contextlib import contextmanager
from typing import Optional, Tuple

import pytorch_warmup as warmup
import torch
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader

from xequinet import keys

from .lr_scheduler import SmoothReduceLROnPlateau, get_polynomial_decay_schedule


@contextmanager
def distributed_zero_first(local_rank: int = None):
    r"""
    Decorator to make all processes in distributed training wait for each local_master to do something.
    Reference:
    https://github.com/xuanzhangyang/yolov5/blob/master/utils/torch_utils.py
    """
    if local_rank is None:
        yield
        return
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def calculate_stats(
    dataloader: DataLoader,
    divided_by_atoms: bool = True,
    target_property: str = keys.TOTAL_ENERGY,
    base_property: Optional[str] = keys.BASE_ENERGY,
    max_num_samples: int = 100000,
) -> Tuple[float, float]:
    """
    Aovid using atom_wise_mean when the dataset is too large.
    First, it will cost a lot of time to calculate the mean atomic energy.
    Second, the mean atomic energy is highly related to the dataset,
    which may lead to a poor performance in inference.
    """
    mean, m2 = 0.0, 0.0
    count = 0
    for data in dataloader:
        sample_y = data[target_property]
        if base_property is not None and base_property in data:
            sample_y -= data[base_property]
        batch_size = sample_y.size(0)
        new_count = count + batch_size
        if divided_by_atoms:
            if keys.BATCH_PTR in data:
                num_atoms = data[keys.BATCH_PTR][1:] - data[keys.BATCH_PTR][:-1]
            else:
                num_atoms = torch.tensor(
                    [data[keys.ATOMIC_NUMBERS].size(0)], device=sample_y.device
                )
            sample_y /= num_atoms
        sample_mean = torch.mean(sample_y, dim=0)
        sample_m2 = torch.sum((sample_y - sample_mean) ** 2, dim=0)

        delta = sample_mean - mean
        mean += delta * batch_size / new_count
        corr = batch_size * count / new_count
        m2 += sample_m2 + delta**2 * corr
        count = new_count
        if count > max_num_samples:
            break

    std = torch.sqrt(m2 / count)
    return mean.item(), std.item()


def resolve_optimizer(
    optim_type: str, params: dict, lr: float, **kwargs
) -> torch.optim.Optimizer:
    """Helper function to return an optimizer"""
    optim_type = optim_type.lower()
    if optim_type == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    elif optim_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported optimizer {optim_type}")


def resolve_lr_scheduler(
    sched_type: str,
    optimizer: torch.optim.Optimizer,
    max_lr: float = 5e-4,
    min_lr: float = 0.0,
    max_epochs: int = 500,
    steps_per_epoch: int = 1,
    **kwargs,
) -> lr_scheduler.LRScheduler:
    """Helper function to return a learning rate scheduler"""
    sched_type = sched_type.lower()
    if sched_type in {"cosine_annealing", "cosine"}:
        T_max = kwargs.pop("T_max", max_epochs) * steps_per_epoch
        return lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=min_lr,
            T_max=T_max,
            **kwargs,
        )
    elif sched_type in {"cosine_annealing_warmup_restart", "cosine_warmup"}:
        T_0 = kwargs.pop("T_0", max_epochs) * steps_per_epoch
        T_mult = kwargs.pop("T_mult", 1)
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=min_lr,
            **kwargs,
        )
    elif sched_type == "exponential":
        return lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            **kwargs,
        )
    elif sched_type == "step":
        step_size = kwargs.pop("step_size", max_epochs // 5) * steps_per_epoch
        return lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=step_size,
            **kwargs,
        )
    elif sched_type in {"plateau", "reduce_on_plateau"}:
        return SmoothReduceLROnPlateau(
            optimizer=optimizer,
            min_lr=min_lr,
            **kwargs,
        )
    elif sched_type == "linear_decay":
        num_steps = max_epochs * steps_per_epoch
        return get_polynomial_decay_schedule(
            optimizer=optimizer,
            num_training_steps=num_steps,
            lr_end=min_lr,
        )
    else:
        raise NotImplementedError(f"Unsupported scheduler {sched_type}")


def resolve_warmup_scheduler(
    warm_type: str,
    optimizer: torch.optim.Optimizer,
    warm_steps: int,
) -> warmup.BaseWarmup:
    """Helper function to return a warmup scheduler"""
    warm_type = warm_type.lower()
    if warm_type == "linear":
        return warmup.LinearWarmup(
            optimizer=optimizer,
            warmup_period=warm_steps,
        )
    elif warm_type == "exponential":
        return warmup.ExponentialWarmup(
            optimizer=optimizer,
            warmup_period=warm_steps,
        )
    elif warm_type == "untuned_linear":
        return warmup.UntunedLinearWarmup(
            optimizer=optimizer,
        )
    elif warm_type == "untuned_exponential":
        return warmup.UntunedExponentialWarmup(
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError(f"Unsupported warmup scheduler {warm_type}")
