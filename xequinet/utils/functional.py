from contextlib import contextmanager
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader
from .lr_scheduler import SmoothReduceLROnPlateau
import pytorch_warmup as warmup


@contextmanager
def distributed_zero_first(local_rank: int):
    r"""
    Decorator to make all processes in distributed training wait for each local_master to do something.
    Reference:
    https://github.com/xuanzhangyang/yolov5/blob/master/utils/torch_utils.py
    """
    if local_rank not in [-1, 0]:
       dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
       dist.barrier(device_ids=[0])


def calculate_stats(
    dataloader: DataLoader,
    divided_by_atoms: bool = True,
):
    """
    Aovid using atom_wise_mean when the dataset is too large.
    First, it will cost a lot of time to calculate the mean atomic energy.
    Second, the mean atomic energy is highly related to the dataset,
    which may lead to a poor performance in inference.
    """
    mean, m2 = 0.0, 0.0
    count = 0
    for data in dataloader:
        sample_y = data.y
        if hasattr(data, "base_y"):
            sample_y -= data.base_y
        batch_size = sample_y.size(0)
        new_count = count + batch_size
        if divided_by_atoms:
            num_atoms = data.ptr[1:] - data.ptr[:-1]
            sample_y /= num_atoms.view(-1, 1)
        sample_mean = torch.mean(sample_y, dim=0)
        sample_m2 = torch.sum((sample_y - sample_mean[:, None]) ** 2, dim=0)

        delta = sample_mean - mean
        mean += delta * batch_size / new_count
        corr = batch_size * count / new_count
        m2 += sample_m2 + delta ** 2 * corr
        count = new_count

    std = torch.sqrt(m2 / count)
    return mean.item(), std.item()


def resolve_lossfn(lossfn: str) -> nn.Module:
    """Helper function to return loss function"""
    lossfn = lossfn.lower()
    if lossfn == "l2" or lossfn == "mse":
        return nn.MSELoss()
    elif lossfn == "l1" or lossfn == "mae":
        return nn.L1Loss()
    elif lossfn == "smoothl1":
        return nn.SmoothL1Loss()
    else:
        raise NotImplementedError(f"Unsupported loss function {lossfn}")


def resolve_actfn(actfn: str) -> nn.Module:
    """Helper function to return activation function"""
    actfn = actfn.lower()
    if actfn == "relu":
        return nn.ReLU()
    elif actfn == "leakyrelu":
        return nn.LeakyReLU()
    elif actfn == "softplus":
        return nn.Softplus()
    elif actfn == "sigmoid":
        return nn.Sigmoid()
    elif actfn == "silu":
        return nn.SiLU()
    elif actfn == "tanh":
        return nn.Tanh()
    elif actfn == "identity":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported activation function {actfn}")


def resolve_norm(
    normtype: str,
    num_features: int,
    momentum: float = 0.1,
    affine: bool = True,
) -> nn.Module:
    """Helper function to return normalization layer"""
    normtype = normtype.lower()
    if normtype == "batch":
        return nn.BatchNorm1d(
            num_features,
            momentum=momentum,
            affine=affine,
        )
    elif normtype == "layer":
        return nn.LayerNorm(
            num_features,
            elementwise_affine=affine,
        )
    elif normtype == "nonorm":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported normalization layer {normtype}")


def resolve_optimizer(optimtype: str, params: dict, lr: float, **kwargs) -> torch.optim.Optimizer:
    """Helper function to return an optimizer"""
    optimtype = optimtype.lower()
    if optimtype == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    elif optimtype == "adamw":
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported optimizer {optimtype}")


def resolve_lr_scheduler(
    schedtype: str,
    optimizer: torch.optim.Optimizer,
    max_lr: float = 5e-4,
    min_lr: float = 0.0,
    num_steps: int = None,
    **kwargs,
) -> lr_scheduler.LRScheduler:
    """Helper function to return a learning rate scheduler"""
    schedtype = schedtype.lower()
    if schedtype == "cosine_annealing":
        return lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_steps,
            eta_min=min_lr,
            **kwargs,
        )
    elif schedtype == "cyclic":
        return lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=min_lr,
            max_lr=max_lr,
            step_size_up=num_steps // 2,
            **kwargs,
        )
    elif schedtype == "exponential":
        return lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            **kwargs,
        )
    elif schedtype == "step":
        return lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=num_steps // 5,
            **kwargs,
        )
    elif schedtype == "plateau":
        return SmoothReduceLROnPlateau(
            optimizer=optimizer,
            min_lr=min_lr,
            **kwargs,
        )
    else:
        raise NotImplementedError(f"Unsupported scheduler {schedtype}")


def resolve_warmup_scheduler(
    warmtype: str,
    optimizer: torch.optim.Optimizer,
    warm_steps: int,
) -> warmup.BaseWarmup:
    """Helper function to return a warmup scheduler"""
    warmtype = warmtype.lower()
    if warmtype == "linear":
        return warmup.LinearWarmup(
            optimizer=optimizer,
            warmup_period=warm_steps,
        )
    elif warmtype == "exponential":
        return warmup.ExponentialWarmup(
            optimizer=optimizer,
            warmup_period=warm_steps,
        )
    elif warmtype == "untuned_linear":
        return warmup.UntunedLinearWarmup(
            optimizer=optimizer,
        )
    elif warmtype == "untuned_exponential":
        return warmup.UntunedExponentialWarmup(
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError(f"Unsupported warmup scheduler {warmtype}")
