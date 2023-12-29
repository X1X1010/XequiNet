from typing import List, Union
from copy import deepcopy
from contextlib import contextmanager
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader
from .lr_scheduler import SmoothReduceLROnPlateau
from .qc import ELEMENTS_LIST
import pytorch_warmup as warmup


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


def resolve_optimizer(optim_type: str, params: dict, lr: float, **kwargs) -> torch.optim.Optimizer:
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
    if sched_type == "cosine_annealing":
        T_max = kwargs.pop("T_max", max_epochs) * steps_per_epoch
        return lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=min_lr,
            T_max=T_max,
            **kwargs,
        )
    elif sched_type == "cyclic":
        step_size_up = kwargs.pop("step_size_up", max_epochs // 2) * steps_per_epoch
        return lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=min_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
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
    elif sched_type == "plateau":
        return SmoothReduceLROnPlateau(
            optimizer=optimizer,
            min_lr=min_lr,
            **kwargs,
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


def gen_3Dinfo_str(
    at_no: torch.Tensor,
    info_3d: Union[List[torch.Tensor], torch.Tensor],
    title: Union[List[str], str],
    precision: Union[List[int], int] = 6
) -> str:
    """
    Generate 3D info string for a molecule.
    """
    titles = [title] if isinstance(title, str) else deepcopy(title)
    info_3ds = [info_3d] if isinstance(info_3d, torch.Tensor) else info_3d
    assert len(titles) == len(info_3ds)
    precisions = [precision] * len(titles) if isinstance(precision, int) else precision
    assert len(titles) == len(precisions)
    assert len(at_no) == info_3ds[0].size(0)
    info_lists = []
    for i, a in enumerate(at_no):
        at_symbol = ELEMENTS_LIST[a.item()]
        lines = [f"  {at_symbol: <2}  "]
        for cs, p in zip(info_3ds, precisions):
            c = cs[i]
            float_fmt = f"{p+6}.{p}f"
            cx = c[0].item(); cy = c[1].item(); cz = c[2].item()
            lines.append(f"{cx:{float_fmt}}{cy:{float_fmt}}{cz:{float_fmt}}")
        info_lists.append(lines)
    num_chs = [len(ac) for ac in info_lists[0]]
    titles.insert(0, "Atom")
    filled_titles = [f"{t: ^{n}}" for t, n in zip(titles, num_chs)]
    title_str = "    ".join(filled_titles)
    info_strs = ["    ".join(line) for line in info_lists]
    parting_line = "-" * len(title_str)
    results = [parting_line, title_str, parting_line]
    results.extend(info_strs)
    results.append(parting_line + "\n")
    return "\n".join(results)
