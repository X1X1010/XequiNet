from contextlib import contextmanager
import torch
import torch.distributed as dist
import torch.nn as nn
from torch_scatter import scatter_sum 
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader
from .lr_scheduler import SmoothReduceLROnPlateau, get_polynomial_decay_schedule
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


class MatCriterion(nn.Module):
    """
    MAE + RMSE
    """
    def __init__(self):
        super().__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        mae = self.mae_loss(pred, target)
        mse = self.mse_loss(pred, target)
        rmse = torch.sqrt(mse)
        loss = mae + rmse 
        return loss


class MatCriterion2(nn.Module):
    """
    data-wise mae + rmse 
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_diag, pred_off_diag, real_diag, real_off_diag, mask_diag, mask_off_diag, batch_index, dst_index):
        edge_batch_index = batch_index[dst_index]
        # per node error statistics 
        diff_diag = pred_diag - real_diag
        mse_diag_per_node = torch.sum(diff_diag**2 * mask_diag, dim=[1, 2])
        mae_diag_per_node = torch.sum(torch.abs(diff_diag) * mask_diag, dim=[1, 2]) 
        num_diag_per_node = torch.sum(mask_diag, dim=[1, 2]) 
        mse_diag_per_mole = scatter_sum(mse_diag_per_node, batch_index) 
        mae_diag_per_mole = scatter_sum(mae_diag_per_node, batch_index) 
        num_diag_per_mole = scatter_sum(num_diag_per_node, batch_index) 
        # per edge error statistics 
        diff_off_diag = pred_off_diag - real_off_diag 
        mse_off_diag_per_edge = torch.sum(diff_off_diag**2 * mask_off_diag, dim=[1, 2])
        mae_off_diag_per_edge = torch.sum(torch.abs(diff_off_diag) * mask_off_diag, dim=[1, 2]) 
        num_off_diag_per_edge = torch.sum(mask_off_diag, dim=[1, 2]) 
        mse_off_diag_per_mole = scatter_sum(mse_off_diag_per_edge, edge_batch_index) 
        mae_off_diag_per_mole = scatter_sum(mae_off_diag_per_edge, edge_batch_index) 
        num_off_diag_per_mole = scatter_sum(num_off_diag_per_edge, edge_batch_index)  

        batch_mae = ((mae_diag_per_mole + mae_off_diag_per_mole) / (num_diag_per_mole + num_off_diag_per_mole)).mean() 
        batch_mse = ((mse_diag_per_mole + mse_off_diag_per_mole) / (num_diag_per_mole + num_off_diag_per_mole)).mean() 
        batch_rmse = torch.sqrt(batch_mse) 
        batch_loss = batch_mae + batch_rmse 
        batch_diag_mae = (mae_diag_per_mole / num_diag_per_mole).mean()
        batch_off_diag_mae = (mae_off_diag_per_mole / num_off_diag_per_mole).mean()
        return batch_loss, batch_mae, batch_diag_mae, batch_off_diag_mae 


def resolve_lossfn(lossfn: str) -> nn.Module:
    """Helper function to return loss function"""
    lossfn = lossfn.lower()
    if lossfn == "l2" or lossfn == "mse":
        return nn.MSELoss()
    elif lossfn == "l1" or lossfn == "mae":
        return nn.L1Loss()
    elif lossfn == "smoothl1":
        return nn.SmoothL1Loss()
    elif lossfn == "matloss":
        return MatCriterion()
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
) -> lr_scheduler._LRScheduler:
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


class ModelWrapper:
    def __init__(self, model: nn.Module, pbc: bool):
        self.model = model
        self.pbc = pbc

    def __call__(self, data):
        if self.pbc:
            return self.model(data.at_no, data.pos, data.shifts,
                              data.edge_index, data.batch, data.at_filter)
        else:
            return self.model(data.at_no, data.pos, data.edge_index, data.batch)
    
    def __getattr__(self, name):
        return getattr(self.model, name)
    