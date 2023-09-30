from typing import Tuple, List, Optional
import os
import heapq

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .functional import (
    resolve_lossfn,
    resolve_optimizer,
    resolve_lr_scheduler,
    resolve_warmup_scheduler,
)
from .config import NetConfig
from .logger import ZeroLogger
from .qc import get_default_unit

class loss2file:
    def __init__(self, loss: float, ptfile: str):
        self.loss = loss
        self.ptfile = ptfile

    def __lt__(self, other: "loss2file"):
        # overloading __lt__ inversely to realize max heap
        return self.loss > other.loss


class AverageMeter:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.sum = torch.zeros((1,), device=self.device)
        self.cnt = torch.zeros((1,), dtype=torch.int32, device=self.device)
    
    def update(self, val: float, n: int = 1):
        self.sum += val
        self.cnt += n
    
    def reduce(self) -> float:
        tmp_sum = self.sum.clone()
        tmp_cnt = self.cnt.clone()
        dist.all_reduce(tmp_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tmp_cnt, op=dist.ReduceOp.SUM)
        avg = tmp_sum / tmp_cnt
        return avg.item()


class EarlyStopping:
    def __init__(self, patience: int = 100, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss, best_loss):
        if val_loss - best_loss > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.counter = 0
        return self.stop


class Trainer:
    """
    General trainer class for training neural networks.
    """
    def __init__(
        self,
        model: nn.parallel.DistributedDataParallel,
        config: NetConfig,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        dist_sampler: Optional[DistributedSampler],
        log: ZeroLogger,
    ):
        """
        Args:
            `model`: DistributedDataParallel model
            `config`: network configuration
            `device`: torch device
            `train_loader`: training data loader
            `valid_loader`: validation data loader
            `dist_sampler`: distributed sampler
            `log`: logger
        """
        self.model = model
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.dist_sampler = dist_sampler

        # set loss function
        self.lossfn = resolve_lossfn(config.lossfn).to(device)
        # set optimizer
        self.optimizer = resolve_optimizer(
            optimtype=config.optimizer,
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.max_lr,
            **config.optim_kwargs,
        )
        # set lr scheduler
        self.lr_scheduler = resolve_lr_scheduler(
            schedtype=config.lr_scheduler,
            optimizer=self.optimizer,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
            num_steps=config.max_epochs * len(train_loader),
            **config.lr_sche_kwargs,
        )
        # set warmup scheduler
        warm_steps = config.warmup_epochs * len(train_loader)
        if config.lr_scheduler == "plateau":
            warm_steps = config.warmup_epochs
        self.warmup_scheduler = resolve_warmup_scheduler(
            warmtype=config.warmup_scheduler,
            optimizer=self.optimizer,
            warm_steps=warm_steps,
        )
        # set early stopping (only work when lr_scheduler is plateau)
        self.early_stop = EarlyStopping(patience=config.early_stop)
        # load checkpoint
        self.start_epoch = 1
        if config.ckpt_file is not None:
            self._load_params(config.ckpt_file)

        # exponential moving average
        self.ema_model = None
        if config.ema_decay is not None:
            self.ema_model = AveragedModel(
                self.model.module,
                avg_fn=lambda avg_param, param, num_avg: \
                    config.ema_decay * avg_param + (1 - config.ema_decay) * param,
                device=device,
            )
        # loss recording, model saving and logging
        self.meter = AverageMeter(device=device)
        self.best_l2fs: List[loss2file] = [
            loss2file(float("inf"), os.path.join(config.save_dir, f"{config.run_name}_{i}.pt"))
            for i in range(config.best_k)
        ]  # a max-heap, actually it is a min-heap
        self.log = log


    def _load_params(self, ckpt_file: str):
        state = torch.load(ckpt_file, map_location=self.device)
        self.model.module.load_state_dict(state["model"])
        if self.config.resumption:
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            self.warmup_scheduler.load_state_dict(state["warmup_scheduler"])
            self.start_epoch = state["epoch"] + 1


    def _save_params(self, model: nn.Module, ckpt_file: str):
        state = {
            "model": model.state_dict(),
            "epoch": self.epoch,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "warmup_scheduler": self.warmup_scheduler.state_dict(),
            "config": self.config.dict(),
        }
        torch.save(state, ckpt_file)


    def save_best_k(self, model: nn.Module, curr_loss: float):
        if curr_loss < self.best_l2fs[0].loss:
            l2f = heapq.heappop(self.best_l2fs)
            l2f.loss = curr_loss
            self._save_params(model, l2f.ptfile)
            heapq.heappush(self.best_l2fs, l2f)


    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch)
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device)
            # TODO: AMP
            # forward propagation
            # output_index = data.fc_edge_index if hasattr(data, "fc_edge_index") else data.batch
            # pred = self.model(data.at_no, data.pos, data.edge_index, output_index)
            pred = self.model(data.at_no, data.pos, data.edge_index, data.batch)
            real = data.y - data.base_y if hasattr(data, "base_y") else data.y
            # print(pred.shape, real.shape)
            loss = self.lossfn(pred, real)
            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.config.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # record l1 loss
            with torch.no_grad():
                l1loss = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(l1loss.item(), real.numel())
            # logging
            if step % self.config.log_interval == 0 or step == len(self.train_loader):
                mae = self.meter.reduce()
                self.log.f.info(
                    "Epoch: [{iepoch:>4}][{step:>4}/{nstep}]   lr: {lr:3e}   train MAE: {mae:10.7f}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        mae=mae,
                    )
                )
    
    def validate(self):
        self.model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                # TODO: AMP
                # output_index = data.fc_edge_index if hasattr(data, "fc_edge_index") else data.batch
                # pred = self.model(data.at_no, data.pos, data.edge_index, output_index)
                pred = self.model(data.at_no, data.pos, data.edge_index, data.batch)
                real = data.y - data.base_y if hasattr(data, "base_y") else data.y
                l1loss = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(l1loss.item(), real.numel())
        mae = self.meter.reduce()
        self.log.f.info(f"Validation MAE: {mae:10.7f}")
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(mae)
            self.early_stop(mae, self.best_l2fs[0].loss)
        self.save_best_k(self.model.module, mae)


    def ema_validate(self):
        if self.ema_model is None:
            return
        self.ema_model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                # TODO: AMP
                output_index = data.fc_edge_index if hasattr(data, "fc_edge_index") else data.batch
                pred = self.model(data.at_no, data.pos, data.edge_index, output_index)
                real = data.y - data.base_y if hasattr(data, "base_y") else data.y
                l1loss = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(l1loss.item(), real.numel())
        mae = self.meter.reduce()
        self.log.f.info("EMA Valid MAE: {mae:10.7f}".format(mae=mae))
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(mae)
            self.early_stop(mae, self.best_l2fs[0].loss)
        self.save_best_k(self.ema_model.module, mae)


    def start(self):
        prop_unit, len_unit = get_default_unit()
        self.log.f.info(" --- Start training")
        self.log.f.info(f" --- Task Name: {self.config.run_name}")
        self.log.f.info(f" --- Property: {self.config.label_name} --- Unit: {prop_unit} {len_unit}")

        # training loop
        for iepoch in range(self.start_epoch, self.config.max_epochs + 1):
            self.epoch = iepoch
            self.train1epoch()
            if self.ema_model is None:
                self.validate()
            else:
                self.ema_validate()
            if self.early_stop.stop:
                self.log.f.info(f" --- Early Stopping at Epoch {iepoch}")
                break
        
        self.log.f.info(" --- Training Completed")
        self.log.f.info(f" --- Best Valid MAE: {self.best_l2fs[-1].loss:.5f}")
        self.log.f.info(f" --- Best Checkpoint: {self.best_l2fs[-1].ptfile}")


class WithForceMeter:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.sum = torch.zeros((2,), device=self.device)
        self.cnt = torch.zeros((2,), dtype=torch.int32, device=self.device)

    def update(self, energy: float, force: float, n_ene: int, n_frc: int):
        self.sum[0] += energy; self.sum[1] += force
        self.cnt[0] += n_ene; self.cnt[1] += n_frc

    def reduce(self) -> Tuple[float, float]:
        tmp_sum = self.sum.clone()
        tmp_cnt = self.cnt.clone()
        dist.all_reduce(tmp_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tmp_cnt, op=dist.ReduceOp.SUM)
        avg = tmp_sum / tmp_cnt
        return avg[0].item(), avg[1].item()
    

class GradTrainer(Trainer):
    """
    Trainer class for scalar and relative gradient property
    """
    def __init__(
        self,
        model: nn.parallel.DistributedDataParallel,
        config: NetConfig,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        dist_sampler: DistributedSampler,
        log: ZeroLogger,
    ):
        """
        Args:
            `model`: DistributedDataParallel model
            `config`: network configuration
            `device`: torch device
            `train_loader`: training data loader
            `valid_loader`: validation data loader
            `dist_sampler`: distributed sampler
            `log`: logger
        """
        super().__init__(
            model, config, device, train_loader, valid_loader, dist_sampler, log
        )
        assert config.force_weight <= 1.0
        self.meter = WithForceMeter(self.device)

    
    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch)
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device)
            # TODO: AMP
            # forward propagation
            data.pos.requires_grad_(True)
            predE, predF = self.model(data.at_no, data.pos, data.edge_index, data.batch)
            realE, realF = data.y, data.force
            if hasattr(data, "base_y") and hasattr(data, "base_force"):
                realE -= data.base_y
                realF -= data.base_force
            loss = (1 - self.config.force_weight) * self.lossfn(predE, realE) \
                 + self.config.force_weight * self.lossfn(predF, realF)
            # backward propagation
            self.optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            # gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.config.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # record l1 loss
            with torch.no_grad():
                l1lossE = F.l1_loss(predE, realE, reduction="sum")
                l1lossF = F.l1_loss(predF, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
            # logging
            if step % self.config.log_interval == 0 or step == len(self.train_loader):
                maeE, maeF = self.meter.reduce()
                self.log.f.info(
                    "Epoch: [{iepoch:>4}][{step:4d}/{nstep}]   lr: {lr:3e}   train MAE: Energy {maeE:10.7f}  Force {maeF:10.7f}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        maeE=maeE,
                        maeF=maeF,
                    )
                )
    
    def validate(self):
        self.model.eval()
        self.meter.reset()
        for data in self.valid_loader:
            data = data.to(self.device)
            # TODO: AMP
            data.pos.requires_grad_(True)
            predE, predF = self.model(data.at_no, data.pos, data.edge_index, data.batch)
            with torch.no_grad():
                realE, realF = data.y, data.force
                if hasattr(data, "base_y") and hasattr(data, "base_force"):
                    realE -= data.base_y
                    realF -= data.base_force
                l1lossE = F.l1_loss(predE, realE, reduction="sum") 
                l1lossF = F.l1_loss(predF, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
        maeE, maeF = self.meter.reduce()
        self.log.f.info(f"Validation MAE: Energy {maeE:10.7f}  Force {maeF:10.7f}")
        mae = (1 - self.config.force_weight) * maeE + self.config.force_weight * maeF
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(mae)
            self.early_stop(mae, self.best_l2fs[0].loss)
        self.save_best_k(self.model.module, mae)

    
    def ema_validate(self):
        if self.ema_model is None:
            return
        self.ema_model.eval()
        self.meter.reset()
        for data in self.valid_loader:
            data = data.to(self.device)
            # TODO: AMP
            data.pos.requires_grad_(True)
            predE, predF = self.ema_model(data.at_no, data.pos, data.edge_index, data.batch)
            with torch.no_grad():
                realE, realF = data.y, data.force
                if hasattr(data, "base_y") and hasattr(data, "base_force"):
                    realE -= data.base_y
                    realF -= data.base_force
                l1lossE = F.l1_loss(predE, realE, reduction="sum") 
                l1lossF = F.l1_loss(predF, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
        maeE, maeF = self.meter.reduce()
        self.log.f.info(f"EMA Validation MAE: Energy {maeE:10.7f}  Force {maeF:10.7f}")
        mae = (1 - self.config.force_weight) * maeE + self.config.force_weight * maeF
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(mae)
            self.early_stop(mae, self.best_l2fs[0].loss)
        self.save_best_k(self.ema_model.module, mae)

    
