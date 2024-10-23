import heapq
import os
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from tabulate import tabulate
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from xequinet import keys
from xequinet.data import XequiData
from xequinet.utils import get_default_units

from .config import XequiConfig
from .functional import (
    resolve_lr_scheduler,
    resolve_optimizer,
    resolve_warmup_scheduler,
)
from .logger import ZeroLogger
from .loss import L1Metric, WeightedLoss


class loss2file:
    def __init__(self, loss: float, ptfile: str, epoch: int) -> None:
        self.loss = loss
        self.ptfile = ptfile
        self.epoch = epoch

    def __lt__(self, other: "loss2file") -> bool:
        # overloading __lt__ inversely to realize max heap
        return self.loss > other.loss


class DistAverageMetric:
    def __init__(self, properties: Iterable, device: torch.device) -> None:
        self.device = device
        self.properties = {
            prop: torch.zeros((1,), device=self.device) for prop in properties
        }
        self.counts = {
            prop: torch.zeros((1,), device=self.device) for prop in properties
        }
        # sort properties
        self.properties = dict(sorted(self.properties.items()))
        self.reset()

    def reset(self) -> None:
        for prop in self.properties:
            self.properties[prop].zero_()
            self.counts[prop].zero_()

    @torch.no_grad()
    def update(self, property: str, value: float, n: int = 1) -> None:
        assert property in self.properties, f"Property {property} not found"
        self.properties[property] += value
        self.counts[property] += n

    @torch.no_grad()
    def reduce(self) -> Dict[str, float]:
        result = {}
        for prop, val in self.properties.items():
            val = val.clone()
            count = self.counts[prop].clone()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            result[prop] = (val / count).item()
        return result


class EarlyStopping:
    def __init__(
        self,
        patience: int = None,
        min_delta: float = 0.0,
        min_lr: float = 1e-6,
    ) -> None:
        self.patience = patience if patience is not None else float("inf")
        self.min_delta = min_delta
        self.min_lr = 1e-6 if min_lr == 0.0 else min_lr
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, best_loss: float, lr: float) -> bool:
        if val_loss - best_loss > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        elif lr <= self.min_lr:
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
        config: XequiConfig,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        dist_sampler: DistributedSampler,
        log: ZeroLogger,
    ) -> None:
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
        self.trainer_conf = config.trainer
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.dist_sampler = dist_sampler
        self.log = log

        self.model_conf = config.model

        # set loss function
        self.lossfn = WeightedLoss(
            self.trainer_conf.lossfn, **self.trainer_conf.losses_weight
        )
        self.compute_forces = keys.FORCES in self.trainer_conf.losses_weight
        self.compute_virial = (
            keys.VIRIAL in self.trainer_conf.losses_weight
            or keys.STRESS in self.trainer_conf.losses_weight
        )
        # set optimizer
        self.optimizer = resolve_optimizer(
            optim_type=self.trainer_conf.optimizer,
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.trainer_conf.max_lr,
            **self.trainer_conf.optimizer_kwargs,
        )
        # set lr scheduler
        self.lr_scheduler = resolve_lr_scheduler(
            sched_type=self.trainer_conf.lr_scheduler,
            optimizer=self.optimizer,
            max_lr=self.trainer_conf.max_lr,
            min_lr=self.trainer_conf.min_lr,
            max_epochs=self.trainer_conf.max_epochs,
            steps_per_epoch=len(train_loader),
            **self.trainer_conf.lr_scheduler_kwargs,
        )
        # set warmup scheduler
        if self.trainer_conf.lr_scheduler == "plateau":
            warm_steps = self.trainer_conf.warmup_epochs
        else:
            warm_steps = self.trainer_conf.warmup_epochs * len(train_loader)
        self.warmup_scheduler = resolve_warmup_scheduler(
            warm_type=self.trainer_conf.warmup_scheduler,
            optimizer=self.optimizer,
            warm_steps=warm_steps,
        )
        # set early stopping (only work when lr_scheduler is plateau)
        self.early_stop = EarlyStopping(
            patience=self.trainer_conf.early_stop, min_lr=self.trainer_conf.min_lr
        )
        # exponential moving average
        self.ema_model: Optional[AveragedModel] = None
        if self.trainer_conf.ema_decay is not None:
            ema_model = AveragedModel(
                self.model.module,
                avg_fn=lambda avg_param, param, num_avg: self.trainer_conf.ema_decay
                * avg_param
                + (1 - self.trainer_conf.ema_decay) * param,
                device=device,
            )
            self.ema_model = ema_model
        # set l1 metric
        self.l1_metrics = L1Metric(*self.trainer_conf.losses_weight.keys())
        # loss recording, model saving and logging
        self.meter = DistAverageMetric(
            properties=self.l1_metrics.properties,
            device=device,
        )
        self.best_l2fs: List[loss2file] = [
            loss2file(
                float("inf"),
                os.path.join(
                    self.trainer_conf.save_dir, f"{self.trainer_conf.run_name}_{i}.pt"
                ),
                0,
            )
            for i in range(self.trainer_conf.best_k)
        ]  # a max-heap, actually it is a min-heap

        # load checkpoint
        self.start_epoch = 1
        if self.trainer_conf.ckpt_file is not None:
            self._load_params(self.trainer_conf.ckpt_file)

    def _load_params(self, ckpt_file: str) -> None:
        state = torch.load(ckpt_file, map_location=self.device)
        self.model.module.load_state_dict(state["model"], strict=False)
        if self.trainer_conf.resume:
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            self.warmup_scheduler.load_state_dict(state["warmup_scheduler"])
            self.start_epoch = state["epoch"] + 1
            for l2f in self.best_l2fs:
                if os.path.isfile(l2f.ptfile):
                    pt_state = torch.load(l2f.ptfile, map_location=self.device)
                    l2f.loss = pt_state["loss"] if "loss" in pt_state else float("inf")
        self.log.f.info(f" --- Loaded checkpoint from {ckpt_file}")

    def _save_params(
        self, model: nn.Module, ckpt_file: str, loss: float = None
    ) -> None:
        state = {
            "model": model.state_dict(),
            "epoch": self.epoch,
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "warmup_scheduler": self.warmup_scheduler.state_dict(),
            "config": self.model_conf,
        }
        torch.save(state, ckpt_file)

    def save_best_k(self, model: nn.Module, curr_loss: float) -> None:
        if curr_loss < self.best_l2fs[0].loss:
            l2f = heapq.heappop(self.best_l2fs)
            l2f.loss = curr_loss
            l2f.epoch = self.epoch
            self._save_params(model, l2f.ptfile, l2f.loss)
            heapq.heappush(self.best_l2fs, l2f)

    def train1epoch(self) -> None:
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch)
        for step, data in enumerate(self.train_loader, start=1):
            data: XequiData  # for type annotation
            self.meter.reset()
            data = data.to(self.device)
            # forward propagation
            result = self.model(
                data.to_dict(), self.compute_forces, self.compute_virial
            )
            loss, _ = self.lossfn(result, data)
            # backward propagation
            self.optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            # gradient clipping
            if self.trainer_conf.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.trainer_conf.grad_clip
                )
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.trainer_conf.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # record l1 loss
            l1_losses = self.l1_metrics(result, data)
            for prop, (l1, n) in l1_losses.items():
                self.meter.update(prop, l1, n)

            # logging
            if self.epoch % self.trainer_conf.log_epochs == 0 and (
                step % self.trainer_conf.log_steps == 0
                or step == len(self.train_loader)
            ):
                reduced_l1 = self.meter.reduce()
                header = ["", "Epoch", "Step", "LR"]
                tabulate_data = [
                    "Train MAE",
                    self.epoch,
                    f"{step}/{len(self.train_loader)}",
                    self.optimizer.param_groups[0]["lr"],
                ]
                header.extend(list(map(lambda x: x.capitalize(), reduced_l1.keys())))
                floatfmt = ["", "", "", ".3e"] + [".7f"] * len(reduced_l1)
                tabulate_data.extend(list(reduced_l1.values()))
                lines = tabulate(
                    [tabulate_data],
                    headers=header,
                    tablefmt="plain",
                    floatfmt=floatfmt,
                ).split("\n")
                for line in lines:
                    self.log.f.info(line)

    def validate(self) -> None:
        valid_model = self.model.module if self.ema_model is None else self.ema_model
        valid_model.eval()
        self.meter.reset()
        for data in self.valid_loader:
            data: XequiData  # for type annotation
            data = data.to(self.device)
            result = valid_model(
                data.to_dict(), self.compute_forces, self.compute_virial
            )
            l1_losses = self.l1_metrics(result, data)
            for prop, (l1, n) in l1_losses.items():
                self.meter.update(prop, l1, n)
        reduced_l1 = self.meter.reduce()
        mae = 0.0
        for prop, w in self.trainer_conf.losses_weight.items():
            mae += w * reduced_l1[prop]
        if self.trainer_conf.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(mae)
            lr = self.optimizer.param_groups[0]["lr"]
            self.early_stop(mae, self.best_l2fs[0].loss, lr)
        self.save_best_k(valid_model.module, mae)
        self._save_params(
            valid_model.module,
            f"{self.trainer_conf.save_dir}/{self.trainer_conf.run_name}_last.pt",
        )
        if self.epoch % self.trainer_conf.log_epochs == 0:
            header = [""]
            tabulate_data = (
                ["Validation MAE"] if self.ema_model is None else ["EMA Valid MAE"]
            )
            header.extend(list(map(lambda x: x.capitalize(), reduced_l1.keys())))
            tabulate_data.extend(list(reduced_l1.values()))
            lines = tabulate(
                [tabulate_data], headers=header, tablefmt="plain", floatfmt=".7f"
            ).split("\n")
            for line in lines:
                self.log.f.info(line)

    def start(self) -> None:
        if not self.trainer_conf.resume:
            self.log.f.info(" --- Start training")
            self.log.f.info(f" --- Task Name: {self.trainer_conf.run_name}")
            default_units = get_default_units()
            for prop, unit in default_units.items():
                if prop in keys.BASE_PROPERTIES:
                    continue
                self.log.f.info(f" --- Property: {prop} --- Unit: {unit}")

        # training loop
        for iepoch in range(self.start_epoch, self.trainer_conf.max_epochs + 1):
            self.epoch = iepoch
            self.train1epoch()
            self.validate()
            if self.early_stop.stop:
                self.log.f.info(f" --- Early Stopping at Epoch {iepoch}")
                break

        self.log.f.info(" --- Training Completed")
        self.log.f.info(f" --- Best Valid MAE: {self.best_l2fs[-1].loss:.5f}")
        self.log.f.info(
            f" --- Best Checkpoint: {self.best_l2fs[-1].ptfile} at Epoch {self.best_l2fs[-1].epoch}"
        )
