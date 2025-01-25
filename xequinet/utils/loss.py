from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from xequinet import keys


class MatCriterion(nn.Module):
    """MSE + RMSE"""

    def __init__(self) -> None:
        super().__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mae = self.mae_loss(pred, target)
        mse = self.mse_loss(pred, target)
        loss = mae + torch.sqrt(mse)
        return loss


class WeightedLoss(nn.Module):
    AVAILABLE_LOSSES = {
        "l1": nn.L1Loss,
        "mae": nn.L1Loss,
        "l2": nn.MSELoss,
        "mse": nn.MSELoss,
        "smoothl1": nn.SmoothL1Loss,
        "matloss": MatCriterion,
    }

    def __init__(
        self,
        loss_fn: str = "l2",
        **kwargs,
    ) -> None:
        super().__init__()
        self.loss_fn = self.AVAILABLE_LOSSES[loss_fn.lower()]()
        assert len(kwargs) > 0, "At least one task should be present"
        for k, v in kwargs.items():
            assert isinstance(v, float), f"Weight for {k} should be a float"
        self.weight_dict = kwargs

    def forward(
        self,
        result: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device = next(iter(result.values())).device
        total_loss = torch.tensor(0.0, device=device)
        losses_dict = {}

        # special case needed for extra processing
        # energy per atom
        if keys.ENERGY_PER_ATOM in self.weight_dict:
            assert result[keys.TOTAL_ENERGY].shape == target[keys.TOTAL_ENERGY].shape
            n_atoms = target[keys.BATCH_PTR][1:] - target[keys.BATCH_PTR][:-1]
            energy_per_atom_loss = self.loss_fn(
                result[keys.TOTAL_ENERGY] / n_atoms,
                target[keys.TOTAL_ENERGY] / n_atoms,
            )
            total_loss += self.weight_dict[keys.ENERGY_PER_ATOM] * energy_per_atom_loss
            losses_dict[keys.ENERGY_PER_ATOM] = energy_per_atom_loss
        # stress
        if keys.STRESS in self.weight_dict:
            assert result[keys.VIRIAL].shape == target[keys.VIRIAL].shape
            volume = torch.det(target[keys.CELL]).abs().reshape(-1, 1, 1)
            stress_loss = self.loss_fn(
                result[keys.VIRIAL] / volume,
                target[keys.VIRIAL] / volume,
            )
            total_loss += self.weight_dict[keys.STRESS] * stress_loss
            losses_dict[keys.STRESS] = stress_loss

        # other cases
        for k, w in self.weight_dict.items():
            if k in {keys.ENERGY_PER_ATOM, keys.STRESS}:
                continue
            assert result[k].shape == target[k].shape
            loss = self.loss_fn(result[k], target[k])
            total_loss += w * loss
            losses_dict[k] = loss

        return total_loss, losses_dict

    def __call__(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return super().__call__(*args, **kwargs)  # for type annotation


class ErrorMetric:
    def __init__(self, *args) -> None:
        self.properties = set()
        for prop in args:
            self.properties.add(prop)
            if prop == keys.TOTAL_ENERGY:
                self.properties.add(keys.ENERGY_PER_ATOM)
            elif prop == keys.ENERGY_PER_ATOM:
                self.properties.add(keys.TOTAL_ENERGY)
            elif prop == keys.VIRIAL:
                self.properties.add(keys.STRESS)
            elif prop in keys.STRESS:
                self.properties.add(keys.VIRIAL)

    @torch.no_grad()
    def __call__(
        self,
        result: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, Tuple[float, int]]:

        metrics_dict = {}

        # special case needed for extra processing
        # energy per atom
        if keys.ENERGY_PER_ATOM in self.properties:
            assert result[keys.TOTAL_ENERGY].shape == target[keys.TOTAL_ENERGY].shape
            n_atoms = target[keys.BATCH_PTR][1:] - target[keys.BATCH_PTR][:-1]
            energy_per_atom_l1 = F.l1_loss(
                result[keys.TOTAL_ENERGY] / n_atoms,
                target[keys.TOTAL_ENERGY] / n_atoms,
                reduction="sum",
            )
            energy_per_atom_l2 = F.mse_loss(
                result[keys.TOTAL_ENERGY] / n_atoms,
                target[keys.TOTAL_ENERGY] / n_atoms,
                reduction="sum",
            )
            metrics_dict[keys.ENERGY_PER_ATOM] = (
                energy_per_atom_l1.item(),
                energy_per_atom_l2.item(),
                target[keys.TOTAL_ENERGY].numel(),
            )
        # stress
        if keys.STRESS in self.properties:
            assert result[keys.VIRIAL].shape == target[keys.VIRIAL].shape
            volume = torch.det(target[keys.CELL]).abs().reshape(-1, 1, 1)
            stress_l1 = F.l1_loss(
                result[keys.VIRIAL] / volume,
                target[keys.VIRIAL] / volume,
                reduction="sum",
            )
            stress_l2 = F.mse_loss(
                result[keys.VIRIAL] / volume,
                target[keys.VIRIAL] / volume,
                reduction="sum",
            )
            metrics_dict[keys.STRESS] = (
                stress_l1.item(),
                stress_l2.item(),
                target[keys.VIRIAL].numel(),
            )

        # other cases
        for prop in self.properties:
            if prop in {keys.ENERGY_PER_ATOM, keys.STRESS}:
                continue
            assert result[prop].shape == target[prop].shape
            l1 = F.l1_loss(result[prop], target[prop], reduction="sum")
            l2 = F.mse_loss(result[prop], target[prop], reduction="sum")
            metrics_dict[prop] = (l1.item(), l2.item(), target[prop].numel())

        return metrics_dict
