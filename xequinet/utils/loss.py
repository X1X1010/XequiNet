from typing import Dict, Tuple

import torch
import torch.nn as nn

from xequinet.utils import keys


class MultiTaskLoss(nn.Module):
    AVAILABLE_LOSSES = {
        "l1": nn.L1Loss,
        "mae": nn.L1Loss,
        "l2": nn.MSELoss,
        "mse": nn.MSELoss,
        "smoothl1": nn.SmoothL1Loss,
    }

    def __init__(
        self,
        loss_type: str = "l2",
        **kwargs,
    ) -> None:
        super().__init__()
        self.loss_fn = self.AVAILABLE_LOSSES[loss_type]()
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
        loss_dict = {}

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
            loss_dict[keys.ENERGY_PER_ATOM] = energy_per_atom_loss
            self.weight_dict.pop(keys.ENERGY_PER_ATOM)
        # stress
        if keys.STRESS in self.weight_dict:
            assert result[keys.VIRIAL].shape == target[keys.VIRIAL].shape
            volume = torch.det(target[keys.CELL]).abs().reshape(-1, 1, 1)
            stress_loss = self.loss_fn(
                -result[keys.VIRIAL] / volume,
                -target[keys.VIRIAL] / volume,
            )
            total_loss += self.weight_dict[keys.STRESS] * stress_loss
            loss_dict[keys.STRESS] = stress_loss
            self.weight_dict.pop(keys.STRESS)

        # other cases
        for k, w in self.weight_dict.items():
            assert result[k].shape == target[k].shape
            loss = self.loss_fn(result[k], target[k])
            total_loss += w * loss
            loss_dict[k] = loss

        return total_loss, loss_dict
