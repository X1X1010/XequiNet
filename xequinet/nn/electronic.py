import math
from typing import Dict

import torch
import torch.nn as nn
from scipy import constants

from xequinet.utils import get_default_units, keys, unit_conversion

from .rbf import resolve_cutoff


class HierarchicalCutoff(nn.Module):
    def __init__(
        self,
        long_cutoff: float = 10.0,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        assert cutoff <= long_cutoff
        self.cutoff = cutoff
        self.long_cutoff = long_cutoff

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        ori_edge_index = data[keys.EDGE_INDEX]
        ori_edge_length = data[keys.EDGE_LENGTH]
        edge_mask = ori_edge_length < self.cutoff
        data[keys.EDGE_INDEX] = ori_edge_index[:, edge_mask]
        data[keys.EDGE_LENGTH] = ori_edge_length[edge_mask]
        data[keys.LONG_EDGE_INDEX] = ori_edge_index
        data[keys.LONG_EDGE_LENGTH] = ori_edge_length

        return data


class CoulombWithCutoff(nn.Module):
    def __init__(
        self,
        coulomb_cutoff: float = 10.0,
    ) -> None:
        super().__init__()
        self.coulomb_cutoff = coulomb_cutoff
        self.flat_envelope = resolve_cutoff("flat", coulomb_cutoff)
        si_constant = 1.0 / (4 * math.pi * constants.epsilon_0)
        units = get_default_units()
        constant_unit = f"{units[keys.TOTAL_ENERGY]}*{units[keys.POSITIONS]}^2/{units[keys.TOTAL_CHARGE]}^2"
        self.constant = si_constant * unit_conversion("J*m/C^2", f"{constant_unit}")

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        long_edge_index = data[keys.LONG_EDGE_INDEX]
        center_idx = long_edge_index[keys.CENTER_IDX]
        neighbor_idx = long_edge_index[keys.NEIGHBOR_IDX]
        long_dist = data[keys.LONG_EDGE_LENGTH]  # [n_edges]
        atomic_charges = data[keys.ATOMIC_CHARGES]

        q1 = atomic_charges[center_idx]
        q2 = atomic_charges[neighbor_idx]
        envelope = self.flat_envelope(long_dist)  # [n_edges]
        # half of the Coulomb energy to avoid double counting
        pair_energies = 0.5 * envelope * self.constant * q1 * q2 / long_dist
        if keys.ATOMIC_ENERGIES in data:
            atomic_energies = data[keys.ATOMIC_ENERGIES]
        else:
            atomic_energies = torch.zeros_like(atomic_charges)
        atomic_energies = atomic_energies.index_add(
            0, center_idx, pair_energies
        )
        data[keys.ATOMIC_ENERGIES] = atomic_energies

        return data
