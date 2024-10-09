import math
from typing import Iterable, Dict

from scipy import constants

import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from e3nn import o3

from xequinet.utils import keys, get_default_unit, unit_conversion
from .o3layer import (
    Invariant,
    EquivariantDot,
    Int2c1eEmbedding,
    resolve_actfn,
    resolve_norm,
    resolve_o3norm,
)
from .rbf import resolve_cutoff, resolve_rbf


class XEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux56",
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "cosine",
    ) -> None:
        """
        Args:
            `embed_dim`: Embedding dimension. (default: 16s + 8p + 4d = 28)
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `num_basis`: Number of the radial basis functions.
            `rbf_kernel`: Radial basis function type.
            `cutoff`: Cutoff distance for the neighbor atoms.
            `cutoff_fn`: Cutoff function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.node_num_irreps = self.node_irreps.num_irreps
        # self.embedding = nn.Embedding(100, self.node_dim)
        self.int2c1e = Int2c1eEmbedding(embed_basis, aux_basis)
        self.node_lin = nn.Linear(self.int2c1e.embed_dim, self.node_dim)
        nn.init.zeros_(self.node_lin.bias)
        self.sph_harm = o3.SphericalHarmonics(
            self.node_irreps, normalize=True, normalization="component"
        )
        self.rbf = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        atomic_numbers = data[keys.ATOMIC_NUMBERS]
        vectors = data[keys.EDGE_VECTOR]
        distances = data[keys.EDGE_LENGTH].unsqueeze(-1)

        # node linear
        node_embed = self.int2c1e(atomic_numbers)
        node_invariant = self.node_lin(node_embed)
        data[keys.NODE_INVARIANT] = node_invariant

        # calculate radial basis function
        rbf = self.rbf(distances)
        fcut = self.cutoff_fn(distances)
        data[keys.RADIAL_BASIS_FUNCTION] = rbf
        data[keys.ENVELOPE_FUNCTION] = fcut
        # calculate spherical harmonics  [x, y, z] -> [y, z, x]
        rsh = self.sph_harm(
            vectors[:, [1, 2, 0]]
        )  # unit vector, normalized by component
        data[keys.SPHERICAL_HARMONICS] = rsh

        node_equivariant = torch.zeros(
            (node_invariant.shape[0], rsh.shape[1]),
            device=node_invariant.device,
        )
        data[keys.NODE_EQUIVARIANT] = node_equivariant

        return data


class XPainnMessage(nn.Module):
    """Message function for XPaiNN"""

    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        num_basis: int = 20,
        actfn: str = "silu",
        norm_type: str = "layer",
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `num_basis`: Number of the radial basis functions.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.node_num_irreps = self.node_irreps.num_irreps
        self.hidden_dim = self.node_dim + self.node_num_irreps * 2
        self.num_basis = num_basis
        # scalar feature
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            resolve_actfn(actfn),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        # spherical feature
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim, bias=True)
        # elementwise tensor product
        self.rsh_conv = o3.ElementwiseTensorProduct(
            self.node_irreps, f"{self.node_num_irreps}x0e"
        )
        # normalization
        self.norm = resolve_norm(norm_type, self.node_dim)
        self.o3norm = resolve_o3norm(norm_type, self.node_irreps)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        node_invt = self.norm(data[keys.NODE_INVARIANT])
        node_equi = self.o3norm(data[keys.NODE_EQUIVARIANT])
        rbf = data[keys.RADIAL_BASIS_FUNCTION]
        fcut = data[keys.ENVELOPE_FUNCTION]
        rsh = data[keys.SPHERICAL_HARMONICS]
        edge_index = data[keys.EDGE_INDEX]
        center_idx = edge_index[keys.CENTER_IDX]
        neighbor_idx = edge_index[keys.NEIGHBOR_IDX]

        inv_out = self.scalar_mlp(node_invt)
        filter_weight = self.rbf_lin(rbf) * fcut
        filter_out = inv_out[neighbor_idx] * filter_weight

        gate_state_equi, gate_edge_equi, message_invt = torch.split(
            filter_out,
            [self.node_num_irreps, self.node_num_irreps, self.node_dim],
            dim=-1,
        )
        message_equi = self.rsh_conv(node_equi[neighbor_idx], gate_state_equi)
        edge_equi = self.rsh_conv(rsh, gate_edge_equi)
        message_equi = message_equi + edge_equi

        ori_invt = data[keys.NODE_INVARIANT]
        ori_equi = data[keys.NODE_EQUIVARIANT]
        data[keys.NODE_INVARIANT] = ori_invt.index_add(0, center_idx, message_invt)
        data[keys.NODE_EQUIVARIANT] = ori_equi.index_add(0, center_idx, message_equi)

        return data


class XPainnUpdate(nn.Module):
    """Update function for XPaiNN"""

    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        actfn: str = "silu",
        norm_type: str = "layer",
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.node_num_irreps = self.node_irreps.num_irreps
        self.hidden_dim = self.node_dim * 2 + self.node_num_irreps
        # spherical feature
        self.update_U = o3.Linear(self.node_irreps, self.node_irreps, biases=True)
        self.update_V = o3.Linear(self.node_irreps, self.node_irreps, biases=True)
        self.invariant = Invariant(self.node_irreps)
        self.equidot = EquivariantDot(self.node_irreps)
        self.dot_lin = nn.Linear(self.node_num_irreps, self.node_dim, bias=False)
        self.rsh_conv = o3.ElementwiseTensorProduct(
            self.node_irreps, f"{self.node_num_irreps}x0e"
        )
        # scalar feature
        self.update_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.node_num_irreps, self.node_dim),
            resolve_actfn(actfn),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        # normalization
        self.norm = resolve_norm(norm_type, self.node_dim)
        self.o3norm = resolve_o3norm(norm_type, self.node_irreps)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        node_invt = self.norm(data[keys.NODE_INVARIANT])
        node_equi = self.o3norm(data[keys.NODE_EQUIVARIANT])

        U_equi = self.update_U(node_equi)
        V_equi = self.update_V(node_equi)

        V_invt = self.invariant(V_equi)
        mlp_in = torch.cat([node_invt, V_invt], dim=-1)
        mlp_out = self.update_mlp(mlp_in)

        a_vv, a_sv, a_ss = torch.split(
            mlp_out, [self.node_num_irreps, self.node_dim, self.node_dim], dim=-1
        )
        d_equi = self.rsh_conv(U_equi, a_vv)
        inner_prod = self.equidot(U_equi, V_equi)
        inner_prod = self.dot_lin(inner_prod)
        d_invt = a_sv * inner_prod + a_ss

        ori_invt = data[keys.NODE_INVARIANT]
        ori_equi = data[keys.NODE_EQUIVARIANT]
        data[keys.NODE_INVARIANT] = ori_invt + d_invt
        data[keys.NODE_EQUIVARIANT] = ori_equi + d_equi
        return data


class EleEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
    ) -> torch.Tensor:
        super().__init__()
        self.node_dim = node_dim
        self.sqrt_dim = math.sqrt(node_dim)
        self.q_linear = nn.Linear(node_dim, node_dim)
        self.k_linear = nn.Linear(1, node_dim)
        self.v_linear = nn.Linear(1, node_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        ele: torch.Tensor,
        batch: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x`: Node features.
            `ele`: Electronic features.
        Returns:
            Atomic features.
        """
        batch_ele = ele.index_select(0, batch).unsqueeze(-1)
        q = self.q_linear(x)
        k = self.k_linear(batch_ele)
        v = self.v_linear(batch_ele)
        dot = torch.sum(q * k, dim=1, keepdim=True) / self.sqrt_dim
        attn = softmax(dot, batch, dim=0)
        return attn * v


class CoulombWithCutoff(nn.Module):
    def __init__(
        self,
        coulomb_cutoff: float = 10.0,
    ) -> None:
        super().__init__()
        self.coulomb_cutoff = coulomb_cutoff
        self.flat_envelope = resolve_cutoff("flat", coulomb_cutoff)
        si_constant = 1.0 / (4 * math.pi * constants.epsilon_0)
        eng_unit, len_unit = get_default_unit()
        self.constant = si_constant * unit_conversion(
            "J*m/C^2", f"{eng_unit}*{len_unit}^2"
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        long_edge_index = data[keys.LONG_EDGE_INDEX]
        center_idx = long_edge_index[keys.CENTER_IDX]
        neighbor_idx = long_edge_index[keys.NEIGHBOR_IDX]
        long_dist = data[keys.LONG_EDGE_LENGTH].unsqueeze(-1)
        atomic_charges = data[keys.ATOMIC_CHARGES]

        q1 = atomic_charges[center_idx]
        q2 = atomic_charges[neighbor_idx]
        envelope = self.flat_envelope(long_dist)
        # half of the Coulomb energy to avoid double counting
        pair_energies = 0.5 * envelope * self.constant * q1 * q2 / long_dist
        if keys.ATOMIC_ENERGIES in data:
            atomic_energies = data[keys.ATOMIC_ENERGIES]
        else:
            atomic_energies = torch.zeros_like(atomic_charges)
        atomic_energies = atomic_energies.index_add(
            0, center_idx, pair_energies.squeeze()
        )
        data[keys.ATOMIC_ENERGIES] = atomic_energies

        return data


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
