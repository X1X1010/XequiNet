from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter

from xequinet import keys

from .basic import Int2c1eEmbedding, resolve_activation
from .rbf import resolve_cutoff, resolve_rbf


class Embedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        num_basis: int = 20,
        embed_basis: str = "one-hot",
        aux_basis: str = "aux56",
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "cosine",
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        if embed_basis == "one-hot":
            self.embedding = nn.Embedding(100, self.node_dim, padding_idx=0)
        else:
            int2c1e = Int2c1eEmbedding(embed_basis, aux_basis)
            self.embedding = nn.Sequential(
                int2c1e,
                nn.Linear(int2c1e.embed_dim, self.node_dim),
            )
        self.rbf = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        atomic_numbers = data[keys.ATOMIC_NUMBERS]
        vectors = data[keys.EDGE_VECTOR]
        distances = data[keys.EDGE_LENGTH].unsqueeze(-1)

        # node linear
        node_invariant = self.embedding(atomic_numbers)
        data[keys.NODE_INVARIANT] = node_invariant

        # calculate radial basis function
        rbf = self.rbf(distances)
        fcut = self.cutoff_fn(distances)
        data[keys.RADIAL_BASIS_FUNCTION] = rbf
        data[keys.ENVELOPE_FUNCTION] = fcut

        uvec = vectors / distances

        data[keys.SPHERICAL_HARMONICS] = uvec

        node_equivariant = torch.zeros(
            (node_invariant.shape[0], 3, self.node_dim),
            device=node_invariant.device,
        )
        data[keys.NODE_EQUIVARIANT] = node_equivariant

        return data


class PainnMessage(nn.Module):
    """Message function for PaiNN"""

    def __init__(
        self,
        node_dim: int = 128,
        num_basis: int = 20,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.num_basis = num_basis
        self.hidden_dim = self.node_dim * 3
        # scalar feature
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            resolve_activation(activation),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        # vector feature
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        node_scalar = data[keys.NODE_INVARIANT]
        node_equi = data[keys.NODE_EQUIVARIANT]
        rbf = data[keys.RADIAL_BASIS_FUNCTION]
        fcut = data[keys.ENVELOPE_FUNCTION]
        uvec = data[keys.SPHERICAL_HARMONICS]
        edge_index = data[keys.EDGE_INDEX]
        center_idx = edge_index[keys.CENTER_IDX]
        neighbor_idx = edge_index[keys.NEIGHBOR_IDX]

        scalar_out = self.scalar_mlp(node_scalar)
        filter_weight = self.rbf_lin(rbf) * fcut
        filter_out = scalar_out.index_select(0, neighbor_idx) * filter_weight

        message_scalar, gate_edge_vector, gate_state_vector = torch.split(
            filter_out,
            self.node_dim,
            dim=-1,
        )
        message_vector = node_equi.index_select(
            0, neighbor_idx
        ) * gate_state_vector.unsqueeze(1)
        edge_vector = gate_edge_vector.unsqueeze(1) * uvec.unsqueeze(-1)
        message_vector = message_vector + edge_vector

        ori_scalar = data[keys.NODE_INVARIANT]
        ori_equi = data[keys.NODE_EQUIVARIANT]
        data[keys.NODE_INVARIANT] = ori_scalar.index_add(0, center_idx, message_scalar)
        data[keys.NODE_EQUIVARIANT] = ori_equi.index_add(0, center_idx, message_vector)

        return data


class PainnUpdate(nn.Module):
    """Update function for PaiNN"""

    def __init__(
        self,
        node_dim: int = 128,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = self.node_dim * 3
        # vector feature
        self.update_U = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.update_V = nn.Linear(self.node_dim, self.node_dim, bias=False)

        # scalar feature
        self.update_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2, self.node_dim),
            resolve_activation(activation),
            nn.Linear(self.node_dim, self.hidden_dim),
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        node_scalar = data[keys.NODE_INVARIANT]
        node_equi = data[keys.NODE_EQUIVARIANT]

        U_vector = self.update_U(node_equi)
        V_vector = self.update_V(node_equi)

        V_invariant = torch.linalg.norm(V_vector, dim=1)
        mlp_in = torch.cat([node_scalar, V_invariant], dim=-1)
        mlp_out = self.update_mlp(mlp_in)

        a_ss, a_vv, a_sv = torch.split(mlp_out, self.node_dim, dim=-1)
        d_vector = a_vv.unsqueeze(1) * U_vector
        inner_prod = torch.sum(U_vector * V_vector, dim=1)
        d_scalar = a_sv * inner_prod + a_ss

        ori_scalar = data[keys.NODE_INVARIANT]
        ori_equi = data[keys.NODE_EQUIVARIANT]
        data[keys.NODE_INVARIANT] = ori_scalar + d_scalar
        data[keys.NODE_EQUIVARIANT] = ori_equi + d_vector

        return data
