from typing import Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter

from .o3layer import resolve_actfn, Int2c1eEmbedding
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
    ):
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

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # calculate distance and relative position
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        # node linear
        x_scalar = self.embedding(at_no)
        # calculate radial basis function
        rbf = self.rbf(dist)
        fcut = self.cutoff_fn(dist)
        # calculate spherical harmonics
        uvec = vec / dist
        return x_scalar, rbf, fcut, uvec


class PainnMessage(nn.Module):
    """Message function for PaiNN"""
    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_basis: int = 20,
        actfn: str = "silu",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_basis = num_basis
        self.hidden_dim = self.node_dim + self.edge_dim * 2
        # scalar feature
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            resolve_actfn(actfn),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.scalar_mlp[0].bias)
        nn.init.zeros_(self.scalar_mlp[2].bias)
        # vector feature
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim)
        nn.init.zeros_(self.rbf_lin.bias)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_vector: torch.Tensor,
        rbf: torch.Tensor,
        fcut: torch.Tensor,
        uvec: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        scalar_out = self.scalar_mlp(x_scalar)
        filter_weight = self.rbf_lin(rbf) * fcut
        filter_out = scalar_out[edge_index[1]] * filter_weight
        
        message_scalar, gate_edge_vector, gate_state_vector = torch.split(
            filter_out,
            [self.node_dim, self.edge_dim, self.edge_dim],
            dim=-1,
        )
        message_vector = x_vector[edge_index[1]] * gate_state_vector.unsqueeze(1)
        edge_vector = gate_edge_vector.unsqueeze(1) * uvec.unsqueeze(-1)
        message_vector = message_vector + edge_vector

        new_scalar = x_scalar + scatter(message_scalar, edge_index[0], dim=0)
        new_vector = x_vector + scatter(message_vector, edge_index[0], dim=0)

        return new_scalar, new_vector



class PainnUpdate(nn.Module):
    """Update function for PaiNN"""
    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 128,
        actfn: str = "silu",
    ):
        super().__init__()
        assert node_dim == edge_dim, "node_dim must be equal to edge_dim"
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = self.node_dim * 2 + self.edge_dim
        # vector feature
        self.update_U = nn.Linear(self.edge_dim, self.edge_dim, bias=False)
        self.update_V = nn.Linear(self.edge_dim, self.edge_dim, bias=False)

        # scalar feature
        self.update_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.edge_dim, self.node_dim),
            resolve_actfn(actfn),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.update_mlp[0].bias)
        nn.init.zeros_(self.update_mlp[2].bias)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        U_vector = self.update_U(x_vector)
        V_vector = self.update_V(x_vector)

        V_invariant = torch.linalg.norm(V_vector, dim=1)
        mlp_in = torch.cat([x_scalar, V_invariant], dim=-1)
        mlp_out = self.update_mlp(mlp_in)

        a_ss, a_vv, a_sv = torch.split(
            mlp_out,
            [self.node_dim, self.edge_dim, self.node_dim],
            dim=-1
        )
        d_vector = a_vv.unsqueeze(1) * U_vector
        inner_prod = torch.sum(U_vector * V_vector, dim=1)
        d_scalar = a_sv * inner_prod + a_ss

        return x_scalar + d_scalar, x_vector + d_vector
