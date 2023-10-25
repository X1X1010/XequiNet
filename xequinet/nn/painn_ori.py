from typing import Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter

from .rbf import GaussianSmearing, SphericalBesselj0, CosineCutoff
from ..utils import NetConfig
from .o3layer import resolve_actfn


def resolve_rbf(rbf_kernel: str, num_basis: int, cutoff: float):
    if rbf_kernel == "bessel":
        return SphericalBesselj0(num_basis, cutoff)
    elif rbf_kernel == "gaussian":
        return GaussianSmearing(num_basis, cutoff)
    else:
        raise NotImplementedError(f"rbf kernel {rbf_kernel} is not implemented")



class OriEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "cosine",

    ):
        super().__init__()
        self.node_dim = node_dim
        self.embedding = nn.Embedding(100, self.node_dim, padding_idx=0)
        self.rbf = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # calculate distance and relative position
        # pos = pos[:, [1, 2, 0]]  # [x, y, z] -> [y, z, x]
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        # node linear
        x_scalar = self.embedding(at_no)
        # calculate radial basis function
        rbf = self.rbf(dist)
        envelope = self.cutoff_fn(dist)
        # calculate spherical harmonics
        rsh = vec / dist
        return x_scalar, rbf, envelope, rsh


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
        envelope: torch.Tensor,
        rsh: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        scalar_out = self.scalar_mlp(x_scalar)
        filter_weight = self.rbf_lin(rbf) * envelope
        filter_out = scalar_out[edge_index[1]] * filter_weight
        
        message_scalar, gate_edge_vector, gate_state_vector = torch.split(
            filter_out,
            [self.node_dim, self.edge_dim, self.edge_dim],
            dim=-1,
        )
        message_vector = x_vector[edge_index[1]] * gate_state_vector.unsqueeze(1)
        edge_vector = gate_edge_vector.unsqueeze(1) * rsh.unsqueeze(-1)
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


class ScalarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 1,
        actfn: str = "silu",
        reduce_op: str = "sum",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.reduce_op = reduce_op
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        nn.init.zeros_(self.out_mlp[0].bias)
        nn.init.zeros_(self.out_mlp[2].bias)
    
    def forward(
        self,
        x_scalar: torch.Tensor,
        at_no: torch.LongTensor,
        coords: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> torch.DoubleTensor:
        """
        Args:
            `atom_out`: Atomic outputs.
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates, must required grads.
            `batch_idx`: Index of the graphs in the batch.
        Returns:
            Scalar outputs.
        """
        assert len(x_scalar) == len(at_no) == len(batch_idx), \
            "x_scalar, at_no and batch_idx must have the same length"
        atom_out = self.out_mlp(x_scalar) - 4.2433421
        res = scatter(
            atom_out,
            batch_idx,
            dim=0,
            reduce=self.reduce_op,
        )
        return res


class PaiNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.embed = OriEmbedding()
        self.message = nn.ModuleList([
            PainnMessage()
            for _ in range(3)
        ])
        self.update = nn.ModuleList([
            PainnUpdate()
            for _ in range(3)
        ])
        self.out = ScalarOut()
    
    def forward(
        self,
        x: torch.Tensor,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.LongTensor,
        batch_idx: torch.LongTensor,
    ) -> torch.Tensor:
        x_scalar, rbf, envelop, rsh = self.embed(at_no, pos, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], 3, 128), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, envelop, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        result = self.out(x_scalar, at_no, pos, batch_idx)
        return result