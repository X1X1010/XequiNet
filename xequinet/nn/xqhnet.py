"""
This Python file holds E3 NN blocks for qc matrix output. 
"""
from typing import Tuple, Iterable, Optional

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter

from .rbf import resolve_rbf, resolve_cutoff
from .o3layer import resolve_actfn
from .tp import get_feasible_tp
from .o3layer import EquivariantDot, Int2c1eEmbedding, Gate
from .matlayer import SelfLayer, PairLayer, Expansion


class MatEmbedding(nn.Module):
    """
    Embedding module for qc matrice output.
    """
    def __init__(
        self, 
        node_dim: int = 128,
        node_channels: int = 128,
        max_l: int = 4,
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux56",
        num_basis: int = 20,
        rbf_kernel: str = "expbern",
        cutoff: float = 8.0,
        cutoff_fn: str ="exponential",
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_channels`: Number of channels for each `Irrep`.
            `max_l`: Maximum angular momentum.
            `embed_basis`: Embedding basis.
            `aux_basis`: Auxiliary basis.
            `num_basis`: Number of basis.
            `rbf_kernel`: Radial basis function type.
            `cutoff`: Cutoff distance.
            `cutoff_fn`: Cutoff function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps([(node_channels, (l, (-1)**l)) for l in range(max_l + 1)])
        self.int2c1e = Int2c1eEmbedding(embed_basis, aux_basis)
        self.embed_dim = self.int2c1e.embed_dim

        self.node_lin = nn.Linear(self.embed_dim, self.node_dim)
        nn.init.zeros_(self.node_lin.bias)

        self.irreps_rsh = o3.Irreps.spherical_harmonics(max_l)
        self.rsh_conv = o3.SphericalHarmonics(self.irreps_rsh, normalize=True, normalization="component")
        self.rbf_kernel = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)
        self.rbf_kernel_full = resolve_rbf(rbf_kernel, num_basis, 100 * cutoff)
        self.cutoff_fn_full = resolve_cutoff(cutoff_fn, 100 * cutoff)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_index_full: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            `at_no`: Atomic numbers.
            `pos`: Atomic positions.
            `edge_index`: Edge index.
            `edge_index_full`: Full edge index.
        Returns:
            `node_0`: Initial node features.
            `rbf`: Radial basis function.
            `rsh`: Real spherical harmonics.
            `rbf_full`: Radial basis function for full edge index.
        """
        # calculate distance and relative position
        pos = pos[:, [1, 2, 0]]  # [x, y, z] -> [y, z, x]
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        vec_full = pos[edge_index_full[0]] - pos[edge_index_full[1]]
        dist_full = torch.linalg.vector_norm(vec_full, dim=-1, keepdim=True)

        x = self.int2c1e(at_no)
        node_0 = self.node_lin(x)

        rbf = self.rbf_kernel(dist) * self.cutoff_fn(dist)
        rbf_full = self.rbf_kernel_full(dist_full) * self.cutoff_fn_full(dist_full)
        rsh = self.rsh_conv(vec)

        return node_0, rbf, rsh, rbf_full


# TODO: use my own gate first to check the performance
class NodewiseInteraction(nn.Module):
    """
    E(3) Graph Convoluton Module.
    """
    def __init__(
        self,
        irreps_node_in: Iterable,
        irreps_node_out: Iterable,
        edge_attr_dim: int = 20,
        actfn: str = "silu",
    ) -> None:
        super().__init__()
        # check input and output irreps
        self.irreps_node_in = o3.Irreps(irreps_node_in)
        self.irreps_node_out = o3.Irreps(irreps_node_out)
        self.not_first_layer = (self.irreps_node_in == self.irreps_node_out)
        self.node_channels = self.irreps_node_in[0].mul

        max_l = self.irreps_node_out.lmax
        self.irreps_rsh = o3.Irreps.spherical_harmonics(max_l)
        self.activation = resolve_actfn(actfn)
        self.inner_dot = EquivariantDot(self.irreps_node_in)
        # tensor product between node and rij
        self.irreps_tp_out, instruct = get_feasible_tp(
            self.irreps_node_in, self.irreps_rsh, self.irreps_node_out, "uvu",
        )
        self.tp = o3.TensorProduct(
            self.irreps_node_in,
            self.irreps_rsh,
            self.irreps_node_out,
            instruct,
            internal_weights=False,
            shared_weights=False,
        )
        self.mlp_edge_scalar = nn.Sequential(
            nn.Linear(self.irreps_node_in.num_irreps + self.node_channels, 32, bias=True),
            self.activation,
            nn.Linear(32, self.tp.weight_numel, bias=True),
        )
        self.mlp_edge_rbf = nn.Sequential(
            nn.Linear(edge_attr_dim, 32, bias=True),
            self.activation,
            nn.Linear(32, self.tp.weight_numel, bias=True),
        )
        if self.not_first_layer:  # use gate with MLP
            self.lin_node0 = o3.Linear(self.irreps_node_in, self.irreps_node_in, biases=True)
            self.lin_node1 = o3.Linear(self.irreps_node_in, self.irreps_node_in, biases=True)
            self.gate = Gate(self.irreps_node_in, refine=True)
        self.lin_node2 = o3.Linear(self.irreps_node_out, self.irreps_node_out, biases=True)
        # the last index of the scalar part
        self.num_scalar = self.irreps_node_in[0].mul

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_rsh: torch.Tensor,
        edge_index : torch.Tensor,
    ) -> torch.Tensor:
        if self.not_first_layer:
            pre_x = self.lin_node0(node_feat)
            s0 = self.inner_dot(pre_x[edge_index[0]], pre_x[edge_index[1]])[:, self.num_scalar :]
            s0 = torch.cat([
                pre_x[edge_index[0]][:, : self.num_scalar],
                pre_x[edge_index[1]][:, : self.num_scalar],
                s0
            ], dim=-1)
            x = self.lin_node1(self.gate(node_feat))
        else:
            x = node_feat
            s0 = self.inner_dot(x[edge_index[0]], x[edge_index[1]])[:, self.num_scalar :]
            s0 = torch.cat([
                x[edge_index[0]][:, : self.num_scalar],
                x[edge_index[1]][:, : self.num_scalar],
                s0
            ], dim=-1)
        x_j = x[edge_index[1]]
        tp_weight = self.mlp_edge_scalar(s0) * self.mlp_edge_rbf(edge_attr)
        msg_j = self.tp(x_j, edge_rsh, tp_weight)
        accu_msg = scatter(msg_j, edge_index[0], dim=0, dim_size=x.shape[0])

        if self.not_first_layer:
            accu_msg += x
        out = self.lin_node2(accu_msg)

        if self.not_first_layer:
            out += node_feat
        return out
    

class MatrixTrans(nn.Module):
    """
    Module to read out convolution layer output.
    """
    def __init__(
        self,
        node_channels: int = 128,
        hidden_channels: int = 64,
        max_l: int = 4,
        edge_attr_dim: int = 20,
        actfn: str = "silu",
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps([(node_channels, (l, (-1)**l)) for l in range(max_l + 1)])
        irreps_hidden = [(hidden_channels, (0, 1))]
        for l in range(1, max_l + 1):
            irreps_hidden.append((hidden_channels, (l, -1)))
            irreps_hidden.append((hidden_channels, (l, 1)))
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        # buliding block
        self.self_layer = SelfLayer(self.irreps_in, self.irreps_hidden, actfn)
        self.pair_layer = PairLayer(self.irreps_in, self.irreps_hidden, edge_attr_dim, actfn)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr_full: torch.Tensor,
        edge_index_full: torch.Tensor,
        fii: Optional[torch.Tensor],
        fij: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fii = self.self_layer(node_feat, fii)
        fij = self.pair_layer(node_feat, edge_attr_full, edge_index_full, fij)
        return fii, fij
    

class MatrixOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_channels: int = 64,
        irreps_out: Iterable = "3x0e + 2x1o + 1x2e",
        max_l: int = 4,
        actfn: str = "silu",
    ) -> None:
        super().__init__()
        block_channels = hidden_channels // 2
        irreps_hidden = [(hidden_channels, (0, 1))]
        irreps_block = [(block_channels, (0, 1))]
        for l in range(1, max_l + 1):
            irreps_hidden.append((hidden_channels, (l, -1)))
            irreps_hidden.append((hidden_channels, (l, 1)))
            irreps_block.append((block_channels, (l, -1)))
            irreps_block.append((block_channels, (l, 1)))
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_block = o3.Irreps(irreps_block)
        self.irreps_out = o3.Irreps(irreps_out)

        self.lin_out_ii = o3.Linear(self.irreps_hidden, self.irreps_block, biases=False)
        self.lin_out_ij = o3.Linear(self.irreps_hidden, self.irreps_block, biases=False)
        self.expand_ii = Expansion(self.irreps_block, self.irreps_out, node_dim, actfn, pair_out=False)
        self.expand_ij = Expansion(self.irreps_block, self.irreps_out, node_dim, actfn, pair_out=True)

    def forward(
        self,
        fii: torch.Tensor,
        fij: torch.Tensor,
        node_embed: torch.Tensor,
        edge_index_full: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, src_indices = torch.sort(edge_index_full[1], stable=True)
        node_embed_pair = torch.cat([node_embed[edge_index_full[0]], node_embed[edge_index_full[1]]], dim=-1)
        fii = self.lin_out_ii(fii)
        fij = self.lin_out_ij(fij)
        diagnol_block = self.expand_ii(fii, node_embed)
        offdiagnol_block = self.expand_ij(fij, node_embed_pair)
        # symmetrize
        diagnol_block = 0.5 * (diagnol_block + diagnol_block.transpose(-1, -2))
        offdiagnol_block_T = torch.index_select(offdiagnol_block.transpose(-1, -2), dim=0, index=src_indices)
        offdiagnol_block = 0.5 * (offdiagnol_block + offdiagnol_block_T)
        return diagnol_block, offdiagnol_block