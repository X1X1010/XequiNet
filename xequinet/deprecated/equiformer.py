from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch_geometric.utils
from torch_scatter import scatter
from e3nn import o3

from xpainn.nn.o3layer import (
    ElementShifts, CGCoupler, Gate,
    o3LayerNorm,
)
from xpainn.nn.rbf import SphericalBesselj0


class EqEmbedding(nn.Module):
    def __init__(
        self,
        node_irreps: str | o3.Irreps | Iterable,
        edge_irreps: str | o3.Irreps | Iterable,
        max_atom_types: int = 119,
        cutoff: float = 5.0,
        cutoff_fn: str = "polynomial",
        num_basis: int = 16,
    ):
        super().__init__()
        # node embedding
        self.max_atom_types = max_atom_types
        self.node_irreps = o3.Irreps(node_irreps).simplify()
        biases = [False for _ in self.node_irreps]
        biases[0] = True
        self.atom_type_lin = o3.Linear(f"{self.max_atom_types}x0e", self.node_irreps, biases=biases)       
        
        # rbf & rsh
        self.edge_irreps = o3.Irreps(edge_irreps).simplify()
        self.rbf = SphericalBesselj0(num_basis, cutoff, cutoff_fn)
        self.rsh = o3.SphericalHarmonics(self.edge_irreps, normalize=True, normalization="component")

        # edge degree embedding
        self.expand = o3.Linear("1x0e", self.node_irreps, biases=True)
        # self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_irreps.num_irreps}x0e")
        filter_ir_out = [ir for _, ir in self.node_irreps]
        self.dtp = CGCoupler(self.node_irreps, self.edge_irreps, filter_ir_out, use_weights=True)
        self.rbf_lin = nn.Linear(num_basis, self.dtp.weight_numel, bias=False)
        self.proj = o3.Linear(self.dtp.irreps_out.simplify(), self.node_irreps)
        
    def forward(self,
            at_no: torch.Tensor,
            pos: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # node embedding
        at_onehot = torch.nn.functional.one_hot(at_no, self.max_atom_types).float()
        node_embedding = self.atom_type_lin(at_onehot)
        # rbf & rsh
        pos = pos[:, [1, 2, 0]]  # [x, y, z] -> [y, z, x]
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        rbf = self.rbf(dist)
        rsh = self.rsh(vec)  # normalize=True, unit vector
        # edge degree embedding
        one_embedding = torch.ones_like(node_embedding.narrow(1, 0, 1))
        one_embedding = self.expand(one_embedding)
        # edge_embedding = self.rsh_conv(rsh, self.rbf_lin(rbf))
        edge_embedding = self.dtp(one_embedding[edge_index[0]], rsh, self.rbf_lin(rbf))
        edge_embedding = self.proj(edge_embedding)
        edge_embedding = scatter(edge_embedding, edge_index[1], dim=0, dim_size=node_embedding.shape[0])
        node_embedding = node_embedding + edge_embedding
        return node_embedding, rbf, rsh


class FeedForward(nn.Module):
    def __init__(
        self,
        node_irreps: str | o3.Irreps | Iterable,
        hidden_irreps: str | o3.Irreps | Iterable,
    ):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps).simplify()
        self.hidden_irreps = o3.Irreps(hidden_irreps).simplify()
        self.layer_norm = o3LayerNorm(self.node_irreps)
        self.pre_lin = o3.Linear(self.node_irreps, self.hidden_irreps)
        self.gate = Gate(self.hidden_irreps, "sigmoid")
        self.post_lin = o3.Linear(self.hidden_irreps, self.node_irreps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x)
        y = self.post_lin(self.gate(self.pre_lin(y)))
        return x + y


def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0

class GraphAttention(nn.Module):
    def __init__(
        self,
        node_irreps: str | o3.Irreps | Iterable,
        node_pre_irreps: Optional[str | o3.Irreps | Iterable],
        attn_irreps: str | o3.Irreps | Iterable,
        edge_irreps: str | o3.Irreps | Iterable,
        num_basis: int = 16,
        num_heads: int = 4,

    ):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps).simplify()
        self.edge_irreps = o3.Irreps(edge_irreps).simplify()
        self.attn_irreps = o3.Irreps(attn_irreps).simplify()
        self.node_pre_irreps = self.node_irreps if node_pre_irreps is None else o3.Irreps(node_pre_irreps).simplify()
        self.num_heads = num_heads

        # layer norm
        self.layer_norm = o3LayerNorm(self.node_irreps)

        # merge src and dst
        self.merge_src = o3.Linear(self.node_irreps, self.node_pre_irreps)
        self.merge_dst = o3.Linear(self.node_irreps, self.node_pre_irreps)

        # dtp1
        # self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_irreps.num_irreps}x0e")
        filter_ir_out = [ir for _, ir in self.edge_irreps]
        self.dtp1 = CGCoupler(self.node_pre_irreps, self.edge_irreps, filter_ir_out, use_weights=True)
        self.rbf_lin1 = nn.Linear(num_basis, self.dtp1.weight_numel, bias=False)

        # alpha
        attn_scalar = get_mul_0(self.attn_irreps)
        self.sep_alpha = o3.Linear(self.dtp1.irreps_out.simplify(), f"{attn_scalar * num_heads}x0e")
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, attn_scalar))
        torch_geometric.nn.inits.glorot(self.alpha_dot)

        # value
        self.sep_value = o3.Linear(self.dtp1.irreps_out.simplify(), self.node_pre_irreps)
        self.gate = Gate(self.node_pre_irreps, "sigmoid")
        self.dtp2 = CGCoupler(self.node_pre_irreps, self.edge_irreps, filter_ir_out, use_weights=True,
            internal_weights=True, shared_weights=True)
        # self.rbf_lin2 = nn.Linear(num_basis, self.dtp2.weight_numel, bias=False)
        self.value_lin = o3.Linear(self.dtp2.irreps_out.simplify(), self.attn_irreps * num_heads)

        # post
        self.proj = o3.Linear(self.attn_irreps * num_heads, self.node_irreps)

    def forward(self, node, edge_index, rbf, rsh):
        # layer norm
        node_norm = self.layer_norm(node)
        # merge src and dst
        msg_src = self.merge_src(node_norm)
        msg_dst = self.merge_dst(node_norm)
        msg = msg_src[edge_index[0]] + msg_dst[edge_index[1]]
        # dtp1
        # edge1 = self.rsh_conv(rsh, self.rbf_lin1(rbf))
        message = self.dtp1(msg, rsh, self.rbf_lin1(rbf))
        # alpha
        alpha = self.sep_alpha(message).view(message.shape[0], self.num_heads, -1)
        alpha = self.leaky_relu(alpha)
        alpha = torch.sum((alpha * self.alpha_dot), dim=-1, keepdim=True)
        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
        # value
        value = self.sep_value(message)
        value = self.gate(value)
        # edge2 = self.rsh_conv(rsh, self.rbf_lin2(rbf))
        value = self.dtp2(value, rsh)
        value = self.value_lin(value).view(value.shape[0], self.num_heads, -1)
        # inner product
        attn = alpha * value
        attn = scatter(attn, index=edge_index[1], dim=0, dim_size=node.shape[0])
        attn = attn.view(attn.shape[0], -1)
        # post
        node = node + self.proj(attn)
        return node


class OutHead(nn.Module):
    def __init__(
        self,
        node_irreps: str | o3.Irreps | Iterable,
        out_dim: int = 1,
        shifts_idx: Optional[torch.LongTensor] = None,
        shifts_value: Optional[torch.DoubleTensor] = None,
        freeze: bool = False,
    ):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps).simplify()
        self.layer_norm = o3LayerNorm(self.node_irreps)
        self.head = nn.Sequential(
            o3.Linear(self.node_irreps, self.node_irreps),
            Gate(self.node_irreps, "sigmoid"),
            o3.Linear(self.node_irreps, f"{out_dim}x0e"),
        )
        self.shifts = ElementShifts(out_dim, shifts_idx, shifts_value, freeze)

    def forward(
        self,
        x: torch.Tensor,
        at_no: torch.LongTensor,
        batch_idx: torch.LongTensor,

    ):
        x = self.layer_norm(x)
        x = self.head(x) / 18.03065905448718
        ele_shifts = self.shifts(at_no)
        res = scatter(x.double() + ele_shifts, batch_idx, dim=0)
        return res


class Equiformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EqEmbedding(
            node_irreps="128x0e + 64x1e + 32x2e",
            edge_irreps="1x0e + 1x1e + 1x2e",
            max_atom_types=10,
            cutoff=5.0,
            cutoff_fn="polynomial",
            num_basis=32,
        )
        self.graph_attns = nn.ModuleList([
            GraphAttention(
                node_irreps="128x0e + 64x1e + 32x2e",
                node_pre_irreps="128x0e + 64x1e + 32x2e",
                edge_irreps="1x0e + 1x1e + 1x2e",
                attn_irreps="32x0e + 16x1e + 8x2e",
                num_heads=4,
                num_basis=32,
            ) for _ in range(6)
        ])
        self.fead_forwards = nn.ModuleList([
            FeedForward(
                node_irreps="128x0e + 64x1e + 32x2e",
                hidden_irreps="384x0e + 192x1e + 96x2e",
            ) for _ in range(6)
        ])
        shifts_value = torch.DoubleTensor([-0.6038066, -38.0740441, -54.7491437, -75.2252374, -99.8658573]) * 27.211386024367243
        shifts_value = shifts_value.view(-1, 1)
        self.out_heads = OutHead(
            node_irreps="128x0e + 64x1e + 32x2e",
            shifts_idx=torch.LongTensor([1, 6, 7, 8, 9]),
            shifts_value=shifts_value,
        )

    def forward(
        self,
        at_no: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ):
        x, rbf, rsh = self.embedding(at_no, pos, edge_index)
        for attn, ffd in zip(self.graph_attns, self.fead_forwards):
            x = attn(x, edge_index, rbf, rsh)
            x = ffd(x)
        res = self.out_heads(x, at_no, batch)
        return res


if __name__ == "__main__":
    at_no = torch.LongTensor([8, 1, 1])
    h2o_coord = torch.Tensor([
        [0.00000000,  0.00000000, -0.11081188],
        [0.00000000, -0.78397589,  0.44324751],
        [0.00000000,  0.78397589,  0.44324751]
    ])
    edge_index = torch.LongTensor([[0, 0, 1, 2], [1, 2, 0, 0]])
    equiformer = Equiformer()

    res = equiformer(at_no, h2o_coord, edge_index, torch.LongTensor([0, 0, 0]))
    print(res)