"""
This Python file holds E3 NN blocks for qc matrice output. 
"""

from typing import Union, Tuple, Iterable 

import torch 
import torch.nn as nn 
from e3nn import o3 
from torch_scatter import scatter 

from .rbf import resolve_rbf, resolve_cutoff
from .o3layer import resolve_actfn 
from .tensorproduct import get_feasible_tp 
from .o3layer import EquivariantDot, Int2c1eEmbedding, NormGate
from .matlayer import SelfLayer, PairLayer


class XMatEmbedding(nn.Module):
    """
    Embedding module for qc matrice learning. 
    """
    def __init__(
        self, 
        node_dim: int = 128, 
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 128x1o + 128x2e + 128x3o + 128x4e",
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux56",
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 8.0,
        cutoff_fn: str ="exponential",
    ):
        """
        Args:
        """
        super().__init__()
        self.node_dim = node_dim 
        self.edge_irreps = edge_irreps if isinstance(edge_irreps, o3.Irreps) else o3.Irreps(edge_irreps)
        self.int2c1e = Int2c1eEmbedding(embed_basis, aux_basis) 
        self.embed_dim = self.int2c1e.embed_dim 
        # self.edge_num_irreps = self.edge_irreps.num_irreps
        self.node_lin = nn.Linear(self.embed_dim, self.node_dim)
        nn.init.zeros_(self.node_lin.bias)
        max_l = self.edge_irreps.lmax
        self.irreps_rshs = o3.Irreps.spherical_harmonics(max_l)
        self.rsh_conv = o3.SphericalHarmonics(self.irreps_rshs, normalize=True, normalization="component")
        self.rbf_kernel = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)
        self.rbf_kernel_full = resolve_rbf(rbf_kernel, num_basis, 1000.0)
        self.cutoff_fn_full = resolve_cutoff(cutoff_fn, 1000.0)

    def forward(
        self, 
        at_no: torch.LongTensor,
        pos: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_index_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
        """
        # calculate distance and relative position
        pos = pos[:, [1, 2, 0]]  # [x, y, z] -> [y, z, x]
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        vec_full = pos[edge_index_full[0]] - pos[edge_index_full[1]] 
        dist_full = torch.linalg.vector_norm(vec_full, dim=-1, keepdim=True)

        x = self.int2c1e(at_no) 
        node_0 = self.node_lin(x)

        rbfs = self.rbf_kernel(dist)
        fcut = self.cutoff_fn(dist) 
        rbfs = rbfs * fcut 
        rbfs_full = self.rbf_kernel_full(dist_full)
        fcut_full = self.cutoff_fn_full(dist_full)
        rbfs_full = rbfs_full * fcut_full 
        rshs = self.rsh_conv(vec) 

        return node_0, rbfs, rshs, rbfs_full
        

class NodewiseInteraction(nn.Module):
    """
    E(3) Graph Convoluton Module.
    """
    def __init__(
        self, 
        irreps_node_in: Union[str, o3.Irreps, Iterable],
        irreps_node_out: Union[str, o3.Irreps, Iterable],
        node_dim: int = 128,
        edge_attr_dim: int = 20, 
        actfn: str = "silu",
        residual_update: bool = True,
        use_normgate: bool = True,
    ):
        super().__init__() 
        self.irreps_node_in = irreps_node_in if isinstance(irreps_node_in, o3.Irreps) else o3.Irreps(irreps_node_in)
        self.irreps_node_out = irreps_node_out if isinstance(irreps_node_out, o3.Irreps) else o3.Irreps(irreps_node_out)
        max_l = self.irreps_node_out.lmax
        self.irreps_rshs = o3.Irreps.spherical_harmonics(max_l)
        self.message_update = self.irreps_node_in == self.irreps_node_out 
        self.residual_update = residual_update and self.irreps_node_in == self.irreps_node_out 
        self.use_normgate = use_normgate
        self.actfn = resolve_actfn(actfn)        
        self.inner_dot = EquivariantDot(self.irreps_node_in) 
        self.irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_node_in, self.irreps_rshs, self.irreps_node_out, tp_mode="uvu"
        )
        self.tp_node = o3.TensorProduct(
            self.irreps_node_in,
            self.irreps_rshs,
            self.irreps_tp_out,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.lin_edge_scalar = nn.Sequential(
            nn.Linear(self.irreps_node_in.num_irreps+node_dim, 32, bias=True),
            self.actfn, 
            nn.Linear(32, self.tp_node.weight_numel, bias=True)
        )
        self.lin_edge_rbf = nn.Sequential(
            nn.Linear(edge_attr_dim, 32, bias=True),
            self.actfn, 
            nn.Linear(32, self.tp_node.weight_numel, bias=True) 
        )
        if use_normgate:
            self.lin_node0 = o3.Linear(self.irreps_node_in, self.irreps_node_in, biases=True)
            self.lin_node1 = o3.Linear(self.irreps_node_in, self.irreps_node_in, biases=True)
            self.normgate = NormGate(self.irreps_node_in, actfn)
        self.lin_node2 = o3.Linear(self.irreps_tp_out, self.irreps_node_out, biases=True) 
        # self.e3norm = EquivariantLayerNorm(self.irreps_node_out) 
        
    def forward(
        self, 
        node_feat:torch.Tensor,  
        edge_attr:torch.Tensor, 
        edge_rshs:torch.Tensor, 
        edge_index:torch.LongTensor
    ) -> torch.Tensor:
        if self.use_normgate:
            pre_x = self.lin_node0(node_feat)
            s0 = self.inner_dot(pre_x[edge_index[0], :], pre_x[edge_index[1], :])[:, self.irreps_node_in.slices()[0].stop:]
            s0 = torch.cat(
                [
                    pre_x[edge_index[0]][:, self.irreps_node_in.slices()[0]], 
                    pre_x[edge_index[1]][:, self.irreps_node_in.slices()[0]], 
                    s0
                ], dim=-1
            )
            x = self.lin_node1(self.normgate(node_feat))
        else:
            x = node_feat 
            s0 = self.inner_dot(x[edge_index[0], :], x[edge_index[1], :])[:, self.irreps_node_in.slices()[0].stop:]
            s0 = torch.cat(
                [
                    x[edge_index[0]][:, self.irreps_node_in.slices()[0]], 
                    x[edge_index[1]][:, self.irreps_node_in.slices()[0]], 
                    s0
                ], dim=-1
            )
        x_j = x[edge_index[1], :] 
        msg_j = self.tp_node(x_j, edge_rshs, self.lin_edge_scalar(s0) * self.lin_edge_rbf(edge_attr)) 
        accu_msg = scatter(msg_j, edge_index[0], dim=0, dim_size=node_feat.size(0))
        if self.message_update:
            accu_msg = accu_msg + x 
        
        out = self.lin_node2(accu_msg)
        if self.residual_update:
            out = node_feat + out
        return out


class MatTrans(nn.Module):
    """
    Module to read out convolution layer output.
    """
    def __init__(
        self,
        node_dim:int = 128,
        hidden_dim:int = 64,
        max_l:int = 4,
        edge_dim:int = 20,
        actfn:str = "silu",
    ):
        super().__init__()
        self.irreps_in = o3.Irreps([(node_dim, (l, (-1)**l)) for l in range(max_l+1)]) 
        irreps_hidden = [(hidden_dim, (0, 1))] 
        for l in range(2, 2*max_l+1):
            irreps_hidden.append((hidden_dim, (l//2, (-1)**l)))
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        # Building block 
        self.node_self_layer = SelfLayer(self.irreps_in, self.irreps_hidden, actfn)
        self.node_pair_layer = PairLayer(self.irreps_in, self.irreps_hidden, edge_dim, actfn)
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.LongTensor,
        fii: torch.Tensor,
        fij: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fii = self.node_self_layer(node_feat, fii)
        fij = self.node_pair_layer(node_feat, edge_attr, edge_index, fij)
        return fii, fij

