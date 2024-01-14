from typing import Union, Tuple, Iterable 

import torch 
import torch.nn as nn 
from e3nn import o3 

from .rbf import resolve_rbf, resolve_cutoff
from .matlayer import SelfLayer, PairLayer 


class FullEdgeKernel(nn.Module):
    """
    RBF embedding layer for fully connected graph.
    """
    def __init__(
        self,
        rbf_kernel: str = "gaussian",
        num_basis: int = 32,
        cutoff: float = 8.0,
        cutoff_fn: str = "cosine",
    ):
        super().__init__()
        self._num_rbf = num_basis
        self._rcut = cutoff 
        self.rbf_kernel = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)
    
    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> torch.Tensor:
        pos = pos[:, [1, 2, 0]] # (x, y, z) to (y, z, x)
        vec = pos[edge_index[0]] - pos[edge_index[1]] 
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True) 
        rbfs = self.rbf_kernel(dist)
        fcut = self.cutoff_fn(dist)
        return rbfs * fcut 


class XMatTrans(nn.Module):
    """
    Read out transform for xPaiNNorb from xPaiNN layer output.
    """
    def __init__(
        self,
        irreps_node: Union[str, o3.Irreps, Iterable] = "128x0e+128x1o+128x2e+128x3o+128x4e",
        hidden_dim: int = 64,
        max_l: int = 4, 
        edge_dim: int = 20,
        actfn: str = "silu",
    ):
        super().__init__()
        self.irreps_in = irreps_node if isinstance(irreps_node, o3.Irreps) else o3.Irreps(irreps_node)
        assert self.irreps_in.lmax == max_l 
        self.irreps_hidden_base = o3.Irreps([(hidden_dim, (l, (-1)**l)) for l in range(max_l + 1)]) 
        irreps_hidden = [(hidden_dim, (0, 1))]
        for l in range(2, 2*max_l+1):
            irreps_hidden.append((hidden_dim, (l//2, (-1)**l)))
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        # Building block 
        if self.irreps_in == self.irreps_hidden_base:
            self.pretrans = False 
        else:
            self.pretrans = True
            self.node_pre_trans = o3.Linear(self.irreps_in, self.irreps_hidden_base, biases=False)
        self.node_self_layer = SelfLayer(self.irreps_hidden_base, self.irreps_hidden, actfn)
        self.node_pair_layer = PairLayer(self.irreps_hidden_base, self.irreps_hidden, edge_dim, actfn)
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.LongTensor,
        fii: Union[torch.Tensor, None],
        fij: Union[torch.Tensor, None],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pretrans:
            node_feat = self.node_pre_trans(node_feat)
        fii = self.node_self_layer(node_feat, fii)
        fij = self.node_pair_layer(node_feat, edge_attr, edge_index, fij)
        return fii, fij
