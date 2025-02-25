from typing import Dict, Iterable

import math 
import torch 
import torch.nn as nn 
from e3nn import o3 

from xequinet import keys 

from .basic import resolve_activation 
from .o3layer import Invariant, EquivariantDot, EquivariantLayerNorm


class EquiFilter(nn.Module):
    def __init__(
        self,
        node_dim:int,
        node_irreps: Iterable,
        num_basis:int,
        activation:str = 'silu',
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `num_basis`: Number of the radial basis functions.
            `activation`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim 
        self.node_irreps = o3.Irreps(node_irreps) 
        self.num_basis = num_basis
        self.node_num_irreps = self.node_irreps.num_irreps

        self.mlp_rbf = nn.Sequential(
            nn.Linear(self.num_basis, node_dim),
            resolve_activation(activation),
            nn.Linear(node_dim, node_dim),
        )
        self.equi_dot = EquivariantDot(self.node_irreps)
        self.equi_inv = Invariant(self.node_irreps)
        self.mlp_inv = nn.Sequential(
            nn.Linear(self.node_num_irreps, node_dim),
            resolve_activation(activation),
            nn.Linear(node_dim, node_dim),
        )

    def forward(
        self, 
        x_equi:torch.Tensor, 
        rbf:torch.Tensor, 
        center_idx:torch.LongTensor,
        neighbor_idx:torch.LongTensor,
    ) -> torch.Tensor:
        x_i = torch.index_select(x_equi, 0, center_idx)
        x_j = torch.index_select(x_equi, 0, neighbor_idx) 
        x_ij = x_j - x_i 
        invariant_x_ij = self.equi_dot(x_ij, x_ij) 
        w_ij_l = self.mlp_inv(invariant_x_ij)
        w_ij_r = self.mlp_rbf(rbf) 
        w_ij = w_ij_l + w_ij_r
        return w_ij 


class InteractionBlock(nn.Module):
    def __init__(
        self, 
        node_irreps: Iterable,
        node_dim:int = 128, 
        activation: str ='silu',
        layer_norm: bool = True,
    ):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps) 
        self.node_dim = node_dim
        self.node_num_irreps = self.node_irreps.num_irreps 
        self.equi_dot = EquivariantDot(self.node_irreps)
        self.scalar_mul = o3.ElementwiseTensorProduct(self.node_irreps, o3.Irreps(f"{self.node_num_irreps}x0e"))
        concat_dim = self.node_dim + self.node_irreps 
        # self.lin_mix = nn.Linear(concat_dim, concat_dim, bias=True)
        self.lin_mix_mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim, bias=True),
            resolve_activation(activation),
            nn.Linear(concat_dim, concat_dim, bias=True),
        )
        # normalization
        self.norm = nn.LayerNorm(self.node_dim) if layer_norm else nn.Identity()
        self.o3norm = EquivariantLayerNorm(self.node_irreps) if layer_norm else nn.Identity()
    
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # interaction block
        node_scalar = self.norm(data[keys.NODE_INVARIANT])
        node_equi = self.o3norm(data[keys.NODE_EQUIVARIANT])
        equi_inv = self.equi_dot(node_equi, node_equi)
        cat_feat = torch.cat([node_scalar, equi_inv], dim=-1)
        mix_feat = self.lin_mix_mlp(cat_feat)
        d_scalar, x_gate = torch.split(mix_feat, [self.node_dim, self.node_num_irreps], dim=-1)
        d_equi = self.scalar_mul(node_equi, x_gate) 
        # update data
        ori_scalar = data[keys.NODE_INVARIANT]
        ori_equi = data[keys.NODE_EQUIVARIANT]
        data[keys.NODE_INVARIANT] = ori_scalar + d_scalar
        data[keys.NODE_EQUIVARIANT] = ori_equi + d_equi

        return data


class EculideanAttention(nn.Module):
    def __init__(
        self,
        node_irreps:Iterable,
        node_dim:int = 120,
        num_heads:int = 4,
        num_basis:int = 20,
        activation:str = 'silu',
        layer_norm:bool = True,
    ):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps) 
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.l_max = self.node_irreps.lmax
        assert self.node_dim % self.num_heads == 0, \
            f"node_dim must be divisible by num_heads, got node_dim={self.node_dim} and num_heads={self.num_heads}."
        assert self.node_dim % (self.l_max + 1) == 0, \
            f"node_dim must be divisible by l_max + 1, got node_dim={self.node_dim} and l_max={self.l_max}."
        self.attn_dim_scalar = node_dim // num_heads
        self.attn_dim_equi = node_dim // (self.l_max + 1)
        self.node_num_irreps = self.node_irreps.num_irreps
        self.scale_scalar = 1.0 / math.sqrt(self.node_dim)
        self.scale_equi = 1.0 / math.sqrt(self.node_num_irreps)

        self.equi_filter = EquiFilter(self.node_dim, self.node_irreps, num_basis, activation) 
        # multi-head self-attention
        ## invariant feature 
        self.query_feat = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.key_feat = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.value_feat = nn.Linear(self.node_dim, self.node_dim, bias=False)
        ## equivariant feature
        self.query_sph = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.key_sph = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.value_sph = nn.Linear(self.node_dim, self.node_num_irreps, bias=False)

        self.rsh_conv = o3.ElementwiseTensorProduct(self.node_irreps, f"{self.node_num_irreps}x0e")

        repeat_scheme = [ir.mul for ir in self.node_irreps]
        self.register_buffer('repeat_scheme', torch.tensor(repeat_scheme, dtype=torch.int32))
        # normalization
        self.norm = nn.LayerNorm(self.node_dim) if layer_norm else nn.Identity()
        self.o3norm = EquivariantLayerNorm(self.node_irreps) if layer_norm else nn.Identity()
    
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node_scalar = self.norm(data[keys.NODE_INVARIANT])
        node_equi = self.o3norm(data[keys.NODE_EQUIVARIANT])
        rbf = data[keys.RADIAL_BASIS_FUNCTION]
        fcut = data[keys.ENVELOPE_FUNCTION]
        rsh = data[keys.SPHERICAL_HARMONICS]
        edge_index = data[keys.EDGE_INDEX]
        center_idx = edge_index[keys.CENTER_IDX]
        neighbor_idx = edge_index[keys.NEIGHBOR_IDX]

        # multi-head self-attention
        q_inv = self.query_feat(node_scalar)
        k_inv = self.key_feat(node_scalar)
        v_inv = self.value_feat(node_scalar)

        q_sph = self.query_sph(node_scalar)
        k_sph = self.key_sph(node_scalar)
        v_sph = self.value_sph(node_scalar)

        w_ij = self.equi_filter(node_scalar, rbf, center_idx, neighbor_idx) 
        w_ij = w_ij * fcut 

        query_0 = torch.index_select(q_inv, 0, center_idx)
        query_scalar = query_0 * w_ij
        query_scalar = query_scalar.view(center_idx.shape[0], self.num_heads, self.attn_dim_scalar)
        key_scalar = torch.index_select(k_inv, 0, neighbor_idx).view(neighbor_idx.shape[0], self.num_heads, self.attn_dim_scalar)
        value_scalar = torch.index_select(v_inv, 0, neighbor_idx).view(neighbor_idx.shape[0], self.num_heads, self.attn_dim_scalar)

        # attn_prim_0 = torch.einsum("bhk, bhk -> bh", query_scalar, key_scalar)
        attn_prim_0:torch.Tensor = (query_scalar * key_scalar).sum(-1)
        attn_scalar = attn_prim_0.unsqueeze(-1) * self.scale_scalar
        # attn_scalar = torch.bmm(query_scalar, key_scalar).view(edge_index.shape[1], self.num_heads, 1)
        attn_msg_scalar = attn_scalar * value_scalar
        attn_msg_scalar = attn_msg_scalar.view(edge_index.shape[1], self.node_dim) 

        query_1 = torch.index_select(q_sph, 0, center_idx) 
        query_equi = query_1 * w_ij 
        query_equi = query_equi.view(center_idx.shape[0], self.l_max+1, self.attn_dim_equi)
        key_equi = torch.index_select(k_sph, 0, neighbor_idx).view(neighbor_idx.shape[0], self.l_max+1, self.attn_dim_equi)
        value_equi = torch.index_select(v_sph, 0, neighbor_idx)

        # attn_prim_1 = torch.einsum("bhk, bhk -> bh", query_spherical, key_spherical) 
        attn_prim_1 = (query_equi * key_equi).sum(-1)
        attn_equi:torch.Tensor = attn_prim_1 * self.scale_equi 
        attn_equi = attn_equi.repeat_interleave(self.repeat_scheme, dim=1)
        attn_gate = attn_equi * value_equi
        attn_msg_equi = self.rsh_conv(rsh, attn_gate) * fcut 

        ori_scalar = data[keys.NODE_INVARIANT]
        ori_equi = data[keys.NODE_EQUIVARIANT]
        data[keys.NODE_INVARIANT] = ori_scalar.index_add(0, center_idx, attn_msg_scalar)
        data[keys.NODE_EQUIVARIANT] = ori_equi.index_add(0, center_idx, attn_msg_equi)

        return data