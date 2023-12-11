"""
This Python file holds E3 NN blocks for qc matrice output. 
"""

from typing import List, Union, Tuple, Iterable 

import torch 
import torch.nn as nn 
from e3nn import o3 
from torch_scatter import scatter 

from .rbf import resolve_rbf, resolve_cutoff
from .o3layer import resolve_actfn, resolve_o3norm 
from .tensorproduct import prod, get_feasible_tp 
from .o3layer import EquivariantDot, EquivariantLayerNorm, Invariant, Int2c1eEmbedding, NormGate


class XMatEmbedding(nn.Module):
    """
    Embedding module for qc matrice learning. 
    """
    def __init__(
        self, 
        node_dim:int=128, 
        edge_irreps:Union[str, o3.Irreps, Iterable]="128x0e+128x1o+128x2e+128x3o+128x4e",
        embed_basis:str="gfn2-xtb",
        aux_basis:str="aux56",
        num_basis:int=20,
        rbf_kernel:str="bessel",
        cutoff:float=8.0,
        cutoff_fn:str="exponential",
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
        at_no:torch.LongTensor,
        pos:torch.Tensor, 
        edge_index:torch.Tensor, 
        edge_index_full:torch.Tensor
    ):
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
        irreps_node_out:Union[str, o3.Irreps, Iterable],
        node_dim: int = 128,
        edge_attr_dim: int = 20, 
        actfn: str = "silu",
    ):
        super().__init__() 
        self.irreps_node_in = irreps_node_in if isinstance(irreps_node_in, o3.Irreps) else o3.Irreps(irreps_node_in)
        self.irreps_node_out = irreps_node_out if isinstance(irreps_node_out, o3.Irreps) else o3.Irreps(irreps_node_out)
        max_l = irreps_node_out.lmax
        self.irreps_rshs = o3.Irreps.spherical_harmonics(max_l)
        self.residual_update = self.irreps_node_in == self.irreps_node_out
        self.actfn = resolve_actfn(actfn)
        self.inner_dot = EquivariantDot(self.irreps_node_in) 
        # tensor product coupler
        self.irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_node_in, self.irreps_rshs, self.irreps_node_out, tp_mode="uvu", trainable=True
        )
        self.tp_node = o3.TensorProduct(
            self.irreps_node_in,
            self.irreps_rshs,
            self.irreps_tp_out,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
            path_normalization="none"
        )

        self.lin_edge_scalar = nn.Sequential(
            nn.Linear(self.irreps_node_in.num_irreps+node_dim, 128, bias=True),
            self.actfn, 
            nn.Linear(128, self.tp_node.weight_numel, bias=True)
        )
        self.lin_edge_rbf = nn.Sequential(
            nn.Linear(edge_attr_dim, 128, bias=True),
            self.actfn, 
            nn.Linear(128, self.tp_node.weight_numel, bias=True) 
        )
        self.lin_node0 = o3.Linear(self.irreps_node_in, self.irreps_node_in, internal_weights=True, shared_weights=True, biases=True)
        self.lin_node1 = o3.Linear(self.irreps_node_in, self.irreps_node_in, internal_weights=True, shared_weights=True, biases=True)
        self.lin_node2 = o3.Linear(self.irreps_tp_out, self.irreps_node_out, internal_weights=True, shared_weights=True, biases=True) 
        self.normgate = NormGate(self.irreps_node_in, actfn)
        self.e3norm = EquivariantLayerNorm(self.irreps_node_out) 
        
    def forward(
        self, 
        node_feat:torch.Tensor,  
        edge_attr:torch.Tensor, 
        edge_rshs:torch.Tensor, 
        edge_index:torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pre_x = self.lin_node0(node_feat)
        s0 = self.inner_dot(pre_x[edge_index[0], :], pre_x[edge_index[1], :])[:, self.irreps_node_in.slices()[0].stop:]
        s0 = torch.cat(
            [
                pre_x[edge_index[0]][:, self.irreps_node_in.slices()[0]], 
                pre_x[edge_index[1]][:, self.irreps_node_in.slices()[0]], 
                s0
            ], dim=-1
        )
        x = self.normgate(node_feat)
        x = self.lin_node1(x) 
        x_j = x[edge_index[1], :] 
        tp_weights = self.lin_edge_scalar(s0) * self.lin_edge_rbf(edge_attr)
        msg_j = self.tp_node(x_j, edge_rshs, tp_weights) 
        accu_msg = scatter(msg_j, edge_index[0], dim=0, dim_size=node_feat.size(0))
        out = self.e3norm(self.lin_node2(accu_msg))
        if self.residual_update:
            out = node_feat + out
        return out


class SelfLayer(nn.Module):
    """
    """
    def __init__(
        self,
        irreps_in:o3.Irreps,
        irreps_hidden:o3.Irreps,
        actfn:str = "silu",
        residual_update:bool = True,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.residual_update = residual_update
        self.irreps_tp_out, instructions = get_feasible_tp(
            irreps_hidden, irreps_hidden, irreps_hidden, "uuu", True
        )
        self.tp_node = o3.TensorProduct(
            self.irreps_hidden,
            self.irreps_hidden,
            self.irreps_tp_out,
            instructions,
            shared_weights=True, 
            internal_weights=True,
        )
        self.linear_node_l = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_hidden,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.linear_node_r = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_hidden,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.linear_node_p = o3.Linear(
            irreps_in=self.irreps_tp_out,
            irreps_out=self.irreps_hidden,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.normgate_l = NormGate(self.irreps_in, actfn)
        self.normgate_r = NormGate(self.irreps_in, actfn)
        self.normgate_p = NormGate(self.irreps_tp_out, actfn)
        # self.e3norm = EquivariantLayerNorm(irreps_hidden)
    
    def forward(self, x:torch.Tensor, old_fii:torch.Tensor=None) -> torch.Tensor:
        old_x = x 
        xl = self.normgate_l(x)
        xl = self.linear_node_l(xl)
        xr = self.normgate_r(x)
        xr = self.linear_node_r(xr) 
        x = self.tp_node(xl, xr)
        if self.residual_update:
            x = x + old_x
        x = self.normgate_p(x)
        fii = self.linear_node_p(x)
        # fii = self.e3norm(fii) 
        if old_fii is not None:
            fii = fii + old_fii 
        return fii

    @property
    def device(self):
        return next(self.parameters()).device


class PairLayer(nn.Module):
    '''
    '''
    def __init__(
        self, 
        irreps_in:o3.Irreps,
        irreps_hidden:o3.Irreps,
        edge_dim:int = 20,
        actfn:str = "silu",
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self._edge_dim = edge_dim 
        self.actfn = resolve_actfn(actfn)

        self.linear_node_pre = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_hidden,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.inner_dot = EquivariantDot(self.irreps_in) 
        # self.invariant = Invariant(self.irreps_node, squared=True)
        self.irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_hidden, self.irreps_hidden, self.irreps_hidden, tp_mode="uuu", trainable=True
        )
        self.tp_node_pair = o3.TensorProduct(
            self.irreps_hidden,
            self.irreps_hidden,
            self.irreps_tp_out,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.linear_edge_scalar = nn.Sequential(
            nn.Linear(self.irreps_in[0][0] + self.irreps_in.num_irreps, 128, bias=True),
            self.actfn,
            nn.Linear(128, self.tp_node_pair.weight_numel, bias=True)
        )
        self.linear_edge_rbf = nn.Sequential(
            nn.Linear(self._edge_dim, 128, bias=True),
            self.actfn,
            nn.Linear(128, self.tp_node_pair.weight_numel, bias=True)
        )
        self.linear_node_post = o3.Linear(
            irreps_in=self.irreps_tp_out,
            irreps_out=self.irreps_hidden,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.normgate_pre = NormGate(irreps_in, actfn) 
        self.normgate_post = NormGate(self.irreps_tp_out, actfn)
        # self.e3norm = EquivariantLayerNorm(irreps_hidden)

    def forward(self, x:torch.Tensor, edge_attr:torch.Tensor, edge_index:torch.LongTensor, old_fij:torch.Tensor=None) -> torch.Tensor:
        node_feat = x 
        s0 = self.inner_dot(node_feat[edge_index[0], :], node_feat[edge_index[1], :])[:, self.irreps_in.slices()[0].stop:]
        s0 = torch.cat(
            [
                x[edge_index[0]][:, self.irreps_in.slices()[0]], 
                x[edge_index[1]][:, self.irreps_in.slices()[0]],
                s0, 
            ], dim=-1
        ) 
        tp_weights = self.linear_edge_scalar(s0) * self.linear_edge_rbf(edge_attr) 
        x_prime = self.normgate_pre(x)
        x_prime = self.linear_node_pre(x_prime)
        fij = self.tp_node_pair(x_prime[edge_index[0], :], x_prime[edge_index[1], :], tp_weights) 
        fij = self.normgate_post(fij)
        fij = self.linear_node_post(fij)
        # fij = self.e3norm(fij)
        if old_fij is not None:
            fij = fij + old_fij 
        return fij 
    
    @property
    def device(self):
        return next(self.parameters()).device


class MatTrans(nn.Module):
    """
    Module to read out convolution layer output.
    """
    def __init__(
        self,
        irreps_node:Union[str, o3.Irreps, Iterable] = "128x0e+128x1o+128x2e+128x3o+128x4e",
        hidden_dim:int = 64,
        edge_dim:int = 20,
        actfn:str = "silu",
    ):
        super().__init__()
        self.irreps_in = irreps_node if isinstance(irreps_node, o3.Irreps) else o3.Irreps(irreps_node)
        max_l = self.irreps_in.lmax
        self.irreps_hidden = o3.Irreps([(hidden_dim, (l, (-1)**l)) for l in range(max_l+1)]) 
        # self.irreps_hidden = o3.Irreps([(hidden_dim, (l, 1)) for l in range(max_l+1)])
        # self.irreps_rshs = o3.Irreps.spherical_harmonics(max_l)
        # Building block 
        self.node_self_layer = SelfLayer(self.irreps_in, self.irreps_hidden, actfn)
        self.node_pair_layer = PairLayer(self.irreps_in, self.irreps_hidden, edge_dim, actfn)
    
    def forward(
        self,
        node_feat:torch.Tensor,
        edge_attr:torch.Tensor,
        edge_index:torch.LongTensor,
        fii:torch.Tensor,
        fij:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fii = self.node_self_layer(node_feat, fii)
        fij = self.node_pair_layer(node_feat, edge_attr, edge_index, fij)
        return fii, fij


class Expansion(nn.Module):
    def __init__(
        self, 
        irreps_node:o3.Irreps, 
        node_dim:int, 
        irreps_block:o3.Irreps, 
        irreps_out:o3.Irreps, 
        actfn:str="silu", 
        pair_out:bool=False
    ):
        super().__init__() 
        self.irreps_node = irreps_node
        self.irreps_block = irreps_block
        self.irreps_out = irreps_out
        self.pair_out = pair_out
        self.actfn = resolve_actfn(actfn)

        self.linear_out = o3.Linear(irreps_node, irreps_block, biases=True)        
        self.instructions = self.get_expansion_path(irreps_block, irreps_out, irreps_out)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
        # if self.num_path_weight > 0:
        #     self.weights = nn.Parameter(torch.rand(self.num_path_weight + self.num_bias))
        self.num_weights = self.num_path_weight + self.num_bias 
        if pair_out:
            self.lin_weight = nn.Sequential(
                nn.Linear(2*node_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_path_weight, bias=True)
            )
            self.lin_bias = nn.Sequential(
                nn.Linear(2*node_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_bias, bias=True)
            )
        else:
            self.lin_weight = nn.Sequential(
                nn.Linear(node_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_path_weight, bias=True)
            )
            self.lin_bias = nn.Sequential(
                nn.Linear(node_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_bias, bias=True)
            )

    def get_expansion_path(self, irrep_in:o3.Irreps, irrep_out_1:o3.Irreps, irrep_out_2:o3.Irreps):
        instructions = []
        for  i, (num_in, ir_in) in enumerate(irrep_in):
            for  j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions 
    
    def forward(self, feat:torch.Tensor, node_emb:torch.Tensor) -> torch.Tensor:
        x_in = self.linear_out(feat) 
        batch_num = x_in.shape[0] 
        if len(self.irreps_block) == 1:
            x_in_s = [x_in.reshape(batch_num, self.irreps_block[0].mul, self.irreps_block[0].ir.dim)]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
                for i, mul_ir in zip(self.irreps_block.slices(), self.irreps_block)
            ]
        weights = self.lin_weight(node_emb)
        bias_weights = self.lin_bias(node_emb)

        outputs = {}
        flat_weight_index = 0 
        bias_weight_index = 0 
        for ins in self.instructions:
            mul_ir_in = self.irreps_block[ins[0]]
            mul_ir_out1 = self.irreps_out[ins[1]]
            mul_ir_out2 = self.irreps_out[ins[2]] 
            x1 = x_in_s[ins[0]] 
            x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim) 
            w3j_matrix = o3.wigner_3j(ins[1], ins[2], ins[0]).to(self.device).type(x1.type()) 
            if ins[3] is True:
                weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
                if ins[0] == 0:
                    bias_weight = bias_weights[:,bias_weight_index:bias_weight_index + prod(ins[-1][1:])].reshape([-1] + ins[-1][1:])
                    bias_weight_index += prod(ins[-1][1:])
                    result = result + bias_weight.unsqueeze(-1)
                result = torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
            flat_weight_index += prod(ins[-1])
            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            key = (ins[1], ins[2])
            if key in outputs.keys():
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result
        
        rows = []
        for i in range(len(self.irreps_out)):
            blocks = []
            for j in range(len(self.irreps_out)):
                if (i, j) not in outputs.keys():
                    blocks += [
                        torch.zeros(
                            (x_in.shape[0], self.irreps_out[i].dim, self.irreps_out[j].dim),
                            device=x_in.device,
                            dtype=x_in.dtype
                        )
                    ]
                else:
                    blocks += [outputs[(i, j)]]
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return f'{self.irreps_block} -> {self.irreps_out}x{self.irreps_out} and bias {self.num_bias}' \
               f'with parameters {self.num_path_weight}'


class MatriceOut(nn.Module):
    def __init__(
        self, 
        irreps_out:Union[str, o3.Irreps, Iterable] = "3x0e+2x1e+1x2e",
        node_dim:int = 128,
        hidden_dim:int = 64,
        block_dim:int = 32, 
        max_l:int = 4,
        actfn:str = "silu",
    ):
        super().__init__()
        self.irreps_hidden_base = o3.Irreps([(hidden_dim, (l, 1)) for l in range(max_l+1)])
        self.irreps_block = o3.Irreps([(block_dim, (l, 1)) for l in range(max_l+1)])  
        self.irreps_out = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        self.expand_ii = Expansion(self.irreps_hidden_base, node_dim, self.irreps_block, self.irreps_out, actfn, pair_out=False) 
        self.expand_ij = Expansion(self.irreps_hidden_base, node_dim, self.irreps_block, self.irreps_out, actfn, pair_out=True) 
    
    def forward(
        self, 
        fii:torch.Tensor, 
        fij:torch.Tensor, 
        node_embed:torch.Tensor, 
        edge_index:torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
        _, src_indices = torch.sort(edge_index[1], stable=True)
        node_embed_pair = torch.cat([node_embed[edge_index[0],:], node_embed[edge_index[1],:]], dim=-1)
        diagnol_block = self.expand_ii(fii, node_embed) 
        off_diagnol_block = self.expand_ij(fij, node_embed_pair)
        # symmertrize 
        diagnol_block = 0.5 * (diagnol_block + diagnol_block.transpose(-1, -2))
        off_diagnol_block_trans = torch.index_select(off_diagnol_block.transpose(-1, -2), dim=0, index=src_indices)
        off_diagnol_block = 0.5 * (off_diagnol_block + off_diagnol_block_trans)
        return diagnol_block, off_diagnol_block
