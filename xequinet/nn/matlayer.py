"""
Scripts containing E(3) NN blocks for qc matrice output from QHNet
Original from https://github.com/divelab/AIRS/OpenDFT/QHBench
"""

from typing import List, Union, Tuple, Iterable
import torch 
import torch.nn as nn 
from e3nn import o3 

from .tensorproduct import prod, get_feasible_tp
from .o3layer import NormGate, EquivariantDot 
from .o3layer import resolve_actfn 


class SelfLayer(nn.Module):
    """
    Block for generate diagnol/onsite hidden irreps.
    """
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        actfn: str = "silu",
    ):
        """
        Args:
            `irreps_in`: input node-wise irreps, i.e. 64x0e+64x1o+64x2e+64x3o+64x4e.
            `irreps_hidden`: output irreps for node-wise representation, i.e. 64x0e+64x1o+64x1e+64x2o+64x2e+64x3o+64x3e+64x4e.
            `actfn`: Activation function.
        """
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_tp_out, instructions = get_feasible_tp(
            irreps_in, irreps_in, irreps_hidden, "uuu",
        )
        self.tp_node = o3.TensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_tp_out,
            instructions,
            shared_weights=True, 
            internal_weights=True,
        )
        self.linear_node_l = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_in, biases=True)
        self.linear_node_r = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_in, biases=True)
        self.linear_node_p = o3.Linear(irreps_in=self.irreps_tp_out, irreps_out=self.irreps_hidden, biases=False)
        self.normgate_l = NormGate(self.irreps_in, actfn)
        self.normgate_r = NormGate(self.irreps_in, actfn)
        self.normgate_p = NormGate(self.irreps_tp_out, actfn)
    
    def forward(self, x: torch.Tensor, old_fii: torch.Tensor = None) -> torch.Tensor:
        xl = self.normgate_l(x)
        xl = self.linear_node_l(xl)
        xr = self.normgate_r(x)
        xr = self.linear_node_r(xr) 
        xtp = self.tp_node(xl, xr)
        xtp = self.normgate_p(xtp)
        fii = self.linear_node_p(xtp)
        if old_fii is not None:
            fii = fii + old_fii 
        return fii

    @property
    def device(self):
        return next(self.parameters()).device


class PairLayer(nn.Module):
    '''
    Block for generate off-diagnol/offsite hidden irreps.
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

        self.linear_node_pre = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_in, biases=True)
        self.inner_dot = EquivariantDot(self.irreps_in) 
        self.irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_in, self.irreps_in, self.irreps_hidden, tp_mode="uuu",
        )
        self.tp_node_pair = o3.TensorProduct(
            self.irreps_in,
            self.irreps_in,
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
        self.linear_node_post = o3.Linear(irreps_in=self.irreps_tp_out, irreps_out=self.irreps_hidden, biases=False)
        self.normgate_pre = NormGate(irreps_in, actfn) 
        self.normgate_post = NormGate(self.irreps_tp_out, actfn)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.LongTensor, old_fij: torch.Tensor = None) -> torch.Tensor:
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
        x_prime = self.normgate_pre(self.linear_node_pre(x))
        fij = self.tp_node_pair(x_prime[edge_index[0], :], x_prime[edge_index[1], :], tp_weights) 
        fij = self.normgate_post(fij)
        fij = self.linear_node_post(fij)
        if old_fij is not None:
            fij = fij + old_fij 
        return fij 
    
    @property
    def device(self):
        return next(self.parameters()).device


class Expansion(nn.Module):
    """
    Tensor Product expansion, inversion of irreps1 \otimes irreps2 --> irreps3.
    """
    def __init__(
        self, 
        irreps_block: o3.Irreps, 
        irreps_out: o3.Irreps, 
        node_dim: int, 
        actfn: str = "silu", 
        pair_out: bool = False
    ):
        super().__init__() 
        self.irreps_block = irreps_block
        self.irreps_out = irreps_out
        self.pair_out = pair_out
        self.actfn = resolve_actfn(actfn)
        
        self.instructions = self.get_expansion_path(irreps_block, irreps_out, irreps_out)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
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

    def get_expansion_path(self, irrep_in: o3.Irreps, irrep_out_1: o3.Irreps, irrep_out_2: o3.Irreps) -> List:
        instructions = []
        for  i, (num_in, ir_in) in enumerate(irrep_in):
            for  j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions 
    
    def forward(self, feat: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        x_in = feat 
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
            w3j_matrix = o3.wigner_3j(mul_ir_out1.ir.l, mul_ir_out2.ir.l, mul_ir_in.ir.l, device=self.device, dtype=x1.dtype) 
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
        return f'{self.irreps_block} -> {self.irreps_out} x {self.irreps_out} and bias {self.num_bias} ' \
               f'with parameters {self.num_path_weight}'


class MatriceOut(nn.Module):
    def __init__(
        self, 
        irreps_out: Union[str, o3.Irreps, Iterable] = "3x0e + 2x1o + 1x2e",
        node_dim: int = 128,
        hidden_dim: int = 64,
        block_dim: int = 32, 
        max_l:int = 4,
        actfn:str = "silu",
    ):
        super().__init__()
        irreps_hidden_base = [(hidden_dim, (0, 1))] 
        irreps_block = [(block_dim, (0, 1))] 
        for l in range(2, 2*max_l + 1):
            irreps_hidden_base.append((hidden_dim, (l//2, (-1)**l)))
            irreps_block.append((block_dim, (l//2, (-1)**l)))
        self.irreps_hidden_base = o3.Irreps(irreps_hidden_base)
        self.irreps_block = o3.Irreps(irreps_block) 
        self.irreps_out = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        self.linear_out_ii = o3.Linear(self.irreps_hidden_base, self.irreps_block, biases=False) 
        self.linear_out_ij = o3.Linear(self.irreps_hidden_base, self.irreps_block, biases=False)
        self.expand_ii = Expansion(self.irreps_block, self.irreps_out, node_dim, actfn, pair_out=False) 
        self.expand_ij = Expansion(self.irreps_block, self.irreps_out, node_dim, actfn, pair_out=True) 
    
    def forward(
        self, 
        fii: torch.Tensor, 
        fij: torch.Tensor, 
        node_embed: torch.Tensor, 
        edge_index: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, src_indices = torch.sort(edge_index[1], stable=True)
        node_embed_pair = torch.cat([node_embed[edge_index[0],:], node_embed[edge_index[1],:]], dim=-1)
        fii = self.linear_out_ii(fii)
        fij = self.linear_out_ij(fij)
        diagnol_block = self.expand_ii(fii, node_embed) 
        off_diagnol_block = self.expand_ij(fij, node_embed_pair)
        # symmertrize 
        diagnol_block = 0.5 * (diagnol_block + diagnol_block.transpose(-1, -2))
        off_diagnol_block_trans = torch.index_select(off_diagnol_block.transpose(-1, -2), dim=0, index=src_indices)
        off_diagnol_block = 0.5 * (off_diagnol_block + off_diagnol_block_trans)
        return diagnol_block, off_diagnol_block
