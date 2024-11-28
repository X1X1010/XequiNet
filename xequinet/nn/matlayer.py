"""
Scripts containing E(3) NN blocks for qc matrice output from QHNet
Original from https://github.com/divelab/AIRS/OpenDFT/QHBench
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from e3nn import o3

from .basic import resolve_activation
from .o3layer import EquivariantDot, Gate
from .tp import get_feasible_tp, prod


class SelfLayer(nn.Module):
    """
    Block for generate diagnol/onsite hidden irreps.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        activation: str = "silu",
    ):
        """
        Args:
            `irreps_in`: Input node-wise irreps.
            `irreps_hidden`: Output irreps for node-wise representation.
            `activation`: Activation function type.
        """
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_tp_out, instruct = get_feasible_tp(
            irreps_in,
            irreps_in,
            irreps_hidden,
            "uuu",
        )
        self.self_tp = o3.TensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_tp_out,
            instruct,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear_l = o3.Linear(self.irreps_in, self.irreps_in, biases=True)
        self.linear_r = o3.Linear(self.irreps_in, self.irreps_in, biases=True)
        self.linear_p = o3.Linear(self.irreps_tp_out, self.irreps_hidden, biases=False)
        self.gate_l = Gate(self.irreps_in, activation=activation, refine=True)
        self.gate_r = Gate(self.irreps_in, activation=activation, refine=True)
        self.gate_p = Gate(self.irreps_tp_out, activation=activation, refine=True)

    def forward(self, x: torch.Tensor, fii_in: Optional[torch.Tensor]) -> torch.Tensor:
        xl = self.linear_l(self.gate_l(x))
        xr = self.linear_r(self.gate_r(x))
        xtp = self.gate_p(self.self_tp(xl, xr))
        fii = self.linear_p(xtp)
        if fii_in is not None:
            fii += fii_in
        return fii


class PairLayer(nn.Module):
    """
    Block for generate off-diagnol/offsite hidden irreps.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        edge_attr_dim: int = 20,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.edge_attr_dim = edge_attr_dim
        self.activation = resolve_activation(activation)

        self.linear_node_pre = o3.Linear(self.irreps_in, self.irreps_in, biases=True)
        self.inner_dot = EquivariantDot(self.irreps_in)
        self.irreps_tp_out, instruct = get_feasible_tp(
            self.irreps_in,
            self.irreps_in,
            self.irreps_hidden,
            "uuu",
        )
        self.pair_tp = o3.TensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_tp_out,
            instruct,
            internal_weights=False,
            shared_weights=False,
        )
        self.mlp_edge_scalar = nn.Sequential(
            nn.Linear(self.irreps_in[0].mul + self.irreps_in.num_irreps, 128),
            self.activation,
            nn.Linear(128, self.pair_tp.weight_numel),
        )
        self.mlp_edge_rbf = nn.Sequential(
            nn.Linear(self.edge_attr_dim, 128),
            self.activation,
            nn.Linear(128, self.pair_tp.weight_numel),
        )
        self.linear_node_post = o3.Linear(
            self.irreps_tp_out, self.irreps_hidden, biases=False
        )
        self.gate_pre = Gate(self.irreps_in, activation=activation, refine=True)
        self.gate_post = Gate(self.irreps_tp_out, activation=activation, refine=True)
        # the last index of the scalar part
        self.num_scalar = self.irreps_in[0].mul

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        fij_in: Optional[torch.Tensor],
    ) -> torch.Tensor:
        node_feat = x
        s0 = self.inner_dot(node_feat[edge_index[0], :], node_feat[edge_index[1], :])[
            :, self.num_scalar :
        ]
        s0 = torch.cat(
            [
                x[edge_index[0]][:, : self.num_scalar],
                x[edge_index[1]][:, : self.num_scalar],
                s0,
            ],
            dim=-1,
        )
        tp_weight = self.mlp_edge_scalar(s0) * self.mlp_edge_rbf(edge_attr)
        x_prime = self.gate_pre(self.linear_node_pre(x))
        fij = self.pair_tp(x_prime[edge_index[0]], x_prime[edge_index[1]], tp_weight)
        fij = self.linear_node_post(self.gate_post(fij))
        if fij_in is not None:
            fij += fij_in
        return fij


class Expansion(nn.Module):
    """
    Tensor Product expansion, inversion of irreps1 \otimes irreps2 --> irreps3.
    """

    def __init__(
        self,
        irreps_block: o3.Irreps,
        irreps_out: o3.Irreps,
        node_dim: int,
        activation: str = "silu",
        pair_out: bool = False,
    ) -> None:
        super().__init__()
        self.irreps_block = irreps_block
        self.irreps_out = irreps_out
        self.activation = resolve_activation(activation)

        self.instructions, self.w3j_matrices = self.get_expansion_path(
            irreps_block, irreps_out, irreps_out
        )
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum(
            [prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0]
        )
        self.num_param = self.num_path_weight + self.num_bias

        node_dim_in = 2 * node_dim if pair_out else node_dim
        self.lin_weight = nn.Sequential(
            nn.Linear(node_dim_in, 64),
            self.activation,
            nn.Linear(64, self.num_path_weight),
        )
        self.lin_bias = nn.Sequential(
            nn.Linear(node_dim_in, 64),
            self.activation,
            nn.Linear(64, self.num_bias),
        )
        # get the slice and shape for each irrep
        blk_slices, blk_shapes = [], []
        for slc, mul_ir in zip(self.irreps_block.slices(), self.irreps_block):
            blk_slices.append([slc.start, slc.stop])
            blk_shapes.append([mul_ir.mul, mul_ir.ir.dim])
        self.register_buffer("blk_slices", torch.tensor(blk_slices))
        self.register_buffer("blk_shapes", torch.tensor(blk_shapes))

    def get_expansion_path(
        self, irreps_in: o3.Irreps, irreps_out1: o3.Irreps, irreps_out2: o3.Irreps
    ) -> Tuple[List]:
        instructions = []
        w3j_matrices = nn.ParameterList()
        for i, (mul_in, ir_in) in enumerate(irreps_in):
            for j, (mul_out1, ir_out1) in enumerate(irreps_out1):
                for k, (mul_out2, ir_out2) in enumerate(irreps_out2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append(
                            (i, j, k, True, 1.0, (mul_in, mul_out1, mul_out2))
                        )
                        w3j_ijk = nn.Parameter(
                            o3.wigner_3j(ir_out1.l, ir_out2.l, ir_in.l),
                            requires_grad=False,
                        )
                        w3j_matrices.append(w3j_ijk)
        return instructions, w3j_matrices

    def __repr__(self) -> str:
        return f"{self.irreps_block} -> {self.irreps_out} x {self.irreps_out} | {self.num_bias} bias, {self.num_path_weight} weight"

    def forward(self, x_in: torch.Tensor, node_embed: torch.Tensor) -> torch.Tensor:
        # split input spherical tensor into blocks
        x_in_s = [
            x_in[:, slc[0] : slc[1]].view(-1, shp[0], shp[1])
            for slc, shp in zip(self.blk_slices, self.blk_shapes)
        ]
        weight = self.lin_weight(node_embed)
        bias = self.lin_bias(node_embed)

        outputs = {}
        flat_weight_index = 0
        bias_index = 0
        for ins, w3j in zip(self.instructions, self.w3j_matrices):
            mul_ir_in = self.irreps_block[ins[0]]
            mul_ir_out1 = self.irreps_out[ins[1]]
            mul_ir_out2 = self.irreps_out[ins[2]]
            x1 = x_in_s[ins[0]]
            if ins[3] is True:
                w = weight[
                    :, flat_weight_index : flat_weight_index + prod(ins[-1])
                ].view(-1, *ins[-1])
                result = torch.einsum("bwuv, bwk -> buvk", w, x1)
                if ins[0] == 0:
                    b = bias[:, bias_index : bias_index + prod(ins[-1][1:])].view(
                        -1, *ins[-1][1:]
                    )
                    bias_index += prod(ins[-1][1:])
                    result += b.unsqueeze(-1)
                result = torch.einsum("ijk, buvk -> buivj", w3j, result) / mul_ir_in[0]
            flat_weight_index += prod(ins[-1])
            # mul_ir_out.dim = mul_ir_out.mul * (2 * mul_ir_out.ir.dim + 1)
            out_dim1 = mul_ir_out1[0] * (2 * mul_ir_out1[1][0] + 1)  # mul * (2l + 1)
            out_dim2 = mul_ir_out2[0] * (2 * mul_ir_out2[1][0] + 1)  # mul * (2l + 1)
            result = result.reshape(-1, out_dim1, out_dim2)
            key = (ins[1], ins[2])
            if key in outputs:
                outputs[key] += result
            else:
                outputs[key] = result

        rows = []
        for i in range(len(self.irreps_out)):
            blocks = []
            for j in range(len(self.irreps_out)):
                if (i, j) not in outputs.keys():
                    # mul_ir_out.dim = mul_ir_out.mul * (2 * mul_ir_out.ir.dim + 1)
                    out_dimi = self.irreps_out[i][0] * (
                        2 * self.irreps_out[i][1][0] + 1
                    )
                    out_dimj = self.irreps_out[j][0] * (
                        2 * self.irreps_out[j][1][0] + 1
                    )
                    blocks.append(
                        torch.zeros(
                            (x_in.shape[0], out_dimi, out_dimj),
                            device=x_in.device,
                            dtype=x_in.dtype,
                        )
                    )
                else:
                    blocks.append(outputs[(i, j)])
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output
