from typing import Iterable

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from .basic import resolve_activation


@compile_mode("trace")
class Invariant(nn.Module):
    """Modolus of each irrep in a direct sum of irreps."""

    def __init__(
        self, irreps_in: Iterable, squared: bool = False, eps: float = 1e-5
    ) -> None:
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [
            (i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)
        ]

        self.tp = o3.TensorProduct(
            irreps_in, irreps_in, irreps_out, instr, irrep_normalization="component"
        )

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.squared = squared
        self.eps = eps

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tp(x, x)
        if self.squared:
            return out
        else:
            return torch.sqrt(out + self.eps**2) - self.eps


class Gate(nn.Module):
    def __init__(
        self,
        irreps_in: Iterable,
        activation: str = "silu",
        refine: bool = False,
    ) -> None:
        super().__init__()
        irreps_in = o3.Irreps(irreps_in).simplify()
        self.invariant = Invariant(irreps_in)
        if refine:
            self.activation = nn.Sequential(
                nn.Linear(irreps_in.num_irreps, irreps_in.num_irreps),
                resolve_activation(activation, devide_x=True),
                nn.Linear(irreps_in.num_irreps, irreps_in.num_irreps),
            )
            nn.init.zeros_(self.activation[0].bias)
            nn.init.zeros_(self.activation[2].bias)
        else:
            self.activation = resolve_activation(activation, devide_x=True)
        self.scalar_mul = o3.ElementwiseTensorProduct(
            irreps_in, f"{irreps_in.num_irreps}x0e"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_invariant = self.invariant(x)
        x_activation = self.activation(x_invariant)
        x_out = self.scalar_mul(x, x_activation)
        return x_out


@compile_mode("trace")
class EquivariantDot(nn.Module):
    def __init__(
        self,
        irreps_in: Iterable,
    ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [
            (i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)
        ]

        self.tp = o3.TensorProduct(
            irreps_in, irreps_in, irreps_out, instr, irrep_normalization="component"
        )

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.input_dim = self.irreps_in.dim

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in})"

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        assert (
            features1.shape[-1] == features2.shape[-1] == self.input_dim
        ), "Input tensor must have the same last dimension as the irreps"
        out = self.tp(features1, features2)
        return out


class EquivariantLayerNorm(nn.Module):
    def __init__(self, irreps, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.dim = self.irreps.dim

        self.num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        self.num_features = self.irreps.num_irreps
        scalar_index = []
        ix = 0
        for mul, ir in self.irreps:
            if ir.l == 0 and ir.p == 1:
                scalar_index.extend(list(range(ix, ix + mul)))
            ix += ir.dim * mul
        self.register_buffer("scalar_index", torch.LongTensor(scalar_index))

        self.invariant = Invariant(self.irreps, squared=True)
        self.scalar_mul = o3.ElementwiseTensorProduct(
            self.irreps, f"{self.num_features}x0e"
        )

        weight = torch.ones(self.num_features)
        bias = torch.zeros(self.num_scalar)
        if affine:
            self.affine_weight = nn.Parameter(weight)
            self.affine_bias = nn.Parameter(bias)
        else:
            self.register_buffer("affine_weight", weight)
            self.register_buffer("affine_bias", bias)

        self.eps = eps

    def forward(self, node_input: torch.Tensor) -> torch.Tensor:
        assert (
            node_input.shape[-1] == self.dim
        ), "Input tensor must have the same last dimension as the irreps"

        scalar_input = node_input[:, self.scalar_index]

        node_input = node_input.index_add(
            dim=1,
            index=self.scalar_index,
            source=-scalar_input.mean(dim=1, keepdim=True).repeat(1, self.num_scalar),
        )

        input_square = self.invariant(node_input)
        input_inv_rms = torch.reciprocal(
            torch.sqrt(torch.mean(input_square, dim=1, keepdim=True) + self.eps)
        )
        node_input = node_input * input_inv_rms

        node_input = self.scalar_mul(node_input, self.affine_weight.unsqueeze(0))
        node_input = node_input.index_add(
            dim=1,
            index=self.scalar_index,
            source=self.affine_bias.unsqueeze(0).repeat(node_input.shape[0], 1),
        )

        return node_input
