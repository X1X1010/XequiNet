from typing import Iterable

import torch
import torch.nn as nn

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from ..utils import get_embedding_tensor


def resolve_actfn(actfn: str, devide_x: bool = False) -> nn.Module:
    """Helper function to return activation function"""
    actfn = actfn.lower()
    actfn_div_x = {"silu": "sigmoid", "relu": "identity", "leakyrelu": "identity"}
    if devide_x and actfn in actfn_div_x:
        actfn = actfn_div_x[actfn]
    if actfn == "relu":
        return nn.ReLU()
    elif actfn == "leakyrelu":
        return nn.LeakyReLU()
    elif actfn == "softplus":
        return nn.Softplus()
    elif actfn == "sigmoid":
        return nn.Sigmoid()
    elif actfn == "silu":
        return nn.SiLU()
    elif actfn == "tanh":
        return nn.Tanh()
    elif actfn == "identity":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported activation function {actfn}")


class Gate(nn.Module):
    def __init__(
        self,
        irreps_in: Iterable,
        actfn: str = "silu",
        refine: bool = False,
    ) -> None:
        super().__init__()
        irreps_in = o3.Irreps(irreps_in).simplify()
        self.invariant = o3.Norm(irreps_in)
        if refine:
            self.activation = nn.Sequential(
                nn.Linear(irreps_in.num_irreps, irreps_in.num_irreps),
                resolve_actfn(actfn, devide_x=True),
                nn.Linear(irreps_in.num_irreps, irreps_in.num_irreps),
            )
            nn.init.zeros_(self.activation[0].bias)
            nn.init.zeros_(self.activation[2].bias)
        else:
            self.activation = resolve_actfn(actfn, devide_x=True)
        self.scalar_mul = o3.ElementwiseTensorProduct(irreps_in, f"{irreps_in.num_irreps}x0e")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_invariant = self.invariant(x)
        x_activation = self.activation(x_invariant)
        x_out = self.scalar_mul(x, x_activation)
        return x_out


class Int2c1eEmbedding(nn.Module):
    def __init__(
        self,
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux28",
    ) -> None:
        """
        Args:
            `embed_basis`: Type of the embedding basis.
            `aux_basis`: Type of the auxiliary basis.
        """
        super().__init__()
        embed_ten = get_embedding_tensor(embed_basis, aux_basis)
        self.register_buffer("embed_ten", embed_ten)
        self.embed_dim = embed_ten.shape[1]
    
    def forward(self, at_no: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `at_no`: Atomic numbers.
        Returns:
            Atomic features.
        """
        return self.embed_ten[at_no]


@compile_mode("trace")
class EquivariantDot(nn.Module):
    def __init__(
        self,
        irreps_in: Iterable,
    ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [(i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)]

        self.tp = o3.TensorProduct(irreps_in, irreps_in, irreps_out, instr, irrep_normalization="component")

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.input_dim = self.irreps_in.dim

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in})"
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        assert features1.shape[-1] == features2.shape[-1] == self.input_dim, \
            "Input tensor must have the same last dimension as the irreps"
        out = self.tp(features1, features2)
        return out


class EquivariantLayerNorm(nn.Module):
    def __init__(self, irreps, affine=True) -> None:
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
        self.register_buffer('scalar_index', torch.LongTensor(scalar_index))

        self.invariant = o3.Norm(self.irreps)
        self.scalar_mul = o3.ElementwiseTensorProduct(self.irreps, f"{self.num_features}x0e")

        weight = torch.ones(self.num_features)
        bias = torch.zeros(self.num_scalar)
        if affine:
            self.affine_weight = nn.Parameter(weight)
            self.affine_bias = nn.Parameter(bias)
        else:
            self.register_buffer('affine_weight', weight)
            self.register_buffer('affine_bias', bias)


    def forward(self, node_input: torch.Tensor) -> torch.Tensor:
        assert node_input.shape[-1] == self.dim, "Input tensor must have the same last dimension as the irreps"

        scalar_input = node_input[:, self.scalar_index]

        node_input = node_input.index_add(
            dim=1,
            index=self.scalar_index,
            source=-scalar_input.mean(dim=1, keepdim=True).repeat(1, self.num_scalar)
        )

        input_norm = self.invariant(node_input)
        input_1_rms = torch.reciprocal(torch.sqrt(torch.square(input_norm).mean(dim=1, keepdim=True)))
        node_input = node_input * input_1_rms
        
        node_input = self.scalar_mul(node_input, self.affine_weight.unsqueeze(0))
        node_input = node_input.index_add(
            dim=1,
            index=self.scalar_index,
            source=self.affine_bias.unsqueeze(0).repeat(node_input.shape[0], 1)
        )

        return node_input


def resolve_norm(
    norm_type: str,
    num_features: int,
    affine: bool = True,
) -> nn.Module:
    """Helper function to return normalization layer"""
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm1d(
            num_features,
            affine=affine,
        )
    elif norm_type == "layer":
        return nn.LayerNorm(
            num_features,
            elementwise_affine=affine,
        )
    elif norm_type == "nonorm":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported normalization layer {norm_type}")


def resolve_o3norm(
    norm_type: str,
    irreps: Iterable,
    affine: bool = True,
) -> nn.Module:
    """Helper function to return equivariant normalization layer"""
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return e3nn.nn.BatchNorm(
            irreps,
            affine=affine,
        )
    elif norm_type == "layer":
        return EquivariantLayerNorm(
            irreps,
            affine=affine,
        )
    elif norm_type == "nonorm":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported equivariant normalization layer {norm_type}")
