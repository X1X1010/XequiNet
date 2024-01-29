from typing import Iterable, Union

import torch
import torch.nn as nn

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from ..utils import get_embedding_tensor


class Invariant(nn.Module):
    """
    Invariant layer.
    """
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
        squared: bool = False,
        eps: float = 1e-6,
    ):
        """
        Args:
            `irreps_in`: Input irreps.
            `squared`: Whether to square the norm.
            `eps`: Epsilon for numerical stability.
        """
        super().__init__()
        self.squared = squared
        self.eps = eps
        self.invariant = o3.Norm(irreps_in, squared=squared)

    def forward(self, x):
        if self.squared:
            x = self.invariant(x)
        else:
            x = self.invariant(x + self.eps ** 2) - self.eps
        return x


class Gate(nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
    ):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in).simplify()
        self.invariant = Invariant(irreps_in)
        self.activation = nn.Sigmoid()
        self.scalar_mul = o3.ElementwiseTensorProduct(irreps_in, f"{irreps_in.num_irreps}x0e")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_invariant = self.invariant(x)
        x_activation = self.activation(x_invariant)
        x_out = self.scalar_mul(x, x_activation)
        return x_out


# from QHNet
class NormGate(torch.nn.Module):
    """
    NormGate activation module.
    """
    def __init__(self, irreps:o3.Irreps, actfn:str="silu"):
        super().__init__()
        self.irreps = irreps
        self.norm = o3.Norm(self.irreps)
        self.activation = resolve_actfn(actfn)

        num_mul, num_mul_wo_0 = 0, 0
        for mul, ir in self.irreps:
            num_mul += mul
            if ir.l != 0:
                num_mul_wo_0 += mul

        self.mul = o3.ElementwiseTensorProduct(
            self.irreps[1:], o3.Irreps(f"{num_mul_wo_0}x0e"))
        self.fc = nn.Sequential(
            nn.Linear(num_mul, num_mul),
            # nn.SiLU(),
            self.activation,
            nn.Linear(num_mul, num_mul)
        )
        self.num_mul = num_mul
        self.num_mul_wo_0 = num_mul_wo_0

    def forward(self, x):
        norm_x = self.norm(x)[:, self.irreps.slices()[0].stop:]
        f0 = torch.cat([x[:, self.irreps.slices()[0]], norm_x], dim=-1)
        gates = self.fc(f0)
        gated = self.mul(x[:, self.irreps.slices()[0].stop:], gates[:, self.irreps.slices()[0].stop:])
        x = torch.cat([gates[:, self.irreps.slices()[0]], gated], dim=-1)
        return x


class Int2c1eEmbedding(nn.Module):
    def __init__(
        self,
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux28",
    ):
        """
        Args:
            `embed_basis`: Type of the embedding basis.
            `aux_basis`: Type of the auxiliary basis.
        """
        super().__init__()
        embed_ten = get_embedding_tensor(embed_basis, aux_basis)
        self.register_buffer("embed_ten", embed_ten)
        self.embed_dim = embed_ten.shape[1]
    
    def forward(self, at_no):
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
        irreps_in: Union[str, o3.Irreps, Iterable],
    ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [(i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)]

        self.tp = o3.TensorProduct(irreps_in, irreps_in, irreps_out, instr, irrep_normalization="component")

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.input_dim = self.irreps_in.dim

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in})"
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        assert features1.shape[-1] == features2.shape[-1] == self.input_dim, \
            "Input tensor must have the same last dimension as the irreps"
        out = self.tp(features1, features2)
        return out


class EquivariantLayerNorm(nn.Module):
    def __init__(self, irreps, eps=1e-5, affine=True):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.dim = self.irreps.dim
        self.eps = eps

        self.num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        self.num_features = self.irreps.num_irreps
        scalar_index = []
        ix = 0
        for mul, ir in self.irreps:
            if ir.l == 0 and ir.p == 1:
                scalar_index.extend(list(range(ix, ix + mul)))
            ix += ir.dim * mul
        self.register_buffer('scalar_index', torch.LongTensor(scalar_index))

        self.invariant = Invariant(self.irreps)
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


def resolve_actfn(actfn: str) -> nn.Module:
    """Helper function to return activation function"""
    actfn = actfn.lower()
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
    irreps: Union[str, o3.Irreps, Iterable],
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
