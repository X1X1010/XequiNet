from typing import Iterable, Union

import math

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3

from .o3layer import Gate, resolve_actfn


class ScalarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 1,
        actfn: str = "silu",
        node_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `out_dim`: Output dimension.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        nn.init.zeros_(self.out_mlp[0].bias)
        nn.init.constant_(self.out_mlp[2].bias, node_bias)
    
    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_index: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates. (Unused in this module)
            `batch_index`: Batch index.
        Returns:
            `res`: Scalar output.
        """
        atom_out = self.out_mlp(x_scalar)
        res = scatter(atom_out, batch_index, dim=0)
        return res


class NegGradOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        node_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.out_mlp[0].bias)
        nn.init.constant_(self.out_mlp[2].bias, node_bias)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_index: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates.
            `batch_index`: Batch index.
        Returns:
            `res`: Scalar output.
            `neg_grad`: Negative gradient.
        """
        atom_out = self.out_mlp(x_scalar)
        res =  scatter(atom_out, batch_index, dim=0)
        grad = torch.autograd.grad(
            [atom_out.sum(),],
            [coord,],
            retain_graph=True,
            create_graph=True,
        )[0]
        return res, -grad


class VectorOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1o",
        output_dim: int = 3,
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `output_dim`: Output dimension. (3 for vector and 1 for norm of the vector)
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps),
            Gate(self.hidden_irreps),
            o3.Linear(self.hidden_irreps, "1x1o"),
        )
        if output_dim != 3 and output_dim != 1:
            raise ValueError(f"output dimension must be either 1 or 3, but got {output_dim}")
        self.output_dim = output_dim

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_index: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
            `coord`: Atomic coordinates.
            `batch_index`: Batch index.
        Returns:
            `res`: Vector output.
        """
        spherical_out = self.spherical_out_mlp(x_spherical)[:, [2, 0, 1]]  # [y, z, x] -> [x, y, z]
        scalar_out = self.scalar_out_mlp(x_scalar)
        atom_out = spherical_out * scalar_out
        res = scatter(atom_out, batch_index, dim=0)
        if self.output_dim == 1:
            res = torch.linalg.norm(res, dim=-1, keepdim=True)
        return res


class PolarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "64x0e + 16x2e",
        output_dim: int = 9,
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `output_dim`: Output dimension. (9 for 3x3 matrix and 1 for trace of the matrix)
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps, biases=True),
            Gate(self.hidden_irreps),
            o3.Linear(self.hidden_irreps, "1x0e + 1x2e", biases=True),
        )
        nn.init.zeros_(self.spherical_out_mlp[0].bias)
        nn.init.zeros_(self.spherical_out_mlp[2].bias)
        self.rsh_conv = o3.ElementwiseTensorProduct("1x0e + 1x2e", "2x0e")
        if output_dim != 9 and output_dim != 1:
            raise ValueError(f"output dimension must be either 1 or 9, but got {output_dim}")
        self.output_dim = output_dim

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_index: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
            `coord`: Atomic coordinates.
            `batch_index`: Batch index.
        Returns:
            `res`: Polarizability.
        """
        spherical_out = self.spherical_out_mlp(x_spherical)
        scalar_out = self.scalar_out_mlp(x_scalar)
        atom_out = self.rsh_conv(spherical_out, scalar_out)
        mol_out = scatter(atom_out, batch_index, dim=0)
        zero_order = mol_out[:, 0:1]
        second_order = mol_out[:, 1:6]
        # build zero order output
        zero_out = torch.diag_embed(torch.repeat_interleave(zero_order, 3, dim=-1))
        # build second order output
        second_out = torch.empty_like(zero_out)
        d_norm = torch.linalg.norm(second_order, dim=-1)
        dxy = second_order[:, 0]; dyz = second_order[:, 1]
        dz2 = second_order[:, 2]
        dzx = second_order[:, 3]; dx2_y2 = second_order[:, 4]
        second_out[:, 0, 0] = (1 / math.sqrt(3)) * (d_norm - dz2) + dx2_y2
        second_out[:, 1, 1] = (1 / math.sqrt(3)) * (d_norm - dz2) - dx2_y2
        second_out[:, 2, 2] = (1 / math.sqrt(3)) * (d_norm + 2 * dz2)
        second_out[:, 0, 1] = second_out[:, 1, 0] = dxy
        second_out[:, 1, 2] = second_out[:, 2, 1] = dyz
        second_out[:, 0, 2] = second_out[:, 2, 0] = dzx
        # add together
        res = zero_out + second_out
        if self.output_dim == 1:
            res = torch.diagonal(res, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3
        return res


class SpatialOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Hidden dimension.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_index: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
            `coord`: Atomic coordinates.
            `batch_index`: Batch index.
        Returns:
            `res`: Spatial output.
        """
        scalar_out = self.scalar_out_mlp(x_scalar)
        spatial = torch.square(coord).sum(dim=1, keepdim=True)
        res = scatter(scalar_out * spatial, batch_index, dim=0)
        return res


class CartTensorOut(nn.Module):
    """
    Output Module for order n tensor via tensor product 1 \otimes 1 \otimes 1 ...
    """
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1o",
        order: int = 2,
        required_symmetry: str = "ij",
        trace_out: bool = False,
        actfn: str = "silu",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.edge_irreps = edge_irreps if isinstance(edge_irreps, o3.Irreps) else o3.Irreps(edge_irreps)
        self.hidden_irreps = hidden_irreps if isinstance(hidden_irreps, o3.Irreps) else o3.Irreps(hidden_irreps)
        self.activation = resolve_actfn(actfn)
        self.order = order
        self.trace_out = trace_out and order == 2

        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.order),
        )
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps, biases=True),
            Gate(self.hidden_irreps),
            o3.Linear(self.hidden_irreps, f"{self.order}x1o", biases=True),
        )
        self.rsh_conv = o3.ElementwiseTensorProduct(f"{self.order}x1o", f"{self.order}x0e")
        self.rtp = o3.ReducedTensorProducts(formula=required_symmetry, **{i: "1o" for i in range(self.order)})
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        nn.init.zeros_(self.spherical_out_mlp[0].bias)
        nn.init.zeros_(self.spherical_out_mlp[2].bias)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_index: torch.LongTensor
    ) -> torch.Tensor:
        spherical_out = self.spherical_out_mlp(x_spherical)
        scalar_out = self.scalar_out_mlp(x_scalar)
        atom_out = self.rsh_conv(spherical_out, scalar_out)
        reduced_out = scatter(atom_out, batch_index, dim=0)
        split_vectors = torch.split(reduced_out, 3, dim=-1)
        spherical_res = self.rtp(*split_vectors)
        Q = self.rtp.change_of_basis
        cartesian_res = spherical_res @ Q.flatten(-len(self.order))
        shape = list(spherical_res.shape[:-1]) + list(Q.shape[1:])
        cartesian_res = cartesian_res.view(shape)
        if self.trace_out:
            res = torch.diagonal(res, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3
            return res
        else:
            return cartesian_res
