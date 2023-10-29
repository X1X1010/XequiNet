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
        graph_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `out_dim`: Output dimension.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
            `graph_bias`: Bias for graphic output.
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
        nn.init.zeros_(self.out_mlp[2].bias)
        self.register_buffer("node_bias", torch.tensor(node_bias))
        self.register_buffer("graph_bias", torch.tensor(graph_bias))
    
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
        atom_out = self.out_mlp(x_scalar) + self.node_bias
        res = scatter(atom_out, batch_index, dim=0)
        return res + self.graph_bias


class NegGradOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        node_bias: float = 0.0,
        graph_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
            `graph_bias`: Bias for graphic output.
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
        nn.init.zeros_(self.out_mlp[2].bias)
        self.register_buffer("node_bias", torch.tensor(node_bias))
        self.register_buffer("graph_bias", torch.tensor(graph_bias))

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
        atom_out = self.out_mlp(x_scalar) + self.node_bias
        res =  scatter(atom_out, batch_index, dim=0)
        grad = torch.autograd.grad(
            [atom_out.sum(),],
            [coord,],
            retain_graph=True,
            create_graph=True,
        )[0]
        neg_grad = torch.zeros_like(coord)      # because the output of `autograd.grad()` is `Tuple[Optional[torch.Tensor],...]` in jit script
        if grad is not None:                    # which means the complier thinks that `neg_grad` may be `torch.Tensor` or `None`
            neg_grad = neg_grad - grad          # but neg_grad there are not allowed to be `None`
        return res + self.graph_bias, neg_grad  # so we add this if statement to let the compiler make sure that `neg_grad` is not `None`


class VectorOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1e",
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
            o3.Linear(self.hidden_irreps, "1x1e"),
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
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1e",
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
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1e",
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
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
            o3.Linear(self.hidden_irreps, "1x1e"),
        )

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
        spherical_out = self.spherical_out_mlp(x_spherical)
        scalar_out = self.scalar_out_mlp(x_scalar)
        atom_out = torch.square(spherical_out).sum(dim=1) * scalar_out
        res = scatter(atom_out, batch_index, dim=0)
        return res


class PBCScalarOut(ScalarOut):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 1,
        actfn: str = "silu",
        node_bias: float = 0.0,
        graph_bias: float = 0.0,
    ):
        super().__init__(node_dim, hidden_dim, out_dim,
                         actfn, node_bias, graph_bias)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_index: torch.LongTensor,
        at_filter: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates. (Unused in this module)
            `batch_index`: Batch index.
            `at_filter`: Atomic filter.
        Returns:
            `res`: Scalar output.
        """
        atom_out = self.out_mlp(x_scalar[at_filter]) + self.node_bias
        res = scatter(atom_out, batch_index, dim=0)
        return res + self.graph_bias


class PBCNegGradOut(NegGradOut):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        node_bias: float = 0.0,
        graph_bias: float = 0.0,
    ):
        super().__init__(node_dim, hidden_dim, actfn, node_bias, graph_bias)

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_index: torch.LongTensor,
        at_filter: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates. (Unused in this module)
            `batch_index`: Batch index.
            `at_filter`: Atomic filter.
        Returns:
            `res`: Scalar output.
            `neg_grad`: Negative gradient.
        """
        atom_out = self.out_mlp(x_scalar[at_filter]) + self.node_bias
        res = scatter(atom_out, batch_index, dim=0)
        grad = torch.autograd.grad(
            [atom_out.sum(),],
            [coord[at_filter],],
            retain_graph=True,
            create_graph=True,
        )[0]
        neg_grad = torch.zeros_like(coord[at_filter])      # because the output of `autograd.grad()` is `Tuple[Optional[torch.Tensor],...]` in jit script
        if grad is not None:                               # which means the complier thinks that `neg_grad` may be `torch.Tensor` or `None`
            neg_grad = neg_grad - grad                     # but neg_grad there are not allowed to be `None`
        return res + self.graph_bias, neg_grad             # so we add this if statement to let the compiler make sure that `neg_grad` is not `None`
