from typing import Iterable, Tuple, Optional

import math

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter
from e3nn import o3

from .o3layer import Gate, resolve_actfn
from .xe3net import SelfMixTP, Sph2Cart
from .tp import get_feasible_tp
from ..utils import NetConfig
from ..utils.qc import ATOM_MASS


class ScalarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 1,
        actfn: str = "silu",
        node_bias: float = 0.0,
        reduce_op: Optional[str] = "sum",
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `out_dim`: Output dimension.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
            `reduce_op`: Reduce operation for the output.
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
        nn.init.constant_(self.out_mlp[2].bias, node_bias)
        self.reduce_op = reduce_op
    
    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
        Returns:
            `res`: Scalar output.
        """
        batch = data.batch
        res = self.out_mlp(x_scalar)
        if self.reduce_op is not None:
            res = scatter(res, batch, dim=0, reduce=self.reduce_op)
        return res


class NegGradOut(ScalarOut):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        node_bias: float = 0.0,
        reduce_op: Optional[str] = "sum",
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
            `reduce_op`: Reduce operation for the output.
        """
        super().__init__(node_dim, hidden_dim, 1, actfn, node_bias, reduce_op)

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
        Returns:
            `res`: Scalar output.
            `neg_grad`: Negative gradient.
        """
        batch = data.batch; coord = data.pos
        res = self.out_mlp(x_scalar)
        if self.reduce_op is not None:
            res = scatter(res, batch, dim=0, reduce=self.reduce_op)
        grad = torch.autograd.grad(
            [res.sum(),],
            [coord,],
            retain_graph=True,
            create_graph=True,
        )[0]
        return res, -grad


class VectorOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Iterable = "32x1o",
        if_norm: bool = False,
        actfn: str = "silu",
        reduce_op: Optional[str] = "sum",
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `if_norm`: Whether to normalize the output.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.node_irreps, self.hidden_irreps),
            Gate(self.hidden_irreps, actfn=actfn),
            o3.Linear(self.hidden_irreps, "1x1o"),
        )
        self.if_norm = if_norm
        self.reduce_op = reduce_op

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
        Returns:
            `res`: Vector output.
        """
        batch = data.batch
        spherical_out = self.spherical_out_mlp(x_spherical)[:, [2, 0, 1]]  # [y, z, x] -> [x, y, z]
        scalar_out = self.scalar_out_mlp(x_scalar)
        res = spherical_out * scalar_out
        if self.reduce_op is not None:
            res = scatter(res, batch, dim=0, reduce=self.reduce_op)
        if self.if_norm:
            res = torch.linalg.norm(res, dim=-1, keepdim=True)
        return res


class PolarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Iterable = "64x0e + 16x2e",
        if_iso: bool = False,
        actfn: str = "silu",
        reduce_op: Optional[str] = "sum",
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `output_dim`: Output dimension. (9 for 3x3 matrix and 1 for trace of the matrix)
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 2),
        )
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.node_irreps, self.hidden_irreps, biases=True),
            Gate(self.hidden_irreps, actfn=actfn),
            o3.Linear(self.hidden_irreps, "1x0e + 1x2e", biases=True),
        )
        self.rsh_conv = o3.ElementwiseTensorProduct("1x0e + 1x2e", "2x0e")
        self.if_iso = if_iso
        self.reduce_op = reduce_op

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
        Returns:
            `res`: Polarizability.
        """
        batch = data.batch
        spherical_out = self.spherical_out_mlp(x_spherical)
        scalar_out = self.scalar_out_mlp(x_scalar)
        polar_out = self.rsh_conv(spherical_out, scalar_out)
        if self.reduce_op is not None:
            polar_out = scatter(polar_out, batch, dim=0, reduce=self.reduce_op)
        zero_order = polar_out[:, 0:1]
        second_order = polar_out[:, 1:6]
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
        if self.if_iso == 1:
            res = torch.diagonal(res, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3.0
        return res


class SpatialOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        reduce_op: Optional[str] = "sum",
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Hidden dimension.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.register_buffer("masses", torch.Tensor(ATOM_MASS))
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        self.reduce_op = reduce_op

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
        Returns:
            `res`: Spatial output.
        """
        batch = data.batch; coord = data.pos; at_no = data.at_no
        masses = self.masses[at_no]
        centroids = scatter(masses * coord, batch, dim=0) / scatter(masses, batch, dim=0)
        coord -= centroids[batch]
        scalar_out = self.scalar_out_mlp(x_scalar)
        spatial = torch.square(coord).sum(dim=1, keepdim=True)
        res = scalar_out * spatial
        if self.reduce_op is not None:
            res = scatter(res, batch, dim=0, reduce=self.reduce_op)
        return res


class CartTensorOut(nn.Module):
    """
    Output Module for order n tensor via tensor product of input hidden irreps and cartesian transform. 
    """
    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_channels: int = 32,
        order: int = 2,
        symmetry: str = "ij",
        # vector_space: Optional[Dict[str, str]] = None,
        if_iso: bool = False,
        actfn: str = "silu",
        # norm_type: str = "layer",
        reduce_op: Optional[str] = "sum",
    ) -> None:
        super().__init__()
        # self-mix tensor product layer to generize high `l`
        # self.selfmix_tp = SelfMixTP(node_irreps, hidden_channels, norm_type)
        # self.mixed_irreps = self.selfmix_tp.irreps_out
        self.node_irreps = o3.Irreps(node_irreps)
        hidden_irreps = []
        for mul, irrep in self.node_irreps:
            hidden_irreps.append((hidden_channels, irrep))
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.pre_lin = o3.Linear(node_irreps, self.hidden_irreps, biases=False)

        # spherical to cartesian transform
        # vec_space = vector_space if vector_space is not None else {i: "1o" for i in indices}
        self.sph2cart = Sph2Cart(formula=symmetry)
        self.rtp_irreps = self.sph2cart.rtp_irreps

        # tensor product to form spherical output
        self.hidden_dim = hidden_dim
        # sph_irreps, instruct = get_feasible_tp(
        #     self.mixed_irreps, self.mixed_irreps, self.rtp_irreps, "uuw"
        # )
        sph_irreps, instruct = get_feasible_tp(
            self.hidden_irreps, self.hidden_irreps, self.rtp_irreps, "uuw"
        )
        self.tp = o3.TensorProduct(
            # self.mixed_irreps,
            # self.mixed_irreps,
            self.hidden_irreps,
            self.hidden_irreps,
            sph_irreps,
            instruct,
            internal_weights=False,
            shared_weights=False,
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.tp.weight_numel),
        )
        if sph_irreps != self.rtp_irreps:
            self.post_lin = o3.Linear(sph_irreps, self.rtp_irreps, biases=False)
        else:
            self.post_lin = None

        self.reduce_op = reduce_op
        self.if_trace = if_iso and order == 2


    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor
    ) -> torch.Tensor:
        # self mix to expand the spherical features
        # x_in = self.selfmix_tp(x_spherical)
        x_in = self.pre_lin(x_spherical)
        # get the weight for tensor product
        tp_weight = self.weight_mlp(x_scalar)
        # tensor product
        x_sph = self.tp(x_in, x_in, tp_weight)
        if self.post_lin is not None:
            x_sph = self.post_lin(x_sph)
        # cartesian transform
        x_cart = self.sph2cart(x_sph)
        if self.reduce_op is not None:
            x_cart = scatter(x_cart, data.batch, dim=0, reduce=self.reduce_op)
        if self.if_trace:
            res = torch.diagonal(x_cart, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3.0
        else:
            # [y, z, x] -> [x, y, z]
            for i_dim in range(1, x_cart.dim()):
                x_cart = x_cart.roll(shifts=1, dims=i_dim)
            res = x_cart
        return res
    

def resolve_output(config: NetConfig):
    if '-' in config.output_mode:
        output_mode, extra = config.output_mode.split('-')
    else:
        output_mode, extra = config.output_mode, ''
    if output_mode == "scalar":
        return ScalarOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.output_dim,
            actfn=config.activation,
            node_bias=config.node_average,
            reduce_op=config.reduce_op,
        )
    elif output_mode == "grad":
        return NegGradOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
            node_bias=config.node_average,
            reduce_op=config.reduce_op,
        )
    elif output_mode == "vector":
        return VectorOut(
            node_dim=config.node_dim,
            node_irreps=config.node_irreps,
            hidden_dim=config.hidden_dim,
            hidden_irreps=config.hidden_irreps,
            if_norm=(extra == "norm"),
            actfn=config.activation,
            reduce_op=config.reduce_op,
        )
    elif output_mode == "polar":
        return PolarOut(
            node_dim=config.node_dim,
            node_irreps=config.node_irreps,
            hidden_dim=config.hidden_dim,
            hidden_irreps=config.hidden_irreps,
            if_iso=(extra == "iso"),
            actfn=config.activation,
            reduce_op=config.reduce_op,
        )
    elif output_mode == "spatial":
        return SpatialOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
            reduce_op=config.reduce_op,
       )
    elif output_mode == "cartesian":
        return CartTensorOut(
            node_dim=config.node_dim,
            node_irreps=config.node_irreps,
            hidden_dim=config.hidden_dim,
            hidden_channels=config.hidden_channels,
            order=config.order,
            symmetry=config.required_symm,
            if_iso=(extra == "iso"),
            actfn=config.activation,
            # norm_type=config.norm_type,
            reduce_op=config.reduce_op,
        )
    else:
        raise NotImplementedError(f"output mode {config.output_mode} is not implemented")
