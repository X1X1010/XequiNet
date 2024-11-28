import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter, scatter_sum

from xequinet import keys
from xequinet.utils import qc

from .basic import resolve_activation
from .o3layer import Gate
from .tp import get_feasible_tp
from .xe3net import SelfMixTP, Sph2Cart


class OutputModule(nn.Module):
    extra_properties: List[str]

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class ScalarOut(OutputModule):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        activation: str = "silu",
        node_shift: float = 0.0,
        node_scale: float = 1.0,
        reduce_op: Optional[str] = "sum",
        output_field: str = keys.SCALAR_OUTPUT,
        **kwargs,
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `out_dim`: Output dimension.
            `activation`: Activation function type.
            `node_shift`: Shift for atomic wise output.
            `node_scale`: Scale for atomic wise output.
            `reduce_op`: Reduce operation for the output.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        final_linear = nn.Linear(self.hidden_dim, 1)
        final_linear.weight.data *= node_scale
        nn.init.constant_(final_linear.bias, node_shift)
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_activation(activation),
            final_linear,
        )
        self.reduce_op = reduce_op
        self.output_field = output_field
        self.extra_properties = [output_field]

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch = data[keys.BATCH]
        node_scalar = data[keys.NODE_INVARIANT]
        res = self.out_mlp(node_scalar).reshape(-1)

        if self.reduce_op is not None:
            res = scatter(res, batch, dim=0, reduce=self.reduce_op)
        data[self.output_field] = res

        return data


class EnergyOut(OutputModule):
    # almost the same as ScalarOut, but ensure jitable.
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        activation: str = "silu",
        node_shift: float = 0.0,
        node_scale: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `out_dim`: Output dimension.
            `activation`: Activation function type.
            `node_shift`: Shift for atomic wise output.
            `node_scale`: Scale for atomic wise output.
            `reduce_op`: Reduce operation for the output.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        final_linear = nn.Linear(self.hidden_dim, 1)
        final_linear.weight.data *= node_scale
        nn.init.constant_(final_linear.bias, node_shift)
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_activation(activation),
            final_linear,
        )
        self.extra_properties = [keys.TOTAL_ENERGY, keys.ATOMIC_ENERGIES]

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch = data[keys.BATCH]
        node_scalar = data[keys.NODE_INVARIANT]
        atom_eng_out = self.out_mlp(node_scalar).reshape(-1)

        if keys.ATOMIC_ENERGIES in data:
            atomic_energies = data[keys.ATOMIC_ENERGIES] + atom_eng_out
        else:
            atomic_energies = atom_eng_out
        total_energy = scatter_sum(atomic_energies, batch, dim=0)
        data[keys.ATOMIC_ENERGIES] = atomic_energies
        data[keys.TOTAL_ENERGY] = total_energy

        return data


class AtomicChargesOut(OutputModule):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        activation: str = "silu",
        conservation: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `activation`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_activation(activation),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.out_mlp[0].bias)
        nn.init.zeros_(self.out_mlp[2].bias)
        self.conservation = conservation
        self.extra_properties = [keys.ATOMIC_CHARGES]

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        node_scalar = data[keys.NODE_INVARIANT]
        batch = data[keys.BATCH]
        atomic_charges = self.out_mlp(node_scalar).reshape(-1)
        if self.conservation:
            raw_total_charge = scatter_sum(atomic_charges, batch, dim=0)
            num_atoms = scatter_sum(
                src=torch.ones_like(atomic_charges),
                index=batch,
                dim=0,
            )
            if keys.TOTAL_CHARGE in data:
                total_charge = data[keys.TOTAL_CHARGE]
            else:
                total_charge = torch.zeros_like(raw_total_charge)
            delta_charge = (total_charge - raw_total_charge) / num_atoms
            atomic_charges += delta_charge.index_select(0, batch)
        data[keys.ATOMIC_CHARGES] = atomic_charges

        return data


class DipoleOut(OutputModule):
    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Iterable = "32x1o",
        activation: str = "silu",
        magnitude: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `activation`: Activation function type.
            `magnitude`: Whether to compute the magnitude of the dipole.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_activation(activation),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.equi_out_mlp = nn.Sequential(
            o3.Linear(self.node_irreps, self.hidden_irreps),
            Gate(self.hidden_irreps, activation=activation),
            o3.Linear(self.hidden_irreps, "1x1o"),
        )
        self.magnitude = magnitude
        self.extra_properties = [
            keys.DIPOLE if not magnitude else keys.DIPOLE_MAGNITUDE
        ]

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch = data[keys.BATCH]
        node_scalar = data[keys.NODE_INVARIANT]
        node_equi = data[keys.NODE_EQUIVARIANT]

        equi_out = self.equi_out_mlp(node_equi)[:, [2, 0, 1]]  # [y, z, x] -> [x, y, z]
        scalar_out = self.scalar_out_mlp(node_scalar)
        dipole = scatter_sum(
            src=equi_out * scalar_out,
            index=batch,
            dim=0,
        )
        data[keys.DIPOLE] = dipole

        if self.magnitude:
            dipole_norm = torch.linalg.norm(dipole, dim=-1)
            data[keys.DIPOLE_MAGNITUDE] = dipole_norm
        return data


class PolarOut(OutputModule):
    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Iterable = "64x0e + 16x2e",
        activation: str = "silu",
        isotropic: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `activation`: Activation function type.
            `isotropic`: Whether to compute the isotropic polarizability.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_activation(activation),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.equi_out_mlp = nn.Sequential(
            o3.Linear(self.node_irreps, self.hidden_irreps, biases=True),
            Gate(self.hidden_irreps, activation=activation),
            o3.Linear(self.hidden_irreps, "1x0e + 1x2e", biases=True),
        )
        self.rsh_conv = o3.ElementwiseTensorProduct("1x0e + 1x2e", "2x0e")
        self.isotropic = isotropic
        self.extra_properties = [
            keys.POLARIZABILITY if not isotropic else keys.ISO_POLARIZABILITY
        ]

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch = data[keys.BATCH]
        node_scalar = data[keys.NODE_INVARIANT]
        node_equi = data[keys.NODE_EQUIVARIANT]

        equi_out = self.equi_out_mlp(node_equi)
        scalar_out = self.scalar_out_mlp(node_scalar)
        polar_out = scatter_sum(
            src=self.rsh_conv(equi_out, scalar_out),
            index=batch,
            dim=0,
        )
        zero_order = polar_out[:, 0:1]
        second_order = polar_out[:, 1:6]
        # build zero order output
        zero_out = torch.diag_embed(torch.repeat_interleave(zero_order, 3, dim=-1))
        # build second order output
        second_out = torch.empty_like(zero_out)
        d_norm = torch.linalg.norm(second_order, dim=-1)
        dxy = second_order[:, 0]
        dyz = second_order[:, 1]
        dz2 = second_order[:, 2]
        dzx = second_order[:, 3]
        dx2_y2 = second_order[:, 4]
        second_out[:, 0, 0] = (1 / math.sqrt(3)) * (d_norm - dz2) + dx2_y2
        second_out[:, 1, 1] = (1 / math.sqrt(3)) * (d_norm - dz2) - dx2_y2
        second_out[:, 2, 2] = (1 / math.sqrt(3)) * (d_norm + 2 * dz2)
        second_out[:, 0, 1] = second_out[:, 1, 0] = dxy
        second_out[:, 1, 2] = second_out[:, 2, 1] = dyz
        second_out[:, 0, 2] = second_out[:, 2, 0] = dzx
        # add together
        polarizability = zero_out + second_out
        data[keys.POLARIZABILITY] = polarizability
        if self.isotropic:
            iso_polar = torch.diagonal(polarizability, dim1=-2, dim2=-1).mean(dim=-1)
            data[keys.ISO_POLARIZABILITY] = iso_polar

        return data


class SpatialOut(OutputModule):
    masses: torch.Tensor

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        activation: str = "silu",
        **kwargs,
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Hidden dimension.
            `activation`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_activation(activation),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.register_buffer("masses", torch.Tensor(qc.ATOM_MASS))
        self.extra_properties = [keys.SPATIAL_EXTENT]

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch = data[keys.BATCH]
        pos = data[keys.POSITIONS]
        atomic_numbers = data[keys.ATOMIC_NUMBERS]
        masses = self.masses[atomic_numbers]
        centroids = scatter(masses * pos, batch, dim=0) / scatter(masses, batch, dim=0)
        pos -= centroids[batch]

        node_scalar = data[keys.NODE_INVARIANT]
        scalar_out = self.scalar_out_mlp(node_scalar)
        spatial = torch.square(pos).sum(dim=1, keepdim=True)
        spatial_extent = scatter_sum(scalar_out * spatial, batch, dim=0)
        data[keys.SPATIAL_EXTENT] = spatial_extent
        return data


class CartTensorOut(OutputModule):
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
        activation: str = "silu",
        reduce_op: Optional[str] = "sum",
        layer_norm: str = True,
        isotropic: bool = False,
        output_field: str = keys.CARTESIAN_TENSOR,
        **kwargs,
    ) -> None:
        super().__init__()

        if order != 2 and isotropic:
            raise ValueError("Isotropic output is only supported for order 2 tensor.")
        self.isotropic = isotropic

        # self-mix tensor product layer to generize high `l`
        self.selfmix_tp = SelfMixTP(node_irreps, hidden_channels, layer_norm)
        self.mixed_irreps = self.selfmix_tp.irreps_out

        # spherical to cartesian transform
        self.sph2cart = Sph2Cart(formula=symmetry)
        self.rtp_irreps = self.sph2cart.rtp_irreps

        # tensor product to form spherical output
        self.hidden_dim = hidden_dim
        sph_irreps, instruct = get_feasible_tp(
            self.mixed_irreps, self.mixed_irreps, self.rtp_irreps, "uuw"
        )
        self.tp = o3.TensorProduct(
            self.mixed_irreps,
            self.mixed_irreps,
            sph_irreps,
            instruct,
            internal_weights=False,
            shared_weights=False,
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(node_dim, self.hidden_dim),
            resolve_activation(activation),
            nn.Linear(self.hidden_dim, self.tp.weight_numel),
        )
        if sph_irreps != self.rtp_irreps:
            self.post_lin = o3.Linear(sph_irreps, self.rtp_irreps, biases=False)
        else:
            self.post_lin = None

        self.reduce_op = reduce_op
        self.output_field = output_field
        self.extra_properties = [output_field]

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch = data[keys.BATCH]
        node_scalar = data[keys.NODE_INVARIANT]
        node_equi = data[keys.NODE_EQUIVARIANT]

        # self mix to expand the spherical features
        tp_in = self.selfmix_tp(node_equi)
        # get the weight for tensor product
        tp_weight = self.weight_mlp(node_scalar)
        # tensor product
        out_equi = self.tp(tp_in, tp_in, tp_weight)
        if self.post_lin is not None:
            out_equi = self.post_lin(out_equi)
        # cartesian transform
        out_cart = self.sph2cart(out_equi)
        if self.reduce_op is not None:
            out_cart = scatter(out_cart, batch, dim=0, reduce=self.reduce_op)

        if self.isotropic:
            assert out_cart.dim() == 3
            cart_tensor = torch.diagonal(out_cart, dim1=-2, dim2=-1).mean(dim=-1)
        else:
            # [y, z, x] -> [x, y, z]
            for i_dim in range(1, out_cart.dim()):
                out_cart = out_cart.roll(shifts=1, dims=i_dim)
            cart_tensor = out_cart
        data[self.output_field] = cart_tensor

        return data


def resolve_output(mode: str, **kwargs) -> OutputModule:

    output_factory = {
        "scalar": ScalarOut,
        "energy": EnergyOut,
        "charges": AtomicChargesOut,
        "atomic_charges": AtomicChargesOut,
        "dipole": DipoleOut,
        "polar": PolarOut,
        "spatial": SpatialOut,
        "cartesian": CartTensorOut,
    }
    return output_factory[mode](**kwargs)
