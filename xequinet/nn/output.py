import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter, scatter_sum

from xequinet import keys
from xequinet.utils import qc

from .electronic import CoulombWithCutoff
from .o3layer import Gate, resolve_activation
from .tp import get_feasible_tp
from .xe3net import SelfMixTP, Sph2Cart


class ScalarOut(nn.Module):
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

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        f: bool = False,
        v: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        batch = data[keys.BATCH]
        node_scalar = data[keys.NODE_INVARIANT]
        res = self.out_mlp(node_scalar).reshape(-1)

        if self.reduce_op is not None:
            res = scatter(res, batch, dim=0, reduce=self.reduce_op)

        return data, {self.output_field: res}


class ForceFieldOut(nn.Module):
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
            `activation`: Activation function type.
            `node_shift`: Shift for atomic wise output.
            `node_scale`: Scale for atomic wise output.
            `compute_forces`: Whether to compute forces.
            `compute_virial`: Whether to compute virial.
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

    def _compute_forces(
        self,
        energy: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
        pos_grad = torch.autograd.grad(
            outputs=[energy],
            inputs=[pos],
            grad_outputs=grad_outputs,
            retain_graph=self.training,
            create_graph=self.training,
        )[0]
        if pos_grad is None:
            pos_grad = torch.zeros_like(pos)
        return -1.0 * pos_grad

    def _compute_virial(
        self,
        energy: torch.Tensor,
        strain: torch.Tensor,
    ) -> torch.Tensor:
        grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
        strain_grad = torch.autograd.grad(
            outputs=[energy],
            inputs=[strain],
            grad_outputs=grad_outputs,
            retain_graph=self.training,
            create_graph=self.training,
        )[0]
        if strain_grad is None:
            strain_grad = torch.zeros_like(strain)
        return -1.0 * strain_grad

    def _compute_forces_and_virial(
        self,
        energy: torch.Tensor,
        pos: torch.Tensor,
        strain: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
        pos_grad, strain_grad = torch.autograd.grad(
            outputs=[energy],
            inputs=[pos, strain],
            grad_outputs=grad_outputs,
            retain_graph=self.training,
            create_graph=self.training,
        )
        if pos_grad is None:
            pos_grad = torch.zeros_like(pos)
        if strain_grad is None:
            strain_grad = torch.zeros_like(strain)
        return -1.0 * pos_grad, -1.0 * strain_grad

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_forces: bool = True,
        compute_virial: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        batch = data[keys.BATCH]
        node_scalar = data[keys.NODE_INVARIANT]
        atom_eng_out = self.out_mlp(node_scalar).reshape(-1)

        if keys.ATOMIC_ENERGIES in data:
            atomic_energies = data[keys.ATOMIC_ENERGIES] + atom_eng_out
        else:
            atomic_energies = atom_eng_out
        data[keys.ATOMIC_ENERGIES] = atomic_energies

        total_energy = scatter_sum(atomic_energies, batch, dim=0)

        result = {
            keys.TOTAL_ENERGY: total_energy,
            keys.ATOMIC_ENERGIES: atomic_energies,
        }
        if compute_forces and compute_virial:
            forces, virial = self._compute_forces_and_virial(
                energy=total_energy,
                pos=data[keys.POSITIONS],
                strain=data[keys.STRAIN],
            )
            result[keys.FORCES] = forces
            result[keys.VIRIAL] = virial
        elif compute_forces:
            forces = self._compute_forces(
                energy=total_energy,
                pos=data[keys.POSITIONS],
            )
            result[keys.FORCES] = forces
        elif compute_virial:
            virial = self._compute_virial(
                energy=total_energy,
                strain=data[keys.STRAIN],
            )
            result[keys.VIRIAL] = virial
        return data, result


class AtomicChargesOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        activation: str = "silu",
        conservation: bool = True,
        coulomb_interaction: bool = False,
        coulomb_cutoff: Optional[float] = 10.0,
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
        self.coulomb_interaction = coulomb_interaction
        if self.coulomb_interaction:
            assert coulomb_cutoff is not None
            self.coulomb = CoulombWithCutoff(coulomb_cutoff)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        f: bool = False,
        v: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

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
                total_charge = torch.zeros(
                    (num_atoms,), dtype=torch.int, device=batch.device
                )
            delta_charge = (total_charge - raw_total_charge) / num_atoms
            atomic_charges += delta_charge.index_select(0, batch)

        data[keys.ATOMIC_CHARGES] = atomic_charges
        result = {keys.ATOMIC_CHARGES: atomic_charges}
        if self.coulomb_interaction:
            data = self.coulomb(data)

        return data, result


class DipoleOut(nn.Module):
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

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        f: bool = False,
        v: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

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
        result = {keys.DIPOLE: dipole}

        if self.magnitude:
            dipole_norm = torch.linalg.norm(dipole, dim=-1)
            result[keys.DIPOLE_MAGNITUDE] = dipole_norm
        return data, result


class PolarOut(nn.Module):
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

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        f: bool = False,
        v: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

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
        result = {keys.POLARIZABILITY: polarizability}
        if self.isotropic:
            iso_polar = torch.diagonal(polarizability, dim1=-2, dim2=-1).mean(dim=-1)
            result[keys.ISO_POLARIZABILITY] = iso_polar
        return data, result


class SpatialOut(nn.Module):
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

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        f: bool = False,
        v: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

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
        return data, {keys.SPATIAL_EXTENT: spatial_extent}


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
        activation: str = "silu",
        reduce_op: Optional[str] = "sum",
        norm_type: str = "layer",
        isotropic: bool = False,
        output_field: str = keys.CARTESIAN_TENSOR,
        **kwargs,
    ) -> None:
        super().__init__()

        if order != 2 and isotropic:
            raise ValueError("Isotropic output is only supported for order 2 tensor.")
        self.isotropic = isotropic

        # self-mix tensor product layer to generize high `l`
        self.selfmix_tp = SelfMixTP(node_irreps, hidden_channels, norm_type)
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

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        f: bool = False,
        v: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

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
        return data, {self.output_field: cart_tensor}


def resolve_output(mode: str, **kwargs) -> nn.Module:

    output_factory = {
        "scalar": ScalarOut,
        "forcefield": ForceFieldOut,
        "charges": AtomicChargesOut,
        "dipole": DipoleOut,
        "polar": PolarOut,
        "spatial": SpatialOut,
        "cartesian": CartTensorOut,
    }
    return output_factory[mode](**kwargs)
