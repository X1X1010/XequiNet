from typing import Dict, Tuple, Union, Iterable

import torch
import torch.nn as nn
from e3nn import o3

from xequinet.utils import NetConfig

from ..nn import XEmbedding, XPainnMessage, XPainnUpdate, resolve_actfn
from ..utils import NetConfig, unit_conversion, get_default_unit, get_atomic_energy



class LmpEmbedding(XEmbedding):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux56",
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "cosine",
    ):
        super().__init__(
            node_dim, edge_irreps, embed_basis, aux_basis,
            num_basis, rbf_kernel, cutoff, cutoff_fn,
        )

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        shifts: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Args:
            `x`: Atomic features.
            `pos`: Atomic coordinates.
            `edge_index`: Edge index.
            `shifts`: Shift vectors.
        Returns:
            `x_scalar`: Scalar features.
            `rbf`: Values under radial basis functions.
            `fcut`: Values under cutoff function.
            `rsh`: Real spherical harmonics.
        """
        # calculate distance and relative position
        vec = pos[edge_index[0]] - pos[edge_index[1]] - shifts
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        # node linear
        x = self.int2c1e(at_no)
        x_scalar = self.node_lin(x)
        # calculate radial basis function
        rbf = self.rbf(dist)
        fcut = self.cutoff_fn(dist)
        # calculate spherical harmonics  [x, y, z] -> [y, z, x]
        rsh = self.sph_harm(vec[:, [1, 2, 0]])  # unit vector, normalized by component
        return x_scalar, vec, rbf, fcut, rsh


class LmpGradOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Dimension of node feature.
            `hidden_dim`: Dimension of hidden layer.
            `actfn`: Activation function.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        x_scalar: torch.Tensor,
        coord: torch.Tensor,
        vec: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates.
            `vec`: rij vectors.
            `edge_index`: Edge index.
        Returns:
            `energies`: Atomic energies.
            `forces`: Atomic forces.
            `virials`: Atomic virials.
        """
        energies = self.out_mlp(x_scalar).view(-1)
        grad = torch.autograd.grad(
            [energies.sum(),],
            [coord, vec],
            retain_graph=True,
            create_graph=True,
        )
        nuc_grad = grad[0]
        if nuc_grad is None:  # condition needed to unwrap optional for torchscript
            raise RuntimeError("Nuclear gradient is None")
        forces = -nuc_grad

        edge_grad = grad[1]
        if edge_grad is None: # condition needed to unwrap optional for torchscript
            raise RuntimeError("Edge gradient is None")
        edge_virials = torch.einsum("zi, zj -> zij", edge_grad, vec)
        virials = torch.zeros((coord.shape[0], 3, 3), device=coord.device)
        # edge_virial is distributed into two nodes equally.
        virials = virials.index_add(0, edge_index[0], edge_virials / 2)
        virials = virials.index_add(0, edge_index[1], edge_virials / 2)

        return energies, forces, virials


class LmpPaiNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.embed = LmpEmbedding(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.message = nn.ModuleList([
            XPainnMessage(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                num_basis=config.num_basis,
                actfn=config.activation,
                norm_type=config.norm_type,
            )
            for _ in range(config.action_blocks)
        ])
        self.update = nn.ModuleList([
            XPainnUpdate(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                actfn=config.activation,
                norm_type=config.norm_type,
            )
            for _ in range(config.action_blocks)
        ])
        self.out = LmpGradOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
        )
        self.prop_unit, self.len_unit = get_default_unit()
        atom_sp = get_atomic_energy(config.atom_ref) - get_atomic_energy(config.batom_ref)
        self.register_buffer("atom_sp", atom_sp)
        self.len_unit_conv = unit_conversion("Angstrom", self.len_unit)
        self.prop_unit_conv = unit_conversion(self.prop_unit, "eV")
        self.grad_unit_conv = unit_conversion(f"{self.prop_unit}/{self.len_unit}", "eV/Angstrom")


    def forward(
        self,
        at_no: torch.LongTensor,
        coord: torch.Tensor,
        edge_index: torch.LongTensor,
        shifts: torch.Tensor,
        charge: int,
        spin: int,
    ) -> Dict[str, torch.Tensor]:
        # set coord.requires_grad_(True) to enable autograd
        coord.requires_grad_(True)
        coord = coord * self.len_unit_conv
        # network forward
        x_scalar, vec, rbf, fcut, rsh = self.embed(at_no, coord, shifts, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        energies, forces, virials = self.out(x_scalar, coord, vec, edge_index)
        # add atomic species energy and convert to eV
        energies = (energies.double() + self.atom_sp[at_no]) * self.prop_unit_conv
        energy = torch.sum(energies)
        # convert forces to eV/Angstrom
        forces = forces.double() * self.grad_unit_conv
        # symmetrize virial tensor and convert to eV
        virial = torch.sum(virials, dim=0)
        virial = 0.5 * (virial + virial.T) * self.prop_unit_conv
        # return result
        result = {
            "energy": energy, "energies": energies,
            "forces": forces,
            "virial": virial, "virials": virials,
        }
        return result

