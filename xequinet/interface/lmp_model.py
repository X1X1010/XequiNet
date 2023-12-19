from typing import Tuple

import torch
import torch.nn as nn

from xequinet.utils import NetConfig

from ..nn import XEmbedding, XPainnMessage, XPainnUpdate, resolve_actfn
from ..utils import NetConfig, unit_conversion, get_default_unit, get_atomic_energy


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates.
        Returns:
            `e_tot`: Total energy.
            `forces`: Forces.
            `e_atom`: Atomic energies.
        """
        e_atom = self.out_mlp(x_scalar).view(-1)
        grad = torch.autograd.grad(
            [e_atom.sum(),],
            [coord,],
            retain_graph=True,
            create_graph=True,
        )[0]
        forces = torch.zeros_like(coord)    # because the output of `autograd.grad()` is `Tuple[Optional[torch.Tensor],...]` in jit script
        if grad is not None:                # which means the complier thinks that `neg_grad` may be `torch.Tensor` or `None`
            forces -= grad                  # but neg_grad there are not allowed to be `None`
        return e_atom, forces               # so we add this if statement to let the compiler make sure that `neg_grad` is not `None`


class LmpPaiNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.embed = XEmbedding(
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
        self.prop_unit_conv = unit_conversion(self.prop_unit, "kcal_per_mol")
        self.grad_unit_conv = unit_conversion(f"{self.prop_unit}/{self.len_unit}", "kcal_per_mol/Angstrom")


    def forward(
        self,
        at_no: torch.LongTensor,
        coord: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coord.requires_grad_(True)
        coord = coord * self.len_unit_conv
        x_scalar, rbf, fcut, rsh = self.embed(at_no, coord, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        e_atom, forces = self.out(x_scalar, coord)
        e_atom = (e_atom.double() + self.atom_sp[at_no]) * self.prop_unit_conv
        forces = forces.double() * self.grad_unit_conv
        e_tot = torch.sum(e_atom, dim=0, keepdim=True)       
        return e_tot, forces, e_atom

