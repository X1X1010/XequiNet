from typing import Dict, Tuple
import math

import torch
import torch.nn as nn
from torch_cluster import radius_graph

from xequinet.utils import NetConfig

from ..nn import XEmbedding, XPainnMessage, XPainnUpdate, resolve_actfn
from ..utils import NetConfig, unit_conversion, get_default_unit, get_atomic_energy


class JitGradOut(nn.Module):
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
            `energy`: Total energy.
            `grads`: Gradients.
        """
        atom_out = self.out_mlp(x_scalar)
        energy = atom_out.sum()
        grad = torch.autograd.grad(
            [energy,],
            [coord,],
            retain_graph=True,
            create_graph=True,
        )
        nuc_grad = grad[0]
        if nuc_grad is None:  # condition needed to unwrap optional for torchscript
            raise RuntimeError("Gradient is None")
        return energy, nuc_grad


class JitPaiNN(nn.Module):
    """
    XPaiNN model for JIT script. This model does not consider batch.
    """
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
        self.out = JitGradOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
        )
        self.cutoff = config.cutoff
        self.max_edges = config.max_edges
        self.prop_unit, self.len_unit = get_default_unit()
        atom_sp = get_atomic_energy(config.atom_ref) - get_atomic_energy(config.batom_ref)
        self.register_buffer("atom_sp", atom_sp)
        self.len_unit_conv = unit_conversion("Bohr", self.len_unit)
        self.prop_unit_conv = unit_conversion(self.prop_unit, "AU")
        self.grad_unit_conv = unit_conversion(f"{self.prop_unit}/{self.len_unit}", "AU")

    def forward(
        self,
        at_no: torch.LongTensor,
        coord: torch.Tensor,
        charge: int,
        spin: int,
    ) -> Dict[str, torch.Tensor]:
        coord.requires_grad_(True)
        coord = coord * self.len_unit_conv
        edge_index = radius_graph(coord, r=self.cutoff, max_num_neighbors=self.max_edges)
        x_scalar, rbf, fcut, rsh = self.embed(at_no, coord, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        energy, nuc_grad = self.out(x_scalar, coord)
        energy = (energy.double() + self.atom_sp[at_no].sum()) * self.prop_unit_conv
        nuc_grad = nuc_grad.double() * self.grad_unit_conv
        result = {"energy": energy, "forces": nuc_grad}
        return result


class JitEleEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.sqrt_dim = math.sqrt(node_dim)
        self.q_linear = nn.Linear(node_dim, node_dim)
        nn.init.zeros_(self.q_linear.bias)
        self.k_linear = nn.Linear(1, node_dim)
        nn.init.zeros_(self.k_linear.bias)
        self.v_linear = nn.Linear(1, node_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        ele: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `x`: Node features.
            `ele`: Electronic features.
        Returns:
            Atomic features.
        """
        q = self.q_linear(x)
        k = self.k_linear(ele)
        v = self.v_linear(ele)
        dot = torch.sum(q * k, dim=1, keepdim=True) / self.sqrt_dim
        attn = torch.softmax(dot, dim=0)
        return attn * v


class JitPaiNNEle(JitPaiNN):
    """XPaiNN-ele model for JIT script. This model does not consider batch."""
    def __init__(self, config: NetConfig):
        super().__init__(config)
        self.charge_ebd = nn.ModuleList([
            JitEleEmbedding(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])
        self.spin_ebd = nn.ModuleList([
            JitEleEmbedding(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])

    def forward(
        self,
        at_no: torch.LongTensor,
        coord: torch.Tensor,
        charge: int,
        spin: int,
    ) -> Dict[str, torch.Tensor]:
        coord.requires_grad_(True)
        charge_t = torch.tensor([[charge]], dtype=coord.dtype, device=coord.device)
        spin_t = torch.tensor([[spin]], dtype=coord.dtype, device=coord.device)
        coord = coord * self.len_unit_conv
        edge_index = radius_graph(coord, r=self.cutoff, max_num_neighbors=self.max_edges)
        x_scalar, rbf, fcut, rsh = self.embed(at_no, coord, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for ce, se, msg, upd in zip(self.charge_ebd, self.spin_ebd, self.message, self.update):
            x_scalar = x_scalar + ce(x_scalar, charge_t) + se(x_scalar, spin_t)
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        energy, nuc_grad = self.out(x_scalar, coord)
        energy = (energy.double() + self.atom_sp[at_no].sum()) * self.prop_unit_conv
        nuc_grad = nuc_grad.double() * self.grad_unit_conv
        result = {"energy": energy, "gradient": nuc_grad}
        return result


def resolve_jit_model(config: NetConfig) -> nn.Module:
    if config.version.lower() == "xpainn":
        return JitPaiNN(config)
    elif config.version.lower() == "xpainn-ele":
        return JitPaiNNEle(config)
    else:
        raise NotImplementedError(f"Unsupported model {config.version}")
    