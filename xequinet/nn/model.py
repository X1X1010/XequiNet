from typing import Tuple, Union

import torch
import torch.nn as nn

from .xpainn import (
    XEmbedding, PBCEmbedding,
    XPainnMessage, XPainnUpdate,
    ElectronicFuse,
)
from .painn import (
    Embedding, PainnMessage, PainnUpdate,
)
from .output import (
    ScalarOut, NegGradOut, VectorOut, PolarOut, SpatialOut,
)
from ..utils import NetConfig


def resolve_output(config: NetConfig):
    if config.output_mode == "scalar":
        return ScalarOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.output_dim,
            actfn=config.activation,
            node_bias=config.node_average,
        )
    elif config.output_mode == "grad":
        return NegGradOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
            node_bias=config.node_average,
        )
    elif config.output_mode == "vector":
        return VectorOut(
            edge_irreps=config.edge_irreps,
            hidden_irreps=config.hidden_irreps,
            output_dim=config.output_dim,
        )
    elif config.output_mode == "polar":
        return PolarOut(
            edge_irreps=config.edge_irreps,
            hidden_irreps=config.hidden_irreps,
            output_dim=config.output_dim,
        )
    elif config.output_mode == "spatial":
        return SpatialOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
        )
    else:
        raise NotImplementedError(f"output mode {config.output_mode} is not implemented")


class XPaiNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config
        self.cutoff = config.cutoff
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
        self.charge_fuse = nn.ModuleList([
            ElectronicFuse(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])
        self.spin_fuse = nn.ModuleList([
            ElectronicFuse(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])
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
        self.out = resolve_output(config)
    
    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.LongTensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        batch: torch.LongTensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            `at_no`: Atomic numbers.
            `pos`: Atomic coordinates.
            `edge_index`: Edge index.
            `charge`: Molecular charges.
            `spin`: Molecular spins. (2S, not 2S+1)
            `batch`: Batch index.
        Returns:
            `result`: Output.
        """
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for cf, sf, msg, upd in zip(self.charge_fuse, self.spin_fuse, self.message, self.update):
            x_scalar = x_scalar + cf(x_scalar, charge, batch) + sf(x_scalar, spin, batch)
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        result = self.out(x_scalar, x_vector, pos, batch)
        return result


class PBCPaiNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config
        self.cutoff = config.cutoff
        self.embed = PBCEmbedding(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.charge_fuse = nn.ModuleList([
            ElectronicFuse(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])
        self.spin_fuse = nn.ModuleList([
            ElectronicFuse(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])
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
        self.out = resolve_output(config)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        shifts: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            `at_no`: Atomic numbers.
            `pos`: Atomic coordinates.
            `shifts`: Shifts.
            `edge_index`: Edge index.
            `charge`: Molecular charges.
            `spin`: Molecular spins. (2S, not 2S+1)
            `batch`: Batch index.
        Returns:
            `result`: Output.
        """
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, shifts, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for cf, sf, msg, upd in zip(self.charge_fuse, self.spin_fuse, self.message, self.update):
            x_scalar = x_scalar + cf(x_scalar, charge, batch) + sf(x_scalar, spin, batch)
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        result = self.out(x_scalar, x_vector, pos, batch)
        return result


class PaiNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.embed = Embedding(
            node_dim=config.node_dim,
            num_basis=config.num_basis,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.message = nn.ModuleList([
            PainnMessage(
                node_dim=config.node_dim,
                edge_dim=config.edge_dim,
                num_basis=config.num_basis,
                actfn=config.activation,
            )
            for _ in range(config.action_blocks)
        ])
        self.update = nn.ModuleList([
            PainnUpdate(
                node_dim=config.node_dim,
                edge_dim=config.edge_dim,
                actfn=config.activation,
            )
            for _ in range(config.action_blocks)
        ])
        self.out = resolve_output(config)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.LongTensor,
        batch_idx: torch.LongTensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_scalar, rbf, envelop, rsh = self.embed(at_no, pos, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], 3, 128), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, envelop, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        result = self.out(x_scalar, at_no, pos, batch_idx)
        return result


def resolve_model(config: NetConfig) -> nn.Module:
    if hasattr(config, "original") and config.original:
        return PaiNN(config)
    if config.pbc:
        return PBCPaiNN(config)
    else:
        return XPaiNN(config)