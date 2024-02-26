from typing import Union, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .xpainn import (
    XEmbedding, XPainnMessage, XPainnUpdate, EleEmbedding,
)
from .painn import (
    Embedding, PainnMessage, PainnUpdate,
)
from .output import resolve_output
from ..utils import NetConfig


class XPaiNN(nn.Module):
    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
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
        self.out = resolve_output(config)
    
    def forward(self, data: Data) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no; pos=data.pos; edge_index=data.edge_index
        if hasattr(data, "shifts"):
            shifts = data.shifts
        else:
            shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device)
        # embed input
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index, shifts)
        # initialize vector with zeros
        x_spherical = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        # message passing and node update
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_spherical = msg(x_scalar, x_spherical, rbf, fcut, rsh, edge_index)
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
        # output
        result = self.out(data, x_scalar, x_spherical)
        return result


class XPaiNNEle(XPaiNN):
    def __init__(self, config: NetConfig) -> None:
        super().__init__(config)
        self.charge_ebd = nn.ModuleList([
            EleEmbedding(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])
        self.spin_ebd = nn.ModuleList([
            EleEmbedding(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])

    def forward(self, data: Data) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no; pos=data.pos; edge_index=data.edge_index; batch=data.batch
        charge = data.charge; spin = data.spin
        if hasattr(data, "shifts"):
            shifts = data.shifts
        else:
            shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device)
        # embed input
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index, shifts)
        # initialize vector with zeros
        x_spherical = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        # electron embedding, message passing and node update
        for ce, se, msg, upd in zip(self.charge_ebd, self.spin_ebd, self.message, self.update):
            x_scalar = x_scalar + ce(x_scalar, charge, batch) + se(x_scalar, spin, batch)
            x_scalar, x_spherical = msg(x_scalar, x_spherical, rbf, fcut, rsh, edge_index)
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
        # output
        result = self.out(data, x_scalar, x_spherical)
        return result


class PaiNN(nn.Module):
    def __init__(self, config: NetConfig) -> None:
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

    def forward(self, data: Data) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no; pos=data.pos; edge_index=data.edge_index; batch_idx=data.batch
        # embed input
        x_scalar, rbf, envelop, rsh = self.embed(at_no, pos, edge_index)
        # initialize vector with zeros
        x_vector = torch.zeros((x_scalar.shape[0], 3, 128), device=x_scalar.device)
        # message passing and node update
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, envelop, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        # output
        result = self.out(data, x_scalar, x_vector)
        return result


def resolve_model(config: NetConfig) -> nn.Module:
    if config.version.lower() in ["xpainn", "xpainn-pbc"]:
        return XPaiNN(config)
    elif config.version.lower() in ["xpainn-ele", "xpainn-elepbc"]:
        return XPaiNNEle(config)
    elif config.version.lower() == "painn":
        return PaiNN(config)
    else:
        raise NotImplementedError(f"Unsupported model {config.version}")
