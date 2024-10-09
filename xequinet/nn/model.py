from typing import Union, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .xpainn import (
    XEmbedding,
    XPainnMessage,
    XPainnUpdate,
    EleEmbedding,
)
from .painn import (
    Embedding,
    PainnMessage,
    PainnUpdate,
)
from .xe3net import LTCEmbeding
from .xqhnet import (
    MatEmbedding,
    NodewiseInteraction,
    NodeInterWithEle,
    MatrixTrans,
    MatrixOut,
)
from .output import resolve_output
from ..utils import NetConfig, MatToolkit


class XPaiNN(nn.Module):
    """
    eXtended PaiNN.
    """

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = XEmbedding(
            node_dim=config.node_dim,
            node_irreps=config.node_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.message = nn.ModuleList(
            [
                XPainnMessage(
                    node_dim=config.node_dim,
                    node_irreps=config.node_irreps,
                    num_basis=config.num_basis,
                    actfn=config.activation,
                    norm_type=config.norm_type,
                )
                for _ in range(config.action_blocks)
            ]
        )
        self.update = nn.ModuleList(
            [
                XPainnUpdate(
                    node_dim=config.node_dim,
                    node_irreps=config.node_irreps,
                    actfn=config.activation,
                    norm_type=config.norm_type,
                )
                for _ in range(config.action_blocks)
            ]
        )
        self.out = resolve_output(config)

    def forward(self, data: Data) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no
        pos = data.pos
        edge_index = data.edge_index
        if hasattr(data, "shifts"):
            shifts = data.shifts
        else:
            shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device)
        # embed input
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index, shifts)
        # initialize vector with zeros
        x_spherical = torch.zeros(
            (x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device
        )
        # message passing and node update
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_spherical = msg(
                x_scalar, x_spherical, rbf, fcut, rsh, edge_index
            )
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
        # output
        result = self.out(data, x_scalar, x_spherical)
        return result


class XPaiNNEle(XPaiNN):
    """
    XPaiNN with electron embedding.
    """

    def __init__(self, config: NetConfig) -> None:
        super().__init__(config)
        self.charge_ebd = nn.ModuleList(
            [
                EleEmbedding(node_dim=config.node_dim)
                for _ in range(config.action_blocks)
            ]
        )
        self.spin_ebd = nn.ModuleList(
            [
                EleEmbedding(node_dim=config.node_dim)
                for _ in range(config.action_blocks)
            ]
        )

    def forward(self, data: Data) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch
        charge = data.charge
        spin = data.spin
        if hasattr(data, "shifts"):
            shifts = data.shifts
        else:
            shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device)
        # embed input
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index, shifts)
        # initialize vector with zeros
        x_spherical = torch.zeros(
            (x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device
        )
        # electron embedding, message passing and node update
        for ce, se, msg, upd in zip(
            self.charge_ebd, self.spin_ebd, self.message, self.update
        ):
            x_scalar = (
                x_scalar + ce(x_scalar, charge, batch) + se(x_scalar, spin, batch)
            )
            x_scalar, x_spherical = msg(
                x_scalar, x_spherical, rbf, fcut, rsh, edge_index
            )
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
        # output
        result = self.out(data, x_scalar, x_spherical)
        return result


class PaiNN(nn.Module):
    """
    Traditional PaiNN.
    """

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
        self.message = nn.ModuleList(
            [
                PainnMessage(
                    node_dim=config.node_dim,
                    edge_dim=config.node_dim,
                    num_basis=config.num_basis,
                    actfn=config.activation,
                )
                for _ in range(config.action_blocks)
            ]
        )
        self.update = nn.ModuleList(
            [
                PainnUpdate(
                    node_dim=config.node_dim,
                    edge_dim=config.node_dim,
                    actfn=config.activation,
                )
                for _ in range(config.action_blocks)
            ]
        )
        self.out = resolve_output(config)

    def forward(self, data: Data) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no
        pos = data.pos
        edge_index = data.edge_index
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


class XPaiNNLTC(nn.Module):
    """
    XPaiNN with lattice embedding. (not Li Tang City 1! 5!)
    """

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.embed = LTCEmbeding(
            node_dim=config.node_dim,
            node_irreps=config.node_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.message = nn.ModuleList(
            [
                XPainnMessage(
                    node_dim=config.node_dim,
                    node_irreps=config.node_irreps,
                    num_basis=config.num_basis,
                    actfn=config.activation,
                    norm_type=config.norm_type,
                )
                for _ in range(config.action_blocks)
            ]
        )
        self.update = nn.ModuleList(
            [
                XPainnUpdate(
                    node_dim=config.node_dim,
                    node_irreps=config.node_irreps,
                    actfn=config.activation,
                    norm_type=config.norm_type,
                )
                for _ in range(config.action_blocks)
            ]
        )
        self.out = resolve_output(config)

    def forward(self, data: Data) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no
        pos = data.pos
        edge_index = data.edge_index
        lattice = data.lattice
        batch = data.batch
        if hasattr(data, "shifts"):
            shifts = data.shifts
        else:
            shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device)
        # embed input
        x_scalar, x_spherical, rbf, fcut, rsh = self.embed(
            at_no, lattice, pos, edge_index, shifts, batch
        )
        # message passing and node update
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_spherical = msg(
                x_scalar, x_spherical, rbf, fcut, rsh, edge_index
            )
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
        # output
        result = self.out(data, x_scalar, x_spherical)
        return result


class MatNetBase(nn.Module):
    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        assert config.mat_conv_blocks >= config.action_blocks
        self.num_pre_conv = config.mat_conv_blocks - config.action_blocks
        self.embed = MatEmbedding(
            node_dim=config.node_dim,
            node_channels=config.node_channels,
            max_l=config.max_l,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        config.node_irreps = (
            f"{self.embed.node_irreps}"  # for hyper parameter saving, not important
        )
        self.mat_conv = nn.ModuleList()
        self.mat_trans = nn.ModuleList(
            [
                MatrixTrans(
                    node_channels=config.node_channels,
                    hidden_channels=config.hidden_channels,
                    max_l=config.max_l,
                    edge_attr_dim=config.num_basis,
                    actfn=config.activation,
                )
                for _ in range(config.action_blocks)
            ]
        )
        mat_toolkit = MatToolkit(config.target_basis, config.possible_elements)
        self.output = MatrixOut(
            node_dim=config.node_dim,
            hidden_channels=config.hidden_channels,
            irreps_out=mat_toolkit.get_basis_irreps(),
            max_l=config.max_l,
            actfn=config.activation,
        )


class XQHNet(MatNetBase):
    def __init__(self, config: NetConfig) -> None:
        super().__init__(config)
        for idx in range(config.mat_conv_blocks):
            irreps_in = f"{config.node_dim}x0e" if idx == 0 else config.node_irreps
            self.mat_conv.append(
                NodewiseInteraction(
                    irreps_node_in=irreps_in,
                    irreps_node_out=config.node_irreps,
                    edge_attr_dim=config.num_basis,
                    actfn=config.activation,
                )
            )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Tuple[diagnol, off-diagnol].
        """
        at_no = data.at_no
        pos = data.pos
        edge_index = data.edge_index
        edge_index_full = data.edge_index_full
        node_feat, rbf, rsh, rbf_full = self.embed(
            at_no, pos, edge_index, edge_index_full
        )
        node_0, fii, fij = node_feat, None, None
        for pre_conv in self.mat_conv[: self.num_pre_conv]:
            node_feat = pre_conv(node_feat, rbf, rsh, edge_index)
        for mat_conv, mat_trans in zip(
            self.mat_conv[self.num_pre_conv :], self.mat_trans
        ):
            node_feat = mat_conv(node_feat, rbf, rsh, edge_index)
            fii, fij = mat_trans(node_feat, rbf_full, edge_index_full, fii, fij)
        return self.output(fii, fij, node_0, edge_index_full)


class XQHNetEle(MatNetBase):
    def __init__(self, config: NetConfig) -> None:
        super().__init__(config)
        for idx in range(config.mat_conv_blocks):
            irreps_in = f"{config.node_dim}x0e" if idx == 0 else config.node_irreps
            self.mat_conv.append(
                NodeInterWithEle(
                    irreps_node_in=irreps_in,
                    irreps_node_out=config.node_irreps,
                    edge_attr_dim=config.num_basis,
                    actfn=config.activation,
                )
            )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Tuple[diagnol, off-diagnol].
        """
        at_no = data.at_no
        pos = data.pos
        batch = data.batch
        charge = data.charge
        spin = data.spin
        edge_index = data.edge_index
        edge_index_full = data.edge_index_full
        node_feat, rbf, rsh, rbf_full = self.embed(
            at_no, pos, edge_index, edge_index_full
        )
        node_0, fii, fij = node_feat, None, None
        for pre_conv in self.mat_conv[: self.num_pre_conv]:
            node_feat = pre_conv(node_feat, rbf, rsh, edge_index, batch, charge, spin)
        for mat_conv, mat_trans in zip(
            self.mat_conv[self.num_pre_conv :], self.mat_trans
        ):
            node_feat = mat_conv(node_feat, rbf, rsh, edge_index, batch, charge, spin)
            fii, fij = mat_trans(node_feat, rbf_full, edge_index_full, fii, fij)
        return self.output(fii, fij, node_0, edge_index_full)


def resolve_model(config: NetConfig) -> nn.Module:
    if config.version.lower() in ["xpainn", "xpainn-pbc"]:
        return XPaiNN(config)
    elif config.version.lower() in ["xpainn-ele", "xpainn-ele-pbc"]:
        return XPaiNNEle(config)
    elif config.version.lower() == "xqhnet-mat":
        return XQHNet(config)
    elif config.version.lower() == "xqhnet-ele-mat":
        return XQHNetEle(config)
    elif config.version.lower() == "xpainn-ltc-pbc":
        return XPaiNNLTC(config)
    elif config.version.lower() == "painn":
        return PaiNN(config)
    else:
        raise NotImplementedError(f"Unsupported model {config.version}")
