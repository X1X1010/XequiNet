from typing import Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .xpainn import (
    XEmbedding, PBCEmbedding,
    XPainnMessage, XPainnUpdate,
    EleEmbedding,
)
from .painn import (
    Embedding, PainnMessage, PainnUpdate,
)
from .output import (
    ScalarOut, NegGradOut, VectorOut, PolarOut, SpatialOut,
)
from .xqhnet import (
    XMatEmbedding, NodewiseInteraction, MatTrans,
)
from .xpainnorb import (
    FullEdgeKernel, XMatTrans
)
from .matlayer import MatriceOut 
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
    
    def forward(self, data: Data) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no; pos=data.pos; edge_index=data.edge_index; batch=data.batch
        # embed input
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index)
        # initialize vector with zeros
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        # message passing and node update
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        # output
        result = self.out(x_scalar, x_vector, pos, batch)
        return result


class XPaiNNEle(XPaiNN):
    def __init__(self, config: NetConfig):
        super().__init__(config)
        self.charge_ebd = nn.ModuleList([
            EleEmbedding(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])
        self.spin_ebd = nn.ModuleList([
            EleEmbedding(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])

    def forward(self, data: Data) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no; pos=data.pos; edge_index=data.edge_index; batch=data.batch
        charge = data.charge; spin = data.spin
        # embed input
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index)
        # initialize vector with zeros
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        # electron embedding, message passing and node update
        for ce, se, msg, upd in zip(self.charge_ebd, self.spin_ebd, self.message, self.update):
            x_scalar = x_scalar + ce(x_scalar, charge, batch) + se(x_scalar, spin, batch)
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        # output
        result = self.out(x_scalar, x_vector, pos, batch)
        return result


class PBCPaiNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config
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
        self.charge_ebd = nn.ModuleList([
            EleEmbedding(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])
        self.spin_ebd = nn.ModuleList([
            EleEmbedding(node_dim=config.node_dim)
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

    def forward(self, data: Data) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no; pos=data.pos; edge_index=data.edge_index; batch=data.batch
        shifts=data.shifts; charge = data.charge; spin = data.spin
        # embed input
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, shifts, edge_index)
        # initialize vector with zeros
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        # electron embedding, message passing and node update
        for ce, se, msg, upd in zip(self.charge_ebd, self.spin_ebd, self.message, self.update):
            x_scalar = x_scalar + ce(x_scalar, charge, batch) + se(x_scalar, spin, batch)
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        # output
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

    def forward(self, data: Data) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        result = self.out(x_scalar, at_no, pos, batch_idx)
        return result


class XQHNet(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        assert config.num_mat_conv >= config.action_blocks
        self.begin_read_idx = config.num_mat_conv - config.action_blocks
        self.embed = XMatEmbedding(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.mat_conv = nn.ModuleList() 
        for idx in range(config.num_mat_conv):
            irreps_in = f"{config.node_dim}x0e" if idx == 0 else config.edge_irreps 
            self.mat_conv.append(
                NodewiseInteraction(
                    irreps_node_in=irreps_in,
                    irreps_node_out=config.edge_irreps,
                    node_dim=config.node_dim,
                    edge_attr_dim=config.num_basis,
                    actfn=config.activation,
                    use_normgate=False if idx == 0 else True,
                )
            )
        self.mat_trans = nn.ModuleList([
            MatTrans(
                node_dim=config.node_dim,
                hidden_dim=config.mat_hidden_dim,
                max_l=config.max_l,
                edge_dim=config.num_basis,
                actfn=config.activation,
            )
            for _ in range(config.action_blocks)
        ])
        self.output = MatriceOut(
            config.irreps_out,
            node_dim=config.node_dim,
            hidden_dim=config.mat_hidden_dim,
            block_dim=config.mat_block_dim,
            max_l=config.max_l,
            actfn=config.activation, 
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `data`: Input Data.
        Returns:
            `result`: Output.
        """
        at_no = data.at_no; pos=data.pos
        edge_index=data.edge_index; edge_index_full=data.fc_edge_index
        node_feat, rbfs, rshs, full_rbfs = self.embed(at_no, pos, edge_index, edge_index_full)
        node_0, node_sph_ten, edge_sph_ten = node_feat, None, None 
        for idx, matconv in enumerate(self.mat_conv):
            node_feat = matconv(node_feat, rbfs, rshs, edge_index)
            if idx >= self.begin_read_idx:
                node_sph_ten, edge_sph_ten = self.mat_trans[idx - self.begin_read_idx](node_feat, full_rbfs, edge_index_full, node_sph_ten, edge_sph_ten)
        result = self.output(node_sph_ten, edge_sph_ten, node_0, edge_index_full)
        
        return result
        

class XPaiNNOrb(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        assert config.num_mat_conv <= config.action_blocks
        self.begin_read_idx = config.action_blocks - config.num_mat_conv
        # xpainn block
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
        # XPaiNNOrb block
        self.edge_full_embed = FullEdgeKernel(
            rbf_kernel=config.rbf_kernel,
            num_basis=config.num_basis,
            cutoff=1000.0,
            cutoff_fn="exponential",
        )
        self.mat_transform = nn.ModuleList([
            XMatTrans(
                irreps_node=config.edge_irreps,
                hidden_dim=config.mat_hidden_dim,
                block_dim=config.mat_block_dim,
                max_l = config.max_l,
                edge_dim=config.num_basis,
                actfn=config.activation,
            )
            for _ in range(config.num_mat_conv)
        ])
        self.output = MatriceOut(
            irreps_out=config.irreps_out,
            node_dim=config.node_dim,
            hidden_dim=config.mat_hidden_dim,
            block_dim=config.mat_block_dim,
            max_l=config.max_l,
            actfn=config.activation, 
        )

    def forward(self, data:Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `data`: Input Data.
        Returns:
            `result`: Output.
        """
        # raw data 
        at_no = data.at_no; pos=data.pos
        edge_index=data.edge_index; edge_index_full=data.fc_edge_index
        # embedding
        x_scalar, rbf, fcut, rshs = self.embed(at_no, pos, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rshs.shape[1]), device=x_scalar.device)
        node_sph_ten, edge_sph_ten = None, None 
        full_rbfs = self.edge_full_embed(pos, edge_index_full)
        # message convolution & representation generation 
        for idx, (msg, upd) in enumerate(zip(self.message, self.update)):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rshs, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
            if idx >= self.begin_read_idx:
                node_sph_ten, edge_sph_ten = self.mat_transform[idx - self.begin_read_idx](x_vector, full_rbfs, edge_index_full, node_sph_ten, edge_sph_ten)
        # output 
        result = self.output(node_sph_ten, edge_sph_ten, x_scalar, edge_index_full)

        return result


def resolve_model(config: NetConfig) -> nn.Module:
    if config.version.lower() == "xpainn":
        return XPaiNN(config)
    elif config.version.lower() == "xpainn-ele":
        return XPaiNNEle(config)
    elif config.version.lower() == "xpainn-pbc":
        return PBCPaiNN(config)
    elif config.version.lower() == "painn":
        return PaiNN(config)
    elif config.version.lower() == "xqhnet-mat":
        return XQHNet(config)
    elif config.version.lower() == "xpainn-mat":
        return XPaiNNOrb(config)
    else:
        raise NotImplementedError(f"Unsupported model {config.version}")
