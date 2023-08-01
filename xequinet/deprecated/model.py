from typing import Union, Tuple
import torch
import torch.nn as nn

from ..utils import NetConfig
from .equiblk import (
    XEmbedding,
    TransPhormer,
    GraphAttention,
    NaiveSelfMix,
    FeedForward,
    ScalarOut,
    NegGradOut,
    VectorOut,
)


def resolve_embedding(config: NetConfig) -> nn.Module:
    if config.embedding == "int2c1e":
        return XEmbedding(
            config.embed_irreps,
            config.node_irreps,
            config.edge_irreps,
            config.num_basis,
            config.rbf_kernel,
            config.cutoff,
            config.cutoff_fn,
            config.ebd_norm,
            config.gate_actfn,
        )
    else:
        raise NotImplementedError(f"embedding type {config.embedding} is not implemented")


def resolve_encoder(config: NetConfig) -> nn.Module:
    if config.encoder == "transphormer":
        return TransPhormer(
            config.node_irreps,
            config.edge_irreps,
            config.num_basis,
            config.msg_irreps,
            config.attn_heads,
            config.enc_norm,
            config.attn_dropout,
            config.feat_dropout,
        )
    elif config.encoder == "graphattn":
        return GraphAttention(
            config.node_irreps,
            config.node_irreps,
            config.edge_irreps,
            config.msg_irreps,
            config.num_basis,
            config.attn_heads,
            config.attn_dropout,
            config.feat_dropout,
        )
    else:
        raise NotImplementedError(f"encoder type {config.encoder} is not implemented")


def resolve_decoder(config: NetConfig) -> nn.Module:
    if config.decoder == "selfmix":
        return NaiveSelfMix(
            config.node_irreps,
            config.dec_norm,
            config.activation,
            feat_dropout=config.feat_dropout,
        )
    elif config.decoder == "feedforward":
        return FeedForward(
            config.node_irreps,
            config.hidden_irreps,
            config.dec_norm,
            config.gate_actfn,
            config.feat_dropout,
        )
    else:
        raise NotImplementedError(f"decoder type {config.decoder} is not implemented")


def resolve_output(config: NetConfig) -> nn.Module:
    if config.output_mode == "scalar":
        return ScalarOut(
            irreps=config.node_irreps,
            out_dim=config.output_dim,
            reduce_op=config.reduce_op,
            actfn=config.activation,
            normtype=config.dec_norm,
            shifts_value=config.element_shifts,
            freeze=config.freeze_shifts,
        )
    elif config.output_mode == "grad":
        return NegGradOut(
            irreps=config.node_irreps,
            reduce_op=config.reduce_op,
            actfn=config.activation,
            normtype=config.dec_norm,
            shifts_value=config.element_shifts,
            freeze=config.freeze_shifts,
        )
    elif config.output_mode == "vector":
        return VectorOut(
            config.node_irreps,
            reduce_op=config.reduce_op,
            actfn=config.gate_actfn,
            normtype=config.dec_norm,
        )                
    else:
        raise NotImplementedError(f"output mode {config.output_mode} is not implemented")



class Xphormer(nn.Module):
    """
    Xphormer model
    """
    def __init__(self, config: NetConfig):
        """
        Args:
            `config`: network configuration
        """
        super().__init__()
        self.embedding = resolve_embedding(config)
        self.encoder = nn.ModuleList([
            resolve_encoder(config) for _ in range(config.action_blocks)
        ])
        self.decoder = nn.ModuleList([
            resolve_decoder(config) for _ in range(config.action_blocks)
        ])
        self.output = resolve_output(config)

    def forward(
        self,
        x: torch.Tensor,
        at_no: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        node, erbf, ersh = self.embedding(x, pos, edge_index)
        for enc, dec in zip(self.encoder, self.decoder):
            node = enc(node, erbf, ersh, edge_index)
            node = dec(node)
        # for enc in self.encoder:
        #     node = enc(node, erbf, ersh, edge_index)
        # for dec in self.decoder:
        #     node = dec(node)
        res = self.output(node, at_no, pos, batch)
        return res
