import math
from typing import Dict

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from xequinet import keys

from .basic import ResidualLayer


class ChargeEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        activation: str = "silu",
    ) -> torch.Tensor:
        super().__init__()
        self.node_dim = node_dim
        self.scale_factor = 1 / math.sqrt(node_dim)
        self.linear_q = nn.Linear(node_dim, node_dim)
        # for charge, positive or negative is different
        self.linear_k = nn.Linear(2, node_dim, bias=False)
        self.linear_v = nn.Linear(2, node_dim, bias=False)
        self.residual = ResidualLayer(
            node_dim=node_dim, n_layers=2, activation=activation
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = data[keys.BATCH]
        node_scalar = data[keys.NODE_INVARIANT]
        charge = data[keys.TOTAL_CHARGE].to(node_scalar.dtype)

        charge_ = nn.functional.relu(torch.stack([charge, -charge], dim=-1))
        charge_norm = torch.maximum(charge_, torch.ones_like(charge_))

        query = self.linear_q(node_scalar)
        key = self.linear_k(charge_ / charge_norm).index_select(0, batch)
        value = self.linear_v(charge_).index_select(0, batch)
        dot = torch.sum(query * key, dim=-1, keepdim=True)
        attn = nn.functional.softplus(dot * self.scale_factor)
        attn_sum = scatter_sum(attn, batch, dim=0).index_select(0, batch)

        charge_embed = self.residual((attn * value) / attn_sum)
        data[keys.NODE_INVARIANT] = node_scalar + charge_embed

        return data


class SpinEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        activation: str = "silu",
    ) -> torch.Tensor:
        super().__init__()
        self.node_dim = node_dim
        self.scale_factor = 1 / math.sqrt(node_dim)
        self.linear_q = nn.Linear(node_dim, node_dim)
        # spin is positive only
        self.linear_k = nn.Linear(1, node_dim, bias=False)
        self.linear_v = nn.Linear(1, node_dim, bias=False)
        self.residual = ResidualLayer(
            node_dim=node_dim, n_layers=2, activation=activation
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = data[keys.BATCH]
        node_scalar = data[keys.NODE_INVARIANT]
        spin = data[keys.TOTAL_SPIN].to(node_scalar.dtype)

        spin_norm = torch.maximum(spin, torch.ones_like(spin))

        query = self.linear_q(node_scalar)
        key = self.linear_k(spin / spin_norm).index_select(0, batch)
        value = self.linear_v(spin).index_select(0, batch)
        dot = torch.sum(query * key, dim=-1, keepdim=True)
        attn = nn.functional.softplus(dot * self.scale_factor)
        attn_sum = scatter_sum(attn, batch, dim=0).index_select(0, batch)

        spin_embed = self.residual((attn * value) / attn_sum)
        data[keys.NODE_INVARIANT] = node_scalar + spin_embed

        return data
