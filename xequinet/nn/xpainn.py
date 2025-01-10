from typing import Dict, Iterable

import torch
import torch.nn as nn
from e3nn import o3

from xequinet import keys

from .basic import Int2c1eEmbedding, resolve_activation
from .o3layer import EquivariantDot, EquivariantLayerNorm, Invariant
from .rbf import resolve_cutoff, resolve_rbf


class XEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux56",
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "cosine",
    ) -> None:
        """
        Args:
            `embed_dim`: Embedding dimension. (default: 16s + 8p + 4d = 28)
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `num_basis`: Number of the radial basis functions.
            `rbf_kernel`: Radial basis function type.
            `cutoff`: Cutoff distance for the neighbor atoms.
            `cutoff_fn`: Cutoff function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.node_num_irreps = self.node_irreps.num_irreps
        if embed_basis == "one-hot":
            self.embedding = nn.Embedding(100, self.node_dim, padding_idx=0)
        else:
            int2c1e = Int2c1eEmbedding(embed_basis, aux_basis)
            self.embedding = nn.Sequential(
                int2c1e,
                nn.Linear(int2c1e.embed_dim, self.node_dim),
            )
            nn.init.zeros_(self.embedding[1].bias)
        self.sph_harm = o3.SphericalHarmonics(
            self.node_irreps, normalize=True, normalization="component"
        )
        self.rbf = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        atomic_numbers = data[keys.ATOMIC_NUMBERS]
        vectors = data[keys.EDGE_VECTOR]
        distances = data[keys.EDGE_LENGTH].unsqueeze(-1)

        # node linear
        node_invariant = self.embedding(atomic_numbers)
        data[keys.NODE_INVARIANT] = node_invariant

        # calculate radial basis function
        rbf = self.rbf(distances)
        fcut = self.cutoff_fn(distances)
        data[keys.RADIAL_BASIS_FUNCTION] = rbf
        data[keys.ENVELOPE_FUNCTION] = fcut
        # calculate spherical harmonics  [x, y, z] -> [y, z, x]
        rsh = self.sph_harm(
            # i.e. vectors[:, [1, 2, 0]]
            vectors.index_select(1, torch.tensor([1, 2, 0], device=vectors.device))
        )  # unit vector, normalized by component
        data[keys.SPHERICAL_HARMONICS] = rsh

        node_equivariant = torch.zeros(
            (node_invariant.shape[0], rsh.shape[1]),
            device=node_invariant.device,
        )
        data[keys.NODE_EQUIVARIANT] = node_equivariant

        return data


class XPainnMessage(nn.Module):
    """Message function for XPaiNN"""

    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        num_basis: int = 20,
        activation: str = "silu",
        layer_norm: bool = True,
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `num_basis`: Number of the radial basis functions.
            `activation`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.node_num_irreps = self.node_irreps.num_irreps
        self.hidden_dim = self.node_dim + self.node_num_irreps * 2
        self.num_basis = num_basis
        # scalar feature
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            resolve_activation(activation),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        # spherical feature
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim, bias=True)
        # elementwise tensor product
        self.rsh_conv = o3.ElementwiseTensorProduct(
            self.node_irreps, f"{self.node_num_irreps}x0e"
        )
        # normalization
        self.norm = nn.LayerNorm(self.node_dim) if layer_norm else nn.Identity()
        self.o3norm = (
            EquivariantLayerNorm(self.node_irreps) if layer_norm else nn.Identity()
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        node_scalar = self.norm(data[keys.NODE_INVARIANT])
        node_equi = self.o3norm(data[keys.NODE_EQUIVARIANT])
        rbf = data[keys.RADIAL_BASIS_FUNCTION]
        fcut = data[keys.ENVELOPE_FUNCTION]
        rsh = data[keys.SPHERICAL_HARMONICS]
        edge_index = data[keys.EDGE_INDEX]
        center_idx = edge_index[keys.CENTER_IDX]
        neighbor_idx = edge_index[keys.NEIGHBOR_IDX]

        scalar_out = self.scalar_mlp(node_scalar)
        filter_weight = self.rbf_lin(rbf) * fcut
        # i.e. scalar_out[neighbor_idx]
        filter_out = scalar_out.index_select(0, neighbor_idx) * filter_weight

        gate_state_equi, gate_edge_equi, message_scalar = torch.split(
            filter_out,
            [self.node_num_irreps, self.node_num_irreps, self.node_dim],
            dim=-1,
        )
        # i.e. node_equi[neighbor_idx]
        message_equi = self.rsh_conv(
            node_equi.index_select(0, neighbor_idx), gate_state_equi
        )
        edge_equi = self.rsh_conv(rsh, gate_edge_equi)
        message_equi = message_equi + edge_equi

        ori_scalar = data[keys.NODE_INVARIANT]
        ori_equi = data[keys.NODE_EQUIVARIANT]
        data[keys.NODE_INVARIANT] = ori_scalar.index_add(0, center_idx, message_scalar)
        data[keys.NODE_EQUIVARIANT] = ori_equi.index_add(0, center_idx, message_equi)

        return data


class XPainnUpdate(nn.Module):
    """Update function for XPaiNN"""

    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: Iterable = "128x0e + 64x1o + 32x2e",
        activation: str = "silu",
        layer_norm: bool = True,
    ) -> None:
        """
        Args:
            `node_dim`: Node dimension.
            `node_irreps`: Node irreps.
            `activation`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.node_irreps = o3.Irreps(node_irreps)
        self.node_num_irreps = self.node_irreps.num_irreps
        self.hidden_dim = self.node_dim * 2 + self.node_num_irreps
        # spherical feature
        self.update_U = o3.Linear(self.node_irreps, self.node_irreps, biases=True)
        self.update_V = o3.Linear(self.node_irreps, self.node_irreps, biases=True)
        self.invariant = Invariant(self.node_irreps)
        self.equidot = EquivariantDot(self.node_irreps)
        self.dot_lin = nn.Linear(self.node_num_irreps, self.node_dim, bias=False)
        self.rsh_conv = o3.ElementwiseTensorProduct(
            self.node_irreps, f"{self.node_num_irreps}x0e"
        )
        # scalar feature
        self.update_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.node_num_irreps, self.node_dim),
            resolve_activation(activation),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        # normalization
        self.norm = nn.LayerNorm(self.node_dim) if layer_norm else nn.Identity()
        self.o3norm = (
            EquivariantLayerNorm(self.node_irreps) if layer_norm else nn.Identity()
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        node_scalar = self.norm(data[keys.NODE_INVARIANT])
        node_equi = self.o3norm(data[keys.NODE_EQUIVARIANT])

        U_equi = self.update_U(node_equi)
        V_equi = self.update_V(node_equi)

        V_scalar = self.invariant(V_equi)
        mlp_in = torch.cat([node_scalar, V_scalar], dim=-1)
        mlp_out = self.update_mlp(mlp_in)

        a_vv, a_sv, a_ss = torch.split(
            mlp_out, [self.node_num_irreps, self.node_dim, self.node_dim], dim=-1
        )
        d_equi = self.rsh_conv(U_equi, a_vv)
        inner_prod = self.equidot(U_equi, V_equi)
        inner_prod = self.dot_lin(inner_prod)
        d_scalar = a_sv * inner_prod + a_ss

        ori_scalar = data[keys.NODE_INVARIANT]
        ori_equi = data[keys.NODE_EQUIVARIANT]
        data[keys.NODE_INVARIANT] = ori_scalar + d_scalar
        data[keys.NODE_EQUIVARIANT] = ori_equi + d_equi

        return data
