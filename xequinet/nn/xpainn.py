from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3

from .o3layer import (
    Invariant, EquivariantDot, Int2c1eEmbedding,
    resolve_actfn, resolve_norm, resolve_o3norm,
)
from .rbf import resolve_cutoff, resolve_rbf


class XEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux28",
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "polynomial",
    ):
        """
        Args:
            `embed_dim`: Embedding dimension. (default: 16s + 8p + 4d = 28)
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `num_basis`: Number of the radial basis functions.
            `rbf_kernel`: Radial basis function type.
            `cutoff`: Cutoff distance for the neighbor atoms.
            `cutoff_fn`: Cutoff function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.int2c1e = Int2c1eEmbedding(embed_basis, aux_basis)
        self.embed_dim = self.int2c1e.embed_dim
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.node_lin = nn.Linear(self.embed_dim, self.node_dim)
        nn.init.zeros_(self.node_lin.bias)
        self.sph_harm = o3.SphericalHarmonics(self.edge_irreps, normalize=True, normalization="component")
        self.rbf = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)


    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            `x`: Atomic features.
            `pos`: Atomic coordinates.
            `edge_index`: Edge index.
        Returns:
            `x_scalar`: Scalar features.
            `rbf`: Radial basis functions.
            `rsh`: Real spherical harmonics.
        """
        # calculate distance and relative position
        pos = pos[:, [1, 2, 0]]  # [x, y, z] -> [y, z, x]
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        # node linear
        x = self.int2c1e(at_no)
        x_scalar = self.node_lin(x)
        # calculate radial basis function
        rbf = self.rbf(dist)
        fcut = self.cutoff_fn(dist)
        # calculate spherical harmonics
        rsh = self.sph_harm(vec)  # unit vector, normalized by component
        return x_scalar, rbf, fcut, rsh



class XPainnMessage(nn.Module):
    """Message function for XPaiNN"""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        num_basis: int = 20,
        actfn: str = "silu",
        norm_type: str = "layer",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `num_basis`: Number of the radial basis functions.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim + self.edge_num_irreps * 2
        self.num_basis = num_basis
        # scalar feature
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            resolve_actfn(actfn),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.scalar_mlp[0].bias)
        nn.init.zeros_(self.scalar_mlp[2].bias)
        # spherical feature
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim, bias=True)
        nn.init.zeros_(self.rbf_lin.bias)
        # elementwise tensor product
        self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_num_irreps}x0e")
        # normalization
        self.norm = resolve_norm(norm_type, self.node_dim)
        self.o3norm = resolve_o3norm(norm_type, self.edge_irreps)


    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        rbf: torch.Tensor,
        fcut: torch.Tensor,
        rsh: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
            `rbf`: Radial basis functions.
            `rsh`: Real spherical harmonics.
            `edge_index`: Edge index.
        Returns:
            `new_scalar`: New scalar features.
            `new_spherical`: New spherical features.
        """
        scalar_in: torch.Tensor = self.norm(x_scalar)
        spherical_in: torch.Tensor = self.o3norm(x_spherical)
        scalar_out = self.scalar_mlp(scalar_in)
        filter_weight = self.rbf_lin(rbf) * fcut
        filter_out = scalar_out[edge_index[1]] * filter_weight
        
        gate_state_spherical, gate_edge_spherical, message_scalar = torch.split(
            filter_out,
            [self.edge_num_irreps, self.edge_num_irreps, self.node_dim],
            dim=-1,
        )
        message_spherical = self.rsh_conv(spherical_in[edge_index[1]], gate_state_spherical)
        edge_spherical = self.rsh_conv(rsh, gate_edge_spherical)
        message_spherical = message_spherical + edge_spherical

        new_scalar = x_scalar.index_add(0, edge_index[0], message_scalar)
        new_spherical = x_spherical.index_add(0, edge_index[0], message_spherical)
        # new_scalar = x_scalar + scatter(message_scalar, edge_index[0], dim=0)
        # new_spherical = x_spherical + scatter(message_spherical, edge_index[0], dim=0)

        return new_scalar, new_spherical



class XPainnUpdate(nn.Module):
    """Update function for XPaiNN"""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        actfn: str = "silu",
        norm_type: str = "layer",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim * 2 + self.edge_num_irreps
        # spherical feature
        self.update_U = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        self.update_V = o3.Linear(self.edge_irreps, self.edge_irreps, biases=True)
        self.invariant = Invariant(self.edge_irreps)
        self.equidot = EquivariantDot(self.edge_irreps)
        self.dot_lin = nn.Linear(self.edge_num_irreps, self.node_dim, bias=False)
        self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_num_irreps}x0e")
        # scalar feature
        self.update_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.edge_num_irreps, self.node_dim),
            resolve_actfn(actfn),
            nn.Linear(self.node_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.update_mlp[0].bias)
        nn.init.zeros_(self.update_mlp[2].bias)
        # normalization
        self.norm = resolve_norm(norm_type, self.node_dim)
        self.o3norm = resolve_o3norm(norm_type, self.edge_irreps)


    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
        Returns:
            `new_scalar`: New scalar features.
            `new_spherical`: New spherical features.
        """
        scalar_in: torch.Tensor = self.norm(x_scalar)
        spherical_in: torch.Tensor = self.o3norm(x_spherical)
        U_spherical = self.update_U(spherical_in)
        V_spherical = self.update_V(spherical_in)

        V_invariant = self.invariant(V_spherical)
        mlp_in = torch.cat([scalar_in, V_invariant], dim=-1)
        mlp_out = self.update_mlp(mlp_in)

        a_vv, a_sv, a_ss = torch.split(
            mlp_out,
            [self.edge_num_irreps, self.node_dim, self.node_dim],
            dim=-1
        )
        d_spherical = self.rsh_conv(U_spherical, a_vv)
        inner_prod = self.equidot(U_spherical, V_spherical)
        inner_prod = self.dot_lin(inner_prod)
        d_scalar = a_sv * inner_prod + a_ss

        return x_scalar + d_scalar, x_spherical + d_spherical
