from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3

from .o3layer import Invariant, EquivariantDot, Int2c1eEmbedding
from .rbf import resolve_cutoff, resolve_rbf
from ..utils import resolve_actfn, resolve_norm, resolve_o3norm



class TpEmbedding(nn.Module):
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
        self.sph_irreps = o3.Irreps([(1, ir) for _, ir in o3.Irreps(edge_irreps)])
        self.int2c1e = Int2c1eEmbedding(embed_basis, aux_basis)
        self.embed_dim = self.int2c1e.embed_dim
        self.node_lin = nn.Linear(self.embed_dim, self.node_dim)
        nn.init.zeros_(self.node_lin.bias)
        self.sph_harm = o3.SphericalHarmonics(self.sph_irreps, normalize=True, normalization="component")
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
        rbf = self.rbf(dist) * self.cutoff_fn(dist)
        # calculate spherical harmonics
        rsh = self.sph_harm(vec)  # unit vector, normalized by component
        return x_scalar, rbf, rsh, vec



class FirstMessage(nn.Module):
    """First message function for Tensor Product PaiNN"""
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
        self.hidden_dim = self.node_dim + self.edge_num_irreps
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
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim, bias=False)
        # elementwise tensor product
        self.sph_harm = o3.SphericalHarmonics(self.edge_irreps, normalize=True, normalization="component")
        self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_num_irreps}x0e")
        # normalization
        self.norm = resolve_norm(norm_type, self.node_dim)
        self.o3norm = resolve_o3norm(norm_type, self.edge_irreps)


    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        rbf: torch.Tensor,
        vec: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
            `rbf`: Radial basis functions.
            `edge_index`: Edge index.
        Returns:
            `new_scalar`: New scalar features.
            `new_spherical`: New spherical features.
        """
        scalar_in: torch.Tensor = self.norm(x_scalar)
        spherical_in: torch.Tensor = self.o3norm(x_spherical)
        scalar_out = self.scalar_mlp(scalar_in)
        filter_weight = self.rbf_lin(rbf)
        filter_out = scalar_out[edge_index[1]] * filter_weight
        
        gate_edge_spherical, message_scalar = torch.split(
            filter_out,
            [self.edge_num_irreps, self.node_dim],
            dim=-1,
        )
        whole_rsh = self.sph_harm(vec)
        message_spherical = self.rsh_conv(whole_rsh, gate_edge_spherical)

        new_scalar = scalar_in.index_add(0, edge_index[0], message_scalar)
        new_spherical = spherical_in.index_add(0, edge_index[0], message_spherical)
        # new_scalar = scalar_in + scatter(message_scalar, edge_index[0], dim=0)
        # new_spherical = spherical_in + scatter(message_spherical, edge_index[0], dim=0)

        return new_scalar, new_spherical


class TPMessage(nn.Module):
    """Message function for PaiNN"""
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
        self.sph_irreps = o3.Irreps([(1, ir) for _, ir in self.edge_irreps])
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.hidden_dim = self.node_dim + self.edge_num_irreps
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
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim, bias=False)
        # elementwise tensor product
        self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_num_irreps}x0e")
        self.fc_tp = o3.FullyConnectedTensorProduct(
            self.edge_irreps, self.sph_irreps, self.edge_irreps,
        )
        # normalization
        self.norm = resolve_norm(norm_type, self.node_dim)
        self.o3norm = resolve_o3norm(norm_type, self.edge_irreps)


    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        rbf: torch.Tensor,
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
        filter_weight = self.rbf_lin(rbf)
        filter_out = scalar_out[edge_index[1]] * filter_weight
        
        gate_state_spherical, message_scalar = torch.split(
            filter_out,
            [self.edge_num_irreps, self.node_dim],
            dim=-1,
        )
        message_spherical = self.rsh_conv(spherical_in[edge_index[1]], gate_state_spherical)
        message_spherical = self.fc_tp(message_spherical, rsh)

        new_scalar = scalar_in.index_add(0, edge_index[0], message_scalar)
        new_spherical = spherical_in.index_add(0, edge_index[0], message_spherical)
        # new_scalar = scalar_in + scatter(message_scalar, edge_index[0], dim=0)
        # new_spherical = spherical_in + scatter(message_spherical, edge_index[0], dim=0)

        return new_scalar, new_spherical
