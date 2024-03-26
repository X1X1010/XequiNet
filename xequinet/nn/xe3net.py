import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from e3nn import o3

from .tp import get_feasible_tp
from .o3layer import resolve_o3norm
from .xpainn import XEmbedding


class LTCEmbeding(XEmbedding):
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
        super().__init__(
            node_dim=node_dim,
            node_irreps=node_irreps,
            embed_basis=embed_basis,
            aux_basis=aux_basis,
            num_basis=num_basis,
            rbf_kernel=rbf_kernel,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
        )
        lattice_irreps = []
        for mul, irrep in self.node_irreps:
            lattice_irreps.append((3, irrep))
        self.lattice_irreps = o3.Irreps(lattice_irreps)
        self.lattice_harm = o3.SphericalHarmonics(self.lattice_irreps, normalize=True, normalization="component")
        self.weight_lin = nn.Linear(self.int2c1e.embed_dim, self.node_num_irreps)
        self.lattice_lin = o3.Linear(self.lattice_irreps, self.node_irreps, biases=False)
        self.lattice_mul = o3.ElementwiseTensorProduct(self.node_irreps, f"{self.node_num_irreps}x0e")

    def forward(
        self,
        at_no: torch.LongTensor,
        lattice: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            `at_no`: Atomic numbers.
            `lattice`: Lattice vectors.
            `pos`: Atomic coordinates.
            `edge_index`: Edge index.
            `shifts`: Shifts.
            `batch`: Batch index.
        Returns:
            `x_scalar`: Scalar features.
            `rbf`: Values under radial basis functions.
            `fcut`: Values under cutoff function.
            `rsh`: Real spherical harmonics.
        """
        # calculate distance and relative position
        vec = pos[edge_index[0]] - pos[edge_index[1]] - shifts
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        # node linear
        x = self.int2c1e(at_no)
        x_scalar = self.node_lin(x)
        x_weight = self.weight_lin(x)
        # cell lattice, [x, y, z] -> [y, z, x]
        lattice_rsh = self.lattice_harm(lattice[:, :, [1, 2, 0]])
        lattice_rsh = torch.sum(self.lattice_lin(lattice_rsh), dim=1)
        lattice_rsh = lattice_rsh.index_select(0, batch)
        x_spherical = self.lattice_mul(lattice_rsh, x_weight)
        # calculate radial basis function
        rbf = self.rbf(dist)
        fcut = self.cutoff_fn(dist)
        # calculate spherical harmonics  [x, y, z] -> [y, z, x]
        rsh = self.sph_harm(vec[:, [1, 2, 0]])  # unit vector, normalized by component
        return x_scalar, x_spherical, rbf, fcut, rsh


class SelfMixTP(nn.Module):
    """
    Self-mix tensor product layer
    """
    def __init__(
        self,
        irreps_in: Iterable = "128x0e + 64x1o + 32x2e",
        hidden_channel: int = 32,
        norm_type: str = "nonorm",
    ) -> None:
        super().__init__()
        # initialize `Irreps`
        self.irreps_in = o3.Irreps(irreps_in)
        lmax = self.irreps_in.lmax
        irreps_hid = []
        for mul, irrep in self.irreps_in:
            irreps_hid.append((hidden_channel, irrep))
        self.irreps_hid = o3.Irreps(irreps_hid)

        # pre linear transformation
        self.lin_U = o3.Linear(self.irreps_in, self.irreps_hid, biases=False)
        self.lin_V = o3.Linear(self.irreps_in, self.irreps_hid, biases=False)
        # generate expanded `Irreps`
        irreps_mix = [(hidden_channel, (0, 1))]
        for l in range(2, 2 * lmax):
            irreps_mix.append((hidden_channel, (l, -1)))
            irreps_mix.append((hidden_channel, (l, +1)))
        irreps_mix.append((hidden_channel, (2 * lmax, 1)))
        self.irreps_mix = o3.Irreps(irreps_mix)
        # expansion to include parity
        self.irreps_out, instruct = get_feasible_tp(
            self.irreps_hid, self.irreps_hid, self.irreps_mix, "uuu",
        )
        self.tp = o3.TensorProduct(
            self.irreps_hid,
            self.irreps_hid,
            self.irreps_out,
            instruct,
            internal_weights=True,
            shared_weights=True,
        )
        self.o3norm = resolve_o3norm(norm_type, self.irreps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `x`: Input tensor.
        Returns:
            `x_out`: Output mixed tensor.
        """
        x_U = self.lin_U(x)
        x_V = self.lin_V(x)
        x_mix = self.tp(x_U, x_V)
        x_out = self.o3norm(x_mix)
        return x_out


class Sph2Cart(nn.Module):
    """
    Spherical to Cartesian tensor layer
    """
    def __init__(
        self,
        formula: str,
    ) -> None:
        super().__init__()
        self.formula = formula
        self.indices = formula.split("=")[0].replace("-", "")
        self.rtp = o3.ReducedTensorProducts(formula, **{i: "1o" for i in self.indices})
        self.rtp_irreps: o3.Irreps = self.rtp.irreps_out

    def forward(self, x_sph: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `x_sph`: Input spherical tensor.
        Returns:
            `x_cart`: Output Cartesian tensor.
        """
        Q = self.rtp.change_of_basis
        x_cart = x_sph @ Q.flatten(-len(self.indices))

        shape = list(x_sph.shape[:-1]) + list(Q.shape[1:])
        x_cart = x_cart.view(shape)
        return x_cart
    