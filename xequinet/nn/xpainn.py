from typing import Iterable, Tuple, Union

import math

import torch
import torch.nn as nn
import torch_geometric.utils
from torch_scatter import scatter
from e3nn import o3

from .o3layer import Invariant, EquivariantDot, Gate, Int2c1eEmbedding
from .rbf import GaussianSmearing, SphericalBesselj0, CosineCutoff, PolynomialCutoff
from ..utils import resolve_actfn


def resolve_rbf(rbf_kernel: str, num_basis: int, cutoff: float):
    if rbf_kernel == "bessel":
        return SphericalBesselj0(num_basis, cutoff)
    elif rbf_kernel == "gaussian":
        return GaussianSmearing(num_basis, cutoff)
    else:
        raise NotImplementedError(f"rbf kernel {rbf_kernel} is not implemented")


def resolve_cutoff(cutoff_fn: str, cutoff: float):
    if cutoff_fn == "cosine":
        return CosineCutoff(cutoff)
    elif cutoff_fn == "polynomial":
        return PolynomialCutoff(cutoff)
    else:
        raise NotImplementedError(f"cutoff function {cutoff_fn} is not implemented")


class XEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        embed_basis: str = "gfn2-xtb",
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
        # self.embedding = nn.Embedding(119, 28)
        # self.embed_dim = self.embedding.embedding_dim
        self.int2c1e = Int2c1eEmbedding(embed_basis)
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
        # x = self.embedding(at_no)
        x = self.int2c1e(at_no)
        x_scalar = self.node_lin(x)
        # calculate radial basis function
        rbf = self.rbf(dist)
        fcut = self.cutoff_fn(dist)
        # calculate spherical harmonics
        rsh = self.sph_harm(vec)  # unit vector, normalized by component
        return x_scalar, rbf, fcut, rsh



class PainnMessage(nn.Module):
    """Message function for PaiNN"""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        num_basis: int = 20,
        actfn: str = "silu",
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
        self.rbf_lin = nn.Linear(self.num_basis, self.hidden_dim)
        nn.init.zeros_(self.rbf_lin.bias)
        # elementwise tensor product
        self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_num_irreps}x0e")

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
        scalar_out = self.scalar_mlp(x_scalar)
        filter_weight = self.rbf_lin(rbf) * fcut
        filter_out = scalar_out[edge_index[1]] * filter_weight
        
        gate_state_spherical, gate_edge_spherical, message_scalar = torch.split(
            filter_out,
            [self.edge_num_irreps, self.edge_num_irreps, self.node_dim],
            dim=-1,
        )
        message_spherical = self.rsh_conv(x_spherical[edge_index[1]], gate_state_spherical)
        edge_spherical = self.rsh_conv(rsh, gate_edge_spherical)
        message_spherical = message_spherical + edge_spherical

        new_scalar = x_scalar.index_add(0, edge_index[0], message_scalar)
        new_spherical = x_spherical.index_add(0, edge_index[0], message_spherical)
        # new_scalar = x_scalar + scatter(message_scalar, edge_index[0], dim=0)
        # new_spherical = x_spherical + scatter(message_spherical, edge_index[0], dim=0)

        return new_scalar, new_spherical



class PainnUpdate(nn.Module):
    """Update function for PaiNN"""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        actfn: str = "silu",
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
        U_spherical = self.update_U(x_spherical)
        V_spherical = self.update_V(x_spherical)

        V_invariant = self.invariant(V_spherical)
        mlp_in = torch.cat([x_scalar, V_invariant], dim=-1)
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


class ScalarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 1,
        actfn: str = "silu",
        reduce_op: str = "sum",
        node_bias: float = 0.0,
        graph_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `out_dim`: Output dimension.
            `actfn`: Activation function type.
            `reduce_op`: Reduce operation.
            `node_bias`: Bias for atomic wise output.
            `graph_bias`: Bias for graphic output.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.reduce_op = reduce_op
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        nn.init.zeros_(self.out_mlp[0].bias)
        nn.init.zeros_(self.out_mlp[2].bias)
        self.register_buffer("node_bias", torch.tensor(node_bias))
        self.register_buffer("graph_bias", torch.tensor(graph_bias))
    
    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates. (Unused in this module)
            `batch_idx`: Batch index.
        Returns:
            `res`: Scalar output.
        """
        atom_out = self.out_mlp(x_scalar) + self.node_bias
        res = scatter(atom_out, batch_idx, dim=0, reduce=self.reduce_op)
        # num_mol = int(batch_idx.max().item() + 1)
        # zero_res = torch.zeros(
        #     (num_mol, self.out_dim),
        #     dtype=atom_out.dtype, device=atom_out.device,
        # )
        # res = zero_res.index_add(0, batch_idx, atom_out)
        return res + self.graph_bias


class NegGradOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        reduce_op: str = "sum",
        node_bias: float = 0.0,
        graph_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `actfn`: Activation function type.
            `reduce_op`: Reduce operation.
            `node_bias`: Bias for atomic wise output.
            `graph_bias`: Bias for graphic output.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.reduce_op = reduce_op
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.out_mlp[0].bias)
        nn.init.zeros_(self.out_mlp[2].bias)
        self.register_buffer("node_bias", torch.tensor(node_bias))
        self.register_buffer("graph_bias", torch.tensor(graph_bias))

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates.
            `batch_idx`: Batch index.
        Returns:
            `res`: Scalar output.
            `neg_grad`: Negative gradient.
        """
        atom_out = self.out_mlp(x_scalar) + self.node_bias
        # res =  scatter(atom_out, batch_idx, dim=0, reduce=self.reduce_op)
        num_mol = int(batch_idx.max().item() + 1)
        zero_res = torch.zeros(
            (num_mol, 1),
            dtype=atom_out.dtype, device=atom_out.device,
        )
        res = zero_res.index_add(0, batch_idx, atom_out)
        grad = torch.autograd.grad(
            [res.sum(),],
            [coord,],
            retain_graph=True,
            create_graph=True,
        )[0]
        neg_grad = torch.zeros_like(coord)      # because the output of `autograd.grad()` is `Tuple[Optional[torch.Tensor],...]` in jit script
        if grad is not None:                    # which means the complier thinks that `neg_grad` may be `torch.Tensor` or `None`
            neg_grad = neg_grad - grad          # but neg_grad there are not allowed to be `None`
        return res + self.graph_bias, neg_grad  # so we add this if statement to let the compiler make sure that `neg_grad` is not `None`


class VectorOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1e",
        output_dim: int = 3,
        actfn: str = "silu",
        gatefn: str = "sigmoid",
        reduce_op: str = "sum",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `output_dim`: Output dimension. (3 for vector and 1 for norm of the vector)
            `actfn`: Activation function type.
            `gatefn`: Gate function type.
            `reduce_op`: Reduce operation.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.reduce_op = reduce_op
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps),
            Gate(self.hidden_irreps, actfn=gatefn),
            o3.Linear(self.hidden_irreps, "1x1e"),
        )
        if output_dim != 3 and output_dim != 1:
            raise ValueError(f"output dimension must be either 1 or 3, but got {output_dim}")
        self.output_dim = output_dim

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
            `coord`: Atomic coordinates.
            `batch_idx`: Batch index.
        Returns:
            `res`: Vector output.
        """
        spherical_out = self.spherical_out_mlp(x_spherical)[:, [2, 0, 1]]  # [y, z, x] -> [x, y, z]
        scalar_out = self.scalar_out_mlp(x_scalar)
        # atom_out = spherical_out + spherical_out * scalar_out
        atom_out = spherical_out * scalar_out
        res = scatter(atom_out, batch_idx, dim=0, reduce=self.reduce_op)
        # num_mol = int(batch_idx.max().item() + 1)
        # zero_res = torch.zeros(
        #     (num_mol + 1, 3),
        #     dtype=atom_out.dtype, device=atom_out.device,
        # )
        # res = zero_res.index_add(0, batch_idx, atom_out)
        if self.output_dim == 1:
            res = torch.linalg.norm(res, dim=-1, keepdim=True)
        return res


class PolarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1e",
        output_dim: int = 9,
        actfn: str = "silu",
        gatefn: str = "sigmoid",
        reduce_op: str = "sum",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `output_dim`: Output dimension. (9 for 3x3 matrix and 1 for trace of the matrix)
            `actfn`: Activation function type.
            `gatefn`: Gate function type.
            `reduce_op`: Reduce operation.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.reduce_op = reduce_op
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps, biases=True),
            Gate(self.hidden_irreps, actfn=gatefn),
            o3.Linear(self.hidden_irreps, "1x0e + 1x2e", biases=True),
        )
        nn.init.zeros_(self.spherical_out_mlp[0].bias)
        nn.init.zeros_(self.spherical_out_mlp[2].bias)
        self.rsh_conv = o3.ElementwiseTensorProduct("1x0e + 1x2e", "2x0e")
        if output_dim != 9 and output_dim != 1:
            raise ValueError(f"output dimension must be either 1 or 9, but got {output_dim}")
        self.output_dim = output_dim

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
            `coord`: Atomic coordinates.
            `batch_idx`: Batch index.
        Returns:
            `res`: Polarizability.
        """
        spherical_out = self.spherical_out_mlp(x_spherical)
        scalar_out = self.scalar_out_mlp(x_scalar)
        atom_out = self.rsh_conv(spherical_out, scalar_out)
        zero_order = atom_out[:, [0]]
        second_order = atom_out[:, 1:6]
        # build zero order output
        zero_out = torch.diag_embed(torch.repeat_interleave(zero_order, 3, dim=-1))
        # build second order output
        second_out = torch.empty_like(zero_out)
        d_norm = torch.linalg.norm(second_order, dim=-1)
        dxy = second_order[:, 0]; dyz = second_order[:, 1]
        dz2 = second_order[:, 2]
        dzx = second_order[:, 3]; dx2_y2 = second_order[:, 4]
        second_out[:, 0, 0] = (1 / math.sqrt(3)) * (d_norm - dz2) + dx2_y2
        second_out[:, 1, 1] = (1 / math.sqrt(3)) * (d_norm - dz2) - dx2_y2
        second_out[:, 2, 2] = (1 / math.sqrt(3)) * (d_norm + 2 * dz2)
        second_out[:, 0, 1] = second_out[:, 1, 0] = dxy
        second_out[:, 1, 2] = second_out[:, 2, 1] = dyz
        second_out[:, 0, 2] = second_out[:, 2, 0] = dzx
        # add together
        res = scatter(zero_out + second_out, batch_idx, dim=0, reduce=self.reduce_op)
        # num_mol = int(batch_idx.max().item() + 1)
        # zero_res = torch.zeros(
        #     (num_mol + 1, 3, 3),
        #     dtype=atom_out.dtype, device=atom_out.device,
        # )
        # res = zero_res.index_add(0, batch_idx, atom_out)
        if self.output_dim == 1:
            res = torch.diagonal(res, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        return res


class ForceOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1e",
        actfn: str = "silu",
        gatefn: str = "sigmoid",
        reduce_op: str = "sum",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `actfn`: Activation function type.
            `gatefn`: Gate function type.
            `reduce_op`: Reduce operation.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.reduce_op = reduce_op
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps),
            Gate(self.hidden_irreps, actfn=gatefn),
            o3.Linear(self.hidden_irreps, "1x1e"),
        )

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
        coord: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
            `coord`: Atomic coordinates.
            `batch_idx`: Batch index.
        Returns:
            `res`: Vector output.
        """
        spherical_out = self.spherical_out_mlp(x_spherical)[:, [2, 0, 1]]  # [y, z, x] -> [x, y, z]
        scalar_out = self.scalar_out_mlp(x_scalar)
        # res = spherical_out + spherical_out * scalar_out
        res = spherical_out * scalar_out
        return res        


class CoulombOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        reduce_op: str = "sum",
        node_bias: float = 0.0,
        graph_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `actfn`: Activation function type.
            `reduce_op`: Reduce operation.
            `node_bias`: Bias for atomic wise output.
            `graph_bias`: Bias for graphic output.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.reduce_op = reduce_op
        self.energy_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.energy_mlp[0].bias)
        nn.init.zeros_(self.energy_mlp[2].bias)
        self.charge_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.charge_mlp[0].bias)
        nn.init.zeros_(self.charge_mlp[2].bias)
        self.register_buffer("node_bias", torch.tensor(node_bias))
        self.register_buffer("graph_bias", torch.tensor(graph_bias))

    def forward(
        self,
        x_scalar: torch.Tensor,
        edge_index: torch.Tensor,
        dist: torch.Tensor,
        mol_charge: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates.
            `batch_idx`: Batch index.
        Returns:
            `res`: Energy output with Coulomb interaction.
        """
        atom_energy_out = self.energy_mlp(x_scalar) + self.node_bias
        atom_charge_out = torch_geometric.utils.softmax(
            src=self.charge_mlp(x_scalar),
            index=batch_idx,
        ) * mol_charge.index_select(0, batch_idx)
        coulomb = atom_charge_out[edge_index[0]] * atom_charge_out[edge_index[1]] / dist
        coulomb_out = scatter(0.5 * coulomb, edge_index[0], dim=0, reduce=self.reduce_op)
        res = scatter(coulomb_out + atom_energy_out, batch_idx, dim=0, reduce=self.reduce_op)
        return res + self.graph_bias
