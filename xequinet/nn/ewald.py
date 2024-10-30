from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from xequinet import keys

from .o3layer import resolve_activation, resolve_norm
from .rbf import resolve_cutoff, resolve_rbf


@torch.no_grad()
def get_k_index_product_set(num_k_x: int, num_k_y: int, num_k_z: int) -> torch.Tensor:
    """Get a box of k-lattice indices around the origin"""
    k_index_sets = (
        torch.arange(-num_k_x, num_k_x + 1),
        torch.arange(-num_k_y, num_k_y + 1),
        torch.arange(-num_k_z, num_k_z + 1),
    )
    k_index_product_set = torch.cartesian_prod(*k_index_sets)
    # cut the box in half
    k_index_product_set = k_index_product_set[k_index_product_set.shape[0] // 2 + 1 :]
    return k_index_product_set.to(torch.get_default_dtype())


@torch.no_grad()
def get_k_voxel_grid(
    k_cutoff: float,
    delta_k: float,
    num_k_basis: int,
    k_offset: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get indices for a cube of k-lattice sites containing the cutoff sphere"""
    num_k = int(k_cutoff / delta_k)
    k_index_product_set = get_k_index_product_set(num_k, num_k, num_k)

    # orthogonalize the k-space basis, norm delta k
    k_cell = torch.eye(3) * delta_k

    # translate lattice indices into k-vectors
    k_grid = torch.matmul(k_index_product_set, k_cell)

    # prune all k-vectors outside the cutoff sphere
    k_grid = k_grid[torch.square(k_grid).sum(dim=-1) < k_cutoff**2]

    # https://github.com/arthurkosmala/EwaldMP/blob/main/ocpmodels/common/utils.py#L1115
    if k_offset is None:
        k_offset = 0.1 if num_k_basis <= 48 else 0.25

    # evaluate with Gaussian RBF and polynomial envelope
    rbf = resolve_rbf("gaussian", num_basis=num_k_basis, cutoff=k_cutoff + k_offset)
    envelope = resolve_cutoff("polynomial", order=5, cutoff=k_cutoff + k_offset)
    k_grid_length = torch.linalg.norm(k_grid, dim=-1, keepdim=True)
    k_rbf_values = rbf(k_grid_length) * envelope(k_grid_length)

    return k_grid, k_rbf_values


class EwaldInitialPBC(nn.Module):

    k_index_product_set: torch.Tensor

    def __init__(
        self,
        num_k_points: List[int],
        projection_dim: int = 8,
    ) -> None:
        super().__init__()
        assert len(num_k_points) == 3 and any(num_k_points)
        k_index_product_set = get_k_index_product_set(*num_k_points)
        self.register_buffer("k_index_product_set", k_index_product_set)

        self.down_projection = nn.Parameter(
            torch.empty(k_index_product_set.shape[0], projection_dim)
        )
        nn.init.xavier_uniform_(self.down_projection)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        k_cell = 2 * torch.pi * torch.inverse(data[keys.CELL])
        # [n_graphs, n_k_points, 3]
        k_grid = torch.matmul(self.k_index_product_set, k_cell)
        batch = data[keys.BATCH]
        # [n_atoms, n_k_points, 3]
        k_grid = k_grid.index_select(0, batch)
        pos = data[keys.POSITIONS]
        # [n_atoms, n_k_points]
        # k_dot_r = torch.sum(k_grid * pos.unsqueeze(1), dim=-1)
        k_dot_r = torch.einsum("aki, ai -> ak", k_grid, pos)
        data[keys.K_DOT_R] = k_dot_r

        data[keys.SINC_DAMPING] = torch.tensor(1.0, device=pos.device, dtype=pos.dtype)
        # [n_k_points, down_dim]
        data[keys.DOWN_PROJECTION] = self.down_projection

        return data


class EwaldInitialNonPBC(nn.Module):
    k_grid: torch.Tensor
    k_rbf_values: torch.Tensor

    def __init__(
        self,
        k_cutoff: float,
        delta_k: float,
        num_k_basis: int,
        k_offset: Optional[float] = None,
        projection_dim: int = 8,
    ) -> None:
        super().__init__()
        k_grid, k_rbf_values = get_k_voxel_grid(
            k_cutoff=k_cutoff,
            delta_k=delta_k,
            num_k_basis=num_k_basis,
            k_offset=k_offset,
        )
        self.register_buffer("k_grid", k_grid)
        self.register_buffer("k_rbf_values", k_rbf_values)
        self.delta_k = delta_k
        self.down = nn.Linear(k_rbf_values.shape[-1], projection_dim, bias=False)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos = data[keys.POSITIONS]
        # [n_atoms, n_k_points]
        # k_dot_r = torch.sum(self.k_grid.unsqueeze(0) * pos.unsqueeze(1), dim=-1)
        k_dot_r = torch.einsum("ki, ai -> ak", self.k_grid, pos)
        data[keys.K_DOT_R] = k_dot_r

        # [n_atoms, 1]
        sinc_damping = torch.sinc(0.5 * self.delta_k * pos).prod(dim=-1, keepdim=True)
        data[keys.SINC_DAMPING] = sinc_damping
        # [n_k_points, down_dim]
        data[keys.DOWN_PROJECTION] = self.down(self.k_rbf_values)

        return data


class EwaldBlock(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        projection_dim: int = 8,
        activation: str = "silu",
        norm_type: str = "layer",
    ) -> None:
        super().__init__()

        self.norm = resolve_norm(norm_type, node_dim)
        act_fn = resolve_activation(activation)
        self.pre_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim, bias=False),
            act_fn,
            nn.Linear(node_dim, node_dim, bias=False),
            act_fn,
        )
        self.up = nn.Linear(projection_dim, node_dim, bias=False)
        # https://github.com/arthurkosmala/EwaldMP/blob/main/ocpmodels/models/ewald_block.py#L154
        with torch.no_grad():
            self.up.weight *= 0.01

        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim, bias=False),
            act_fn,
            nn.Linear(node_dim, node_dim, bias=False),
            act_fn,
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_scalar = data[keys.NODE_INVARIANT]
        k_dot_r = data[keys.K_DOT_R]  # [n_atoms, n_k_points]
        sinc_damping = data[keys.SINC_DAMPING]  # [n_atoms, 1]
        batch = data[keys.BATCH]

        # residual connection [n_atoms, node_dim]
        if keys.ATOMIC_CHARGES in data:
            atomic_charges = data[keys.ATOMIC_CHARGES]
            x_residual = x_scalar + self.pre_mlp(
                x_scalar * atomic_charges.unsqueeze(-1)
            )
        else:
            x_residual = x_scalar + self.pre_mlp(x_scalar)

        x_residual = self.norm(x_residual)
        # compute real and imaginary parts of structure factor
        # [n_atoms, n_k_points, 1]
        real_part = (torch.cos(k_dot_r) * sinc_damping).unsqueeze(-1)
        imag_part = (torch.sin(k_dot_r) * sinc_damping).unsqueeze(-1)
        # [n_graph, n_k_points, node_dim]
        sf_real = scatter_sum(
            src=real_part * x_residual.unsqueeze(1),
            index=batch,
            dim=0,
        )
        sf_imag = scatter_sum(
            src=imag_part * x_residual.unsqueeze(1),
            index=batch,
            dim=0,
        )
        # apply fourier space filter and scatter back to position space
        # [1, n_k_points, node_dim]
        kfilter = self.up(data[keys.DOWN_PROJECTION]).unsqueeze(0)
        # [n_atoms, n_k_points, node_dim]
        filter_real = torch.index_select(kfilter * sf_real, 0, batch)
        filter_imag = torch.index_select(kfilter * sf_imag, 0, batch)
        # [n_atoms, node_dim]
        ewald_message = torch.sum(
            input=filter_real * real_part + filter_imag * imag_part,
            dim=1,
        )
        x_scalar = x_scalar.add(ewald_message)

        # residual update
        data[keys.NODE_INVARIANT] = x_scalar + self.update_mlp(x_scalar)

        return data
