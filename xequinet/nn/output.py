from typing import Iterable, Union

import math

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter
from e3nn import o3

from .o3layer import Gate, resolve_actfn
from .tensorproduct import get_feasible_tp
from ..utils.config import NetConfig


class ScalarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 1,
        actfn: str = "silu",
        node_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `out_dim`: Output dimension.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        nn.init.zeros_(self.out_mlp[0].bias)
        nn.init.constant_(self.out_mlp[2].bias, node_bias)
    
    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
        Returns:
            `res`: Scalar output.
        """
        batch = data.batch
        atom_out = self.out_mlp(x_scalar)
        res = scatter(atom_out, batch, dim=0)
        return res


class NegGradOut(ScalarOut):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        node_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
        """
        super().__init__(node_dim, hidden_dim, 1, actfn, node_bias)

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
        Returns:
            `res`: Scalar output.
            `neg_grad`: Negative gradient.
        """
        batch = data.batch; coord = data.pos
        atom_out = self.out_mlp(x_scalar)
        res =  scatter(atom_out, batch, dim=0)
        grad = torch.autograd.grad(
            [atom_out.sum(),],
            [coord,],
            retain_graph=True,
            create_graph=True,
        )[0]
        return res, -grad


class VectorOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "32x1o",
        output_dim: int = 3,
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `output_dim`: Output dimension. (3 for vector and 1 for norm of the vector)
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps),
            Gate(self.hidden_irreps),
            o3.Linear(self.hidden_irreps, "1x1o"),
        )
        if output_dim != 3 and output_dim != 1:
            raise ValueError(f"output dimension must be either 1 or 3, but got {output_dim}")
        self.output_dim = output_dim

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
        Returns:
            `res`: Vector output.
        """
        batch = data.batch
        spherical_out = self.spherical_out_mlp(x_spherical)[:, [2, 0, 1]]  # [y, z, x] -> [x, y, z]
        scalar_out = self.scalar_out_mlp(x_scalar)
        atom_out = spherical_out * scalar_out
        res = scatter(atom_out, batch, dim=0)
        if self.output_dim == 1:
            res = torch.linalg.norm(res, dim=-1, keepdim=True)
        return res


class PolarOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "64x0e + 16x2e",
        output_dim: int = 9,
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `hidden_dim`: Hidden dimension.
            `hidden_irreps`: Hidden irreps.
            `output_dim`: Output dimension. (9 for 3x3 matrix and 1 for trace of the matrix)
            `actfn`: Activation function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = hidden_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)
        self.spherical_out_mlp = nn.Sequential(
            o3.Linear(self.edge_irreps, self.hidden_irreps, biases=True),
            Gate(self.hidden_irreps),
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
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
        Returns:
            `res`: Polarizability.
        """
        batch = data.batch
        spherical_out = self.spherical_out_mlp(x_spherical)
        scalar_out = self.scalar_out_mlp(x_scalar)
        atom_out = self.rsh_conv(spherical_out, scalar_out)
        mol_out = scatter(atom_out, batch, dim=0)
        zero_order = mol_out[:, 0:1]
        second_order = mol_out[:, 1:6]
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
        res = zero_out + second_out
        if self.output_dim == 1:
            res = torch.diagonal(res, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3
        return res


class SpatialOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Hidden dimension.
            `actfn`: Activation function type.
        """
        super().__init__()
        from ..utils.qc import ATOM_MASS
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.register_buffer("masses", ATOM_MASS)
        self.scalar_out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scalar_out_mlp[0].bias)
        nn.init.zeros_(self.scalar_out_mlp[2].bias)

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
        Returns:
            `res`: Spatial output.
        """
        batch = data.batch; coord = data.pos; at_no = data.at_no
        masses = self.masses[at_no]
        centroids = scatter(masses * coord, batch, dim=0) / scatter(masses, batch, dim=0)
        coord -= centroids[batch]
        scalar_out = self.scalar_out_mlp(x_scalar)
        spatial = torch.square(coord).sum(dim=1, keepdim=True)
        res = scatter(scalar_out * spatial, batch, dim=0)
        return res


class AtomicCharge(ScalarOut):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
        node_bias: float = 0.0,
    ):
        """
        Args:
            `node_dim`: Node dimension.
            `hidden_dim`: Dimension of hidden layer.
            `actfn`: Activation function type.
            `node_bias`: Bias for atomic wise output.
        """
        super().__init__(node_dim, hidden_dim, 1, actfn, node_bias)

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features. (Unused in this module)
        Returns:
            `res`: Atomic charges.
        """
        charge = data.charge; batch = data.batch
        tot_chrg = charge.index_select(0, batch)
        atom_out = self.out_mlp(x_scalar).squeeze()
        # force the sum of atomic charge equals to the total charge
        sum_chrg = scatter(atom_out, data.batch)
        num_atoms = scatter(torch.ones_like(atom_out), batch)
        at_chrgs = atom_out + (tot_chrg - sum_chrg) / num_atoms
        return at_chrgs        


class CartTensorOut(nn.Module):
    """
    Output Module for order n tensor via tensor product of input hidden irreps and cartesian transform. 
    """
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "16x0e + 16x1o + 16x2e",
        order: int = 2,
        required_symmetry: str = "ij",
        output_dim: int = 9,
        actfn: str = "silu",
    ):
        super().__init__()
        self.irreps_in = edge_irreps if isinstance(edge_irreps, o3.Irreps) else o3.Irreps(edge_irreps)
        self.irreps_hidden = hidden_irreps if isinstance(hidden_irreps, o3.Irreps) else o3.Irreps(hidden_irreps)
        lmin = min(self.irreps_in.lmax, self.irreps_hidden.lmax)
        assert lmin * 2 >= order, f"Input irreps must have 2*lmax >= order, got lmax={lmin} and order={order}."
        self.trace_out = output_dim == 1 and order == 2 
        self.indices = required_symmetry.split("=")[0].replace("-", "")
        # determine various metadata
        if output_dim == 3**(order) or self.trace_out:
            rtp = o3.ReducedTensorProducts(formula=required_symmetry, **{i: "1o" for i in self.indices}) 
        elif output_dim % 2 == 1:
            required_l = output_dim // 2
            filter_ir_out = [o3.Irrep((required_l, (-1)**required_l))] 
            rtp = o3.ReducedTensorProducts(formula=required_symmetry, filter_ir_out=filter_ir_out, **{i: "1o" for i in self.indices})
        else:
            raise NotImplementedError(f"Current CartTensorOut module does not support the required output dim={output_dim}, please check.")
        self.rtp = rtp 
        self.irreps_out = self.rtp.irreps_out 
        self.activation = resolve_actfn(actfn)
        # tensor product path 
        irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_hidden, self.irreps_hidden, self.irreps_out, "uuw",
        )
        self.tp = o3.TensorProduct(
            self.irreps_hidden,
            self.irreps_hidden,
            irreps_tp_out,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )
        # gate block 
        self.lin_out_pre = o3.Linear(self.irreps_in, self.irreps_hidden, biases=False) 
        self.gate_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim, bias=True),
            self.activation,
            nn.Linear(hidden_dim, self.tp.weight_numel, bias=True)
        )
        if irreps_tp_out == self.irreps_out:
            self.post_trans = False 
        else:
            self.post_trans = True
            self.lin_out_post = o3.Linear(irreps_tp_out, self.irreps_out, biases=False) 
        self.register_buffer("cartesian_index", torch.LongTensor([2, 0 ,1]))

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `data`: pyg `Data` object.
            `x_scalar`: Scalar features.
            `x_spherical`: Spherical features.
        Returns:
            `res`: high-order `Tensor` output.
        """
        batch = data.batch
        x_sph = self.lin_out_pre(x_spherical)
        tp_weights = self.gate_mlp(x_scalar)
        x_tp = self.tp(x_sph, x_sph, tp_weights) 
        atom_out = self.lin_out_post(x_tp) if self.post_trans else x_tp
        res_sph = scatter(atom_out, batch, dim=0)
        # cartesian transform 
        Q = self.rtp.change_of_basis
        res_cart = res_sph @ Q.flatten(-len(self.indices))
        shape = list(res_sph.shape[:-1]) + list(Q.shape[1:])
        res_cart = res_cart.view(shape)
        if self.trace_out:
            res = torch.diagonal(res_cart, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3
            return res
        else:
            # permute (y, z, x) to (x, y, z)
            for i_dim in range(1, len(self.indices)+1):
                res_cart = torch.index_select(res_cart, dim=-i_dim, index=self.cartesian_index)
            return res_cart 


class CSCOut(nn.Module):
    """
    Output Module for chemical shielding tensors as well as general atom-wise order-2 tensor.
    """
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        hidden_dim: int = 64,
        hidden_irreps: Union[str, o3.Irreps, Iterable] = "16x0e + 16x1o + 16x2e",
        required_symmetry: str = "ij",
        trace_out: bool = False,
        actfn: str = "silu",
    ):
        super().__init__()
        self.irreps_in = edge_irreps if isinstance(edge_irreps, o3.Irreps) else o3.Irreps(edge_irreps)
        self.irreps_hidden = hidden_irreps if isinstance(hidden_irreps, o3.Irreps) else o3.Irreps(hidden_irreps)
        self.trace_out = trace_out 
        self.indices = required_symmetry.split("=")[0].replace("-", "")
        assert len(self.indices) == 2, "Only 2-order tensor is supported."
        self.rtp = o3.ReducedTensorProducts(formula=required_symmetry, **{i: "1o" for i in self.indices}) 
        self.irreps_out = self.rtp.irreps_out 
        self.activation = resolve_actfn(actfn)
        # tensor product path 
        irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_hidden, self.irreps_hidden, self.irreps_out, "uuw",
        )
        self.tp = o3.TensorProduct(
            self.irreps_hidden,
            self.irreps_hidden,
            irreps_tp_out,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )
        # gate block 
        self.lin_out_pre = o3.Linear(self.irreps_in, self.irreps_hidden, biases=False) 
        self.gate_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim, bias=True),
            self.activation,
            nn.Linear(hidden_dim, self.tp.weight_numel, bias=True)
        )
        if irreps_tp_out == self.irreps_out:
            self.post_trans = False 
        else:
            self.post_trans = True
            self.lin_out_post = o3.Linear(irreps_tp_out, self.irreps_out, biases=False) 
        self.register_buffer("cartesian_index", torch.LongTensor([2, 0 ,1]))

    def forward(
        self,
        data: Data,
        x_scalar: torch.Tensor,
        x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        x_sph = self.lin_out_pre(x_spherical)
        tp_weights = self.gate_mlp(x_scalar)
        x_tp = self.tp(x_sph, x_sph, tp_weights) 
        atom_out = self.lin_out_post(x_tp) if self.post_trans else x_tp 
        
        # cartesian transform 
        Q = self.rtp.change_of_basis
        res_cart = atom_out @ Q.flatten(-len(self.indices))
        shape = list(atom_out.shape[:-1]) + list(Q.shape[1:])
        res_cart = res_cart.view(shape)
        if self.trace_out:
            res = torch.diagonal(res_cart, dim1=-2, dim2=-1).sum(dim=-1) / 3
            return res
        else:
            # permute (y, z, x) to (x, y, z)
            for i_dim in range(1, 3):
                res_cart = torch.index_select(res_cart, dim=-i_dim, index=self.cartesian_index)
            return res_cart 


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
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            hidden_dim=config.hidden_dim,
            hidden_irreps=config.hidden_irreps,
            output_dim=config.output_dim,
            actfn=config.activation,
        )
    elif config.output_mode == "polar":
        return PolarOut(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            hidden_dim=config.hidden_dim,
            hidden_irreps=config.hidden_irreps,
            output_dim=config.output_dim,
            actfn=config.activation,
        )
    elif config.output_mode == "spatial":
        return SpatialOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
        )
    elif config.output_mode == "atomic_charge":
        return AtomicCharge(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
            node_bias=config.node_average,
        )
    elif config.output_mode == "cart_tensor":
        return CartTensorOut(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            hidden_dim=config.hidden_dim,
            hidden_irreps=config.hidden_irreps,
            order=config.order,
            required_symmetry=config.required_symm,
            output_dim=config.output_dim,
            actfn=config.activation,
        )
    elif config.output_mode == "atomic_2_tensor":
        return CSCOut(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            hidden_dim=config.hidden_dim,
            hidden_irreps=config.hidden_irreps,
            required_symmetry=config.required_symm,
            trace_out=config.output_dim==1,
            actfn=config.activation,
        )
    else:
        raise NotImplementedError(f"output mode {config.output_mode} is not implemented")
