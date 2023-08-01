from typing import Iterable, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch_geometric.utils
from torch_scatter import scatter
import e3nn
from e3nn import o3

from .o3layer import Invariant, NormGate, ElementShifts, CGCoupler, Gate
from .rbf import GaussianSmearing, SphericalBesselj0, AuxGTO
from ..utils import resolve_o3norm, resolve_actfn


class XEmbedding(nn.Module):
    """
    eXtendable Embedding Module.
    """
    def __init__(
        self,
        embed_irreps: str | o3.Irreps | Iterable,
        node_irreps: str | o3.Irreps | Iterable,
        edge_irreps: str | o3.Irreps | Iterable,
        edge_nbasis: int = 16,
        rbf_kernel: str = "bessel",
        cutoff: float = 10.0,
        cutoff_fn: str = "polynomial",
        normtype: str = "batch",
        actfn: str = "sigmoid",
    ):
        """
        Args:
            `embed_irreps`: Irreps of input node features.
            `node_irreps`: Irreps of output node features.
            `edge_irreps`: Irreps of edge features.
            `edge_nbasis`: Number of radial basis functions.
            `rbf_kernel`: Radial basis function kernel.
            `cutoff`: Cutoff distance.
            `cutoff_fn`: Cutoff function.
            `normtype`: Type of normalization layer.
            `actfn`: Type of activation function.
        """
        super().__init__()
        embed_irreps = o3.Irreps(embed_irreps).simplify()
        node_irreps = o3.Irreps(node_irreps).simplify()
        edge_irreps = o3.Irreps(edge_irreps).simplify()
        self._ndim_in = embed_irreps.dim
        # node embedding
        self.nrbf = AuxGTO(cutoff, cutoff_fn)
        self.nrsh = o3.SphericalHarmonics(embed_irreps, normalize=True, normalization='component')
        self.rsh_conv = o3.ElementwiseTensorProduct(embed_irreps, f"{embed_irreps.num_irreps}x0e")
        filter_ir_out = [ir for _, ir in embed_irreps]
        self.selfmix = CGCoupler(embed_irreps, embed_irreps, filter_ir_out, internal_weights=True, shared_weights=True)
        self.node_mlp = nn.Sequential(
            o3.Linear(self.selfmix.irreps_out.simplify(), node_irreps),
            Gate(node_irreps, actfn),
            o3.Linear(node_irreps, node_irreps),
        )
        # batch embedding
        self.norm = resolve_o3norm(normtype, node_irreps)
        # rbf & rsh
        if rbf_kernel.lower() == 'gaussian':
            self.erbf = GaussianSmearing(edge_nbasis, cutoff, cutoff_fn)
        elif rbf_kernel.lower() == 'bessel':
            self.erbf = SphericalBesselj0(edge_nbasis, cutoff, cutoff_fn)
        else:
            raise NotImplementedError(f"Unsupported radial basis function kernel: {rbf_kernel}")
        self.ersh = o3.SphericalHarmonics(edge_irreps, normalize=True, normalization='component')

    def forward(
            self,
            x: torch.Tensor,
            pos: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            `x`: Raw node features (data.x).
            `pos`: Node coordinates (data.pos).
            `edge_index`: Edge indices (data.edge_index).
        Returns:
            `node`: Node features.
            `rbf`: Radial basis function features.
            `rsh`: Spherical harmonics features.
        """
        assert x.shape[-1] == self._ndim_in,\
            "Input node features must match the specified irreps."
        # calculate distance and vector
        pos = pos[:, [1, 2, 0]]  # [x, y, z] -> [y, z, x]
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        # node embedding
        nrsh_conv = self.rsh_conv(self.nrsh(vec), self.nrbf(dist))
        sph_x = scatter(
            x[edge_index[0]] * nrsh_conv,
            index=edge_index[1], dim=0, reduce="sum",
        )
        sph_x = self.selfmix(sph_x, sph_x)
        node = self.node_mlp(sph_x)
        node = self.norm(node)
        # rbf & rsh
        erbf = self.erbf(dist)
        ersh = self.ersh(-vec)
        return node, erbf, ersh


class TransPhormer(nn.Module):
    """
    O(3) Equivariant Graph Transformer Interaction Module.
    """
    def __init__(
        self,
        node_irreps: str | o3.Irreps | Iterable,
        edge_irreps: str | o3.Irreps | Iterable,
        edge_nbasis: int = 16,
        msg_irreps: Optional[str | o3.Irreps | Iterable] = None,
        nheads: int = 8,
        normtype: str = "layer",
        attn_dropout: float = 0.0,
        msg_dropout: float = 0.0,
    ):
        """
        Args:
            `node_irreps`: Irreps of embedded node features.
            `edge_irreps`: Irreps of edge features.
            `edge_nbasis`: Number of basis functions.
            `msg_irreps`: Irreps of messages.
            `nheads`: Number of attention heads.
            `normtype`: Type of normalization layer.
            `actfn`: Type of activation function.
            `attn_dropout`: Dropout rate of attention weights.
            `feat_dropout`: Dropout rate of node features.
        """
        super().__init__()
        node_irreps = o3.Irreps(node_irreps).simplify()
        edge_irreps = o3.Irreps(edge_irreps).simplify()
        msg_irreps = node_irreps if msg_irreps is None else o3.Irreps(msg_irreps).simplify()
        self._ndim = node_irreps.dim
        self._edim = edge_irreps.dim
        self._ebasis = edge_nbasis
        self._nheads = nheads
        self._scale = 1 / math.sqrt(msg_irreps.num_irreps)
        # normalization
        self.norm = resolve_o3norm(normtype, node_irreps)
        # linear layers for node features
        self.query_lin = o3.Linear(node_irreps, msg_irreps * nheads)
        self.src_lin = o3.Linear(node_irreps, msg_irreps)
        self.dst_lin = o3.Linear(node_irreps, msg_irreps)
        # coupling of angular mommentum
        filter_ir_out = [ir for _, ir in msg_irreps]
        self.cg_coupling = CGCoupler(
            msg_irreps, edge_irreps, filter_ir_out, trainable=True,
            shared_weights=False, internal_weights=False,
        )
        self.rbf_lin = nn.Linear(edge_nbasis, self.cg_coupling.weight_numel, bias=False)
        # linear layers for edge features
        coupled_irreps = self.cg_coupling.irreps_out.simplify()
        self.key_value_lin = o3.Linear(coupled_irreps, msg_irreps * nheads * 2)
        # messages
        self.msg_lin = o3.Linear(msg_irreps * nheads, node_irreps)
        # dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.msg_dropout = e3nn.nn.Dropout(node_irreps, msg_dropout)

    def forward(
        self,
        node: torch.Tensor,
        rbf: torch.Tensor,
        rsh: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        """
        Args:
            `node` (in): Input node features.
            `rbf`: Radial basis function features.
            `rsh`: Spherical harmonics features.
            `edge_index`: Edge indices.
        Returns:
            `node` (out): Updated node features.
        """
        assert node.shape[-1] == self._ndim, "Input node dimension must match the specified irreps."
        assert rbf.shape[-1] == self._ebasis, "Input rbf must match the specified number of basis."
        assert rsh.shape[-1] == self._edim, "Input rsh dimension must match the specified irreps."
        # norm
        node_norm = self.norm(node)
        # linear layers for node features
        # [natom, nheads, ^ndim]
        query = self.query_lin(node_norm).view(node.shape[0], self._nheads, -1)
        # [nedge, ^ndim]
        src = self.src_lin(node_norm)[edge_index[0]]
        dst = self.dst_lin(node_norm)[edge_index[1]]   
        # spherical harmonics expansion for edge features
        coupled_node = self.cg_coupling((src + dst), rsh, self.rbf_lin(rbf))
        key_value = self.key_value_lin(coupled_node)
        # [nedge, nhead * 2, ^ndim] -> [nedge, nhead, ^ndim] * 2
        key_value = key_value.view(edge_index.shape[1], self._nheads * 2, -1)
        key, value = key_value.split(self._nheads, dim=1)
        # transformer
        projection = (query[edge_index[0]] * key).sum(dim=-1, keepdim=True)
        # a[nedge, nhead, 1] x v[nedge, nhead, ^ndim] --scatter, reshape-> m[natom, nhead * ^ndim]
        attn = torch_geometric.utils.softmax(projection * self._scale, edge_index[1])
        attn = self.attn_dropout(attn)
        message = scatter(
            attn * value, edge_index[1], dim=0, dim_size=node.shape[0],
        ).view(node.shape[0], -1)
        # message passing
        message = self.msg_lin(message)
        message = self.msg_dropout(message)
        return node + message


class GraphAttention(nn.Module):
    def __init__(
        self,
        node_irreps: str | o3.Irreps | Iterable,
        node_pre_irreps: Optional[str | o3.Irreps | Iterable],
        edge_irreps: str | o3.Irreps | Iterable,
        msg_irreps: str | o3.Irreps | Iterable,
        num_basis: int = 16,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        feat_dropout: float = 0.0,
    ):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps).simplify()
        self.node_irreps = o3.Irreps(node_irreps).simplify()
        self.edge_irreps = o3.Irreps(edge_irreps).simplify()
        self.msg_irreps = o3.Irreps(msg_irreps).simplify()
        self.node_pre_irreps = self.node_irreps if node_pre_irreps is None else o3.Irreps(node_pre_irreps).simplify()
        self.num_heads = num_heads

        # layer norm
        self.layer_norm = resolve_o3norm(self.node_irreps)

        # merge src and dst
        self.merge_src = o3.Linear(self.node_irreps, self.node_pre_irreps)
        self.merge_dst = o3.Linear(self.node_irreps, self.node_pre_irreps)

        # dtp1
        # self.rsh_conv = o3.ElementwiseTensorProduct(self.edge_irreps, f"{self.edge_irreps.num_irreps}x0e")
        filter_ir_out = [ir for _, ir in self.edge_irreps]
        self.dtp1 = CGCoupler(self.node_pre_irreps, self.edge_irreps, filter_ir_out, use_weights=True)
        self.rbf_lin1 = nn.Linear(num_basis, self.dtp1.weight_numel, bias=False)

        # alpha
        attn_scalar = self.msg_irreps.count("0e")
        self.sep_alpha = o3.Linear(self.dtp1.irreps_out.simplify(), f"{attn_scalar * num_heads}x0e")
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, attn_scalar))
        torch_geometric.nn.inits.glorot(self.alpha_dot)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # value
        self.sep_value = o3.Linear(self.dtp1.irreps_out.simplify(), self.node_pre_irreps)
        self.gate = Gate(self.node_pre_irreps, "sigmoid")
        self.dtp2 = CGCoupler(self.node_pre_irreps, self.edge_irreps, filter_ir_out, use_weights=True,
            shared_weights=True, internal_weights=True)
        # self.rbf_lin2 = nn.Linear(num_basis, self.dtp2.weight_numel, bias=False)
        self.value_lin = o3.Linear(self.dtp2.irreps_out.simplify(), self.msg_irreps * num_heads)

        # post
        self.proj = o3.Linear(self.msg_irreps * num_heads, self.node_irreps)
        self.feat_dropout = e3nn.nn.Dropout(self.node_irreps, feat_dropout)

    def forward(
        self,
        node: torch.Tensor,
        edge_index: torch.LongTensor,
        rbf: torch.Tensor,
        rsh: torch.Tensor,
    ) -> torch.Tensor:
        # layer norm
        node_norm = self.layer_norm(node)
        # merge src and dst
        msg_src = self.merge_src(node_norm)
        msg_dst = self.merge_dst(node_norm)
        msg = msg_src[edge_index[0]] + msg_dst[edge_index[1]]
        # dtp1
        # edge1 = self.rsh_conv(rsh, self.rbf_lin1(rbf))
        message = self.dtp1(msg, rsh, self.rbf_lin1(rbf))
        # alpha
        alpha = self.sep_alpha(message).view(message.shape[0], self.num_heads, -1)
        alpha = self.leaky_relu(alpha)
        alpha = torch.sum((alpha * self.alpha_dot), dim=-1, keepdim=True)
        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
        alpha = self.attn_dropout(alpha)
        # value
        value = self.sep_value(message)
        value = self.gate(value)
        # edge2 = self.rsh_conv(rsh, self.rbf_lin2(rbf))
        value = self.dtp2(value, rsh)
        value = self.value_lin(value).view(value.shape[0], self.num_heads, -1)
        # inner product
        attn = alpha * value
        attn = scatter(attn, index=edge_index[1], dim=0, dim_size=node.shape[0])
        attn = attn.view(attn.shape[0], -1)
        # post
        feat = self.feat_dropout(self.proj(attn))
        node = node + feat
        return node


class NaiveSelfMix(nn.Module):
    def __init__(
        self,
        feat_irreps: str,
        normtype: str = "node",
        actfn: str = "silu",
        use_weights: bool = True,
        feat_dropout: float = 0.0,
    ):
        """
        Args:
            `feat_irreps`: Irreps of input features.
            `normtype`: Type of normalization.
            `actfn`: Activation function.
        """
        super().__init__()
        feat_irreps = o3.Irreps(feat_irreps).simplify()
        # pre-transformation
        self.feat_pre_lin = o3.Linear(feat_irreps, feat_irreps)
        self.pre_normgate = NormGate(feat_irreps, normtype, actfn, num_mlp=2)
        # coupling of angular mommentum
        filter_ir_out = [ir for _, ir in feat_irreps]
        self.cg_couping = CGCoupler(
            feat_irreps, feat_irreps, filter_ir_out,
            internal_weights=use_weights, shared_weights=use_weights
        )
        coupled_irreps = self.cg_couping.irreps_out.simplify()
        # post-transformation
        self.feat_post_lin = o3.Linear(coupled_irreps, feat_irreps)
        self.post_normgate = NormGate(feat_irreps, normtype, actfn, num_mlp=2)
        self.feat_dropout = e3nn.nn.Dropout(feat_irreps, feat_dropout)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `feat` (in): Input features.
        Returns:
            `feat` (out): Updated features.
        """
        pre_feat = self.pre_normgate(self.feat_pre_lin(feat))
        coupled_feat = self.cg_couping(feat, pre_feat)
        post_feat = self.post_normgate(self.feat_post_lin(coupled_feat))
        pre_feat = self.feat_dropout(pre_feat)
        post_feat = self.feat_dropout(post_feat)
        return feat + pre_feat + post_feat
        # return feat + post_feat


class FeedForward(nn.Module):
    def __init__(
        self,
        node_irreps: str | o3.Irreps | Iterable,
        hidden_irreps: str | o3.Irreps | Iterable,
        normtype: str = "layer",
        actfn: str = "sigmoid",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps).simplify()
        self.hidden_irreps = o3.Irreps(hidden_irreps).simplify()
        self.norm = resolve_o3norm(normtype, self.node_irreps)
        self.pre_lin = o3.Linear(self.node_irreps, self.hidden_irreps)
        self.gate = Gate(self.hidden_irreps, actfn)
        self.post_lin = o3.Linear(self.hidden_irreps, self.node_irreps)
        self.dropout = e3nn.nn.Dropout(self.node_irreps, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.post_lin(self.gate(self.pre_lin(y)))
        y = self.dropout(y)
        return x + y


class ScalarOut(nn.Module):
    """
    Scalar output layer (e.g. energy).
    """
    def __init__(
        self,
        irreps: str | o3.Irreps | Iterable,
        hidden_dim: int = 512,
        out_dim: int = 1,
        reduce_op: str = "sum",
        actfn: str = "silu",
        normtype: str = "layer",
        shifts_idx: Optional[torch.LongTensor] = None,
        shifts_value: Optional[torch.DoubleTensor] = None,
        freeze: bool = False,
    ):
        """
        Args:
            `irreps`: Input irreps.
            `out_dim`: Output dimension.
            `reduce_op`: Reduction operation.
            `shift_idx`: Index of the elements to be shifted.
            `shift_value`: Value of the elements to be shifted.
            `freeze`: Whether to freeze the shift.
        """
        super().__init__()
        irreps = o3.Irreps(irreps).simplify()
        self.reduce_op = reduce_op
        self.norm = resolve_o3norm(normtype, irreps)
        self.invariant = Invariant(irreps)
        self.out_head = nn.Sequential(
            nn.Linear(irreps.num_irreps, hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.out_head[0].bias)
        nn.init.zeros_(self.out_head[2].bias)
        self.shifts = ElementShifts(out_dim, shifts_idx, shifts_value, freeze)
        assert self.shifts.shifts.embedding_dim == out_dim, \
            "shifts dimension must match the output dimension"

    def forward(
        self,
        atom_out: torch.Tensor,
        at_no: torch.LongTensor,
        coords: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> torch.DoubleTensor:
        """
        Args:
            `atom_out`: Atomic outputs.
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates, must required grads.
            `batch_idx`: Index of the graphs in the batch.
        Returns:
            Scalar outputs.
        """
        assert len(atom_out) == len(at_no) == len(batch_idx), \
            "atom_out, at_no and batch_idx must have the same length"
        atom_out = self.norm(atom_out)
        atom_res = self.out_head(self.invariant(atom_out)) / 18.03065905448718
        ele_shifts = self.shifts(at_no)
        res = scatter(atom_res.double() + ele_shifts, batch_idx, dim=0, reduce=self.reduce_op)
        return res



class NegGradOut(nn.Module):
    """
    Negative gradient output layer (e.g. forces).
    """
    def __init__(
        self,
        irreps: str | o3.Irreps | Iterable,
        hidden_dim: int = 512,
        reduce_op: str = "sum",
        actfn: str = "silu",
        normtype: str = "layer",
        shifts_idx: Optional[torch.LongTensor] = None,
        shifts_value: Optional[torch.DoubleTensor] = None,
        freeze: bool = False,
    ):
        """
        Args:
            `irreps`: Input irreps.
            `reduce_op`: Reduction operation.
            `shift_idx`: Index of the elements to be shifted.
            `shift_value`: Value of the elements to be shifted.
            `freeze`: Whether to freeze the shift.
        """
        super().__init__()
        irreps = o3.Irreps(irreps).simplify()
        self.reduce_op = reduce_op
        self.norm = resolve_o3norm(normtype, irreps)
        self.invariant = Invariant(irreps)
        self.out_head = nn.Sequential(
            nn.Linear(irreps.num_irreps, hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.out_head[0].bias)
        nn.init.zeros_(self.out_head[2].bias)
        self.shifts = ElementShifts(1, shifts_idx, shifts_value, freeze)
        assert self.shifts.shifts.embedding_dim == 1, \
            "shifts dimension must be 1"
        
    def forward(
        self,
        atom_out: torch.Tensor,
        at_no: torch.LongTensor,
        coord: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> tuple[torch.DoubleTensor, torch.Tensor]:
        """
        Args:
            `atom_out`: Atomic outputs.
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates, must required grads.
            `batch_idx`: Index of the graphs in the batch.
        Note:
            `coord` must be in the computational graph, i.e. `atom_out` computed by `coords`.
        Returns:
            Scalar outputs.
            Negative gradients.
        """
        assert len(atom_out) == len(at_no) == len(batch_idx) == len(coord), \
            "atom_out, at_no and batch_idx must have the same length"
        atom_out = self.norm(atom_out)
        atom_res = self.out_head(self.invariant(atom_out)) / 18.03065905448718
        ele_shifts = self.shifts(at_no)
        res = scatter(atom_res.double() + ele_shifts, batch_idx, dim=0, reduce=self.reduce_op)
        grad = torch.autograd.grad(
            [atom_res.sum(),],    # requires List[torch.Tensor] in jit.sript
            [coord,],             # also requires List[torch.Tensor]
            retain_graph=True,
            create_graph=True,
        )[0]
        neg_grad = torch.zeros_like(coord)   # because the output of `autograd.grad()` is `Tuple[Optional[torch.Tensor],...]` in jit script
        if grad is not None:                 # which means the complier thinks that `neg_grad` may be `torch.Tensor` or `None`
            neg_grad = neg_grad - grad       # but neg_grad there are not allowed to be `None`
        return res, neg_grad                 # so we add this if statement to let the compiler make sure that `neg_grad` is not `None`



class VectorOut(nn.Module):
    """
    Vector output layer (e.g. dipole moments).
    """
    def __init__(
        self,
        irreps: str | o3.Irreps | Iterable,
        hidden_irreps: str | o3.Irreps | Iterable = "128x0e+64x1e+32x2e",
        reduce_op: str = "sum",
        actfn: str = "sigmoid",
        normtype: str = "layer",
    ):
        """
        Args:
            `irreps`: Input irreps.
            `reduce_op`: Reduction operation.
        """
        super().__init__()
        irreps = o3.Irreps(irreps).simplify()
        self.reduce_op = reduce_op
        self.norm = resolve_o3norm(normtype, irreps)
        self.out_head = nn.Sequential(
            o3.Linear(irreps, hidden_irreps),
            Gate(hidden_irreps, actfn),
            o3.Linear(hidden_irreps, "1x1e"),
        )

    def forward(
        self,
        atom_out: torch.Tensor,
        at_no: torch.LongTensor,
        coords: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            `atom_out`: Atomic outputs.
            `at_no`: Atomic numbers.
            `batch_idx`: Index of the graphs in the batch.
        Returns:
            Vector outputs.
        """
        assert len(atom_out) == len(at_no) == len(batch_idx), \
            "atom_out, at_no and batch_idx must have the same length"
        atom_res = self.out_head(atom_out)[:, 2, 0, 1]  # [y, z, x] -> [x, y, z]
        res = scatter(atom_res, batch_idx, dim=0, reduce=self.reduce_op)
        return res



# if __name__ == "__main__":
    # from torch_cluster import radius_graph
    # from torch_geometric.data import Data
    # torch.manual_seed(1016)
    # embed_irreps = "16x0e + 8x1e + 4x2e"
    # node_irreps = "4x0e + 2x1e"
    # edge_irreps = "4x0e + 2x1e"
    # p = torch.load("/share/home/ycchen/PyTorch/Xphormer_pyg/xphormer/interface/pm7_embedding.pt")
    # nodex = torch.stack([p['O'], p['H'], p['H']], dim=0).to(torch.float32)
    # h2o_coord = torch.Tensor([
    #     [0.00000000,  0.00000000, -0.11081188],
    #     [0.00000000, -0.78397589,  0.44324751],
    #     [0.00000000,  0.78397589,  0.44324751]
    # ])
    # rot_mat = o3.rand_matrix()
    # rot_h2o_coord = torch.einsum("ij,nj->ni", rot_mat, h2o_coord)
    # edge_index = torch.LongTensor([[0, 0, 1, 2], [1, 2, 0, 0]])
    # h2o_data = Data(x=nodex, pos=h2o_coord, edge_index=edge_index)
    # rot_h2o_data = Data(x=nodex, pos=rot_h2o_coord, edge_index=edge_index)
    # xemb = XEmbedding(
    #     embed_irreps,
    #     node_irreps,
    #     edge_irreps,
    #     edge_nbasis=16,
    #     rbf_kernel='gaussian',
    #     cutoff=0.5,
    #     cutoff_fn='cosine',
    # )
    # # xemb = torch.jit.script(xemb)
    # node, rbf, rsh = xemb(h2o_data.x, h2o_data.pos, h2o_data.edge_index)
    # rot_node, rot_rbf, rot_rsh = xemb(rot_h2o_data.x, rot_h2o_data.pos, rot_h2o_data.edge_index)
    
    # # D = o3.Irreps(edge_irreps).D_from_matrix(rot_mat)
    # # print(torch.allclose(torch.einsum("ij,nj->ni", D, rsh), rot_rsh))

    # gphormer = PhormerInteraction(node_irreps, edge_irreps, edge_nbasis=16, nheads=1)
    # # phormer = torch.jit.script(gphormer)
    # node_out = gphormer(node, rbf, rsh, h2o_data.edge_index)
    # rot_node_out = gphormer(rot_node, rot_rbf, rot_rsh, rot_h2o_data.edge_index)

    # D = o3.Irreps(edge_irreps).D_from_matrix(rot_mat)
    # print(torch.allclose(torch.einsum("ij,nj->ni", D, node_out), rot_node_out))
    # selfinteract = SelfInteraction1d(node_irreps)
    # selfinteract_compiled = torch.jit.script(selfinteract)
    # node_out2 = selfinteract_compiled(node)
    # print(node_out2)
    