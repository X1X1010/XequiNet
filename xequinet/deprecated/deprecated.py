from typing import Iterable, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch_geometric.utils
from torch_scatter import scatter
import e3nn
from e3nn import o3

from .o3layer import Invariant, NormGate, ElementShifts, CGCoupler
from .rbf import GaussianSmearing, SphericalBesselj0, AuxGTO


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
        actfn: str = "silu",
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
        # rbf & rsh
        if rbf_kernel.lower() == 'gaussian':
            self.rbf = GaussianSmearing(edge_nbasis, cutoff, cutoff_fn)
        elif rbf_kernel.lower() == 'bessel':
            self.rbf = SphericalBesselj0(edge_nbasis, cutoff, cutoff_fn)
        else:
            raise NotImplementedError(f"Unsupported radial basis function kernel: {rbf_kernel}")
        self.rsh = o3.SphericalHarmonics(edge_irreps, normalize=True)
        # normalization
        self.pseduo_pre_linear = o3.Linear(embed_irreps, node_irreps, biases=True)
        self.normgate = NormGate(node_irreps, normtype, actfn, num_mlp=1)
        self.pseduo_post_linear = o3.Linear(node_irreps, node_irreps, biases=True)

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
        # rbf & rsh
        rbf = self.rbf(dist)
        rsh = self.rsh(-vec)  # normalize=True, unit vector
        # pseduo node embedding
        pseduo_x = self.pseduo_pre_linear(x)
        pseduo_x = pseduo_x + self.pseduo_post_linear(self.normgate(pseduo_x))
        return pseduo_x, rbf, rsh




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
            config.activation,
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
            config.activation,
        )
    else:
        raise NotImplementedError(f"encoder type {config.encoder} is not implemented")


def resolve_decoder(config: NetConfig) -> nn.Module:
    if config.decoder == "point-wise":
        return SelfInteraction1d(
            config.node_irreps,
            config.dec_norm,
            config.activation,
        )
    else:
        raise NotImplementedError(f"decoder type {config.decoder} is not implemented")


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
        normtype: str = "node",
        actfn: str = "silu",
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
        # linear layers for node features
        self.query_lin = o3.Linear(node_irreps, msg_irreps * nheads)
        self.src_lin = o3.Linear(node_irreps, msg_irreps)
        self.dst_lin = o3.Linear(node_irreps, msg_irreps)
        # spherical harmonics expansion for edge features
        self.rbf_lin = nn.Linear(edge_nbasis, edge_irreps.num_irreps)
        self.rsh_conv = o3.ElementwiseTensorProduct(edge_irreps, f"{edge_irreps.num_irreps}x0e")
        # coupling of angular mommentum
        filter_ir_out = [ir for _, ir in msg_irreps]
        self.cg_couping = CGCoupler(msg_irreps, edge_irreps, filter_ir_out)
        # linear layers for edge features
        coupled_irreps = self.cg_couping.irreps_out.simplify()
        self.key_value_lin = o3.Linear(coupled_irreps, msg_irreps * nheads * 2)
        # linear layers for messages
        self.msg_lin = o3.Linear(msg_irreps * nheads, node_irreps)
        # norm gate
        self.normgate = NormGate(node_irreps, normtype, actfn, num_mlp=2)
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
        # linear layers for node features
        # [natom, nheads, ^ndim]
        query = self.query_lin(node).view(node.shape[0], self._nheads, -1)
        # [nedge, ^ndim]
        src = self.src_lin(node)[edge_index[0]]
        dst = self.dst_lin(node)[edge_index[1]]   
        # spherical harmonics expansion for edge features
        edge = self.rsh_conv(rsh, self.rbf_lin(rbf))
        key_value = self.key_value_lin(self.cg_couping((src + dst), edge))
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
        message = self.normgate(self.msg_lin(message))
        message = self.msg_dropout(message)
        node = node + message
        return node


class AuxGTO(nn.Module):
    """
    Auxiliary Gaussian-type orbital (GTO) basis.
    Note that `r^l` is not included in the basis, because Y_l,m(r->) = r^l * Y_l,m(r^),
    which is included in the spherical harmonics.
    """
    def __init__(
        self,
        cutoff: float,
        cutoff_fn="polynomial",
    ):
        super().__init__()
        alpha, d = load_aux_basis()
        self.cutoff = cutoff
        if cutoff_fn == "cosine":
            self.cutoff_fn = CosineCutoff(cutoff)
        elif cutoff_fn == "polynomial":
            self.cutoff_fn = PolynomialCutoff(cutoff)
        else:
            raise NotImplementedError
        self.register_buffer("alpha", alpha.unsqueeze(0))
        self.register_buffer("d", d.unsqueeze(0))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `dist`: distances, ``(nedge, 1)``
        """
        envelope = self.cutoff_fn(dist)
        gto = self.d * torch.exp(-self.alpha * dist.pow(2))
        # return envelope
        return gto * envelope


class PseduoPhormer(nn.Module):
    """
    Graph Transformer for pseduo node feature.
    """
    def __init__(
        self,
        pseduo_irreps: str | o3.Irreps | Iterable,
        edge_irreps: str | o3.Irreps | Iterable,
        edge_nbasis: int = 16,
        msg_irreps: Optional[str | o3.Irreps | Iterable] = None,
        nheads: int = 8,
        normtype: str = "node",
        actfn: str = "silu",
        attn_dropout: float = 0.0,
        msg_dropout: float = 0.0,
    ):
        super().__init__()
        pseduo_irreps = o3.Irreps(pseduo_irreps).simplify()
        edge_irreps = o3.Irreps(edge_irreps).simplify()
        msg_irreps = pseduo_irreps if msg_irreps is None else o3.Irreps(msg_irreps).simplify()
        self._ndim = pseduo_irreps.dim
        self._edim = edge_irreps.dim
        self._ebasis = edge_nbasis
        self._nheads = nheads
        self._scale = 1 / math.sqrt(msg_irreps.dim)
        # linear layers for pseduo node features
        self.query_lin = o3.Linear(pseduo_irreps, msg_irreps * nheads)
        self.key_lin = o3.Linear(pseduo_irreps * 2, msg_irreps * nheads)
        # spherical harmonics expansion and convolution
        self.pseduo_lin = o3.Linear(pseduo_irreps * 2, edge_irreps)
        self.rbf_lin = nn.Linear(edge_nbasis, edge_irreps.num_irreps, bias=False)
        self.rsh_conv = o3.ElementwiseTensorProduct(edge_irreps, f"{edge_irreps.num_irreps}x0e")
        # self.value_lin = o3.Linear(edge_irreps, msg_irreps * nheads)
        filter_ir_out = [ir for _, ir in edge_irreps]
        self.self_coupling = CGCoupler(edge_irreps, edge_irreps, filter_ir_out)
        self.value_lin = o3.Linear(self.self_coupling.irreps_out.simplify(), msg_irreps * nheads)
        # linear layers for messages
        self.msg_lin = o3.Linear(msg_irreps * nheads, pseduo_irreps)
        # normgate
        self.normgate = NormGate(pseduo_irreps, normtype, actfn, num_mlp=2)
        # dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.msg_dropout = e3nn.nn.Dropout(pseduo_irreps, msg_dropout)

    def forward(
        self,
        pseduo_x: torch.Tensor,
        rbf: torch.Tensor,
        rsh: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        assert pseduo_x.shape[-1] == self._ndim, "Input node features must match the specified irreps."
        assert rbf.shape[-1] == self._ebasis, "Input rbf must match the specified number of basis."
        assert rsh.shape[-1] == self._edim, "Input rsh dimension must match the specified irreps."
        # linear layers for pseduo node features
        # [natom, nheads, ndim]
        pseduo_ij = torch.cat([pseduo_x[edge_index[0]], pseduo_x[edge_index[1]]], dim=-1)
        query = self.query_lin(pseduo_x).view(pseduo_x.shape[0], self._nheads, -1)
        key = self.key_lin(pseduo_ij).view(pseduo_ij.shape[0], self._nheads, -1)
        # spherical harmonics expansion and convolution for value
        edge = self.rsh_conv(rsh, self.rbf_lin(rbf))
        sph_ij = edge * self.pseduo_lin(pseduo_ij)
        # value = self.value_lin(sph_ij).view(sph_ij.shape[0], self._nheads, -1)
        mixed_sph = self.self_coupling(sph_ij, sph_ij)
        value = self.value_lin(mixed_sph).view(mixed_sph.shape[0], self._nheads, -1)
        # transformer
        projection = (query[edge_index[0]] * key).sum(dim=-1, keepdim=True)
        # a[nedge, nhead, 1] x v[nedge, nhead, ^ndim] --scatter, reshape-> m[natom, nhead * ^ndim]
        attn = torch_geometric.utils.softmax(projection * self._scale, edge_index[1])
        attn = self.attn_dropout(attn)
        message = scatter(
            attn * value, edge_index[1], dim=0, dim_size=pseduo_x.shape[0],
        ).view(pseduo_x.shape[0], -1)
        # spherical node feature
        message = self.normgate(self.msg_lin(message))
        node = self.msg_dropout(message)
        return node


class o3LayerNorm(nn.Module):
    
    def __init__(self, irreps, eps=1e-6, affine=True, normalization='component'):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps})"


    def forward(self, node_input, **kwargs):
        # batch, *size, dim = node_input.shape  # TODO: deal with batch
        # node_input = node_input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # node_input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            #field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul*d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = torch.mean(field, dim=1, keepdim=True) # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean
                
            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)    

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]
            
            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]
            
            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]
            
            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


def resolve_o3norm(
    normtype: str,
    irreps: str | o3.Irreps | Iterable,
    momentum: float = 0.1,
    affine: bool = True,
) -> nn.Module:
    """Helper function to return O(3) normalization layer"""
    normtype = normtype.lower()
    if normtype == "batch":
        return e3nn.nn.BatchNorm(
            irreps,
            momentum=momentum,
            affine=affine,
        )
    elif normtype == "layer":
        return o3LayerNorm(
            irreps,
            affine=affine,
        )
    elif normtype == "nonorm":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported O(3) normalization layer {normtype}")


def load_aux_basis(basisname="orbaux.dat"):
    """
    """
    if (BASIS_FOLDER / f"{basisname}").exists():
        basis = str(BASIS_FOLDER / f"{basisname}")
    else:
        basis = basisname
    aux = gto.load(basis, 'X')
    exp, coeff = [], []
    for cgto in aux:
        norm = calc_cgto_norm(cgto)
        exp_coeff = torch.Tensor(cgto[1:])
        exp.append(exp_coeff[:, 0])
        coeff.append(exp_coeff[:, 1] * norm)
    exp = torch.cat(exp)
    coeff = torch.cat(coeff)
    return exp, coeff


class AuxGTO(nn.Module):
    """
    Auxiliary Gaussian-type orbital (GTO) basis.
    Note that `r^l` is not included in the basis, because Y_l,m(r->) = r^l * Y_l,m(r^),
    which is included in the spherical harmonics.
    """
    def __init__(
        self,
        cutoff: float,
        cutoff_fn="polynomial",
    ):
        super().__init__()
        alpha, d = load_aux_basis()
        self.cutoff = cutoff
        if cutoff_fn == "cosine":
            self.cutoff_fn = CosineCutoff(cutoff)
        elif cutoff_fn == "polynomial":
            self.cutoff_fn = PolynomialCutoff(cutoff)
        else:
            raise NotImplementedError
        self.register_buffer("alpha", alpha.unsqueeze(0))
        self.register_buffer("d", d.unsqueeze(0))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `dist`: distances, ``(nedge, 1)``
        """
        envelope = self.cutoff_fn(dist)
        gto = self.d * torch.exp(-self.alpha * dist.pow(2))
        # return envelope
        return gto * envelope

class EvNorm1d(nn.Module):
    r"""
    The Equivariant Normalization Layer.
    https://doi.org/10.1073/pnas.2205221119
    .. math::

        \mathrm{EvNorm}(\mathbf{h})_{l,m} = \mathrm{Norm}(||\mathbf{h}_l||)
        \cdot \frac{\mathbf{h}_{l}}{ ||\mathbf{h}_{l}|| + \beta + \epsilon}

    Heuristically, the trainable :math:`\mathbf{\beta}` controls the fraction of norm
    information to be retained.
    """
    def __init__(
        self,
        irreps: str | o3.Irreps | Iterable,
        normtype="batch",
        momentum=0.1,
        eps=1e-6,
        squared=False,
    ):
        """
        Args:
            `irreps`: Input irreps.
            `normtype`: Type of norm.
            `momentum`: Momentum of the running mean and variance.
            `eps`: Epsilon for numerical stability.
            `squared`: Whether to square the norm.
        """
        super().__init__()
        irreps = o3.Irreps(irreps)
        self._nirreps = irreps.num_irreps
        self._ndim = irreps.dim
        self._eps = eps
        self.norm = resolve_norm(normtype, self._nirreps, momentum=momentum)
        # self.betainv = torch.nn.Parameter(
        #     torch.rand(self._nirreps) * 10
        # )
        self.invariant = Invariant(irreps, squared=squared)
        self.scalar_mul = o3.ElementwiseTensorProduct(irreps, f"{self._nirreps}x0e")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x`: Spherical tensor of shape ``(..., irreps.dim)``.
        Returns:
            Tuple of two tensors.
            - scalar tensor of shape ``(..., irreps.num_irreps)``
            - spherical tensor of shape ``(..., irreps.dim)``
        """
        # assert x.dim() == 2, "Input tensor must be 2D"
        assert x.shape[-1] == self._ndim, "Input tensor must have the same last dimension as the irreps"
        x0 = self.invariant(x)
        x1 = self.norm(x0)
        # beta = self.betainv.abs().add(self._eps).reciprocal()
        # devisor = x0.add(beta).abs().add(self._eps)
        devisor = x0.abs().add(self._eps)
        x2 = self.scalar_mul(x, devisor.reciprocal())
        return x1, x2

class NormGate(nn.Module):
    """
    EvNorm and MLP. The MLP is applied to the invariant part of the norm.
    https://doi.org/10.1073/pnas.2205221119
    """
    def __init__(
        self,
        irreps_in: str | o3.Irreps | Iterable,
        normtype: str = "batch",
        activation: str = "silu",
        num_mlp: int = 1,
        **kwargs,
    ):
        """
        Args:
            `irreps_in`: Input irreps.
            `normtype`: Type of norm.
            `activation`: Activation function.
            `num_mlp`: Number of MLP layers.
            `**kwargs`: Keyword arguments for EvNorm1d layer.
        """
        super().__init__()
        irreps_in = o3.Irreps(irreps_in).simplify()
        nirreps = irreps_in.num_irreps
        self._ndim = irreps_in.dim
        self.evnorm = EvNorm1d(irreps_in, normtype=normtype, **kwargs)
        self.activation = resolve_actfn(activation)
        self.mlp = nn.Sequential()
        for _ in range(num_mlp):
            self.mlp.append(nn.Linear(nirreps, nirreps))
            self.mlp.append(self.activation)
        self.scalar_mul = o3.ElementwiseTensorProduct(irreps_in, f"{nirreps}x0e")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `x`: Spherical tensor of shape ``(..., irreps.dim)``.
        Returns:
            Spherical tensor of shape ``(..., irreps.dim)``.
        """
        assert x.shape[-1] == self._ndim, "Input tensor match the specified irreps."
        x_scalar, x_spherical = self.evnorm(x)
        new_scalar = self.mlp(x_scalar)
        x_out = self.scalar_mul(x_spherical, new_scalar)
        return x_out


