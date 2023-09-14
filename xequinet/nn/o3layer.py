from typing import Optional, Iterable, Union

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from ..utils import resolve_actfn, get_atomic_energy, get_embedding_tensor


class Invariant(nn.Module):
    """
    Invariant layer.
    """
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
        squared: bool = False,
        eps: float = 1e-6,
    ):
        """
        Args:
            `irreps_in`: Input irreps.
            `squared`: Whether to square the norm.
            `eps`: Epsilon for numerical stability.
        """
        super().__init__()
        self.squared = squared
        self.eps = eps
        self.invariant = o3.Norm(irreps_in, squared=squared)

    def forward(self, x):
        if self.squared:
            x = self.invariant(x)
        else:
            x = self.invariant(x + self.eps ** 2) - self.eps
        return x


class Gate(nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
        actfn: str = "sigmoid",
    ):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in).simplify()
        self.invariant = Invariant(irreps_in)
        self.activation = resolve_actfn(actfn)
        self.scalar_mul = o3.ElementwiseTensorProduct(irreps_in, f"{irreps_in.num_irreps}x0e")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_invariant = self.invariant(x)
        x_activation = self.activation(x_invariant)
        x_out = self.scalar_mul(x, x_activation)
        return x_out


class ElementShifts(nn.Module):
    """
    Perform elementwise shift to the outputs.
    """
    def __init__(
        self,
        atom_ref: Optional[str] = None,
        bias: float = 0.0,
        freeze: bool = False,
    ):
        """
        Args:
            `atom_ref`: Type of atomic reference.
            `freeze`: Whether to freeze the shift.
        """
        super().__init__()
        if ele_shifts is None:
            freeze = False
        ele_shifts = (get_atomic_energy(atom_ref) + bias)
        if ele_shifts.dim() == 1:
            ele_shifts = ele_shifts.unsqueeze(1)
        self.shifts = nn.Embedding.from_pretrained(ele_shifts, freeze=freeze)

    def forward(self, at_no: torch.LongTensor) -> torch.DoubleTensor:
        """
        Args:
            `at_no`: Atomic numbers
        Returns:
            Elementwise shifts
        """
        return self.shifts(at_no)


class Int2c1eEmbedding(nn.Module):
    def __init__(
        self,
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux28",
    ):
        """
        Args:
            `embed_basis`: Type of the embedding basis.
            `aux_basis`: Type of the auxiliary basis.
        """
        super().__init__()
        embed_ten = get_embedding_tensor(embed_basis, aux_basis)
        self.register_buffer("embed_ten", embed_ten)
        self.embed_dim = embed_ten.shape[1]
    
    def forward(self, at_no):
        """
        Args:
            `at_no`: Atomic numbers.
        Returns:
            Atomic features.
        """
        return self.embed_ten[at_no]


class CGCoupler(o3.TensorProduct):
    def __init__(
        self,
        irreps_in1: Union[str, o3.Irreps, Iterable],
        irreps_in2: Union[str, o3.Irreps, Iterable],
        filter_ir_out: Iterable[o3.Irrep] = None,
        irrep_normalization: str = None,
        trainable: bool = False,
        **kwargs,
    ):
        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            try:
                filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep")
        
        out = []
        instr = []
        for i, (mul, ir_in1) in enumerate(irreps_in1):
            for j, (_, ir_in2) in enumerate(irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue
                    k = len(out)
                    out.append((mul, ir_out))
                    instr.append((i, j, k, "uvu", trainable))
        out = o3.Irreps(out)
        out, p, _ = out.sort()
        instr = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instr]

        super().__init__(irreps_in1, irreps_in2, out, instr, irrep_normalization=irrep_normalization, **kwargs)


@compile_mode("trace")
class EquivariantDot(nn.Module):
    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps, Iterable],
    ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [(i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)]

        self.tp = o3.TensorProduct(irreps_in, irreps_in, irreps_out, instr, irrep_normalization="component")

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.input_dim = self.irreps_in.dim

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in})"
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        assert features1.shape[-1] == features2.shape[-1] == self.input_dim, \
            "Input tensor must have the same last dimension as the irreps"
        out = self.tp(features1, features2)
        return out