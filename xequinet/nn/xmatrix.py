from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3

from .o3layer import Invariant, EquivariantDot, Int2c1eEmbedding
from .rbf import resolve_cutoff, resolve_rbf
from ..utils import resolve_actfn



class MatMessage(nn.Module):
    """Message function for the matrix representation."""
    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1e + 32x2e",
        num_basis: int = 20,
        actfn: str = "silu",
    ):
        pass