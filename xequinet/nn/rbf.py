import math
import torch
import torch.nn as nn



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


class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)


class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff: float, order: int = 3):
        super().__init__()
        self.cutoff = cutoff
        self.order = order

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        p = self.order
        return 1 - 0.5 * (p+1) * (p+2) * torch.pow(dist/self.cutoff, p) \
             + p * (p+2) * torch.pow(dist/self.cutoff, p+1) \
             - 0.5 * p * (p+1) * torch.pow(dist/self.cutoff, p+2)


class GaussianSmearing(nn.Module):
    def __init__(self, num_basis: int, cutoff: float, eps=1e-8):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.eps = eps
        self.mean = torch.nn.Parameter(torch.empty((1, num_basis)))
        self.std = torch.nn.Parameter(torch.empty((1, num_basis)))
        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.uniform_(self.mean, 0, self.cutoff)
        torch.nn.init.uniform_(self.std, 1.0 / self.num_basis, 1)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [nedge, 1]
        std = self.std.abs() + self.eps
        coeff = 1 / (std * math.sqrt(2 * math.pi))
        rbf = coeff * torch.exp(-0.5 * ((dist - self.mean) / std) ** 2)
        return rbf


class SphericalBesselj0(nn.Module):
    """
    spherical Bessel function of the first kind.
    """
    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        freq = math.pi * torch.arange(1, num_basis + 1) / cutoff
        self.freq = torch.nn.Parameter(freq.view(1, -1))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [nedge, 1]
        coeff = math.sqrt(2 / self.cutoff)
        norm = torch.where(dist == 0, torch.tensor(1.0, device=dist.device), dist)
        rbf = coeff * torch.sin(self.freq * dist) / norm
        return rbf
