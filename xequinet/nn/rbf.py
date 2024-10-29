import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_rbf(rbf_kernel: str, num_basis: int, cutoff: float) -> nn.Module:
    if rbf_kernel == "bessel":
        return SphericalBesselj0(num_basis, cutoff)
    elif rbf_kernel == "gaussian":
        return GaussianSmearing(num_basis, cutoff)
    elif rbf_kernel == "expbern":
        return ExponentialBernstein(num_basis, cutoff)
    else:
        raise NotImplementedError(f"rbf kernel {rbf_kernel} is not implemented")


def resolve_cutoff(cutoff_fn: str, cutoff: float, **kwargs) -> nn.Module:
    if cutoff_fn == "cosine":
        return CosineCutoff(cutoff)
    elif cutoff_fn == "polynomial":
        return PolynomialCutoff(cutoff, **kwargs)
    elif cutoff_fn == "exponential":
        return ExponentialCutoff(cutoff)
    elif cutoff_fn == "flat":
        return FlatCutoff(cutoff, **kwargs)
    else:
        raise NotImplementedError(f"cutoff function {cutoff_fn} is not implemented")


class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)


class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff: float, order: int = 3) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.order = order

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        p = self.order
        return (
            1
            - 0.5 * (p + 1) * (p + 2) * torch.pow(dist / self.cutoff, p)
            + p * (p + 2) * torch.pow(dist / self.cutoff, p + 1)
            - 0.5 * p * (p + 1) * torch.pow(dist / self.cutoff, p + 2)
        )


class ExponentialCutoff(nn.Module):
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(dist)
        dist_ = torch.where(dist < self.cutoff, dist, zeros)
        return torch.where(
            dist < self.cutoff,
            torch.exp(-(dist_**2) / ((self.cutoff - dist_) * (self.cutoff + dist_))),
            zeros,
        )


class FlatCutoff(nn.Module):
    def __init__(self, cutoff: float, offset_factor: float = 0.1) -> None:
        super().__init__()
        assert 0.0 < offset_factor < 1.0
        self.offset_factor = offset_factor
        self.inv_offset = 1.0 / offset_factor
        self.inv_cutoff = 1.0 / cutoff

    def _steep_cutoff(self, d_prime: torch.Tensor) -> torch.Tensor:
        """steep cutoff function for d_prime > 1.0 - offset_factor"""
        d_tilde = (1.0 - d_prime) * self.inv_offset
        steep_cutoff = (3.0 - 2.0 * d_tilde) * d_tilde**2
        return steep_cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:

        d_prime = dist * self.inv_cutoff  # [nedge, 1] relative distance
        return torch.where(
            d_prime < (1.0 - self.offset_factor),
            torch.ones_like(d_prime),
            self._steep_cutoff(d_prime),
        )


class GaussianSmearing(nn.Module):
    def __init__(self, num_basis: int, cutoff: float, eps=1e-8) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.eps = eps
        self.mean = torch.nn.Parameter(torch.empty((1, num_basis)))
        self.std = torch.nn.Parameter(torch.empty((1, num_basis)))
        self._init_parameters()

    def _init_parameters(self) -> None:
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

    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        freq = math.pi * torch.arange(1, num_basis + 1) / cutoff
        self.freq = torch.nn.Parameter(freq.view(1, -1))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [nedge, 1]
        coeff = math.sqrt(2 / self.cutoff)
        rbf = (
            torch.where(dist == 0, self.freq, torch.sin(self.freq * dist) / dist)
            * coeff
        )
        return rbf


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


class ExponentialBernstein(nn.Module):
    def __init__(self, num_basis: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.alpha = alpha
        self.dtype = torch.get_default_dtype()
        # buffers
        logfactorial = np.zeros((num_basis))
        for i in range(2, num_basis):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis)
        n = (num_basis - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        self.register_buffer("logc", torch.tensor(logbinomial, dtype=self.dtype))
        self.register_buffer("n", torch.tensor(n, dtype=self.dtype))
        self.register_buffer("v", torch.tensor(v, dtype=self.dtype))
        self.register_parameter(
            "_alpha", nn.Parameter(torch.tensor(1.0, dtype=self.dtype))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self._alpha, softplus_inverse(self.alpha))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self._alpha)
        x = -alpha * dist
        x = self.logc + self.n * x + self.v * torch.log(-torch.expm1(x))
        rbf = torch.exp(x)
        return rbf
