"""Helpers shared by matrix-function tests."""

import pytest
import torch

from torch_matfunc.linalg.ops import TRITON_AVAILABLE

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton path"
)
requires_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE), reason="CUDA and Triton required"
)

# 2x the worst error measured across the suite; badly conditioned tests set looser ones.
TOL_FP64 = {"rtol": 1e-11, "atol": 1e-12}
TOL_FP32 = {"rtol": 1e-5, "atol": 5e-6}


def random_spd(batch: tuple[int, ...], n: int, dtype=torch.float64, device="cpu"):
    """Random SPD / Hermitian PD matrices with spectrum off the negative real axis."""
    B = torch.randn(*batch, n, n, dtype=dtype, device=device)
    return B @ B.mH + n * torch.eye(n, dtype=dtype, device=device)


def random_nonnormal(batch: tuple[int, ...], n: int, dtype=torch.float64, device="cpu"):
    """Upper-triangular matrices with positive spectrum, non-normal when n > 1."""
    rd = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    d = 0.5 + torch.rand(*batch, n, dtype=rd, device=device)
    A = torch.diag_embed(d.to(dtype))
    fill = torch.randn(*batch, n, n, dtype=dtype, device=device)
    if not fill.is_complex():
        fill = 0.5 + fill.abs()
    triu = torch.triu(fill, diagonal=1)
    return A + triu


def random_near_identity_spd(batch: tuple[int, ...], n: int, dtype=torch.float64, device="cpu"):
    """SPD / Hermitian PD matrices close to I, so Frechet checks and gradcheck stay stable."""
    B = torch.randn(*batch, n, n, dtype=dtype, device=device) * 0.1
    I = torch.eye(n, dtype=dtype, device=device)
    return I + B @ B.mH


def tols_for(dtype):
    """The suite's measured comparison tolerances for ``dtype``."""
    if dtype in (torch.float32, torch.complex64):
        return dict(TOL_FP32)
    return dict(TOL_FP64)
