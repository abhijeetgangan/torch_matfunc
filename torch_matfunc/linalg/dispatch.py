"""Public dispatch for matrix_exp, matrix_sqrt and matrix_log."""

import torch

from torch_matfunc.linalg import ops
from torch_matfunc.reference.spectral import SUPPORTED_DTYPES, hermitian_apply

# Accepted like torch.linalg.matrix_exp: computed in float32, cast back.
UPCAST_DTYPES = (torch.float16, torch.bfloat16)


def dispatch(A: torch.Tensor, name: str, hermitian: bool) -> torch.Tensor:
    if not isinstance(A, torch.Tensor):
        raise TypeError(f"{name} expects a torch.Tensor")
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"{name} expects a square matrix tensor of shape (..., n, n)")
    if A.dtype not in SUPPORTED_DTYPES and A.dtype not in UPCAST_DTYPES:
        raise TypeError(
            f"{name} supports float16/bfloat16/float32/float64/complex64/complex128, got {A.dtype}"
        )
    if A.shape[-1] == 0:
        # f of a 0x0 matrix is the 0x0 matrix; the pure-Torch norm would raise on zero size.
        return A.clone()
    kind = name.removeprefix("matrix_")
    if A.dtype in UPCAST_DTYPES:
        A32 = A.to(torch.float32)
        out = hermitian_apply(kind, A32) if hermitian else ops.library_apply(name, A32)
        return out.to(A.dtype)
    return hermitian_apply(kind, A) if hermitian else ops.library_apply(name, A)


def matrix_exp(A: torch.Tensor, *, hermitian: bool = False) -> torch.Tensor:
    """Matrix exponential of batched square matrices: Triton kernels on CUDA where measured
    faster, torch.linalg.matrix_exp or a pure-Torch implementation otherwise; hermitian=True
    uses batched eigh."""
    return dispatch(A, "matrix_exp", hermitian)


def matrix_sqrt(A: torch.Tensor, *, hermitian: bool = False) -> torch.Tensor:
    """Principal square root of batched square matrices; the spectrum must avoid the
    closed negative real axis. Triton on CUDA where measured faster; hermitian=True uses eigh."""
    return dispatch(A, "matrix_sqrt", hermitian)


def matrix_log(A: torch.Tensor, *, hermitian: bool = False) -> torch.Tensor:
    """Principal matrix logarithm of batched square matrices; the spectrum must avoid the
    closed negative real axis. Triton on CUDA where measured faster; hermitian=True uses eigh."""
    return dispatch(A, "matrix_log", hermitian)
