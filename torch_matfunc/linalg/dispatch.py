"""Public dispatch for matrix_exp."""

import torch

from torch_matfunc.linalg import ops
from torch_matfunc.reference.spectral import SUPPORTED_DTYPES, hermitian_apply

# Accepted like torch.linalg.matrix_exp: computed in float32, cast back.
UPCAST_DTYPES = (torch.float16, torch.bfloat16)


def matrix_exp(A: torch.Tensor, *, hermitian: bool = False) -> torch.Tensor:
    """Matrix exponential of batched square matrices: Triton kernels on CUDA at supported
    sizes, a pure-Torch implementation otherwise; hermitian=True uses batched eigh."""
    if not isinstance(A, torch.Tensor):
        raise TypeError("matrix_exp expects a torch.Tensor")
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("matrix_exp expects a square matrix tensor of shape (..., n, n)")
    if A.dtype not in SUPPORTED_DTYPES and A.dtype not in UPCAST_DTYPES:
        raise TypeError(
            f"matrix_exp supports float16/bfloat16/float32/float64/complex64/complex128,"
            f" got {A.dtype}"
        )
    if A.shape[-1] == 0:
        # exp of a 0x0 matrix is the 0x0 matrix; the pure-Torch norm would raise on zero size.
        return A.clone()
    if A.dtype in UPCAST_DTYPES:
        A32 = A.to(torch.float32)
        out = hermitian_apply(A32) if hermitian else ops.library_apply(A32)
        return out.to(A.dtype)
    return hermitian_apply(A) if hermitian else ops.library_apply(A)
