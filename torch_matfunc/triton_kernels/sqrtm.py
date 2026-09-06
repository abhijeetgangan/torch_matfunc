"""Triton batched principal matrix square root via Denman-Beavers."""

import torch
import triton
import triton.language as tl

from torch_matfunc.triton_kernels._core import (
    launch_complex,
    launch_kernel,
    load_matrix,
    store_matrix,
)
from torch_matfunc.triton_kernels._iterations import (
    INV_TOL_FP32,
    INV_TOL_FP64,
    SCHULZ_PREFIX,
    SCHULZ_REST,
    denman_beavers,
    denman_beavers_c,
)

SUPPORTED_N = (2, 4, 8, 16, 32)
MAX_ITERS = 16

# DB steps before the single residual check: a prescaled well-conditioned sqrt converges in
# 6 to 8 steps, so checking at 7 skips the rest; ISS passes 4 since later roots start near I.
DB_PREFIX = 7


@triton.jit
def matrix_sqrt_kernel(
    A_ptr,
    Out_ptr,
    stride_b,
    stride_r,
    stride_c,
    B,
    db_prefix,
    db_rest,
    schulz_prefix,
    schulz_rest,
    inv_tol,
    N: tl.constexpr,
    TOL: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    A = load_matrix(A_ptr, pid, stride_b, stride_r, stride_c, N)
    Y = denman_beavers(A, N, db_prefix, db_rest, schulz_prefix, schulz_rest, inv_tol, TOL)
    store_matrix(Out_ptr, Y, pid, stride_b, stride_r, stride_c, N)


@triton.jit
def matrix_sqrt_kernel_c(
    Ar_ptr,
    Ai_ptr,
    Outr_ptr,
    Outi_ptr,
    stride_b,
    stride_r,
    stride_c,
    B,
    db_prefix,
    db_rest,
    schulz_prefix,
    schulz_rest,
    inv_tol,
    N: tl.constexpr,
    TOL: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    Ar = load_matrix(Ar_ptr, pid, stride_b, stride_r, stride_c, N)
    Ai = load_matrix(Ai_ptr, pid, stride_b, stride_r, stride_c, N)
    Yr, Yi = denman_beavers_c(
        Ar, Ai, N, db_prefix, db_rest, schulz_prefix, schulz_rest, inv_tol, TOL
    )
    store_matrix(Outr_ptr, Yr, pid, stride_b, stride_r, stride_c, N)
    store_matrix(Outi_ptr, Yi, pid, stride_b, stride_r, stride_c, N)


def triton_matrix_sqrt(A: torch.Tensor) -> torch.Tensor:
    """Batched principal sqrt for contiguous ``(B, N, N)`` CUDA tensors. The host prescales
    ``sqrt(A) = sqrt(c) * DB(A/c)`` with ``c = ||A||_1`` per matrix so the iteration count does
    not depend on ``||A||``; an in-kernel prescale would nest data-dependent control flow."""
    if A.ndim != 3:
        raise ValueError("triton_matrix_sqrt expects shape (B, N, N)")
    B, N, N2 = A.shape
    if N != N2:
        raise ValueError("square matrices required")
    if N not in SUPPORTED_N:
        raise ValueError(f"N={N} not in supported sizes {SUPPORTED_N}")
    if not A.is_cuda:
        raise ValueError("CUDA tensor required")
    low_prec = A.dtype in (torch.float32, torch.complex64)
    # The in-kernel skip tolerance tracks the attainable floor near n*eps instead of a fixed value.
    real_dt = torch.float32 if low_prec else torch.float64
    tol = 1.5 * N * torch.finfo(real_dt).eps
    inv_tol = INV_TOL_FP32 if low_prec else INV_TOL_FP64

    c = A.abs().sum(dim=-2).amax(dim=-1).clamp(min=torch.finfo(real_dt).tiny)[:, None, None]
    As = (A / c).contiguous()
    sqrt_c = torch.sqrt(c)

    if A.is_complex():
        out = launch_complex(
            matrix_sqrt_kernel_c,
            As,
            DB_PREFIX,
            MAX_ITERS - DB_PREFIX,
            SCHULZ_PREFIX,
            SCHULZ_REST,
            inv_tol,
            N=N,
            TOL=tol,
            num_warps=1 if N <= 8 else 4,
        )
        return out * sqrt_c
    out = torch.empty_like(As)
    sb, sr, sc = As.stride(0), As.stride(1), As.stride(2)
    launch_kernel(
        matrix_sqrt_kernel,
        (B,),
        As,
        out,
        sb,
        sr,
        sc,
        B,
        DB_PREFIX,
        MAX_ITERS - DB_PREFIX,
        SCHULZ_PREFIX,
        SCHULZ_REST,
        inv_tol,
        N=N,
        TOL=tol,
        num_warps=1 if N <= 8 else 4,
    )
    return out * sqrt_c
