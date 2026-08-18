"""Triton batched matrix exponential, one matrix per program."""

import torch
import triton
import triton.language as tl

from torch_matfunc.triton_kernels._core import (
    eye,
    launch_complex,
    launch_kernel,
    load_matrix,
    matrix_1_norm,
    matrix_1_norm_c,
    mm,
    mm_c,
    store_matrix,
)

SUPPORTED_N = (2, 4, 8, 16, 32, 64)
# The complex kernel working set exceeds consumer-GPU shared memory at n=64.
SUPPORTED_N_COMPLEX = (2, 4, 8, 16, 32)

# Bader-Blanes-Casas T18 thresholds.
THETA18_FP32 = 3.010066362817634
THETA18_FP64 = 1.090863719290036

# T18 coefficients as in reference/expm.py; jit globals must be tl.constexpr.
LN2 = tl.constexpr(0.6931471805599453)
T18_B1_A = tl.constexpr(-1.00365581030144618291e-01)
T18_B1_A2 = tl.constexpr(-8.02924648241156932449e-03)
T18_B1_A3 = tl.constexpr(-8.92138498045729985177e-04)
T18_B2_A = tl.constexpr(3.97849749499645077844e-01)
T18_B2_A2 = tl.constexpr(1.36783778460411720168e00)
T18_B2_A3 = tl.constexpr(4.98289622525382669416e-01)
T18_B2_A6 = tl.constexpr(-6.37898194594723280150e-04)
T18_B3_I = tl.constexpr(-1.09676396052962061844e01)
T18_B3_A = tl.constexpr(1.68015813878906206114e00)
T18_B3_A2 = tl.constexpr(5.71779846478865511061e-02)
T18_B3_A3 = tl.constexpr(-6.98210122488052056106e-03)
T18_B3_A6 = tl.constexpr(3.34975017086070470649e-05)
T18_B4_I = tl.constexpr(-9.04316832390810593223e-02)
T18_B4_A = tl.constexpr(-6.76404519071381882256e-02)
T18_B4_A2 = tl.constexpr(6.75961301770459654925e-02)
T18_B4_A3 = tl.constexpr(2.95552570429315521194e-02)
T18_B4_A6 = tl.constexpr(-1.39180257516060693404e-05)
T18_B5_A2 = tl.constexpr(-9.23364619367118555360e-02)
T18_B5_A3 = tl.constexpr(-1.69364939002081722752e-02)
T18_B5_A6 = tl.constexpr(-1.40086798182036094347e-05)
# Cap the squaring count; 0.5^s must stay representable.
SMAX_FP32 = 128.0
SMAX_FP64 = 1200.0


@triton.jit
def taylor18(A, N: tl.constexpr):
    """Bader-Blanes-Casas T18 in 5 matmuls, matching PyTorch ``compute_T18``."""
    I = eye(N, A.dtype)
    A2 = mm(A, A, N)
    A3 = mm(A, A2, N)
    A6 = mm(A3, A3, N)

    B1 = T18_B1_A * A + T18_B1_A2 * A2 + T18_B1_A3 * A3
    B2 = T18_B2_A * A + T18_B2_A2 * A2 + T18_B2_A3 * A3 + T18_B2_A6 * A6
    B3 = T18_B3_I * I + T18_B3_A * A + T18_B3_A2 * A2 + T18_B3_A3 * A3 + T18_B3_A6 * A6
    B4 = T18_B4_I * I + T18_B4_A * A + T18_B4_A2 * A2 + T18_B4_A3 * A3 + T18_B4_A6 * A6
    B5 = T18_B5_A2 * A2 + T18_B5_A3 * A3 + T18_B5_A6 * A6
    A9 = mm(B1, B5, N) + B4
    return B2 + mm(B3 + A9, A9, N)


@triton.jit
def matrix_exp_kernel(
    A_ptr,
    Out_ptr,
    stride_b,
    stride_r,
    stride_c,
    B,
    smax,
    N: tl.constexpr,
    THETA: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    A = load_matrix(A_ptr, pid, stride_b, stride_r, stride_c, N)
    nrm = matrix_1_norm(A, N)

    ratio = nrm / THETA
    need_scale = ratio > 1.0
    log2_ratio = tl.log(tl.maximum(ratio, 1.0)) / LN2
    log2_ratio = tl.minimum(log2_ratio, smax)
    s = tl.where(need_scale, tl.ceil(log2_ratio).to(tl.int32), tl.full((), 0, tl.int32))

    As = A
    for _ in range(s):
        As = As * 0.5

    X = taylor18(As, N)

    for _ in range(s):
        X = mm(X, X, N)

    store_matrix(Out_ptr, X, pid, stride_b, stride_r, stride_c, N)


@triton.jit
def taylor18_c(Ar, Ai, N: tl.constexpr):
    """Complex T18 as four real matmuls per product."""
    I = eye(N, Ar.dtype)
    A2r, A2i = mm_c(Ar, Ai, Ar, Ai, N)
    A3r, A3i = mm_c(Ar, Ai, A2r, A2i, N)
    A6r, A6i = mm_c(A3r, A3i, A3r, A3i, N)

    B1r = T18_B1_A * Ar + T18_B1_A2 * A2r + T18_B1_A3 * A3r
    B1i = T18_B1_A * Ai + T18_B1_A2 * A2i + T18_B1_A3 * A3i
    B2r = T18_B2_A * Ar + T18_B2_A2 * A2r + T18_B2_A3 * A3r + T18_B2_A6 * A6r
    B2i = T18_B2_A * Ai + T18_B2_A2 * A2i + T18_B2_A3 * A3i + T18_B2_A6 * A6i
    # The I term is real only.
    B3r = T18_B3_I * I + T18_B3_A * Ar + T18_B3_A2 * A2r + T18_B3_A3 * A3r + T18_B3_A6 * A6r
    B3i = T18_B3_A * Ai + T18_B3_A2 * A2i + T18_B3_A3 * A3i + T18_B3_A6 * A6i
    B4r = T18_B4_I * I + T18_B4_A * Ar + T18_B4_A2 * A2r + T18_B4_A3 * A3r + T18_B4_A6 * A6r
    B4i = T18_B4_A * Ai + T18_B4_A2 * A2i + T18_B4_A3 * A3i + T18_B4_A6 * A6i
    B5r = T18_B5_A2 * A2r + T18_B5_A3 * A3r + T18_B5_A6 * A6r
    B5i = T18_B5_A2 * A2i + T18_B5_A3 * A3i + T18_B5_A6 * A6i
    A9r, A9i = mm_c(B1r, B1i, B5r, B5i, N)
    A9r = A9r + B4r
    A9i = A9i + B4i
    Tr, Ti = mm_c(B3r + A9r, B3i + A9i, A9r, A9i, N)
    return B2r + Tr, B2i + Ti


@triton.jit
def matrix_exp_kernel_c(
    Ar_ptr,
    Ai_ptr,
    Outr_ptr,
    Outi_ptr,
    stride_b,
    stride_r,
    stride_c,
    B,
    smax,
    N: tl.constexpr,
    THETA: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    Ar = load_matrix(Ar_ptr, pid, stride_b, stride_r, stride_c, N)
    Ai = load_matrix(Ai_ptr, pid, stride_b, stride_r, stride_c, N)
    nrm = matrix_1_norm_c(Ar, Ai, N)

    ratio = nrm / THETA
    need_scale = ratio > 1.0
    log2_ratio = tl.log(tl.maximum(ratio, 1.0)) / LN2
    log2_ratio = tl.minimum(log2_ratio, smax)
    s = tl.where(need_scale, tl.ceil(log2_ratio).to(tl.int32), tl.full((), 0, tl.int32))

    Asr = Ar
    Asi = Ai
    for _ in range(s):
        Asr = Asr * 0.5
        Asi = Asi * 0.5

    Xr, Xi = taylor18_c(Asr, Asi, N)
    for _ in range(s):
        Xr, Xi = mm_c(Xr, Xi, Xr, Xi, N)

    store_matrix(Outr_ptr, Xr, pid, stride_b, stride_r, stride_c, N)
    store_matrix(Outi_ptr, Xi, pid, stride_b, stride_r, stride_c, N)


def triton_matrix_exp(A: torch.Tensor) -> torch.Tensor:
    """Launch Triton matrix_exp for contiguous ``(B, N, N)`` CUDA tensors."""
    if A.ndim != 3:
        raise ValueError("triton_matrix_exp expects shape (B, N, N)")
    B, N, N2 = A.shape
    if N != N2:
        raise ValueError(f"triton_matrix_exp expects square matrices, got {tuple(A.shape)}")
    supported = SUPPORTED_N_COMPLEX if A.is_complex() else SUPPORTED_N
    if N not in supported:
        raise ValueError(f"N={N} not in supported sizes {supported}")
    if not A.is_cuda:
        raise ValueError("triton_matrix_exp requires a CUDA tensor")
    low_prec = A.dtype in (torch.float32, torch.complex64)
    theta = THETA18_FP32 if low_prec else THETA18_FP64
    smax = SMAX_FP32 if low_prec else SMAX_FP64
    if A.is_complex():
        return launch_complex(
            matrix_exp_kernel_c,
            A,
            smax,
            N=N,
            THETA=theta,
            num_warps=1 if N <= 8 else 4,
        )
    A = A.contiguous()
    out = torch.empty_like(A)
    sb, sr, sc = A.stride(0), A.stride(1), A.stride(2)
    launch_kernel(
        matrix_exp_kernel,
        (B,),
        A,
        out,
        sb,
        sr,
        sc,
        B,
        smax,
        N=N,
        THETA=theta,
        # 1 warp measured faster through n=32.
        num_warps=1 if N <= 32 else 4,
    )
    return out
