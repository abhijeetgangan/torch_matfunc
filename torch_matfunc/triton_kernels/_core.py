"""Triton nxn linear-algebra primitives kept in SRAM, one matrix per program."""

import torch
import triton
import triton.language as tl


def launch_kernel(kernel, grid, *args, num_warps=4, **kwargs):
    """Launch via wrap_triton on the tensor's device; num_stages=1 avoids pipeliner miscompiles."""
    device = next(a.device for a in args if isinstance(a, torch.Tensor))
    launcher = torch.library.wrap_triton(kernel)
    with torch.cuda.device(device):
        launcher[grid](*args, num_warps=num_warps, num_stages=1, **kwargs)


@triton.jit
def load_matrix(ptr, batch_idx, stride_b, stride_r, stride_c, N: tl.constexpr):
    offs_i = tl.arange(0, N)
    offs_j = tl.arange(0, N)
    ptrs = ptr + batch_idx * stride_b + offs_i[:, None] * stride_r + offs_j[None, :] * stride_c
    return tl.load(ptrs)


@triton.jit
def store_matrix(ptr, mat, batch_idx, stride_b, stride_r, stride_c, N: tl.constexpr):
    offs_i = tl.arange(0, N)
    offs_j = tl.arange(0, N)
    ptrs = ptr + batch_idx * stride_b + offs_i[:, None] * stride_r + offs_j[None, :] * stride_c
    tl.store(ptrs, mat)


@triton.jit
def eye(N: tl.constexpr, dtype):
    offs_i = tl.arange(0, N)
    offs_j = tl.arange(0, N)
    ones = tl.full((N, N), 1.0, dtype)
    zeros = tl.zeros((N, N), dtype=dtype)
    return tl.where(offs_i[:, None] == offs_j[None, :], ones, zeros)


@triton.jit
def mm(A, B, N: tl.constexpr):
    """nxn on-chip product: tl.dot for n >= 16 (its minimum size), summed products below."""
    if N >= 16:
        return tl.dot(A, B, input_precision="ieee")
    return tl.sum(A[:, :, None] * B[None, :, :], axis=1)


@triton.jit
def mm_c(Ar, Ai, Br, Bi, N: tl.constexpr):
    """Complex mm as four real matmuls: (Ar+iAi)(Br+iBi)."""
    real = mm(Ar, Br, N) - mm(Ai, Bi, N)
    imag = mm(Ar, Bi, N) + mm(Ai, Br, N)
    return real, imag


@triton.jit
def matrix_1_norm(A, N: tl.constexpr):
    """Induced 1-norm: max column sum."""
    col_sums = tl.sum(tl.abs(A), axis=0)
    return tl.max(col_sums)


@triton.jit
def matrix_1_norm_c(Ar, Ai, N: tl.constexpr):
    return tl.max(tl.sum(tl.sqrt(Ar * Ar + Ai * Ai), axis=0))


def launch_complex(kernel, A, *extra, num_warps=4, **kwargs):
    """Launch a kernel that takes real/imag pointer pairs and returns complex."""
    Ar = A.real.contiguous()
    Ai = A.imag.contiguous()
    outr = torch.empty_like(Ar)
    outi = torch.empty_like(Ai)
    B = Ar.shape[0]
    sb, sr, sc = Ar.stride(0), Ar.stride(1), Ar.stride(2)
    launch_kernel(
        kernel,
        (B,),
        Ar,
        Ai,
        outr,
        outi,
        sb,
        sr,
        sc,
        B,
        *extra,
        num_warps=num_warps,
        **kwargs,
    )
    return torch.complex(outr, outi)
