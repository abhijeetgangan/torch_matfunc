"""Host-driven inverse scaling and squaring: up to MAX_SQRT masked Denman-Beavers root launches
with per-matrix ``s`` kept on device, then one degree-7 Pade launch. The root loop stays on the
host because nesting it around the DB loops inside one kernel miscompiles."""

import torch
import triton
import triton.language as tl

from torch_matfunc.triton_kernels._core import (
    eye,
    launch_kernel,
    load_matrix,
    matrix_1_norm,
    matrix_1_norm_c,
    mm,
    mm_c,
    store_matrix,
)
from torch_matfunc.triton_kernels._iterations import (
    INV_TOL_FP32,
    INV_TOL_FP64,
    SCHULZ_PREFIX,
    SCHULZ_REST,
    denman_beavers,
    denman_beavers_c,
    invert,
    invert_c,
)

SUPPORTED_N = (2, 4, 8, 16, 32)
MAX_SQRT = 24
SQRT_ITERS = 16
# Later ISS roots sit near I and converge within 4 DB steps, so check the residual early.
ISS_DB_PREFIX = 4
# theta_7 from Al-Mohy & Higham; the fp32 radius matches reference/logm.py.
THETA_FP64 = 0.288
THETA_FP32 = 0.20


@triton.jit
def pade_term(E, node, weight, N: tl.constexpr, nprefix, nrest, inv_tol):
    return mm(invert(eye(N, E.dtype) + node * E, N, nprefix, nrest, inv_tol), weight * E, N)


@triton.jit
def log1p_pade(E, N: tl.constexpr, nprefix, nrest, inv_tol):
    """Gauss-Legendre Pade of log(I+E), m=7, with nodes/weights on [0, 1]."""
    U = tl.zeros((N, N), dtype=E.dtype)
    U += pade_term(E, 0.0254460438286207, 0.06474248308443487, N, nprefix, nrest, inv_tol)
    U += pade_term(E, 0.12923440720030277, 0.13985269574463843, N, nprefix, nrest, inv_tol)
    U += pade_term(E, 0.2970774243113014, 0.19091502525255935, N, nprefix, nrest, inv_tol)
    U += pade_term(E, 0.5, 0.20897959183673465, N, nprefix, nrest, inv_tol)
    U += pade_term(E, 0.7029225756886985, 0.19091502525255935, N, nprefix, nrest, inv_tol)
    U += pade_term(E, 0.8707655927996972, 0.13985269574463843, N, nprefix, nrest, inv_tol)
    U += pade_term(E, 0.9745539561713793, 0.06474248308443487, N, nprefix, nrest, inv_tol)
    return U


@triton.jit
def log_root_kernel(
    X_ptr,
    S_ptr,
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
    THETA: tl.constexpr,
    TOL: tl.constexpr,
):
    """One masked ISS root in place: ``X <- sqrt(X)`` where ``||X - I||_1 >= THETA``; converged
    programs run zero DB steps. Each program owns S[pid] and holds its matrix in SRAM."""
    pid = tl.program_id(0)
    if pid >= B:
        return

    X = load_matrix(X_ptr, pid, stride_b, stride_r, stride_c, N)
    I = eye(N, X.dtype)
    need = matrix_1_norm(X - I, N) >= THETA
    pref_eff = tl.where(need, db_prefix, 0)
    rest_eff = tl.where(need, db_rest, 0)
    Y = denman_beavers(X, N, pref_eff, rest_eff, schulz_prefix, schulz_rest, inv_tol, TOL)
    store_matrix(X_ptr, Y, pid, stride_b, stride_r, stride_c, N)
    s_old = tl.load(S_ptr + pid)
    tl.store(S_ptr + pid, s_old + tl.where(need, 1, 0))


@triton.jit
def log_pade_kernel(
    X_ptr,
    Out_ptr,
    S_ptr,
    stride_b,
    stride_r,
    stride_c,
    B,
    schulz_prefix,
    schulz_rest,
    inv_tol,
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    X = load_matrix(X_ptr, pid, stride_b, stride_r, stride_c, N)
    I = eye(N, X.dtype)
    s = tl.load(S_ptr + pid)
    L = log1p_pade(X - I, N, schulz_prefix, schulz_rest, inv_tol) * tl.exp2(s.to(X.dtype))
    store_matrix(Out_ptr, L, pid, stride_b, stride_r, stride_c, N)


@triton.jit
def pade_term_c(Er, Ei, node, weight, N: tl.constexpr, nprefix, nrest, inv_tol):
    I = eye(N, Er.dtype)
    Inv_r, Inv_i = invert_c(I + node * Er, node * Ei, N, nprefix, nrest, inv_tol)
    return mm_c(Inv_r, Inv_i, weight * Er, weight * Ei, N)


@triton.jit
def log1p_pade_c(Er, Ei, N: tl.constexpr, nprefix, nrest, inv_tol):
    Ur = tl.zeros((N, N), dtype=Er.dtype)
    Ui = tl.zeros((N, N), dtype=Ei.dtype)

    Pr, Pi = pade_term_c(
        Er, Ei, 0.0254460438286207, 0.06474248308443487, N, nprefix, nrest, inv_tol
    )
    Ur = Ur + Pr
    Ui = Ui + Pi
    Pr, Pi = pade_term_c(
        Er, Ei, 0.12923440720030277, 0.13985269574463843, N, nprefix, nrest, inv_tol
    )
    Ur = Ur + Pr
    Ui = Ui + Pi
    Pr, Pi = pade_term_c(
        Er, Ei, 0.2970774243113014, 0.19091502525255935, N, nprefix, nrest, inv_tol
    )
    Ur = Ur + Pr
    Ui = Ui + Pi
    Pr, Pi = pade_term_c(Er, Ei, 0.5, 0.20897959183673465, N, nprefix, nrest, inv_tol)
    Ur = Ur + Pr
    Ui = Ui + Pi
    Pr, Pi = pade_term_c(
        Er, Ei, 0.7029225756886985, 0.19091502525255935, N, nprefix, nrest, inv_tol
    )
    Ur = Ur + Pr
    Ui = Ui + Pi
    Pr, Pi = pade_term_c(
        Er, Ei, 0.8707655927996972, 0.13985269574463843, N, nprefix, nrest, inv_tol
    )
    Ur = Ur + Pr
    Ui = Ui + Pi
    Pr, Pi = pade_term_c(
        Er, Ei, 0.9745539561713793, 0.06474248308443487, N, nprefix, nrest, inv_tol
    )
    return Ur + Pr, Ui + Pi


@triton.jit
def log_root_kernel_c(
    Xr_ptr,
    Xi_ptr,
    S_ptr,
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
    THETA: tl.constexpr,
    TOL: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    Xr = load_matrix(Xr_ptr, pid, stride_b, stride_r, stride_c, N)
    Xi = load_matrix(Xi_ptr, pid, stride_b, stride_r, stride_c, N)
    I = eye(N, Xr.dtype)
    need = matrix_1_norm_c(Xr - I, Xi, N) >= THETA
    pref_eff = tl.where(need, db_prefix, 0)
    rest_eff = tl.where(need, db_rest, 0)
    Yr, Yi = denman_beavers_c(
        Xr, Xi, N, pref_eff, rest_eff, schulz_prefix, schulz_rest, inv_tol, TOL
    )
    store_matrix(Xr_ptr, Yr, pid, stride_b, stride_r, stride_c, N)
    store_matrix(Xi_ptr, Yi, pid, stride_b, stride_r, stride_c, N)
    s_old = tl.load(S_ptr + pid)
    tl.store(S_ptr + pid, s_old + tl.where(need, 1, 0))


@triton.jit
def log_pade_kernel_c(
    Xr_ptr,
    Xi_ptr,
    Outr_ptr,
    Outi_ptr,
    S_ptr,
    stride_b,
    stride_r,
    stride_c,
    B,
    schulz_prefix,
    schulz_rest,
    inv_tol,
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    Xr = load_matrix(Xr_ptr, pid, stride_b, stride_r, stride_c, N)
    Xi = load_matrix(Xi_ptr, pid, stride_b, stride_r, stride_c, N)
    I = eye(N, Xr.dtype)
    s = tl.load(S_ptr + pid)
    Lr, Li = log1p_pade_c(Xr - I, Xi, N, schulz_prefix, schulz_rest, inv_tol)
    scale = tl.exp2(s.to(Xr.dtype))
    store_matrix(Outr_ptr, Lr * scale, pid, stride_b, stride_r, stride_c, N)
    store_matrix(Outi_ptr, Li * scale, pid, stride_b, stride_r, stride_c, N)


def triton_matrix_log(A: torch.Tensor) -> torch.Tensor:
    """Batched principal log for contiguous ``(B, N, N)`` CUDA tensors: host prescale
    ``log(A) = log(A/c) + log(c) I`` with real ``c = ||A||_1 > 0``, up to MAX_SQRT masked root
    launches, then one Pade launch."""
    if A.ndim != 3:
        raise ValueError("triton_matrix_log expects shape (B, N, N)")
    B, N, N2 = A.shape
    if N != N2:
        raise ValueError("square matrices required")
    if N not in SUPPORTED_N:
        raise ValueError(f"N={N} not in supported sizes {SUPPORTED_N}")
    if not A.is_cuda:
        raise ValueError("CUDA tensor required")
    low_prec = A.dtype in (torch.float32, torch.complex64)
    theta = THETA_FP32 if low_prec else THETA_FP64
    # DB residual tolerance at the dtype rounding floor, scaled by n.
    real_dt = torch.float32 if low_prec else torch.float64
    tol = 1.5 * N * torch.finfo(real_dt).eps
    inv_tol = INV_TOL_FP32 if low_prec else INV_TOL_FP64

    c = A.abs().sum(dim=-2).amax(dim=-1).clamp(min=torch.finfo(real_dt).tiny)[:, None, None]
    I = torch.eye(N, dtype=real_dt, device=A.device)
    shift = torch.log(c) * I
    s = torch.zeros(B, dtype=torch.int32, device=A.device)
    warps = 1 if N <= 8 else 4

    # Eager only: tracers cannot follow the break, and inductor retraces on FunctionalTensor.
    can_break = not torch.compiler.is_compiling() and type(A) is torch.Tensor

    if A.is_complex():
        Xr = (A.real / c).contiguous()
        Xi = (A.imag / c).contiguous()
        sb, sr, sc = Xr.stride(0), Xr.stride(1), Xr.stride(2)
        for i in range(MAX_SQRT):
            launch_kernel(
                log_root_kernel_c,
                (B,),
                Xr,
                Xi,
                s,
                sb,
                sr,
                sc,
                B,
                ISS_DB_PREFIX,
                SQRT_ITERS - ISS_DB_PREFIX,
                SCHULZ_PREFIX,
                SCHULZ_REST,
                inv_tol,
                N=N,
                THETA=theta,
                TOL=tol,
                num_warps=warps,
            )
            if can_break and i % 4 == 3:
                norms = ((Xr - I).square() + Xi.square()).sqrt().sum(-2).amax(-1)
                if not bool((norms >= theta).any()):
                    break
        outr = torch.empty_like(Xr)
        outi = torch.empty_like(Xi)
        launch_kernel(
            log_pade_kernel_c,
            (B,),
            Xr,
            Xi,
            outr,
            outi,
            s,
            sb,
            sr,
            sc,
            B,
            SCHULZ_PREFIX,
            SCHULZ_REST,
            inv_tol,
            N=N,
            num_warps=warps,
        )
        return torch.complex(outr, outi) + shift

    X = (A / c).contiguous()
    sb, sr, sc = X.stride(0), X.stride(1), X.stride(2)
    for i in range(MAX_SQRT):
        launch_kernel(
            log_root_kernel,
            (B,),
            X,
            s,
            sb,
            sr,
            sc,
            B,
            ISS_DB_PREFIX,
            SQRT_ITERS - ISS_DB_PREFIX,
            SCHULZ_PREFIX,
            SCHULZ_REST,
            inv_tol,
            N=N,
            THETA=theta,
            TOL=tol,
            num_warps=warps,
        )
        if can_break and i % 4 == 3:
            if not bool(((X - I).abs().sum(-2).amax(-1) >= theta).any()):
                break
    out = torch.empty_like(X)
    launch_kernel(
        log_pade_kernel,
        (B,),
        X,
        out,
        s,
        sb,
        sr,
        sc,
        B,
        SCHULZ_PREFIX,
        SCHULZ_REST,
        inv_tol,
        N=N,
        num_warps=warps,
    )
    return out + shift
