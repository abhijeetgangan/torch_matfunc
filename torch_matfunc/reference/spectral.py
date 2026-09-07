"""Hermitian fast path via eigh; Daleckii-Krein gradients stay finite at repeated eigenvalues."""

import contextlib

import torch
from torch.autograd import forward_ad as fwAD

SUPPORTED_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)


@contextlib.contextmanager
def magma_pin(A: torch.Tensor):
    """Prefer MAGMA for batched LU work on CUDA fp64 at n >= 32: the default dispatch there
    pairs a MAGMA factor that hides a device sync with slow cuBLAS batched trsm solves."""
    use = A.is_cuda and A.dtype is torch.float64 and A.shape[-1] >= 32 and torch.cuda.has_magma
    if not use:
        yield
        return
    # The preferred backend is process-global state; save and restore around the calls.
    prev = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library("magma")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_linalg_library(prev)


def scalar_fn(kind: str, w: torch.Tensor, dtype) -> torch.Tensor:
    if dtype.is_complex:
        w = w.to(dtype)
    fw = torch.exp(w) if kind == "exp" else torch.sqrt(w) if kind == "sqrt" else torch.log(w)
    return fw.to(dtype)


def eigh_forward(kind: str, A: torch.Tensor):
    w, Q = torch.linalg.eigh(0.5 * (A + A.mH))
    if kind in ("sqrt", "log"):
        # Zero out rounding-negative eigenvalues of numerically PSD input; keep genuine ones.
        eps = torch.finfo(w.dtype).eps
        tol = A.shape[-1] * eps * w.abs().amax(dim=-1, keepdim=True)
        w = torch.where((w < 0) & (w >= -tol), torch.zeros_like(w), w)
    return w, Q


def frechet(kind: str, w, Q, E: torch.Tensor, conj_gamma: bool) -> torch.Tensor:
    """Daleckii-Krein derivative Q (Gamma * (Q^H E Q)) Q^H; close eigenvalues use f'(midpoint)."""
    eps = torch.finfo(w.dtype).eps
    if kind == "sqrt":
        # The divided difference is exactly 1 / (sqrt(w_i) + sqrt(w_j)), which never cancels.
        sw = scalar_fn("sqrt", w, E.dtype)
        gamma = 1.0 / (sw.unsqueeze(-1) + sw.unsqueeze(-2))
    else:
        wi, wj = w.unsqueeze(-1), w.unsqueeze(-2)
        dw = wi - wj
        # log cancels for relatively close pairs, exp for absolutely close ones.
        tol = eps**0.5 * torch.maximum(wi.abs(), wj.abs()) if kind == "log" else eps**0.5
        small = dw.abs() <= tol
        fw = scalar_fn(kind, w, E.dtype)
        dfw = fw.unsqueeze(-1) - fw.unsqueeze(-2)
        quot = dfw / torch.where(small, torch.ones_like(dw), dw).to(dfw.dtype)
        mid = 0.5 * (wi + wj)
        if E.dtype.is_complex:
            mid = mid.to(E.dtype)
        deriv = (torch.exp(mid) if kind == "exp" else 1.0 / mid).to(E.dtype)
        gamma = torch.where(small, deriv, quot)
    if conj_gamma:
        gamma = gamma.conj()
    Et = Q.mH @ E @ Q
    prod = gamma * Et
    # gamma is inf on the clamped-zero-eigenvalue block; a zero cotangent must give 0, not NaN.
    prod = torch.where(Et == 0, torch.zeros_like(prod), prod)
    return Q @ prod @ Q.mH


class HermitianFn(torch.autograd.Function):
    """``f((A + A^H)/2)`` via eigh; backward recomputes eigh so higher-order autograd works."""

    generate_vmap_rule = True

    @staticmethod
    def forward(A, kind):
        w, Q = eigh_forward(kind, A)
        return Q @ torch.diag_embed(scalar_fn(kind, w, A.dtype)) @ Q.mH

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, kind = inputs
        ctx.kind = kind
        ctx.save_for_backward(A)
        ctx.save_for_forward(A)

    @staticmethod
    def backward(ctx, grad):
        (A,) = ctx.saved_tensors
        w, Q = eigh_forward(ctx.kind, A)
        # Wirtinger convention: conjugate Gamma in the VJP.
        out = frechet(ctx.kind, w, Q, grad, conj_gamma=True)
        return 0.5 * (out + out.mH), None

    @staticmethod
    def jvp(ctx, tangent, _):
        (A,) = ctx.saved_tensors
        w, Q = eigh_forward(ctx.kind, A)
        return frechet(ctx.kind, w, Q, 0.5 * (tangent + tangent.mH), conj_gamma=False)


def needs_custom_function(A: torch.Tensor) -> bool:
    """True when ``A`` tracks reverse- or forward-mode AD at any functorch level."""
    if A.requires_grad:
        return True
    if torch.compiler.is_compiling():
        # Dynamo cannot trace the functorch bindings below.
        return False
    f = torch._C._functorch
    t = A
    while f.is_batchedtensor(t) or f.is_gradtrackingtensor(t):
        if f.is_gradtrackingtensor(t):
            return True
        t = f.get_unwrapped(t)
    if t.requires_grad:
        return True
    # unpack_dual raises on BatchedTensors, so unwrap them first.
    return fwAD.unpack_dual(t).tangent is not None


def hermitian_apply(kind: str, A: torch.Tensor) -> torch.Tensor:
    if needs_custom_function(A):
        return HermitianFn.apply(A, kind)
    w, Q = eigh_forward(kind, A)
    return Q @ torch.diag_embed(scalar_fn(kind, w, A.dtype)) @ Q.mH


def apply_eigenfun(A: torch.Tensor, fn) -> torch.Tensor:
    """``Q fn(w) Q^H`` test oracle; the public path is hermitian_apply, which adds gradients."""
    w, Q = torch.linalg.eigh(A)
    fw = fn(w)
    if fw.dtype != A.dtype:
        fw = fw.to(A.dtype)
    return Q @ torch.diag_embed(fw) @ Q.mH
