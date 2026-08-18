"""Hermitian fast path via eigh; Daleckii-Krein gradients stay finite at repeated eigenvalues."""

import torch
from torch.autograd import forward_ad as fwAD

SUPPORTED_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)


def frechet(w, Q, E: torch.Tensor, conj_gamma: bool) -> torch.Tensor:
    """Daleckii-Krein derivative Q (Gamma * (Q^H E Q)) Q^H; close eigenvalues use exp(midpoint)."""
    eps = torch.finfo(w.dtype).eps
    wi, wj = w.unsqueeze(-1), w.unsqueeze(-2)
    dw = wi - wj
    small = dw.abs() <= eps**0.5
    fw = torch.exp(w.to(E.dtype))
    dfw = fw.unsqueeze(-1) - fw.unsqueeze(-2)
    quot = dfw / torch.where(small, torch.ones_like(dw), dw).to(dfw.dtype)
    deriv = torch.exp((0.5 * (wi + wj)).to(E.dtype))
    gamma = torch.where(small, deriv, quot)
    if conj_gamma:
        gamma = gamma.conj()
    return Q @ (gamma * (Q.mH @ E @ Q)) @ Q.mH


class HermitianFn(torch.autograd.Function):
    """``exp((A + A^H)/2)`` via eigh; backward recomputes eigh so higher-order autograd works."""

    generate_vmap_rule = True

    @staticmethod
    def forward(A):
        w, Q = torch.linalg.eigh(0.5 * (A + A.mH))
        return Q @ torch.diag_embed(torch.exp(w.to(A.dtype))) @ Q.mH

    @staticmethod
    def setup_context(ctx, inputs, output):
        (A,) = inputs
        ctx.save_for_backward(A)
        ctx.save_for_forward(A)

    @staticmethod
    def backward(ctx, grad):
        (A,) = ctx.saved_tensors
        w, Q = torch.linalg.eigh(0.5 * (A + A.mH))
        # Wirtinger convention: conjugate Gamma in the VJP.
        out = frechet(w, Q, grad, conj_gamma=True)
        return 0.5 * (out + out.mH)

    @staticmethod
    def jvp(ctx, tangent):
        (A,) = ctx.saved_tensors
        w, Q = torch.linalg.eigh(0.5 * (A + A.mH))
        return frechet(w, Q, 0.5 * (tangent + tangent.mH), conj_gamma=False)


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


def hermitian_apply(A: torch.Tensor) -> torch.Tensor:
    if needs_custom_function(A):
        return HermitianFn.apply(A)
    w, Q = torch.linalg.eigh(0.5 * (A + A.mH))
    return Q @ torch.diag_embed(torch.exp(w.to(A.dtype))) @ Q.mH


def apply_eigenfun(A: torch.Tensor, fn) -> torch.Tensor:
    """``Q fn(w) Q^H`` test oracle; the public path is hermitian_apply, which adds gradients."""
    w, Q = torch.linalg.eigh(A)
    fw = fn(w)
    if fw.dtype != A.dtype:
        fw = fw.to(A.dtype)
    return Q @ torch.diag_embed(fw) @ Q.mH
