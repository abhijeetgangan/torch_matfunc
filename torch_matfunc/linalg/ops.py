"""torch.library.triton_op registration for matrix_exp and matrix_log."""

import torch

from torch_matfunc.reference import expm as ref_expm
from torch_matfunc.reference import logm as ref_logm
from torch_matfunc.reference.spectral import needs_custom_function

TRITON_AVAILABLE = False
EXP_SUPPORTED_N = EXP_SUPPORTED_N_COMPLEX = LOG_SUPPORTED_N = ()
TRITON_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
triton_matrix_exp = triton_matrix_log = None  # type: ignore
try:
    import triton  # noqa: F401

    from torch_matfunc.triton_kernels.expm import SUPPORTED_N as EXP_SUPPORTED_N
    from torch_matfunc.triton_kernels.expm import (
        SUPPORTED_N_COMPLEX as EXP_SUPPORTED_N_COMPLEX,
    )
    from torch_matfunc.triton_kernels.expm import triton_matrix_exp
    from torch_matfunc.triton_kernels.logm import SUPPORTED_N as LOG_SUPPORTED_N
    from torch_matfunc.triton_kernels.logm import triton_matrix_log

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    pass


def can_use_triton(A: torch.Tensor, sizes: tuple) -> bool:
    return TRITON_AVAILABLE and A.is_cuda and A.dtype in TRITON_DTYPES and A.shape[-1] in sizes


# Measured max Triton-winning batch per (dtype, n); absent dtype = always, absent n = never.
LOG_TRITON_MAX_B = {
    torch.float64: {2: 4096, 4: 512, 8: 128, 16: 32},
    torch.complex128: {2: 8192, 4: 256},
}


def triton_wins(A: torch.Tensor, max_b: dict) -> bool:
    caps = max_b.get(A.dtype)
    if caps is None:
        return True
    cap = caps.get(A.shape[-1])
    return cap is not None and A.shape[:-2].numel() <= cap


def forward_exp(A: torch.Tensor) -> torch.Tensor:
    sizes = EXP_SUPPORTED_N_COMPLEX if A.is_complex() else EXP_SUPPORTED_N
    if can_use_triton(A, sizes):
        n = A.shape[-1]
        return triton_matrix_exp(A.reshape(-1, n, n).contiguous()).reshape(A.shape)
    return ref_expm.matrix_exp(A)


def forward_log(A: torch.Tensor) -> torch.Tensor:
    if can_use_triton(A, LOG_SUPPORTED_N) and triton_wins(A, LOG_TRITON_MAX_B):
        n = A.shape[-1]
        return triton_matrix_log(A.reshape(-1, n, n).contiguous()).reshape(A.shape)
    return ref_logm.matrix_log(A)


def block_triangular(A: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Stack [[A, E], [0, A]]; f of the block holds the Frechet derivative L_f(A, E) top right."""
    zeros = torch.zeros_like(A)
    return torch.cat([torch.cat([A, E], dim=-1), torch.cat([zeros, A], dim=-1)], dim=-2)


def register(name: str, forward_fn):
    op = torch.library.triton_op(f"torch_matfunc::{name}", forward_fn, mutates_args=())

    @op.register_fake
    def fake_impl(A: torch.Tensor) -> torch.Tensor:
        # The op always returns contiguous output; empty_like would copy fake strides.
        return torch.empty_like(A, memory_format=torch.contiguous_format)

    @op.register_kernel("cpu")
    def cpu_impl(A: torch.Tensor) -> torch.Tensor:
        return forward_fn(A)

    def setup_context(ctx, inputs, output):
        (A,) = inputs
        ctx.save_for_backward(A)

    def backward(ctx, grad):
        (A,) = ctx.saved_tensors
        n = A.shape[-1]
        f_block = op(block_triangular(A.mH.contiguous(), grad.contiguous()))
        return f_block[..., :n, n:]

    op.register_autograd(backward, setup_context=setup_context)

    def vmap_impl(info, in_dims, A):
        # Redispatch to the op so nested vmap unwraps one level per rule invocation.
        from torch._library.triton import set_wrap_triton_enabled

        # wrap_triton lacks FuncTorchVmapMode support; the kernel batches leading dims.
        dim = in_dims[0]
        with set_wrap_triton_enabled(False):
            if dim is None:
                return op(A), None
            if dim != 0:
                return op(A.movedim(dim, 0)).movedim(0, dim), dim
            return op(A), 0

    op.register_vmap(vmap_impl)
    return op


register("matrix_exp", forward_exp)
register("matrix_log", forward_log)


def frechet_function(op_name: str, class_name: str):
    """Autograd for the op: backward and jvp take f of the block matrix [[A, E], [0, A]].
    jacfwd(jacfwd) silently returns zeros, a PyTorch limitation; use hessian or jacfwd(jacrev)."""

    class FrechetFn(torch.autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def forward(A):
            return getattr(torch.ops.torch_matfunc, op_name)(A)

        @staticmethod
        def setup_context(ctx, inputs, output):
            (A,) = inputs
            ctx.save_for_backward(A)
            ctx.save_for_forward(A)

        @staticmethod
        def backward(ctx, grad):
            (A,) = ctx.saved_tensors
            n = A.shape[-1]
            f_block = FrechetFn.apply(block_triangular(A.mH.contiguous(), grad.contiguous()))
            return f_block[..., :n, n:]

        @staticmethod
        def jvp(ctx, tangent):
            (A,) = ctx.saved_tensors
            n = A.shape[-1]
            f_block = FrechetFn.apply(block_triangular(A.contiguous(), tangent.contiguous()))
            # Forward AD needs the tangent layout to match the contiguous primal on CUDA.
            return f_block[..., :n, n:].contiguous()

    FrechetFn.__name__ = FrechetFn.__qualname__ = class_name
    return FrechetFn


MatrixExpFn = frechet_function("matrix_exp", "MatrixExpFn")
MatrixLogFn = frechet_function("matrix_log", "MatrixLogFn")
FRECHET_FNS = {"matrix_exp": MatrixExpFn, "matrix_log": MatrixLogFn}


def routes_to_triton(name: str, A: torch.Tensor) -> bool:
    """Shape/dtype/device-only routing predicate, safe to evaluate at trace time."""
    if name == "matrix_exp":
        return can_use_triton(A, EXP_SUPPORTED_N_COMPLEX if A.is_complex() else EXP_SUPPORTED_N)
    return can_use_triton(A, LOG_SUPPORTED_N) and triton_wins(A, LOG_TRITON_MAX_B)


def library_apply(name: str, A: torch.Tensor) -> torch.Tensor:
    if needs_custom_function(A):
        return FRECHET_FNS[name].apply(A)
    if torch.compiler.is_compiling() and not routes_to_triton(name, A):
        # The pure-Torch path loops data-dependently, which compile cannot trace; run eager.
        torch._dynamo.graph_break()
    return getattr(torch.ops.torch_matfunc, name)(A)
