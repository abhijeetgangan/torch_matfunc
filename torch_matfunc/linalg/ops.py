"""torch.library.triton_op registration for matrix_exp."""

import torch

from torch_matfunc.reference import expm as ref_expm
from torch_matfunc.reference.spectral import needs_custom_function

TRITON_AVAILABLE = False
EXP_SUPPORTED_N = ()
EXP_SUPPORTED_N_COMPLEX = ()
TRITON_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
triton_matrix_exp = None  # type: ignore
try:
    import triton  # noqa: F401

    from torch_matfunc.triton_kernels.expm import SUPPORTED_N as EXP_SUPPORTED_N
    from torch_matfunc.triton_kernels.expm import (
        SUPPORTED_N_COMPLEX as EXP_SUPPORTED_N_COMPLEX,
    )
    from torch_matfunc.triton_kernels.expm import triton_matrix_exp

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    pass


def can_use_triton(A: torch.Tensor) -> bool:
    """Shape/dtype/device-only routing predicate, safe to evaluate at trace time."""
    sizes = EXP_SUPPORTED_N_COMPLEX if A.is_complex() else EXP_SUPPORTED_N
    return TRITON_AVAILABLE and A.is_cuda and A.dtype in TRITON_DTYPES and A.shape[-1] in sizes


def forward_exp(A: torch.Tensor) -> torch.Tensor:
    if can_use_triton(A):
        n = A.shape[-1]
        return triton_matrix_exp(A.reshape(-1, n, n).contiguous()).reshape(A.shape)
    return ref_expm.matrix_exp(A)


def block_triangular(A: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Stack [[A, E], [0, A]]; exp of it puts the derivative of exp at A toward E top right."""
    zeros = torch.zeros_like(A)
    return torch.cat([torch.cat([A, E], dim=-1), torch.cat([zeros, A], dim=-1)], dim=-2)


matrix_exp_op = torch.library.triton_op("torch_matfunc::matrix_exp", forward_exp, mutates_args=())


@matrix_exp_op.register_fake
def fake_impl(A: torch.Tensor) -> torch.Tensor:
    # The op always returns contiguous output; empty_like would copy fake strides.
    return torch.empty_like(A, memory_format=torch.contiguous_format)


@matrix_exp_op.register_kernel("cpu")
def cpu_impl(A: torch.Tensor) -> torch.Tensor:
    return forward_exp(A)


def setup_context(ctx, inputs, output):
    (A,) = inputs
    ctx.save_for_backward(A)


def backward(ctx, grad):
    (A,) = ctx.saved_tensors
    n = A.shape[-1]
    f_block = matrix_exp_op(block_triangular(A.mH.contiguous(), grad.contiguous()))
    return f_block[..., :n, n:]


matrix_exp_op.register_autograd(backward, setup_context=setup_context)


def vmap_impl(info, in_dims, A):
    # Redispatch to the op so nested vmap unwraps one level per rule invocation.
    from torch._library.triton import set_wrap_triton_enabled

    # wrap_triton lacks FuncTorchVmapMode support; the kernel batches leading dims.
    dim = in_dims[0]
    with set_wrap_triton_enabled(False):
        if dim is None:
            return matrix_exp_op(A), None
        if dim != 0:
            return matrix_exp_op(A.movedim(dim, 0)).movedim(0, dim), dim
        return matrix_exp_op(A), 0


matrix_exp_op.register_vmap(vmap_impl)


class MatrixExpFn(torch.autograd.Function):
    """Autograd for matrix_exp: backward and jvp take exp of the block matrix [[A, E], [0, A]]."""

    # PyTorch limitation: jacfwd(jacfwd) silently returns zeros; use hessian or jacfwd(jacrev).

    generate_vmap_rule = True

    @staticmethod
    def forward(A):
        return torch.ops.torch_matfunc.matrix_exp(A)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (A,) = inputs
        ctx.save_for_backward(A)
        ctx.save_for_forward(A)

    @staticmethod
    def backward(ctx, grad):
        (A,) = ctx.saved_tensors
        n = A.shape[-1]
        f_block = MatrixExpFn.apply(block_triangular(A.mH.contiguous(), grad.contiguous()))
        return f_block[..., :n, n:]

    @staticmethod
    def jvp(ctx, tangent):
        (A,) = ctx.saved_tensors
        n = A.shape[-1]
        f_block = MatrixExpFn.apply(block_triangular(A.contiguous(), tangent.contiguous()))
        # Forward AD needs the tangent layout to match the contiguous primal on CUDA.
        return f_block[..., :n, n:].contiguous()


def library_apply(A: torch.Tensor) -> torch.Tensor:
    if needs_custom_function(A):
        return MatrixExpFn.apply(A)
    if torch.compiler.is_compiling() and not can_use_triton(A):
        # The pure-Torch path loops data-dependently, which compile cannot trace; run eager.
        torch._dynamo.graph_break()
    return torch.ops.torch_matfunc.matrix_exp(A)
