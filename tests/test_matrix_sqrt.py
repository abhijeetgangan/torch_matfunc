"""Tests for reference / public matrix_sqrt."""

import math

import pytest
import torch

from torch_matfunc.linalg import matrix_log, matrix_sqrt, ops
from torch_matfunc.reference.spectral import apply_eigenfun
from torch_matfunc.reference.sqrtm import matrix_sqrt as ref_sqrtm

# 2x the worst error measured across the suite; badly conditioned tests set looser ones.
TOL_FP32 = {"rtol": 1e-5, "atol": 5e-6}
TOL_FP64 = {"rtol": 1e-11, "atol": 1e-12}

dtypes = pytest.mark.parametrize(
    "dtype,tol", [(torch.float32, TOL_FP32), (torch.float64, TOL_FP64)], ids=["fp32", "fp64"]
)


def spd(batch, n, dtype=torch.float64, device="cpu"):
    B = torch.randn(*batch, n, n, dtype=dtype, device=device)
    return B @ B.mH + n * torch.eye(n, dtype=dtype, device=device)


def triangular(batch, n, dtype=torch.float64, device="cpu"):
    # Upper triangular with positive diagonal: non-normal, and its principal sqrt is triangular.
    d = 0.5 + torch.rand(*batch, n, dtype=dtype, device=device)
    U = torch.triu(torch.randn(*batch, n, n, dtype=dtype, device=device), diagonal=1)
    return torch.diag_embed(d) + U


def ill_conditioned(batch, n, cond, device="cpu"):
    # SPD with eigenvalues log-spaced over [1/cond, 1].
    Q, _ = torch.linalg.qr(torch.randn(*batch, n, n, dtype=torch.float64, device=device))
    w = torch.logspace(0.0, -math.log10(cond), n, dtype=torch.float64, device=device)
    return Q @ torch.diag_embed(w.expand(*batch, n)) @ Q.mT


def rel_residual(S, A):
    return float((torch.linalg.matrix_norm(S @ S - A) / torch.linalg.matrix_norm(A)).max())


class TestRefMatrixSqrt:
    def test_identity(self):
        I = torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(ref_sqrtm(I), I, rtol=0.0, atol=1e-12)

    def test_diagonal(self):
        d = torch.tensor([1.0, 4.0, 0.25], dtype=torch.float64)
        out = ref_sqrtm(torch.diag(d))
        torch.testing.assert_close(out, torch.diag(torch.sqrt(d)), **TOL_FP64)

    @dtypes
    def test_square_recovers(self, dtype, tol):
        torch.manual_seed(0)
        A = spd((3,), 4, dtype=dtype)
        S = ref_sqrtm(A)
        torch.testing.assert_close(S @ S, A, **tol)

    @dtypes
    def test_sqrt_of_exp(self, dtype, tol):
        # sqrt(exp(B)) = exp(B / 2) while ||B|| is small enough to stay on the principal branch.
        torch.manual_seed(1)
        B = 0.4 * torch.randn(4, 5, 5, dtype=dtype)
        out = ref_sqrtm(torch.linalg.matrix_exp(B))
        torch.testing.assert_close(out, torch.linalg.matrix_exp(0.5 * B), **tol)

    def test_triangular_stays_triangular(self):
        torch.manual_seed(2)
        A = triangular((3,), 5)
        S = ref_sqrtm(A)
        torch.testing.assert_close(S @ S, A, **TOL_FP64)
        strict_lower = torch.tril(S, diagonal=-1)
        torch.testing.assert_close(strict_lower, torch.zeros_like(S), rtol=0.0, atol=1e-13)
        diag = A.diagonal(dim1=-2, dim2=-1).sqrt()
        torch.testing.assert_close(S.diagonal(dim1=-2, dim2=-1), diag, **TOL_FP64)

    @pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
    def test_complex_sqrt_of_exp(self, dtype):
        torch.manual_seed(3)
        B = 0.4 * torch.randn(4, 5, 5, dtype=dtype)
        out = ref_sqrtm(torch.linalg.matrix_exp(B))
        tol = TOL_FP64 if dtype is torch.complex128 else TOL_FP32
        torch.testing.assert_close(out, torch.linalg.matrix_exp(0.5 * B), **tol)

    def test_ill_conditioned_residual(self):
        # Determinantal scaling and the stall exit keep wide spectra at the rounding floor.
        torch.manual_seed(4)
        A = ill_conditioned((2,), 6, cond=1e6)
        assert rel_residual(ref_sqrtm(A), A) < 1e-12

    def test_unscaled_iteration_converges(self):
        torch.manual_seed(5)
        A = spd((2,), 4)
        S = ref_sqrtm(A, scaled=False)
        torch.testing.assert_close(S @ S, A, **TOL_FP64)

    def test_unbatched(self):
        torch.manual_seed(6)
        A = spd((), 4)
        S = ref_sqrtm(A)
        assert S.shape == A.shape
        torch.testing.assert_close(S @ S, A, **TOL_FP64)


class TestPublicMatrixSqrt:
    """Public dispatch on CPU: routes through the registered op to the reference."""

    @dtypes
    def test_sqrt_of_exp(self, dtype, tol):
        torch.manual_seed(7)
        B = 0.4 * torch.randn(4, 5, 5, dtype=dtype)
        out = matrix_sqrt(torch.linalg.matrix_exp(B))
        torch.testing.assert_close(out, torch.linalg.matrix_exp(0.5 * B), **tol)

    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
    def test_hermitian_matches_eigh(self, dtype):
        torch.manual_seed(8)
        A = spd((3,), 4, dtype=dtype)
        expected = apply_eigenfun(A, torch.sqrt)
        tight = dtype is torch.float64
        tol = {"rtol": 1e-12, "atol": 1e-12} if tight else {"rtol": 1e-5, "atol": 1e-6}
        torch.testing.assert_close(matrix_sqrt(A, hermitian=True), expected, **tol)

    def test_hermitian_psd_rounding(self):
        # A rank-deficient Gram matrix has rounding-negative eigenvalues; they must clamp to 0.
        torch.manual_seed(9)
        B = torch.randn(3, 4, 2, dtype=torch.float64)
        A = B @ B.mT
        S = matrix_sqrt(A, hermitian=True)
        assert torch.isfinite(S).all()
        torch.testing.assert_close(S @ S, A, rtol=1e-11, atol=1e-11)

    @pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_half_upcast(self, dtype):
        torch.manual_seed(10)
        B = torch.randn(2, 4, 4, dtype=torch.float32) * 0.3
        A32 = B @ B.mT + torch.eye(4, dtype=torch.float32)
        A = A32.to(dtype)
        out = matrix_sqrt(A)
        assert out.dtype == dtype
        expected = matrix_sqrt(A.to(torch.float32))
        torch.testing.assert_close(out.to(torch.float32), expected, rtol=2e-2, atol=2e-2)

    def test_gradcheck(self):
        torch.manual_seed(11)
        A = spd((2,), 3).requires_grad_()
        assert torch.autograd.gradcheck(matrix_sqrt, (A,), atol=1e-6, check_forward_ad=True)

    def test_hermitian_gradcheck(self):
        # Exercises the Daleckii-Krein gradients with the sqrt divided differences.
        torch.manual_seed(12)
        A = spd((2,), 3).requires_grad_()

        def f(X):
            return matrix_sqrt(0.5 * (X + X.mH), hermitian=True)

        assert torch.autograd.gradcheck(f, (A,), atol=1e-6, check_forward_ad=True)

    def test_hermitian_grad_repeated_eigenvalue(self):
        # At 4I every eigenvalue pair coincides; the divided difference must give f'(4) = 1/4.
        A = (4 * torch.eye(3, dtype=torch.float64)).requires_grad_()
        matrix_sqrt(A, hermitian=True).sum().backward()
        expected = torch.full((3,), 0.25, dtype=torch.float64)
        torch.testing.assert_close(A.grad.diagonal(), expected, rtol=1e-12, atol=1e-12)

    def test_gradgradcheck(self):
        torch.manual_seed(13)
        A = spd((), 3).requires_grad_()
        assert torch.autograd.gradgradcheck(matrix_sqrt, (A,), atol=1e-4)

    def test_vmap_matches_batched(self):
        torch.manual_seed(14)
        A = spd((5,), 4)
        torch.testing.assert_close(torch.vmap(matrix_sqrt)(A), matrix_sqrt(A), **TOL_FP64)

    def test_vmap_of_grad_matches_loop(self):
        # vmap of grad must keep the per-sample autograd association.
        torch.manual_seed(15)
        A = spd((4,), 3)

        def f(X):
            return matrix_sqrt(X).sum()

        batched = torch.func.vmap(torch.func.grad(f))(A)
        loop = torch.stack([torch.func.grad(f)(A[i]) for i in range(4)])
        torch.testing.assert_close(batched, loop, **TOL_FP64)

    @pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
    def test_complex_sqrt_of_exp(self, dtype):
        torch.manual_seed(16)
        B = 0.4 * torch.randn(4, 5, 5, dtype=dtype)
        out = matrix_sqrt(torch.linalg.matrix_exp(B))
        tol = TOL_FP64 if dtype is torch.complex128 else TOL_FP32
        torch.testing.assert_close(out, torch.linalg.matrix_exp(0.5 * B), **tol)

    # The prescale must keep accuracy independent of ||A||.
    @pytest.mark.parametrize("scale", (1e-12, 1e12))
    def test_extreme_scale(self, scale):
        torch.manual_seed(17)
        A = spd((8,), 4) * scale
        torch.testing.assert_close(matrix_sqrt(A), apply_eigenfun(A, torch.sqrt), **TOL_FP64)

    def test_log_of_sqrt_is_half_log(self):
        torch.manual_seed(18)
        A = spd((3,), 4)
        torch.testing.assert_close(2 * matrix_log(matrix_sqrt(A)), matrix_log(A), **TOL_FP64)

    def test_triton_batch_caps(self):
        # Pins the measured routing table; absent dtype always wins, absent n never.
        def wins(B, n, dtype):
            A = torch.empty(B, n, n, dtype=dtype, device="meta")
            return ops.triton_wins(A, ops.SQRT_TRITON_MAX_B)

        assert wins(2048, 2, torch.float64)
        assert not wins(2049, 2, torch.float64)
        assert wins(256, 4, torch.float64)
        assert not wins(257, 4, torch.float64)
        assert wins(64, 8, torch.float64)
        assert not wins(65, 8, torch.float64)
        assert not wins(8, 16, torch.float64)
        assert not wins(8, 32, torch.float64)
        assert wins(100000, 32, torch.float32)
        assert wins(32768, 2, torch.complex128)
        assert wins(128, 4, torch.complex128)
        assert not wins(129, 4, torch.complex128)
        assert not wins(8, 8, torch.complex128)

    @pytest.mark.parametrize("shape", ((3, 0, 0), (0, 4, 4)))
    def test_zero_size(self, shape):
        A = torch.zeros(*shape, dtype=torch.float64)
        out = matrix_sqrt(A)
        assert out.shape == A.shape and out.dtype == A.dtype

    def test_nonsquare_raises(self):
        with pytest.raises(ValueError, match="square"):
            matrix_sqrt(torch.zeros(2, 3))

    def test_int_dtype_raises(self):
        with pytest.raises(TypeError, match="supports"):
            matrix_sqrt(torch.zeros(2, 2, dtype=torch.int64))

    def test_cpu_compile_falls_back_to_eager(self):
        # Compile must graph-break to eager on the CPU reference route.
        torch.manual_seed(19)
        A = spd((2,), 4)
        compiled = torch.compile(matrix_sqrt)(A)
        torch.testing.assert_close(compiled, matrix_sqrt(A), rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and ops.TRITON_AVAILABLE), reason="CUDA and Triton required"
)
class TestPublicMatrixSqrtCuda:
    @dtypes
    @pytest.mark.parametrize("n", (2, 4, 8, 16, 32))
    def test_triton_kernel_matches_reference(self, n, dtype, tol):
        # Calls the kernel directly; the public routing skips Triton where it measured slower.
        torch.manual_seed(0)
        A = torch.linalg.matrix_exp(0.4 * torch.randn(8, n, n, dtype=dtype, device="cuda"))
        out = ops.triton_matrix_sqrt(A.contiguous())
        tol = dict(tol)
        if dtype is torch.float32:
            # Both sides iterate to fp32 floors; the measured worst gap is 1.2e-5 at n=32.
            tol["atol"] = 2.5e-5
        torch.testing.assert_close(out, ref_sqrtm(A), **tol)

    @dtypes
    def test_public_matches_reference(self, dtype, tol):
        torch.manual_seed(1)
        A = torch.linalg.matrix_exp(0.4 * torch.randn(8, 4, 4, dtype=dtype, device="cuda"))
        torch.testing.assert_close(matrix_sqrt(A), ref_sqrtm(A), **tol)

    # n=2 hits the closed-form complex 2x2 inverse; n=32 the warm-started Schulz inverse.
    @pytest.mark.parametrize("n", (2, 4, 32))
    def test_complex128_kernel(self, n):
        torch.manual_seed(2)
        B = 0.3 * torch.randn(4, n, n, dtype=torch.complex128, device="cuda")
        A = torch.linalg.matrix_exp(B)
        out = ops.triton_matrix_sqrt(A.contiguous())
        torch.testing.assert_close(out, ref_sqrtm(A), **TOL_FP64)

    def test_complex64_public(self):
        # complex64 always routes to Triton; keep that route exercised.
        torch.manual_seed(3)
        B = 0.4 * torch.randn(4, 8, 8, dtype=torch.complex64, device="cuda")
        A = torch.linalg.matrix_exp(B)
        torch.testing.assert_close(matrix_sqrt(A), ref_sqrtm(A), **TOL_FP32)

    def test_batch_cap_routes_reference(self):
        # fp64 n=8 Triton wins only through B=64; above the cap the op must run the reference.
        torch.manual_seed(4)
        A = spd((128,), 8, device="cuda")
        assert not ops.routes_to_triton("matrix_sqrt", A)
        assert ops.routes_to_triton("matrix_sqrt", A[:64])
        torch.testing.assert_close(matrix_sqrt(A), ref_sqrtm(A), rtol=0.0, atol=0.0)

    def test_singular_leading_block(self):
        # The 4x4 adjugate inverse must survive a singular top-left 2x2 block.
        rows = [[0.0, 0, 1, 0], [0, 0, 0, 1], [-2, 0, 3, 0], [0, -2, 0, 3]]
        A = torch.tensor(rows, dtype=torch.float64, device="cuda").unsqueeze(0)
        S = matrix_sqrt(A)
        torch.testing.assert_close(S @ S, A, **TOL_FP64)

    # Pins the kernel's host prescale: sqrt(A) = sqrt(c) sqrt(A / c).
    @pytest.mark.parametrize("scale", (1e-12, 1e12))
    def test_extreme_scale_through_triton(self, scale):
        torch.manual_seed(5)
        A = spd((8,), 4, device="cuda") * scale
        torch.testing.assert_close(matrix_sqrt(A), apply_eigenfun(A, torch.sqrt), **TOL_FP64)

    @dtypes
    def test_odd_n_routes_reference(self, dtype, tol):
        torch.manual_seed(6)
        B = 0.4 * torch.randn(4, 5, 5, dtype=dtype, device="cuda")
        out = matrix_sqrt(torch.linalg.matrix_exp(B))
        torch.testing.assert_close(out, torch.linalg.matrix_exp(0.5 * B), **tol)

    def test_n64_routes_reference(self):
        torch.manual_seed(7)
        A = spd((2,), 64, device="cuda")
        assert not ops.routes_to_triton("matrix_sqrt", A)
        S = matrix_sqrt(A)
        torch.testing.assert_close(S @ S, A, **TOL_FP64)

    def test_compile_fullgraph_matches_eager(self):
        torch.manual_seed(8)
        A = spd((8,), 4, device="cuda")
        compiled = torch.compile(matrix_sqrt, fullgraph=True)(A)
        torch.testing.assert_close(compiled, matrix_sqrt(A), **TOL_FP64)

    def test_opcheck(self):
        torch.manual_seed(9)
        A = spd((2,), 4, device="cuda")
        torch.library.opcheck(
            torch.ops.torch_matfunc.matrix_sqrt,
            (A,),
            test_utils=("test_schema", "test_faketensor"),
        )

    def test_gradcheck_through_triton(self):
        torch.manual_seed(10)
        A = spd((2,), 4, device="cuda").requires_grad_()
        assert torch.autograd.gradcheck(matrix_sqrt, (A,), atol=1e-6)

    def test_vmap_through_triton(self):
        torch.manual_seed(11)
        A = spd((6,), 4, device="cuda")
        torch.testing.assert_close(torch.vmap(matrix_sqrt)(A), matrix_sqrt(A), **TOL_FP64)
