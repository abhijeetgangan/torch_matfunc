"""Tests for reference / public matrix_log."""

import pytest
import torch

from torch_matfunc.linalg import matrix_log, ops
from torch_matfunc.reference.logm import matrix_log as ref_logm
from torch_matfunc.reference.spectral import apply_eigenfun

# 2x the worst error measured across the suite; badly conditioned tests set looser ones.
TOL_FP32 = {"rtol": 1e-5, "atol": 5e-6}
TOL_FP64 = {"rtol": 1e-11, "atol": 1e-12}

dtypes = pytest.mark.parametrize(
    "dtype,tol", [(torch.float32, TOL_FP32), (torch.float64, TOL_FP64)], ids=["fp32", "fp64"]
)


def hetero_batch():
    # Deterministic mixed-convergence batch: the mild matrix needs one square root, the wide four.
    c, s = torch.cos(torch.tensor(0.3)), torch.sin(torch.tensor(0.3))
    Q = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    mild = Q @ torch.diag(torch.tensor([1.9, 0.5], dtype=torch.float64)) @ Q.mT
    wide = Q @ torch.diag(torch.tensor([50.0, 0.02], dtype=torch.float64)) @ Q.mT
    return torch.stack([mild, wide])


class TestRefMatrixLog:
    def test_identity(self):
        I = torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(ref_logm(I), torch.zeros_like(I), rtol=0.0, atol=1e-12)

    def test_diagonal(self):
        d = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
        out = ref_logm(torch.diag(d))
        torch.testing.assert_close(out, torch.diag(torch.log(d)), **TOL_FP64)

    @dtypes
    def test_log_of_exp(self, dtype, tol):
        torch.manual_seed(0)
        B = 0.4 * torch.randn(4, 5, 5, dtype=dtype)
        out = ref_logm(torch.linalg.matrix_exp(B))
        torch.testing.assert_close(out, B, **tol)

    def test_exp_of_log_spd(self):
        torch.manual_seed(1)
        B = torch.randn(2, 4, 4, dtype=torch.float64)
        A = B @ B.mT + 4 * torch.eye(4, dtype=torch.float64)
        torch.testing.assert_close(torch.linalg.matrix_exp(ref_logm(A)), A, **TOL_FP64)

    def test_heterogeneous_batch_subset_gather(self, monkeypatch):
        import torch_matfunc.reference.logm as logm_mod

        A = hetero_batch()
        batch_sizes = []
        real_sqrt = logm_mod.matrix_sqrt

        def spy(X, **kw):
            batch_sizes.append(X.shape[0])
            return real_sqrt(X, **kw)

        monkeypatch.setattr(logm_mod, "matrix_sqrt", spy)
        L = logm_mod.matrix_log(A)
        # The mild matrix converges first, so later roots must run on the gathered subset.
        assert 1 in batch_sizes, f"gather branch never fired: sqrt batch sizes {batch_sizes}"
        torch.testing.assert_close(L, apply_eigenfun(A, torch.log), **TOL_FP64)
        # Batched result must agree with each matrix computed alone via the full-batch path.
        solo = torch.stack([ref_logm(A[0]), ref_logm(A[1])])
        torch.testing.assert_close(L, solo, rtol=1e-13, atol=1e-13)


class TestPublicMatrixLog:
    """Public dispatch on CPU: routes through the registered op to the reference."""

    @dtypes
    def test_log_of_exp(self, dtype, tol):
        torch.manual_seed(2)
        B = 0.4 * torch.randn(4, 5, 5, dtype=dtype)
        out = matrix_log(torch.linalg.matrix_exp(B))
        torch.testing.assert_close(out, B, **tol)

    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
    def test_hermitian_matches_eigh(self, dtype):
        torch.manual_seed(3)
        B = torch.randn(3, 4, 4, dtype=dtype)
        A = B @ B.mT + 4 * torch.eye(4, dtype=dtype)
        expected = apply_eigenfun(A, torch.log)
        tight = dtype is torch.float64
        tol = {"rtol": 1e-12, "atol": 1e-12} if tight else {"rtol": 1e-5, "atol": 1e-6}
        torch.testing.assert_close(matrix_log(A, hermitian=True), expected, **tol)

    @pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_half_upcast(self, dtype):
        torch.manual_seed(4)
        B = torch.randn(2, 4, 4, dtype=torch.float32) * 0.3
        A32 = B @ B.mT + torch.eye(4, dtype=torch.float32)
        A = A32.to(dtype)
        out = matrix_log(A)
        assert out.dtype == dtype
        expected = matrix_log(A.to(torch.float32))
        torch.testing.assert_close(out.to(torch.float32), expected, rtol=2e-2, atol=2e-2)

    def test_gradcheck(self):
        torch.manual_seed(5)
        B = torch.randn(2, 3, 3, dtype=torch.float64)
        A = (B @ B.mT + 3 * torch.eye(3, dtype=torch.float64)).requires_grad_()
        assert torch.autograd.gradcheck(matrix_log, (A,), atol=1e-6, check_forward_ad=True)

    def test_hermitian_gradcheck(self):
        # Exercises the Daleckii-Krein gradients with the log divided differences.
        torch.manual_seed(6)
        B = torch.randn(2, 3, 3, dtype=torch.float64)
        A = (B @ B.mT + 3 * torch.eye(3, dtype=torch.float64)).requires_grad_()

        def f(X):
            return matrix_log(0.5 * (X + X.mH), hermitian=True)

        assert torch.autograd.gradcheck(f, (A,), atol=1e-6, check_forward_ad=True)

    def test_gradgradcheck(self):
        torch.manual_seed(7)
        B = torch.randn(3, 3, dtype=torch.float64)
        A = (B @ B.mT + 3 * torch.eye(3, dtype=torch.float64)).requires_grad_()
        assert torch.autograd.gradgradcheck(matrix_log, (A,), atol=1e-4)

    def test_heterogeneous_batch_autograd(self):
        # The gather branch scatters with an in-place indexed write; backward must survive it.
        A = hetero_batch().requires_grad_()
        ref_logm(A).sum().backward()
        assert torch.isfinite(A.grad).all()
        B = hetero_batch().requires_grad_()
        assert torch.autograd.gradcheck(matrix_log, (B,), atol=1e-7)

    def test_vmap_matches_batched(self):
        torch.manual_seed(8)
        B = 0.4 * torch.randn(5, 4, 4, dtype=torch.float64)
        A = torch.linalg.matrix_exp(B)
        out = torch.vmap(matrix_log)(A)
        torch.testing.assert_close(out, matrix_log(A), **TOL_FP64)

    def test_vmap_of_grad_matches_loop(self):
        # vmap of grad must keep the per-sample autograd association.
        torch.manual_seed(9)
        B = torch.randn(4, 3, 3, dtype=torch.float64)
        A = B @ B.mT + 3 * torch.eye(3, dtype=torch.float64)

        def f(X):
            return matrix_log(X).sum()

        batched = torch.func.vmap(torch.func.grad(f))(A)
        loop = torch.stack([torch.func.grad(f)(A[i]) for i in range(4)])
        torch.testing.assert_close(batched, loop, **TOL_FP64)

    @pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
    def test_complex_log_of_exp(self, dtype):
        torch.manual_seed(10)
        B = 0.4 * torch.randn(4, 5, 5, dtype=dtype)
        out = matrix_log(torch.linalg.matrix_exp(B))
        tol = TOL_FP64 if dtype is torch.complex128 else TOL_FP32
        torch.testing.assert_close(out, B, **tol)

    # The sqrt prescale must keep accuracy independent of ||A||.
    @pytest.mark.parametrize("scale", (1e-12, 1e12))
    def test_extreme_scale(self, scale):
        torch.manual_seed(11)
        B = torch.randn(8, 4, 4, dtype=torch.float64)
        A = (B @ B.mT + 4 * torch.eye(4, dtype=torch.float64)) * scale
        torch.testing.assert_close(matrix_log(A), apply_eigenfun(A, torch.log), **TOL_FP64)

    def test_triton_batch_caps(self):
        # Pins the measured routing table; absent dtype always wins, absent n never.
        def wins(B, n, dtype):
            A = torch.empty(B, n, n, dtype=dtype, device="meta")
            return ops.triton_wins(A, ops.LOG_TRITON_MAX_B)

        assert wins(128, 8, torch.float64)
        assert not wins(129, 8, torch.float64)
        assert wins(32, 16, torch.float64)
        assert not wins(8, 32, torch.float64)
        assert wins(4096, 2, torch.float64)
        assert not wins(8192, 2, torch.float64)
        assert wins(100000, 32, torch.float32)
        assert wins(8192, 2, torch.complex128)
        assert wins(256, 4, torch.complex128)
        assert not wins(257, 4, torch.complex128)
        assert not wins(8, 8, torch.complex128)

    def test_zero_size(self):
        A = torch.zeros(3, 0, 0, dtype=torch.float64)
        out = matrix_log(A)
        assert out.shape == A.shape

    def test_nonsquare_raises(self):
        with pytest.raises(ValueError, match="square"):
            matrix_log(torch.zeros(2, 3))

    def test_int_dtype_raises(self):
        with pytest.raises(TypeError, match="supports"):
            matrix_log(torch.zeros(2, 2, dtype=torch.int64))

    def test_cpu_compile_falls_back_to_eager(self):
        # Compile must graph-break to eager on the CPU reference route.
        A = hetero_batch()
        compiled = torch.compile(matrix_log)(A)
        torch.testing.assert_close(compiled, matrix_log(A), rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and ops.TRITON_AVAILABLE), reason="CUDA and Triton required"
)
class TestPublicMatrixLogCuda:
    @dtypes
    @pytest.mark.parametrize("n", (2, 4, 8, 16, 32))
    def test_triton_kernel_matches_reference(self, n, dtype, tol):
        # Calls the kernel directly; the public routing skips Triton where it measured slower.
        torch.manual_seed(0)
        B = 0.4 * torch.randn(8, n, n, dtype=dtype, device="cuda")
        A = torch.linalg.matrix_exp(B)
        out = ops.triton_matrix_log(A.contiguous())
        tol = dict(tol)
        if dtype is torch.float32:
            # Both sides iterate to fp32 floors near 4e-6; the entrywise gap grows with n.
            tol["atol"] = 2e-4
        torch.testing.assert_close(out, ref_logm(A), **tol)

    @dtypes
    def test_public_matches_reference(self, dtype, tol):
        torch.manual_seed(1)
        B = 0.4 * torch.randn(8, 4, 4, dtype=dtype, device="cuda")
        A = torch.linalg.matrix_exp(B)
        tol = dict(tol)
        if dtype is torch.float32:
            # The Triton and reference fp32 iterations round independently.
            tol["atol"] = 2e-5
        torch.testing.assert_close(matrix_log(A), ref_logm(A), **tol)

    # n=2 hits the closed-form complex 2x2 inverse, reachable only through the log kernel.
    @pytest.mark.parametrize("n", (2, 4, 32))
    def test_complex128_kernel(self, n):
        torch.manual_seed(2)
        B = 0.3 * torch.randn(4, n, n, dtype=torch.complex128, device="cuda")
        A = torch.linalg.matrix_exp(B)
        out = ops.triton_matrix_log(A.contiguous())
        torch.testing.assert_close(out, ref_logm(A), **TOL_FP64)

    def test_complex64_public(self):
        # complex64 always routes to Triton; keep that route exercised.
        torch.manual_seed(8)
        B = 0.4 * torch.randn(4, 8, 8, dtype=torch.complex64, device="cuda")
        A = torch.linalg.matrix_exp(B)
        torch.testing.assert_close(matrix_log(A), ref_logm(A), rtol=1e-4, atol=2e-5)

    def test_batch_cap_routes_reference(self):
        # fp64 n=8 Triton wins only through B=128; above the cap the op must run the reference.
        torch.manual_seed(9)
        M = 0.4 * torch.randn(256, 8, 8, dtype=torch.float64, device="cuda")
        A = torch.linalg.matrix_exp(M)
        assert not ops.routes_to_triton("matrix_log", A)
        assert ops.routes_to_triton("matrix_log", A[:128])
        torch.testing.assert_close(matrix_log(A), ref_logm(A), rtol=0.0, atol=0.0)

    def test_singular_leading_block(self):
        # The 4x4 adjugate inverse must survive a singular top-left 2x2 block.
        rows = [[0.0, 0, 1, 0], [0, 0, 0, 1], [-2, 0, 3, 0], [0, -2, 0, 3]]
        A = torch.tensor(rows, dtype=torch.float64, device="cuda").unsqueeze(0)
        out = matrix_log(A)
        torch.testing.assert_close(torch.linalg.matrix_exp(out), A, rtol=1e-11, atol=1e-12)

    # Pins the kernel's host prescale: log(A) = log(A/c) + log(c) I.
    @pytest.mark.parametrize("scale", (1e-12, 1e12))
    def test_extreme_scale_through_triton(self, scale):
        torch.manual_seed(10)
        B = torch.randn(8, 4, 4, dtype=torch.float64, device="cuda")
        A = (B @ B.mT + 4 * torch.eye(4, dtype=torch.float64, device="cuda")) * scale
        torch.testing.assert_close(matrix_log(A), apply_eigenfun(A, torch.log), **TOL_FP64)

    @dtypes
    def test_odd_n_routes_reference(self, dtype, tol):
        torch.manual_seed(3)
        B = 0.4 * torch.randn(4, 5, 5, dtype=dtype, device="cuda")
        A = torch.linalg.matrix_exp(B)
        torch.testing.assert_close(matrix_log(A), B, **tol)

    def test_compile_fullgraph_matches_eager(self):
        torch.manual_seed(4)
        B = 0.4 * torch.randn(8, 4, 4, dtype=torch.float64, device="cuda")
        A = torch.linalg.matrix_exp(B)
        compiled = torch.compile(matrix_log, fullgraph=True)(A)
        torch.testing.assert_close(compiled, matrix_log(A), **TOL_FP64)

    def test_opcheck(self):
        torch.manual_seed(5)
        B = 0.25 * torch.randn(2, 4, 4, dtype=torch.float64, device="cuda")
        A = torch.linalg.matrix_exp(B)
        torch.library.opcheck(
            torch.ops.torch_matfunc.matrix_log,
            (A,),
            test_utils=("test_schema", "test_faketensor"),
        )

    def test_gradcheck_through_triton(self):
        torch.manual_seed(6)
        B = torch.randn(2, 4, 4, dtype=torch.float64, device="cuda")
        A = (B @ B.mT + 4 * torch.eye(4, dtype=torch.float64, device="cuda")).requires_grad_()
        assert torch.autograd.gradcheck(matrix_log, (A,), atol=1e-6)

    def test_vmap_through_triton(self):
        torch.manual_seed(7)
        B = 0.4 * torch.randn(6, 4, 4, dtype=torch.float64, device="cuda")
        A = torch.linalg.matrix_exp(B)
        out = torch.vmap(matrix_log)(A)
        torch.testing.assert_close(out, matrix_log(A), **TOL_FP64)
