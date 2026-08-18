"""Tests for reference / public matrix_exp."""

import pytest
import torch

from torch_matfunc.linalg import matrix_exp
from torch_matfunc.reference.expm import matrix_exp as ref_expm
from torch_matfunc.reference.spectral import apply_eigenfun

# 2x the worst error measured across the suite; badly conditioned tests set looser ones.
TOL_FP32 = {"rtol": 1e-5, "atol": 5e-6}
TOL_FP64 = {"rtol": 1e-11, "atol": 1e-12}

dtypes = pytest.mark.parametrize(
    "dtype,tol", [(torch.float32, TOL_FP32), (torch.float64, TOL_FP64)], ids=["fp32", "fp64"]
)


class TestRefMatrixExp:
    def test_identity(self):
        I = torch.eye(3, dtype=torch.float64)
        out = ref_expm(I)
        torch.testing.assert_close(out, torch.exp(torch.tensor(1.0, dtype=torch.float64)) * I)

    def test_diagonal(self):
        d = torch.tensor([0.0, 1.0, -0.5], dtype=torch.float64)
        A = torch.diag(d)
        out = ref_expm(A)
        expected = torch.diag(torch.exp(d))
        torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-12)

    @dtypes
    def test_matches_torch_linalg(self, dtype, tol):
        torch.manual_seed(0)
        A = torch.randn(4, 5, 5, dtype=dtype) * 0.5
        out = ref_expm(A)
        torch.testing.assert_close(out, torch.linalg.matrix_exp(A), **tol)

    def test_zero(self):
        A = torch.zeros(2, 2, dtype=torch.float64)
        out = ref_expm(A)
        torch.testing.assert_close(out, torch.eye(2, dtype=torch.float64))

    def test_spd_smoke(self):
        B = torch.randn(3, 3, dtype=torch.float64)
        A = B @ B.mT + 3 * torch.eye(3, dtype=torch.float64)
        out = ref_expm(A)
        assert out.shape == A.shape
        assert torch.isfinite(out).all()


class TestPublicMatrixExp:
    """Public dispatch on CPU: routes through the registered op to the reference."""

    @dtypes
    def test_matches_torch_linalg(self, dtype, tol):
        torch.manual_seed(2)
        A = torch.randn(4, 5, 5, dtype=dtype)
        out = matrix_exp(A)
        torch.testing.assert_close(out, torch.linalg.matrix_exp(A), **tol)

    # Scale 8 is the largest that stays finite in fp32.
    @dtypes
    @pytest.mark.parametrize("scale", (1e-4, 1.0, 8.0))
    def test_squaring_branches(self, scale, dtype, tol):
        torch.manual_seed(3)
        A = scale * torch.randn(4, 8, 8, dtype=dtype)
        out = matrix_exp(A)
        torch.testing.assert_close(out, torch.linalg.matrix_exp(A), **tol)

    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
    def test_hermitian_matches_eigh(self, dtype):
        torch.manual_seed(4)
        B = torch.randn(3, 4, 4, dtype=dtype)
        A = B @ B.mT + 4 * torch.eye(4, dtype=dtype)
        expected = apply_eigenfun(A, torch.exp)
        tight = dtype is torch.float64
        tol = {"rtol": 1e-12, "atol": 1e-12} if tight else {"rtol": 1e-5, "atol": 1e-6}
        torch.testing.assert_close(matrix_exp(A, hermitian=True), expected, **tol)

    @pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_half_upcast(self, dtype):
        torch.manual_seed(5)
        A = (torch.randn(2, 4, 4, dtype=torch.float32) * 0.3).to(dtype)
        out = matrix_exp(A)
        assert out.dtype == dtype
        expected = torch.linalg.matrix_exp(A.to(torch.float32))
        torch.testing.assert_close(out.to(torch.float32), expected, rtol=2e-2, atol=2e-2)

    def test_gradcheck(self):
        torch.manual_seed(6)
        A = (0.3 * torch.randn(2, 3, 3, dtype=torch.float64)).requires_grad_()
        assert torch.autograd.gradcheck(matrix_exp, (A,), atol=1e-7, check_forward_ad=True)

    def test_hermitian_gradcheck(self):
        # Exercises the Daleckii-Krein gradients.
        torch.manual_seed(7)
        B = torch.randn(2, 3, 3, dtype=torch.float64)
        A = (B @ B.mT + 3 * torch.eye(3, dtype=torch.float64)).requires_grad_()

        def f(X):
            return matrix_exp(0.5 * (X + X.mH), hermitian=True)

        assert torch.autograd.gradcheck(f, (A,), atol=1e-7, check_forward_ad=True)

    def test_zero_size(self):
        A = torch.zeros(3, 0, 0, dtype=torch.float64)
        out = matrix_exp(A)
        assert out.shape == A.shape

    def test_nonsquare_raises(self):
        with pytest.raises(ValueError, match="square"):
            matrix_exp(torch.zeros(2, 3))

    def test_int_dtype_raises(self):
        with pytest.raises(TypeError, match="supports"):
            matrix_exp(torch.zeros(2, 2, dtype=torch.int64))

    def test_cpu_compile_falls_back_to_eager(self):
        # Compile must graph-break to eager on the CPU reference route.
        torch.manual_seed(8)
        A = torch.randn(2, 4, 4, dtype=torch.float64) * 0.5
        compiled = torch.compile(matrix_exp)(A)
        torch.testing.assert_close(compiled, matrix_exp(A), rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton path")
class TestPublicMatrixExpCuda:
    @dtypes
    @pytest.mark.parametrize("n", (2, 4, 8, 16, 32, 64))
    def test_triton_matches_torch(self, n, dtype, tol):
        torch.manual_seed(0)
        A = torch.randn(8, n, n, dtype=dtype, device="cuda") * 0.5
        out = matrix_exp(A)
        tol = dict(tol)
        if dtype is torch.float32 and n >= 32:
            # The worst per-entry gap vs torch grows with n in fp32; normwise it stays near 2e-6.
            tol["atol"] = 3e-5
        torch.testing.assert_close(out, torch.linalg.matrix_exp(A), **tol)

    @pytest.mark.parametrize("n", (8, 32, 64))
    def test_complex128(self, n):
        # Complex n=64 falls back to the reference.
        torch.manual_seed(2)
        A = torch.randn(4, n, n, dtype=torch.complex128, device="cuda") * 0.4
        out = matrix_exp(A)
        torch.testing.assert_close(out, torch.linalg.matrix_exp(A), **TOL_FP64)

    @dtypes
    def test_odd_n_routes_reference(self, dtype, tol):
        torch.manual_seed(3)
        A = torch.randn(4, 5, 5, dtype=dtype, device="cuda") * 0.5
        out = matrix_exp(A)
        torch.testing.assert_close(out, torch.linalg.matrix_exp(A), **tol)

    def test_compile_fullgraph_matches_torch(self):
        torch.manual_seed(4)
        A = torch.randn(8, 4, 4, dtype=torch.float64, device="cuda") * 0.5
        compiled = torch.compile(matrix_exp, fullgraph=True)(A)
        torch.testing.assert_close(compiled, torch.linalg.matrix_exp(A), **TOL_FP64)

    def test_opcheck(self):
        A = torch.randn(2, 4, 4, dtype=torch.float64, device="cuda") * 0.25
        torch.library.opcheck(
            torch.ops.torch_matfunc.matrix_exp,
            (A,),
            test_utils=("test_schema", "test_faketensor"),
        )

    def test_gradcheck_through_triton(self):
        torch.manual_seed(5)
        A = (0.3 * torch.randn(2, 4, 4, dtype=torch.float64, device="cuda")).requires_grad_()
        assert torch.autograd.gradcheck(matrix_exp, (A,), atol=1e-6)
