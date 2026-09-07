"""Cross-op regression tests: Triton activation, non-finite input, torch.func composition."""

import time

import pytest
import torch
from helpers import random_near_identity_spd, random_spd, requires_cuda, requires_triton, tols_for

from torch_matfunc.linalg import matrix_exp, matrix_log, matrix_sqrt, ops

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def rel_residual_sqrt(S, A):
    return ((S @ S - A).norm() / A.norm()).item()


def rel_residual_log(L, A):
    return ((matrix_exp(L) - A).norm() / A.norm()).item()


@requires_cuda
class TestTritonIsActive:
    """The Triton kernels load on the CUDA suite so kernel tests exercise the kernels."""

    def test_triton_available(self):
        assert ops.TRITON_AVAILABLE
        assert 4 in ops.EXP_SUPPORTED_N


class TestNonFiniteExp:
    """matrix_exp on inf or NaN input returns promptly and propagates the non-finite entries."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_inf_nan_bounded(self, device):
        A = torch.randn(4, 4, 4, dtype=torch.float64, device=device)
        A[0, 0, 0] = float("inf")
        A[1, 0, 0] = float("nan")
        t0 = time.perf_counter()
        E = matrix_exp(A)
        if device == "cuda":
            torch.cuda.synchronize()
        assert time.perf_counter() - t0 < 30.0
        assert not torch.isfinite(E[0]).all()
        assert not torch.isfinite(E[1]).all()
        assert torch.isfinite(E[2:]).all()


class TestTorchFunc:
    """torch.func transforms compose through the ops: hessian, nested vmap, batched grads."""

    def _spd(self, n=3):
        torch.manual_seed(0)
        G = torch.randn(n, n, dtype=torch.float64)
        return G @ G.mT / n + 2 * torch.eye(n, dtype=torch.float64)

    def test_hessian(self):
        H = torch.func.hessian(lambda x: matrix_sqrt(x).sum())(self._spd())
        assert torch.isfinite(H).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_nested_vmap(self, device):
        A = torch.randn(3, 5, 4, 4, dtype=torch.float64, device=device) * 0.3
        out = torch.vmap(torch.vmap(matrix_exp))(A)
        torch.testing.assert_close(out, torch.linalg.matrix_exp(A), **tols_for(torch.float64))

    @pytest.mark.parametrize("device", DEVICES)
    def test_is_grads_batched(self, device):
        A = random_spd((4,), 3, device=device).requires_grad_(True)
        out = matrix_exp(A)
        cotangents = torch.eye(3, dtype=torch.float64, device=device).expand(2, 4, 3, 3)
        g = torch.autograd.grad(out, A, cotangents, is_grads_batched=True)[0]
        assert g.shape == (2, 4, 3, 3) and torch.isfinite(g).all()


@requires_cuda
class TestJvpValues:
    """CUDA forward-mode JVPs match central finite differences; CPU is covered by gradcheck."""

    # Near-identity draws keep the FD roundoff ||f(A)|| * eps / h below the comparison tolerance.
    @pytest.mark.parametrize("fn", (matrix_exp, matrix_sqrt, matrix_log))
    def test_jvp_matches_fd(self, fn):
        torch.manual_seed(1)
        A = random_near_identity_spd((2,), 4).to("cuda")
        T = (torch.randn(2, 4, 4, dtype=torch.float64) * 0.3).to("cuda")
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(A, T)
            _, tangent = torch.autograd.forward_ad.unpack_dual(fn(dual))
        assert tangent is not None
        eps = 1e-6
        fd = (fn(A + eps * T) - fn(A - eps * T)) / (2 * eps)
        torch.testing.assert_close(tangent, fd, rtol=1e-5, atol=1e-5)


class TestHermitianGrads:
    """hermitian=True grads are finite at repeated eigenvalues; indefinite input still gives NaN."""

    def test_grad_at_identity(self):
        I4 = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1).requires_grad_(True)
        matrix_exp(I4, hermitian=True).sum().backward()
        e = torch.exp(torch.tensor(1.0, dtype=torch.float64))
        assert torch.isfinite(I4.grad).all()
        torch.testing.assert_close(
            I4.grad.diagonal(dim1=-2, dim2=-1), e.expand(3, 4), rtol=1e-12, atol=1e-12
        )

    def test_gradcheck_complex(self):
        def f(x):
            return matrix_log(x @ x.mH + 4 * torch.eye(3, dtype=torch.complex128), hermitian=True)

        x = torch.randn(3, 3, dtype=torch.complex128, requires_grad=True) * 0.3
        assert torch.autograd.gradcheck(f, (x,))

    def test_genuinely_indefinite_still_signals(self):
        A = torch.diag(torch.tensor([-1.0, 2.0], dtype=torch.float64))
        assert not torch.isfinite(matrix_sqrt(A, hermitian=True)).all()


@requires_triton
class TestComplexTritonLargeN:
    """complex128 Triton launchers at n in {16, 32}, called directly since dispatch skips them."""

    @pytest.mark.parametrize("n", (16, 32))
    def test_sqrt_log_cdouble(self, n):
        from torch_matfunc.triton_kernels.logm import triton_matrix_log
        from torch_matfunc.triton_kernels.sqrtm import triton_matrix_sqrt

        torch.manual_seed(0)
        G = torch.randn(8, n, n, dtype=torch.complex128, device="cuda")
        A = G @ G.mH / n + 4 * torch.eye(n, dtype=torch.complex128, device="cuda")
        assert rel_residual_sqrt(triton_matrix_sqrt(A), A) < 1e-12
        assert rel_residual_log(triton_matrix_log(A), A) < 1e-11


@requires_triton
class TestFp32TritonLargeN:
    """fp32 Triton sqrt and log at n in {8, 16, 32} through public dispatch."""

    @pytest.mark.parametrize("n", (8, 16, 32))
    def test_sqrt_log_fp32(self, n):
        torch.manual_seed(0)
        G = torch.randn(64, n, n, dtype=torch.float32, device="cuda")
        A = G @ G.mT / n + 4 * torch.eye(n, dtype=torch.float32, device="cuda")
        assert rel_residual_sqrt(matrix_sqrt(A), A) < 5e-6
        L = matrix_log(A)
        assert rel_residual_log(L, A) < 5e-5


@requires_triton
class TestComplex64Triton:
    """complex64 Triton sqrt, log and exp at n in {8, 16, 32}; c64 dispatch has no max_b gate."""

    @pytest.mark.parametrize("n", (8, 16, 32))
    def test_sqrt_log_cfloat(self, n):
        torch.manual_seed(0)
        G = torch.randn(16, n, n, dtype=torch.complex64, device="cuda")
        A = G @ G.mH / n + 4 * torch.eye(n, dtype=torch.complex64, device="cuda")
        assert rel_residual_sqrt(matrix_sqrt(A), A) < 5e-6
        assert rel_residual_log(matrix_log(A), A) < 5e-5

    @pytest.mark.parametrize("n", (8, 16, 32))
    def test_exp_cfloat(self, n):
        torch.manual_seed(0)
        A = torch.randn(16, n, n, dtype=torch.complex64, device="cuda") * (0.5 / n**0.5)
        expected = torch.linalg.matrix_exp(A.to(torch.complex128)).to(torch.complex64)
        torch.testing.assert_close(matrix_exp(A), expected, **tols_for(torch.complex64))
