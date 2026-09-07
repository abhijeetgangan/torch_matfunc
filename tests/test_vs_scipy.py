"""Public matrix_sqrt and matrix_log against SciPy as an independent oracle."""

import pytest
import torch
from helpers import random_nonnormal, random_spd, tols_for

pytest.importorskip("scipy")
import numpy as np
from scipy.linalg import logm, sqrtm

from torch_matfunc.linalg import matrix_log, matrix_sqrt

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def scipy_batch(fn, A: torch.Tensor) -> torch.Tensor:
    """Apply a SciPy matrix function per matrix and rebuild the batch on A's device."""
    n = A.shape[-1]
    flat = A.detach().cpu().numpy().reshape(-1, n, n)
    out = []
    for M in flat:
        F = np.asarray(fn(M))
        out.append(F if A.is_complex() else F.real)
    return torch.tensor(np.stack(out), dtype=A.dtype, device=A.device).reshape(A.shape)


@pytest.mark.parametrize("device", DEVICES)
class TestNonnormalReal:
    def test_sqrt(self, device):
        torch.manual_seed(4)
        A = random_nonnormal((4,), 4, device=device)
        torch.testing.assert_close(matrix_sqrt(A), scipy_batch(sqrtm, A), **tols_for(A.dtype))

    def test_log(self, device):
        torch.manual_seed(3)
        A = random_nonnormal((4,), 4, device=device)
        torch.testing.assert_close(matrix_log(A), scipy_batch(logm, A), **tols_for(A.dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", (3, 8))
class TestNonnormalComplex128:
    def test_sqrt(self, device, n):
        torch.manual_seed(13)
        A = random_nonnormal((2,), n, dtype=torch.complex128, device=device)
        torch.testing.assert_close(matrix_sqrt(A), scipy_batch(sqrtm, A), **tols_for(A.dtype))

    def test_log(self, device, n):
        torch.manual_seed(14)
        A = random_nonnormal((2,), n, dtype=torch.complex128, device=device)
        torch.testing.assert_close(matrix_log(A), scipy_batch(logm, A), **tols_for(A.dtype))


@pytest.mark.parametrize("device", DEVICES)
class TestHermitianSPD:
    # The eigh-based spectral helpers share the implementation's code path, so SciPy is the oracle.
    def _spd(self, device):
        torch.manual_seed(5)
        return random_spd((3,), 5, device=device)

    def test_sqrt(self, device):
        A = self._spd(device)
        out = matrix_sqrt(A, hermitian=True)
        torch.testing.assert_close(out, scipy_batch(sqrtm, A), **tols_for(A.dtype))

    def test_log(self, device):
        A = self._spd(device)
        out = matrix_log(A, hermitian=True)
        torch.testing.assert_close(out, scipy_batch(logm, A), **tols_for(A.dtype))
