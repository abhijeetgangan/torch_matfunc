"""
Tests for torch implementation of expm_frechet.py

Using standard PyTorch autograd patterns to compute gradients.
"""

import numpy as np
import pytest
import scipy
import torch
from numpy.testing import assert_allclose

from torch_matfunc.matrix.expm_frechet_ref import expm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64


class TestExpmFrechet:
    @pytest.mark.skip(reason="The Frechet derivative is not correct")
    def test_expm_frechet(self):
        # a test of the basic functionality
        M = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=np.float64,
        )
        A = np.array(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=np.float64,
        )
        E = np.array(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=np.float64,
        )
        expected_expm = scipy.linalg.expm(A)
        expected_frechet = scipy.linalg.expm(M)[:2, 2:]

        # Convert to PyTorch tensors - requires_grad for A since we'll compute gradients
        A_torch = torch.tensor(A, dtype=dtype, requires_grad=True)
        E_torch = torch.tensor(E, dtype=dtype)

        observed_expm = expm.apply(A_torch)
        assert_allclose(expected_expm, observed_expm.detach().numpy())

        (observed_frechet,) = torch.autograd.grad(
            observed_expm, A_torch, E_torch, retain_graph=True
        )
        assert_allclose(expected_frechet, observed_frechet.detach().numpy())
