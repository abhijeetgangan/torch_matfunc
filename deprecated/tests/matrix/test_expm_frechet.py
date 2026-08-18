"""
Scipy tests for torch implementation of expm_frechet.py

The tests are organized into three classes:
1. TestExpmFrechet - Tests using numpy arrays converted to torch tensors
2. TestExpmFrechetTorch - Tests using native torch tensors
3. TestExpmFrechetTorchGrad - Tests for gradient computation

See also:
https://github.com/scipy/scipy/blob/52a09130dae506c207d459d574ab22e636fe4b06/scipy/linalg/tests/test_matfuncs.py#L790
"""

import numpy as np
import scipy
import torch
from numpy.testing import assert_allclose

from torch_matfunc.matrix.expm_frechet import ell_table_61, expm, expm_frechet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64


class TestExpmFrechet:
    """Test suite for expm_frechet using numpy arrays converted to torch tensors."""

    def test_expm_frechet(self):
        """Test basic functionality of expm_frechet against scipy implementation."""
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

        A = torch.from_numpy(A).to(device=device)
        E = torch.from_numpy(E).to(device=device)
        for kwargs in ({}, {"method": "SPS"}, {"method": "blockEnlarge"}):
            # Convert it to numpy arrays before passing it to the function
            observed_expm, observed_frechet = expm_frechet(A, E, **kwargs)
            assert_allclose(expected_expm, observed_expm.cpu().numpy())
            assert_allclose(expected_frechet, observed_frechet.cpu().numpy())

    def test_small_norm_expm_frechet(self):
        """Test matrices with a range of norms for better coverage."""
        M_original = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=np.float64,
        )
        A_original = np.array(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=np.float64,
        )
        E_original = np.array(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=np.float64,
        )
        A_original_norm_1 = scipy.linalg.norm(A_original, 1)
        selected_m_list = [1, 3, 5, 7, 9, 11, 13, 15]
        m_neighbor_pairs = zip(selected_m_list[:-1], selected_m_list[1:], strict=True)
        for ma, mb in m_neighbor_pairs:
            ell_a = scipy.linalg._expm_frechet.ell_table_61[ma]
            ell_b = scipy.linalg._expm_frechet.ell_table_61[mb]
            target_norm_1 = 0.5 * (ell_a + ell_b)
            scale = target_norm_1 / A_original_norm_1
            M = scale * M_original
            A = scale * A_original
            E = scale * E_original
            expected_expm = scipy.linalg.expm(A)
            expected_frechet = scipy.linalg.expm(M)[:2, 2:]
            A = torch.from_numpy(A).to(device=device, dtype=dtype)
            E = torch.from_numpy(E).to(device=device, dtype=dtype)
            # Convert it to numpy arrays before passing it to the function
            observed_expm, observed_frechet = expm_frechet(A, E)
            assert_allclose(expected_expm, observed_expm.cpu().numpy())
            assert_allclose(expected_frechet, observed_frechet.cpu().numpy())

    def test_fuzz(self):
        """Test with a variety of random inputs to ensure robustness."""
        rng = np.random.default_rng(1726500908359153)
        # try a bunch of crazy inputs
        rfuncs = (
            np.random.uniform,
            np.random.normal,
            np.random.standard_cauchy,
            np.random.exponential,
        )
        ntests = 100
        for _ in range(ntests):
            rfunc = rfuncs[rng.choice(4)]
            target_norm_1 = rng.exponential()
            n = rng.integers(2, 16)
            A_original = rfunc(size=(n, n))
            E_original = rfunc(size=(n, n))
            A_original_norm_1 = scipy.linalg.norm(A_original, 1)
            scale = target_norm_1 / A_original_norm_1
            A = scale * A_original
            E = scale * E_original
            M = np.vstack([np.hstack([A, E]), np.hstack([np.zeros_like(A), A])])
            expected_expm = scipy.linalg.expm(A)
            expected_frechet = scipy.linalg.expm(M)[:n, n:]
            A = torch.from_numpy(A).to(device=device, dtype=dtype)
            E = torch.from_numpy(E).to(device=device, dtype=dtype)
            # Convert it to numpy arrays before passing it to the function
            observed_expm, observed_frechet = expm_frechet(A, E)
            assert_allclose(expected_expm, observed_expm.cpu().numpy(), atol=5e-8)
            assert_allclose(expected_frechet, observed_frechet.cpu().numpy(), atol=1e-7)

    def test_problematic_matrix(self):
        """Test a specific matrix that previously uncovered a bug."""
        A = np.array(
            [
                [1.50591997, 1.93537998],
                [0.41203263, 0.23443516],
            ],
            dtype=np.float64,
        )
        E = np.array(
            [
                [1.87864034, 2.07055038],
                [1.34102727, 0.67341123],
            ],
            dtype=np.float64,
        )
        A = torch.from_numpy(A).to(device=device, dtype=dtype)
        E = torch.from_numpy(E).to(device=device, dtype=dtype)
        # Convert it to numpy arrays before passing it to the function
        sps_expm, sps_frechet = expm_frechet(A, E, method="SPS")
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(A, E, method="blockEnlarge")
        assert_allclose(sps_expm.cpu().numpy(), blockEnlarge_expm.cpu().numpy())
        assert_allclose(sps_frechet.cpu().numpy(), blockEnlarge_frechet.cpu().numpy())

    def test_medium_matrix(self):
        """Test with a medium-sized matrix to compare performance between methods."""
        n = 1000
        A = np.random.exponential(size=(n, n))
        E = np.random.exponential(size=(n, n))

        A = torch.from_numpy(A).to(device=device, dtype=dtype)
        E = torch.from_numpy(E).to(device=device, dtype=dtype)
        # Convert it to numpy arrays before passing it to the function
        sps_expm, sps_frechet = expm_frechet(A, E, method="SPS")
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(A, E, method="blockEnlarge")
        assert_allclose(sps_expm.cpu().numpy(), blockEnlarge_expm.cpu().numpy())
        assert_allclose(sps_frechet.cpu().numpy(), blockEnlarge_frechet.cpu().numpy())


class TestExpmFrechetTorch:
    """Test suite for expm_frechet using native torch tensors."""

    def test_expm_frechet(self):
        """Test basic functionality of expm_frechet against torch.linalg.matrix_exp."""
        M = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        A = torch.tensor(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        E = torch.tensor(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=dtype,
            device=device,
        )
        expected_expm = torch.linalg.matrix_exp(A)
        expected_frechet = torch.linalg.matrix_exp(M)[:2, 2:]

        for kwargs in ({}, {"method": "SPS"}, {"method": "blockEnlarge"}):
            observed_expm, observed_frechet = expm_frechet(A, E, **kwargs)
            torch.testing.assert_close(expected_expm, observed_expm)
            torch.testing.assert_close(expected_frechet, observed_frechet)

    def test_small_norm_expm_frechet(self):
        """Test matrices with a range of norms for better coverage using torch tensors."""
        M_original = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        A_original = torch.tensor(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        E_original = torch.tensor(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=dtype,
            device=device,
        )
        A_original_norm_1 = torch.linalg.norm(A_original, 1)
        selected_m_list = [1, 3, 5, 7, 9, 11, 13, 15]
        m_neighbor_pairs = zip(selected_m_list[:-1], selected_m_list[1:], strict=True)
        for ma, mb in m_neighbor_pairs:
            ell_a = ell_table_61[ma]
            ell_b = ell_table_61[mb]
            target_norm_1 = 0.5 * (ell_a + ell_b)
            scale = target_norm_1 / A_original_norm_1
            M = scale * M_original
            A = scale * A_original
            E = scale * E_original
            expected_expm = torch.linalg.matrix_exp(A)
            expected_frechet = torch.linalg.matrix_exp(M)[:2, 2:]
            observed_expm, observed_frechet = expm_frechet(A, E)
            torch.testing.assert_close(expected_expm, observed_expm)
            torch.testing.assert_close(expected_frechet, observed_frechet)

    def test_fuzz(self):
        """Test with a variety of random inputs using torch tensors."""
        rng = np.random.default_rng(1726500908359153)
        # try a bunch of crazy inputs
        # Convert random functions to tensor-generating functions
        tensor_rfuncs = (
            lambda size, device="cpu": torch.tensor(np.random.uniform(size=size), device=device),
            lambda size, device="cpu": torch.tensor(np.random.normal(size=size), device=device),
            lambda size, device="cpu": torch.tensor(
                np.random.standard_cauchy(size=size), device=device
            ),
            lambda size, device="cpu": torch.tensor(
                np.random.exponential(size=size), device=device
            ),
        )
        ntests = 100
        for _ in range(ntests):
            rfunc = tensor_rfuncs[torch.tensor(rng.choice(4))]
            target_norm_1 = torch.tensor(rng.exponential())
            n = torch.tensor(rng.integers(2, 16))
            A_original = rfunc(size=(n, n))
            E_original = rfunc(size=(n, n))
            A_original_norm_1 = torch.linalg.norm(A_original, 1)
            scale = target_norm_1 / A_original_norm_1
            A = scale * A_original
            E = scale * E_original
            M = torch.vstack([torch.hstack([A, E]), torch.hstack([torch.zeros_like(A), A])])
            expected_expm = torch.linalg.matrix_exp(A)
            expected_frechet = torch.linalg.matrix_exp(M)[:n, n:]
            observed_expm, observed_frechet = expm_frechet(A, E)
            torch.testing.assert_close(expected_expm, observed_expm, atol=5e-8, rtol=1e-5)
            torch.testing.assert_close(expected_frechet, observed_frechet, atol=1e-7, rtol=1e-5)

    def test_problematic_matrix(self):
        """Test a specific matrix that previously uncovered a bug using torch tensors."""
        A = torch.tensor(
            [
                [1.50591997, 1.93537998],
                [0.41203263, 0.23443516],
            ],
            dtype=dtype,
            device=device,
        )
        E = torch.tensor(
            [
                [1.87864034, 2.07055038],
                [1.34102727, 0.67341123],
            ],
            dtype=dtype,
            device=device,
        )
        sps_expm, sps_frechet = expm_frechet(A, E, method="SPS")
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(A, E, method="blockEnlarge")
        torch.testing.assert_close(sps_expm, blockEnlarge_expm)
        torch.testing.assert_close(sps_frechet, blockEnlarge_frechet)

    def test_medium_matrix(self):
        """Test with a medium-sized matrix to compare performance
        between methods using torch tensors."""
        n = 1000
        A = torch.tensor(np.random.exponential(size=(n, n)))
        E = torch.tensor(np.random.exponential(size=(n, n)))

        sps_expm, sps_frechet = expm_frechet(A, E, method="SPS")
        blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(A, E, method="blockEnlarge")
        torch.testing.assert_close(sps_expm, blockEnlarge_expm)
        torch.testing.assert_close(sps_frechet, blockEnlarge_frechet)


class TestExpmFrechetTorchGrad:
    """Test suite for gradient computation with expm and its Frechet derivative."""

    def test_expm_frechet(self):
        """Test gradient computation for matrix exponential and its Frechet derivative."""
        M = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [0, 0, 1, 2],
                [0, 0, 5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        A = torch.tensor(
            [
                [1, 2],
                [5, 6],
            ],
            dtype=dtype,
            device=device,
        )
        E = torch.tensor(
            [
                [3, 4],
                [7, 8],
            ],
            dtype=dtype,
            device=device,
        )
        expected_expm = torch.linalg.matrix_exp(A)
        expected_frechet = torch.linalg.matrix_exp(M)[:2, 2:]
        # expm will use the SPS method as default
        observed_expm = expm.apply(A)
        torch.testing.assert_close(expected_expm, observed_expm)
        # Compute the Frechet derivative in the direction of grad_output
        A.requires_grad = True
        observed_expm = expm.apply(A)
        (observed_frechet,) = torch.autograd.grad(observed_expm, A, E, retain_graph=True)
        torch.testing.assert_close(expected_frechet, observed_frechet)
