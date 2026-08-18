"""Tests for the 3x3 matrix logarithm implementation."""

import scipy
import torch

from torch_matfunc.matrix._logm_33 import _matrix_log_33
from torch_matfunc.matrix.logm import matrix_log_33, matrix_log_scipy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64


class TestLogM_33:
    """Test suite for the 3x3 matrix logarithm implementation.

    This class contains tests that verify the correctness of the matrix logarithm
    implementation for 3x3 matrices against analytical solutions, scipy implementation,
    and various edge cases.
    """

    def test_logm_33_reference(self):
        """Test matrix logarithm implementation for 3x3 matrices against analytical solutions.

        Tests against scipy implementation as well.

        This test verifies the implementation against known analytical solutions from the paper:
        https://link.springer.com/article/10.1007/s10659-008-9169-x

        I test several cases:
        - Case 1b: All eigenvalues equal with q(T) = (T - λI)²
        - Case 1c: All eigenvalues equal with q(T) = (T - λI)³
        - Case 2b: Two distinct eigenvalues with q(T) = (T - μI)(T - λI)²
        - Identity matrix (should return zero matrix)
        - Diagonal matrix with distinct eigenvalues (Case 3)
        """
        # Set precision for comparisons
        rtol = 1e-5
        atol = 1e-8

        # Case 1b: All eigenvalues equal with q(T) = (T - λI)²
        # Example: T = [[e, 1, 0], [0, e, 0], [0, 0, e]]
        e_val = torch.exp(torch.tensor(1.0))  # e = exp(1)
        T_1b = torch.tensor(
            [[e_val, 1.0, 0.0], [0.0, e_val, 0.0], [0.0, 0.0, e_val]], dtype=dtype, device=device
        )

        # Expected solution: log T = [[1, 1/e, 0], [0, 1, 0], [0, 0, 1]]
        expected_1b = torch.tensor(
            [[1.0, 1.0 / e_val, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device
        )

        # Compute using our implementation and compare
        result_1b = _matrix_log_33(T_1b)
        (
            torch.testing.assert_close(result_1b, expected_1b, rtol=rtol, atol=atol),
            f"Case 1b failed: \nExpected:\n{expected_1b}\nGot:\n{result_1b}",
        )

        # Compare with scipy
        scipy_result_1b = matrix_log_scipy(T_1b)
        (
            torch.testing.assert_close(result_1b, scipy_result_1b, rtol=rtol, atol=atol),
            f"Case 1b differs from scipy: \nExpected:\n{scipy_result_1b}\nGot:\n{result_1b}",
        )

        # Case 1c: All eigenvalues equal with q(T) = (T - λI)³
        # Example: T = [[e, 1, 1], [0, e, 1], [0, 0, e]]
        T_1c = torch.tensor(
            [[e_val, 1.0, 1.0], [0.0, e_val, 1.0], [0.0, 0.0, e_val]], dtype=dtype, device=device
        )

        # Expected solution: log T = [[1, 1/e, (2e-1)/(2e²)], [0, 1, 1/e], [0, 0, 1]]
        expected_1c = torch.tensor(
            [
                [1.0, 1.0 / e_val, (2 * e_val - 1) / (2 * e_val * e_val)],
                [0.0, 1.0, 1.0 / e_val],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )

        # Compute using our implementation and compare
        result_1c = _matrix_log_33(T_1c)
        (
            torch.testing.assert_close(result_1c, expected_1c, rtol=rtol, atol=atol),
            f"Case 1c failed: \nExpected:\n{expected_1c}\nGot:\n{result_1c}",
        )

        # Compare with scipy
        scipy_result_1c = matrix_log_scipy(T_1c)
        (
            torch.testing.assert_close(result_1c, scipy_result_1c, rtol=rtol, atol=atol),
            f"Case 1c differs from scipy: \nExpected:\n{scipy_result_1c}\nGot:\n{result_1c}",
        )

        # Case 2b: Two distinct eigenvalues with q(T) = (T - μI)(T - λI)²
        # Example: T = [[e, 1, 1], [0, e², 1], [0, 0, e²]]
        e_squared = e_val * e_val
        e_cubed = e_squared * e_val
        T_2b = torch.tensor(
            [[e_val, 1.0, 1.0], [0.0, e_squared, 1.0], [0.0, 0.0, e_squared]],
            dtype=dtype,
            device=device,
        )

        # Expected solution: log T = [[1, 1/(e(e-1)), (e³-e²-1)/(e³(e-1)²)],
        # [0, 2, 1/e²], [0, 0, 2]]
        expected_2b = torch.tensor(
            [
                [
                    1.0,
                    1.0 / (e_val * (e_val - 1.0)),
                    (e_cubed - e_squared - 1) / (e_cubed * (e_val - 1.0) * (e_val - 1.0)),
                ],
                [0.0, 2.0, 1.0 / e_squared],
                [0.0, 0.0, 2.0],
            ],
            dtype=dtype,
            device=device,
        )

        # Compute using our implementation and compare
        result_2b = _matrix_log_33(T_2b)
        (
            torch.testing.assert_close(result_2b, expected_2b, rtol=rtol, atol=atol),
            f"Case 2b failed: \nExpected:\n{expected_2b}\nGot:\n{result_2b}",
        )

        # Compare with scipy
        scipy_result_2b = matrix_log_scipy(T_2b)
        (
            torch.testing.assert_close(result_2b, scipy_result_2b, rtol=rtol, atol=atol),
            f"Case 2b differs from scipy: \nExpected:\n{scipy_result_2b}\nGot:\n{result_2b}",
        )

        # Additional test: identity matrix (should return zero matrix)
        identity = torch.eye(3, dtype=dtype, device=device)
        log_identity = _matrix_log_33(identity)
        expected_log_identity = torch.zeros((3, 3), dtype=dtype, device=device)
        (
            torch.testing.assert_close(log_identity, expected_log_identity, rtol=rtol, atol=atol),
            f"log(I) failed: \nExpected:\n{expected_log_identity}\nGot:\n{log_identity}",
        )

        # Additional test: diagonal matrix with distinct eigenvalues (Case 3)
        D = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=dtype, device=device))
        log_D = _matrix_log_33(D)
        expected_log_D = torch.diag(
            torch.log(torch.tensor([2.0, 3.0, 4.0], dtype=dtype, device=device))
        )
        (
            torch.testing.assert_close(log_D, expected_log_D, rtol=rtol, atol=atol),
            f"log(diag) failed: \nExpected:\n{expected_log_D}\nGot:\n{log_D}",
        )

    def test_random_float(self):
        """Test matrix logarithm on random 3x3 matrices.

        This test generates a random 3x3 matrix and compares the implementation
        against scipy's implementation to ensure consistency.
        """
        torch.manual_seed(1234)
        n = 3
        M = torch.randn(n, n, dtype=dtype, device=device)
        M_logm = matrix_log_33(M)
        scipy_logm = scipy.linalg.logm(M.cpu().numpy())
        torch.testing.assert_close(M_logm, torch.tensor(scipy_logm, dtype=dtype, device=device))

    def test_nearly_degenerate(self):
        """Test matrix logarithm on nearly degenerate matrices.

        This test verifies that the implementation handles matrices with
        nearly degenerate eigenvalues correctly by comparing against scipy's
        implementation.
        """
        eps = 1e-6
        M = torch.tensor(
            [[1.0, 1.0, eps], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device
        )
        M_logm = matrix_log_33(M)
        scipy_logm = scipy.linalg.logm(M.cpu().numpy())
        torch.testing.assert_close(M_logm, torch.tensor(scipy_logm, dtype=dtype, device=device))
