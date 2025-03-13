"""
This module provides implementation of the matrix logarithm for square matrices.
"""

import warnings

import torch
from scipy.linalg import logm as logm_scipy

from torch_matfunc.matrix._logm_33 import _matrix_log_33


def matrix_log_scipy(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the matrix logarithm of a square matrix using scipy.linalg.logm.

    This function handles tensors on CPU or GPU and preserves gradients.

    Args:
        matrix: A square matrix tensor

    Returns:
        The matrix logarithm of the input matrix
    """
    # Save original device and dtype
    device = matrix.device
    dtype = matrix.dtype
    requires_grad = matrix.requires_grad

    # Detach and move to CPU for scipy
    matrix_cpu = matrix.detach().cpu().numpy()

    # Compute the logarithm using scipy
    result_np = logm_scipy(matrix_cpu)

    # Convert back to tensor and move to original device
    result = torch.tensor(result_np, dtype=dtype, device=device)

    # If input requires gradient, make the output require gradient too
    if requires_grad:
        result = result.requires_grad_()

    return result


def matrix_log_33(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the matrix logarithm of a square 3x3 matrix.

    This function attempts to use the exact formula for 3x3 matrices first,
    and falls back to scipy implementation if that fails.
    """
    try:
        return _matrix_log_33(matrix)
    except Exception as e:
        print(f"Error computing matrix logarithm with _matrix_log_33 {e} \n Falling back to scipy")
        # Fall back to scipy implementation
        return matrix_log_scipy(matrix)


def matrix_log_taylor(matrix: torch.Tensor, order: int = 10) -> torch.Tensor:
    """Compute the matrix logarithm of a square matrix using power series expansion.

    This function calculates log(matrix) using the Taylor series expansion:
    log(I + X) = X - X^2/2 + X^3/3 - X^4/4 + ...

    The function works best when the input matrix is close to the identity matrix.

    Args:
        matrix: A square matrix of shape (..., n, n)
        order: Number of terms to use in the power series expansion (default: 10)

    Returns:
        The matrix logarithm of the input matrix
    """
    warnings.warn("matrix_log_taylor untested use matrix_log_scipy instead", stacklevel=2)
    # Create identity matrix with same shape as input
    identity = torch.eye(matrix.shape[-1], device=matrix.device).expand_as(matrix)

    # Compute X = matrix - I for the expansion
    X = matrix - identity

    # Initialize result and first power of X
    result = torch.zeros_like(matrix)
    X_power = X.clone()

    # Compute the Taylor series up to the specified order
    for n in range(1, order + 1):
        # Add the nth term: (-1)^(n+1) * X^n / n
        term_sign = 1 if n % 2 == 1 else -1
        result = result + term_sign * X_power / n

        # Compute the next power of X
        if n < order:
            X_power = torch.matmul(X_power, X)

    return result
