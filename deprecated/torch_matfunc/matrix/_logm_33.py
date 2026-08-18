"""
This module provides implementation of the matrix logarithm for 3x3 matrices.

This is the analytical solution for the matrix logarithm of 3x3 matrices.
It is based on the paper:
https://link.springer.com/article/10.1007/s10659-008-9169-x

The implementation is based on the paper and the references therein.
"""

import torch


def _is_valid_matrix(T: torch.Tensor, n: int = 3) -> bool:
    """
    Check if T is a valid nxn matrix.

    Parameters:
    -----------
    T : torch.Tensor
        The matrix to check
    n : int, default=3
        The expected dimension of the matrix

    Returns:
    --------
    bool
        True if T is a valid nxn tensor, False otherwise
    """
    if not isinstance(T, torch.Tensor):
        return False
    if T.shape != (n, n):
        return False
    return True


def _determine_eigenvalue_case(
    T: torch.Tensor, eigenvalues: torch.Tensor, num_tol: float = 1e-10
) -> str:
    """
    Determine the eigenvalue structure case of matrix T.

    Parameters:
    -----------
    T : torch.Tensor
        The 3x3 matrix to analyze
    eigenvalues : torch.Tensor
        The eigenvalues of T
    num_tol : float, default=1e-10
        Numerical tolerance for comparing eigenvalues

    Returns:
    --------
    str
        The case identifier ("case1a", "case1b", etc.)

    Raises:
    -------
    ValueError
        If the eigenvalue structure cannot be determined
    """
    # Get unique values and their counts directly with one call
    unique_vals, counts = torch.unique(eigenvalues, return_counts=True)

    # Use np.isclose to group eigenvalues that are numerically close
    # We can create a mask for each unique value to see if other values are close to it
    if len(unique_vals) > 1:
        # Check if some "unique" values should actually be considered the same
        i = 0
        while i < len(unique_vals):
            # Find all values close to the current one
            close_mask = torch.isclose(unique_vals, unique_vals[i], rtol=0, atol=num_tol)
            close_count = torch.sum(close_mask)

            if close_count > 1:  # If there are other close values
                # Merge them (keep the first one, remove the others)
                counts[i] = torch.sum(counts[close_mask])
                unique_vals = unique_vals[~(close_mask & torch.arange(len(close_mask)) != i)]
                counts = counts[~(close_mask & torch.arange(len(counts)) != i)]
            else:
                i += 1

    # Now determine the case based on the number of unique eigenvalues
    if len(unique_vals) == 1:
        # Case 1: All eigenvalues are equal (λ, λ, λ)
        lambda_val = unique_vals[0]
        Identity = torch.eye(3, dtype=lambda_val.dtype, device=lambda_val.device)
        T_minus_lambdaI = T - lambda_val * Identity

        rank1 = torch.linalg.matrix_rank(T_minus_lambdaI)
        if rank1 == 0:
            return "case1a"  # q(T) = (T - λI)

        rank2 = torch.linalg.matrix_rank(T_minus_lambdaI @ T_minus_lambdaI)
        if rank2 == 0:
            return "case1b"  # q(T) = (T - λI)²

        return "case1c"  # q(T) = (T - λI)³

    elif len(unique_vals) == 2:
        # Case 2: Two distinct eigenvalues
        # The counts array already tells us which eigenvalue is repeated
        if counts.max() != 2 or counts.min() != 1:
            raise ValueError("Unexpected eigenvalue pattern for Case 2")

        mu = unique_vals[torch.argmin(counts)]  # The non-repeated eigenvalue
        lambda_val = unique_vals[torch.argmax(counts)]  # The repeated eigenvalue

        Identity = torch.eye(3, dtype=lambda_val.dtype, device=lambda_val.device)
        T_minus_muI = T - mu * Identity
        T_minus_lambdaI = T - lambda_val * Identity

        # Check if (T - μI)(T - λI) annihilates T
        if torch.allclose(
            T_minus_muI @ T_minus_lambdaI @ T,
            torch.zeros((3, 3), dtype=lambda_val.dtype, device=lambda_val.device),
        ):
            return "case2a"  # q(T) = (T - λI)(T - μI)
        else:
            return "case2b"  # q(T) = (T - μI)(T - λI)²

    elif len(unique_vals) == 3:
        # Case 3: Three distinct eigenvalues (λ, μ, ν)
        return "case3"  # q(T) = (T - λI)(T - μI)(T - νI)

    else:
        raise ValueError("Could not determine eigenvalue structure")


def _matrix_log_case1a(T: torch.Tensor, lambda_val: complex) -> torch.Tensor:
    """
    Compute log(T) when q(T) = (T - λI).
    This is the case where T is a scalar multiple of the identity matrix.

    Parameters:
    -----------
    T : torch.Tensor
        The matrix whose logarithm is to be computed
    lambda_val : complex
        The eigenvalue of T

    Returns:
    --------
    torch.Tensor
        The logarithm of T, which is log(λ)·I
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    return torch.log(lambda_val) * Identity


def _matrix_log_case1b(T: torch.Tensor, lambda_val: complex) -> torch.Tensor:
    """
    Compute log(T) when q(T) = (T - λI)².
    This is the case where T has a Jordan block of size 2.

    Parameters:
    -----------
    T : torch.Tensor
        The matrix whose logarithm is to be computed
    lambda_val : complex
        The eigenvalue of T

    Returns:
    --------
    torch.Tensor
        The logarithm of T
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    T_minus_lambdaI = T - lambda_val * Identity

    # For numerical stability, scale appropriately
    if abs(lambda_val) > 1:
        scaled_T_minus_lambdaI = T_minus_lambdaI / lambda_val
        return torch.log(lambda_val) * Identity + scaled_T_minus_lambdaI
    else:
        # Alternative computation for small lambda
        return torch.log(lambda_val) * Identity + T_minus_lambdaI / max(lambda_val, 1e-10)


def _matrix_log_case1c(T: torch.Tensor, lambda_val: complex) -> torch.Tensor:
    """
    Compute log(T) when q(T) = (T - λI)³.
    This is the case where T has a Jordan block of size 3.

    Parameters:
    -----------
    T : torch.Tensor
        The matrix whose logarithm is to be computed
    lambda_val : complex
        The eigenvalue of T

    Returns:
    --------
    torch.Tensor
        The logarithm of T
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    T_minus_lambdaI = T - lambda_val * Identity

    # Compute (T - λI)² with better numerical stability
    T_minus_lambdaI_squared = T_minus_lambdaI @ T_minus_lambdaI

    # For numerical stability
    lambda_squared = lambda_val * lambda_val

    term1 = torch.log(lambda_val) * Identity
    term2 = T_minus_lambdaI / max(lambda_val, 1e-10)
    term3 = T_minus_lambdaI_squared / max(2 * lambda_squared, 1e-10)

    return term1 + term2 - term3


def _matrix_log_case2a(
    T: torch.Tensor, lambda_val: complex, mu: complex, num_tol: float = 1e-10
) -> torch.Tensor:
    """
    Compute log(T) when q(T) = (T - λI)(T - μI) with λ≠μ.
    This is the case with two distinct eigenvalues.

    Formula: log T = log μ((T - λI)/(μ - λ)) + log λ((T - μI)/(λ - μ))

    Parameters:
    -----------
    T : torch.Tensor
        The matrix whose logarithm is to be computed
    lambda_val : complex
        The repeated eigenvalue of T
    mu : complex
        The non-repeated eigenvalue of T
    num_tol : float, default=1e-10
        Numerical tolerance for stability checks

    Returns:
    --------
    torch.Tensor
        The logarithm of T

    Raises:
    -------
    ValueError
        If λ and μ are too close for numerical stability
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    lambda_minus_mu = lambda_val - mu

    # Check for numerical stability
    if torch.abs(lambda_minus_mu) < num_tol:
        raise ValueError("λ and μ are too close, computation may be unstable")

    T_minus_lambdaI = T - lambda_val * Identity
    T_minus_muI = T - mu * Identity

    # Compute each term separately for better numerical stability
    term1 = torch.log(mu) * (T_minus_lambdaI / (mu - lambda_val))
    term2 = torch.log(lambda_val) * (T_minus_muI / (lambda_val - mu))

    return term1 + term2


def _matrix_log_case2b(
    T: torch.Tensor, lambda_val: complex, mu: complex, num_tol: float = 1e-10
) -> torch.Tensor:
    """
    Compute log(T) when q(T) = (T - μI)(T - λI)² with λ≠μ.
    This is the case with one eigenvalue of multiplicity 2 and one distinct eigenvalue.

    Formula: log T = log μ((T - λI)²/(λ - μ)²) -
             log λ((T - μI)(T - (2λ - μ)I)/(λ - μ)²) +
             ((T - λI)(T - μI)/(λ(λ - μ)))

    Parameters:
    -----------
    T : torch.Tensor
        The matrix whose logarithm is to be computed
    lambda_val : complex
        The repeated eigenvalue of T
    mu : complex
        The non-repeated eigenvalue of T
    num_tol : float, default=1e-10
        Numerical tolerance for stability checks

    Returns:
    --------
    torch.Tensor
        The logarithm of T

    Raises:
    -------
    ValueError
        If λ and μ are too close for numerical stability or if λ is too close to zero
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)
    lambda_minus_mu = lambda_val - mu
    lambda_minus_mu_squared = lambda_minus_mu * lambda_minus_mu

    # Check for numerical stability
    if torch.abs(lambda_minus_mu) < num_tol:
        raise ValueError("λ and μ are too close, computation may be unstable")

    if torch.abs(lambda_val) < num_tol:
        raise ValueError("λ is too close to zero, computation may be unstable")

    T_minus_lambdaI = T - lambda_val * Identity
    T_minus_muI = T - mu * Identity
    T_minus_lambdaI_squared = T_minus_lambdaI @ T_minus_lambdaI

    # The term (T - (2λ - μ)I)
    T_minus_2lambda_plus_muI = T - (2 * lambda_val - mu) * Identity

    # Compute each term separately for better numerical stability
    term1 = torch.log(mu) * (T_minus_lambdaI_squared / lambda_minus_mu_squared)
    term2 = -torch.log(lambda_val) * (
        (T_minus_muI @ T_minus_2lambda_plus_muI) / lambda_minus_mu_squared
    )
    term3 = (T_minus_lambdaI @ T_minus_muI) / (lambda_val * lambda_minus_mu)

    return term1 + term2 + term3


def _matrix_log_case3(
    T: torch.Tensor, lambda_val: complex, mu: complex, nu: complex, num_tol: float = 1e-10
) -> torch.Tensor:
    """
    Compute log(T) when q(T) = (T - λI)(T - μI)(T - νI) with λ≠μ≠ν≠λ.
    This is the case with three distinct eigenvalues.

    Formula: log T = log λ((T - μI)(T - νI)/((λ - μ)(λ - ν)))
                    + log μ((T - λI)(T - νI)/((μ - λ)(μ - ν)))
                    + log ν((T - λI)(T - μI)/((ν - λ)(ν - μ)))

    Parameters:
    -----------
    T : torch.Tensor
        The matrix whose logarithm is to be computed
    lambda_val : complex
        First eigenvalue of T
    mu : complex
        Second eigenvalue of T
    nu : complex
        Third eigenvalue of T
    num_tol : float, default=1e-10
        Numerical tolerance for stability checks

    Returns:
    --------
    torch.Tensor
        The logarithm of T

    Raises:
    -------
    ValueError
        If any pair of eigenvalues are too close for numerical stability
    """
    n = T.shape[0]
    Identity = torch.eye(n, dtype=lambda_val.dtype, device=lambda_val.device)

    # Check if eigenvalues are distinct enough for numerical stability
    if min(torch.abs(lambda_val - mu), torch.abs(lambda_val - nu), torch.abs(mu - nu)) < num_tol:
        raise ValueError("Eigenvalues are too close, computation may be unstable")

    T_minus_lambdaI = T - lambda_val * Identity
    T_minus_muI = T - mu * Identity
    T_minus_nuI = T - nu * Identity

    # Compute the terms for λ
    lambda_term_numerator = T_minus_muI @ T_minus_nuI
    lambda_term_denominator = (lambda_val - mu) * (lambda_val - nu)
    lambda_term = torch.log(lambda_val) * (lambda_term_numerator / lambda_term_denominator)

    # Compute the terms for μ
    mu_term_numerator = T_minus_lambdaI @ T_minus_nuI
    mu_term_denominator = (mu - lambda_val) * (mu - nu)
    mu_term = torch.log(mu) * (mu_term_numerator / mu_term_denominator)

    # Compute the terms for ν
    nu_term_numerator = T_minus_lambdaI @ T_minus_muI
    nu_term_denominator = (nu - lambda_val) * (nu - mu)
    nu_term = torch.log(nu) * (nu_term_numerator / nu_term_denominator)

    return lambda_term + mu_term + nu_term


def _matrix_log_33(
    T: torch.Tensor, case: str = "auto", dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """
    Compute the logarithm of 3x3 matrix T based on its eigenvalue structure.
    The logarithm of this matrix is known exactly as given the in the references.

    Parameters:
    -----------
    T : torch.Tensor
        The matrix whose logarithm is to be computed
    case : str
        One of "auto", "case1a", "case1b", "case1c", "case2a", "case2b", "case3"
        - "auto": Automatically determine the structure
        - "case1a": All eigenvalues are equal, q(T) = (T - λI)
        - "case1b": All eigenvalues are equal, q(T) = (T - λI)²
        - "case1c": All eigenvalues are equal, q(T) = (T - λI)³
        - "case2a": Two distinct eigenvalues, q(T) = (T - λI)(T - μI)
        - "case2b": Two distinct eigenvalues, q(T) = (T - μI)(T - λI)²
        - "case3": Three distinct eigenvalues, q(T) = (T - λI)(T - μI)(T - νI)
    dtype : torch.dtype, default=torch.float64
        The data type to use for numerical tolerance

    Returns:
    --------
    torch.Tensor
        The logarithm of T

    References:
    -----------
    - https://link.springer.com/article/10.1007/s10659-008-9169-x
    """

    num_tol = 1e-16 if dtype == torch.float64 else 1e-8

    if not _is_valid_matrix(T):
        raise ValueError("Input must be a 3x3 matrix")

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(T)
    # Convert eigenvalues to real if they're complex but with tiny imaginary parts
    eigenvalues = (
        torch.real(eigenvalues)
        if torch.allclose(
            torch.imag(eigenvalues), torch.zeros_like(torch.imag(eigenvalues)), atol=num_tol
        )
        else eigenvalues
    )

    # If automatic detection, determine the structure
    if case == "auto":
        case = _determine_eigenvalue_case(T, eigenvalues, num_tol)

    # Case 1: All eigenvalues are equal (λ, λ, λ)
    if case in ["case1a", "case1b", "case1c"]:
        lambda_val = eigenvalues[0]

        # Check for numerical stability
        if torch.abs(lambda_val) < num_tol:
            raise ValueError("Eigenvalue too close to zero, computation may be unstable")

        if case == "case1a":
            return _matrix_log_case1a(T, lambda_val)
        elif case == "case1b":
            return _matrix_log_case1b(T, lambda_val)
        elif case == "case1c":
            return _matrix_log_case1c(T, lambda_val)

    # Case 2: Two distinct eigenvalues (μ, λ, λ)
    elif case in ["case2a", "case2b"]:
        # Find the unique eigenvalue (μ) and the repeated eigenvalue (λ)
        unique_vals, counts = torch.unique(
            torch.round(eigenvalues, decimals=10), return_counts=True
        )
        if len(unique_vals) != 2 or counts.max() != 2:
            raise ValueError("Case 2 requires exactly two distinct eigenvalues with one repeated")

        mu = unique_vals[torch.argmin(counts)]  # The non-repeated eigenvalue
        lambda_val = unique_vals[torch.argmax(counts)]  # The repeated eigenvalue

        if case == "case2a":
            return _matrix_log_case2a(T, lambda_val, mu, num_tol)
        elif case == "case2b":
            return _matrix_log_case2b(T, lambda_val, mu, num_tol)

    # Case 3: Three distinct eigenvalues (λ, μ, ν)
    elif case == "case3":
        if len(torch.unique(torch.round(eigenvalues, decimals=10))) != 3:
            raise ValueError("Case 3 requires three distinct eigenvalues")

        lambda_val, mu, nu = torch.sort(eigenvalues).values  # Sort for consistency
        return _matrix_log_case3(T, lambda_val, mu, nu, num_tol)

    else:
        raise ValueError(f"Unknown case: {case}")
