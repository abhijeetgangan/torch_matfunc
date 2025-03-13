"""Frechet derivative of the matrix exponential.

Adapted from scipy.linalg._expm_frechet.py
https://github.com/scipy/scipy/blob/v1.15.2/scipy/linalg/_expm_frechet.py#L0-L1
"""

import torch
from torch.autograd import Function

__all__ = ["expm_frechet", "expm_cond"]


def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):
    """
    Frechet derivative of the matrix exponential of A in the direction E.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix of which to take the matrix exponential.
    E : (N, N) array_like
        Matrix direction in which to take the Frechet derivative.
    method : str, optional
        Choice of algorithm. Should be one of

        - `SPS` (default)
        - `blockEnlarge`

    compute_expm : bool, optional
        Whether to compute also `expm_A` in addition to `expm_frechet_AE`.
        Default is True.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    expm_A : ndarray
        Matrix exponential of A.
    expm_frechet_AE : ndarray
        Frechet derivative of the matrix exponential of A in the direction E.
    For ``compute_expm = False``, only `expm_frechet_AE` is returned.
    """
    if check_finite:
        if not torch.isfinite(A).all():
            raise ValueError("Matrix A contains non-finite values")
        if not torch.isfinite(E).all():
            raise ValueError("Matrix E contains non-finite values")

    # Convert inputs to torch tensors if they aren't already
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float64)
    if not isinstance(E, torch.Tensor):
        E = torch.tensor(E, dtype=torch.float64)

    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected A to be a square matrix")
    if E.dim() != 2 or E.shape[0] != E.shape[1]:
        raise ValueError("expected E to be a square matrix")
    if A.shape != E.shape:
        raise ValueError("expected A and E to be the same shape")

    if method is None:
        method = "SPS"

    if method == "SPS":
        expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
    elif method == "blockEnlarge":
        expm_A, expm_frechet_AE = expm_frechet_block_enlarge(A, E)
    else:
        raise ValueError(f"Unknown implementation {method}")

    if compute_expm:
        return expm_A, expm_frechet_AE
    else:
        return expm_frechet_AE


def expm_frechet_block_enlarge(A, E):
    """
    This is a helper function, mostly for testing and profiling.
    Return expm(A), frechet(A, E)
    """
    n = A.shape[0]
    # Create block matrix M = [[A, E], [0, A]]
    M = torch.zeros((2 * n, 2 * n), dtype=A.dtype, device=A.device)
    M[:n, :n] = A
    M[:n, n:] = E
    M[n:, n:] = A

    # Use matrix exponential
    expm_M = matrix_exp(M)
    return expm_M[:n, :n], expm_M[:n, n:]


# Maximal values ell_m of ||2**-s A|| such that the backward error bound
# does not exceed 2**-53.
ell_table_61 = (
    None,
    # 1
    2.11e-8,
    3.56e-4,
    1.08e-2,
    6.49e-2,
    2.00e-1,
    4.37e-1,
    7.83e-1,
    1.23e0,
    1.78e0,
    2.42e0,
    # 11
    3.13e0,
    3.90e0,
    4.74e0,
    5.63e0,
    6.56e0,
    7.52e0,
    8.53e0,
    9.56e0,
    1.06e1,
    1.17e1,
)


def _diff_pade3(A, E, ident):
    b = (120.0, 60.0, 12.0, 1.0)
    A2 = torch.matmul(A, A)
    M2 = torch.matmul(A, E) + torch.matmul(E, A)
    U = torch.matmul(A, b[3] * A2 + b[1] * ident)
    V = b[2] * A2 + b[0] * ident
    Lu = torch.matmul(A, b[3] * M2) + torch.matmul(E, b[3] * A2 + b[1] * ident)
    Lv = b[2] * M2
    return U, V, Lu, Lv


def _diff_pade5(A, E, ident):
    b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
    A2 = torch.matmul(A, A)
    M2 = torch.matmul(A, E) + torch.matmul(E, A)
    A4 = torch.matmul(A2, A2)
    M4 = torch.matmul(A2, M2) + torch.matmul(M2, A2)
    U = torch.matmul(A, b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = torch.matmul(A, b[5] * M4 + b[3] * M2) + torch.matmul(
        E, b[5] * A4 + b[3] * A2 + b[1] * ident
    )
    Lv = b[4] * M4 + b[2] * M2
    return U, V, Lu, Lv


def _diff_pade7(A, E, ident):
    b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
    A2 = torch.matmul(A, A)
    M2 = torch.matmul(A, E) + torch.matmul(E, A)
    A4 = torch.matmul(A2, A2)
    M4 = torch.matmul(A2, M2) + torch.matmul(M2, A2)
    A6 = torch.matmul(A2, A4)
    M6 = torch.matmul(A4, M2) + torch.matmul(M4, A2)
    U = torch.matmul(A, b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = torch.matmul(A, b[7] * M6 + b[5] * M4 + b[3] * M2) + torch.matmul(
        E, b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
    )
    Lv = b[6] * M6 + b[4] * M4 + b[2] * M2
    return U, V, Lu, Lv


def _diff_pade9(A, E, ident):
    b = (
        17643225600.0,
        8821612800.0,
        2075673600.0,
        302702400.0,
        30270240.0,
        2162160.0,
        110880.0,
        3960.0,
        90.0,
        1.0,
    )
    A2 = torch.matmul(A, A)
    M2 = torch.matmul(A, E) + torch.matmul(E, A)
    A4 = torch.matmul(A2, A2)
    M4 = torch.matmul(A2, M2) + torch.matmul(M2, A2)
    A6 = torch.matmul(A2, A4)
    M6 = torch.matmul(A4, M2) + torch.matmul(M4, A2)
    A8 = torch.matmul(A4, A4)
    M8 = torch.matmul(A4, M4) + torch.matmul(M4, A4)
    U = torch.matmul(A, b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = torch.matmul(A, b[9] * M8 + b[7] * M6 + b[5] * M4 + b[3] * M2) + torch.matmul(
        E, b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
    )
    Lv = b[8] * M8 + b[6] * M6 + b[4] * M4 + b[2] * M2
    return U, V, Lu, Lv


def expm_frechet_algo_64(A, E):
    n = A.shape[0]
    s = None
    ident = torch.eye(n, dtype=A.dtype, device=A.device)
    A_norm_1 = torch.norm(A, p=1)
    m_pade_pairs = ((3, _diff_pade3), (5, _diff_pade5), (7, _diff_pade7), (9, _diff_pade9))

    for m, pade in m_pade_pairs:
        if A_norm_1 <= ell_table_61[m]:
            U, V, Lu, Lv = pade(A, E, ident)
            s = 0
            break

    if s is None:
        # scaling
        s = max(0, int(torch.ceil(torch.log2(A_norm_1 / ell_table_61[13]))))
        A = A * 2.0**-s
        E = E * 2.0**-s
        # pade order 13
        A2 = torch.matmul(A, A)
        M2 = torch.matmul(A, E) + torch.matmul(E, A)
        A4 = torch.matmul(A2, A2)
        M4 = torch.matmul(A2, M2) + torch.matmul(M2, A2)
        A6 = torch.matmul(A2, A4)
        M6 = torch.matmul(A4, M2) + torch.matmul(M4, A2)
        b = (
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        )
        W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
        W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
        Z1 = b[12] * A6 + b[10] * A4 + b[8] * A2
        Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
        W = torch.matmul(A6, W1) + W2
        U = torch.matmul(A, W)
        V = torch.matmul(A6, Z1) + Z2
        Lw1 = b[13] * M6 + b[11] * M4 + b[9] * M2
        Lw2 = b[7] * M6 + b[5] * M4 + b[3] * M2
        Lz1 = b[12] * M6 + b[10] * M4 + b[8] * M2
        Lz2 = b[6] * M6 + b[4] * M4 + b[2] * M2
        Lw = torch.matmul(A6, Lw1) + torch.matmul(M6, W1) + Lw2
        Lu = torch.matmul(A, Lw) + torch.matmul(E, W)
        Lv = torch.matmul(A6, Lz1) + torch.matmul(M6, Z1) + Lz2

    # Solve the system (-U + V)X = (U + V) for R
    R = torch.linalg.solve(-U + V, U + V)

    # Solve the system (-U + V)X = (Lu + Lv + (Lu - Lv)R) for L
    L = torch.linalg.solve(-U + V, Lu + Lv + torch.matmul(Lu - Lv, R))

    # squaring
    for _ in range(s):
        L = torch.matmul(R, L) + torch.matmul(L, R)
        R = torch.matmul(R, R)

    return R, L


def matrix_exp(A):
    """
    Compute the matrix exponential of A using PyTorch's matrix_exp.

    This is a replacement for scipy.linalg.expm
    """
    return torch.matrix_exp(A)


def vec(M):
    """
    Stack columns of M to construct a single vector.

    This is somewhat standard notation in linear algebra.

    Parameters
    ----------
    M : 2-D array_like
        Input matrix

    Returns
    -------
    v : 1-D ndarray
        Output vector
    """
    return M.t().reshape(-1)


def expm_frechet_kronform(A, method=None, check_finite=True):
    """
    Construct the Kronecker form of the Frechet derivative of expm.

    Parameters
    ----------
    A : array_like with shape (N, N)
        Matrix to be expm'd.
    method : str, optional
        Extra keyword to be passed to expm_frechet.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    K : 2-D ndarray with shape (N*N, N*N)
        Kronecker form of the Frechet derivative of the matrix exponential.
    """
    if check_finite:
        if not torch.isfinite(A).all():
            raise ValueError("Matrix A contains non-finite values")

    # Convert input to torch tensor if it isn't already
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float64)

    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected a square matrix")

    n = A.shape[0]
    ident = torch.eye(n, dtype=A.dtype, device=A.device)
    cols = []

    for i in range(n):
        for j in range(n):
            E = torch.outer(ident[i], ident[j])
            F = expm_frechet(A, E, method=method, compute_expm=False, check_finite=False)
            cols.append(vec(F))

    return torch.stack(cols, dim=1)


def expm_cond(A, check_finite=True):
    """
    Relative condition number of the matrix exponential in the Frobenius norm.

    Parameters
    ----------
    A : 2-D array_like
        Square input matrix with shape (N, N).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    kappa : float
        The relative condition number of the matrix exponential
        in the Frobenius norm
    """
    if check_finite:
        if not torch.isfinite(A).all():
            raise ValueError("Matrix A contains non-finite values")

    # Convert input to torch tensor if it isn't already
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float64)

    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected a square matrix")

    X = matrix_exp(A)
    K = expm_frechet_kronform(A, check_finite=False)

    # The following norm choices are deliberate.
    # The norms of A and X are Frobenius norms,
    # and the norm of K is the induced 2-norm.
    A_norm = torch.norm(A, p="fro")
    X_norm = torch.norm(X, p="fro")
    K_norm = torch.linalg.matrix_norm(K, ord=2)

    kappa = (K_norm * A_norm) / X_norm
    return kappa


class expm(Function):
    @staticmethod
    def forward(ctx, A):
        """
        Compute the matrix exponential of A.

        Parameters
        ----------
        A : torch.Tensor
            Input matrix or batch of matrices

        Returns
        -------
        torch.Tensor
            Matrix exponential of A
        """
        # Save A for backward pass
        ctx.save_for_backward(A)
        # Use the matrix_exp function we already have
        return matrix_exp(A)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradient of matrix exponential.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient with respect to the output

        Returns
        -------
        torch.Tensor
            Gradient with respect to the input
        """
        # Retrieve saved tensor
        (A,) = ctx.saved_tensors

        # Compute the Frechet derivative in the direction of grad_output
        gradient = expm_frechet(
            A, grad_output, method="SPS", compute_expm=False, check_finite=False
        )
        return gradient
