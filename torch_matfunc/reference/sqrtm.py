"""Pure-Torch reference principal matrix square root: determinantally scaled Denman-Beavers
(Higham ch. 6) on ``A/c`` with ``c = ||A||_F`` per matrix, so ``sqrt(A) = sqrt(c) * Y``."""

import torch

from torch_matfunc.reference.spectral import SUPPORTED_DTYPES


def matrix_sqrt(
    A: torch.Tensor,
    *,
    max_iters: int = 40,
    tol: float | None = None,
    scaled: bool = True,
) -> torch.Tensor:
    """Principal square root of batched square matrices ``(..., n, n)``; the spectrum must
    avoid the closed negative real axis. ``tol`` is a relative Frobenius residual tolerance.

    Accuracy is normwise: eigencomponents far below ``||A||`` converge relatively only up to
    ``tol * ||A|| / lambda``; use an eigh-based path for spectrum-accurate Hermitian input.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("matrix_sqrt expects a square matrix tensor of shape (..., n, n)")
    if A.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"Unsupported dtype {A.dtype}; expected float32/float64/complex64/complex128"
        )

    if tol is None:
        tol = 1e-7 if A.dtype in (torch.float32, torch.complex64) else 1e-14

    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device).expand_as(A)

    # Prescale so the iteration count does not depend on ||A||.
    tiny = torch.finfo(A.dtype).tiny
    c = torch.linalg.matrix_norm(A, ord="fro").clamp(min=tiny)
    if A.is_complex():
        c = c[..., None, None].to(A.dtype)
    else:
        c = c[..., None, None]
    As = A / c

    Y = As.clone()
    Z = I.clone()

    eps = torch.finfo(A.dtype).eps
    # Freeze scaling once r < eps^0.25, where g ~ 1; below eps^0.5 exit when r stops halving.
    freeze, stall_gate = eps**0.25, eps**0.5
    prev_r = float("inf")
    scale_on = scaled

    # One batched LU covers both inverses; stacking measured slower past n=16.
    stack_solves = n <= 16
    for _ in range(max_iters):
        if stack_solves:
            YZ = torch.stack([Y, Z])
            LU, piv = torch.linalg.lu_factor(YZ)
            invs = torch.linalg.lu_solve(LU, piv, I.expand(YZ.shape))
            Y_inv, Z_inv = invs[0], invs[1]
            logdet = LU.diagonal(dim1=-2, dim2=-1).abs().clamp(min=tiny).log().sum(-1).sum(0)
        else:
            LUy, pivy = torch.linalg.lu_factor(Y)
            LUz, pivz = torch.linalg.lu_factor(Z)
            Y_inv = torch.linalg.lu_solve(LUy, pivy, I)
            Z_inv = torch.linalg.lu_solve(LUz, pivz, I)
            logdet = LUy.diagonal(dim1=-2, dim2=-1).abs().clamp(min=tiny).log().sum(
                -1
            ) + LUz.diagonal(dim1=-2, dim2=-1).abs().clamp(min=tiny).log().sum(-1)
        if scale_on:
            # Determinantal scaling g = |det(Y) det(Z)|^(-1/(2n)), free from the LU diagonals.
            g = torch.exp((-logdet / (2 * n)).clamp(min=-20.0, max=20.0))[..., None, None]
            Y, Z = 0.5 * (g * Y + Z_inv / g), 0.5 * (g * Z + Y_inv / g)
        else:
            Y, Z = 0.5 * (Y + Z_inv), 0.5 * (Z + Y_inv)

        # ||As||_F = 1 by construction, so this residual is relative.
        residual = torch.linalg.matrix_norm(Y @ Y - As, ord="fro")
        if residual.numel() == 0:
            break
        r = float(residual.detach().max())
        if r < tol or (r < stall_gate and r >= 0.5 * prev_r):
            break
        scale_on = scaled and r > freeze
        prev_r = r

    return torch.sqrt(c) * Y
