"""Pure-Torch reference matrix logarithm: transformation-free inverse scaling-and-squaring per
Al-Mohy & Higham (2012): square roots until near ``I``, degree-13 Pade, scale by ``2^s``."""

import torch

from torch_matfunc.reference.spectral import SUPPORTED_DTYPES, magma_pin
from torch_matfunc.reference.sqrtm import matrix_sqrt

MAX_SQRT = 24
# Degree-13 Pade convergence radii; only fp64 is the Al-Mohy & Higham theta_13.
THETA_FP64 = 0.652
THETA_FP32 = 0.20

# Degree-13 Gauss-Legendre nodes/weights, mapped from [-1, 1] to [0, 1].
PADE_NODES = (
    0.007908472640705932,
    0.04120080038851104,
    0.09921095463334506,
    0.17882533027982983,
    0.27575362448177654,
    0.3847708420224326,
    0.5,
    0.6152291579775674,
    0.7242463755182235,
    0.8211746697201702,
    0.9007890453666549,
    0.958799199611489,
    0.9920915273592941,
)
PADE_WEIGHTS = (
    0.02024200238265807,
    0.0460607499188638,
    0.0694367551098938,
    0.08907299038097277,
    0.1039080237684443,
    0.11314159013144873,
    0.116275776615437,
    0.11314159013144873,
    0.1039080237684443,
    0.08907299038097277,
    0.0694367551098938,
    0.0460607499188638,
    0.02024200238265807,
)


def log1p_pade(E: torch.Tensor) -> torch.Tensor:
    """m=13 Gauss-Legendre Pade approximant of ``log(I + E)``: ``sum_j w_j (I + x_j E)^{-1} E``."""
    n = E.shape[-1]
    I = torch.eye(n, dtype=E.dtype, device=E.device).expand_as(E)
    # The m partial-fraction solves are independent, so run them as one solve batched over m.
    shape = (len(PADE_NODES),) + (1,) * E.ndim
    nodes = torch.tensor(PADE_NODES, dtype=E.dtype, device=E.device).view(shape)
    weights = torch.tensor(PADE_WEIGHTS, dtype=E.dtype, device=E.device).view(shape)
    # solve_ex skips solve's hidden info sync; magma_pin fixes the slow n >= 32 dispatch.
    with magma_pin(E):
        sols, _ = torch.linalg.solve_ex(I + nodes * E, weights * E)
    return sols.sum(dim=0)


def matrix_log(A: torch.Tensor, *, sqrt_iters: int = 40) -> torch.Tensor:
    """Principal matrix log of batched square matrices ``(..., n, n)``; the spectrum must
    avoid the closed negative real axis. ``sqrt_iters`` caps iterations per square root."""
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("matrix_log expects a square matrix tensor of shape (..., n, n)")
    if A.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"Unsupported dtype {A.dtype}; expected float32/float64/complex64/complex128"
        )

    theta = THETA_FP32 if A.dtype in (torch.float32, torch.complex64) else THETA_FP64
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device).expand_as(A)
    X = A.clone()
    s = torch.zeros(A.shape[:-2], dtype=torch.long, device=A.device)

    for _ in range(MAX_SQRT):
        norms = (X - I).abs().sum(dim=-2).amax(dim=-1)
        need = norms >= theta
        n_need = int(need.sum())
        if n_need == 0:
            break
        # Once under half the batch still needs roots, gather that subset instead of masking.
        if A.ndim > 2 and n_need <= need.numel() // 2:
            flat = X.reshape(-1, n, n)
            mask = need.reshape(-1)
            flat[mask] = matrix_sqrt(flat[mask], max_iters=sqrt_iters)
            X = flat.reshape(X.shape)
        else:
            X_sqrt = matrix_sqrt(X, max_iters=sqrt_iters)
            X = torch.where(need.unsqueeze(-1).unsqueeze(-1), X_sqrt, X)
        s = s + need.to(s.dtype)

    scale = torch.pow(2.0, s.to(A.real.dtype)).unsqueeze(-1).unsqueeze(-1)
    return scale * log1p_pade(X - I)
