"""Pure-Torch matrix exponential: Bader-Blanes-Casas scaling-and-squaring with a degree-18
Taylor polynomial (T18 below), the same algorithm as torch.linalg.matrix_exp."""

import torch

from torch_matfunc.reference.spectral import SUPPORTED_DTYPES

# T18 1-norm thresholds from PyTorch mexp_impl.
THETA18_FP32 = 3.010066362817634
THETA18_FP64 = 1.090863719290036

# T18 coefficients from PyTorch compute_T18; the Triton kernels copy these.
T18_B1_A = -1.00365581030144618291e-01
T18_B1_A2 = -8.02924648241156932449e-03
T18_B1_A3 = -8.92138498045729985177e-04
T18_B2_A = 3.97849749499645077844e-01
T18_B2_A2 = 1.36783778460411720168e00
T18_B2_A3 = 4.98289622525382669416e-01
T18_B2_A6 = -6.37898194594723280150e-04
T18_B3_I = -1.09676396052962061844e01
T18_B3_A = 1.68015813878906206114e00
T18_B3_A2 = 5.71779846478865511061e-02
T18_B3_A3 = -6.98210122488052056106e-03
T18_B3_A6 = 3.34975017086070470649e-05
T18_B4_I = -9.04316832390810593223e-02
T18_B4_A = -6.76404519071381882256e-02
T18_B4_A2 = 6.75961301770459654925e-02
T18_B4_A3 = 2.95552570429315521194e-02
T18_B4_A6 = -1.39180257516060693404e-05
T18_B5_A2 = -9.23364619367118555360e-02
T18_B5_A3 = -1.69364939002081722752e-02
T18_B5_A6 = -1.40086798182036094347e-05


def taylor18(A: torch.Tensor) -> torch.Tensor:
    """Bader-Blanes-Casas T18 in 5 matmuls, matching PyTorch ``compute_T18``."""
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A)
    A2 = A @ A
    A3 = A @ A2
    A6 = A3 @ A3
    B1 = T18_B1_A * A + T18_B1_A2 * A2 + T18_B1_A3 * A3
    B2 = T18_B2_A * A + T18_B2_A2 * A2 + T18_B2_A3 * A3 + T18_B2_A6 * A6
    B3 = T18_B3_I * eye + T18_B3_A * A + T18_B3_A2 * A2 + T18_B3_A3 * A3 + T18_B3_A6 * A6
    B4 = T18_B4_I * eye + T18_B4_A * A + T18_B4_A2 * A2 + T18_B4_A3 * A3 + T18_B4_A6 * A6
    B5 = T18_B5_A2 * A2 + T18_B5_A3 * A3 + T18_B5_A6 * A6
    A9 = B1 @ B5 + B4
    return B2 + (B3 + A9) @ A9


def matrix_exp(A: torch.Tensor) -> torch.Tensor:
    """Matrix exponential of a batch of square matrices ``(..., n, n)``."""
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("matrix_exp expects a square matrix tensor of shape (..., n, n)")
    if A.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"Unsupported dtype {A.dtype}; expected float32/float64/complex64/complex128"
        )

    norms = A.abs().sum(dim=-2).amax(dim=-1)
    low_prec = A.dtype in (torch.float32, torch.complex64)
    theta18 = THETA18_FP32 if low_prec else THETA18_FP64
    # float->long of NaN/inf is undefined; keep the bound finite.
    ratio = torch.nan_to_num(norms.double() / theta18, nan=1.0, posinf=2.0**1023).clamp(min=1.0)
    # 0.5^s must stay representable.
    s_cap = 128 if low_prec else 1200
    s = torch.ceil(torch.log2(ratio)).to(torch.long).clamp(min=0, max=s_cap)
    s_max = int(s.max().item()) if s.numel() else 0

    scale = torch.pow(0.5, s.to(A.real.dtype)).unsqueeze(-1).unsqueeze(-1)
    X = taylor18(A * scale)

    # The mask stops squaring each matrix after its own s steps.
    remaining = s.clone()
    for _ in range(s_max):
        need = (remaining > 0).unsqueeze(-1).unsqueeze(-1)
        X = torch.where(need, X @ X, X)
        remaining = (remaining - 1).clamp(min=0)

    return X
