# torch_matfunc

## A collection of PyTorch matrix functions.

[![CI](https://github.com/abhijeetgangan/torch_matfunc/actions/workflows/ci.yml/badge.svg)](https://github.com/abhijeetgangan/torch_matfunc/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/abhijeetgangan/torch_matfunc/branch/main/graph/badge.svg)](https://codecov.io/gh/abhijeetgangan/torch_matfunc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

### Implemented functions
 - `expm_frechet`: Matrix exponential and its Fréchet derivative.
 - `matrix_log_33`: Analytical matrix logarithm for a 3x3 matrix.

#### Example Usage

##### Matrix exponential and its Fréchet derivative

```python
import torch
from torch_matfunc.matrix.expm_frechet import expm_frechet
from scipy.linalg import expm_frechet as scipy_expm_frechet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

A = torch.tensor([[1, 2], [5, 6]], dtype=dtype, device=device)
E = torch.tensor([[3, 4], [7, 8]], dtype=dtype, device=device)

A_numpy = A.cpu().numpy()
E_numpy = E.cpu().numpy()

# Compute the matrix exponential and its Fréchet derivative
expm, expm_frechet = expm_frechet(A, E, method="SPS", compute_expm=True)
expm_scipy, expm_frechet_scipy = scipy_expm_frechet(A_numpy, E_numpy, method="SPS", compute_expm=True)

# Compare with scipy
assert torch.allclose(expm.cpu(), torch.tensor(expm_scipy))
assert torch.allclose(expm_frechet.cpu(), torch.tensor(expm_frechet_scipy))
```
#### Matrix exponential and its Fréchet derivative (autograd)

```python
import torch
from torch_matfunc.matrix.expm_frechet import expm
from scipy.linalg import expm_frechet as scipy_expm_frechet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

A = torch.tensor([[1, 2], [5, 6]], dtype=dtype, device=device, requires_grad=True)
E = torch.tensor([[3, 4], [7, 8]], dtype=dtype, device=device)

A_numpy = A.cpu().detach().numpy()
E_numpy = E.cpu().numpy()

# Compute the matrix exponential
expm = expm.apply(A)

# Compute the gradient of the matrix exponential
expm_frechet = torch.autograd.grad(expm, A, E)[0]
expm_scipy, expm_frechet_scipy = scipy_expm_frechet(A_numpy, E_numpy, method="SPS", compute_expm=True)

# Compare with scipy
assert torch.allclose(expm.cpu(), torch.tensor(expm_scipy))
assert torch.allclose(expm_frechet.cpu(), torch.tensor(expm_frechet_scipy))
```
