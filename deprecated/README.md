# Deprecated implementations

Frozen copies of the pre-Triton `torch_matfunc` matrix code:

- SciPy-backed `expm_frechet` / Fréchet derivative of the matrix exponential
- Analytical 3×3 matrix logarithm (`_logm_33`)
- SciPy round-trip `logm`

These are **not** installed as part of the package and are kept only for historical reference.
The active library lives under `torch_matfunc/` and targets a Triton / pure-Torch path
aligned with [pytorch/pytorch#9983](https://github.com/pytorch/pytorch/issues/9983).
