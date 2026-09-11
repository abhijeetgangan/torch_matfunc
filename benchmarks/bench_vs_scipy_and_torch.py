# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.11.0",
#     "triton==3.6.0",
#     "scipy",
#     "torch_matfunc",
# ]
#
# [tool.uv.sources]
# torch_matfunc = { path = "..", editable = true }
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cu128"
#
# [tool.uv]
# index-strategy = "unsafe-best-match"
# ///
"""Benchmark public matrix_exp/sqrt/log vs torch.linalg.matrix_exp, eigh and serial SciPy:
uv run benchmarks/bench_vs_scipy_and_torch.py"""

import statistics
import time

import numpy as np
import torch
from scipy.linalg import logm, sqrtm

from torch_matfunc.linalg import matrix_exp, matrix_log, matrix_sqrt, ops

GRID = (
    (64, 4),
    (256, 4),
    (1024, 4),
    (64, 8),
    (256, 8),
    (64, 16),
    (64, 32),
)
# SciPy takes the whole batch in one call; logm still loops per matrix inside.
SCIPY_GRID = (
    (64, 4),
    (1024, 4),
    (1024, 8),
    (64, 32),
    (1024, 32),
)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, warmup=5, iters=20):
    """Median per-call seconds over ``iters`` runs after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def rel(x, ref):
    return (torch.linalg.matrix_norm(x - ref).max() / torch.linalg.matrix_norm(ref).max()).item()


def rel_diff(x: torch.Tensor, ref: np.ndarray) -> float:
    """Largest entrywise difference from a SciPy result, relative to its largest entry."""
    return float(np.abs(x.cpu().numpy() - ref).max() / np.abs(ref).max())


def scipy_batch(fn, A_np: np.ndarray) -> np.ndarray:
    return np.asarray(fn(A_np))


def inputs(kind: str, B: int, N: int, dtype, device):
    torch.manual_seed(0)
    A = torch.randn(B, N, N, dtype=dtype, device=device) * 0.5
    if kind == "randn":
        return A
    if kind == "spd":
        return A @ A.mT + N * torch.eye(N, dtype=dtype, device=device)
    # exp of a small random matrix: non-normal, spectrum off the negative axis.
    return torch.linalg.matrix_exp(0.3 * A)


def route(name: str, A: torch.Tensor) -> str:
    return "triton" if ops.routes_to_triton(name, A) else "torch"


def print_expm_table(title: str, rows: list[tuple]):
    print()
    print(f"### {title}")
    print()
    print("| batch | n | ours_expm_ms | torch_expm_ms | speedup | rel_err_vs_torch |")
    print("| --- | --- | --- | --- | --- | --- |")
    for B, N, t_ours, t_torch, err in rows:
        speedup = t_torch / t_ours if t_ours > 0 else float("inf")
        print(
            f"| {B} | {N} | {t_ours * 1e3:.3f} | {t_torch * 1e3:.3f} | {speedup:.2f}x | {err:.1e} |"
        )


def print_sqrt_log_table(title: str, rows: list[tuple]):
    print()
    print(f"### {title}")
    print()
    print(
        "| batch | n | ours_sqrt_ms | eigh_sqrt_ms | sqrt_vs_eigh | sqrt_resid | "
        "ours_log_ms | eigh_log_ms | log_vs_eigh | log_resid |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for B, N, t_sqrt, t_esqrt, e_sqrt, t_log, t_elog, e_log in rows:
        s_vs = t_esqrt / t_sqrt if t_sqrt > 0 else float("inf")
        l_vs = t_elog / t_log if t_log > 0 else float("inf")
        print(
            f"| {B} | {N} | {t_sqrt * 1e3:.3f} | {t_esqrt * 1e3:.3f} | {s_vs:.2f}x | "
            f"{e_sqrt:.1e} | {t_log * 1e3:.3f} | {t_elog * 1e3:.3f} | {l_vs:.2f}x | "
            f"{e_log:.1e} |"
        )


def print_scipy_table(rows: list[tuple]):
    print()
    print("### General sqrt/log vs SciPy sqrtm/logm on the CPU, one call per batch")
    print()
    print(
        "| input | batch | n | route sqrt/log | ours_sqrt_ms | scipy_sqrtm_ms | sqrt_speedup | "
        "sqrt_rel_diff | ours_log_ms | scipy_logm_ms | log_speedup | log_rel_diff |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for kind, B, N, routes, t_s, t_ss, e_s, t_l, t_sl, e_l in rows:
        print(
            f"| {kind} | {B} | {N} | {routes} | {t_s * 1e3:.3f} | {t_ss * 1e3:.2f} | "
            f"{t_ss / t_s:.1f}x | {e_s:.1e} | {t_l * 1e3:.3f} | {t_sl * 1e3:.1f} | "
            f"{t_sl / t_l:.1f}x | {e_l:.1e} |"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    triton_active = ops.TRITON_AVAILABLE and device.type == "cuda"
    print(f"device={device}, dtype={dtype}, triton_active={triton_active}")
    print(f"torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    expm_rows = []
    sqrt_log_rows = []
    herm_rows = []
    for B, N in GRID:
        A = inputs("randn", B, N, dtype, device)
        As = inputs("spd", B, N, dtype, device)

        t_ours = bench(lambda A=A: matrix_exp(A), device)
        t_torch = bench(lambda A=A: torch.linalg.matrix_exp(A), device)
        t_sqrt = bench(lambda As=As: matrix_sqrt(As), device)
        t_log = bench(lambda As=As: matrix_log(As), device)
        t_esqrt = bench(lambda As=As: matrix_sqrt(As, hermitian=True), device)
        t_elog = bench(lambda As=As: matrix_log(As, hermitian=True), device)

        # Every row also reports a relative residual so a fast-but-wrong kernel cannot hide.
        err_exp = rel(matrix_exp(A), torch.linalg.matrix_exp(A))
        S = matrix_sqrt(As)
        err_sqrt = rel(S @ S, As)
        err_log = rel(matrix_exp(matrix_log(As)), As)

        expm_rows.append((B, N, t_ours, t_torch, err_exp))
        sqrt_log_rows.append((B, N, t_sqrt, t_esqrt, err_sqrt, t_log, t_elog, err_log))

        t_hexp = bench(lambda As=As: matrix_exp(As, hermitian=True), device)
        t_htorch = bench(lambda As=As: torch.linalg.matrix_exp(As), device)
        herm_rows.append((B, N, t_hexp, t_htorch, t_esqrt, t_elog))

    print_expm_table("General matrix_exp (public dispatch vs torch.linalg.matrix_exp)", expm_rows)
    print_sqrt_log_table(
        "General sqrt/log on SPD (public dispatch, Triton or reference by routing table, vs eigh)",
        sqrt_log_rows,
    )
    print()
    print("### Hermitian (eigh, hermitian=True)")
    print()
    print("| batch | n | ours_expm_ms | torch_expm_ms | speedup | ours_sqrt_ms | ours_log_ms |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for B, N, t_hexp, t_htorch, t_hsqrt, t_hlog in herm_rows:
        speedup = t_htorch / t_hexp if t_hexp > 0 else float("inf")
        print(
            f"| {B} | {N} | {t_hexp * 1e3:.3f} | {t_htorch * 1e3:.3f} | "
            f"{speedup:.2f}x | {t_hsqrt * 1e3:.3f} | {t_hlog * 1e3:.3f} |"
        )

    scipy_rows = []
    for kind in ("spd", "nonnormal"):
        for B, N in SCIPY_GRID:
            A = inputs(kind, B, N, dtype, device)
            A_np = A.cpu().numpy()
            iters = 3 if B * N * N > 20000 else 5
            t_ss = bench(lambda A_np=A_np: scipy_batch(sqrtm, A_np), device, warmup=1, iters=iters)
            t_sl = bench(lambda A_np=A_np: scipy_batch(logm, A_np), device, warmup=1, iters=iters)
            t_s = bench(lambda A=A: matrix_sqrt(A), device)
            t_l = bench(lambda A=A: matrix_log(A), device)
            ref_s = scipy_batch(sqrtm, A_np)
            ref_l = scipy_batch(logm, A_np)
            e_s = rel_diff(matrix_sqrt(A), ref_s)
            e_l = rel_diff(matrix_log(A), ref_l)
            routes = f"{route('matrix_sqrt', A)}/{route('matrix_log', A)}"
            scipy_rows.append((kind, B, N, routes, t_s, t_ss, e_s, t_l, t_sl, e_l))
    print_scipy_table(scipy_rows)


if __name__ == "__main__":
    main()
