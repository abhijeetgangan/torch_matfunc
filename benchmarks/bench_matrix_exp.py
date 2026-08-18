# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.11.0",
#     "triton==3.6.0",
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
"""Benchmark matrix_exp vs torch.linalg.matrix_exp: uv run benchmarks/bench_matrix_exp.py"""

import statistics
import time

import torch

from torch_matfunc.linalg import matrix_exp

GRID = (
    (64, 4),
    (256, 4),
    (1024, 4),
    (64, 8),
    (256, 8),
    (64, 16),
    (64, 32),
    (64, 64),
    (256, 64),
)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, warmup=5, iters=20):
    """Median per-call time in seconds."""
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


def print_table(title: str, rows: list[tuple]):
    print()
    print(f"### {title}")
    print()
    print("| batch | n | ours_ms | torch_ms | speedup | rel_err_vs_torch |")
    print("| --- | --- | --- | --- | --- | --- |")
    for B, N, t_ours, t_torch, err in rows:
        speedup = t_torch / t_ours if t_ours > 0 else float("inf")
        print(
            f"| {B} | {N} | {t_ours * 1e3:.3f} | {t_torch * 1e3:.3f} | {speedup:.2f}x | {err:.1e} |"
        )


def run_grid(device, dtype, hermitian):
    rows = []
    for B, N in GRID:
        torch.manual_seed(0)
        A = torch.randn(B, N, N, dtype=dtype, device=device) * 0.5
        if hermitian:
            A = A @ A.mH + N * torch.eye(N, dtype=dtype, device=device)
        t_ours = bench(lambda A=A: matrix_exp(A, hermitian=hermitian), device)
        t_torch = bench(lambda A=A: torch.linalg.matrix_exp(A), device)
        err = rel(matrix_exp(A, hermitian=hermitian), torch.linalg.matrix_exp(A))
        rows.append((B, N, t_ours, t_torch, err))
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print_table("General fp64, randn", run_grid(device, torch.float64, hermitian=False))
    print_table("General fp32, randn", run_grid(device, torch.float32, hermitian=False))
    print_table(
        "General complex128, randn; n=64 routes to the reference",
        run_grid(device, torch.complex128, hermitian=False),
    )
    print_table(
        "Hermitian fp64: SPD, hermitian=True eigh vs torch general",
        run_grid(device, torch.float64, hermitian=True),
    )


if __name__ == "__main__":
    main()
