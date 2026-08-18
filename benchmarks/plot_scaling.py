# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.11.0",
#     "triton==3.6.0",
#     "matplotlib",
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
"""Batch-size scaling plots vs torch.linalg.matrix_exp: uv run benchmarks/plot_scaling.py"""

import pathlib
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from torch_matfunc.linalg import matrix_exp  # noqa: E402

NS = (4, 8, 16, 32, 64)
BS = (8, 32, 128, 512, 2048, 8192)
OUT_DIR = pathlib.Path(__file__).parent


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def sweep(dtype):
    rows = {}
    for n in NS:
        for B in BS:
            # The n=64 fp64 cell at B=8192 runs for minutes; skip it.
            if n == 64 and B == 8192 and dtype is torch.float64:
                continue
            torch.manual_seed(0)
            A = torch.randn(B, n, n, dtype=dtype, device="cuda") * 0.5
            t_ours = bench(lambda A=A: matrix_exp(A))
            t_torch = bench(lambda A=A: torch.linalg.matrix_exp(A))
            rows[(n, B)] = (t_ours, t_torch)
            print(
                f"{str(dtype).split('.')[-1]} n={n:3d} B={B:5d} "
                f"ours={t_ours * 1e3:8.3f}ms torch={t_torch * 1e3:8.3f}ms "
                f"speedup={t_torch / t_ours:5.2f}x",
                flush=True,
            )
            del A
    return rows


def plot(dtype, rows, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n in NS:
        bs = [B for B in BS if (n, B) in rows]
        ax.plot(bs, [rows[(n, B)][1] / rows[(n, B)][0] for B in bs], marker="o", label=f"n={n}")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size B")
    ax.set_ylabel("speedup vs torch.linalg.matrix_exp")
    ax.set_title(f"matrix_exp, {str(dtype).split('.')[-1]}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="matrix size")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def main():
    for dtype, name in ((torch.float64, "scaling_fp64.png"), (torch.float32, "scaling_fp32.png")):
        plot(dtype, sweep(dtype), OUT_DIR / name)


if __name__ == "__main__":
    main()
