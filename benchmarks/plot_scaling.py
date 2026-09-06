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
"""Batch-size scaling plots: matrix_exp vs torch.linalg.matrix_exp, and matrix_sqrt or
matrix_log vs the pure-Torch reference on GPU and on CPU:
uv run benchmarks/plot_scaling.py [exp|sqrt|log|all]"""

import os
import pathlib
import statistics
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from torch_matfunc.linalg import matrix_exp, matrix_log, matrix_sqrt  # noqa: E402
from torch_matfunc.reference.logm import matrix_log as ref_logm  # noqa: E402
from torch_matfunc.reference.sqrtm import matrix_sqrt as ref_sqrtm  # noqa: E402

NS = (4, 8, 16, 32, 64)
NS_REF = (2, 4, 8, 16, 32)
BS = (8, 32, 128, 512, 2048, 8192)
OUT_DIR = pathlib.Path(__file__).parent
# Ops benchmarked against their own pure-Torch reference: public function and reference.
REF_OPS = {"sqrt": (matrix_sqrt, ref_sqrtm), "log": (matrix_log, ref_logm)}


def system_busy():
    """Return a reason string if another GPU process or high CPU load is present, else None."""
    try:
        query = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
        apps = subprocess.run(query, capture_output=True, text=True, timeout=10).stdout.strip()
        if apps:
            return f"GPU busy with pid {apps.splitlines()[0]}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    load = os.getloadavg()[0]
    if load > os.cpu_count() / 2:
        return f"CPU load average {load:.1f}"
    return None


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


def bench_pair(fa, fb, warmup=3, iters=10):
    """Time two functions interleaved per iteration so clock drift affects both equally."""
    for _ in range(warmup):
        fa()
        fb()
    torch.cuda.synchronize()
    ta, tb = [], []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fa()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fb()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ta.append(t1 - t0)
        tb.append(t2 - t1)
    return statistics.median(ta), statistics.median(tb)


def sweep(dtype):
    rows = {}
    for n in NS:
        for B in BS:
            # The n=64 fp64 cell at B=8192 runs for minutes; skip it.
            if n == 64 and B == 8192 and dtype is torch.float64:
                continue
            torch.manual_seed(0)
            A = torch.randn(B, n, n, dtype=dtype, device="cuda") * 0.5
            t_ours, t_torch = bench_pair(
                lambda A=A: matrix_exp(A), lambda A=A: torch.linalg.matrix_exp(A)
            )
            rows[(n, B)] = (t_ours, t_torch)
            print(
                f"exp {str(dtype).split('.')[-1]} n={n:3d} B={B:5d} "
                f"ours={t_ours * 1e3:8.3f}ms torch={t_torch * 1e3:8.3f}ms "
                f"speedup={t_torch / t_ours:5.2f}x",
                flush=True,
            )
            del A
    return rows


def sweep_ref(which, dtype):
    public, ref = REF_OPS[which]
    rows = {}
    for n in NS_REF:
        for B in BS:
            # The log reference's 13-node batched Pade solve overflows 8 GB at this cell.
            if which == "log" and n == 32 and B == 8192 and dtype is torch.float64:
                continue
            torch.manual_seed(0)
            A = torch.linalg.matrix_exp(0.4 * torch.randn(B, n, n, dtype=dtype, device="cuda"))
            A = A.contiguous()
            # Baseline is the reference path, so the ratio is the per-cell gain from routing.
            t_pub, t_ref = bench_pair(lambda A=A: public(A), lambda A=A: ref(A))
            Ac = A.cpu()
            t_cpu = bench(lambda Ac=Ac: ref(Ac), warmup=1, iters=3)
            rows[(n, B)] = (t_pub, t_ref, t_cpu)
            print(
                f"{which} {str(dtype).split('.')[-1]} n={n:3d} B={B:5d} "
                f"public={t_pub * 1e3:8.3f}ms ref={t_ref * 1e3:8.3f}ms "
                f"cpu={t_cpu * 1e3:9.3f}ms speedup={t_ref / t_pub:5.2f}x "
                f"gpu_vs_cpu={t_cpu / t_pub:6.2f}x",
                flush=True,
            )
            del A, Ac
    return rows


def plot(ns, rows, path, title, ylabel):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n in ns:
        bs = [B for B in BS if (n, B) in rows]
        ax.plot(bs, [rows[(n, B)][1] / rows[(n, B)][0] for B in bs], marker="o", label=f"n={n}")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size B")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="matrix size")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def main():
    args = [a for a in sys.argv[1:] if a != "--force"]
    which = args[0] if args else "all"
    if which not in ("exp", "sqrt", "log", "all"):
        sys.exit(f"usage: {sys.argv[0]} [exp|sqrt|log|all] [--force]")
    if "--force" not in sys.argv:
        busy = system_busy()
        if busy:
            sys.exit(f"not benchmarking: {busy}; pass --force to override")
    for dtype, tag in ((torch.float64, "fp64"), (torch.float32, "fp32")):
        name = str(dtype).split(".")[-1]
        if which in ("exp", "all"):
            plot(
                NS,
                sweep(dtype),
                OUT_DIR / f"scaling_{tag}.png",
                f"matrix_exp, {name}",
                "speedup vs torch.linalg.matrix_exp",
            )
        for op in REF_OPS:
            if which not in (op, "all"):
                continue
            rows = sweep_ref(op, dtype)
            plot(
                NS_REF,
                rows,
                OUT_DIR / f"scaling_{op}_{tag}.png",
                f"matrix_{op}, {name}: public op vs reference path",
                "speedup vs reference implementation",
            )
            vs_cpu = {k: (v[0], v[2]) for k, v in rows.items()}
            plot(
                NS_REF,
                vs_cpu,
                OUT_DIR / f"scaling_{op}_vs_cpu_{tag}.png",
                f"matrix_{op}, {name}: public op vs CPU reference",
                "speedup vs CPU reference",
            )


if __name__ == "__main__":
    main()
