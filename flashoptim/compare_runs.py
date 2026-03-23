"""
Compare training runs side-by-side.

Usage:
  python compare_runs.py --runs adamw flash_adamw
  python compare_runs.py --runs adamw flash_adamw --plot   # requires matplotlib
"""

import argparse
import json
import os
import sys


def load_run(log_dir: str, name: str) -> dict:
    path = os.path.join(log_dir, f"{name}.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found"); sys.exit(1)
    with open(path) as f:
        return json.load(f)


def print_summary(runs: dict):
    names = list(runs.keys())
    header = f"{'Metric':<28s}" + "".join(f"{n:>18s}" for n in names)
    sep = "-" * len(header)

    print(f"\n{sep}\n{header}\n{sep}")

    rows = [
        ("Final loss",        lambda r: f"{r['final_loss']:.4f}"),
        ("Final perplexity",  lambda r: f"{r['final_perplexity']:.2f}"),
        ("Avg step time (s)", lambda r: f"{r['avg_step_time']:.4f}"),
        ("Avg tokens/sec",    lambda r: f"{r['avg_tokens_per_sec']:,.0f}"),
        ("Peak GPU mem (MB)", lambda r: f"{r['peak_gpu_mem_mb']:,.0f}"),
        ("Total steps",       lambda r: f"{r['total_steps']}"),
    ]

    for label, fmt in rows:
        line = f"{label:<28s}"
        for name in names:
            line += f"{fmt(runs[name]):>18s}"
        print(line)

    # Delta row if exactly 2 runs
    if len(names) == 2:
        a, b = runs[names[0]], runs[names[1]]
        print(sep)
        print(f"{'Delta (B vs A)':<28s}")

        loss_diff = b["final_loss"] - a["final_loss"]
        ppl_diff = b["final_perplexity"] - a["final_perplexity"]
        time_diff_pct = (b["avg_step_time"] - a["avg_step_time"]) / a["avg_step_time"] * 100
        mem_diff_pct = ((b["peak_gpu_mem_mb"] - a["peak_gpu_mem_mb"])
                        / max(a["peak_gpu_mem_mb"], 1) * 100)

        print(f"  {'Loss diff':<26s}{loss_diff:>+18.4f}")
        print(f"  {'Perplexity diff':<26s}{ppl_diff:>+18.2f}")
        print(f"  {'Step time':<26s}{time_diff_pct:>+17.1f}%")
        print(f"  {'GPU memory':<26s}{mem_diff_pct:>+17.1f}%")

    print(sep + "\n")


def plot_comparison(runs: dict, out_path: str = "comparison.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot"); return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Run Comparison", fontsize=14, fontweight="bold")

    for name, data in runs.items():
        steps = [s["step"] for s in data["steps"]]
        losses = [s["loss"] for s in data["steps"]]
        ppls = [s["perplexity"] for s in data["steps"]]
        mems = [s["gpu_mem_allocated_mb"] for s in data["steps"]]
        tps = [s["tokens_per_sec"] for s in data["steps"]]

        axes[0, 0].plot(steps, losses, label=name, alpha=0.8)
        axes[0, 1].plot(steps, ppls, label=name, alpha=0.8)
        axes[1, 0].plot(steps, mems, label=name, alpha=0.8)
        axes[1, 1].plot(steps, tps, label=name, alpha=0.8)

    axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_title("Perplexity"); axes[0, 1].set_xlabel("Step"); axes[0, 1].set_ylabel("PPL")
    axes[1, 0].set_title("GPU Memory (MB)"); axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("MB")
    axes[1, 1].set_title("Throughput"); axes[1, 1].set_xlabel("Step"); axes[1, 1].set_ylabel("tok/s")

    for ax in axes.flat:
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run names to compare")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plot")
    parser.add_argument("--plot-out", default="comparison.png")
    args = parser.parse_args()

    runs = {name: load_run(args.log_dir, name) for name in args.runs}
    print_summary(runs)

    if args.plot:
        plot_comparison(runs, args.plot_out)