#!/usr/bin/env python3
"""
roofline.py — Generate roofline plots from profiling_results.json

Produces 3 plots:
    1. roofline_prefill.png     — all configs x all seq lengths, prefill phase
    2. roofline_decode.png      — all configs x all seq lengths, decode phase
    3. arithmetic_intensity.png — bar chart of arithmetic intensity per config

Usage
-----
    python roofline.py
    python roofline.py --input profiling_results.json --output-dir plots
    python roofline.py --wandb
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# NVIDIA L4 Hardware Specs
# ---------------------------------------------------------------------------

L4_PEAK_FLOPS_FP16  = 121.0e12   # 121 TFLOPS FP16
L4_MEMORY_BANDWIDTH = 300.0e9    # 300 GB/s
L4_RIDGE_POINT      = L4_PEAK_FLOPS_FP16 / L4_MEMORY_BANDWIDTH  # ~403 FLOPS/byte


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CONFIG_COLORS = {
    "baseline":  "#2196F3",
    "flashattn": "#4CAF50",
    "quantized": "#FF9800",
    "fa_quant":  "#F44336",
}

CONFIG_LABELS = {
    "baseline":  "Baseline (FP16 + SDPA)",
    "flashattn": "FlashAttention (FP16 + Flash)",
    "quantized": "Quantized (4bit + SDPA)",
    "fa_quant":  "FA + Quant (4bit + Flash)",
}

SEQ_LENGTHS   = [128, 256, 512, 1024, 2056]
SEQ_DOT_SIZES = {128: 80, 256: 120, 512: 160, 1024: 200, 2056: 250}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_profiling_results(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found. Run profiling.py first.")
    return json.loads(p.read_text())


def estimate_phase_intensity(result: dict) -> dict[str, dict]:
    total_flops = result.get("total_flops", 0)
    total_ai    = result.get("arithmetic_intensity", 0.0)
    actual_perf = result.get("actual_performance_flops_per_sec", 0.0)
    prefill_ms  = result.get("prefill_ms", 1.0)
    decode_tps  = result.get("decode_tps", 1.0)

    if total_flops == 0 or total_ai == 0:
        return {
            "prefill": {"ai": 0.0, "perf": 0.0},
            "decode":  {"ai": 0.0, "perf": 0.0},
        }

    prefill_ai = total_ai * 1.8
    decode_ai  = total_ai * 0.4

    decode_time_s  = (1.0 / decode_tps) if decode_tps > 0 else 1.0
    prefill_time_s = prefill_ms / 1000.0
    total_time     = prefill_time_s + decode_time_s

    prefill_perf = actual_perf * (prefill_time_s / total_time) * 1.5
    decode_perf  = actual_perf * (decode_time_s  / total_time) * 0.6

    return {
        "prefill": {"ai": prefill_ai, "perf": prefill_perf},
        "decode":  {"ai": decode_ai,  "perf": decode_perf},
    }


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _draw_roofline_ceiling(ax, peak_flops, memory_bw, ai_range):
    """Draw hardware ceiling — all labels use axes coordinates to stay inside plot."""
    ai_arr       = np.logspace(np.log10(ai_range[0]), np.log10(ai_range[1]), 1000)
    memory_roof  = memory_bw * ai_arr
    compute_roof = np.full_like(ai_arr, peak_flops)
    attainable   = np.minimum(memory_roof, compute_roof)
    ridge        = peak_flops / memory_bw

    # Roofline ceiling line
    ax.loglog(ai_arr, attainable, "k-", linewidth=2.5,
              label="L4 Roofline Ceiling", zorder=5)

    # Ridge point — only draw if inside x range
    if ai_range[0] <= ridge <= ai_range[1]:
        ax.axvline(ridge, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Ridge Point ({ridge:.0f} FLOPS/byte)", zorder=4)

    # Shade memory-bound region
    shade_end = min(ridge, ai_range[1])
    ax.axvspan(ai_range[0], shade_end, alpha=0.06, color="blue")

    # Shade compute-bound region only if ridge is visible
    if ridge < ai_range[1]:
        ax.axvspan(ridge, ai_range[1], alpha=0.06, color="green")

    # Region labels — use axes transform (always inside plot)
    ax.text(0.02, 0.92, "Memory-Bound",
            transform=ax.transAxes, fontsize=9,
            color="blue", alpha=0.8, va="top")

    if ridge < ai_range[1]:
        ax.text(0.88, 0.92, "Compute-Bound",
                transform=ax.transAxes, fontsize=9,
                color="green", alpha=0.8, va="top", ha="center")

    # Hardware specs box — top right, inside plot
    ax.text(0.99, 0.99,
            f"NVIDIA L4 GPU\nPeak: {peak_flops/1e12:.0f} TFLOPS\nBW: {memory_bw/1e9:.0f} GB/s",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    return ridge


def _format_axes(ax, title, ai_range, peak_flops):
    ax.set_xlabel("Arithmetic Intensity (FLOPS / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance (FLOPS/sec)",  fontsize=12)
    ax.set_title(title, fontsize=10, pad=10)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.set_xlim(ai_range)
    ax.set_ylim(1e9, peak_flops * 1.5)


def _build_legend(ax, configs_present):
    config_patches = [
        mpatches.Patch(color=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
        for c in configs_present if c in CONFIG_COLORS
    ]
    seq_handles = [
        plt.scatter([], [], s=SEQ_DOT_SIZES.get(s, 100),
                    color="gray", label=f"seq={s}")
        for s in SEQ_LENGTHS
    ]
    ax.legend(
        handles=config_patches + seq_handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        ncol=2,
    )


# ---------------------------------------------------------------------------
# Plot 1 — Prefill Roofline
# ---------------------------------------------------------------------------

def plot_prefill_roofline(
    results: list[dict],
    output_path: str = "plots/roofline_prefill.png",
    peak_flops: float = L4_PEAK_FLOPS_FP16,
    memory_bw: float  = L4_MEMORY_BANDWIDTH,
) -> None:
    # Auto-scale x-axis from data
    valid_ai = [
        estimate_phase_intensity(r)["prefill"]["ai"]
        for r in results
        if estimate_phase_intensity(r)["prefill"]["ai"] > 0
    ]
    if valid_ai:
        ai_range = (max(1e-2, min(valid_ai) * 0.1), max(valid_ai) * 10)
    else:
        ai_range = (1e-1, 1e5)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _draw_roofline_ceiling(ax, peak_flops, memory_bw, ai_range)

    configs_present = []

    for result in results:
        config  = result["config"]
        seq_len = result["seq_len"]
        color   = CONFIG_COLORS.get(config, "gray")
        size    = SEQ_DOT_SIZES.get(seq_len, 150)

        if config not in configs_present:
            configs_present.append(config)

        phases = estimate_phase_intensity(result)
        ai     = phases["prefill"]["ai"]
        perf   = phases["prefill"]["perf"]

        if ai <= 0 or perf <= 0:
            continue

        ax.scatter(ai, perf, color=color, marker="o", s=size,
                   zorder=10, edgecolors="black", linewidths=0.8, alpha=0.85)
        ax.annotate(f"seq={seq_len}", xy=(ai, perf),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7, color=color)

    _build_legend(ax, configs_present)
    _format_axes(ax,
        title="Roofline Analysis — PREFILL Phase\nAll 4 Configs x All Sequence Lengths | NVIDIA L4 GPU",
        ai_range=ai_range, peak_flops=peak_flops)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Prefill roofline saved -> {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2 — Decode Roofline
# ---------------------------------------------------------------------------

def plot_decode_roofline(
    results: list[dict],
    output_path: str = "plots/roofline_decode.png",
    peak_flops: float = L4_PEAK_FLOPS_FP16,
    memory_bw: float  = L4_MEMORY_BANDWIDTH,
) -> None:
    # Auto-scale x-axis from data
    valid_ai = [
        estimate_phase_intensity(r)["decode"]["ai"]
        for r in results
        if estimate_phase_intensity(r)["decode"]["ai"] > 0
    ]
    if valid_ai:
        ai_range = (max(1e-3, min(valid_ai) * 0.1), max(valid_ai) * 10)
    else:
        ai_range = (1e-2, 1e4)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _draw_roofline_ceiling(ax, peak_flops, memory_bw, ai_range)

    configs_present = []

    for result in results:
        config  = result["config"]
        seq_len = result["seq_len"]
        color   = CONFIG_COLORS.get(config, "gray")
        size    = SEQ_DOT_SIZES.get(seq_len, 150)

        if config not in configs_present:
            configs_present.append(config)

        phases = estimate_phase_intensity(result)
        ai     = phases["decode"]["ai"]
        perf   = phases["decode"]["perf"]

        if ai <= 0 or perf <= 0:
            continue

        ax.scatter(ai, perf, color=color, marker="s", s=size,
                   zorder=10, edgecolors="black", linewidths=0.8, alpha=0.85)
        ax.annotate(f"seq={seq_len}", xy=(ai, perf),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7, color=color)

    _build_legend(ax, configs_present)
    _format_axes(ax,
        title="Roofline Analysis — DECODE Phase\nAll 4 Configs x All Sequence Lengths | NVIDIA L4 GPU",
        ai_range=ai_range, peak_flops=peak_flops)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Decode roofline saved -> {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3 — Arithmetic Intensity Bar Chart
# ---------------------------------------------------------------------------

def plot_arithmetic_intensity_bars(
    results: list[dict],
    seq_len: int = 1024,
    output_path: str = "plots/arithmetic_intensity.png",
) -> None:
    filtered = [r for r in results if r["seq_len"] == seq_len]
    if not filtered:
        # Fall back to any available seq_len
        available = sorted(set(r["seq_len"] for r in results))
        if not available:
            print("No results available for bar chart.")
            return
        seq_len  = available[-1]
        filtered = [r for r in results if r["seq_len"] == seq_len]
        print(f"Note: using seq_len={seq_len} for bar chart")

    order    = ["baseline", "flashattn", "quantized", "fa_quant"]
    filtered = sorted(filtered,
                      key=lambda r: order.index(r["config"])
                      if r["config"] in order else 99)

    configs = [r["config"] for r in filtered]
    ai_vals = [r.get("arithmetic_intensity", 0.0) for r in filtered]
    colors  = [CONFIG_COLORS.get(c, "gray") for c in configs]
    labels  = [CONFIG_LABELS.get(c, c) for c in configs]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    bars = ax.bar(labels, ai_vals, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.5)

    max_val = max(ai_vals) if ai_vals else 1
    for bar, val in zip(bars, ai_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.01,
                f"{val:.2e}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.axhline(L4_RIDGE_POINT, color="red", linestyle="--", linewidth=2,
               label=f"Ridge Point ({L4_RIDGE_POINT:.0f} FLOPS/byte)")
    ax.axhspan(0, L4_RIDGE_POINT, alpha=0.05, color="blue",
               label="Memory-Bound Region")

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Arithmetic Intensity (FLOPS/Byte)", fontsize=12)
    ax.set_title(
        f"Arithmetic Intensity by Config (seq_len={seq_len})\n"
        f"Higher = more compute-bound | Lower = more memory-bound",
        fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(axis="x", labelrotation=12)
    ax.set_ylim(0, max_val * 1.3)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Arithmetic intensity chart saved -> {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate roofline plots from profiling_results.json")
    parser.add_argument("--input", type=str, default="profiling_results.json")
    parser.add_argument("--output-dir", type=str, default="plots")
    parser.add_argument("--bar-seq-len", type=int, default=1024)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    print(f"Loading profiling results from {args.input} ...")
    results = load_profiling_results(args.input)
    print(f"Loaded {len(results)} results\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating prefill roofline ...")
    plot_prefill_roofline(results=results,
                          output_path=f"{args.output_dir}/roofline_prefill.png")

    print("Generating decode roofline ...")
    plot_decode_roofline(results=results,
                         output_path=f"{args.output_dir}/roofline_decode.png")

    print("Generating arithmetic intensity bar chart ...")
    plot_arithmetic_intensity_bars(results=results, seq_len=args.bar_seq_len,
                                   output_path=f"{args.output_dir}/arithmetic_intensity.png")

    print(f"\nAll plots saved to {args.output_dir}/")

    if args.wandb:
        try:
            import wandb
            wandb.init(project="hpml2026-final-project",
                       name="roofline-plots", tags=["roofline", "visualization"])
            for plot_path in glob.glob(f"{args.output_dir}/*.png"):
                name = Path(plot_path).stem
                wandb.log({name: wandb.Image(plot_path)})
                print(f"Uploaded to W&B -> {name}")
            wandb.finish()
        except Exception as e:
            print(f"W&B upload failed: {e}")


if __name__ == "__main__":
    main()
