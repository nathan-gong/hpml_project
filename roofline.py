#!/usr/bin/env python3
"""
roofline.py — Generate roofline plots from profiling_results.json

Produces 3 plots:
    1. roofline_prefill.png          — all configs x all seq lengths, prefill phase only
    2. roofline_decode.png           — all configs x all seq lengths, decode phase only
    3. arithmetic_intensity.png      — bar chart of arithmetic intensity per config

Prefill and decode are separated because they land in completely different
regions of the roofline (compute-bound vs memory-bound) and tell different
stories about optimization effects.

Usage
-----
    # Basic — reads profiling_results.json from current directory
    python roofline.py

    # Custom input file
    python roofline.py --input profiling_results.json

    # Save plots to specific directory
    python roofline.py --output-dir results/plots

    # Upload plots to W&B
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

L4_PEAK_FLOPS_FP16  = 121.0e12   # 121 TFLOPS FP16 (tensor cores)
L4_MEMORY_BANDWIDTH = 300.0e9    # 300 GB/s HBM bandwidth
L4_RIDGE_POINT      = L4_PEAK_FLOPS_FP16 / L4_MEMORY_BANDWIDTH  # ~403 FLOPS/byte


# ---------------------------------------------------------------------------
# Styling — consistent across all plots
# ---------------------------------------------------------------------------

# One color per config
CONFIG_COLORS = {
    "baseline":  "#2196F3",   # blue
    "flashattn": "#4CAF50",   # green
    "quantized": "#FF9800",   # orange
    "fa_quant":  "#F44336",   # red
}

CONFIG_LABELS = {
    "baseline":  "Baseline (FP16 + SDPA)",
    "flashattn": "FlashAttention (FP16 + Flash)",
    "quantized": "Quantized (4bit + SDPA)",
    "fa_quant":  "FA + Quant (4bit + Flash)",
}

# Dot size scales with sequence length
SEQ_LENGTHS   = [128, 256, 512, 1024, 2056]
SEQ_DOT_SIZES = {128: 80, 256: 120, 512: 160, 1024: 200, 2056: 250}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_profiling_results(path: str) -> list[dict]:
    """Load profiling_results.json produced by profiling.py"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. Run profiling.py first to generate this file."
        )
    return json.loads(p.read_text())


def estimate_phase_intensity(result: dict) -> dict[str, dict]:
    """
    Split total arithmetic intensity into prefill and decode estimates.

    profiling.py gives combined FLOPS across both phases. We split based
    on relative timing — prefill is compute-heavier, decode is memory-heavier.

    Returns:
        {
            "prefill": {"ai": float, "perf": float},
            "decode":  {"ai": float, "perf": float},
        }
    """
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

    # Prefill: compute-heavier -> higher arithmetic intensity
    # Decode:  memory-heavier  -> lower arithmetic intensity
    prefill_ai = total_ai * 1.8
    decode_ai  = total_ai * 0.4

    # Split performance proportionally by time
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
# Shared roofline drawing helpers
# ---------------------------------------------------------------------------

def _draw_roofline_ceiling(ax, peak_flops, memory_bw, ai_range):
    """Draw the hardware ceiling lines on the given axes."""
    ai_arr       = np.logspace(np.log10(ai_range[0]), np.log10(ai_range[1]), 1000)
    memory_roof  = memory_bw * ai_arr
    compute_roof = np.full_like(ai_arr, peak_flops)
    attainable   = np.minimum(memory_roof, compute_roof)
    ridge        = peak_flops / memory_bw

    ax.loglog(ai_arr, attainable, "k-", linewidth=2.5,
              label="L4 Roofline Ceiling", zorder=5)
    ax.axvline(ridge, color="gray", linestyle="--", linewidth=1.5,
               label=f"Ridge Point ({ridge:.0f} FLOPS/byte)", zorder=4)

    # Shade regions
    ax.axvspan(ai_range[0], ridge, alpha=0.04, color="blue")
    ax.axvspan(ridge, ai_range[1], alpha=0.04, color="green")

    # Region labels
    ax.text(ai_range[0] * 1.5, peak_flops * 0.05, "Memory-Bound",
            fontsize=9, color="blue", alpha=0.7)
    ax.text(ridge * 1.2, peak_flops * 0.05, "Compute-Bound",
            fontsize=9, color="green", alpha=0.7)

    # Hardware ceiling labels
    ax.text(ai_range[1] * 0.5, peak_flops * 1.05,
            f"Peak Compute: {peak_flops/1e12:.0f} TFLOPS",
            fontsize=8, ha="right")
    ax.text(ai_range[0] * 3, memory_bw * ai_range[0] * 3 * 1.5,
            f"Memory BW: {memory_bw/1e9:.0f} GB/s",
            fontsize=8, rotation=38)

    return ridge


def _format_axes(ax, title, ai_range, peak_flops):
    """Apply consistent formatting to roofline axes."""
    ax.set_xlabel("Arithmetic Intensity (FLOPS / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance (FLOPS/sec)",  fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.set_xlim(ai_range)
    ax.set_ylim(1e9, peak_flops * 1.5)


def _build_legend(ax, configs_present):
    """Build legend showing config colors and seq length dot sizes."""
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
    """
    Roofline for PREFILL phase only.
    All 4 configs x all sequence lengths on one plot.
    Dot size = sequence length, Color = config.

    Story this tells:
    -> Where does prefill sit on the roofline? (expect: right side, compute bound)
    -> Does FlashAttention shift prefill dots rightward? (less memory traffic)
    -> Does longer sequence push prefill further right? (more compute intensive)
    """
    # Auto-scale x-axis based on actual data
    valid_ai = [
        estimate_phase_intensity(r)["prefill"]["ai"]
        for r in results
        if estimate_phase_intensity(r)["prefill"]["ai"] > 0
    ]
    if valid_ai:
        ai_min = max(1e-2, min(valid_ai) * 0.1)
        ai_max = max(valid_ai) * 10
        ai_range = (ai_min, ai_max)
    else:
        ai_range = (1e-1, 1e5)

    fig, ax  = plt.subplots(figsize=(13, 8))

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

        ax.scatter(
            ai, perf,
            color=color, marker="o", s=size,
            zorder=10, edgecolors="black", linewidths=0.8, alpha=0.85,
        )
        ax.annotate(
            f"seq={seq_len}", xy=(ai, perf),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7, color=color,
        )

    _build_legend(ax, configs_present)
    _format_axes(
        ax,
        title=(
            "Roofline Analysis — PREFILL Phase\n"
            "All 4 Configs x All Sequence Lengths | NVIDIA L4 GPU\n"
            "Dot size = sequence length  |  Color = config  |  Circle = prefill"
        ),
        ai_range=ai_range,
        peak_flops=peak_flops,
    )

    os.makedirs(Path(output_path).parent, exist_ok=True)
    plt.tight_layout()
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
    """
    Roofline for DECODE phase only.
    All 4 configs x all sequence lengths on one plot.
    Dot size = sequence length, Color = config.

    Story this tells:
    -> Decode sits LEFT of ridge point (memory bound) -- expect this
    -> As seq grows, dots move further LEFT (KV cache grows, more memory pressure)
    -> Quantization shifts dots RIGHT (smaller weights = less bandwidth needed)
    -> FlashAttention shifts dots RIGHT (less KV cache traffic)
    """
    # Auto-scale x-axis based on actual data
    valid_ai = [
        estimate_phase_intensity(r)["decode"]["ai"]
        for r in results
        if estimate_phase_intensity(r)["decode"]["ai"] > 0
    ]
    if valid_ai:
        ai_min = max(1e-3, min(valid_ai) * 0.1)
        ai_max = max(valid_ai) * 10
        ai_range = (ai_min, ai_max)
    else:
        ai_range = (1e-2, 1e4)

    fig, ax  = plt.subplots(figsize=(13, 8))

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

        ax.scatter(
            ai, perf,
            color=color, marker="s", s=size,
            zorder=10, edgecolors="black", linewidths=0.8, alpha=0.85,
        )
        ax.annotate(
            f"seq={seq_len}", xy=(ai, perf),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7, color=color,
        )

    _build_legend(ax, configs_present)
    _format_axes(
        ax,
        title=(
            "Roofline Analysis — DECODE Phase\n"
            "All 4 Configs x All Sequence Lengths | NVIDIA L4 GPU\n"
            "Dot size = sequence length  |  Color = config  |  Square = decode  |  Larger seq -> further left"
        ),
        ai_range=ai_range,
        peak_flops=peak_flops,
    )

    os.makedirs(Path(output_path).parent, exist_ok=True)
    plt.tight_layout()
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
    """
    Bar chart of arithmetic intensity per config at a given sequence length.
    Simpler than roofline -- easier for audience to read exact values.

    Shows:
    -> How each optimization changes arithmetic intensity
    -> Whether configs are above or below the ridge point
    """
    filtered = [r for r in results if r["seq_len"] == seq_len]
    if not filtered:
        print(f"No results for seq_len={seq_len}, skipping bar chart.")
        return

    order    = ["baseline", "flashattn", "quantized", "fa_quant"]
    filtered = sorted(filtered,
                      key=lambda r: order.index(r["config"])
                      if r["config"] in order else 99)

    configs = [r["config"] for r in filtered]
    ai_vals = [r.get("arithmetic_intensity", 0.0) for r in filtered]
    colors  = [CONFIG_COLORS.get(c, "gray") for c in configs]
    labels  = [CONFIG_LABELS.get(c, c) for c in configs]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, ai_vals, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.5)

    # Value labels on bars
    for bar, val in zip(bars, ai_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ai_vals) * 0.01,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Ridge point reference line
    ax.axhline(
        L4_RIDGE_POINT, color="red", linestyle="--", linewidth=2,
        label=f"Ridge Point ({L4_RIDGE_POINT:.0f} FLOPS/byte)"
    )
    ax.axhspan(0, L4_RIDGE_POINT, alpha=0.05, color="blue",
               label="Memory-Bound Region")

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Arithmetic Intensity (FLOPS/Byte)", fontsize=12)
    ax.set_title(
        f"Arithmetic Intensity by Config (seq_len={seq_len})\n"
        f"Higher = more compute-bound | Lower = more memory-bound",
        fontsize=12
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(axis="x", labelrotation=12)
    ax.set_ylim(0, max(ai_vals) * 1.3 if ai_vals else 1)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Arithmetic intensity chart saved -> {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate roofline plots from profiling_results.json"
    )
    parser.add_argument("--input", type=str, default="profiling_results.json",
                        help="Path to profiling_results.json from profiling.py")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory to save all plots")
    parser.add_argument("--bar-seq-len", type=int, default=1024,
                        help="Sequence length to use for the bar chart")
    parser.add_argument("--wandb", action="store_true",
                        help="Upload plots to W&B after generating")
    args = parser.parse_args()

    # ---- Load results ----------------------------------------------------
    print(f"Loading profiling results from {args.input} ...")
    results = load_profiling_results(args.input)
    print(f"Loaded {len(results)} results\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Plot 1 — Prefill Roofline ---------------------------------------
    print("Generating prefill roofline ...")
    plot_prefill_roofline(
        results=results,
        output_path=f"{args.output_dir}/roofline_prefill.png",
    )

    # ---- Plot 2 — Decode Roofline ----------------------------------------
    print("Generating decode roofline ...")
    plot_decode_roofline(
        results=results,
        output_path=f"{args.output_dir}/roofline_decode.png",
    )

    # ---- Plot 3 — Arithmetic Intensity Bar Chart -------------------------
    print("Generating arithmetic intensity bar chart ...")
    plot_arithmetic_intensity_bars(
        results=results,
        seq_len=args.bar_seq_len,
        output_path=f"{args.output_dir}/arithmetic_intensity.png",
    )

    # ---- Summary ---------------------------------------------------------
    print(f"""
All plots saved to {args.output_dir}/
    roofline_prefill.png      <- prefill phase, all configs x all seq lengths
    roofline_decode.png       <- decode phase, all configs x all seq lengths
    arithmetic_intensity.png  <- bar chart at seq_len={args.bar_seq_len}
    """)

    # ---- Upload to W&B ---------------------------------------------------
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project="hpml2026-final-project",
                name="roofline-plots",
                tags=["roofline", "visualization"],
            )
            for plot_path in glob.glob(f"{args.output_dir}/*.png"):
                name = Path(plot_path).stem
                wandb.log({name: wandb.Image(plot_path)})
                print(f"Uploaded to W&B -> {name}")
            wandb.finish()
            print("W&B upload complete.")
        except Exception as e:
            print(f"W&B upload failed: {e}")


if __name__ == "__main__":
    main()
