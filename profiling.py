#!/usr/bin/env python3
"""
profiling.py — Deep hardware profiling using PyTorch Profiler.

This script wraps PyTorch Profiler around the existing benchmark_single()
function to extract kernel-level FLOPS and memory traffic data needed
for the roofline plot.

It does NOT replace run_baseline.py or run_quantized.py — those collect
performance metrics. This collects hardware-level explanation of WHY
those metrics look the way they do.

Usage
-----
    # Profile all 4 configs
    python profiling.py --model meta-llama/Meta-Llama-3.1-8B-Instruct

    # Quick test with smaller model
    python profiling.py --model gpt2 --seq-lengths 128 --configs baseline --no-wandb

    # Skip W&B logging
    python profiling.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --no-wandb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from src.benchmark import benchmark_single
from src.config import BenchmarkConfig
from src.data import build_prompt
from src.model import AttentionBackend, load_model_and_tokenizer


# ---------------------------------------------------------------------------
# Config — 4 experiment configs from the proposal Table I
# ---------------------------------------------------------------------------

CONFIGS = [
    {"name": "baseline",  "precision": "fp16", "backend": AttentionBackend.MATH},
    {"name": "flashattn", "precision": "fp16", "backend": AttentionBackend.FLASH},
    {"name": "quantized", "precision": "4bit", "backend": AttentionBackend.MATH},
    {"name": "fa_quant",  "precision": "4bit", "backend": AttentionBackend.FLASH},
]


# ---------------------------------------------------------------------------
# Profiler helpers
# ---------------------------------------------------------------------------

def _get_cuda_time(evt) -> float:
    """
    Safely get CUDA time from a profiler event.
    Handles different PyTorch versions that use different attribute names.
    """
    for attr in ("self_cuda_time_total", "cuda_time_total", "cpu_time_total"):
        val = getattr(evt, attr, None)
        if val is not None:
            return float(val)
    return 0.0


def _get_flops(evt) -> int:
    """Safely get FLOPS from a profiler event."""
    for attr in ("flops", "total_flops"):
        val = getattr(evt, attr, None)
        if val:
            return int(val)
    return 0


def extract_flops_and_bytes(prof) -> dict:
    """
    Extract total estimated FLOPS and memory bytes moved from profiler output.
    These are the two numbers needed for the roofline plot:
        Arithmetic Intensity = flops / bytes_moved
        Performance          = flops / elapsed_seconds
    """
    total_flops          = 0
    total_cuda_time_us   = 0
    total_self_cpu_memory = 0

    for evt in prof.key_averages():
        total_flops        += _get_flops(evt)
        total_cuda_time_us += _get_cuda_time(evt)
        mem = getattr(evt, "self_cpu_memory_usage", 0) or 0
        total_self_cpu_memory += abs(mem)

    elapsed_sec = total_cuda_time_us / 1e6
    arithmetic_intensity = (
        total_flops / total_self_cpu_memory
        if total_self_cpu_memory > 0 else 0.0
    )
    actual_performance = (
        total_flops / elapsed_sec
        if elapsed_sec > 0 else 0.0
    )

    return {
        "total_flops":                     total_flops,
        "total_cuda_time_us":              total_cuda_time_us,
        "total_self_cpu_memory_bytes":     total_self_cpu_memory,
        "arithmetic_intensity":            round(arithmetic_intensity, 4),
        "actual_performance_flops_per_sec": actual_performance,
    }


def get_top_kernels(prof, top_n: int = 10) -> list[dict]:
    """
    Return the top N most expensive CUDA kernels.
    Useful for understanding which operations dominate time.
    """
    top = sorted(
        prof.key_averages(),
        key=lambda e: _get_cuda_time(e),
        reverse=True
    )[:top_n]

    return [
        {
            "kernel":       evt.key,
            "cuda_time_ms": round(_get_cuda_time(evt) / 1e3, 3),
            "cpu_time_ms":  round(getattr(evt, "cpu_time_total", 0) / 1e3, 3),
            "calls":        evt.count,
            "flops":        _get_flops(evt),
        }
        for evt in top
    ]


# ---------------------------------------------------------------------------
# Core profiling function
# ---------------------------------------------------------------------------

def profile_single_config(
    model,
    tokenizer,
    config_name: str,
    backend: AttentionBackend,
    precision: str,
    seq_len: int,
    decode_tokens: int,
    device: torch.device,
    trace_dir: str = "traces",
) -> dict:
    """
    Run PyTorch Profiler on one (config, seq_len) combination.
    Returns a dict of profiling stats ready for roofline plotting and W&B logging.
    """
    print(f"  Profiling {config_name} | seq_len={seq_len} ...")

    # Respect model's max sequence length (important for small models like gpt2)
    safe_len = min(seq_len, getattr(tokenizer, "model_max_length", seq_len))
    if safe_len != seq_len:
        print(f"    Note: seq_len capped to {safe_len} (model max length)")

    # Build prompt of exact length
    inputs = build_prompt(tokenizer, safe_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ---- Run with PyTorch Profiler ---------------------------------------
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        with record_function("full_benchmark"):
            result = benchmark_single(
                model=model,
                inputs=inputs,
                decode_tokens=decode_tokens,
                backend=backend,
                device=device,
                precision=precision,
            )

    # ---- Extract profiler stats ------------------------------------------
    roofline_stats = extract_flops_and_bytes(prof)
    top_kernels    = get_top_kernels(prof, top_n=10)

    # ---- Save Chrome trace -----------------------------------------------
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = f"{trace_dir}/{config_name}_seq{seq_len}.json"
    prof.export_chrome_trace(trace_path)
    print(f"    Chrome trace saved -> {trace_path}")

    # ---- Print summary ---------------------------------------------------
    print(f"    TTFT          = {result.prefill.elapsed_ms:.2f} ms")
    print(f"    Decode        = {result.decode.tokens_per_sec:.2f} tok/s")
    print(f"    Est. FLOPS    = {roofline_stats['total_flops']:.3e}")
    print(f"    Arith. Int.   = {roofline_stats['arithmetic_intensity']:.4f} FLOPS/byte")
    print(f"    Top kernel    = {top_kernels[0]['kernel'] if top_kernels else 'N/A'}")

    # ---- Package results -------------------------------------------------
    return {
        "config":         config_name,
        "precision":      precision,
        "backend":        backend.value,
        "seq_len":        seq_len,
        "prefill_ms":     round(result.prefill.elapsed_ms, 3),
        "decode_tps":     round(result.decode.tokens_per_sec, 3),
        "prefill_mem_mb": round(result.prefill.peak_memory_mb, 2),
        "decode_mem_mb":  round(result.decode.peak_memory_mb, 2),
        "kv_cache_mb":    round(result.prefill.kv_cache_mb, 4),
        **roofline_stats,
        "top_kernels":    top_kernels,
        "trace_path":     trace_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch Profiler — roofline data collection")
    parser.add_argument("--model", type=str, default=BenchmarkConfig.model_name,
                        help="HuggingFace model ID")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048],
                        help="Sequence lengths to profile")
    parser.add_argument("--decode-tokens", type=int,
                        default=BenchmarkConfig.decode_tokens)
    parser.add_argument("--configs", type=str, nargs="+",
                        default=["baseline", "flashattn", "quantized", "fa_quant"],
                        help="Which configs to profile")
    parser.add_argument("--trace-dir", type=str, default="traces",
                        help="Directory to save Chrome trace files")
    parser.add_argument("--output", type=str, default="profiling_results.json",
                        help="Path to save JSON results")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--append", action="store_true",
                        help="Append results to existing JSON instead of overwriting")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No CUDA device — profiling results will be limited.", file=sys.stderr)

    # ---- W&B init --------------------------------------------------------
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=BenchmarkConfig.wandb_project,
                name="profiling-roofline",
                tags=["profiling", "roofline"],
                config={
                    "model":         args.model,
                    "seq_lengths":   args.seq_lengths,
                    "decode_tokens": args.decode_tokens,
                    "configs":       args.configs,
                },
            )
        except Exception as e:
            print(f"W&B init failed ({e}); continuing without logging.", file=sys.stderr)

    # ---- Filter configs to run -------------------------------------------
    configs_to_run = [c for c in CONFIGS if c["name"] in args.configs]

    # ---- Group by precision to minimize model reloads --------------------
    all_results: list[dict] = []

    for precision in ["fp16", "4bit"]:
        precision_configs = [c for c in configs_to_run if c["precision"] == precision]
        if not precision_configs:
            continue

        print(f"\nLoading model: {args.model} ({precision}) ...")
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            device=str(device),
            precision=precision,
        )
        print("Model loaded.\n")

        for cfg in precision_configs:
            print(f"\n{'='*50}")
            print(f"Config: {cfg['name']} | precision={precision} | backend={cfg['backend'].value}")
            print(f"{'='*50}")

            for seq_len in args.seq_lengths:
                result = profile_single_config(
                    model=model,
                    tokenizer=tokenizer,
                    config_name=cfg["name"],
                    backend=cfg["backend"],
                    precision=precision,
                    seq_len=seq_len,
                    decode_tokens=args.decode_tokens,
                    device=device,
                    trace_dir=args.trace_dir,
                )
                all_results.append(result)

                # Log to W&B
                if wandb_run is not None:
                    import wandb
                    log_key = f"{cfg['name']}/seq{seq_len}"
                    wandb.log({
                        f"{log_key}/prefill_ms":           result["prefill_ms"],
                        f"{log_key}/decode_tps":           result["decode_tps"],
                        f"{log_key}/prefill_mem_mb":       result["prefill_mem_mb"],
                        f"{log_key}/kv_cache_mb":          result["kv_cache_mb"],
                        f"{log_key}/total_flops":          result["total_flops"],
                        f"{log_key}/arithmetic_intensity": result["arithmetic_intensity"],
                        f"{log_key}/actual_performance":   result["actual_performance_flops_per_sec"],
                    })

        # Free GPU memory before loading next precision
        del model
        torch.cuda.empty_cache()

    # ---- Save results to JSON (append or overwrite) ----------------------
    out_path = Path(args.output)
    results_for_json = [
        {k: v for k, v in r.items() if k != "top_kernels"}
        for r in all_results
    ]

    if args.append and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            existing_keys = {(r["config"], r["seq_len"]) for r in results_for_json}
            existing = [r for r in existing if (r["config"], r["seq_len"]) not in existing_keys]
            results_for_json = existing + results_for_json
            print(f"Appending: {len(existing)} existing + {len(all_results)} new results")
        except Exception as e:
            print(f"Could not read existing file ({e}), overwriting.")
    elif args.append:
        print(f"No existing file found at {out_path}, creating new.")

    out_path.write_text(json.dumps(results_for_json, indent=2))
    print(f"\nProfiling results saved -> {out_path} ({len(results_for_json)} total results)")

    kernels_path = Path("kernel_breakdown.json")
    kernels_data = [
        {"config": r["config"], "seq_len": r["seq_len"], "top_kernels": r["top_kernels"]}
        for r in all_results
    ]
    if args.append and kernels_path.exists():
        try:
            existing_k = json.loads(kernels_path.read_text())
            existing_keys = {(r["config"], r["seq_len"]) for r in kernels_data}
            existing_k = [r for r in existing_k if (r["config"], r["seq_len"]) not in existing_keys]
            kernels_data = existing_k + kernels_data
        except Exception:
            pass
    kernels_path.write_text(json.dumps(kernels_data, indent=2))
    print(f"Kernel breakdown saved -> {kernels_path}")

    if wandb_run is not None:
        import wandb
        artifact = wandb.Artifact("profiling-results", type="profiling")
        artifact.add_file(str(out_path))
        artifact.add_file(str(kernels_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print("\nDone! Next step: run roofline.py to generate the roofline plot.")


if __name__ == "__main__":
    main()
