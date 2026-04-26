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
    python profiling.py --model mistralai/Mistral-7B-v0.1

    # Quick test with smaller model
    python profiling.py --model gpt2 --seq-lengths 128 256

    # Skip W&B logging
    python profiling.py --model mistralai/Mistral-7B-v0.1 --no-wandb
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



# Config — 4 experiment configs from the proposal Table I

CONFIGS = [
    {"name": "baseline",   "precision": "fp16", "backend": AttentionBackend.MATH},
    {"name": "flashattn",  "precision": "fp16", "backend": AttentionBackend.FLASH},
    {"name": "quantized",  "precision": "4bit", "backend": AttentionBackend.MATH},
    {"name": "fa_quant",   "precision": "4bit", "backend": AttentionBackend.FLASH},
]


# Profiler helpers

def extract_flops_and_bytes(prof) -> dict:
    """
    Extract total estimated FLOPS and memory bytes moved from profiler output.
    These are the two numbers needed for the roofline plot:
        Arithmetic Intensity = flops / bytes_moved
        Performance          = flops / elapsed_seconds
    """
    total_flops = 0
    total_cuda_time_us = 0
    total_self_cpu_memory = 0

    for evt in prof.key_averages():
        # FLOPS — PyTorch estimates these with with_flops=True
        if evt.flops:
            total_flops += evt.flops

        # CUDA time in microseconds
        total_cuda_time_us += evt.cuda_time_total

        # Memory moved (self CPU memory as proxy for data movement)
        if evt.self_cpu_memory_usage:
            total_self_cpu_memory += abs(evt.self_cpu_memory_usage)

    elapsed_sec = total_cuda_time_us / 1e6  # convert us to seconds
    arithmetic_intensity = (
        total_flops / total_self_cpu_memory
        if total_self_cpu_memory > 0
        else 0.0
    )
    actual_performance = (
        total_flops / elapsed_sec
        if elapsed_sec > 0
        else 0.0
    )

    return {
        "total_flops":           total_flops,
        "total_cuda_time_us":    total_cuda_time_us,
        "total_self_cpu_memory_bytes": total_self_cpu_memory,
        "arithmetic_intensity":  round(arithmetic_intensity, 4),
        "actual_performance_flops_per_sec": actual_performance,
    }


def get_top_kernels(prof, top_n: int = 10) -> list[dict]:
    """
    Return the top N most expensive CUDA kernels.
    Useful for understanding which operations dominate time.
    """
    top = sorted(
        prof.key_averages(),
        key=lambda e: e.cuda_time_total,
        reverse=True
    )[:top_n]

    return [
        {
            "kernel":          evt.key,
            "cuda_time_ms":    round(evt.cuda_time_total / 1e3, 3),
            "cpu_time_ms":     round(evt.cpu_time_total / 1e3, 3),
            "calls":           evt.count,
            "flops":           evt.flops or 0,
        }
        for evt in top
    ]


# Core profiling function

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

    # Build prompt of exact length
    inputs = build_prompt(tokenizer, seq_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run with PyTorch Profiler 
    # with_flops=True  → estimates FLOPS per operation (needed for roofline)
    # profile_memory   → tracks memory allocations per kernel
    # record_shapes    → records tensor shapes (helps identify attention ops)
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

    # Extract profiler stats
    roofline_stats = extract_flops_and_bytes(prof)
    top_kernels    = get_top_kernels(prof, top_n=10)

    os.makedirs(trace_dir, exist_ok=True)
    trace_path = f"{trace_dir}/{config_name}_seq{seq_len}.json"
    prof.export_chrome_trace(trace_path)
    print(f"    Chrome trace saved → {trace_path}")

    # Print summary to console
    print(f"    TTFT          = {result.prefill.elapsed_ms:.2f} ms")
    print(f"    Decode        = {result.decode.tokens_per_sec:.2f} tok/s")
    print(f"    Est. FLOPS    = {roofline_stats['total_flops']:.3e}")
    print(f"    Arith. Int.   = {roofline_stats['arithmetic_intensity']:.4f} FLOPS/byte")
    print(f"    Top kernel    = {top_kernels[0]['kernel'] if top_kernels else 'N/A'}")

    # Package everything into one result dict
    return {
        # Identity
        "config":        config_name,
        "precision":     precision,
        "backend":       backend.value,
        "seq_len":       seq_len,
        # Performance metrics (from benchmark_single)
        "prefill_ms":    round(result.prefill.elapsed_ms, 3),
        "decode_tps":    round(result.decode.tokens_per_sec, 3),
        "prefill_mem_mb": round(result.prefill.peak_memory_mb, 2),
        "decode_mem_mb": round(result.decode.peak_memory_mb, 2),
        "kv_cache_mb":   round(result.prefill.kv_cache_mb, 4),
        # Roofline data (from profiler)
        **roofline_stats,
        # Kernel breakdown
        "top_kernels":   top_kernels,
        # Trace path
        "trace_path":    trace_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch Profiler — roofline data collection")
    parser.add_argument("--model", type=str, default=BenchmarkConfig.model_name,
                        help="HuggingFace model ID")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2056], 
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No CUDA device — profiling results will be limited.", file=sys.stderr)

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

    configs_to_run = [c for c in CONFIGS if c["name"] in args.configs]

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

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    out_path = Path(args.output)

    results_for_json = [
        {k: v for k, v in r.items() if k != "top_kernels"}
        for r in all_results
    ]
    out_path.write_text(json.dumps(results_for_json, indent=2))
    print(f"\nProfiling results saved → {out_path}")

    # Save kernel breakdowns separately
    kernels_path = Path("kernel_breakdown.json")
    kernels_data = [
        {"config": r["config"], "seq_len": r["seq_len"], "top_kernels": r["top_kernels"]}
        for r in all_results
    ]
    kernels_path.write_text(json.dumps(kernels_data, indent=2))
    print(f"Kernel breakdown saved → {kernels_path}")

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
