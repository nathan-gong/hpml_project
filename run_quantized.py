#!/usr/bin/env python3
"""Run quantized / precision-controlled inference benchmarks.

This mirrors the baseline runner but exposes a ``--precision`` flag so the
same benchmark flow can be used for FP16, INT8, and 4-bit weights.

Usage
-----
    # FP16 (same behaviour as baseline, but through the quantized-capable path)
    python run_quantized.py --model meta-llama/Llama-2-7b-hf --precision fp16

    # INT8 sweep
    python run_quantized.py --model meta-llama/Llama-2-7b-hf --precision int8

    # 4-bit sweep
    python run_quantized.py --model meta-llama/Llama-2-7b-hf --precision 4bit

    # Quick sanity check
    python run_quantized.py --model gpt2 --precision 4bit --warmup 0 --repeats 1 --no-wandb
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

from src.benchmark import BenchmarkResult, run_benchmark
from src.config import BenchmarkConfig
from src.data import build_prompt
from src.model import AttentionBackend, load_model_and_tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise(results: list[BenchmarkResult]) -> dict:
    """Compute mean ± stdev for key metrics over repeated runs."""
    prefill_ms = [r.prefill.elapsed_ms for r in results]
    decode_tps = [r.decode.tokens_per_sec for r in results]
    prefill_mem = [r.prefill.peak_memory_mb for r in results]
    prefill_reserved = [r.prefill.peak_reserved_memory_mb for r in results]
    decode_mem = [r.decode.peak_memory_mb for r in results]
    decode_reserved = [r.decode.peak_reserved_memory_mb for r in results]
    prefill_delta = [r.prefill.memory_delta_mb for r in results]
    decode_delta = [r.decode.memory_delta_mb for r in results]
    prefill_kv = [r.prefill.kv_cache_mb for r in results]
    decode_kv = [r.decode.kv_cache_mb for r in results]
    parameter_mb = [r.parameter_bytes / (1024 ** 2) for r in results]

    def _stat(vals):
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return m, s

    pm, ps = _stat(prefill_ms)
    dm, ds = _stat(decode_tps)
    pmem_m, pmem_s = _stat(prefill_mem)
    prm_m, prm_s = _stat(prefill_reserved)
    dmem_m, dmem_s = _stat(decode_mem)
    drm_m, drm_s = _stat(decode_reserved)
    pdel_m, pdel_s = _stat(prefill_delta)
    ddel_m, ddel_s = _stat(decode_delta)
    pkv_m, pkv_s = _stat(prefill_kv)
    dkv_m, dkv_s = _stat(decode_kv)
    param_m, param_s = _stat(parameter_mb)

    return {
        "prefill_ttft_ms_mean": round(pm, 2),
        "prefill_ttft_ms_std": round(ps, 2),
        "decode_tok_per_sec_mean": round(dm, 2),
        "decode_tok_per_sec_std": round(ds, 2),
        "prefill_peak_mem_mb_mean": round(pmem_m, 1),
        "prefill_peak_mem_mb_std": round(pmem_s, 1),
        "prefill_peak_reserved_mb_mean": round(prm_m, 1),
        "prefill_peak_reserved_mb_std": round(prm_s, 1),
        "decode_peak_mem_mb_mean": round(dmem_m, 1),
        "decode_peak_mem_mb_std": round(dmem_s, 1),
        "decode_peak_reserved_mb_mean": round(drm_m, 1),
        "decode_peak_reserved_mb_std": round(drm_s, 1),
        "prefill_mem_delta_mb_mean": round(pdel_m, 1),
        "prefill_mem_delta_mb_std": round(pdel_s, 1),
        "decode_mem_delta_mb_mean": round(ddel_m, 1),
        "decode_mem_delta_mb_std": round(ddel_s, 1),
        "prefill_kv_cache_mb_mean": round(pkv_m, 2),
        "prefill_kv_cache_mb_std": round(pkv_s, 2),
        "decode_kv_cache_mb_mean": round(dkv_m, 2),
        "decode_kv_cache_mb_std": round(dkv_s, 2),
        "parameter_mb_mean": round(param_m, 2),
        "parameter_mb_std": round(param_s, 2),
    }


def _print_table(all_summaries: list[dict]) -> None:
    header = (
        f"{'SeqLen':>8} | {'TTFT (ms)':>16} | {'Decode tok/s':>18} | "
        f"{'Prefill Mem MB':>18} | {'KV Cache MB':>16}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(
            f"{s['seq_len']:>8} | "
            f"{s['prefill_ttft_ms_mean']:>8.2f} ± {s['prefill_ttft_ms_std']:<6.2f} | "
            f"{s['decode_tok_per_sec_mean']:>8.2f} ± {s['decode_tok_per_sec_std']:<6.2f} | "
            f"{s['prefill_peak_mem_mb_mean']:>8.1f} ± {s['prefill_peak_mem_mb_std']:<6.1f} | "
            f"{s['prefill_kv_cache_mb_mean']:>8.2f} ± {s['prefill_kv_cache_mb_std']:<6.2f}"
        )
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Precision-controlled inference benchmark")
    parser.add_argument("--model", type=str, default=BenchmarkConfig.model_name,
                        help="HuggingFace model ID")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "int8", "4bit"],
                        help="Weight precision / quantization mode")
    parser.add_argument("--backend", type=str, default=AttentionBackend.MATH.value,
                        choices=[b.value for b in AttentionBackend],
                        help="Attention backend to force")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=BenchmarkConfig().sequence_lengths,
                        help="Prompt lengths to sweep")
    parser.add_argument("--decode-tokens", type=int,
                        default=BenchmarkConfig.decode_tokens,
                        help="Tokens to generate in decode phase")
    parser.add_argument("--warmup", type=int, default=BenchmarkConfig.warmup)
    parser.add_argument("--repeats", type=int, default=BenchmarkConfig.repeats)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--output", type=str, default="quantized_results.json",
                        help="Path to save JSON results")
    args = parser.parse_args()

    backend = AttentionBackend(args.backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No CUDA device detected — quantized timing/memory results will be limited.", file=sys.stderr)

    # ---- W&B init --------------------------------------------------------
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=BenchmarkConfig.wandb_project,
                config={
                    "model": args.model,
                    "backend": backend.value,
                    "precision": args.precision,
                    "decode_tokens": args.decode_tokens,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "seq_lengths": args.seq_lengths,
                },
                tags=["quantized", backend.value, args.precision],
            )
        except Exception as e:
            print(f"W&B init failed ({e}); continuing without logging.", file=sys.stderr)

    # ---- Load model ------------------------------------------------------
    print(f"Loading model: {args.model} ({args.precision}, backend={backend.value}) ...")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=str(device),
        precision=args.precision,
    )
    print("Model loaded.\n")

    # ---- Sweep over sequence lengths -------------------------------------
    all_summaries: list[dict] = []

    for seq_len in args.seq_lengths:
        print(f">>> Benchmarking seq_len={seq_len} ...")
        inputs = build_prompt(tokenizer, seq_len)

        results = run_benchmark(
            model=model,
            inputs=inputs,
            decode_tokens=args.decode_tokens,
            backend=backend,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            precision=args.precision,
        )

        summary = _summarise(results)
        summary["seq_len"] = seq_len
        summary["model"] = args.model
        summary["backend"] = backend.value
        summary["precision"] = args.precision
        all_summaries.append(summary)

        print(f"    TTFT = {summary['prefill_ttft_ms_mean']:.2f} ± {summary['prefill_ttft_ms_std']:.2f} ms")
        print(f"    Decode = {summary['decode_tok_per_sec_mean']:.2f} ± {summary['decode_tok_per_sec_std']:.2f} tok/s")
        print(f"    Param footprint ≈ {summary['parameter_mb_mean']:.2f} MB")
        print(f"    Prefill peak mem = {summary['prefill_peak_mem_mb_mean']:.1f} MB")
        print(f"    Decode  peak mem = {summary['decode_peak_mem_mb_mean']:.1f} MB")
        print(f"    Prefill KV-cache ≈ {summary['prefill_kv_cache_mb_mean']:.2f} MB")
        print(f"    Decode  KV-cache ≈ {summary['decode_kv_cache_mb_mean']:.2f} MB")

        if wandb_run is not None:
            import wandb
            wandb.log({f"seq{seq_len}/{k}": v for k, v in summary.items()
                       if isinstance(v, (int, float))})

    # ---- Print summary table ---------------------------------------------
    _print_table(all_summaries)

    # ---- Persist results -------------------------------------------------
    out_path = Path(args.output)
    out_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"Results saved to {out_path}")

    if wandb_run is not None:
        import wandb
        artifact = wandb.Artifact("quantized-results", type="benchmark")
        artifact.add_file(str(out_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()
