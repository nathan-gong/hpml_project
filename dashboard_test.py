#!/usr/bin/env python3
"""
dashboard_test.py — W&B dashboard preview with dummy data.

This script is NOT a real experiment — it logs realistic dummy data to W&B
so you can build and arrange your dashboard panels BEFORE real GCP results
are available.

Once real data comes in from teammates, the same dashboard panels will
automatically display real numbers instead of dummy ones.

Checklist of what your W&B dashboard should contain:
    1. TTFT comparison across 4 configs
    2. Decode throughput comparison across 4 configs
    3. Performance scaling vs sequence length
    4. Peak memory usage comparison
    5. KV-cache growth vs sequence length
    6. Roofline plot images (uploaded after roofline.py runs)

Usage
-----
    python dashboard_test.py

    # Then go to wandb.ai and arrange your dashboard panels
"""

from __future__ import annotations

import time
import numpy as np

# ---------------------------------------------------------------------------
# Realistic dummy data ranges based on Mistral-7B expectations on L4 GPU
# These approximate real values so your dashboard looks meaningful
# ---------------------------------------------------------------------------

# Expected TTFT ranges (ms) per config — prefill gets faster with optimization
TTFT_RANGES = {
    "baseline":  (350, 420),   # slowest — no optimization
    "flashattn": (260, 310),   # faster — less memory traffic
    "quantized": (320, 380),   # slightly faster — smaller weights
    "fa_quant":  (230, 280),   # fastest — both optimizations
}

# Expected decode throughput (tokens/sec) — higher is better
DECODE_TPS_RANGES = {
    "baseline":  (25, 35),     # slowest — memory bound, large weights
    "flashattn": (30, 42),     # moderate gain
    "quantized": (45, 60),     # big gain — 4x smaller weights
    "fa_quant":  (55, 75),     # biggest gain — both optimizations
}

# Expected peak memory (MB) — lower is better
PEAK_MEM_RANGES = {
    "baseline":  (14000, 15000),  # FP16 — large footprint
    "flashattn": (13500, 14500),  # slightly less (no N x N matrix)
    "quantized": (8000,  9500),   # much less — 4bit weights
    "fa_quant":  (7500,  9000),   # smallest — both reduce memory
}

# KV-cache grows linearly with sequence length
# Base MB per 128 tokens — scales up with seq length
KV_CACHE_BASE_MB = {
    "baseline":  2.0,
    "flashattn": 2.0,    # same KV-cache size — FA doesn't change cache size
    "quantized": 2.0,    # same KV-cache size — quantization is weights only
    "fa_quant":  2.0,
}

CONFIGS     = ["baseline", "flashattn", "quantized", "fa_quant"]
SEQ_LENGTHS = [128, 256, 512, 1024, 2056]

CONFIG_LABELS = {
    "baseline":  "Baseline (FP16 + SDPA)",
    "flashattn": "FlashAttention (FP16 + Flash)",
    "quantized": "Quantized (4bit + SDPA)",
    "fa_quant":  "FA + Quant (4bit + Flash)",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. Run: pip install wandb")
        return

    print("Logging dummy data to W&B for dashboard preview...")
    print("This will create one W&B run per config.\n")

    # ---- One W&B run per config ------------------------------------------
    # This mirrors how run_baseline.py and run_quantized.py log data
    for config in CONFIGS:
        print(f"  Logging config: {config} ...")

        run = wandb.init(
            project="hpml2026-final-project",
            name=f"DUMMY-{config}",
            tags=["dummy", "dashboard-test", config],
            config={
                "model":     "mistralai/Mistral-7B-v0.1",
                "config":    config,
                "precision": "4bit" if "quant" in config else "fp16",
                "backend":   "flash" if "flash" in config else "math",
                "note":      "DUMMY DATA — for dashboard layout only",
            },
        )

        # Log metrics for each sequence length
        for seq_len in SEQ_LENGTHS:

            # Scale metrics with sequence length — longer = slower + more memory
            seq_scale = seq_len / 128   # relative to shortest seq

            # TTFT grows with sequence length (more tokens to process)
            ttft_base = np.random.uniform(*TTFT_RANGES[config])
            ttft      = ttft_base * (seq_scale ** 0.9)

            # Decode throughput drops slightly with longer sequences (bigger KV-cache)
            tps_base = np.random.uniform(*DECODE_TPS_RANGES[config])
            tps      = tps_base / (seq_scale ** 0.15)

            # Peak memory grows with sequence length
            mem_base = np.random.uniform(*PEAK_MEM_RANGES[config])
            mem      = mem_base + (seq_len * 12)   # ~12 MB per token

            # KV-cache grows linearly with sequence length
            kv_cache = KV_CACHE_BASE_MB[config] * seq_scale

            wandb.log({
                # Identity
                "seq_len":               seq_len,
                "config":                config,

                # Core performance metrics (from run_baseline / run_quantized)
                "prefill/ttft_ms":       round(ttft, 2),
                "decode/tokens_per_sec": round(tps, 2),

                # Memory metrics
                "prefill/peak_mem_mb":   round(mem, 1),
                "decode/peak_mem_mb":    round(mem + 200, 1),
                "kv_cache/size_mb":      round(kv_cache, 4),

                # Profiling metrics (from profiling.py)
                "profiling/arithmetic_intensity": round(
                    np.random.uniform(0.1, 0.8) * (1.3 if "flash" in config else 1.0),
                    4
                ),
                "profiling/total_flops": round(
                    np.random.uniform(40, 80) * 1e12, 2
                ),
            })

            time.sleep(0.1)   # small delay so W&B receives logs cleanly

        wandb.finish()
        print(f"    Done — {config} logged for seq_lens {SEQ_LENGTHS}\n")

    # ---- Print dashboard building instructions ---------------------------
    print("=" * 60)
    print("DUMMY DATA LOGGED SUCCESSFULLY")
    print("=" * 60)
    print("""
Next steps — build your W&B dashboard:

1. Go to wandb.ai -> your project "hpml2026-final-project"
2. Click "Dashboard" -> "New Dashboard"
3. Add these panels:

   PANEL 1 — TTFT Comparison (Bar Chart)
   Metric: prefill/ttft_ms
   Group by: config
   Title: "Time To First Token by Config"

   PANEL 2 — Decode Throughput (Bar Chart)
   Metric: decode/tokens_per_sec
   Group by: config
   Title: "Decode Throughput (tokens/sec)"

   PANEL 3 — Performance vs Sequence Length (Line Chart)
   X-axis: seq_len
   Y-axis: prefill/ttft_ms
   Group by: config
   Title: "TTFT Scaling with Sequence Length"

   PANEL 4 — Peak Memory Usage (Bar Chart)
   Metric: prefill/peak_mem_mb
   Group by: config
   Title: "Peak GPU Memory by Config"

   PANEL 5 — KV-Cache Growth (Line Chart)
   X-axis: seq_len
   Y-axis: kv_cache/size_mb
   Group by: config
   Title: "KV-Cache Growth vs Sequence Length"

   PANEL 6 — Arithmetic Intensity (Bar Chart)
   Metric: profiling/arithmetic_intensity
   Group by: config
   Title: "Arithmetic Intensity by Config"

   PANEL 7 — Roofline Plots (Image Panel)
   Upload: roofline_prefill.png, roofline_decode.png
   (generated by roofline.py after profiling.py runs)

4. Arrange panels in a logical order:
   Row 1: TTFT + Decode throughput    (what happened)
   Row 2: Memory + KV-cache           (memory story)
   Row 3: Arithmetic intensity        (why it happened)
   Row 4: Roofline plots              (the deep dive)

When real data comes in from teammates:
   -> Filter dashboard to hide DUMMY runs
   -> Real runs automatically fill the same panels
""")


if __name__ == "__main__":
    main()
