# HPML Final Project — Roofline-Aware LLM Inference Analysis

## Overview

This project conducts a roofline-aware performance analysis of LLM inference, studying how **FlashAttention-2** and **weight quantization** shift compute vs. memory bottlenecks across inference regimes.

LLM inference has two distinct phases:
- **Prefill** — processes the full prompt in parallel (transitions from memory-bound to compute-bound as sequence length grows)
- **Decode** — generates tokens one at a time via KV-cache (always memory-bandwidth bound)

### 4 Experiment Configurations

| Config | Attention Backend | Weight Precision |
|--------|------------------|-----------------|
| Baseline | Standard SDPA | FP16 |
| FlashAttention | FlashAttention-2 | FP16 |
| Quantized | Standard SDPA | 4bit |
| FA + Quant | FlashAttention-2 | 4bit |

### Sequence Lengths: `[128, 256, 512, 1024, 2048]`

---

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login
wandb login

# Run once on GCP
python setup/download_model.py
python setup/generate_data.py
```

---

## Project Structure

```
hpml_project/
│
├── setup/
│   ├── download_model.py      ← download model to GCP cache
│   ├── generate_data.py       ← generate exact-length prompt .pt files
│   └── baseline_inference.py  ← quick environment sanity check
│
├── src/
│   ├── config.py              ← all settings (seq lengths, W&B project, etc.)
│   ├── data.py                ← exact-length prompt generation from Wikipedia
│   ├── model.py               ← model loading + attention backend control
│   ├── quantization.py        ← INT8 / 4bit quantization config
│   ├── benchmark.py           ← phase-isolated prefill + decode measurement
│   ├── metrics.py             ← CUDA memory + nvidia-smi helpers
│   └── kv_cache_utils.py      ← KV-cache memory tracking per layer
│
├── run_baseline.py            ← Config 1: FP16 + Standard SDPA
├── run_quantized.py           ← Configs 2, 3, 4: flexible precision + backend
├── profiling.py               ← PyTorch Profiler + roofline data collection
├── roofline.py                ← roofline plot generation
├── dashboard_test.py          ← W&B dashboard preview with dummy data
└── requirements.txt
```

---

## Part 1 — Infrastructure Setup

```bash
python setup/download_model.py    # download model to GCP cache
python setup/generate_data.py     # generate prompt files
python setup/baseline_inference.py # verify environment
```

---

## Part 2 — Core Inference & Instrumentation

The `src/` folder implements the core benchmarking engine used by all experiment scripts. Key design decisions:

1. **Explicit SDPA backend** — prevents PyTorch from auto-selecting FlashAttention
2. **Phase isolation** — prefill and decode timed independently with CUDA events and memory resets
3. **CUDA event timing** — avoids CPU/GPU sync noise from `time.time()`
4. **Deterministic prompts** — Wikipedia text truncated to exact token lengths

---

## Part 3 — Experiments (2 warmup + 5 timed repeats → mean ± stdev)

### Config 1 — Baseline
```bash
python run_baseline.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output baseline_results.json
```

### Config 2 — FlashAttention
```bash
python run_quantized.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --precision fp16 --backend flash \
    --output results_flashattn.json
```

### Config 3 — Quantized
```bash
python run_quantized.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --precision 4bit --backend math \
    --output results_quantized.json
```

### Config 4 — FA + Quant
```bash
python run_quantized.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --precision 4bit --backend flash \
    --output results_fa_quant.json
```

### Quick Test
```bash
python run_baseline.py --model gpt2 \
    --seq-lengths 128 --warmup 0 --repeats 1 --no-wandb
```

### Metrics Collected (Prefill and Decode Separately)

| Metric | Description |
|--------|-------------|
| TTFT (ms) | Time from prompt to first output token |
| Tokens/sec | Decode throughput |
| Peak GPU Memory (MB) | Per phase |
| KV-Cache Size (MB) | Per layer, per phase |
| Parameter Footprint (MB) | Model size in GPU memory |

---

## Part 4 — Analysis & Visualization

### Why 3 Scripts?

```
run_baseline/quantized → reliable performance numbers (5 averaged runs, no profiler overhead)
profiling.py           → hardware-level WHY (FLOPS, arithmetic intensity, kernel breakdown)
roofline.py            → visualizes the WHY (roofline plots)
```

### Step 1 — Profiling
```bash
python profiling.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --warmup 1
```

Outputs: `profiling_results.json`, `kernel_breakdown.json`, `traces/`

### Step 2 — Roofline Plots
```bash
python roofline.py \
    --input profiling_results.json \
    --output-dir plots \
    --wandb
```

Outputs: `roofline_prefill.png`, `roofline_decode.png`, `arithmetic_intensity.png`

### Step 3 — W&B Dashboard

[Link to Dashboard](wandb.ai/hpml2026-final-project/hpml2026-final-project)

```bash
python dashboard_test.py  # builds layout with dummy data
```

---

## Full Run Order

```bash
python run_baseline.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --output baseline_results.json
python run_quantized.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --precision fp16 --backend flash --output results_flashattn.json
python run_quantized.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --precision 4bit --backend math --output results_quantized.json
python run_quantized.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --precision 4bit --backend flash --output results_fa_quant.json
python profiling.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --warmup 1
python roofline.py --input profiling_results.json --output-dir plots --wandb
```

---

## Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA L4 (24 GB) |
| Peak Compute (FP16) | 121 TFLOPS |
| Memory Bandwidth | 300 GB/s |
| Ridge Point | ~403 FLOPS/byte |
| Platform | Google Cloud Platform |

---

## AI USAGE DISCLOSURE

**Did you use any AI tool in completing this assignment?**
Yes, I used AI assistance as described below.

**Tool(s) used:** ChatGPT, Claude

**Specific purpose:** Debugged implementation issues, clarified GPU roofline concepts, assisted with code structure and visualization

**Questions/sections affected:** Infrastructure setup, profiling implementation, roofline plot generation, W&B dashboard

**How you verified correctness:** Ran all code on GCP, confirmed outputs matched expected results, manually verified calculation formulas against hardware specs

*By submitting this assignment, I confirm that the analysis, interpretations, and conclusions are my own, and that any AI assistance is fully disclosed above.*
