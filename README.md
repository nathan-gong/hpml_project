# hpml_project

##  Roofline-Aware LLM Inference Analysis

## Week 1: Unoptimised Baseline (SDPA + FP16)

### Setup

```bash
pip install -r requirements.txt
```

If using a gated model (e.g. Llama-2), authenticate first:

```bash
huggingface-cli login
```

### Running the Baseline

Full sweep across all sequence lengths (128, 256, 512, 1024):

```bash
python run_baseline.py --model meta-llama/Llama-2-7b-hf
```

Quick sanity check (1 repeat, no warmup, no W&B):

```bash
python run_baseline.py --model meta-llama/Llama-2-7b-hf \
    --warmup 0 --repeats 1 --no-wandb
```

Custom sequence lengths:

```bash
python run_baseline.py --model mistralai/Mistral-7B-v0.1 \
    --seq-lengths 128 512 --decode-tokens 64
```

### What Gets Measured

| Metric | Phase | Description |
|--------|-------|-------------|
| **TTFT (ms)** | Prefill | Wall-clock time from prompt submission to first output logit |
| **Tokens/sec** | Decode | Autoregressive generation throughput after first token |
| **Peak GPU Memory (MB)** | Both | `torch.cuda.max_memory_allocated` per phase |

Phases are isolated:
- **Prefill**: A single forward pass over the full prompt (produces KV-cache).
- **Decode**: Token-by-token generation reusing the KV-cache.

Timing uses `torch.cuda.Event` for accurate GPU-side measurement.

### Output

- Console summary table
- `baseline_results.json` — full metrics per sequence length
- W&B dashboard (unless `--no-wandb`)

### Project Structure

```
hpml_project/
│
├── setup/
│   ├── download_model.py      ← download model to GCP cache
│   ├── generate_data.py       ← generate prompt .pt files
│   └── baseline_inference.py  ← quick environment sanity check
│
├── src/
│   ├── config.py              ← all settings (seq lengths, W&B project, etc.)
│   ├── data.py                ← exact-length prompt generation (Wikipedia)
│   ├── model.py               ← model loading + attention backend control
│   ├── quantization.py        ← INT8 / 4bit quantization config
│   ├── benchmark.py           ← phase-isolated benchmarking core
│   ├── metrics.py             ← CUDA memory + nvidia-smi helpers
│   └── kv_cache_utils.py      ← KV-cache size tracking
│
├── run_baseline.py            ← Config 1: FP16 + SDPA
├── run_quantized.py           ← Configs 2,3,4: flexible precision + backend
├── profiling.py               ← Part 4: PyTorch Profiler + roofline data
├── roofline.py                ← Part 4: roofline plot generation
├── dashboard_test.py          ← Part 4: W&B dashboard preview with dummy data
└── requirements.txt
```

### Key Design Decisions

1. **Explicit SDPA backend selection** via `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` prevents PyTorch from auto-selecting FlashAttention, ensuring a valid unoptimised baseline.

2. **Phase isolation** — prefill and decode are timed independently with separate CUDA events and memory tracking. The KV-cache produced during prefill is passed to the decode loop.

3. **CUDA event timing** — avoids CPU/GPU sync noise inherent in `time.time()`.

4. **Deterministic prompts** — Wikipedia text is tokenised and truncated to exact target lengths for reproducibility.

## Part 4 — Analysis & Visualization

This section covers the hardware profiling, roofline plot generation, and W&B dashboard building on top of the benchmark pipeline.

### Step 1 — Run System Profiling

Wraps PyTorch Profiler around `benchmark_single()` to collect kernel-level FLOPS, memory traffic, and arithmetic intensity needed for the roofline plot.

```bash
# Full run — all 4 configs x all 5 sequence lengths
python profiling.py --model meta-llama/Meta-Llama-3.1-8B-Instruct

# Quick test with small model
python profiling.py --model gpt2 --seq-lengths 128 --configs baseline --no-wandb
```

**Outputs:**
```
profiling_results.json   ← key input for roofline.py
kernel_breakdown.json    ← top 10 kernels per config
traces/                  ← Chrome trace files (open in chrome://tracing)
```

### Step 2 — Generate Roofline Plots

Reads `profiling_results.json` and maps each config onto the NVIDIA L4 hardware roofline. Prefill and decode phases are plotted separately for clarity.

```bash
# Generate plots locally
python roofline.py --input profiling_results.json --output-dir plots

# Generate and upload to W&B
python roofline.py --input profiling_results.json --output-dir plots --wandb
```

**Outputs:**
```
plots/roofline_prefill.png       ← prefill phase roofline (all configs x all seq lengths)
plots/roofline_decode.png        ← decode phase roofline (all configs x all seq lengths)
plots/arithmetic_intensity.png   ← bar chart of arithmetic intensity per config
```

### Step 3 — W&B Dashboard Preview

Run `dashboard_test.py` to populate W&B with realistic dummy data and build the dashboard layout before real GCP results arrive:

```bash
python dashboard_test.py
```

Then go to **wandb.ai** and arrange these panels:

| Panel | Chart Type | Metric |
|-------|-----------|--------|
| TTFT Comparison | Bar chart | `prefill/ttft_ms` |
| Decode Throughput | Bar chart | `decode/tokens_per_sec` |
| Performance Scaling | Line chart | `prefill/ttft_ms` vs `seq_len` |
| Peak Memory | Bar chart | `prefill/peak_mem_mb` |
| KV-Cache Growth | Line chart | `kv_cache/size_mb` vs `seq_len` |
| Arithmetic Intensity | Bar chart | `profiling/arithmetic_intensity` |
| Roofline Plots | Image panel | uploaded PNGs from `roofline.py` |

### Full Run Order

```bash
# Step 1 — baseline performance
python run_baseline.py --model meta-llama/Meta-Llama-3.1-8B-Instruct

# Step 2 — FlashAttention
python run_quantized.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --precision fp16 --backend flash

# Step 3 — Quantized
python run_quantized.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --precision 4bit --backend math

# Step 4 — FA + Quant
python run_quantized.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --precision 4bit --backend flash

# Step 5 — hardware profiling
python profiling.py --model meta-llama/Meta-Llama-3.1-8B-Instruct

# Step 6 — roofline plots
python roofline.py --wandb
