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
src/
  config.py      — Constants and BenchmarkConfig dataclass
  data.py        — Synthetic prompt generation (Wikipedia source)
  model.py       — Model loading + attention backend control
  benchmark.py   — Phase-isolated benchmarking core
run_baseline.py  — CLI entry point for the baseline sweep
```

### Key Design Decisions

1. **Explicit SDPA backend selection** via `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` prevents PyTorch from auto-selecting FlashAttention, ensuring a valid unoptimised baseline.

2. **Phase isolation** — prefill and decode are timed independently with separate CUDA events and memory tracking. The KV-cache produced during prefill is passed to the decode loop.

3. **CUDA event timing** — avoids CPU/GPU sync noise inherent in `time.time()`.

4. **Deterministic prompts** — Wikipedia text is tokenised and truncated to exact target lengths for reproducibility.
