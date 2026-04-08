"""Core benchmarking: isolate prefill (TTFT) and decode throughput.

Design
------
*Prefill* is measured by running a **single forward pass** with the full
prompt and recording wall-clock time until the first new token logit is
produced.  This corresponds to the Time-To-First-Token (TTFT) metric.

*Decode* is measured by continuing autoregressive generation for a fixed
number of new tokens with KV-cache reuse, recording wall-clock time and
computing tokens/sec.

Both phases use ``torch.cuda.Event`` based timing (not ``time.time()``)
to avoid CPU/GPU synchronisation noise.  Peak GPU memory is captured via
``torch.cuda.max_memory_allocated``.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .model import AttentionBackend, force_attention_backend


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PhaseMetrics:
    """Timing and memory results for a single phase."""

    phase: str  # "prefill" or "decode"
    elapsed_ms: float
    tokens: int
    tokens_per_sec: float
    peak_memory_mb: float


@dataclass
class BenchmarkResult:
    """Complete result for one (model, seq_len, config) run."""

    model_name: str
    sequence_length: int
    decode_tokens: int
    backend: str
    prefill: PhaseMetrics
    decode: PhaseMetrics


# ---------------------------------------------------------------------------
# CUDA timing helpers
# ---------------------------------------------------------------------------

def _cuda_sync_and_reset_memory(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    torch.cuda.empty_cache()


def _timed_region(device: torch.device):
    """Return (start, end) CUDA events for a timed region."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


# ---------------------------------------------------------------------------
# Phase-isolated benchmark
# ---------------------------------------------------------------------------

@torch.inference_mode()
def benchmark_single(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
    decode_tokens: int,
    backend: AttentionBackend,
    device: torch.device,
) -> BenchmarkResult:
    """Run one complete prefill+decode pass and return per-phase metrics.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace causal-LM already on *device*.
    inputs : dict
        ``{"input_ids": Tensor, "attention_mask": Tensor}`` on *device*.
    decode_tokens : int
        Number of new tokens to generate in the decode phase.
    backend : AttentionBackend
        Which SDPA kernel to force.
    device : torch.device
        CUDA device.

    Returns
    -------
    BenchmarkResult
    """
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = input_ids.shape[1]

    # ---- PREFILL PHASE ---------------------------------------------------
    # A single forward pass over the full prompt.  The output includes the
    # KV-cache (past_key_values) that will feed the decode phase.
    _cuda_sync_and_reset_memory(device)

    prefill_start, prefill_end = _timed_region(device)
    with force_attention_backend(backend):
        prefill_start.record()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        prefill_end.record()

    torch.cuda.synchronize(device)
    prefill_ms = prefill_start.elapsed_time(prefill_end)
    prefill_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Greedy-select the first new token from the prefill logits
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values

    # ---- DECODE PHASE ----------------------------------------------------
    # Autoregressive token-by-token generation reusing past KV-cache.
    _cuda_sync_and_reset_memory(device)

    generated_ids = [next_token]
    cur_attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
        dim=1,
    )

    decode_start, decode_end = _timed_region(device)
    with force_attention_backend(backend):
        decode_start.record()
        for _ in range(decode_tokens - 1):
            outputs = model(
                input_ids=next_token,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            generated_ids.append(next_token)
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones((1, 1), device=device, dtype=cur_attention_mask.dtype)],
                dim=1,
            )
        decode_end.record()

    torch.cuda.synchronize(device)
    decode_ms = decode_start.elapsed_time(decode_end)
    decode_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # ---- Assemble results ------------------------------------------------
    prefill_metrics = PhaseMetrics(
        phase="prefill",
        elapsed_ms=prefill_ms,
        tokens=seq_len,
        tokens_per_sec=(seq_len / (prefill_ms / 1000.0)) if prefill_ms > 0 else 0.0,
        peak_memory_mb=prefill_peak_mb,
    )
    decode_metrics = PhaseMetrics(
        phase="decode",
        elapsed_ms=decode_ms,
        tokens=decode_tokens,
        tokens_per_sec=(decode_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0,
        peak_memory_mb=decode_peak_mb,
    )
    return BenchmarkResult(
        model_name=model.config._name_or_path,
        sequence_length=seq_len,
        decode_tokens=decode_tokens,
        backend=backend.value,
        prefill=prefill_metrics,
        decode=decode_metrics,
    )


# ---------------------------------------------------------------------------
# Multi-iteration driver (warmup + repeats)
# ---------------------------------------------------------------------------

def run_benchmark(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
    decode_tokens: int,
    backend: AttentionBackend,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> List[BenchmarkResult]:
    """Execute *warmup* + *repeats* runs and return only the timed results."""
    for _ in range(warmup):
        benchmark_single(model, inputs, decode_tokens, backend, device)

    results: List[BenchmarkResult] = []
    for _ in range(repeats):
        r = benchmark_single(model, inputs, decode_tokens, backend, device)
        results.append(r)
    return results
