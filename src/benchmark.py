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

Phase A extension
-----------------
This version preserves the original phase-isolated benchmark structure and
adds:
- richer CUDA memory snapshots (before/after/peak allocated + reserved)
- KV-cache size estimates from actual ``past_key_values`` tensors
- optional ``nvidia-smi`` utilization proxy snapshots
- precision / parameter-byte metadata for quantized-model comparisons
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from transformers import PreTrainedModel

from .kv_cache_utils import get_past_key_values_bytes, summarize_past_key_values
from .metrics import (
    MemorySnapshot,
    PeakMemoryStats,
    get_peak_cuda_memory,
    query_nvidia_smi,
    reset_cuda_peak_stats,
    snapshot_cuda_memory,
)
from .model import AttentionBackend, force_attention_backend
from .quantization import estimate_parameter_bytes


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
    peak_reserved_memory_mb: float
    memory_before_mb: float
    memory_after_mb: float
    reserved_before_mb: float
    reserved_after_mb: float
    memory_delta_mb: float
    reserved_delta_mb: float
    kv_cache_bytes: int = 0
    kv_cache_mb: float = 0.0
    kv_cache_summary: dict[str, Any] = field(default_factory=dict)
    utilization_proxy: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete result for one (model, seq_len, config) run."""

    model_name: str
    sequence_length: int
    decode_tokens: int
    backend: str
    precision: str
    parameter_bytes: int
    prefill: PhaseMetrics
    decode: PhaseMetrics


# ---------------------------------------------------------------------------
# CUDA timing helpers
# ---------------------------------------------------------------------------

def _cuda_sync_and_reset_memory(device: torch.device) -> None:
    """Synchronize CUDA, clear caches, and reset peak-memory statistics."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    torch.cuda.empty_cache()


def _timed_region(device: torch.device):
    """Return (start, end) CUDA events for a timed region."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("benchmark.py expects a CUDA device for timed benchmarking")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def _build_phase_metrics(
    *,
    phase: str,
    elapsed_ms: float,
    tokens: int,
    before: MemorySnapshot,
    after: MemorySnapshot,
    peaks: PeakMemoryStats,
    kv_cache: Any,
    utilization_proxy: dict[str, Any] | None = None,
) -> PhaseMetrics:
    """Assemble the extended PhaseMetrics record for one benchmark phase."""
    kv_cache_bytes = get_past_key_values_bytes(kv_cache)
    return PhaseMetrics(
        phase=phase,
        elapsed_ms=elapsed_ms,
        tokens=tokens,
        tokens_per_sec=(tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
        peak_memory_mb=peaks.peak_allocated_mb,
        peak_reserved_memory_mb=peaks.peak_reserved_mb,
        memory_before_mb=before.allocated_mb,
        memory_after_mb=after.allocated_mb,
        reserved_before_mb=before.reserved_mb,
        reserved_after_mb=after.reserved_mb,
        memory_delta_mb=after.allocated_mb - before.allocated_mb,
        reserved_delta_mb=after.reserved_mb - before.reserved_mb,
        kv_cache_bytes=kv_cache_bytes,
        kv_cache_mb=kv_cache_bytes / (1024 ** 2),
        kv_cache_summary=summarize_past_key_values(kv_cache),
        utilization_proxy=utilization_proxy or {},
    )


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
    precision: str = "fp16",
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
    precision : str
        Precision label for logging (e.g. ``fp16``, ``int8``, ``4bit``).

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

    prefill_before = snapshot_cuda_memory(device)
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
    prefill_after = snapshot_cuda_memory(device)
    prefill_peaks = get_peak_cuda_memory(device)
    prefill_util = query_nvidia_smi() if device.type == "cuda" else {}

    # Greedy-select the first new token from the prefill logits
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    prefill_cache = past_key_values

    # ---- DECODE PHASE ----------------------------------------------------
    # Autoregressive token-by-token generation reusing past KV-cache.
    _cuda_sync_and_reset_memory(device)

    cur_attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
        dim=1,
    )

    decode_before = snapshot_cuda_memory(device)
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
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones((1, 1), device=device, dtype=cur_attention_mask.dtype)],
                dim=1,
            )
        decode_end.record()

    torch.cuda.synchronize(device)
    decode_ms = decode_start.elapsed_time(decode_end)
    decode_after = snapshot_cuda_memory(device)
    decode_peaks = get_peak_cuda_memory(device)
    decode_util = query_nvidia_smi() if device.type == "cuda" else {}

    # ---- Assemble results ------------------------------------------------
    prefill_metrics = _build_phase_metrics(
        phase="prefill",
        elapsed_ms=prefill_ms,
        tokens=seq_len,
        before=prefill_before,
        after=prefill_after,
        peaks=prefill_peaks,
        kv_cache=prefill_cache,
        utilization_proxy=prefill_util,
    )

    decode_metrics = _build_phase_metrics(
        phase="decode",
        elapsed_ms=decode_ms,
        tokens=decode_tokens,
        before=decode_before,
        after=decode_after,
        peaks=decode_peaks,
        kv_cache=past_key_values,
        utilization_proxy=decode_util,
    )

    return BenchmarkResult(
        model_name=model.config._name_or_path,
        sequence_length=seq_len,
        decode_tokens=decode_tokens,
        backend=backend.value,
        precision=precision,
        parameter_bytes=estimate_parameter_bytes(model),
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
    precision: str = "fp16",
) -> List[BenchmarkResult]:
    """Execute *warmup* + *repeats* runs and return only the timed results."""
    for _ in range(warmup):
        benchmark_single(model, inputs, decode_tokens, backend, device, precision=precision)

    results: List[BenchmarkResult] = []
    for _ in range(repeats):
        r = benchmark_single(model, inputs, decode_tokens, backend, device, precision=precision)
        results.append(r)
    return results
