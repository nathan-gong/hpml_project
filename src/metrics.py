"""CUDA memory/profiling helpers for inference benchmarking."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MemorySnapshot:
    allocated_mb: float
    reserved_mb: float


@dataclass
class PeakMemoryStats:
    peak_allocated_mb: float
    peak_reserved_mb: float



def bytes_to_mb(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024 ** 2)



def cuda_available_for_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()



def reset_cuda_peak_stats(device: torch.device) -> None:
    if not cuda_available_for_device(device):
        return
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)



def snapshot_cuda_memory(device: torch.device) -> MemorySnapshot:
    if not cuda_available_for_device(device):
        return MemorySnapshot(allocated_mb=0.0, reserved_mb=0.0)
    return MemorySnapshot(
        allocated_mb=bytes_to_mb(torch.cuda.memory_allocated(device)),
        reserved_mb=bytes_to_mb(torch.cuda.memory_reserved(device)),
    )



def get_peak_cuda_memory(device: torch.device) -> PeakMemoryStats:
    if not cuda_available_for_device(device):
        return PeakMemoryStats(peak_allocated_mb=0.0, peak_reserved_mb=0.0)
    return PeakMemoryStats(
        peak_allocated_mb=bytes_to_mb(torch.cuda.max_memory_allocated(device)),
        peak_reserved_mb=bytes_to_mb(torch.cuda.max_memory_reserved(device)),
    )



def query_nvidia_smi() -> dict[str, Any]:
    """Best-effort utilization proxy via nvidia-smi.

    This is intentionally lightweight and optional. If nvidia-smi is missing or
    inaccessible, an empty dict is returned.
    """
    fields = [
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception:
        return {}

    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    if not line:
        return {}

    values = [part.strip() for part in line.split(",")]
    if len(values) != len(fields):
        return {"raw": line}
    parsed: dict[str, Any] = dict(zip(fields, values))
    for key in ["utilization.gpu", "utilization.memory", "memory.used", "memory.total"]:
        if key in parsed:
            try:
                parsed[key] = float(parsed[key])
            except ValueError:
                pass
    return parsed



def serialize_dataclass(instance: Any) -> dict[str, Any]:
    """Convert nested dataclasses into plain JSON-ready dicts."""
    return json.loads(json.dumps(instance, default=lambda o: o.__dict__))
