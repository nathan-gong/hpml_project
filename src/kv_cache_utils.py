"""Utilities for estimating KV-cache size from Hugging Face model outputs."""

from __future__ import annotations

from typing import Any

import torch



def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()



def get_past_key_values_bytes(past_key_values: Any) -> int:
    """Return total bytes occupied by the KV-cache tensors.

    Works with the standard Hugging Face tuple-of-tuples cache format where
    each layer exposes ``(key, value)``.
    """
    if past_key_values is None:
        return 0

    total = 0
    for layer in past_key_values:
        if isinstance(layer, (list, tuple)):
            for tensor in layer:
                if torch.is_tensor(tensor):
                    total += _tensor_bytes(tensor)
        elif torch.is_tensor(layer):
            total += _tensor_bytes(layer)
    return total



def summarize_past_key_values(past_key_values: Any) -> dict[str, float | int | list[dict[str, Any]]]:
    """Return a structured KV-cache summary useful for logging/debugging."""
    if past_key_values is None:
        return {
            "num_layers": 0,
            "total_bytes": 0,
            "total_mb": 0.0,
            "layers": [],
        }

    layer_summaries: list[dict[str, Any]] = []
    total = 0
    for idx, layer in enumerate(past_key_values):
        key_bytes = 0
        value_bytes = 0
        key_shape = None
        value_shape = None
        dtype = None
        if isinstance(layer, (list, tuple)) and len(layer) >= 2:
            key, value = layer[0], layer[1]
            if torch.is_tensor(key):
                key_bytes = _tensor_bytes(key)
                key_shape = tuple(key.shape)
                dtype = str(key.dtype)
            if torch.is_tensor(value):
                value_bytes = _tensor_bytes(value)
                value_shape = tuple(value.shape)
                dtype = dtype or str(value.dtype)
        layer_total = key_bytes + value_bytes
        total += layer_total
        layer_summaries.append(
            {
                "layer": idx,
                "key_shape": key_shape,
                "value_shape": value_shape,
                "dtype": dtype,
                "bytes": layer_total,
                "mb": round(layer_total / (1024 ** 2), 4),
            }
        )

    return {
        "num_layers": len(layer_summaries),
        "total_bytes": total,
        "total_mb": round(total / (1024 ** 2), 4),
        "layers": layer_summaries,
    }
