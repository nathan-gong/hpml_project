"""Quantization helpers for Hugging Face causal LMs.

This module keeps quantization-specific logic out of the main model loader so
benchmark code can request a precision mode without worrying about the
underlying HF / bitsandbytes configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - depends on transformers install
    BitsAndBytesConfig = None  # type: ignore[assignment]


SUPPORTED_PRECISIONS = {"fp16", "int8", "4bit"}


@dataclass(frozen=True)
class QuantizationSpec:
    """Normalized quantization configuration used by the model loader."""

    precision: str
    torch_dtype: torch.dtype
    model_kwargs: dict[str, Any]



def _require_bitsandbytes() -> None:
    if BitsAndBytesConfig is None:
        raise ImportError(
            "bitsandbytes/transformers quantization support is unavailable. "
            "Install bitsandbytes and a recent transformers version."
        )



def build_quantization_spec(
    precision: str = "fp16",
    *,
    compute_dtype: torch.dtype = torch.float16,
) -> QuantizationSpec:
    """Return normalized model-loading settings for the requested precision.

    Parameters
    ----------
    precision:
        One of ``fp16``, ``int8``, or ``4bit``.
    compute_dtype:
        Internal compute dtype for 4-bit kernels. FP16 is a good default for
        the project baseline and common cloud GPUs.
    """
    precision = precision.lower()
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision '{precision}'. Expected one of "
            f"{sorted(SUPPORTED_PRECISIONS)}."
        )

    if precision == "fp16":
        return QuantizationSpec(
            precision=precision,
            torch_dtype=torch.float16,
            model_kwargs={},
        )

    _require_bitsandbytes()

    if precision == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return QuantizationSpec(
            precision=precision,
            torch_dtype=torch.float16,
            model_kwargs={"quantization_config": bnb_config},
        )

    # 4-bit weights with FP16 compute is a strong default for inference.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    return QuantizationSpec(
        precision=precision,
        torch_dtype=compute_dtype,
        model_kwargs={"quantization_config": bnb_config},
    )



def estimate_parameter_bytes(model: torch.nn.Module) -> int:
    """Estimate bytes currently occupied by model parameters.

    For quantized modules this is still useful as a practical approximation of
    the live parameter footprint exposed through PyTorch tensors.
    """
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    return total
