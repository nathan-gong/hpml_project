"""Model loading with explicit attention-backend control.

The proposal requires that the baseline use **Standard SDPA** (the MATH
kernel), *not* FlashAttention or the memory-efficient backend that modern
PyTorch may auto-select.  We enforce this via context managers from
``torch.nn.attention.sdpa_kernel``.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Generator

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .quantization import build_quantization_spec


class AttentionBackend(str, Enum):
    """Attention backends matching the proposal's Table I."""

    MATH = "math"  # Standard SDPA (baseline)
    FLASH = "flash"  # FlashAttention-2
    MEMORY_EFFICIENT = "memory_efficient"


_BACKEND_MAP = {
    AttentionBackend.MATH: [SDPBackend.MATH],
    AttentionBackend.FLASH: [SDPBackend.FLASH_ATTENTION],
    AttentionBackend.MEMORY_EFFICIENT: [SDPBackend.EFFICIENT_ATTENTION],
}


@contextmanager
def force_attention_backend(
    backend: AttentionBackend,
) -> Generator[None, None, None]:
    """Context manager that forces PyTorch to use *only* the given SDPA backend.

    Example::

        with force_attention_backend(AttentionBackend.MATH):
            outputs = model.generate(...)
    """
    with sdpa_kernel(_BACKEND_MAP[backend]):
        yield


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    precision: str = "fp16",
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load a HuggingFace causal-LM with optional quantized weights.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``meta-llama/Llama-2-7b-hf``).
    device : str
        Target device (``"cuda"`` or ``"cpu"``).
    dtype : torch.dtype
        Baseline weight precision.  Baseline = ``torch.float16``.
    precision : str
        Requested model precision mode. Supported values are ``"fp16"``,
        ``"int8"``, and ``"4bit"``. Quantized modes route through
        ``src.quantization`` and currently use bitsandbytes-backed loading.

    Returns
    -------
    model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    spec = build_quantization_spec(precision=precision, compute_dtype=dtype)

    model_kwargs = dict(
        torch_dtype=spec.torch_dtype,
        device_map=device,
        attn_implementation="sdpa",  # use PyTorch-native SDPA path
    )
    model_kwargs.update(spec.model_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.eval()
    return model, tokenizer
