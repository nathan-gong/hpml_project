"""Model loading with explicit attention-backend control.

The proposal requires that the baseline use **Standard SDPA** (the MATH
kernel), *not* FlashAttention or the memory-efficient backend that modern
PyTorch may auto-select.  We enforce this via context managers from
``torch.nn.attention.sdpa_kernel``.
"""

from contextlib import contextmanager
from enum import Enum
from typing import Generator

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


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
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load a HuggingFace causal-LM in FP16 on *device*.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``meta-llama/Llama-2-7b-hf``).
    device : str
        Target device (``"cuda"`` or ``"cpu"``).
    dtype : torch.dtype
        Weight precision.  Baseline = ``torch.float16``.

    Returns
    -------
    model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",  # use PyTorch-native SDPA path
    )
    model.eval()
    return model, tokenizer
