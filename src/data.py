"""Synthetic prompt generation for controlled-length inference benchmarks.

Builds prompts from Wikipedia text via HuggingFace `datasets` so that the
*tokenised* input has exactly the requested number of tokens.
"""

from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


# We cache a chunk of raw text on first call so repeated prompt builds are
# fast and deterministic.
_TEXT_CACHE: str = ""


def _load_source_text(min_chars: int = 500_000) -> str:
    """Return a large block of Wikipedia text (cached after first call)."""
    global _TEXT_CACHE
    if _TEXT_CACHE:
        return _TEXT_CACHE

    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    buf: List[str] = []
    total = 0
    for row in ds:
        text = row["text"]
        buf.append(text)
        total += len(text)
        if total >= min_chars:
            break
    _TEXT_CACHE = "\n".join(buf)
    return _TEXT_CACHE


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    target_length: int,
    *,
    source_text: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Return token-ids of exactly *target_length* tokens.

    Strategy: tokenise a large source text, then truncate / pad to the
    desired length.  Because we always truncate from the same document the
    content is deterministic across runs.
    """
    if source_text is None:
        source_text = _load_source_text()

    all_ids = tokenizer.encode(source_text, add_special_tokens=False)

    if len(all_ids) < target_length:
        # Repeat to fill (unlikely for Wikipedia, but safe)
        repeats = (target_length // len(all_ids)) + 1
        all_ids = (all_ids * repeats)[:target_length]
    else:
        all_ids = all_ids[:target_length]

    input_ids = torch.tensor([all_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
