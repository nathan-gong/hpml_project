"""Central configuration for benchmark experiments."""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Sequence lengths to sweep (Table I in proposal)
# ---------------------------------------------------------------------------
SEQUENCE_LENGTHS: List[int] = [128, 256, 512, 1024]

# Number of *new* tokens to generate during decode
DEFAULT_DECODE_TOKENS: int = 128

# Warm-up iterations discarded before timing
DEFAULT_WARMUP: int = 2

# Timed iterations per (model, seq_len) pair
DEFAULT_REPEATS: int = 5

# Candidate primary models (pick whichever fits VRAM)
PRIMARY_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-v0.1",
]

# Default model for quick local testing
DEFAULT_MODEL: str = "meta-llama/Llama-2-7b-hf"


@dataclass
class BenchmarkConfig:
    """Single benchmark run configuration."""

    model_name: str = DEFAULT_MODEL
    sequence_lengths: List[int] = field(default_factory=lambda: list(SEQUENCE_LENGTHS))
    decode_tokens: int = DEFAULT_DECODE_TOKENS
    warmup: int = DEFAULT_WARMUP
    repeats: int = DEFAULT_REPEATS
    batch_size: int = 1
    seed: int = 42
    wandb_project: str = "hpml2026-final-project"
    wandb_enabled: bool = True
    device: str = "cuda"
