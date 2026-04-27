"""Switch trace export utilities.

Classes:
- TraceAccumulator: per-stage / per-layer ring buffers for router records
- ContractWriter:   serialize accumulator buffers to train_predict_model contract
- SwitchRunner:     load HF Switch model, install hooks, run batched generate
- Verifier:         contract-level assertions on a written trace directory
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


STAGES = ("encoder", "decoder")
NUM_SPARSE_LAYERS = 6
EXPECTED_NUM_EXPERTS = 128
PER_TOKEN_EXPERT = 1


@dataclass
class TraceAccumulator:
    """Holds raw per-token router records for one stage."""
    num_layers: int = NUM_SPARSE_LAYERS

    def __post_init__(self) -> None:
        raise NotImplementedError

    def add_token(
        self,
        seq_id: int,
        token_idx_in_seq: int,
        token_id: int,
        per_layer_logits: list[torch.Tensor],
    ) -> None:
        raise NotImplementedError

    def total_tokens(self) -> int:
        raise NotImplementedError


class ContractWriter:
    """Serialize a TraceAccumulator to a directory matching train_predict_model contract."""

    def __init__(self, out_dir: Path, stage: str):
        raise NotImplementedError

    def write(self, acc: TraceAccumulator, extra_metadata: Optional[dict] = None) -> None:
        raise NotImplementedError


class SwitchRunner:
    """Load Switch model, install hooks, run batched generate."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        raise NotImplementedError

    def run(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int,
    ) -> tuple[TraceAccumulator, TraceAccumulator]:
        """Returns (encoder_acc, decoder_acc)."""
        raise NotImplementedError


class Verifier:
    @staticmethod
    def verify(trace_dir: Path) -> None:
        """Raise AssertionError on any contract violation."""
        raise NotImplementedError
