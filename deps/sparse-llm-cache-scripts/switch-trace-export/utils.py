"""Switch trace export utilities.

Classes:
- TraceAccumulator: per-stage / per-layer ring buffers for router records
- ContractWriter:   serialize accumulator buffers to train_predict_model contract
- SwitchRunner:     load HF Switch model, install hooks, run batched generate
- Verifier:         contract-level assertions on a written trace directory
"""
from __future__ import annotations

import json
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
    num_experts: int = EXPECTED_NUM_EXPERTS

    def __post_init__(self) -> None:
        self._seq_ids: list[int] = []
        self._token_idx: list[int] = []
        self._token_ids: list[int] = []
        self._per_layer: list[list[torch.Tensor]] = [[] for _ in range(self.num_layers)]

    def add_token(
        self,
        seq_id: int,
        token_idx_in_seq: int,
        token_id: int,
        per_layer_logits: list[torch.Tensor],
    ) -> None:
        if len(per_layer_logits) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} per-layer logits, got {len(per_layer_logits)}"
            )
        for i, t in enumerate(per_layer_logits):
            if t.dim() != 1:
                raise ValueError(f"layer {i}: expected 1-D logits, got shape {tuple(t.shape)}")
            if t.shape[0] != self.num_experts:
                raise ValueError(
                    f"layer {i}: expected expert dim {self.num_experts}, got {t.shape[0]}"
                )
        self._seq_ids.append(int(seq_id))
        self._token_idx.append(int(token_idx_in_seq))
        self._token_ids.append(int(token_id))
        for i, t in enumerate(per_layer_logits):
            self._per_layer[i].append(t.detach().to(torch.float32).cpu())

    def total_tokens(self) -> int:
        return len(self._seq_ids)

    def seq_ids(self) -> list[int]:
        return list(self._seq_ids)

    def token_idx_in_seq(self) -> list[int]:
        return list(self._token_idx)

    def token_ids(self) -> list[int]:
        return list(self._token_ids)

    def stacked_logits(self) -> torch.Tensor:
        if self.total_tokens() == 0:
            return torch.zeros((0, self.num_layers, self.num_experts), dtype=torch.float32)
        per_layer_stacked = [torch.stack(layer_list, dim=0) for layer_list in self._per_layer]
        return torch.stack(per_layer_stacked, dim=1)  # [N, L, V]


class ContractWriter:
    """Serialize a TraceAccumulator to a directory matching train_predict_model contract."""

    def __init__(self, out_dir: Path, stage: str):
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        self.out_dir = Path(out_dir)
        self.stage = stage

    def write(self, acc: TraceAccumulator, extra_metadata: Optional[dict] = None) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        n = acc.total_tokens()

        gate = acc.stacked_logits()  # [N, L, V] float32
        feat = gate.clone()  # input feature == gate (per design)
        freq = torch.softmax(gate.to(torch.float32), dim=-1)  # [N, L, V]
        sel = gate.argmax(dim=-1, keepdim=True).to(torch.int64)  # [N, L, 1]

        seq_ids = torch.tensor(acc.seq_ids(), dtype=torch.int64)
        tok_idx = torch.tensor(acc.token_idx_in_seq(), dtype=torch.int64)
        tok_ids = torch.tensor(acc.token_ids(), dtype=torch.int64)

        torch.save(sel, self.out_dir / "expert_selection.pt")
        torch.save(feat, self.out_dir / "decode_stage_moe_layer_logits_per_token.pt")
        torch.save(gate, self.out_dir / "decode_stage_moe_layer_gate_logits_per_token.pt")
        torch.save(freq, self.out_dir / "decode_stage_expert_freq_per_token.pt")
        torch.save(tok_ids, self.out_dir / "decode_stage_token_ids_per_token.pt")
        torch.save(seq_ids, self.out_dir / "decode_stage_seq_id_of_token.pt")
        torch.save(tok_idx, self.out_dir / "decode_stage_token_idx_in_seq.pt")

        meta = {
            "stage": self.stage,
            "num_expert": int(gate.shape[-1]),
            "num_moe_layer": int(gate.shape[1]),
            "per_token_expert": PER_TOKEN_EXPERT,
            "N": n,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        (self.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


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
