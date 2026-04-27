"""Switch trace export utilities.

Classes:
- TraceAccumulator: per-stage / per-layer ring buffers for router records
- ContractWriter:   serialize accumulator buffers to train_predict_model contract
- SwitchRunner:     load HF Switch model, install hooks, run batched generate
- Verifier:         contract-level assertions on a written trace directory
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
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
        n = acc.total_tokens()
        if n == 0:
            raise ValueError(
                "refusing to write an empty trace: TraceAccumulator has no tokens. "
                "Downstream train_predict_model.Trace.prepare_tensors() cannot handle empty expert_selection."
            )

        gate = acc.stacked_logits()  # [N, L, V] float32
        feat = gate.clone()  # input feature == gate (per design)
        freq = torch.softmax(gate.to(torch.float32), dim=-1)  # [N, L, V]
        sel = gate.argmax(dim=-1, keepdim=True).to(torch.int64)  # [N, L, 1]

        v = int(gate.shape[-1])
        max_eid = int(sel.max().item())
        if max_eid + 1 != v:
            raise ValueError(
                "observed expert indices in argmax do not cover the full last-dim of gate logits: "
                f"max selected index is {max_eid} but the router tensor has {v} experts "
                f"(downstream infers num_expert from max index + 1; at least one token/layer must select "
                f"expert {v - 1} so the written directory matches the zero-change training contract)."
            )

        self.out_dir.mkdir(parents=True, exist_ok=True)

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
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.model = None
        self.tokenizer = None
        self._hook_handles: list = []
        self._encoder_acc: Optional[TraceAccumulator] = None
        self._decoder_acc: Optional[TraceAccumulator] = None
        self._current_batch_seq_ids: list[int] = []
        self._current_batch_token_ids_per_step: list[list[int]] = []
        self._current_batch_alive_mask: list[bool] = []
        self._current_batch_step: int = 0
        self._stage: str = "idle"  # "idle" | "encoder" | "decoder"
        self._encoder_attn_mask: Optional[torch.Tensor] = None
        self._decoder_step_buffer: dict[int, list[torch.Tensor]] = {}
        self._encoder_slot_buffer: dict[int, torch.Tensor] = {}

    def _seed_all(self) -> None:
        from transformers import set_seed as hf_set_seed

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        hf_set_seed(self.seed)

    def _load_model(self) -> None:
        from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

        self._seed_all()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.float32
        )
        self.model.eval()
        self.model.to(self.device)

    def _install_hooks(self) -> None:
        from transformers.models.switch_transformers.modeling_switch_transformers import (
            SwitchTransformersSparseMLP,
            SwitchTransformersTop1Router,
        )

        if self._hook_handles:
            self._remove_hooks()

        for m in self.model.modules():
            if isinstance(m, SwitchTransformersTop1Router):
                m.jitter_noise = 0.0

        for stack_name, stack in (("encoder", self.model.encoder), ("decoder", self.model.decoder)):
            sparse_id = 0
            for block in stack.block:
                for layer in block.layer:
                    mlp = getattr(layer, "mlp", None)
                    if isinstance(mlp, SwitchTransformersSparseMLP):
                        mlp._promoe_stage = stack_name
                        mlp._promoe_sparse_layer_id = sparse_id
                        h = mlp.register_forward_hook(self._sparse_mlp_hook)
                        self._hook_handles.append(h)
                        sparse_id += 1
            if sparse_id != NUM_SPARSE_LAYERS:
                raise RuntimeError(
                    f"{stack_name}: found {sparse_id} sparse MLPs, expected {NUM_SPARSE_LAYERS}"
                )

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _sparse_mlp_hook(self, module, inputs, output):
        router_logits = self._extract_router_logits(module, inputs, output)
        layer_id = module._promoe_sparse_layer_id
        if module._promoe_stage == "encoder":
            self._encoder_slot_buffer[layer_id] = router_logits.detach().cpu()
        else:
            self._decoder_step_buffer.setdefault(layer_id, []).append(router_logits.detach().cpu())

    def _extract_router_logits(self, module, inputs, output) -> torch.Tensor:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError(
                "Switch sparse MLP hook expected output as (hidden_states, router_logits), "
                f"got {type(output).__name__}"
            )

        router_payload = output[1]
        if isinstance(router_payload, torch.Tensor):
            router_logits = router_payload
        elif isinstance(router_payload, (tuple, list)) and router_payload:
            router_logits = router_payload[0]
        else:
            raise RuntimeError(
                "Switch sparse MLP hook could not find router logits in output[1]; "
                f"got {type(router_payload).__name__}"
            )

        if not isinstance(router_logits, torch.Tensor):
            raise RuntimeError(
                "Switch sparse MLP hook expected router logits tensor, "
                f"got {type(router_logits).__name__}"
            )
        if router_logits.shape[-1] != EXPECTED_NUM_EXPERTS:
            raise RuntimeError(
                f"{module._promoe_stage} sparse layer {module._promoe_sparse_layer_id}: "
                f"expected router logits last dim {EXPECTED_NUM_EXPERTS}, "
                f"got shape {tuple(router_logits.shape)}"
            )
        if router_logits.dim() == 2:
            if not inputs or not isinstance(inputs[0], torch.Tensor) or inputs[0].dim() < 2:
                raise RuntimeError(
                    "Switch sparse MLP hook needs input hidden_states [B, S, ...] "
                    f"to reshape 2-D router logits, got inputs={type(inputs).__name__}"
                )
            B, S = inputs[0].shape[0], inputs[0].shape[1]
            if router_logits.shape[0] != B * S:
                raise RuntimeError(
                    f"{module._promoe_stage} sparse layer {module._promoe_sparse_layer_id}: "
                    f"cannot reshape router logits {tuple(router_logits.shape)} to "
                    f"({B}, {S}, {EXPECTED_NUM_EXPERTS})"
                )
            router_logits = router_logits.view(B, S, -1)
        elif router_logits.dim() != 3:
            raise RuntimeError(
                f"{module._promoe_stage} sparse layer {module._promoe_sparse_layer_id}: "
                f"expected router logits dim 2 or 3, got shape {tuple(router_logits.shape)}"
            )
        return router_logits

    def run(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int,
    ) -> tuple[TraceAccumulator, TraceAccumulator]:
        """Returns (encoder_acc, decoder_acc)."""
        if self.model is None:
            self._load_model()
        if not self._hook_handles:
            self._install_hooks()
        self._encoder_acc = TraceAccumulator()
        self._decoder_acc = TraceAccumulator()

        try:
            for batch_start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[batch_start : batch_start + batch_size]
                global_seq_ids = list(range(batch_start, batch_start + len(batch_prompts)))
                self._run_one_batch(batch_prompts, global_seq_ids, max_new_tokens)
        finally:
            self._remove_hooks()

        return self._encoder_acc, self._decoder_acc

    def _run_one_batch(
        self,
        batch_prompts: list[str],
        global_seq_ids: list[int],
        max_new_tokens: int,
    ) -> None:
        enc_inputs = self.tokenizer(batch_prompts, padding="longest", return_tensors="pt")
        input_ids = enc_inputs["input_ids"].to(self.device)
        attn_mask = enc_inputs["attention_mask"].to(self.device)

        self._encoder_slot_buffer = {}
        self._decoder_step_buffer = {}
        decoder_start = self._decoder_start_token_id()

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                decoder_start_token_id=decoder_start,
                output_scores=False,
                return_dict_in_generate=True,
            )

        gen_seqs = output.sequences
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.model.config.eos_token_id

        self._drain_encoder(global_seq_ids, input_ids.cpu(), attn_mask.cpu())
        self._drain_decoder(global_seq_ids, gen_seqs.cpu(), pad_id, eos_id, decoder_start)

    def _decoder_start_token_id(self) -> int:
        decoder_start = self.model.config.decoder_start_token_id
        if decoder_start is None:
            decoder_start = self.model.config.bos_token_id
        if decoder_start is None:
            decoder_start = self.tokenizer.pad_token_id
        if decoder_start is None:
            raise RuntimeError("could not determine decoder_start_token_id for generation")
        return int(decoder_start)

    def _drain_encoder(
        self,
        global_seq_ids: list[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if self._encoder_acc is None:
            raise RuntimeError("encoder accumulator is not initialized")
        for layer_id in range(NUM_SPARSE_LAYERS):
            if layer_id not in self._encoder_slot_buffer:
                raise RuntimeError(f"missing encoder router logits for sparse layer {layer_id}")
            tensor = self._encoder_slot_buffer[layer_id]
            if tensor.shape[:2] != input_ids.shape:
                raise RuntimeError(
                    f"encoder layer {layer_id}: router logits shape {tuple(tensor.shape)} "
                    f"does not match input ids shape {tuple(input_ids.shape)}"
                )

        for batch_idx, seq_id in enumerate(global_seq_ids):
            token_idx_in_seq = 0
            for slot_idx in range(input_ids.shape[1]):
                if int(attention_mask[batch_idx, slot_idx]) == 0:
                    continue
                layers = [
                    self._encoder_slot_buffer[layer_id][batch_idx, slot_idx]
                    for layer_id in range(NUM_SPARSE_LAYERS)
                ]
                self._encoder_acc.add_token(
                    seq_id=seq_id,
                    token_idx_in_seq=token_idx_in_seq,
                    token_id=int(input_ids[batch_idx, slot_idx]),
                    per_layer_logits=layers,
                )
                token_idx_in_seq += 1

    def _drain_decoder(
        self,
        global_seq_ids: list[int],
        gen_seqs: torch.Tensor,
        pad_id: Optional[int],
        eos_id: Optional[int],
        decoder_start: Optional[int],
    ) -> None:
        if self._decoder_acc is None:
            raise RuntimeError("decoder accumulator is not initialized")

        batch_size = len(global_seq_ids)
        per_layer_steps: list[list[torch.Tensor]] = []
        for layer_id in range(NUM_SPARSE_LAYERS):
            if layer_id not in self._decoder_step_buffer:
                raise RuntimeError(f"missing decoder router logits for sparse layer {layer_id}")

            steps: list[torch.Tensor] = []
            for tensor in self._decoder_step_buffer[layer_id]:
                if tensor.dim() != 3:
                    raise RuntimeError(
                        f"decoder layer {layer_id}: expected [B, S, V] router logits, "
                        f"got shape {tuple(tensor.shape)}"
                    )
                if tensor.shape[0] != batch_size:
                    raise RuntimeError(
                        f"decoder layer {layer_id}: router batch {tensor.shape[0]} "
                        f"!= prompt batch {batch_size}"
                    )
                for slot_idx in range(tensor.shape[1]):
                    steps.append(tensor[:, slot_idx, :])
            per_layer_steps.append(steps)

        if gen_seqs.shape[0] != batch_size:
            raise RuntimeError(
                f"generated sequence batch {gen_seqs.shape[0]} != prompt batch {batch_size}"
            )

        max_routed_steps = min(len(steps) for steps in per_layer_steps)
        max_sequence_steps = max(int(gen_seqs.shape[1]) - 1, 0)
        steps_to_drain = min(max_routed_steps, max_sequence_steps)
        alive = [True] * batch_size

        for step_idx in range(steps_to_drain):
            for batch_idx, seq_id in enumerate(global_seq_ids):
                if not alive[batch_idx]:
                    continue
                input_token = int(gen_seqs[batch_idx, step_idx])
                is_decoder_start = decoder_start is not None and input_token == decoder_start
                if pad_id is not None and input_token == pad_id and not (
                    step_idx == 0 and is_decoder_start
                ):
                    continue
                if step_idx > 0 and is_decoder_start:
                    continue
                layers = [
                    per_layer_steps[layer_id][step_idx][batch_idx]
                    for layer_id in range(NUM_SPARSE_LAYERS)
                ]
                self._decoder_acc.add_token(
                    seq_id=seq_id,
                    token_idx_in_seq=step_idx,
                    token_id=input_token,
                    per_layer_logits=layers,
                )

            for batch_idx in range(batch_size):
                next_token = int(gen_seqs[batch_idx, step_idx + 1])
                if (eos_id is not None and next_token == eos_id) or (
                    pad_id is not None and next_token == pad_id
                ):
                    alive[batch_idx] = False


class Verifier:
    @staticmethod
    def verify(trace_dir: Path) -> None:
        """Raise AssertionError on any contract violation."""
        d = Path(trace_dir)

        sel = torch.load(d / "expert_selection.pt")
        feat = torch.load(d / "decode_stage_moe_layer_logits_per_token.pt")
        gate = torch.load(d / "decode_stage_moe_layer_gate_logits_per_token.pt")
        freq = torch.load(d / "decode_stage_expert_freq_per_token.pt")
        tok = torch.load(d / "decode_stage_token_ids_per_token.pt")
        seq = torch.load(d / "decode_stage_seq_id_of_token.pt")
        idx = torch.load(d / "decode_stage_token_idx_in_seq.pt")

        assert sel.dtype == torch.int64, f"expert_selection dtype {sel.dtype} != int64"
        assert tok.dtype == seq.dtype == idx.dtype == torch.int64
        assert feat.dtype == gate.dtype == freq.dtype == torch.float32

        n = sel.shape[0]
        assert n > 0, "trace must be non-empty"

        assert feat.shape[0] == gate.shape[0] == freq.shape[0] == n
        assert tok.shape == (n,) and seq.shape == (n,) and idx.shape == (n,)

        assert gate.shape == feat.shape, f"gate shape {tuple(gate.shape)} != feat {tuple(feat.shape)}"
        assert freq.shape == gate.shape == feat.shape, (
            f"freq/gate/feat shape mismatch: freq {tuple(freq.shape)}, "
            f"gate {tuple(gate.shape)}, feat {tuple(feat.shape)}"
        )

        assert sel.shape == (n, NUM_SPARSE_LAYERS, PER_TOKEN_EXPERT), (
            f"expert_selection shape {tuple(sel.shape)} != ({n}, {NUM_SPARSE_LAYERS}, {PER_TOKEN_EXPERT})"
        )
        assert feat.shape == (n, NUM_SPARSE_LAYERS, EXPECTED_NUM_EXPERTS), (
            f"feat shape {tuple(feat.shape)} != ({n}, {NUM_SPARSE_LAYERS}, {EXPECTED_NUM_EXPERTS})"
        )
        assert freq.shape == (n, NUM_SPARSE_LAYERS, EXPECTED_NUM_EXPERTS), (
            f"freq shape {tuple(freq.shape)} != ({n}, {NUM_SPARSE_LAYERS}, {EXPECTED_NUM_EXPERTS})"
        )

        sel_min = int(sel.min())
        assert sel_min >= 0, f"expert id out of range: min {sel_min}, expected >= 0"
        assert int(sel.max()) + 1 == EXPECTED_NUM_EXPERTS, (
            f"expert_selection max+1 must equal {EXPECTED_NUM_EXPERTS} (downstream num_expert), "
            f"got {int(sel.max()) + 1}"
        )

        row_sums = freq.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            "freq must sum to ~1 along expert dim"
        )

        # token_idx_in_seq strictly +1 monotonic within each seq, in seq's appearance order
        for s_val in seq.unique().tolist():
            mask = seq == s_val
            sub = idx[mask]
            expected = torch.arange(sub.shape[0], dtype=torch.int64)
            assert torch.equal(sub, expected), (
                f"token_idx not monotonic within seq {s_val}: {sub.tolist()}"
            )
