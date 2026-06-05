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
SWITCH_NUM_ENCODER_MOE_LAYERS = NUM_SPARSE_LAYERS
SWITCH_NUM_DECODER_MOE_LAYERS = NUM_SPARSE_LAYERS
SWITCH_NUM_GLOBAL_MOE_LAYERS = SWITCH_NUM_ENCODER_MOE_LAYERS + SWITCH_NUM_DECODER_MOE_LAYERS


@dataclass
class TraceAccumulator:
    """Holds raw per-token router records for one stage."""
    num_layers: int = NUM_SPARSE_LAYERS
    num_experts: int = EXPECTED_NUM_EXPERTS
    num_feature_layers: int = NUM_SPARSE_LAYERS + 1

    def __post_init__(self) -> None:
        self._seq_ids: list[int] = []
        self._token_idx: list[int] = []
        self._token_ids: list[int] = []
        self._per_layer: list[list[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        self._per_feature_layer: list[list[torch.Tensor]] = [
            [] for _ in range(self.num_feature_layers)
        ]
        self._feature_dim: Optional[int] = None

    def add_token(
        self,
        seq_id: int,
        token_idx_in_seq: int,
        token_id: int,
        per_layer_logits: list[torch.Tensor],
        per_layer_features: list[torch.Tensor],
    ) -> None:
        if len(per_layer_logits) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} per-layer logits, got {len(per_layer_logits)}"
            )
        if len(per_layer_features) != self.num_feature_layers:
            raise ValueError(
                f"expected {self.num_feature_layers} per-layer features, "
                f"got {len(per_layer_features)}"
            )
        for i, t in enumerate(per_layer_logits):
            if t.dim() != 1:
                raise ValueError(f"layer {i}: expected 1-D logits, got shape {tuple(t.shape)}")
            if t.shape[0] != self.num_experts:
                raise ValueError(
                    f"layer {i}: expected expert dim {self.num_experts}, got {t.shape[0]}"
                )
        for i, t in enumerate(per_layer_features):
            if t.dim() != 1:
                raise ValueError(
                    f"feature layer {i}: expected 1-D hidden state, got shape {tuple(t.shape)}"
                )
            if self._feature_dim is None:
                self._feature_dim = int(t.shape[0])
            elif t.shape[0] != self._feature_dim:
                raise ValueError(
                    f"feature layer {i}: expected hidden dim {self._feature_dim}, "
                    f"got {t.shape[0]}"
                )
        self._seq_ids.append(int(seq_id))
        self._token_idx.append(int(token_idx_in_seq))
        self._token_ids.append(int(token_id))
        for i, t in enumerate(per_layer_logits):
            self._per_layer[i].append(t.detach().to(torch.float32).cpu())
        for i, t in enumerate(per_layer_features):
            self._per_feature_layer[i].append(t.detach().to(torch.float32).cpu())

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

    def stacked_features(self) -> torch.Tensor:
        if self.total_tokens() == 0:
            feature_dim = 0 if self._feature_dim is None else self._feature_dim
            return torch.zeros(
                (0, self.num_feature_layers, feature_dim), dtype=torch.float32
            )
        per_layer_stacked = [
            torch.stack(layer_list, dim=0) for layer_list in self._per_feature_layer
        ]
        return torch.stack(per_layer_stacked, dim=1)  # [N, L+1, H]


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
        feat = acc.stacked_features()  # [N, L+1, H] float32, aligns cpp moe_layer_logits
        freq = torch.softmax(gate.to(torch.float32), dim=-1)  # [N, L, V]
        sel = gate.argmax(dim=-1, keepdim=True).to(torch.int64)  # [N, L, 1]

        v = int(gate.shape[-1])

        trace_metas = None
        if self.stage == "decoder":
            encoder_layers = SWITCH_NUM_ENCODER_MOE_LAYERS
            decoder_layers = int(gate.shape[1])
            if decoder_layers != SWITCH_NUM_DECODER_MOE_LAYERS:
                raise ValueError(
                    f"decoder writer expected {SWITCH_NUM_DECODER_MOE_LAYERS} local sparse layers, "
                    f"got {decoder_layers}"
                )
            global_layers = encoder_layers + decoder_layers
            global_gate = torch.zeros((n, global_layers, v), dtype=gate.dtype)
            global_freq = torch.zeros((n, global_layers, v), dtype=freq.dtype)
            global_sel = torch.zeros((n, global_layers, PER_TOKEN_EXPERT), dtype=sel.dtype)
            global_feat = torch.zeros((n, global_layers + 1, int(feat.shape[-1])), dtype=feat.dtype)
            global_gate[:, encoder_layers:global_layers, :] = gate
            global_freq[:, encoder_layers:global_layers, :] = freq
            global_sel[:, encoder_layers:global_layers, :] = sel
            global_feat[:, encoder_layers:global_layers + 1, :] = feat
            gate = global_gate
            freq = global_freq
            sel = global_sel
            feat = global_feat
            trace_metas = {
                "schema_version": 2,
                "id_space": "global",
                "num_layer": global_layers,
                "num_encoder_moe_layer": encoder_layers,
                "num_decoder_moe_layer": decoder_layers,
                "predict_stage": "decoder",
                "decoder_global_layer_start": encoder_layers,
                "decoder_global_layer_stop": global_layers,
            }

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
            "num_feature_layer": int(feat.shape[1]),
            "feature_dim": int(feat.shape[-1]),
            "per_token_expert": PER_TOKEN_EXPERT,
            "N": n,
        }
        if trace_metas is not None:
            meta.update(trace_metas)
            meta["layout"] = "global-dense"
        if extra_metadata:
            meta.update(extra_metadata)
        (self.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        if trace_metas is not None:
            (self.out_dir / "trace_metas.json").write_text(json.dumps(trace_metas, indent=2))


class SwitchRunner:
    """Load Switch model, install hooks, run batched generate."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        seed: int = 42,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.verbose = verbose
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
        self._decoder_feature_step_buffer: dict[int, list[torch.Tensor]] = {}
        self._encoder_feature_slot_buffer: dict[int, torch.Tensor] = {}

    def _status(self, message: str) -> None:
        if self.verbose:
            print(f"[switch-trace-export] {message}", flush=True)

    def _print_batch_io(
        self,
        global_seq_ids: list[int],
        batch_prompts: list[str],
        generated_texts: list[str],
    ) -> None:
        if not self.verbose:
            return
        for seq_id, prompt_text, generated_text in zip(
            global_seq_ids, batch_prompts, generated_texts
        ):
            print(
                (
                    f"\n[switch-trace-export][seq_id={seq_id}] PROMPT:\n"
                    f"{prompt_text}\n"
                    "[switch-trace-export] OUTPUT:\n"
                    f"{generated_text}\n"
                    "[switch-trace-export] ---"
                ),
                flush=True,
            )

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

        self._status(f"loading tokenizer/model from {self.model_path} on {self.device}")
        self._seed_all()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.float32
        )
        self.model.eval()
        self.model.to(self.device)
        self._status("model loaded")

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
        self._status("sparse MLP hooks installed")

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _sparse_mlp_hook(self, module, inputs, output):
        router_logits = self._extract_router_logits(module, inputs, output)
        input_hidden = self._extract_input_hidden_states(module, inputs)
        output_hidden = self._extract_output_hidden_states(module, inputs, output)
        layer_id = module._promoe_sparse_layer_id
        if module._promoe_stage == "encoder":
            self._encoder_slot_buffer[layer_id] = router_logits.detach().cpu()
            if layer_id == 0:
                self._encoder_feature_slot_buffer[0] = input_hidden.detach().cpu()
            self._encoder_feature_slot_buffer[layer_id + 1] = output_hidden.detach().cpu()
        else:
            self._decoder_step_buffer.setdefault(layer_id, []).append(router_logits.detach().cpu())
            if layer_id == 0:
                self._decoder_feature_step_buffer.setdefault(0, []).append(input_hidden.detach().cpu())
            self._decoder_feature_step_buffer.setdefault(layer_id + 1, []).append(
                output_hidden.detach().cpu()
            )

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

    def _extract_input_hidden_states(self, module, inputs) -> torch.Tensor:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError(
                "Switch sparse MLP hook expected input hidden_states tensor in inputs[0]"
            )
        hidden_states = inputs[0]
        if hidden_states.dim() != 3:
            raise RuntimeError(
                f"{module._promoe_stage} sparse layer {module._promoe_sparse_layer_id}: "
                f"expected input hidden_states [B, S, H], got shape {tuple(hidden_states.shape)}"
            )
        return hidden_states

    def _extract_output_hidden_states(self, module, inputs, output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            hidden_states = output
        elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            hidden_states = output[0]
        else:
            raise RuntimeError(
                "Switch sparse MLP hook expected output hidden_states tensor in output[0]"
            )

        if hidden_states.dim() == 2:
            input_hidden = self._extract_input_hidden_states(module, inputs)
            B, S = input_hidden.shape[0], input_hidden.shape[1]
            if hidden_states.shape[0] != B * S:
                raise RuntimeError(
                    f"{module._promoe_stage} sparse layer {module._promoe_sparse_layer_id}: "
                    f"cannot reshape output hidden_states {tuple(hidden_states.shape)} "
                    f"to ({B}, {S}, H)"
                )
            hidden_states = hidden_states.view(B, S, -1)
        elif hidden_states.dim() != 3:
            raise RuntimeError(
                f"{module._promoe_stage} sparse layer {module._promoe_sparse_layer_id}: "
                f"expected output hidden_states dim 2 or 3, got shape {tuple(hidden_states.shape)}"
            )
        return hidden_states

    def run(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int,
        max_input_tokens: Optional[int] = None,
    ) -> tuple[TraceAccumulator, TraceAccumulator]:
        """Returns (encoder_acc, decoder_acc)."""
        self._status(
            f"run started: prompts={len(prompts)}, batch_size={batch_size}, "
            f"max_new_tokens={max_new_tokens}, max_input_tokens={max_input_tokens}"
        )
        if self.model is None:
            self._load_model()
        if not self._hook_handles:
            self._install_hooks()
        self._encoder_acc = TraceAccumulator()
        self._decoder_acc = TraceAccumulator()
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        try:
            for batch_idx, batch_start in enumerate(range(0, len(prompts), batch_size), start=1):
                batch_prompts = prompts[batch_start : batch_start + batch_size]
                global_seq_ids = list(range(batch_start, batch_start + len(batch_prompts)))
                self._status(
                    f"batch {batch_idx}/{total_batches}: prompts={len(batch_prompts)} "
                    f"(seq_id {global_seq_ids[0]}..{global_seq_ids[-1]})"
                )
                self._run_one_batch(
                    batch_prompts,
                    global_seq_ids,
                    max_new_tokens,
                    max_input_tokens,
                )
                self._status(f"batch {batch_idx}/{total_batches}: done")
        finally:
            self._remove_hooks()
            self._status("router hooks removed")

        self._status(
            f"run finished: N_enc={self._encoder_acc.total_tokens()}, N_dec={self._decoder_acc.total_tokens()}"
        )
        return self._encoder_acc, self._decoder_acc

    def _run_one_batch(
        self,
        batch_prompts: list[str],
        global_seq_ids: list[int],
        max_new_tokens: int,
        max_input_tokens: Optional[int],
    ) -> None:
        token_kwargs = {
            "padding": "longest",
            "return_tensors": "pt",
            "truncation": max_input_tokens is not None,
        }
        if max_input_tokens is not None:
            token_kwargs["max_length"] = int(max_input_tokens)
        enc_inputs = self.tokenizer(batch_prompts, **token_kwargs)
        input_ids = enc_inputs["input_ids"].to(self.device)
        attn_mask = enc_inputs["attention_mask"].to(self.device)
        self._status(
            f"tokenized batch: input_ids shape={tuple(input_ids.shape)}, attn_mask shape={tuple(attn_mask.shape)}"
        )

        self._encoder_slot_buffer = {}
        self._decoder_step_buffer = {}
        self._encoder_feature_slot_buffer = {}
        self._decoder_feature_step_buffer = {}
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
        self._status(f"generation done: sequences shape={tuple(gen_seqs.shape)}")
        generated_texts = self.tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)
        self._print_batch_io(global_seq_ids, batch_prompts, generated_texts)
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
            if layer_id not in self._encoder_feature_slot_buffer:
                raise RuntimeError(f"missing encoder hidden features for sparse layer {layer_id}")
            feature_tensor = self._encoder_feature_slot_buffer[layer_id]
            if feature_tensor.shape[:2] != input_ids.shape:
                raise RuntimeError(
                    f"encoder feature layer {layer_id}: hidden shape {tuple(feature_tensor.shape)} "
                    f"does not match input ids shape {tuple(input_ids.shape)}"
                )

        if NUM_SPARSE_LAYERS not in self._encoder_feature_slot_buffer:
            raise RuntimeError("missing encoder final hidden features")
        final_feature = self._encoder_feature_slot_buffer[NUM_SPARSE_LAYERS]
        if final_feature.shape[:2] != input_ids.shape:
            raise RuntimeError(
                f"encoder final hidden shape {tuple(final_feature.shape)} "
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
                features = [
                    self._encoder_feature_slot_buffer[layer_id][batch_idx, slot_idx]
                    for layer_id in range(NUM_SPARSE_LAYERS + 1)
                ]
                self._encoder_acc.add_token(
                    seq_id=seq_id,
                    token_idx_in_seq=token_idx_in_seq,
                    token_id=int(input_ids[batch_idx, slot_idx]),
                    per_layer_logits=layers,
                    per_layer_features=features,
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

        per_feature_steps: list[list[torch.Tensor]] = []
        for feature_layer_id in range(NUM_SPARSE_LAYERS + 1):
            if feature_layer_id not in self._decoder_feature_step_buffer:
                raise RuntimeError(
                    f"missing decoder hidden features for feature layer {feature_layer_id}"
                )

            steps: list[torch.Tensor] = []
            for tensor in self._decoder_feature_step_buffer[feature_layer_id]:
                if tensor.dim() != 3:
                    raise RuntimeError(
                        f"decoder feature layer {feature_layer_id}: expected [B, S, H] hidden, "
                        f"got shape {tuple(tensor.shape)}"
                    )
                if tensor.shape[0] != batch_size:
                    raise RuntimeError(
                        f"decoder feature layer {feature_layer_id}: hidden batch {tensor.shape[0]} "
                        f"!= prompt batch {batch_size}"
                    )
                for slot_idx in range(tensor.shape[1]):
                    steps.append(tensor[:, slot_idx, :])
            per_feature_steps.append(steps)

        if gen_seqs.shape[0] != batch_size:
            raise RuntimeError(
                f"generated sequence batch {gen_seqs.shape[0]} != prompt batch {batch_size}"
            )

        step_counts = [len(steps) for steps in per_layer_steps]
        if len(set(step_counts)) != 1:
            raise RuntimeError(f"decoder sparse layer step counts differ: {step_counts}")
        feature_step_counts = [len(steps) for steps in per_feature_steps]
        if len(set(feature_step_counts)) != 1:
            raise RuntimeError(
                f"decoder hidden feature layer step counts differ: {feature_step_counts}"
            )
        if feature_step_counts[0] != step_counts[0]:
            raise RuntimeError(
                f"decoder hidden feature steps {feature_step_counts[0]} "
                f"!= router steps {step_counts[0]}"
            )

        routed_steps = step_counts[0]
        max_sequence_steps = max(int(gen_seqs.shape[1]) - 1, 0)
        steps_to_drain = min(routed_steps, max_sequence_steps)
        alive = [True] * batch_size

        for step_idx in range(steps_to_drain):
            for batch_idx, seq_id in enumerate(global_seq_ids):
                if not alive[batch_idx]:
                    continue
                # Router step k predicts generated token sequences[:, k + 1].
                # Record EOS when generated, then suppress all later padded steps.
                generated_token = int(gen_seqs[batch_idx, step_idx + 1])
                if pad_id is not None and generated_token == pad_id:
                    continue
                layers = [
                    per_layer_steps[layer_id][step_idx][batch_idx]
                    for layer_id in range(NUM_SPARSE_LAYERS)
                ]
                features = [
                    per_feature_steps[feature_layer_id][step_idx][batch_idx]
                    for feature_layer_id in range(NUM_SPARSE_LAYERS + 1)
                ]
                self._decoder_acc.add_token(
                    seq_id=seq_id,
                    token_idx_in_seq=step_idx,
                    token_id=generated_token,
                    per_layer_logits=layers,
                    per_layer_features=features,
                )

            for batch_idx in range(batch_size):
                generated_token = int(gen_seqs[batch_idx, step_idx + 1])
                if (eos_id is not None and generated_token == eos_id) or (
                    pad_id is not None and generated_token == pad_id
                ):
                    alive[batch_idx] = False


class Verifier:
    @staticmethod
    def verify(trace_dir: Path) -> None:
        """Raise AssertionError on any contract violation."""
        d = Path(trace_dir)

        sel = torch.load(d / "expert_selection.pt", weights_only=True)
        feat = torch.load(d / "decode_stage_moe_layer_logits_per_token.pt", weights_only=True)
        gate = torch.load(d / "decode_stage_moe_layer_gate_logits_per_token.pt", weights_only=True)
        freq = torch.load(d / "decode_stage_expert_freq_per_token.pt", weights_only=True)
        tok = torch.load(d / "decode_stage_token_ids_per_token.pt", weights_only=True)
        seq = torch.load(d / "decode_stage_seq_id_of_token.pt", weights_only=True)
        idx = torch.load(d / "decode_stage_token_idx_in_seq.pt", weights_only=True)

        trace_metas = {}
        trace_metas_path = d / "trace_metas.json"
        if trace_metas_path.exists():
            trace_metas = json.loads(trace_metas_path.read_text())
        is_global_decoder = (
            trace_metas.get("id_space") == "global"
            and trace_metas.get("predict_stage", "decoder") == "decoder"
        )

        assert sel.dtype == torch.int64, f"expert_selection dtype {sel.dtype} != int64"
        assert tok.dtype == seq.dtype == idx.dtype == torch.int64
        assert feat.dtype == gate.dtype == freq.dtype == torch.float32

        n = sel.shape[0]
        assert n > 0, "trace must be non-empty"

        assert feat.shape[0] == gate.shape[0] == freq.shape[0] == n
        assert tok.shape == (n,) and seq.shape == (n,) and idx.shape == (n,)

        assert feat.dim() == 3, f"feat must be [N, F, H], got {tuple(feat.shape)}"
        assert feat.shape[2] > 0, f"feat hidden dim must be non-empty, got {tuple(feat.shape)}"
        assert freq.shape == gate.shape, (
            f"freq/gate shape mismatch: freq {tuple(freq.shape)}, gate {tuple(gate.shape)}"
        )

        if is_global_decoder:
            assert int(trace_metas.get("schema_version", 0)) == 2, "global decoder trace requires schema_version=2"
            expected_layers = int(trace_metas["num_layer"])
            encoder_layers = int(trace_metas["num_encoder_moe_layer"])
            decoder_layers = int(trace_metas["num_decoder_moe_layer"])
            decoder_start = int(trace_metas["decoder_global_layer_start"])
            decoder_stop = int(trace_metas["decoder_global_layer_stop"])
            assert encoder_layers + decoder_layers == expected_layers, (
                "global decoder metadata layer counts must satisfy encoder + decoder == total"
            )
            assert decoder_start == encoder_layers and decoder_stop == expected_layers, (
                "decoder global range must match encoder boundary and total layer count"
            )
            assert sel.shape == (n, expected_layers, PER_TOKEN_EXPERT), (
                f"expert_selection shape {tuple(sel.shape)} != ({n}, {expected_layers}, {PER_TOKEN_EXPERT})"
            )
            assert freq.shape == (n, expected_layers, EXPECTED_NUM_EXPERTS), (
                f"freq shape {tuple(freq.shape)} != ({n}, {expected_layers}, {EXPECTED_NUM_EXPERTS})"
            )
            assert feat.shape[1] == expected_layers + 1, (
                f"feat layer count {feat.shape[1]} must be {expected_layers + 1}"
            )
            assert torch.count_nonzero(sel[:, :encoder_layers, :]) == 0, "encoder expert slots must be zero"
            assert torch.count_nonzero(gate[:, :encoder_layers, :]) == 0, "encoder gate slots must be zero"
            assert torch.count_nonzero(freq[:, :encoder_layers, :]) == 0, "encoder freq slots must be zero"
            assert torch.count_nonzero(feat[:, :encoder_layers, :]) == 0, "encoder hidden slots must be zero"
            freq_for_sum = freq[:, decoder_start:decoder_stop, :]
        else:
            assert feat.shape[1] in (NUM_SPARSE_LAYERS, NUM_SPARSE_LAYERS + 1), (
                f"feat layer count {feat.shape[1]} must be {NUM_SPARSE_LAYERS} "
                f"or {NUM_SPARSE_LAYERS + 1}"
            )
            assert sel.shape == (n, NUM_SPARSE_LAYERS, PER_TOKEN_EXPERT), (
                f"expert_selection shape {tuple(sel.shape)} != ({n}, {NUM_SPARSE_LAYERS}, {PER_TOKEN_EXPERT})"
            )
            assert freq.shape == (n, NUM_SPARSE_LAYERS, EXPECTED_NUM_EXPERTS), (
                f"freq shape {tuple(freq.shape)} != ({n}, {NUM_SPARSE_LAYERS}, {EXPECTED_NUM_EXPERTS})"
            )
            freq_for_sum = freq

        sel_min = int(sel.min())
        sel_max = int(sel.max())
        assert sel_min >= 0, f"expert id out of range: min {sel_min}, expected >= 0"
        assert sel_max < gate.shape[-1], (
            f"expert id out of range: max {sel_max}, expected < {gate.shape[-1]}"
        )

        row_sums = freq_for_sum.sum(dim=-1)
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
