from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_ROUTER_PATTERN = re.compile(r".*(encoder|decoder)\.block\.(\d+)\.")


def _router_stage_block(router_key: str) -> tuple[str, int] | None:
  match = _ROUTER_PATTERN.match(str(router_key))
  if not match:
    return None
  return match.group(1), int(match.group(2))


def _global_layer_id_or_none(adapter, stage: str, block_id: int) -> int | None:
  try:
    stage_layer = adapter.stage_layer_id(stage, block_id)
  except ValueError:
    return None
  return adapter.global_layer_id(stage, stage_layer)


def _dedupe_valid_eids(raw_eids: Iterable[Any], num_experts: int) -> list[int]:
  seen = set()
  result = []
  for raw in raw_eids:
    eid = int(raw)
    if eid < 0 or eid >= int(num_experts) or eid in seen:
      continue
    seen.add(eid)
    result.append(eid)
  return result


def _pairs_from_usage_info(info: dict[str, Any], num_experts: int) -> list[tuple[int, int]]:
  raw_counts = info.get("top_token_counts")
  if isinstance(raw_counts, list) and raw_counts:
    pairs = []
    for rank, item in enumerate(raw_counts):
      if isinstance(item, dict):
        if "eid" not in item:
          continue
        eid = int(item["eid"])
        count = int(item.get("count", 0) or 0)
      else:
        top_eids = info.get("top_token_eids") or info.get("frozen_top_eids") or []
        if rank >= len(top_eids):
          continue
        eid = int(top_eids[rank])
        count = int(item or 0)
      if 0 <= eid < int(num_experts):
        pairs.append((eid, max(0, count)))
    if pairs:
      seen = set()
      deduped = []
      for eid, count in pairs:
        if eid in seen:
          continue
        seen.add(eid)
        deduped.append((eid, count))
      return deduped

  fallback = info.get("frozen_top_eids") or info.get("top_token_eids") or info.get("top_eids")
  if isinstance(fallback, list):
    return [(eid, 0) for eid in _dedupe_valid_eids(fallback, num_experts)]
  return []


def _hot_pairs_by_layer(
    payload: dict[str, Any],
    adapter,
    stage: str,
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, int]]:
  num_experts = int(adapter.num_expert_per_layer)
  by_layer: dict[int, list[tuple[int, int]]] = {}
  token_totals: dict[int, int] = {}

  frozen = payload.get("frozen_hot_experts") or payload.get("hot_experts")
  if isinstance(frozen, dict):
    for router_key, eids in frozen.items():
      parsed = _router_stage_block(router_key)
      if parsed is None or parsed[0] != stage or not isinstance(eids, list):
        continue
      global_layer = _global_layer_id_or_none(adapter, stage, parsed[1])
      if global_layer is None:
        continue
      pairs = [(eid, 0) for eid in _dedupe_valid_eids(eids, num_experts)]
      if pairs:
        by_layer[global_layer] = pairs

  summary = payload.get("expert_usage_summary")
  if isinstance(summary, dict):
    for router_key, info in summary.items():
      parsed = _router_stage_block(router_key)
      if parsed is None or parsed[0] != stage or not isinstance(info, dict):
        continue
      global_layer = _global_layer_id_or_none(adapter, stage, parsed[1])
      if global_layer is None:
        continue
      pairs = _pairs_from_usage_info(info, num_experts)
      if pairs:
        by_layer[global_layer] = pairs
        token_totals[global_layer] = int(info.get("token_total_hits") or 0)

  return by_layer, token_totals


def _encoder_hot_pairs_by_layer(payload: dict[str, Any], adapter) -> dict[int, list[tuple[int, int]]]:
  by_layer, _token_totals = _hot_pairs_by_layer(payload, adapter, "encoder")
  return by_layer


def _read_hot_expert_payload(hot_expert_file: str | Path) -> dict[str, Any]:
  with Path(hot_expert_file).expanduser().open("r", encoding="utf-8") as f:
    payload = json.load(f)
  if not isinstance(payload, dict):
    raise ValueError(f"hot expert snapshot must be a JSON object: {hot_expert_file}")
  return payload


def _encoder_k_coverage_plan(
    by_layer: dict[int, list[tuple[int, int]]],
    token_totals: dict[int, int],
    encoder_coverage: float,
) -> list[tuple[int, int]]:
  plan: list[tuple[int, int]] = []
  for layer_idx in sorted(by_layer):
    pairs = by_layer[layer_idx]
    token_total = int(token_totals.get(layer_idx, 0))
    if token_total <= 0:
      token_total = sum(count for _eid, count in pairs)
    covered = 0
    for eid, count in pairs:
      plan.append((layer_idx, eid))
      covered += int(count)
      if token_total <= 0 or covered / token_total >= encoder_coverage:
        break
  return plan


def _decoder_even_plan(
    by_layer: dict[int, list[tuple[int, int]]],
    adapter,
    total_slots: int,
) -> list[tuple[int, int]]:
  num_decoder_layers = int(adapter.num_decoder_sparse_layers)
  if total_slots <= 0 or num_decoder_layers <= 0:
    return []
  decoder_capacity = num_decoder_layers * int(adapter.num_expert_per_layer)
  if int(total_slots) > decoder_capacity:
    raise ValueError(
      f"decoder capacity {decoder_capacity} is smaller than remaining slots {int(total_slots)}"
    )

  base_quota, extra = divmod(int(total_slots), num_decoder_layers)
  by_decoder_layer: dict[int, list[tuple[int, int]]] = {}
  for stage_layer in range(num_decoder_layers):
    global_layer = adapter.global_layer_id("decoder", stage_layer)
    quota = base_quota + (1 if stage_layer < extra else 0)
    selected: list[tuple[int, int]] = []
    seen = set()
    for eid, _count in by_layer.get(global_layer, []):
      if len(selected) >= quota:
        break
      item = (global_layer, eid)
      if item in seen:
        continue
      selected.append(item)
      seen.add(item)
    for eid in range(int(adapter.num_expert_per_layer)):
      if len(selected) >= quota:
        break
      item = (global_layer, eid)
      if item in seen:
        continue
      selected.append(item)
      seen.add(item)
    by_decoder_layer[global_layer] = selected

  plan: list[tuple[int, int]] = []
  for layer_idx in sorted(by_decoder_layer, reverse=True):
    plan.extend(by_decoder_layer[layer_idx])
  return plan


def _append_fallback_experts(
    plan: list[tuple[int, int]],
    *,
    total_slots: int,
    adapter,
) -> None:
  seen = set(plan)
  for layer_idx in range(int(adapter.num_encoder_sparse_layers)):
    global_layer = adapter.global_layer_id("encoder", layer_idx)
    for eid in range(int(adapter.num_expert_per_layer)):
      item = (global_layer, eid)
      if item in seen:
        continue
      plan.append(item)
      seen.add(item)
      if len(plan) >= total_slots:
        return

  for layer_idx in range(int(adapter.num_decoder_sparse_layers)):
    global_layer = adapter.global_layer_id("decoder", layer_idx)
    for eid in range(int(adapter.num_expert_per_layer)):
      item = (global_layer, eid)
      if item in seen:
        continue
      plan.append(item)
      seen.add(item)
      if len(plan) >= total_slots:
        return


def _order_initial_plan_for_encoder_lru(plan: list[tuple[int, int]], adapter) -> list[tuple[int, int]]:
  indexed = [(idx, layer_idx, expert_idx) for idx, (layer_idx, expert_idx) in enumerate(plan)]
  first_decoder_layer = int(adapter.num_encoder_sparse_layers)

  def sort_key(item: tuple[int, int, int]) -> tuple[int, int, int]:
    original_idx, layer_idx, _expert_idx = item
    if int(layer_idx) < first_decoder_layer:
      return (1, -int(layer_idx), original_idx)
    return (0, int(layer_idx), original_idx)

  ordered = sorted(indexed, key=sort_key)
  return [(layer_idx, expert_idx) for _idx, layer_idx, expert_idx in ordered]


def build_encoder_hot_initial_plan(
    hot_expert_file: str | Path,
    adapter,
    *,
    total_slots: int,
    allow_sequential_fallback: bool = False,
) -> list[tuple[int, int]]:
  """Build an exact-size initial cache plan from encoder hot expert stats."""
  total_slots = max(0, int(total_slots))
  if total_slots <= 0:
    return []

  payload = _read_hot_expert_payload(hot_expert_file)
  by_layer = _encoder_hot_pairs_by_layer(payload, adapter)
  if not by_layer:
    raise ValueError(f"hot expert snapshot has no encoder expert entries: {hot_expert_file}")
  plan: list[tuple[int, int]] = []
  quota = {layer_idx: 0 for layer_idx in by_layer}
  remaining = total_slots

  if remaining >= len(by_layer):
    for layer_idx in sorted(by_layer):
      eid, _count = by_layer[layer_idx][0]
      plan.append((layer_idx, eid))
      quota[layer_idx] = 1
      remaining -= 1

  candidates: list[tuple[int, int, int, int]] = []
  for layer_idx, pairs in by_layer.items():
    start_rank = int(quota.get(layer_idx, 0))
    for rank, (eid, count) in enumerate(pairs[start_rank:], start=start_rank):
      candidates.append((int(count), int(layer_idx), int(rank), int(eid)))
  candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

  seen = set(plan)
  for _count, layer_idx, _rank, eid in candidates:
    if remaining <= 0:
      break
    item = (layer_idx, eid)
    if item in seen:
      continue
    plan.append(item)
    seen.add(item)
    remaining -= 1

  if len(plan) < total_slots and allow_sequential_fallback:
    _append_fallback_experts(plan, total_slots=total_slots, adapter=adapter)

  if len(plan) != total_slots:
    raise ValueError(
      f"initial hot expert plan has {len(plan)} hot entries, expected {total_slots}; "
      "reduce cache_rate or pass allow_sequential_fallback=True"
    )
  return _order_initial_plan_for_encoder_lru(plan, adapter)


def build_hot_initial_plan(
    hot_expert_file: str | Path,
    adapter,
    *,
    total_slots: int,
    encoder_coverage: float = 0.90,
) -> list[tuple[int, int]]:
  """Build an initial cache plan from encoder k-coverage and decoder hot stats."""
  total_slots = max(0, int(total_slots))
  if total_slots <= 0:
    return []

  payload = _read_hot_expert_payload(hot_expert_file)
  encoder_by_layer, encoder_token_totals = _hot_pairs_by_layer(payload, adapter, "encoder")
  if not encoder_by_layer:
    raise ValueError(f"hot expert snapshot has no encoder expert entries: {hot_expert_file}")

  encoder_plan = _encoder_k_coverage_plan(
    encoder_by_layer,
    encoder_token_totals,
    float(encoder_coverage),
  )
  if len(encoder_plan) > total_slots:
    raise ValueError(
      f"encoder k90 requires {len(encoder_plan)} slots, but total_slots={total_slots}"
    )

  decoder_by_layer, _decoder_token_totals = _hot_pairs_by_layer(payload, adapter, "decoder")
  decoder_plan = _decoder_even_plan(
    decoder_by_layer,
    adapter,
    total_slots=total_slots - len(encoder_plan),
  )
  plan = decoder_plan + sorted(encoder_plan, key=lambda item: -int(item[0]))
  if len(plan) != total_slots:
    raise ValueError(
      f"initial hot expert plan has {len(plan)} entries, expected {total_slots}"
    )
  return plan


def build_decoder_warmup_overlap_plan(
    hot_expert_file: str | Path,
    adapter,
    *,
    initial_plan: Iterable[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
  payload = _read_hot_expert_payload(hot_expert_file)
  decoder_by_layer, _decoder_token_totals = _hot_pairs_by_layer(payload, adapter, "decoder")
  initial_seen = {
    (int(layer_idx), int(expert_idx))
    for layer_idx, expert_idx in (initial_plan or [])
  }

  by_stage_layer: list[list[tuple[int, int]]] = []
  max_depth = 0
  for stage_layer in range(int(adapter.num_decoder_sparse_layers)):
    global_layer = adapter.global_layer_id("decoder", stage_layer)
    layer_plan: list[tuple[int, int]] = []
    seen_eids = set()
    for eid, _count in decoder_by_layer.get(global_layer, []):
      item = (global_layer, int(eid))
      if item in initial_seen or int(eid) in seen_eids:
        continue
      layer_plan.append(item)
      seen_eids.add(int(eid))
    by_stage_layer.append(layer_plan)
    max_depth = max(max_depth, len(layer_plan))

  plan: list[tuple[int, int]] = []
  for depth in range(max_depth):
    for layer_plan in by_stage_layer:
      if depth < len(layer_plan):
        plan.append(layer_plan[depth])
  return plan


def format_initial_expert_plan(plan: Iterable[tuple[int, int]]) -> str:
  return ",".join(f"{int(layer_idx)}:{int(expert_idx)}" for layer_idx, expert_idx in plan)
