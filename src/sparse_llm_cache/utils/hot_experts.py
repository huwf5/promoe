from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, NamedTuple


_SWITCH_ROUTER_PATTERN = re.compile(r".*(encoder|decoder)\.block\.(\d+)\.")
_LAYERS_ROUTER_PATTERN = re.compile(r".*(encoder|decoder)\.layers\.(\d+)\.")


class EncoderCoverageInitialPlan(NamedTuple):
  plan: list[tuple[int, int]]
  coverage: float
  coverage_slots: int


def _router_stage_block(router_key: str) -> tuple[str, int] | None:
  text = str(router_key)
  for pattern in (_SWITCH_ROUTER_PATTERN, _LAYERS_ROUTER_PATTERN):
    match = pattern.match(text)
    if match:
      return match.group(1), int(match.group(2))
  return None


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


def _layer_prefix_for_coverage(
    pairs: list[tuple[int, int]],
    token_total: int,
    coverage: float,
) -> list[tuple[int, int]]:
  if coverage <= 0.0:
    return []
  if token_total <= 0:
    token_total = sum(count for _eid, count in pairs)
  if token_total <= 0:
    return pairs[:1] if pairs else []
  selected: list[tuple[int, int]] = []
  covered = 0
  for eid, count in pairs:
    selected.append((eid, count))
    covered += int(count)
    if covered / token_total >= coverage:
      break
  return selected


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


def build_encoder_coverage_initial_plan(
    hot_expert_file: str | Path,
    adapter,
    *,
    total_slots: int,
    coverage_step: float = 0.01,
    allow_sequential_fallback: bool = False,
) -> EncoderCoverageInitialPlan:
  total_slots = max(0, int(total_slots))
  coverage_step = float(coverage_step)
  if coverage_step <= 0.0 or coverage_step > 1.0:
    raise ValueError(f"coverage_step must be in (0, 1], got {coverage_step}")

  payload = _read_hot_expert_payload(hot_expert_file)
  by_layer, token_totals = _hot_pairs_by_layer(payload, adapter, "encoder")
  if not by_layer:
    raise ValueError(f"hot expert snapshot has no encoder expert entries: {hot_expert_file}")

  if total_slots <= 0:
    return EncoderCoverageInitialPlan([], 0.0, 0)

  encoder_capacity = int(adapter.num_encoder_sparse_layers) * int(adapter.num_expert_per_layer)
  if total_slots > encoder_capacity:
    raise ValueError(
      f"encoder capacity {encoder_capacity} is smaller than initial slots {total_slots}"
    )

  best_coverage = 0.0
  best_plan: list[tuple[int, int]] = []
  steps = int(1.0 / coverage_step)
  coverages = [round(i * coverage_step, 10) for i in range(steps + 1)]
  if coverages[-1] < 1.0:
    coverages.append(1.0)

  encoder_layers = [
    adapter.global_layer_id("encoder", stage_layer)
    for stage_layer in range(int(adapter.num_encoder_sparse_layers))
  ]

  for coverage in coverages:
    candidate: list[tuple[int, int]] = []
    coverage_possible = True
    for layer_idx in encoder_layers:
      prefix = _layer_prefix_for_coverage(
        by_layer.get(layer_idx, []),
        int(token_totals.get(layer_idx, 0)),
        coverage,
      )
      if coverage > 0.0 and not prefix:
        coverage_possible = False
        break
      candidate.extend((layer_idx, eid) for eid, _count in prefix)
    if not coverage_possible:
      break
    if len(candidate) <= total_slots:
      best_coverage = coverage
      best_plan = candidate
    else:
      break

  plan = list(best_plan)
  seen = set(plan)
  candidates: list[tuple[int, int, int, int]] = []
  for layer_idx, pairs in by_layer.items():
    for rank, (eid, count) in enumerate(pairs):
      if (layer_idx, eid) in seen:
        continue
      candidates.append((int(count), int(layer_idx), int(rank), int(eid)))
  candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

  for _count, layer_idx, _rank, eid in candidates:
    if len(plan) >= total_slots:
      break
    item = (layer_idx, eid)
    if item in seen:
      continue
    plan.append(item)
    seen.add(item)

  if len(plan) < total_slots and allow_sequential_fallback:
    for stage_layer in range(int(adapter.num_encoder_sparse_layers)):
      layer_idx = adapter.global_layer_id("encoder", stage_layer)
      for eid in range(int(adapter.num_expert_per_layer)):
        if len(plan) >= total_slots:
          break
        item = (layer_idx, eid)
        if item in seen:
          continue
        plan.append(item)
        seen.add(item)

  if len(plan) != total_slots:
    raise ValueError(
      f"encoder coverage initial plan has {len(plan)} entries, expected {total_slots}; "
      "reduce cache_rate or pass allow_sequential_fallback=True"
    )
  return EncoderCoverageInitialPlan(
    _order_initial_plan_for_encoder_lru(plan, adapter),
    best_coverage,
    len(best_plan),
  )




def _select_hot_prefix_with_fallback(
    *,
    layer_idx: int,
    pairs: list[tuple[int, int]],
    quota: int,
    num_experts: int,
    allow_sequential_fallback: bool,
) -> list[tuple[int, int]]:
  selected: list[tuple[int, int]] = []
  seen_eids = set()
  for eid, _count in pairs:
    if len(selected) >= quota:
      break
    eid = int(eid)
    if eid in seen_eids:
      continue
    selected.append((int(layer_idx), eid))
    seen_eids.add(eid)
  if len(selected) < quota and allow_sequential_fallback:
    for eid in range(int(num_experts)):
      if len(selected) >= quota:
        break
      if eid in seen_eids:
        continue
      selected.append((int(layer_idx), eid))
      seen_eids.add(eid)
  return selected


def build_encoder_balanced_hot_initial_plan(
    hot_expert_file: str | Path,
    adapter,
    *,
    total_slots: int,
    allow_sequential_fallback: bool = False,
) -> list[tuple[int, int]]:
  """Build an encoder-first plan with balanced encoder quotas and decoder spillover.

  This policy spreads the initial cache budget across encoder layers first, then
  uses the hot ranking inside each layer. If the requested budget exceeds encoder
  capacity, the remaining slots are split evenly across decoder layers.
  """
  total_slots = max(0, int(total_slots))
  if total_slots <= 0:
    return []

  payload = _read_hot_expert_payload(hot_expert_file)
  by_layer = _encoder_hot_pairs_by_layer(payload, adapter)
  if not by_layer:
    raise ValueError(f"hot expert snapshot has no encoder expert entries: {hot_expert_file}")

  encoder_layers = [
    adapter.global_layer_id("encoder", stage_layer)
    for stage_layer in range(int(adapter.num_encoder_sparse_layers))
  ]
  num_encoder_layers = len(encoder_layers)
  if num_encoder_layers <= 0:
    raise ValueError("adapter has no encoder sparse layers")
  encoder_capacity = num_encoder_layers * int(adapter.num_expert_per_layer)
  encoder_slots = min(total_slots, encoder_capacity)
  decoder_slots = total_slots - encoder_slots

  base_quota, extra_slots = divmod(encoder_slots, num_encoder_layers)
  plan: list[tuple[int, int]] = []
  seen = set()
  selected_per_layer: dict[int, int] = {}

  for layer_idx in encoder_layers:
    selected = _select_hot_prefix_with_fallback(
      layer_idx=layer_idx,
      pairs=by_layer.get(layer_idx, []),
      quota=base_quota,
      num_experts=int(adapter.num_expert_per_layer),
      allow_sequential_fallback=allow_sequential_fallback,
    )
    if len(selected) != base_quota:
      raise ValueError(
        f"balanced hot initial plan layer {layer_idx} has {len(selected)} entries, "
        f"expected base quota {base_quota}; pass allow_sequential_fallback=True"
      )
    for item in selected:
      plan.append(item)
      seen.add(item)
    selected_per_layer[layer_idx] = len(selected)

  candidates: list[tuple[int, int, int, int]] = []
  for layer_idx in encoder_layers:
    pairs = by_layer.get(layer_idx, [])
    for rank, (eid, count) in enumerate(pairs):
      item = (layer_idx, int(eid))
      if item in seen:
        continue
      candidates.append((int(count), int(layer_idx), int(rank), int(eid)))
  candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

  for _count, layer_idx, _rank, eid in candidates:
    if extra_slots <= 0:
      break
    item = (layer_idx, eid)
    if item in seen:
      continue
    plan.append(item)
    seen.add(item)
    selected_per_layer[layer_idx] = selected_per_layer.get(layer_idx, 0) + 1
    extra_slots -= 1

  if extra_slots > 0 and allow_sequential_fallback:
    for layer_idx in encoder_layers:
      for eid in range(int(adapter.num_expert_per_layer)):
        if extra_slots <= 0:
          break
        item = (layer_idx, eid)
        if item in seen:
          continue
        plan.append(item)
        seen.add(item)
        selected_per_layer[layer_idx] = selected_per_layer.get(layer_idx, 0) + 1
        extra_slots -= 1
      if extra_slots <= 0:
        break

  if len(plan) != encoder_slots:
    raise ValueError(
      f"balanced hot initial plan has {len(plan)} encoder entries, expected {encoder_slots}; "
      "reduce cache_rate or pass allow_sequential_fallback=True"
    )

  if decoder_slots > 0:
    decoder_by_layer, _decoder_token_totals = _hot_pairs_by_layer(payload, adapter, "decoder")
    plan.extend(
      _decoder_even_plan(
        decoder_by_layer,
        adapter,
        total_slots=decoder_slots,
      )
    )

  if len(plan) != total_slots:
    raise ValueError(
      f"balanced hot initial plan has {len(plan)} entries, expected {total_slots}"
    )
  return _order_initial_plan_for_encoder_lru(plan, adapter)

def _encoder_l0_priority_quotas(
    *,
    total_slots: int,
    num_encoder_layers: int,
    num_experts: int,
    l0_fraction: float,
) -> list[int]:
  total_slots = max(0, int(total_slots))
  num_encoder_layers = int(num_encoder_layers)
  num_experts = int(num_experts)
  if num_encoder_layers <= 0:
    raise ValueError("adapter has no encoder sparse layers")
  if total_slots > num_encoder_layers * num_experts:
    raise ValueError(
      f"encoder capacity {num_encoder_layers * num_experts} is smaller than initial slots {total_slots}"
    )
  quotas = [0 for _ in range(num_encoder_layers)]
  l0_quota = min(total_slots, num_experts, int(num_experts * float(l0_fraction)))
  quotas[0] = l0_quota

  remaining = total_slots - l0_quota
  if remaining > 0 and num_encoder_layers > 1:
    base_quota, extra = divmod(remaining, num_encoder_layers - 1)
    for stage_layer in range(1, num_encoder_layers):
      quotas[stage_layer] = min(num_experts, base_quota + (1 if stage_layer - 1 < extra else 0))

  while sum(quotas) < total_slots:
    progressed = False
    for stage_layer in range(num_encoder_layers):
      if sum(quotas) >= total_slots:
        break
      if quotas[stage_layer] >= num_experts:
        continue
      quotas[stage_layer] += 1
      progressed = True
    if not progressed:
      break

  if sum(quotas) != total_slots:
    raise ValueError(
      f"L0 priority quotas have {sum(quotas)} slots, expected {total_slots}"
    )
  return quotas


def build_encoder_l0_priority_hot_initial_plan(
    hot_expert_file: str | Path,
    adapter,
    *,
    total_slots: int,
    l0_fraction: float = 0.75,
    allow_sequential_fallback: bool = False,
) -> list[tuple[int, int]]:
  """Build an encoder-only hot plan that gives L0 a larger fixed quota first."""
  total_slots = max(0, int(total_slots))
  if total_slots <= 0:
    return []

  payload = _read_hot_expert_payload(hot_expert_file)
  by_layer = _encoder_hot_pairs_by_layer(payload, adapter)
  if not by_layer:
    raise ValueError(f"hot expert snapshot has no encoder expert entries: {hot_expert_file}")

  encoder_layers = [
    adapter.global_layer_id("encoder", stage_layer)
    for stage_layer in range(int(adapter.num_encoder_sparse_layers))
  ]
  quotas = _encoder_l0_priority_quotas(
    total_slots=total_slots,
    num_encoder_layers=len(encoder_layers),
    num_experts=int(adapter.num_expert_per_layer),
    l0_fraction=float(l0_fraction),
  )

  plan: list[tuple[int, int]] = []
  for stage_layer, layer_idx in enumerate(encoder_layers):
    quota = quotas[stage_layer]
    selected = _select_hot_prefix_with_fallback(
      layer_idx=layer_idx,
      pairs=by_layer.get(layer_idx, []),
      quota=quota,
      num_experts=int(adapter.num_expert_per_layer),
      allow_sequential_fallback=allow_sequential_fallback,
    )
    if len(selected) != quota:
      raise ValueError(
        f"L0 priority hot initial plan layer {layer_idx} has {len(selected)} entries, "
        f"expected quota {quota}; pass allow_sequential_fallback=True"
      )
    plan.extend(selected)

  if len(plan) != total_slots:
    raise ValueError(
      f"L0 priority hot initial plan has {len(plan)} entries, expected {total_slots}; "
      "reduce cache_rate or pass allow_sequential_fallback=True"
    )
  return _order_initial_plan_for_encoder_lru(plan, adapter)


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
