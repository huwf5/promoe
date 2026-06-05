from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ERPP_CPP = REPO_ROOT / "src/cpp_worker/erpp_encoder_predictor.cpp"
ERPP_HPP = REPO_ROOT / "src/cpp_worker/erpp_encoder_predictor.hpp"
PREFETCHER_CPP = REPO_ROOT / "src/cpp_worker/prefetcher.cpp"
WORKER_CPP = REPO_ROOT / "src/cpp_worker/worker.cpp"
RUN_SWITCH = REPO_ROOT / "performance/run_switch_mmlu_validation.sh"


def _text(path):
  return path.read_text()


def _function_body(text, signature):
  start = text.index(signature)
  brace = text.index("{", start)
  depth = 0
  for idx in range(brace, len(text)):
    if text[idx] == "{":
      depth += 1
    elif text[idx] == "}":
      depth -= 1
      if depth == 0:
        return text[brace + 1:idx]
  raise AssertionError(f"could not find function body for {signature}")


def test_diagnostics_env_gate_exists_in_predictor_worker_and_scheduler():
  for path in (ERPP_CPP, PREFETCHER_CPP, WORKER_CPP):
    text = _text(path)
    assert "SPARSE_CACHE_LOG_ERPP_ENCODER_DIAGNOSTICS" in text
    assert "log_erpp_encoder_diagnostics_enabled" in text


def test_predictor_records_epoch_and_logs_copy_wait_and_predict_timing():
  hpp = _text(ERPP_HPP)
  cpp = _text(ERPP_CPP)
  record_body = _function_body(cpp, "void ErppEncoderPredictor::record_encoder_layer0")
  predict_body = _function_body(cpp, "ErppEncoderPrediction ErppEncoderPredictor::predict_recorded")

  assert "recorded_forward_epoch" in hpp
  assert "recorded_generate_epoch" in hpp
  assert "recorded_forward_epoch = forward_epoch" in record_body
  assert "recorded_generate_epoch = generate_epoch" in record_body
  assert "record_wait_us=" in predict_body
  assert "predict_cpu_us=" in predict_body
  assert "erpp_encoder_diagnostics: predictor_timing" in predict_body


def test_worker_logs_prediction_summary_with_budget_and_ranking_totals():
  body = _function_body(
    _text(WORKER_CPP),
    "void ErppEncoderPredictWorker::do_one_task_impl",
  )

  assert "ranking_experts_total" in body
  assert "budget_experts_total" in body
  assert "enabled_layers" in body
  assert "erpp_encoder_diagnostics: prediction_summary" in body


def test_scheduler_logs_jit_conversion_and_demand_coverage_summaries():
  cpp = _text(PREFETCHER_CPP)
  maybe_body = _function_body(cpp, "void FetchScheduleWorker::maybe_enqueue_encoder_jit_refill")
  demand_body = _function_body(cpp, "void FetchScheduleWorker::log_encoder_jit_layer_entry_diagnostics")

  assert "erpp_encoder_diagnostics: jit_enqueue_summary" in maybe_body
  assert "inspected_layers=" in maybe_body
  assert "required_experts=" in maybe_body
  assert "enqueued_experts=" in maybe_body
  assert "per_idle_limited=" in maybe_body
  assert "erpp_encoder_diagnostics: demand_coverage" in demand_body
  assert "predicted_cover=" in demand_body
  assert "ready_cover=" in demand_body
  assert "submitted_not_ready=" in demand_body


def test_validation_script_exports_diagnostics_and_collects_summary_lines():
  text = _text(RUN_SWITCH)

  assert 'ERPP_ENCODER_DIAGNOSTICS="${ERPP_ENCODER_DIAGNOSTICS:-0}"' in text
  assert "SPARSE_CACHE_LOG_ERPP_ENCODER_DIAGNOSTICS" in text
  assert "SPARSE_CACHE_LOG_ENCODER_LAYER_STATS" in text
  assert "erpp_encoder_diagnostics:" in text
  assert "encoder_layer_stats:" in text


def test_scheduler_logs_dispatch_and_block_reasons_with_diagnostics_prefix():
  body = _function_body(_text(PREFETCHER_CPP), "bool FetchScheduleWorker::pop_next_prefetch_for_class")

  assert "erpp_encoder_diagnostics: scheduler_block" in body
  assert "reason=no_reclaimable_encoder" in body
  assert "reason=no_safe_victim" in body
  assert "erpp_encoder_diagnostics: scheduler_dispatch" in body
  assert "request=encoder_jit_refill" in body
  assert "request=encoder_predictor_prefetch" in body


def test_scheduler_supports_auto_refill_scoring_and_wait_feedback():
  hpp = _text(REPO_ROOT / "src/cpp_worker/prefetcher.hpp")
  cpp = _text(PREFETCHER_CPP)
  maybe_body = _function_body(cpp, "void FetchScheduleWorker::maybe_enqueue_encoder_jit_refill")
  score_body = _function_body(cpp, "double FetchScheduleWorker::score_encoder_jit_refill_candidate")
  done_body = _function_body(cpp, "void PrefetchMngr::log_encoder_layer_done_stats")

  assert 'spec == "all" || spec == "auto"' in hpp
  assert "encoder_jit_auto_refill_enabled" in hpp
  assert "EncoderJitRefillCandidate" in hpp
  assert "encoder_jit_wait_penalty_ema_us" in hpp
  assert "encoder_jit_auto_floor" in hpp
  assert "remaining_future_layers" in hpp

  assert "mode=auto" in maybe_body
  assert "effective_per_idle_limit" in maybe_body
  assert "build_encoder_jit_refill_candidate" in maybe_body
  assert "encoder_jit_auto_floor" in cpp
  assert "candidate.score" in maybe_body
  assert "best.required.front()" in maybe_body
  assert "wait_penalty_us=" in maybe_body

  assert "candidate.budget = candidate.floor_value" in cpp
  assert "occupancy_gap" in score_body
  assert "predicted_missing" in score_body
  assert "deadline_weight" in score_body
  assert "slack_weight" in score_body
  assert "encoder_jit_wait_penalty_ema_us" in score_body
  assert "update_encoder_jit_wait_penalty" in done_body


def test_encoder_layer_stats_collection_is_gated_before_work():
  cpp = _text(PREFETCHER_CPP)
  for signature in (
      "void PrefetchMngr::log_encoder_layer_entry_stats",
      "void PrefetchMngr::log_encoder_layer_use_stats",
      "void PrefetchMngr::log_encoder_layer_done_stats",
  ):
    body = _function_body(cpp, signature)
    gate_pos = body.index("!log_encoder_layer_stats_enabled()")
    assert "!log_erpp_encoder_diagnostics_enabled()" in body[:gate_pos + 120]
    for marker in (
        "ensure_encoder_layer_stats_size",
        "encoder_layer_expert_entry_hit",
        "ensure_encoder_prefetch_metrics",
        "update_encoder_jit_wait_penalty",
    ):
      if marker in body:
        assert gate_pos < body.index(marker)
