from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFETCHER_CPP = REPO_ROOT / "src/cpp_worker/prefetcher.cpp"
PREFETCHER_HPP = REPO_ROOT / "src/cpp_worker/prefetcher.hpp"
WORKER_CPP = REPO_ROOT / "src/cpp_worker/worker.cpp"
WORKER_HPP = REPO_ROOT / "src/cpp_worker/worker.hpp"


def _read(path: Path) -> str:
    return path.read_text()


def _text_between(source: str, start: str, end: str) -> str:
    start_idx = source.index(start)
    end_idx = source.index(end, start_idx)
    return source[start_idx:end_idx]


def _function_body(source: str, signature: str) -> str:
    start_idx = source.index(signature)
    body_start = source.index("{", start_idx)
    depth = 0
    for idx in range(body_start, len(source)):
        char = source[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start_idx : idx + 1]
    raise AssertionError(f"could not find end of function body for {signature!r}")


def test_forward_epoch_start_task_uses_explicit_decoder_warmup_action():
    hpp = _read(PREFETCHER_HPP)

    assert "DecoderWarmupAction" in hpp
    assert "bool rebuild_decoder_warmup" not in hpp


def test_forward_epoch_names_the_forward_boundary_not_generate_call():
    sources = {
        "prefetcher.hpp": _read(PREFETCHER_HPP),
        "prefetcher.cpp": _read(PREFETCHER_CPP),
        "worker.hpp": _read(WORKER_HPP),
        "worker.cpp": _read(WORKER_CPP),
    }

    combined = "\n".join(sources.values())
    assert "forward_epoch" in combined
    assert "ForwardEpochStartTask" in combined
    assert "prefetch_generation" not in combined
    assert "current_generation" not in combined
    assert "GenerationStartTask" not in combined
    assert "kGenerationStart" not in combined
    assert "start_generation" not in combined
    assert ".generation" not in combined
    assert "current_forward_epoch" in combined
    assert "kForwardEpochStart" in combined
    assert "start_forward_epoch" in combined


def test_preserve_forward_epoch_does_not_touch_decoder_warmup_state():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::start_forward_epoch")
    preserve_case = _text_between(
        body,
        "case DecoderWarmupAction::kPreserve",
        "case DecoderWarmupAction::kClear",
    )

    assert "set_phase(kEncoderPhase)" not in preserve_case
    assert "rebuild_decoder_warmup_queue()" not in preserve_case
    assert "clear_decoder_warmup_plan_queue()" not in preserve_case


def test_clearing_normal_job_queues_preserves_decoder_warmup_queue():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::clear_all_job_queues")

    assert "clear_all_prefetch_queues()" in body
    assert "precise_job_queue.clear()" in body
    assert "clear_decoder_warmup_plan_queue()" not in body


def test_actual_decoder_layer_preserves_decoder_warmup_queue():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::advance_actual_layer")
    set_phase_body = _function_body(cpp, "void FetchScheduleWorker::set_phase")

    assert "metas->is_decoder_layer(layer_idx)" in body
    assert "set_phase(kDecoderPredictorPhase)" in body
    assert "phase == kDecoderPredictorPhase" in set_phase_body
    assert "clear_decoder_warmup_plan_queue()" not in set_phase_body
    assert "clear_prefetch_class_up_to_layer(PrefetchClass::kDecoderWarmup" not in body


def test_last_layer_forward_epoch_boundary_preserves_decoder_warmup_overlap():
    cpp = _read(PREFETCHER_CPP)
    body = _text_between(
        cpp,
        "void PrefetchMngr::one_moe_layer_done",
        "void PrefetchMngr::report_one_expert",
    )

    assert "rebuild_decoder_warmup = true" not in body
    assert "DecoderWarmupAction::kPreserve" in body


def test_forward_boundary_logits_reports_do_not_unconditionally_rebuild_decoder_warmup():
    cpp = _read(PREFETCHER_CPP)

    for signature in (
        "void PrefetchMngr::report_moe_attn_logits",
        "void PrefetchMngr::report_moe_layer_logits",
    ):
        body = _function_body(cpp, signature)
        assert "rebuild_decoder_warmup = true" not in body
        assert "DecoderWarmupAction::kPreserve" in body


def test_reset_and_load_initial_cache_still_rebuilds_for_generate_start():
    hpp = _read(PREFETCHER_HPP)
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void PrefetchMngr::reset_for_generate")

    assert "void reset_and_load_initial_cache() { reset_for_generate(); }" in hpp
    assert "DecoderWarmupAction::kRebuildForGenerateStart" in body


def test_forward_epoch_boundary_filters_prefetch_queues_by_task_epoch():
    cpp = _read(PREFETCHER_CPP)
    hpp = _read(PREFETCHER_HPP)

    assert "void clear_stale_prefetch_queues_before_epoch(int64_t min_forward_epoch)" in hpp
    body = _function_body(
        cpp,
        "void FetchScheduleWorker::clear_stale_prefetch_queues_before_epoch",
    )
    assert "prefetch_queues.encoder_predictor_by_layer" in body
    assert "prefetch_queues.decoder_predictor_by_layer" in body
    assert "prefetch_queues.decoder_warmup_plan_queue" not in body
    assert "TaskQueue kept" in body
    assert "CopyTask task = queue.front()" in body
    assert "queue.pop()" in body
    assert "task.forward_epoch >= min_forward_epoch" in body
    assert "kept.push(task)" in body
    assert "queue.push(task)" in body
    assert body.index("kept.push(task)") < body.rindex("queue.push(task)")
    assert "per_layer_job_queues" not in body
    assert "precise_job_queue" not in body


def test_normal_forward_epoch_advance_preserves_future_prefetch_tasks():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::start_forward_epoch")
    boundary = _text_between(
        body,
        "if (forward_epoch > current_forward_epoch)",
        "switch (decoder_warmup_action)",
    )

    assert "clear_stale_prefetch_queues_before_epoch(current_forward_epoch)" in boundary
    assert "precise_job_queue.clear()" in boundary
    assert "clear_all_job_queues()" not in boundary


def test_decoder_warmup_is_not_pruned_by_forward_epoch_or_layer_progress():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::prune_prefetch_class")
    warmup_branch = body[body.rindex("} else {"):]

    assert "prune_decoder_warmup_queue" in warmup_branch
    assert "prune_queue(prefetch_queues.decoder_warmup_plan_queue" not in warmup_branch


def test_send_one_job_does_not_treat_decoder_warmup_as_layer_stale():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::send_one_job")
    stale_guard = body[body.index("if (task->is_precise == false"):body.index("// nullptr and 0: first time task")]

    assert "task->request_type != kCacheRequestDecoderWarmupPrefetch" in stale_guard


def test_decoder_warmup_prefetch_can_run_after_decoder_phase_starts():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::requires_encoder_phase")

    assert "cls == PrefetchClass::kEncoderPredictor" in body
    assert "PrefetchClass::kDecoderWarmup" not in body


def test_reset_still_discards_all_prefetch_and_precise_work():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(ResetTask *task)")

    assert "clear_all_job_queues()" in body
    assert "clear_stale_prefetch_queues_before_epoch" not in body


def test_cross_token_prediction_paths_advance_epoch_before_submit():
    cpp = _read(PREFETCHER_CPP)

    attn_body = _function_body(cpp, "void PrefetchMngr::report_moe_attn_logits")
    attn_bump_pos = attn_body.index("forward_epoch += 1")
    attn_submit_pos = attn_body.index(
        "predict_thread->on_moe_attn_input_logits_recorded(layer_id, forward_epoch, generate_epoch)"
    )
    assert attn_bump_pos < attn_submit_pos

    layer_body = _function_body(cpp, "void PrefetchMngr::report_moe_layer_logits")
    layer_bump_pos = layer_body.index("forward_epoch += 1")
    layer_submit_pos = layer_body.index(
        "predict_thread->on_moe_layer_logits_recorded(layer_id, predict_forward_epoch, generate_epoch)"
    )
    layer_bump_guard = layer_body[layer_body.rindex("if", 0, layer_bump_pos):layer_bump_pos]
    assert layer_bump_pos < layer_submit_pos
    assert "metas->predict_input_mode == kMoeLayerLogits" in layer_bump_guard
    assert "layer_id == metas->first_decoder_layer()" in layer_bump_guard
    assert "layer_id == metas->num_layer" not in layer_bump_guard
    assert "predict_forward_epoch = forward_epoch + 1" in layer_body
    next_token_pos = layer_body.index("predict_forward_epoch = forward_epoch + 1")
    assert next_token_pos < layer_submit_pos

    last_body = _function_body(cpp, "void PrefetchMngr::one_moe_layer_done")
    bump_pos = last_body.index("forward_epoch += 1")
    submit_pos = last_body.index("predict_thread->on_one_iter_done(forward_epoch, generate_epoch)")
    assert bump_pos < submit_pos

