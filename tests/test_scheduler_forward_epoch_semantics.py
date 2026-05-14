from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFETCHER_CPP = REPO_ROOT / "src/cpp_worker/prefetcher.cpp"
PREFETCHER_HPP = REPO_ROOT / "src/cpp_worker/prefetcher.hpp"


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


def test_generation_start_task_uses_explicit_decoder_warmup_action():
    hpp = _read(PREFETCHER_HPP)

    assert "DecoderWarmupAction" in hpp
    assert "bool rebuild_decoder_warmup" not in hpp


def test_preserve_forward_epoch_does_not_touch_decoder_warmup_state():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::start_generation")
    preserve_case = _text_between(
        body,
        "case DecoderWarmupAction::kPreserve",
        "case DecoderWarmupAction::kClear",
    )

    assert "set_phase(kEncoderPhase)" not in preserve_case
    assert "rebuild_decoder_warmup_queue()" not in preserve_case
    assert "clear_decoder_warmup_queue()" not in preserve_case


def test_clearing_normal_job_queues_preserves_decoder_warmup_queue():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::clear_all_job_queues")

    assert "clear_all_prefetch_queues()" in body
    assert "precise_job_queue.clear()" in body
    assert "clear_decoder_warmup_queue()" not in body


def test_actual_decoder_layer_clears_decoder_warmup_queue():
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::advance_actual_layer")
    set_phase_body = _function_body(cpp, "void FetchScheduleWorker::set_phase")

    assert "metas->is_decoder_layer(layer_idx)" in body
    assert "set_phase(kDecoderPredictorPhase)" in body
    assert "phase == kDecoderPredictorPhase" in set_phase_body
    assert "clear_decoder_warmup_queue()" in set_phase_body


def test_last_layer_generation_boundary_preserves_decoder_warmup_overlap():
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
    cpp = _read(PREFETCHER_CPP)
    body = _function_body(cpp, "void PrefetchMngr::reset_and_load_initial_cache")

    assert "DecoderWarmupAction::kRebuildForGenerateStart" in body
