import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFETCHER_CPP = REPO_ROOT / "src/cpp_worker/prefetcher.cpp"
PREFETCHER_HPP = REPO_ROOT / "src/cpp_worker/prefetcher.hpp"
CACHE_HPP = REPO_ROOT / "src/cpp_worker/cache.hpp"
CACHE_CPP = REPO_ROOT / "src/cpp_worker/cache.cpp"


def _text(path):
    return path.read_text()


def _compact(text):
    return " ".join(text.split())


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


def test_prefetch_manager_hooks_enqueue_reclaimable_updates_instead_of_touching_cache_policy():
    cpp = _text(PREFETCHER_CPP)
    expected_calls = {
        "void PrefetchMngr::report_one_layer(int layer_id, int64_t* experts, int64_t num_expert)":
            "enqueue_layer_reclaimable_except",
        "void PrefetchMngr::one_moe_layer_done(int layer_id)":
            "enqueue_layer_reclaimable",
        "void PrefetchMngr::one_expert_done(int layer_id, int expert_id)":
            "enqueue_expert_reclaimable",
    }
    for signature, expected_call in expected_calls.items():
        body = _function_body(cpp, signature)
        assert "cache->mark_reclaimable" not in body
        assert "cache->mark_layer_reclaimable" not in body
        assert "cache->" not in body
        assert "policy->" not in body
        assert f"fetch_schedule_thread->{expected_call}" in body


def test_fetch_scheduler_declares_coalesced_reclaimable_pending_state():
    hpp = _text(PREFETCHER_HPP)
    assert "struct PendingReclaimableUpdate" in hpp
    assert "kSomeExperts" in hpp
    assert "kLayerExcept" in hpp
    assert "kLayerAll" in hpp
    assert "pending_reclaimable_updates" in hpp
    assert "pending_reclaimable_layers" in hpp
    assert "pending_reclaimable_layer_mask" in hpp
    assert "reclaimable_update_lock" in hpp
    assert "has_pending_reclaimable_updates" in hpp
    compact_hpp = _compact(hpp)
    assert (
        "void enqueue_layer_reclaimable_except("
        "int layer_idx, const std::vector<uint8_t>& needed_mask)"
    ) in compact_hpp
    assert "void enqueue_expert_reclaimable(int layer_idx, int expert_idx)" in compact_hpp
    assert "void enqueue_layer_reclaimable(int layer_idx)" in compact_hpp
    assert "void drain_reclaimable_updates(int max_updates = -1)" in hpp


def test_enqueue_methods_do_not_touch_cache_or_policy_state():
    cpp = _text(PREFETCHER_CPP)
    for signature in [
        "void FetchScheduleWorker::enqueue_layer_reclaimable_except("
        "\n    int layer_idx,"
        "\n    const std::vector<uint8_t>& needed_mask)",
        "void FetchScheduleWorker::enqueue_expert_reclaimable(int layer_idx, int expert_idx)",
        "void FetchScheduleWorker::enqueue_layer_reclaimable(int layer_idx)",
    ]:
        body = _function_body(cpp, signature)
        assert "cache->" not in body
        assert "policy->" not in body


def test_precise_queue_remains_before_reclaimable_drain():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::pop_next_task")
    precise_pos = body.index("precise_job_queue")
    drain_pos = body.index("drain_reclaimable_updates")
    normal_prefetch_pos = body.index("pop_next_normal_prefetch")
    assert precise_pos < drain_pos
    assert drain_pos < normal_prefetch_pos


def test_decoder_warmup_drains_before_has_reclaimable_check():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::pop_next_decoder_warmup")
    drain_pos = body.index("drain_reclaimable_updates")
    has_pos = body.index("cache->has_reclaimable_encoder")
    assert drain_pos < has_pos


def test_decoder_warmup_overlap_uses_chunked_tasks_when_chunk_prefetch_is_enabled():
    hpp = _text(PREFETCHER_HPP)
    cpp = _text(PREFETCHER_CPP)

    assert "struct DecoderWarmupEntry" in hpp
    assert "int start_mem_buf_idx" in hpp
    assert "int stop_mem_buf_idx" in hpp
    assert "std::queue<DecoderWarmupEntry> decoder_warmup_queue" in hpp

    rebuild_body = _function_body(cpp, "void FetchScheduleWorker::rebuild_decoder_warmup_queue")
    assert "metas->chunk_prefetch" in rebuild_body
    assert "decoder_warmup_queue.push({layer_idx, expert_idx, j, j + 1})" in _compact(rebuild_body)
    assert (
        "decoder_warmup_queue.push({layer_idx, expert_idx, 0, metas->num_per_expert_param})"
        in _compact(rebuild_body)
    )

    pop_body = _function_body(cpp, "bool FetchScheduleWorker::pop_next_decoder_warmup")
    assert "auto entry = decoder_warmup_queue.front()" in pop_body
    assert "task.start_mem_buf_idx = entry.start_mem_buf_idx" in pop_body
    assert "task.stop_mem_buf_idx = entry.stop_mem_buf_idx" in pop_body
    assert "expert->num_ready >= entry.stop_mem_buf_idx" in pop_body
    assert "expert->gpu_data == nullptr && entry.start_mem_buf_idx > 0" in pop_body
    assert "task.request_type = kCacheRequestDecoderWarmupOverlap" in pop_body


def test_cache_reset_discards_partial_chunk_prefetch_entries_but_not_active_experts():
    cpp = _text(CACHE_CPP)
    body = _function_body(cpp, "void CacheMngr::reset_cache_contents")

    assert "status == kFetching" in body
    assert "expert->expert_status.transfer(kFetching, kIdle)" in body
    assert "status == kLaunching" not in body
    assert "status == kUsing" not in body


def test_cache_reset_retires_scheduler_aware_policy_instead_of_destroying_it():
    cpp = _text(CACHE_CPP)
    body = _function_body(cpp, "void CacheMngr::reset_cache_contents")

    assert 'metas->cache_policy == "scheduler_aware"' in body
    assert "retired_scheduler_aware_policies().push_back(cache_slot.policy)" in body
    assert "reset slot policy retired" in body
    assert "policy_factory.create_policy(metas->cache_policy)" in body


def test_cache_manager_destructor_retires_scheduler_aware_policy():
    cpp = _text(CACHE_CPP)
    body = _function_body(cpp, "CacheMngr::~CacheMngr()")

    assert 'metas->cache_policy == "scheduler_aware"' in body
    assert "retired_scheduler_aware_policies().push_back(l.policy)" in body
    assert "l.policy.reset()" in body


def test_scheduler_aware_reset_does_not_depend_on_list_traversal_or_destruction():
    cpp = _text(CACHE_CPP)
    hpp = _text(CACHE_HPP)
    body = _function_body(cpp, "bool CachePolicySchedulerAware::reset_for_cache_reset()")

    assert "virtual bool reset_for_cache_reset()" in hpp
    assert "bool reset_for_cache_reset() override" in hpp
    assert "global_map.clear()" in body
    assert "encoder_map.clear()" in body
    assert "decoder_map.clear()" in body
    assert "reclaimable_map.clear()" in body
    assert "node_free_buffer.clear()" in body
    assert "reset_list(global_lru)" in body
    assert "reset_list(encoder_lru)" in body
    assert "reset_list(decoder_lru)" in body
    assert "reset_list(reclaimable_encoder_lru)" in body
    assert "delete pop_front()" not in body
    assert "delete " not in body


def test_preempt_one_expert_drains_after_demand_preempt():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(PreemptOneExpertTask *task)")
    preempt_pos = body.index("preempt_one_expert")
    drain_pos = body.index("drain_reclaimable_updates")
    assert preempt_pos < drain_pos


def test_preempt_task_drains_after_layer_preempt_and_queue_cleanup():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(PreemptTask *task)")
    advance_pos = body.index("advance_actual_layer")
    queue_cleanup_pos = body.rindex("per_layer_job_queues")
    drain_pos = body.index("drain_reclaimable_updates")
    assert advance_pos < drain_pos
    assert queue_cleanup_pos < drain_pos


def test_preempt_layer_deduplicates_experts_before_launch_transition():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::preempt_one_layer_without_reorder_")
    compact_body = _compact(body)

    assert "std::unordered_set<int64_t> seen_demand_experts" in body
    assert "flatten_expert(layer_idx, e->expert_idx)" in body
    assert "seen_demand_experts.insert" in body
    assert "duplicate demand expert" in body
    assert re.search(
        r"if\s*\(\s*!seen_demand_experts\.insert\s*\([^)]*\)\.second\s*\)\s*"
        r"\{[^{}]*duplicate demand expert[^{}]*continue\s*;",
        compact_body,
    )
    dedupe_pos = body.index("seen_demand_experts.insert")
    for later in (
        "cache->is_in_cache(e)",
        "add_single_tasks_for_one_expert",
        "e->expert_status.transfer(kReady, kLaunching, false)",
        "cache_hit(e, true)",
        "current_task.is_precise",
    ):
        assert dedupe_pos < body.index(later)


def test_ready_expert_launch_transition_is_idempotent_for_redundant_demands():
    cpp = _text(PREFETCHER_CPP)

    for signature in [
        "void FetchScheduleWorker::preempt_one_expert",
        "void FetchScheduleWorker::preempt_one_layer_without_reorder_",
    ]:
        body = _function_body(cpp, signature)
        compact_body = _compact(body)
        assert "transfer(kReady, kLaunching, false)" in body
        assert "launch_status == kReady" in body
        assert "launch_status == kLaunching" in body
        assert re.search(
            r"if\s*\(\s*launch_status\s*==\s*kReady\s*\)\s*\{[^{}]*cache_hit\(e,\s*true\);",
            compact_body,
        )
        assert re.search(
            r"else\s+if\s*\(\s*launch_status\s*==\s*kLaunching\s*\)",
            compact_body,
        )
        assert "CHECK(launch_status == kReady || launch_status == kLaunching)" in body
        assert "e->expert_status.transfer(kReady, kLaunching);" not in body


def test_coalescing_rules_are_encoded_in_enqueue_methods():
    cpp = _text(PREFETCHER_CPP)

    layer_except_body = _function_body(
        cpp,
        "void FetchScheduleWorker::enqueue_layer_reclaimable_except",
    )
    assert "pending.mode == PendingReclaimableUpdate::kLayerAll" in layer_except_body
    assert re.search(
        r"pending\.mode\s*==\s*PendingReclaimableUpdate::kLayerAll\s*\)\s*\{\s*return;",
        layer_except_body,
        re.MULTILINE,
    )
    assert "pending.mode == PendingReclaimableUpdate::kSomeExperts" in layer_except_body
    assert "merged[expert_idx] = 0" in layer_except_body
    assert "pending.mode == PendingReclaimableUpdate::kLayerExcept" in layer_except_body
    assert "merged[expert_idx] = merged[expert_idx] && pending.needed_mask[expert_idx]" in _compact(layer_except_body)
    assert "pending.mode = PendingReclaimableUpdate::kLayerExcept" in layer_except_body
    assert "pending.needed_mask = std::move(merged)" in layer_except_body
    assert "pending.expert_mask.clear()" in layer_except_body

    one_expert_body = _function_body(
        cpp,
        "void FetchScheduleWorker::enqueue_expert_reclaimable",
    )
    assert "pending.mode == PendingReclaimableUpdate::kLayerAll" in one_expert_body
    assert re.search(
        r"pending\.mode\s*==\s*PendingReclaimableUpdate::kLayerAll\s*\)\s*\{\s*return;",
        one_expert_body,
        re.MULTILINE,
    )
    assert "pending.mode == PendingReclaimableUpdate::kLayerExcept" in one_expert_body
    assert "pending.needed_mask[expert_idx] = 0" in one_expert_body
    assert "pending.mode = PendingReclaimableUpdate::kSomeExperts" in one_expert_body
    assert "pending.expert_mask[expert_idx] = 1" in one_expert_body

    layer_all_body = _function_body(
        cpp,
        "void FetchScheduleWorker::enqueue_layer_reclaimable(int layer_idx)",
    )
    assert "pending.mode = PendingReclaimableUpdate::kLayerAll" in layer_all_body
    assert "pending.expert_mask.clear()" in layer_all_body
    assert "pending.needed_mask.clear()" in layer_all_body


def test_pending_layer_dedup_and_drain_budget_are_encoded():
    cpp = _text(PREFETCHER_CPP)

    note_body = _function_body(
        cpp,
        "void FetchScheduleWorker::note_pending_reclaimable_layer_locked",
    )
    assert "if (!pending_reclaimable_layer_mask[layer_idx])" in note_body
    assert "pending_reclaimable_layer_mask[layer_idx] = 1" in note_body
    assert "pending_reclaimable_layers.push_back(layer_idx)" in note_body

    drain_body = _function_body(cpp, "void FetchScheduleWorker::drain_reclaimable_updates")
    assert "max_updates >= 0 && drained >= max_updates" in drain_body
    assert "remaining_layers.push_back(layer_idx)" in drain_body
    assert "pending_reclaimable_layers.swap(remaining_layers)" in drain_body
    assert "!pending_reclaimable_layers.empty()" in drain_body
    assert "pending_reclaimable_layer_mask[layer_idx] = 0" in drain_body


def test_cache_manager_has_mask_overload_for_layer_except():
    hpp = _text(CACHE_HPP)
    cpp = _text(CACHE_CPP)
    vector_mask_param = r"const\s+std::vector<uint8_t>\s*&?\s*\w*"
    assert re.search(
        rf"mark_layer_reclaimable_except\s*\(\s*int\s+\w+\s*,\s*{vector_mask_param}\s*\)",
        hpp,
        re.MULTILINE,
    )
    assert re.search(
        rf"void\s+CacheMngr::mark_layer_reclaimable_except\s*"
        rf"\(\s*int\s+\w+\s*,\s*{vector_mask_param}\s*\)",
        cpp,
        re.MULTILINE,
    )


def test_new_prefetcher_path_uses_vector_mask_for_layer_except():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(
        cpp,
        "void PrefetchMngr::report_one_layer(int layer_id, int64_t* experts, int64_t num_expert)",
    )
    assert "std::vector<uint8_t>" in body
    assert "std::unordered_set<int> needed" not in body
    assert "enqueue_layer_reclaimable_except" in body
    assert re.search(
        r"enqueue_layer_reclaimable_except\s*\(\s*layer_id\s*,\s*needed_mask\s*\)",
        body,
        re.MULTILINE,
    )


def test_encoder_report_one_layer_does_not_wait_for_predictor_progress():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(
        cpp,
        "void PrefetchMngr::report_one_layer(int layer_id, int64_t* experts, int64_t num_expert)",
    )
    compact_body = _compact(body)

    assert "consume_prefetch_layer_progress" in body
    assert re.search(
        r"if\s*\(\s*metas->is_decoder_layer\s*\(\s*layer_id\s*\)\s*\)\s*"
        r"\{[^{}]*consume_prefetch_layer_progress\s*\(",
        compact_body,
    )


def test_idle_task_does_not_overwrite_inflight_fetch_task():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(IdleTask *idle_task)")
    compact_body = _compact(body)

    assert re.search(
        r"current_task\.expert\s*!=\s*nullptr.*"
        r"add_one_task\s*\(\s*&this->idle_task\s*\).*"
        r"return\s*;",
        compact_body,
    )
    guard_idx = body.index("current_task.expert != nullptr")
    pop_idx = body.index("pop_next_task")
    assert guard_idx < pop_idx


def test_stale_partial_task_restarts_only_for_precise_demand():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::send_one_job(CopyTask *task)")
    compact_body = _compact(body)

    assert "task->expert->num_ready > task->start_mem_buf_idx" in body
    assert "task->expert->num_ready < task->start_mem_buf_idx" in body
    assert re.search(
        r"task->expert->num_ready\s*<\s*task->start_mem_buf_idx.*"
        r"if\s*\(\s*task->is_precise\s*\).*"
        r"task->start_mem_buf_idx\s*=\s*task->expert->num_ready",
        compact_body,
    )
    assert re.search(
        r"task->expert->num_ready\s*<\s*task->start_mem_buf_idx.*"
        r"if\s*\(\s*task->is_precise\s*\).*"
        r"else\s*\{[^{}]*return false\s*;",
        compact_body,
    )
