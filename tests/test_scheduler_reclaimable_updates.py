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
    idx = brace
    state = "code"
    while idx < len(text):
        ch = text[idx]
        nxt = text[idx + 1] if idx + 1 < len(text) else ""

        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line_comment"
                idx += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block_comment"
                idx += 2
                continue
            if ch == '"':
                state = "string"
                idx += 1
                continue
            if ch == "'":
                state = "char"
                idx += 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[brace + 1:idx]
        elif state == "line_comment":
            if ch == "\n":
                state = "code"
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                idx += 2
                continue
        elif state == "string":
            if ch == "\\":
                idx += 2
                continue
            if ch == '"':
                state = "code"
        elif state == "char":
            if ch == "\\":
                idx += 2
                continue
            if ch == "'":
                state = "code"

        idx += 1
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
    assert "struct PendingCachePolicyUpdate" in hpp
    assert "kSomeExperts" in hpp
    assert "kLayerExcept" in hpp
    assert "kLayerAll" in hpp
    assert "pending_cache_policy_updates" in hpp
    assert "pending_cache_policy_layers" in hpp
    assert "pending_cache_policy_layer_mask" in hpp
    assert "cache_policy_update_lock" in hpp
    assert "has_pending_cache_policy_updates" in hpp
    compact_hpp = _compact(hpp)
    assert (
        "void enqueue_layer_reclaimable_except("
        "int layer_idx, const std::vector<uint8_t>& needed_mask)"
    ) in compact_hpp
    assert "void enqueue_expert_reclaimable(int layer_idx, int expert_idx, bool clear_demand_protection)" in compact_hpp
    assert "void enqueue_layer_reclaimable(int layer_idx)" in compact_hpp
    assert "void drain_cache_policy_updates(int max_updates = -1)" in hpp


def test_enqueue_methods_do_not_touch_cache_or_policy_state():
    cpp = _text(PREFETCHER_CPP)
    for signature in [
        "void FetchScheduleWorker::enqueue_layer_reclaimable_except("
        "\n    int layer_idx,"
        "\n    const std::vector<uint8_t>& needed_mask)",
        "void FetchScheduleWorker::enqueue_expert_reclaimable",
        "void FetchScheduleWorker::enqueue_layer_reclaimable(int layer_idx)",
    ]:
        body = _function_body(cpp, signature)
        assert "cache->" not in body
        assert "policy->" not in body


def test_pop_next_task_drains_reclaimable_updates_before_precise_queue():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::pop_next_task")
    precise_pos = body.index("precise_job_queue")
    drain_pos = body.index("drain_cache_policy_updates")
    prefetch_pos = body.index("pop_next_prefetch_for_class(PrefetchClass::kEncoderPredictor")
    assert drain_pos < precise_pos
    assert drain_pos < prefetch_pos


def test_decoder_warmup_drains_before_has_reclaimable_check():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::pop_next_prefetch_for_class")
    warmup_pos = body.index("PrefetchClass::kDecoderWarmup")
    drain_pos = body.index("drain_cache_policy_updates", warmup_pos)
    has_pos = body.index("cache->has_reclaimable_encoder", drain_pos)
    assert drain_pos < has_pos


def test_decoder_warmup_prefetch_uses_chunked_tasks_in_unified_queue_set():
    hpp = _text(PREFETCHER_HPP)
    cpp = _text(PREFETCHER_CPP)

    assert "TaskQueue decoder_warmup_plan_queue" in hpp
    assert "std::queue<DecoderWarmupEntry> decoder_warmup_queue" not in hpp

    rebuild_body = _function_body(cpp, "void FetchScheduleWorker::rebuild_decoder_warmup_queue")
    assert "metas->chunk_prefetch" in rebuild_body
    assert "prefetch_queues.decoder_warmup_plan_queue.push" in rebuild_body
    assert "task.start_mem_buf_idx = metas->chunk_prefetch ? j : 0" in rebuild_body
    assert "task.stop_mem_buf_idx = metas->chunk_prefetch ? j + 1 : metas->num_per_expert_param" in rebuild_body

    pop_body = _function_body(cpp, "bool FetchScheduleWorker::pop_next_prefetch_for_class")
    assert "PrefetchClass::kDecoderWarmup" in pop_body
    assert (
        "CopyTask candidate = prefetch_queues.decoder_warmup_plan_queue.front()" in pop_body
        or "auto candidate = prefetch_queues.decoder_warmup_plan_queue.front()" in pop_body
    )
    assert "candidate.stop_mem_buf_idx" in pop_body
    assert "candidate.start_mem_buf_idx" in pop_body
    assert "task = candidate" in pop_body
    assert "task.request_type = kCacheRequestDecoderWarmupPrefetch" in rebuild_body


def test_decoder_warmup_remains_plan_order_fifo_in_unified_queue_set():
    hpp = _text(PREFETCHER_HPP)
    cpp = _text(PREFETCHER_CPP)

    assert "TaskQueue decoder_warmup_plan_queue" in hpp
    rebuild_body = _function_body(cpp, "void FetchScheduleWorker::rebuild_decoder_warmup_queue")
    pop_body = _function_body(cpp, "bool FetchScheduleWorker::pop_next_prefetch_for_class")

    assert "prefetch_queues.decoder_warmup_plan_queue.push" in rebuild_body
    assert "for (auto [layer_idx, expert_idx] : parsed)" in rebuild_body
    assert "prefetch_queues.decoder_warmup_plan_queue.front()" in pop_body
    assert "prefetch_queues.decoder_warmup_plan_queue.pop()" in pop_body


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




def test_encoder_jit_occupancy_uses_cache_count_directly_and_submitted_mask_only_dedupes():
    cache_hpp = _text(CACHE_HPP)
    cache_cpp = _text(CACHE_CPP)
    prefetcher_hpp = _text(PREFETCHER_HPP)
    prefetcher_cpp = _text(PREFETCHER_CPP)

    assert "encoder_layer_cached_count" in cache_hpp
    assert "int encoder_layer_cache_occupancy(int layer_idx) const" in cache_hpp
    assert "CacheMngr::encoder_layer_cache_occupancy(int layer_idx) const" in cache_cpp

    assert "encoder_jit_submitted_mask" in prefetcher_hpp
    assert "encoder_jit_submitted_count" not in prefetcher_hpp
    assert "encoder_jit_submitted_count" not in prefetcher_cpp
    assert "encoder_jit_occupancy" not in prefetcher_hpp
    assert "FetchScheduleWorker::encoder_jit_occupancy" not in prefetcher_cpp

    mark_body = _function_body(prefetcher_hpp, "void mark_encoder_jit_submitted(int layer_idx, int expert_idx)")
    assert "encoder_jit_submitted_mask[layer_idx][expert_idx] = 1" in mark_body
    assert "encoder_jit_submitted_count" not in mark_body

    refill_body = _function_body(prefetcher_cpp, "void FetchScheduleWorker::maybe_enqueue_encoder_jit_refill")
    assert "const int occupancy = cache->encoder_layer_cache_occupancy(layer_idx);" in refill_body
    assert "encoder_jit_occupancy" not in refill_body


def test_encoder_jit_refill_per_idle_minus_one_disables_enqueue_limit():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::maybe_enqueue_encoder_jit_refill")

    assert "metas->erpp_encoder_jit_refill_per_idle > 0" in body
    assert "enqueued >= metas->erpp_encoder_jit_refill_per_idle" in body
    assert "reason=per_idle_limit" in body


def test_encoder_jit_refill_dispatch_only_requires_reclaimable_or_unused_slot():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::encoder_jit_can_dispatch")

    assert "cache->has_unused_slot_for(task.expert)" in body
    assert "cache->has_reclaimable_encoder()" in body
    assert "has_safe_encoder_jit_refill_victim" not in body
    assert "encoder_jit_floor()" not in body
    assert "encoder_jit_protected_layers()" not in body


def test_encoder_jit_entry_diagnostics_correlate_demand_with_ranking_and_readiness():
    hpp = _text(PREFETCHER_HPP)
    cpp = _text(PREFETCHER_CPP)

    assert "void log_encoder_jit_layer_entry_diagnostics" in hpp

    preempt_body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(PreemptTask *task)")
    advance_pos = preempt_body.index("advance_actual_layer")
    diag_pos = preempt_body.index("log_encoder_jit_layer_entry_diagnostics")
    preempt_pos = preempt_body.index("preempt_one_layer_without_reorder_")
    assert advance_pos < diag_pos < preempt_pos

    body = _function_body(cpp, "void FetchScheduleWorker::log_encoder_jit_layer_entry_diagnostics")
    for expected in [
        "erpp_encoder_jit_refill: demand_summary",
        "predicted_cover=",
        "ready_cover=",
        "submitted_not_ready=",
        "predicted_not_ready=",
        "outside_ranking=",
        "ranking_available=",
        "budget=",
        "occupancy=",
        "current=L",
    ]:
        assert expected in body
    for expected in [
        "rank=",
        "submitted=",
        "in_cache=",
        "ready=",
        "is_current_copy=",
        "num_ready=",
        "status=",
    ]:
        assert expected in body


def test_encoder_jit_refill_cache_policy_is_reclaimable_only_without_context():
    cpp = _text(CACHE_CPP)
    hpp = _text(CACHE_HPP)
    select_body = _function_body(
        cpp,
        "ExpertHandler* CachePolicySchedulerAware::select_for_evict(\n"
        "    ExpertHandler* incoming,\n"
        "    CacheRequestType request_type)",
    )

    assert "first_loaded_candidate(reclaimable_map, reclaimable_encoder_lru)" in select_body
    assert "request_type == kCacheRequestEncoderJitRefill" not in select_body
    assert "encoder_jit_refill_context" not in hpp
    assert "has_safe_encoder_jit_refill_victim" not in hpp
    assert "set_encoder_jit_refill_context" not in hpp
    assert "clear_encoder_jit_refill_context" not in hpp
    assert "first_safe_encoder_jit_refill_candidate" not in cpp
    prefetcher_cpp = _text(PREFETCHER_CPP)
    prefetcher_hpp = _text(PREFETCHER_HPP)

    assert "encoder_layer_occupancy" not in cpp
    assert "encoder_jit_protected_layers" not in prefetcher_cpp
    assert "encoder_jit_protected_layers" not in prefetcher_hpp


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
    drain_pos = body.index("drain_cache_policy_updates")
    assert preempt_pos < drain_pos


def test_preempt_task_drains_after_layer_preempt_and_queue_cleanup():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(PreemptTask *task)")
    advance_pos = body.index("advance_actual_layer")
    queue_cleanup_pos = body.index("clear_prefetch_class_for_layer")
    drain_pos = body.index("drain_cache_policy_updates")
    assert advance_pos < drain_pos
    assert queue_cleanup_pos < drain_pos
    assert "per_layer_job_queues" not in body


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
    assert "pending.mode == PendingCachePolicyUpdate::kLayerAll" in layer_except_body
    assert re.search(
        r"pending\.mode\s*==\s*PendingCachePolicyUpdate::kLayerAll\s*\)\s*\{\s*return;",
        layer_except_body,
        re.MULTILINE,
    )
    assert "pending.mode == PendingCachePolicyUpdate::kSomeExperts" in layer_except_body
    assert "merged[expert_idx] = 0" in layer_except_body
    assert "pending.mode == PendingCachePolicyUpdate::kLayerExcept" in layer_except_body
    assert "merged[expert_idx] = merged[expert_idx] && pending.needed_mask[expert_idx]" in _compact(layer_except_body)
    assert "pending.mode = PendingCachePolicyUpdate::kLayerExcept" in layer_except_body
    assert "pending.needed_mask = std::move(merged)" in layer_except_body
    assert "pending.expert_mask.clear()" in layer_except_body

    one_expert_body = _function_body(
        cpp,
        "void FetchScheduleWorker::enqueue_expert_reclaimable",
    )
    assert "pending.mode == PendingCachePolicyUpdate::kLayerAll" in one_expert_body
    assert "pending.mode == PendingCachePolicyUpdate::kLayerAll" in one_expert_body
    assert "return;" in one_expert_body[one_expert_body.index("pending.mode == PendingCachePolicyUpdate::kLayerAll"):]
    assert "pending.mode == PendingCachePolicyUpdate::kLayerExcept" in one_expert_body
    assert "pending.needed_mask[expert_idx] = 0" in one_expert_body
    assert "pending.mode = PendingCachePolicyUpdate::kSomeExperts" in one_expert_body
    assert "pending.expert_mask[expert_idx] = 1" in one_expert_body

    layer_all_body = _function_body(
        cpp,
        "void FetchScheduleWorker::enqueue_layer_reclaimable(int layer_idx)",
    )
    assert "pending.mode = PendingCachePolicyUpdate::kLayerAll" in layer_all_body
    assert "pending.expert_mask.clear()" in layer_all_body
    assert "pending.needed_mask.clear()" in layer_all_body


def test_pending_layer_dedup_and_drain_budget_are_encoded():
    cpp = _text(PREFETCHER_CPP)

    note_body = _function_body(
        cpp,
        "void FetchScheduleWorker::note_pending_cache_policy_layer_locked",
    )
    assert "if (!pending_cache_policy_layer_mask[layer_idx])" in note_body
    assert "pending_cache_policy_layer_mask[layer_idx] = 1" in note_body
    assert "pending_cache_policy_layers.push_back(layer_idx)" in note_body

    drain_body = _function_body(cpp, "void FetchScheduleWorker::drain_cache_policy_updates")
    assert "max_updates >= 0 && drained >= max_updates" in drain_body
    assert "remaining_layers.push_back(layer_idx)" in drain_body
    assert "pending_cache_policy_layers.swap(remaining_layers)" in drain_body
    assert "!pending_cache_policy_layers.empty()" in drain_body
    assert "pending_cache_policy_layer_mask[layer_idx] = 0" in drain_body


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



def test_precise_demand_does_not_restart_from_zero_when_partial_cache_line_was_evicted():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::send_one_job(CopyTask *task)")
    compact_body = _compact(body)

    assert "task->expert->gpu_data == nullptr" in body
    assert "task->start_mem_buf_idx > 0" in body
    assert "task->request_type == kCacheRequestDemand" in body
    assert "task->is_precise" in body
    assert "restart stale partial demand" not in body
    assert "task->start_mem_buf_idx = 0" not in body
    assert "task->stop_mem_buf_idx = metas->num_per_expert_param" not in body
    assert re.search(
        r"task->expert->gpu_data\s*==\s*nullptr.*"
        r"task->start_mem_buf_idx\s*>\s*0.*"
        r"return\s+false\s*;",
        compact_body,
    )


def test_stale_partial_task_does_not_restart_before_launch():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::send_one_job(CopyTask *task)")
    compact_body = _compact(body)

    assert "task->expert->num_ready > task->start_mem_buf_idx" in body
    assert "task->expert->num_ready < task->start_mem_buf_idx" in body
    stale_branch = body[body.index("task->expert->num_ready < task->start_mem_buf_idx"):]
    stale_branch = stale_branch[:stale_branch.index("} else {")]
    assert "task->start_mem_buf_idx = task->expert->num_ready" not in stale_branch
    assert re.search(
        r"task->expert->num_ready\s*<\s*task->start_mem_buf_idx.*"
        r"return\s+false\s*;",
        compact_body,
    )


def test_scheduler_marks_current_demand_experts_protected_before_enqueue():
    cpp = _text(PREFETCHER_CPP)

    one_body = _function_body(cpp, "void FetchScheduleWorker::preempt_one_expert")
    one_compact = _compact(one_body)
    assert "cache->mark_demand_protected(e)" in one_body
    assert one_compact.index("cache->mark_demand_protected(e)") < one_compact.index("cache->is_in_cache(e)")
    assert one_compact.index("cache->mark_demand_protected(e)") < one_compact.index("add_single_tasks_for_one_expert")
    assert one_compact.index("cache->mark_demand_protected(e)") < one_compact.index("e->expert_status.transfer(kReady, kLaunching, false)")

    layer_body = _function_body(cpp, "void FetchScheduleWorker::preempt_one_layer_without_reorder_")
    layer_compact = _compact(layer_body)
    assert "cache->mark_demand_protected(e)" in layer_body
    assert layer_compact.index("seen_demand_experts.insert") < layer_compact.index("cache->mark_demand_protected(e)")
    assert layer_compact.index("cache->mark_demand_protected(e)") < layer_compact.index("cache->is_in_cache(e)")
    assert layer_compact.index("cache->mark_demand_protected(e)") < layer_compact.index("add_single_tasks_for_one_expert")
    assert layer_compact.index("cache->mark_demand_protected(e)") < layer_compact.index("e->expert_status.transfer(kReady, kLaunching, false)")


def test_scheduler_clears_demand_protection_on_layer_advance_and_reset():
    cpp = _text(PREFETCHER_CPP)
    advance_body = _function_body(cpp, "void FetchScheduleWorker::advance_actual_layer")
    reset_body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(ResetTask *task)")
    compact_advance = _compact(advance_body)

    assert "cache->clear_demand_protected_for_layer(current_layer)" in advance_body
    assert "cache->clear_all_demand_protected()" in advance_body
    assert "cache->clear_all_demand_protected()" in reset_body
    assert compact_advance.index("cache->clear_demand_protected_for_layer(current_layer)") < compact_advance.index("current_layer = layer_idx")


def test_lru_policy_skips_busy_and_protected_victims_before_returning_no_victim():
    hpp = _text(CACHE_HPP)
    cpp = _text(CACHE_CPP)

    assert "std::unordered_set<ExpertHandler*> demand_protected_experts" in hpp
    for decl in [
        "ExpertHandler *select_for_evict(ExpertHandler *) override;",
        "void mark_demand_protected(ExpertHandler* expert) override;",
        "void clear_demand_protected(ExpertHandler* expert) override;",
        "void clear_demand_protected_for_layer(int layer_idx) override;",
        "void clear_all_demand_protected() override;",
        "bool is_demand_protected(ExpertHandler* expert) const override;",
    ]:
        assert decl in hpp

    body = _function_body(cpp, "ExpertHandler *CachePolicyLRU::select_for_evict(ExpertHandler *)")
    compact = _compact(body)
    for expected in [
        "node = linked_list.front()",
        "node != &linked_list.guard_tail",
        "cache->is_in_cache_ptr(e)",
        "status == kLaunching",
        "status == kUsing",
        "is_demand_protected(e)",
        "return e",
        "return nullptr",
    ]:
        assert expected in compact
    assert compact.index("status == kLaunching") < compact.index("return e")
    assert compact.index("status == kUsing") < compact.index("return e")
    assert compact.index("is_demand_protected(e)") < compact.index("return e")
    assert compact.index("return e") < compact.index("return nullptr")

    evict_body = _function_body(cpp, "void CachePolicyLRU::evict(ExpertHandler *e)")
    assert "demand_protected_experts.erase(e)" in evict_body

    clear_layer_body = _function_body(cpp, "void CachePolicyLRU::clear_demand_protected_for_layer(int layer_idx)")
    assert "expert->layer_idx == layer_idx" in clear_layer_body
    assert "clear_demand_protected(expert)" in clear_layer_body

def test_scheduler_aware_policy_tracks_protected_demand_experts_like_reclaimable():
    hpp = _text(CACHE_HPP)
    cpp = _text(CACHE_CPP)

    assert "demand_protected_map" in hpp
    assert "demand_protected_lru" in hpp
    assert "void mark_demand_protected(ExpertHandler* expert) override" in hpp
    assert "void clear_demand_protected(ExpertHandler* expert) override" in hpp
    assert "void clear_demand_protected_for_layer(int layer_idx) override" in hpp
    assert "void clear_all_demand_protected() override" in hpp
    assert "bool is_demand_protected(ExpertHandler* expert) const override" in hpp
    assert "protected_demand_experts" not in hpp

    mark_body = _function_body(cpp, "void CachePolicySchedulerAware::mark_demand_protected(ExpertHandler* expert)")
    clear_one_body = _function_body(cpp, "void CachePolicySchedulerAware::clear_demand_protected(ExpertHandler* expert)")
    clear_layer_body = _function_body(cpp, "void CachePolicySchedulerAware::clear_demand_protected_for_layer(int layer_idx)")
    clear_all_body = _function_body(cpp, "void CachePolicySchedulerAware::clear_all_demand_protected()")
    is_protected_body = _function_body(cpp, "bool CachePolicySchedulerAware::is_demand_protected(ExpertHandler* expert) const")
    evict_body = _function_body(cpp, "void CachePolicySchedulerAware::evict")

    assert "touch(demand_protected_map, demand_protected_lru, expert)" in mark_body
    assert "demand_protected_lru.remove" in clear_one_body
    assert "demand_protected_map.erase" in clear_one_body
    assert "clear_demand_protected(expert)" in clear_layer_body
    assert "demand_protected_map.clear()" in clear_all_body
    assert "demand_protected_map.find(expert)" in is_protected_body
    assert "erase_from(demand_protected_map, demand_protected_lru)" in evict_body


def test_cache_manager_forwards_protected_demand_operations_to_slot_policy():
    cpp = _text(CACHE_CPP)

    mark_body = _function_body(cpp, "void CacheMngr::mark_demand_protected(ExpertHandler* expert)")
    clear_one_body = _function_body(cpp, "void CacheMngr::clear_demand_protected(ExpertHandler* expert)")
    clear_layer_body = _function_body(cpp, "void CacheMngr::clear_demand_protected_for_layer(int layer_idx)")
    clear_all_body = _function_body(cpp, "void CacheMngr::clear_all_demand_protected()")
    is_protected_body = _function_body(cpp, "bool CacheMngr::is_demand_protected(ExpertHandler* expert) const")

    assert "cache_slots->to_slot(expert)->policy->mark_demand_protected(expert)" in mark_body
    assert "cache_slots->to_slot(expert)->policy->clear_demand_protected(expert)" in clear_one_body
    assert "cache_slots->to_slot(layer_idx)->policy->clear_demand_protected_for_layer(layer_idx)" in clear_layer_body
    assert "slot.policy->clear_all_demand_protected()" in clear_all_body
    assert "cache_slots->to_slot(expert)->policy->is_demand_protected(expert)" in is_protected_body
    assert "protected_demand_experts" not in mark_body + clear_one_body + clear_layer_body + clear_all_body + is_protected_body


def test_demand_expert_done_clears_protection_through_reclaimable_drain():
    hpp = _text(PREFETCHER_HPP)
    cpp = _text(PREFETCHER_CPP)

    assert "clear_demand_protection" in hpp
    assert "void enqueue_clear_demand_protection(int layer_idx, int expert_idx)" in hpp
    assert "void enqueue_expert_reclaimable(int layer_idx, int expert_idx, bool clear_demand_protection)" in hpp

    one_done_body = _function_body(cpp, "void PrefetchMngr::one_expert_done")
    one_done_compact = _compact(one_done_body)
    assert "enqueue_clear_demand_protection(layer_id, expert_id)" in one_done_body
    assert "enqueue_expert_reclaimable(layer_id, expert_id, false)" in one_done_body
    assert one_done_compact.index("enqueue_clear_demand_protection(layer_id, expert_id)") < one_done_compact.index("metas->is_encoder_layer(layer_id)")
    assert "cache->clear_demand_protected" not in one_done_body

    clear_enqueue_body = _function_body(cpp, "void FetchScheduleWorker::enqueue_clear_demand_protection")
    assert "metas->is_encoder_layer" not in clear_enqueue_body
    assert "pending.clear_demand_protection_mask[expert_idx] = 1" in clear_enqueue_body
    assert "pending.mode = PendingCachePolicyUpdate::kSomeExperts" not in clear_enqueue_body
    assert "cache->clear_demand_protected" not in clear_enqueue_body

    enqueue_body = _function_body(cpp, "void FetchScheduleWorker::enqueue_expert_reclaimable")
    assert "clear_demand_protection" in enqueue_body
    assert "pending.clear_demand_protection_mask[expert_idx] = 1" in enqueue_body
    assert "cache->clear_demand_protected" not in enqueue_body

    drain_body = _function_body(cpp, "void FetchScheduleWorker::drain_cache_policy_updates")
    assert "update.clear_demand_protection_mask" in drain_body
    assert "cache->clear_demand_protected(layer_idx, expert_idx)" in drain_body
    clear_pos = drain_body.index("cache->clear_demand_protected(layer_idx, expert_idx)")
    mark_pos = drain_body.index("cache->mark_reclaimable(layer_idx, expert_idx)")
    assert clear_pos < mark_pos


def test_no_victim_precise_retry_drains_before_requeue_and_uses_conditional_log():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::send_one_job(CopyTask *task)")
    retry_pos = body.index("retry precise demand without legal victim")
    drain_pos = body.rindex("drain_cache_policy_updates", 0, retry_pos)
    requeue_pos = body.index("requeue_precise_task_front(task)", retry_pos)
    assert drain_pos < retry_pos < requeue_pos
    assert "if (log_prefetch_decision_enabled() || log_demand_fetch_enabled())" in body[drain_pos:requeue_pos]


def test_scheduler_aware_victim_selection_skips_protected_demand_experts():
    cpp = _text(CACHE_CPP)
    body = _function_body(cpp, "ExpertHandler* CachePolicySchedulerAware::first_loaded_candidate")
    compact = _compact(body)

    assert "is_demand_protected(expert)" in body
    assert re.search(
        r"if\s*\(\s*is_demand_protected\s*\(\s*expert\s*\)\s*\)\s*\{[^{}]*continue\s*;",
        compact,
    )
    protected_pos = compact.index("is_demand_protected(expert)")
    continue_pos = compact.index("continue", protected_pos)
    return_pos = compact.index("return expert")
    assert protected_pos < continue_pos < return_pos


def test_precise_demand_without_legal_victim_retries_instead_of_fatal():
    cpp = _text(CACHE_CPP)
    select_body = _function_body(
        cpp,
        "ExpertHandler* CachePolicySchedulerAware::select_for_evict(\n"
        "    ExpertHandler* incoming,\n"
        "    CacheRequestType request_type)",
    )
    select_compact = _compact(select_body)

    reclaimable_only_pos = select_compact.index("is_reclaimable_only_request(request_type, cache->metas.get())")
    encoder_pos = select_compact.index("first_loaded_candidate(encoder_map, encoder_lru)", reclaimable_only_pos)
    decoder_pos = select_compact.index("first_loaded_candidate(decoder_map, decoder_lru)", encoder_pos)
    global_pos = select_compact.index("first_loaded_candidate(global_map, global_lru)", decoder_pos)
    demand_pos = select_compact.index("request_type == kCacheRequestDemand", global_pos)
    null_return_pos = select_compact.index("return nullptr", demand_pos)
    check_pos = select_compact.index("CHECK(false)", null_return_pos)
    assert reclaimable_only_pos < encoder_pos < decoder_pos < global_pos < demand_pos < null_return_pos < check_pos

    miss_body = _function_body(
        cpp,
        "CacheMngr::CacheLineOccupancyWaiter CacheMngr::miss(\n"
        "    ExpertHandler *incoming_e,\n"
        "    bool is_precise,\n"
        "    CacheRequestType request_type)",
    )
    compact = _compact(miss_body)

    demand_precise_branch = re.search(
        r"if\s*\(\s*(?:"
        r"request_type\s*==\s*kCacheRequestDemand\s*&&\s*is_precise|"
        r"is_precise\s*&&\s*request_type\s*==\s*kCacheRequestDemand"
        r")\s*\)",
        compact,
    )
    assert demand_precise_branch
    assert "demand miss has no legal victim" in miss_body
    assert "CHECK(is_reclaimable_only_request(request_type, metas.get())" in miss_body
    assert demand_precise_branch.start() < compact.index("CHECK(is_reclaimable_only_request(request_type, metas.get())")

    prefetcher_cpp = _text(PREFETCHER_CPP)
    send_body = _function_body(prefetcher_cpp, "bool FetchScheduleWorker::send_one_job(CopyTask *task)")
    send_compact = _compact(send_body)

    assert "requeue_precise_task_front(task)" in send_body
    assert re.search(
        r"cache_miss\(task->expert,\s*task->is_precise,\s*task->request_type\).*"
        r"task->request_type\s*==\s*kCacheRequestDemand.*"
        r"task->is_precise.*"
        r"task->expert->gpu_data\s*==\s*nullptr.*"
        r"requeue_precise_task_front\(task\).*"
        r"return\s+false\s*;",
        send_compact,
    )
    assert send_compact.index("requeue_precise_task_front(task)") < send_compact.index("task->expert->expert_status.transfer(kIdle, kFetching)")

def test_enable_encoder_reclaim_gates_encoder_reclaim_updates():
    cpp = _text(PREFETCHER_CPP)

    report_body = _function_body(cpp, "void PrefetchMngr::report_one_layer(int layer_id, int64_t* experts, int64_t num_expert)")
    assert "metas->enable_encoder_reclaim" in report_body
    assert "enqueue_layer_reclaimable_except" in report_body
    assert report_body.index("metas->enable_encoder_reclaim") < report_body.index("enqueue_layer_reclaimable_except")

    layer_done_body = _function_body(cpp, "void PrefetchMngr::one_moe_layer_done")
    assert "metas->enable_encoder_reclaim" in layer_done_body
    assert "enqueue_layer_reclaimable(layer_id)" in layer_done_body
    assert layer_done_body.index("metas->enable_encoder_reclaim") < layer_done_body.index("enqueue_layer_reclaimable(layer_id)")

    expert_done_body = _function_body(cpp, "void PrefetchMngr::one_expert_done")
    assert "metas->enable_encoder_reclaim" in expert_done_body
    assert "enqueue_expert_reclaimable" in expert_done_body
    assert expert_done_body.index("metas->enable_encoder_reclaim") < expert_done_body.index("enqueue_expert_reclaimable")

    drain_body = _function_body(cpp, "void FetchScheduleWorker::drain_cache_policy_updates")
    assert "if (!metas->enable_encoder_reclaim)" in drain_body
    assert "cache->mark_reclaimable" in drain_body


def test_disable_encoder_reclaim_disables_reclaimable_scheduler_gates():
    cpp = _text(PREFETCHER_CPP)

    requires_body = _function_body(cpp, "bool FetchScheduleWorker::requires_reclaimable_encoder")
    assert "!metas->enable_encoder_reclaim" in requires_body
    assert "return false" in requires_body
    assert requires_body.index("!metas->enable_encoder_reclaim") < requires_body.index("cls == PrefetchClass::kEncoderPredictor")

    jit_body = _function_body(cpp, "bool FetchScheduleWorker::encoder_jit_can_dispatch")
    assert "!metas->enable_encoder_reclaim" in jit_body
    assert "return true" in jit_body
    assert jit_body.index("!metas->enable_encoder_reclaim") < jit_body.index("cache->has_reclaimable_encoder()")


def test_disable_encoder_reclaim_uses_global_lru_without_stage_fallback():
    cpp = _text(CACHE_CPP)
    body = _function_body(
        cpp,
        "ExpertHandler* CachePolicySchedulerAware::select_for_evict(\n"
        "    ExpertHandler* incoming,\n"
        "    CacheRequestType request_type)",
    )
    compact = _compact(body)

    assert "!cache->metas->enable_encoder_reclaim" in body
    no_reclaim_pos = compact.index("!cache->metas->enable_encoder_reclaim")
    global_pos = compact.index("first_loaded_candidate(global_map, global_lru)", no_reclaim_pos)
    no_reclaim_null_pos = compact.index("return nullptr", global_pos)
    encoder_pos = compact.index("first_loaded_candidate(encoder_map, encoder_lru)")
    decoder_pos = compact.index("first_loaded_candidate(decoder_map, decoder_lru)")
    assert no_reclaim_pos < global_pos < no_reclaim_null_pos < encoder_pos < decoder_pos

def test_non_precise_prefetch_without_victim_returns_before_fetching():
    cpp = _text(PREFETCHER_CPP)
    body = _function_body(cpp, "bool FetchScheduleWorker::send_one_job(CopyTask *task)")
    compact = _compact(body)

    match = re.search(
        r"if\s*\(\s*!task->is_precise\s*&&\s*task->expert->gpu_data\s*==\s*nullptr\s*\)",
        body,
    )
    assert match
    no_victim_pos = match.start()
    return_pos = body.index("return false", no_victim_pos)
    fetching_pos = body.index("task->expert->expert_status.transfer(kIdle, kFetching)")
    assert no_victim_pos < return_pos < fetching_pos
    assert "skip prefetch request without victim" in body

