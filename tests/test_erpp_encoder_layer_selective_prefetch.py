from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ERPP_HPP = REPO_ROOT / "src/cpp_worker/erpp_encoder_predictor.hpp"
ERPP_CPP = REPO_ROOT / "src/cpp_worker/erpp_encoder_predictor.cpp"
PREFETCHER_CPP = REPO_ROOT / "src/cpp_worker/prefetcher.cpp"
WORKER_CPP = REPO_ROOT / "src/cpp_worker/worker.cpp"
PREFETCHER_HPP = REPO_ROOT / "src/cpp_worker/prefetcher.hpp"
CACHE_HPP = REPO_ROOT / "src/cpp_worker/cache.hpp"
CACHE_CPP = REPO_ROOT / "src/cpp_worker/cache.cpp"
RUN_SWITCH = REPO_ROOT / "performance/run_switch_mmlu_validation.sh"
COMPARE_RUN = REPO_ROOT / "performance_baseline/compare/run.sh"
OUR_RUN_ALL = REPO_ROOT / "experiment/baseline/LSP/our/run_all.sh"


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


def test_erpp_predictor_declares_layer_selection_api():
  hpp = _text(ERPP_HPP)

  assert "std::vector<uint8_t> enabled_layers" in hpp
  assert "std::vector<uint8_t> parse_enabled_layers(const std::string& spec) const" in hpp
  assert "bool should_prefetch_layer(int layer_idx) const" in hpp


def test_erpp_layer_parser_supports_all_positive_and_negative_ids():
  cpp = _text(ERPP_CPP)
  body = _function_body(
    cpp,
    "std::vector<uint8_t> ErppEncoderPredictor::parse_enabled_layers",
  )
  compact = _compact(body)

  assert 'normalized == "all"' in body
  assert "std::getline(ss, item, ',')" in body
  assert "std::stoi(item" in body
  assert "layer_idx < 0" in body
  assert "layer_idx += metas->num_encoder_moe_layer" in body
  assert "enabled[layer_idx]" in body
  assert "duplicate ERPP encoder layer" in body
  assert "ERPP encoder layer out of range" in body
  assert "empty ERPP encoder layer entry" in body
  assert "normalized.back() != ','" in body
  assert "all" in compact and "cannot be mixed" in compact

def test_erpp_layer_parser_supports_non_first_alias():
  cpp = _text(ERPP_CPP)
  body = _function_body(
    cpp,
    "std::vector<uint8_t> ErppEncoderPredictor::parse_enabled_layers",
  )
  compact = _compact(body)

  assert 'normalized == "non_first"' in body
  assert "std::fill(enabled.begin() + 1, enabled.end(), 1)" in compact
  assert "enabled.size() > 1" in body


def test_scheduler_jit_refill_layer_parser_supports_non_first_alias():
  hpp = _text(PREFETCHER_HPP)
  body = _function_body(hpp, "void initialize_encoder_jit_enabled_layer_mask()")
  compact = _compact(body)

  assert 'spec == "non_first"' in body
  assert "std::fill(encoder_jit_enabled_layer_mask.begin() + 1," in compact
  assert "encoder_jit_enabled_layer_mask.end(), 1)" in compact


def test_our_run_all_defaults_erpp_encoder_layers_to_non_first():
  text = _text(OUR_RUN_ALL)

  assert "ERPP_ENCODER_LAYERS=" in text
  assert ":-non_first}" in text
  assert ":-2,3,4,5}" not in text
  assert "--erpp_encoder_layers" in text
  assert "ERPP_ENCODER_LAYERS" in text


def test_our_run_all_defaults_jit_refill_layers_to_non_first():
  text = _text(OUR_RUN_ALL)

  assert "ERPP_ENCODER_JIT_REFILL_LAYERS=" in text
  assert "ERPP_ENCODER_JIT_REFILL_LAYERS:-non_first" in text
  assert "ERPP_ENCODER_JIT_REFILL_WINDOW:-1" in text
  assert "ENABLE_ERPP_ENCODER_JIT_TOPK_COVER:-True" in text
  assert "--erpp_encoder_jit_refill_layers" in text
  assert "--enable_erpp_encoder_jit_topk_cover" in text


def test_our_run_all_does_not_pass_unsupported_erpp_timing_args():
  text = _text(OUR_RUN_ALL)

  assert "--erpp_encoder_expert_copy_us" not in text
  assert "--erpp_encoder_expert_compute_us" not in text


def test_our_run_all_uses_hidden_only_switch_base_128_ble_predictor():
  text = _text(OUR_RUN_ALL)

  assert "performance_predictor/encoder/ERPP/implement/model/sida-gru-sa-hard-ce/erpp_encoder_predictor_v2.ts" not in text
  assert "switch-base-128/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim/encoder_predictor_ble.ts" in text


def test_our_run_all_default_sweep_uses_all_explicit_gpu_profiles_without_default():
  text = _text(OUR_RUN_ALL)
  gpu_line = next(line for line in text.splitlines() if line.startswith("GPU_CONFIGS="))
  model_line = next(line for line in text.splitlines() if line.startswith("MODELS="))

  assert ":-default" not in gpu_line
  for profile in ("gpu4gb", "gpu8gb", "gpu12gb", "gpu16gb", "gpu24gb", "gpu32gb", "gpu40gb", "gpu48gb"):
    assert profile in gpu_line
  for model in ("switch-base-128", "switch-base-256", "switch-large-128", "nllb"):
    assert model in model_line


def test_cache_request_type_uses_explicit_predictor_prefetch_names():
  hpp = _text(CACHE_HPP)
  compact = _compact(hpp)

  assert "kCacheRequestEncoderPredictorPrefetch" in hpp
  assert "kCacheRequestDecoderPredictorPrefetch" in hpp
  assert "kCacheRequestDecoderWarmupPrefetch" in hpp
  assert "kCacheRequestInitialLoad" in hpp
  assert (
    "kCacheRequestDemand = 0, kCacheRequestEncoderPredictorPrefetch, "
    "kCacheRequestEncoderJitRefill, kCacheRequestDecoderPredictorPrefetch, "
    "kCacheRequestDecoderWarmupPrefetch, kCacheRequestInitialLoad"
  ) in compact


def test_cache_request_type_includes_encoder_jit_refill():
  hpp = _text(CACHE_HPP)
  compact = _compact(hpp)

  assert "kCacheRequestEncoderJitRefill" in hpp
  assert (
    "kCacheRequestDemand = 0, kCacheRequestEncoderPredictorPrefetch, "
    "kCacheRequestEncoderJitRefill, kCacheRequestDecoderPredictorPrefetch, "
    "kCacheRequestDecoderWarmupPrefetch, kCacheRequestInitialLoad"
  ) in compact


def test_initial_cache_load_uses_initial_load_request_type():
  cpp = _text(CACHE_CPP)
  body = _function_body(cpp, "void CacheMngr::load_initial_plan_sync")

  assert "kCacheRequestInitialLoad" in body
  assert "miss(expert, false, kCacheRequestDecoderPredictorPrefetch)" not in body


def test_worker_labels_encoder_jit_refill_request():
  cpp = _text(WORKER_CPP)
  body = _function_body(cpp, "const char* cache_request_label")

  assert 'case kCacheRequestEncoderJitRefill: return "encoder_jit_refill";' in body


def test_erpp_predictor_declares_jit_budget_and_floor_helpers():
  hpp = _text(ERPP_HPP)

  assert "int encoder_budget_for_layer(int layer_idx) const" in hpp
  assert "int encoder_jit_floor() const" in hpp
  assert "int encoder_jit_ranking_limit(int layer_idx) const" in hpp


def test_erpp_predictor_skips_topk_and_predicted_log_for_disabled_layers():
  cpp = _text(ERPP_CPP)
  body = _function_body(cpp, "ErppEncoderPrediction ErppEncoderPredictor::predict_from_cpu_tensors")
  compact = _compact(body)

  assert "!should_prefetch_layer(layer)" in body
  assert "result.rankings.emplace_back()" in body
  assert "result.budgets.push_back(0)" in body
  assert compact.index("!should_prefetch_layer(layer)") < compact.index(".topk(limit, -1, true, true)")
  disabled_block = body[body.index("!should_prefetch_layer(layer)"):body.index(".topk(limit, -1, true, true)")]
  assert "erpp_encoder_prefetch: predicted layer L" not in disabled_block
  assert "array_to_str" not in disabled_block


def test_erpp_predictor_is_hidden_only_and_does_not_use_attention_mask():
  hpp = _text(ERPP_HPP)
  cpp = _text(ERPP_CPP)
  load_body = _function_body(cpp, "void ErppEncoderPredictor::load_model_from")
  record_body = _function_body(cpp, "void ErppEncoderPredictor::record_encoder_layer0")
  predict_body = _function_body(cpp, "ErppEncoderPrediction ErppEncoderPredictor::predict_from_cpu_tensors")
  compact_predict = _compact(predict_body)

  assert "forward_accepts_attention_mask" not in hpp
  assert "attention_mask_buffer" not in hpp
  assert "normalize_attention_mask" not in hpp
  assert 'model.get_method("forward")' in load_body
  assert "getSchema()" in load_body
  assert "arguments().size()" in load_body
  assert "forward_arg_count == 2" in load_body
  assert "forward(hidden)" in load_body
  assert "attention_mask_buffer" not in record_body
  assert "attention_mask_d2h" not in record_body
  assert "normalize_attention_mask" not in predict_body
  assert "inputs.push_back(mask)" not in predict_body
  assert "std::vector<torch::jit::IValue> inputs{hidden_float_cpu}" in predict_body
  assert compact_predict.index("inputs{hidden_float_cpu}") < compact_predict.index("model.forward(inputs)")


def test_erpp_worker_sets_zero_jit_budget_for_disabled_layers():
  cpp = _text(WORKER_CPP)
  body = _function_body(cpp, "void ErppEncoderPredictWorker::do_one_task_impl")
  compact = _compact(body)

  assert "auto prediction = erpp_encoder_predictor->predict_recorded();" in body
  assert "auto& predictions = prediction.rankings;" in body
  assert "auto& budgets = prediction.budgets;" in body
  assert "task.rankings = predictions;" in body
  assert "task.budgets = budgets;" in body
  jit_body = body[body.index("if (metas->enable_erpp_encoder_jit_refill)"):body.index("// Not JIT refill")]
  assert "encoder_budget_for_layer(layer_idx)" not in jit_body
  assert "task.budgets.push_back" not in jit_body
  assert "const int budget = layer_idx < static_cast<int>(budgets.size()) ? budgets[layer_idx] : 0;" in body
  assert "task.num_expert = std::min<size_t>(experts.size(), static_cast<size_t>(std::max(0, budget)));" in body
  assert "task.num_expert == 0" in body
  assert compact.index("task.num_expert = std::min<size_t>") < compact.index("array_to_str(task.expert_idxs, task.num_expert)")


def test_erpp_predictor_uses_max_floor_budget_ranking_limit_in_jit_mode():
  cpp = _text(ERPP_CPP)
  legacy_limit_body = _function_body(cpp, "int ErppEncoderPredictor::encoder_jit_ranking_limit(int layer_idx) const")
  limit_body = _function_body(cpp, "int ErppEncoderPredictor::encoder_jit_ranking_limit(int layer_idx, int budget) const")
  floor_body = _function_body(cpp, "int ErppEncoderPredictor::encoder_jit_floor")
  predict_body = _function_body(cpp, "ErppEncoderPrediction ErppEncoderPredictor::predict_from_cpu_tensors")
  compact_floor = _compact(floor_body)
  compact_predict = _compact(predict_body)

  assert "encoder_budget_for_layer(layer_idx)" in legacy_limit_body
  assert "encoder_jit_floor()" in limit_body
  assert "std::max" in limit_body
  assert "std::min" in limit_body
  assert "metas->num_expert" in limit_body
  assert 'metas->initial_cache_policy == "fixed"' in floor_body
  assert "metas->initial_layer_budgets" in floor_body
  assert "std::floor(metas->cache_rate * metas->num_layer * metas->num_expert)" in compact_floor
  assert "/ metas->num_encoder_moe_layer" in compact_floor
  assert "std::clamp" in floor_body
  assert "const int limit = encoder_jit_ranking_limit(layer, budget);" in predict_body
  assert ".topk(limit, -1, true, true)" in predict_body
  assert "budget=" in predict_body
  assert "ranking_limit=" in predict_body
  assert "array_to_str(data, limit)" in predict_body
  assert "result.rankings.emplace_back(data, data + limit)" in predict_body
  assert ".topk(budgets[layer], -1, true, true)" not in compact_predict


def test_prefetch_scheduler_declares_class_aware_queue_set():
  hpp = _text(PREFETCHER_HPP)

  assert "enum class PrefetchClass" in hpp
  assert "kEncoderPredictor" in hpp
  assert "kDecoderPredictor" in hpp
  assert "kDecoderWarmup" in hpp
  assert "struct PrefetchQueueSet" in hpp
  assert "std::vector<TaskQueue> encoder_predictor_by_layer" in hpp
  assert "std::vector<TaskQueue> decoder_predictor_by_layer" in hpp
  assert "TaskQueue decoder_warmup_plan_queue" in hpp
  assert "PrefetchQueueSet prefetch_queues" in hpp
  assert "std::vector<TaskQueue> per_layer_job_queues" not in hpp
  assert "TaskQueue encoder_prefetch_queue" not in hpp
  assert "DecoderWarmupEntry" not in hpp
  assert "std::queue<DecoderWarmupEntry> decoder_warmup_queue" not in hpp
  assert "pop_next_normal_prefetch" not in hpp
  assert "pop_next_encoder_prefetch" not in hpp
  assert "pop_next_decoder_warmup" not in hpp


def test_prefetch_class_policy_helpers_are_declared():
  hpp = _text(PREFETCHER_HPP)

  assert "bool pop_next_prefetch_for_class(PrefetchClass cls, CopyTask& task)" in hpp
  assert "bool has_pending_prefetch_for_class(PrefetchClass cls)" in hpp
  assert "void prune_prefetch_class(PrefetchClass cls)" in hpp
  assert "bool requires_encoder_phase(PrefetchClass cls) const" in hpp
  assert "bool requires_reclaimable_encoder(PrefetchClass cls) const" in hpp
  assert "bool blocks_lower_priority_when_pending(PrefetchClass cls) const" in hpp
  assert "bool replace_same_layer_on_enqueue(PrefetchClass cls) const" in hpp


def test_scheduler_aware_encoder_prefetch_is_reclaimable_only():
  cpp = _text(CACHE_CPP)
  body = _function_body(
    cpp,
    "ExpertHandler* CachePolicySchedulerAware::select_for_evict(\n"
    "    ExpertHandler* incoming,\n"
    "    CacheRequestType request_type)",
  )
  compact = _compact(body)

  assert "cache->metas->enable_encoder_reclaim" in body
  assert "is_reclaimable_only_request(request_type" in body
  assert "return nullptr" in body
  reclaimable_idx = compact.index("first_loaded_candidate(reclaimable_map, reclaimable_encoder_lru)")
  helper_idx = compact.index("is_reclaimable_only_request(request_type, cache->metas.get())")
  encoder_fallback_idx = compact.index("first_loaded_candidate(encoder_map, encoder_lru)")
  assert reclaimable_idx < helper_idx
  assert helper_idx < encoder_fallback_idx


def test_scheduler_aware_single_arg_evict_hard_fails():
  cpp = _text(CACHE_CPP)
  body = _function_body(
    cpp,
    "ExpertHandler* CachePolicySchedulerAware::select_for_evict(ExpertHandler* incoming)",
  )

  assert "requires CacheRequestType" in body
  assert "kCacheRequestDecoderPredictorPrefetch" not in body


def test_cache_access_legacy_entrypoint_hard_fails():
  cpp = _text(CACHE_CPP)
  body = _function_body(
    cpp,
    "void CacheMngr::access(ExpertHandler *expert, bool is_precise)",
  )

  assert "requires explicit hit/miss" in body
  assert "miss(expert, is_precise)" not in body


def test_cache_two_arg_miss_hard_fails():
  cpp = _text(CACHE_CPP)
  body = _function_body(
    cpp,
    "CacheMngr::CacheLineOccupancyWaiter CacheMngr::miss(ExpertHandler *incoming_e, bool is_precise)",
  )

  assert "requires CacheRequestType" in body
  assert "kCacheRequestDecoderPredictorPrefetch" not in body


def test_initial_plan_uses_explicit_cache_request_type():
  cpp = _text(CACHE_CPP)
  body = _function_body(
    cpp,
    "void CacheMngr::load_initial_plan_sync(cudaStream_t stream)",
  )

  assert "miss(expert, false, kCacheRequestInitialLoad)" in body
  assert "miss(expert, false, kCacheRequestDecoderPredictorPrefetch)" not in body
  assert "miss(expert, false);" not in body


def test_cache_miss_skips_encoder_prefetch_without_reclaimable_victim():
  cpp = _text(CACHE_CPP)
  signature = (
    "CacheMngr::CacheLineOccupancyWaiter CacheMngr::miss(\n"
    "    ExpertHandler *incoming_e,\n"
    "    bool is_precise,\n"
    "    CacheRequestType request_type)"
  )
  start = cpp.index(signature)
  end = cpp.index("void CacheMngr::mark_reclaimable", start)
  body = cpp[start:end]

  assert "is_reclaimable_only_request(request_type, metas.get())" in body
  assert "may skip eviction" in body

def test_prefetch_layer_task_carries_request_type():
  hpp = _text(PREFETCHER_HPP)
  body = _function_body(hpp, "class PrefetchLayerTask : public FetchScheduleTaskBase")

  assert "CacheRequestType request_type = kCacheRequestDecoderPredictorPrefetch" in body


def test_prefetch_layer_task_passes_request_type_to_copy_tasks():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(PrefetchLayerTask *task)")

  assert "task->request_type" in body
  assert "kCacheRequestDecoderPredictorPrefetch" not in body
  assert "add_separate_tasks_for_one_expert" in body
  assert "add_single_tasks_for_one_expert" in body

def test_erpp_predictor_declares_record_lifecycle_api():
  hpp = _text(ERPP_HPP)

  assert "struct ErppRecordedInputHandle" in hpp
  assert "ErppRecordedInputHandle record_encoder_layer0(" in hpp
  assert "std::vector<std::vector<int64_t>> predict_recorded(" in hpp
  assert "void reset_sequence_state()" in hpp
  assert "struct ErppRecordedInput" in hpp
  assert "cudaEvent_t ready_event" in hpp
  assert "torch::Tensor hidden_cpu" in hpp
  assert "torch::Tensor attention_mask_cpu" in hpp


def test_erpp_report_records_input_and_enqueues_lightweight_predict_job():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "void PrefetchMngr::report_erpp_encoder_layer0")

  assert "erpp_encoder_predictor->record_encoder_layer0(" in body
  assert "erpp_encoder_predict_thread->add_predict_job(handle)" in body
  assert "ErppEncoderPredictJob job" not in body
  assert "erpp_encoder_predict_thread->add_one_task" not in body
  assert "erpp_encoder_predict_thread->enqueue(" not in body
  assert "erpp_encoder_predictor->predict" not in body
  assert "cudaEventCreateWithFlags" not in body
  assert "cudaEventRecord" not in body
  assert "kCacheRequestEncoderPredictorPrefetch" not in body


def test_generate_reset_resets_erpp_predictor_state_after_workers_are_idle():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "void PrefetchMngr::reset_for_generate")
  compact = _compact(body)

  assert "erpp_encoder_predict_thread->begin_reset_for_generate()" in body
  assert "erpp_encoder_predict_thread->wait_until_idle()" in body
  assert "erpp_encoder_predictor->reset_sequence_state()" in body
  assert compact.index("erpp_encoder_predict_thread->wait_until_idle()") < compact.index(
      "erpp_encoder_predictor->reset_sequence_state()"
  )
  assert compact.index("erpp_encoder_predictor->reset_sequence_state()") < compact.index(
      "cache->reset_cache_contents()"
  )


def test_erpp_worker_job_no_longer_owns_tensors_or_cuda_events():
  hpp = _text(REPO_ROOT / "src/cpp_worker/worker.hpp")
  job_body = _function_body(hpp, "struct ErppEncoderPredictJob")
  worker_body = _function_body(hpp, "class ErppEncoderPredictWorker : public WorkerThread<ErppEncoderPredictJob>")

  assert "int64_t record_id" in job_body
  assert "int64_t forward_epoch" in job_body
  assert "int64_t generate_epoch" in job_body
  assert "torch::Tensor" not in job_body
  assert "cudaEvent_t" not in job_body
  assert "void enqueue(" not in worker_body
  assert "add_predict_job" in worker_body
  assert "reset_requested" not in worker_body
  assert "enqueue_inflight" not in worker_body
  assert "reset_mutex" not in worker_body


def test_erpp_worker_uses_recorded_prediction_and_encoder_prefetch_request_type():
  cpp = _text(WORKER_CPP)
  body = _function_body(cpp, "void ErppEncoderPredictWorker::do_one_task_impl")

  assert "erpp_encoder_predictor->predict_recorded(handle)" in body
  assert "erpp_encoder_predictor->predict(job.hidden, job.attention_mask)" not in body
  assert "cudaEventSynchronize(job.ready_event)" not in body
  assert "cudaEventDestroy(job.ready_event)" not in body
  assert "should_prefetch_layer(layer_idx)" in body
  assert "task.request_type = kCacheRequestEncoderPredictorPrefetch" in body
  assert "NVTX_RANGE(\"erpp/submit_encoder_prefetch" in body


def test_erpp_observability_uses_ranges_for_record_predict_and_scheduler_submit():
  erpp_cpp = _text(ERPP_CPP)
  worker_cpp = _text(WORKER_CPP)

  assert "NVTX_RANGE(\"erpp/record_encoder_layer0" in erpp_cpp
  assert "NVTX_RANGE(\"erpp/wait_record_event" in erpp_cpp
  assert "NVTX_RANGE(\"erpp/predict_forward" in erpp_cpp
  assert "NVTX_RANGE(\"erpp/submit_encoder_prefetch" in worker_cpp

  for label in (
      "erpp/record_encoder_layer0",
      "erpp/wait_record_event",
      "erpp/predict_forward",
      "erpp/submit_encoder_prefetch",
  ):
    assert f"NVTX_MARK(\"{label}" not in erpp_cpp + worker_cpp
    assert f"NVTX_DETAIL_MARK(\"{label}" not in erpp_cpp + worker_cpp



def test_erpp_encoder_prefetch_logs_full_runtime_pipeline():
  erpp_cpp = _text(ERPP_CPP)
  worker_cpp = _text(WORKER_CPP)
  prefetcher_cpp = _text(PREFETCHER_CPP)

  assert "erpp_encoder_prefetch: predicted layer L" in erpp_cpp
  assert "erpp_encoder_prefetch: submit scheduler layer L" in worker_cpp
  assert "experts=[" in worker_cpp
  assert "prefetch_decision: enqueue ERPP encoder prefetch L" in prefetcher_cpp
  assert "prefetch_decision: dispatch encoder_predictor_prefetch" in prefetcher_cpp
  assert "fetcher: start encoder_predictor_prefetch" in worker_cpp
  assert "fetcher: done encoder_predictor_prefetch" in worker_cpp
  assert "cache_miss: reclaimable-only request has no victim" in prefetcher_cpp + _text(CACHE_CPP)

def test_send_one_job_skips_encoder_prefetch_when_cache_miss_cannot_allocate():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "bool FetchScheduleWorker::send_one_job")
  compact = _compact(body)

  assert "task->request_type == kCacheRequestEncoderPredictorPrefetch" in body
  assert "task->request_type == kCacheRequestDecoderWarmupPrefetch" in body
  assert "task->expert->gpu_data == nullptr" in body
  assert "return false" in body
  cache_miss_idx = compact.index("cache_miss(task->expert, task->is_precise, task->request_type)")
  encoder_skip_idx = compact.index("task->request_type == kCacheRequestEncoderPredictorPrefetch", cache_miss_idx)
  assert cache_miss_idx < encoder_skip_idx


def test_send_one_job_drops_later_reclaimable_only_chunks_without_cache_line():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "bool FetchScheduleWorker::send_one_job")
  compact = _compact(body)

  assert "task->start_mem_buf_idx > 0" in body
  assert "task->request_type == kCacheRequestEncoderPredictorPrefetch" in body
  assert "task->request_type == kCacheRequestDecoderWarmupPrefetch" in body
  assert compact.index("task->start_mem_buf_idx > 0") < compact.index("CHECK(task->start_mem_buf_idx == 0)")


def test_encoder_predictor_prefetch_has_reclaimable_gate():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "bool FetchScheduleWorker::pop_next_prefetch_for_class")

  assert "PrefetchClass::kEncoderPredictor" in body
  assert "requires_reclaimable_encoder(cls)" in body
  assert "cache->has_reclaimable_encoder()" in body
  assert "blocked_on_encoder_predictor_prefetch_reclaimable = true" in body
  assert "sched/wait_encoder_predictor_prefetch_reclaimable" in body


def test_prefetch_layer_task_routes_predictor_classes_to_symmetric_queues():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(PrefetchLayerTask *task)")
  compact = _compact(body)

  assert "prefetch_class_for_request(task->request_type)" in body
  assert "queue_for_class_and_layer(cls, task->layer_idx)" in body
  assert "encoder_predictor_by_layer" in cpp
  assert "decoder_predictor_by_layer" in cpp
  assert "add_single_tasks_for_one_expert(task->layer_idx, task->expert_idxs[i], target_queue" in compact


def test_erpp_encoder_prefetch_prunes_before_waiting_for_reclaimable():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "bool FetchScheduleWorker::pop_next_prefetch_for_class")
  compact = _compact(body)

  assert "prune_prefetch_class(cls)" in body
  assert compact.index("prune_prefetch_class(cls)") < compact.index("cache->has_reclaimable_encoder()")
  assert 'NVTX_DETAIL_MARK("sched/wait_encoder_predictor_prefetch_reclaimable")' in body


def test_erpp_encoder_prefetch_prune_keeps_later_chunks_for_missing_expert():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "void FetchScheduleWorker::prune_prefetch_class")

  assert "task.expert->num_ready >= task.stop_mem_buf_idx" in body
  assert "task.expert->gpu_data == nullptr && task.start_mem_buf_idx > 0" not in body


def test_erpp_encoder_prefetch_wait_keeps_original_polling_behavior():
  cpp = _text(PREFETCHER_CPP)
  pop_body = _function_body(cpp, "bool FetchScheduleWorker::pop_next_prefetch_for_class")
  idle_body = _function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(IdleTask *idle_task)")

  assert "blocked_on_encoder_predictor_prefetch_reclaimable = true" in pop_body
  assert "blocked_on_encoder_predictor_prefetch_reclaimable = false" in cpp
  assert "usleep(50)" not in idle_body
  assert "this->add_one_task(&this->idle_task)" in idle_body



def test_prefetch_scan_decision_log_is_rate_limited():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "void FetchScheduleWorker::pop_next_task")

  assert "bool should_log_prefetch_scan_decision()" in cpp
  assert "kScanLogEveryCalls = 10000" in cpp
  assert "std::atomic<int64_t> scan_log_counter" in cpp
  assert "log_prefetch_decision_enabled() && should_log_prefetch_scan_decision()" in body
  assert 'LOG(INFO) << "prefetch_decision: scan phase="' in body


def test_scheduler_prioritizes_erpp_encoder_prefetch_before_other_prefetches():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "void FetchScheduleWorker::pop_next_task")
  compact = _compact(body)

  erpp_idx = compact.index("pop_next_prefetch_for_class(PrefetchClass::kEncoderPredictor, task)")
  blocked_idx = compact.index("blocked_on_encoder_predictor_prefetch_reclaimable", erpp_idx)
  decoder_idx = compact.index("pop_next_prefetch_for_class(PrefetchClass::kDecoderPredictor, task)")
  warmup_idx = compact.index("pop_next_prefetch_for_class(PrefetchClass::kDecoderWarmup, task)")

  assert compact.index("precise_job_queue.front()") < erpp_idx
  assert erpp_idx < blocked_idx
  assert blocked_idx < decoder_idx
  assert decoder_idx < warmup_idx


def test_prefetch_nvtx_labels_use_symmetric_predictor_names():
  cpp = _text(PREFETCHER_CPP)

  assert 'request_label = "encoder_predictor_prefetch"' in cpp
  assert 'request_label = "decoder_predictor_prefetch"' in cpp
  assert 'request_label = "decoder_warmup_prefetch"' in cpp
  assert '"sched/enqueue_encoder_predictor_prefetch"' in cpp
  assert '"sched/pop_encoder_predictor_prefetch"' in cpp
  assert '"sched/wait_encoder_predictor_prefetch_reclaimable"' in cpp
  assert '"sched/drop_stale_encoder_predictor_prefetch"' in cpp
  assert '"sched/enqueue_decoder_predictor_prefetch"' in cpp
  assert '"sched/pop_decoder_predictor_prefetch"' in cpp
  assert '"sched/enqueue_decoder_warmup_prefetch"' in cpp
  assert '"sched/pop_decoder_warmup_prefetch"' in cpp
  assert 'request_label = "erpp_encoder_prefetch"' not in cpp
  assert 'request_label = "predictor_prefetch"' not in cpp
  assert 'request_label = "decoder_warmup"' not in cpp


def test_erpp_encoder_prefetch_queue_is_stale_cleaned():
  cpp = _text(PREFETCHER_CPP)
  hpp = _text(PREFETCHER_HPP)

  clear_all = _function_body(cpp, "void FetchScheduleWorker::clear_all_prefetch_queues()")
  clear_before_epoch = _function_body(
      cpp,
      "void FetchScheduleWorker::clear_stale_prefetch_queues_before_epoch(",
  )
  clear_up_to_layer = _function_body(
      cpp,
      "void FetchScheduleWorker::clear_prefetch_queues_up_to_layer(int layer_idx)",
  )

  assert "prefetch_queues.encoder_predictor_by_layer" in clear_all
  assert "prefetch_queues.decoder_predictor_by_layer" in clear_all
  assert "prefetch_queues.decoder_warmup_plan_queue" in clear_all
  assert "prefetch_queues.encoder_predictor_by_layer" in clear_before_epoch
  assert "clear_prefetch_class_up_to_layer(PrefetchClass::kEncoderPredictor" in clear_up_to_layer
  assert "clear_prefetch_class_up_to_layer(PrefetchClass::kDecoderPredictor" in clear_up_to_layer
  assert "is_stale_prefetch" in cpp
  assert "const_cast" not in hpp
  assert "const_cast" not in cpp


def test_erpp_encoder_prefetch_does_not_use_unused_cache_slots():
  cpp = _text(CACHE_CPP)
  signature = (
    "CacheMngr::CacheLineOccupancyWaiter CacheMngr::miss(\n"
    "    ExpertHandler *incoming_e,\n"
    "    bool is_precise,\n"
    "    CacheRequestType request_type)"
  )
  start = cpp.index(signature)
  end = cpp.index("void CacheMngr::mark_reclaimable", start)
  body = cpp[start:end]

  assert "!is_reclaimable_only_request(request_type, metas.get())" in body


def test_validation_script_passes_erpp_encoder_layers():
  text = _text(RUN_SWITCH)

  assert 'ERPP_ENCODER_LAYERS="${ERPP_ENCODER_LAYERS:-all}"' in text
  assert '--erpp_encoder_layers' in text
  assert '"${ERPP_ENCODER_LAYERS}"' in text


def test_compare_script_passes_erpp_encoder_layers_to_nsys_cases():
  text = _text(COMPARE_RUN)

  assert 'ERPP_ENCODER_LAYERS="${ERPP_ENCODER_LAYERS:-all}"' in text
  assert '"ERPP_ENCODER_LAYERS=${ERPP_ENCODER_LAYERS}"' in text


def test_scheduler_declares_erpp_encoder_jit_ranking_task_and_state():
  hpp = _text(PREFETCHER_HPP)

  assert "kErppEncoderJitRankings" in hpp
  assert "class ErppEncoderJitRankingsTask" in hpp
  assert "std::vector<std::vector<int64_t>> rankings" in hpp
  assert "std::vector<std::vector<int64_t>> encoder_jit_rankings" in hpp
  assert "std::vector<int> encoder_jit_budgets" in hpp
  assert "std::vector<std::vector<uint8_t>> encoder_jit_pending_mask" in hpp
  assert "void store_erpp_encoder_jit_rankings" in hpp


def test_erpp_worker_jit_mode_submits_rankings_not_legacy_prefetch():
  cpp = _text(WORKER_CPP)
  body = _function_body(cpp, "void ErppEncoderPredictWorker::do_one_task_impl")

  assert "metas->enable_erpp_encoder_jit_refill" in body
  assert "ErppEncoderJitRankingsTask" in body
  assert "submit scheduler rankings" in body
  assert "kCacheRequestEncoderPredictorPrefetch" in body


def test_scheduler_declares_encoder_jit_candidate_helpers():
  hpp = _text(PREFETCHER_HPP)

  assert "int encoder_jit_floor(int layer_idx) const" in hpp
  assert "int encoder_jit_low_watermark(int layer_idx) const" in hpp
  assert "int encoder_jit_occupancy(int layer_idx) const" in hpp
  assert "bool encoder_jit_layer_enabled(int layer_idx) const" in hpp
  assert "void maybe_enqueue_encoder_jit_refill()" in hpp
  assert "std::vector<int> build_encoder_jit_required_experts" in hpp


def test_scheduler_budget_floor_mode_uses_per_layer_prediction_budget():
  hpp = _text(PREFETCHER_HPP)
  cpp = _text(PREFETCHER_CPP)
  floor_body = _function_body(hpp, "int encoder_jit_floor(int layer_idx) const")
  low_body = _function_body(hpp, "int encoder_jit_low_watermark(int layer_idx) const")
  maybe_body = _function_body(cpp, "void FetchScheduleWorker::maybe_enqueue_encoder_jit_refill()")
  compact_maybe = _compact(maybe_body)

  assert "int encoder_jit_floor(int layer_idx) const" in hpp
  assert "int encoder_jit_low_watermark(int layer_idx) const" in hpp
  assert 'erpp_encoder_jit_refill_floor_mode == "budget"' in floor_body
  assert "encoder_jit_budgets" in floor_body
  assert "layer_idx" in floor_body
  assert "return std::max(1, std::min(metas->num_expert, floor_value));" in floor_body
  assert "encoder_jit_floor(layer_idx)" in low_body
  assert "encoder_jit_floor()" not in hpp
  assert "encoder_jit_floor()" not in cpp
  assert "encoder_jit_low_watermark()" not in hpp
  assert "encoder_jit_low_watermark()" not in cpp

  budget_expr = (
      "const int budget = layer_idx < static_cast<int>(encoder_jit_budgets.size()) "
      "? encoder_jit_budgets[layer_idx] : 0;"
  )
  assert budget_expr in compact_maybe
  assert "const int floor_value = encoder_jit_floor(layer_idx);" in maybe_body
  assert "const int low_watermark = encoder_jit_low_watermark(layer_idx);" in maybe_body
  assert compact_maybe.index(budget_expr) < compact_maybe.index(
      "const int floor_value = encoder_jit_floor(layer_idx);"
  )
  assert compact_maybe.index("const int low_watermark = encoder_jit_low_watermark(layer_idx);") < compact_maybe.index(
      "build_encoder_jit_required_experts("
  )


def test_scheduler_candidate_logic_has_floor_deficit_and_topk_cover():
  cpp = _text(PREFETCHER_CPP)
  body = _function_body(cpp, "std::vector<int> FetchScheduleWorker::build_encoder_jit_required_experts")
  compact = _compact(body)

  assert "build_encoder_jit_required_experts" in cpp
  assert "floor_deficit" in cpp
  assert "enable_erpp_encoder_jit_topk_cover" in cpp
  assert "reason=floor_deficit" in cpp
  assert "reason=topk_cover" in cpp
  assert "kCacheRequestEncoderJitRefill" in cpp
  assert "erpp_encoder_jit_refill_per_idle" in cpp
  assert compact.index("if (occupancy < low_watermark)") < compact.index(
      "if (metas->enable_erpp_encoder_jit_topk_cover && budget > 0)"
  )


def test_cache_policy_treats_encoder_jit_refill_as_safe_victim_request():
  cpp = _text(CACHE_CPP)

  assert "kCacheRequestEncoderJitRefill" in cpp
  assert "set_encoder_jit_refill_context" in cpp
  assert "clear_encoder_jit_refill_context" in cpp
  assert "encoder_jit_refill_floor" in cpp
  assert "encoder_jit_refill_protected_layers" in cpp


def test_scheduler_clears_encoder_jit_pending_mask():
  hpp = _text(PREFETCHER_HPP)
  cpp = _text(PREFETCHER_CPP)

  assert "void clear_encoder_jit_pending(int layer_idx, int expert_idx)" in hpp
  assert "clear_encoder_jit_pending" in cpp
  assert "encoder_jit_pending_mask[layer_idx][expert_idx] = 0" in cpp



def test_erpp_predictor_jit_ranking_limit_respects_budget_floor_mode():
  cpp = _text(ERPP_CPP)
  limit_body = _function_body(cpp, "int ErppEncoderPredictor::encoder_jit_ranking_limit(int layer_idx, int budget) const")
  budget_mode = 'metas->erpp_encoder_jit_refill_floor_mode == "budget"'

  assert budget_mode in limit_body
  assert "encoder_jit_floor()" in limit_body
  assert "std::max(floor_value, budget)" in limit_body

  budget_branch_idx = limit_body.index(budget_mode)
  floor_idx = limit_body.index("encoder_jit_floor()")
  assert budget_branch_idx < floor_idx
  assert "return budget;" in limit_body[budget_branch_idx:floor_idx]


def test_erpp_predictor_jit_ranking_limit_respects_fixed_floor_and_legacy_budget():
  cpp = _text(ERPP_CPP)
  floor_body = _function_body(cpp, "int ErppEncoderPredictor::encoder_jit_floor")
  legacy_limit_body = _function_body(cpp, "int ErppEncoderPredictor::encoder_jit_ranking_limit(int layer_idx) const")
  limit_body = _function_body(cpp, "int ErppEncoderPredictor::encoder_jit_ranking_limit(int layer_idx, int budget) const")

  assert 'metas->erpp_encoder_jit_refill_floor_mode == "fixed"' in floor_body
  assert "metas->erpp_encoder_jit_refill_floor_value" in floor_body
  assert "encoder_budget_for_layer(layer_idx)" in legacy_limit_body
  assert "!metas->enable_erpp_encoder_jit_refill" in limit_body
  assert "return budget" in limit_body
  assert "std::max(floor_value, budget)" in limit_body


def test_jit_refill_can_use_unused_slot_before_safe_victim():
  cpp = _text(CACHE_CPP)
  signature = (
    "CacheMngr::CacheLineOccupancyWaiter CacheMngr::miss(\n"
    "    ExpertHandler *incoming_e,\n"
    "    bool is_precise,\n"
    "    CacheRequestType request_type)"
  )
  start = cpp.index(signature)
  end = cpp.index("void CacheMngr::mark_reclaimable", start)
  body = cpp[start:end]
  assert "request_type == kCacheRequestEncoderJitRefill" in body
  assert "cache_slot->unused_mems.size() > 0" in body
  assert body.index("request_type == kCacheRequestEncoderJitRefill") < body.index("select_for_evict")


def test_scheduler_checks_jit_safe_victim_per_candidate_without_blocking_lower_priority():
  cpp = _text(PREFETCHER_CPP)
  hpp = _text(PREFETCHER_HPP)
  body = _function_body(cpp, "bool FetchScheduleWorker::pop_next_prefetch_for_class")
  compact = _compact(body)

  assert "bool encoder_jit_can_dispatch(const CopyTask& task) const" in hpp
  assert "cache->has_unused_slot_for(task.expert)" in cpp
  assert "cache->has_safe_encoder_jit_refill_victim" in cpp
  assert "reason=no_safe_victim" in body
  assert "break" in body
  assert "legacy_waiting_for_reclaimable" in body
  assert compact.index("reason=no_safe_victim") < compact.index("ERPP encoder queue exhausted without dispatch")
  assert "blocked_on_encoder_predictor_prefetch_reclaimable = true" in body
  assert "reason=no_safe_victim" in body


def test_encoder_jit_pending_is_counted_per_queued_chunk():
  cpp = _text(PREFETCHER_CPP)
  hpp = _text(PREFETCHER_HPP)

  assert "void note_encoder_jit_pending(int layer_idx, int expert_idx, int count)" in hpp
  assert "void release_encoder_jit_pending_for_task(const CopyTask& task)" in hpp
  assert "std::min(next, 255)" in cpp
  assert "encoder_jit_pending_mask[task.expert->layer_idx][task.expert->expert_idx] -= 1" in cpp
  assert "const int queued_tasks = metas->chunk_prefetch ? metas->num_per_expert_param : 1" in cpp
  assert "note_encoder_jit_pending(layer_idx, expert_idx, queued_tasks)" in cpp
  assert "release_encoder_jit_pending_for_task(current_task)" in cpp


def test_encoder_jit_refill_logs_skip_and_enqueue_reasons():
  cpp = _text(PREFETCHER_CPP)

  for needle in (
      "reason=no_ranking",
      "reason=stale_ranking",
      "reason=not_in_window",
      "reason=per_idle_limit",
      "reason=already_cache_or_pending",
      "reason=no_safe_victim",
      "rank=",
      "reason=",
  ):
    assert needle in cpp



def test_encoder_layer_stats_logs_entry_use_and_done_phases():
  cpp = _text(PREFETCHER_CPP)
  hpp = _text(PREFETCHER_HPP)
  report_body = _function_body(cpp, "void PrefetchMngr::report_one_layer(int layer_id, int64_t* experts, int64_t num_expert)")
  wait_body = _function_body(cpp, "void PrefetchMngr::wait_expert")
  done_body = _function_body(cpp, "void PrefetchMngr::one_moe_layer_done")

  assert "SPARSE_CACHE_LOG_ENCODER_LAYER_STATS" in cpp
  assert "struct EncoderLayerStats" in hpp
  assert "std::vector<EncoderLayerStats> encoder_layer_stats" in hpp
  assert "void log_encoder_layer_entry_stats" in hpp
  assert "void log_encoder_layer_use_stats" in hpp
  assert "void log_encoder_layer_done_stats" in hpp
  assert "log_encoder_layer_entry_stats(layer_id, experts, num_expert)" in report_body
  assert "log_encoder_layer_use_stats(layer_id, expert_id" in wait_body
  assert "log_encoder_layer_done_stats(layer_id)" in done_body

  for phase in ("phase=entry", "phase=use", "phase=done"):
    assert phase in cpp
  for field in (
      " needed=", " entry_hit=", " entry_miss=", " actual_hit=",
      " actual_miss=", " waited=", " wait_us_total=", " wait_us_max=",
  ):
    assert field in cpp
