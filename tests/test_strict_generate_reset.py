from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read(path):
  return (REPO_ROOT / path).read_text()


def top_level_function(text, signature):
  start = text.index(signature)
  next_function = text.find("\ndef ", start + len(signature))
  if next_function == -1:
    return text[start:]
  return text[start:next_function]


def cpp_function_body(text, signature):
  start = text.index(signature)
  body_start = text.index("{", start)
  depth = 0
  for pos in range(body_start, len(text)):
    if text[pos] == "{":
      depth += 1
    elif text[pos] == "}":
      depth -= 1
      if depth == 0:
        return text[body_start:pos + 1]
  raise AssertionError(f"Could not find end of function {signature}")


def assert_in_order(text, expected_items):
  previous = -1
  for expected in expected_items:
    current = text.index(expected, previous + 1)
    assert current > previous
    previous = current


def test_python_generate_wrapper_uses_reset_for_generate():
  utils = read("src/sparse_llm_cache/utils/__init__.py")
  wrapper = top_level_function(utils, "def wrap_generate_with_initial_cache")

  assert "prefetch_mngr.reset_for_generate()" in wrapper
  assert "prefetch_mngr.reset_and_load_initial_cache()" not in wrapper


def test_pybind_exposes_reset_for_generate():
  adapter = read("src/cpp_worker/adapter.cpp")

  assert '.def("reset_for_generate", &PrefetchMngr::reset_for_generate)' in adapter


def test_hook_timing_diagnostics_are_removed_from_hot_path():
  hooks = read("src/sparse_llm_cache/utils/hooks.py")

  assert "_HookTimingStats" not in hooks
  assert "hook_timing_stats" not in hooks
  assert "SPARSE_CACHE_HOOK_TIMING" not in hooks
  assert "time.perf_counter" not in hooks


def test_generate_reset_checks_are_not_on_hook_hot_path():
  prefetcher = read("src/cpp_worker/prefetcher.cpp")
  reset_body = cpp_function_body(prefetcher, "void PrefetchMngr::reset_for_generate")

  assert "reset_in_progress.exchange(true" in reset_body
  assert "reset_in_progress.store(false" in reset_body

  for signature in (
      "void PrefetchMngr::report_one_layer(int layer_id, torch::Tensor experts)",
      "void PrefetchMngr::report_one_layer(int layer_id, int64_t* experts, int64_t num_expert)",
      "void PrefetchMngr::one_moe_layer_done(int layer_id)",
      "void PrefetchMngr::report_one_expert(int layer_id, int expert_id)",
      "void PrefetchMngr::one_expert_done(int layer_id, int expert_id)",
      "void PrefetchMngr::report_moe_attn_logits(int layer_id, torch::Tensor attn_logits)",
      "void PrefetchMngr::report_moe_layer_logits(int layer_id, torch::Tensor layer_logits)",
  ):
    assert "check_not_resetting" not in cpp_function_body(prefetcher, signature)


def test_predict_worker_does_not_poll_reset_requested_on_runtime_path():
  worker_cpp = read("src/cpp_worker/worker.cpp")
  body = cpp_function_body(worker_cpp, "void PredictWorker::do_one_task_impl")

  assert "reset_requested.load" not in body
  assert "CHECK(!(job.generate_epoch != current_generate_epoch))" in body


def test_predict_reset_releases_prefetch_budget_before_waiting_for_idle():
  worker_hpp = read("src/cpp_worker/worker.hpp")
  begin_body = cpp_function_body(worker_hpp, "void begin_reset_for_generate()")
  reset_body = cpp_function_body(worker_hpp, "void reset_for_generate(int64_t next_generate_epoch)")

  assert "clear_pending_tasks();" in begin_body
  assert "release_prefetch_layer_budget_for_reset();" in begin_body
  assert begin_body.index("clear_pending_tasks();") < begin_body.index("release_prefetch_layer_budget_for_reset();")
  assert "wait_until_idle();" in reset_body


def test_generate_epoch_is_separate_from_forward_epoch():
  prefetcher_hpp = read("src/cpp_worker/prefetcher.hpp")
  prefetcher_cpp = read("src/cpp_worker/prefetcher.cpp")
  worker_hpp = read("src/cpp_worker/worker.hpp")
  worker_cpp = read("src/cpp_worker/worker.cpp")

  scheduler_fields = prefetcher_hpp[
      prefetcher_hpp.index("class FetchScheduleWorker"):
      prefetcher_hpp.index("class PrefetchMngr")
  ]
  prefetch_mngr_fields = prefetcher_hpp[prefetcher_hpp.index("class PrefetchMngr"):]
  predict_job_fields = worker_hpp[
      worker_hpp.index("struct PredictJob"):
      worker_hpp.index("class AtomicQueue")
  ]
  predict_worker_body = worker_cpp[
      worker_cpp.index("void PredictWorker::do_one_task_impl"):
      worker_cpp.index("void ExpertUnlockWorker::do_one_task_impl")
  ]

  assert "current_forward_epoch" in scheduler_fields
  assert "forward_epoch" in prefetcher_cpp
  assert "forward_epoch" in predict_job_fields
  assert "forward_epoch" in worker_cpp

  assert "int64_t generate_epoch" in prefetch_mngr_fields
  assert "int64_t current_generate_epoch" in scheduler_fields
  assert "int64_t generate_epoch" in predict_job_fields

  assert "CHECK(!(job.generate_epoch != current_generate_epoch))" in predict_worker_body


def test_scheduler_reset_is_a_control_task_without_hot_path_non_idle_accounting():
  hpp = read("src/cpp_worker/prefetcher.hpp")
  cpp = read("src/cpp_worker/prefetcher.cpp")
  scheduler_fields = hpp[
      hpp.index("class FetchScheduleWorker"):
      hpp.index("\nclass PrefetchMngr")
  ]

  assert "kReset" in hpp
  assert "class ResetTask" in hpp
  assert "ResetTask reset_task" in scheduler_fields
  assert "void begin_reset_for_generate()" in scheduler_fields
  assert "void do_one_task_impl(ResetTask *task)" in scheduler_fields

  for removed in (
      "pending_non_idle_tasks",
      "running_non_idle_task",
      "queued_non_idle_tasks",
      "completed_non_idle_tasks",
  ):
    assert removed not in scheduler_fields

  assert "int64_t FetchScheduleWorker::add_one_task" not in cpp
  assert "void FetchScheduleWorker::wait_progress" not in cpp


def test_scheduler_reset_task_is_fifo_barrier_and_clears_only_internal_state():
  cpp = read("src/cpp_worker/prefetcher.cpp")
  reset_wrapper = cpp_function_body(
      cpp,
      "void FetchScheduleWorker::reset_for_generate",
  )
  dispatch = cpp_function_body(
      cpp,
      "void FetchScheduleWorker::do_one_task_impl(FetchScheduleTaskBase *task)",
  )
  reset_handler = cpp_function_body(
      cpp,
      "void FetchScheduleWorker::do_one_task_impl(ResetTask *task)",
  )

  assert "WorkerThread<FetchScheduleTaskBase*>::add_one_task(&reset_task)" in reset_wrapper
  assert "WorkerThread<FetchScheduleTaskBase*>::wait_progress(handle)" in reset_wrapper

  assert "case FetchScheduleTaskBase::kReset" in dispatch
  assert "do_one_task_impl(dynamic_cast<ResetTask*>(task))" in dispatch

  assert "clear_pending_tasks()" not in reset_handler
  for expected in (
      "reset_requested.store(true",
      "current_task.expert = nullptr",
      "clear_all_job_queues()",
      "clear_decoder_warmup_queue()",
      "reset_pending_reclaimable_updates()",
      "current_forward_epoch = task->next_forward_epoch",
      "current_generate_epoch = task->next_generate_epoch",
      "set_phase(kEncoderPhase)",
      "DecoderWarmupAction::kRebuildForGenerateStart",
      "reset_requested.store(false",
      "add_one_task(&idle_task)",
  ):
    assert expected in reset_handler


def test_scheduler_clear_reset_keeps_idle_scheduler_paused_until_cache_reset_finishes():
  cpp = read("src/cpp_worker/prefetcher.cpp")
  reset_handler = cpp_function_body(
      cpp,
      "void FetchScheduleWorker::do_one_task_impl(ResetTask *task)",
  )

  rebuild_branch_start = reset_handler.index(
      "if (task->action == DecoderWarmupAction::kRebuildForGenerateStart)"
  )
  rebuild_body_start = reset_handler.index("{", rebuild_branch_start)
  depth = 0
  for pos in range(rebuild_body_start, len(reset_handler)):
    if reset_handler[pos] == "{":
      depth += 1
    elif reset_handler[pos] == "}":
      depth -= 1
      if depth == 0:
        rebuild_branch = reset_handler[rebuild_body_start:pos + 1]
        after_rebuild_branch = reset_handler[pos + 1:]
        break
  else:
    raise AssertionError("Could not find rebuild reset branch")

  assert "reset_requested.store(false" in rebuild_branch
  assert "add_one_task(&idle_task)" in rebuild_branch
  assert "reset_requested.store(false" not in after_rebuild_branch
  assert "add_one_task(&idle_task)" not in after_rebuild_branch
  assert_in_order(reset_handler, (
      "reset_requested.store(true",
      "if (task->action == DecoderWarmupAction::kRebuildForGenerateStart)",
      "reset_requested.store(false",
      "add_one_task(&idle_task)",
  ))


def test_scheduler_rebuild_reset_resumes_idle_after_cache_reload():
  prefetcher = read("src/cpp_worker/prefetcher.cpp")
  reset_body = cpp_function_body(
      prefetcher,
      "void FetchScheduleWorker::do_one_task_impl(ResetTask *task)",
  )
  manager_body_start = prefetcher.index("void PrefetchMngr::reset_for_generate")
  manager_body = prefetcher[
      manager_body_start:prefetcher.index(
          "void PrefetchMngr::preempt_and_launch_one_layer", manager_body_start
      )
  ]

  assert "DecoderWarmupAction::kRebuildForGenerateStart" in reset_body
  assert_in_order(manager_body, (
      "DecoderWarmupAction::kClear",
      "cache->reset_cache_contents()",
      "cache->load_initial_plan_sync",
      "DecoderWarmupAction::kRebuildForGenerateStart",
  ))


def test_idle_task_preserves_current_task_during_reset_request():
  cpp = read("src/cpp_worker/prefetcher.cpp")
  body = cpp_function_body(cpp, "void FetchScheduleWorker::do_one_task_impl(IdleTask *idle_task)")
  reset_branch_start = body.index("if (reset_requested.load")
  reset_branch_end = body.index("if (current_task.expert != nullptr)", reset_branch_start)
  reset_branch = body[reset_branch_start:reset_branch_end]

  assert "current_task.expert = nullptr" not in reset_branch
  assert "return;" in reset_branch


def test_reset_for_generate_resets_all_runtime_components():
  prefetcher = read("src/cpp_worker/prefetcher.cpp")
  body_start = prefetcher.index("void PrefetchMngr::reset_for_generate")
  body = prefetcher[
      body_start:prefetcher.index(
          "void PrefetchMngr::preempt_and_launch_one_layer", body_start
      )
  ]

  for expected in (
      "reset_in_progress.exchange(true",
      "generate_epoch += 1",
      "predict_thread->begin_reset_for_generate()",
      "predict_thread->wait_until_idle()",
      "fetch_schedule_thread->begin_reset_for_generate()",
      "fetch_schedule_thread->wait_until_idle()",
      "fetch_thread->wait_until_idle()",
      "expert_unlocker_thread->wait_until_idle()",
      "CUDA_CALL(cudaDeviceSynchronize())",
      "predict_thread->reset_for_generate",
      "predictor->reset_sequence_state()",
      "fetch_schedule_thread->reset_for_generate",
      "cache->reset_cache_contents()",
      "cache->load_initial_plan_sync",
      "DecoderWarmupAction::kRebuildForGenerateStart",
      "reset_in_progress.store(false",
  ):
    assert expected in body

  assert_in_order(body, (
      "reset_in_progress.exchange(true",
      "generate_epoch += 1",
      "predict_thread->begin_reset_for_generate()",
      "predict_thread->wait_until_idle()",
      "fetch_schedule_thread->begin_reset_for_generate()",
      "fetch_schedule_thread->wait_until_idle()",
      "fetch_thread->wait_until_idle()",
      "CUDA_CALL(cudaDeviceSynchronize())",
      "predict_thread->reset_for_generate",
      "predictor->reset_sequence_state()",
      "fetch_schedule_thread->reset_for_generate",
      "cache->reset_cache_contents()",
      "cache->load_initial_plan_sync",
      "DecoderWarmupAction::kRebuildForGenerateStart",
      "reset_in_progress.store(false",
  ))
  assert "while (!fetch_schedule_thread->is_idle())" in body


def test_reset_for_generate_waits_for_fetch_done_before_scheduler_reset_task():
  prefetcher = read("src/cpp_worker/prefetcher.cpp")
  body_start = prefetcher.index("void PrefetchMngr::reset_for_generate")
  body = prefetcher[
      body_start:prefetcher.index(
          "void PrefetchMngr::preempt_and_launch_one_layer", body_start
      )
  ]

  assert "while (!fetch_schedule_thread->is_idle())" in body
  assert_in_order(body, (
      "fetch_schedule_thread->begin_reset_for_generate()",
      "fetch_schedule_thread->wait_until_idle()",
      "fetch_thread->wait_until_idle()",
      "while (!fetch_schedule_thread->is_idle())",
      "predict_thread->reset_for_generate",
      "fetch_schedule_thread->reset_for_generate",
      "cache->reset_cache_contents()",
  ))


def test_reset_for_generate_waits_for_expert_unlocker_before_cache_reset():
  prefetcher = read("src/cpp_worker/prefetcher.cpp")
  body_start = prefetcher.index("void PrefetchMngr::reset_for_generate")
  body = prefetcher[
      body_start:prefetcher.index(
          "void PrefetchMngr::preempt_and_launch_one_layer", body_start
      )
  ]

  assert "expert_unlocker_thread->wait_until_idle()" in body
  assert_in_order(body, (
      "fetch_thread->wait_until_idle()",
      "while (!fetch_schedule_thread->is_idle())",
      "expert_unlocker_thread->wait_until_idle()",
      "CUDA_CALL(cudaDeviceSynchronize())",
      "fetch_schedule_thread->reset_for_generate",
      "cache->reset_cache_contents()",
      "cache->load_initial_plan_sync",
  ))


def test_predictor_declares_and_implements_sequence_reset():
  hpp = read("src/cpp_worker/predictor.hpp")
  cpp = read("src/cpp_worker/predictor.cpp")
  legacy_body = cpp_function_body(cpp, "void LegacyPredictor::reset_sequence_state()")
  sep_body = cpp_function_body(cpp, "void SepPredictor::reset_sequence_state()")

  assert "virtual void reset_sequence_state()" in hpp
  assert "void LegacyPredictor::reset_sequence_state()" in cpp
  assert "void SepPredictor::reset_sequence_state()" in cpp
  assert "moe_layer_logits_buffer_list.clear()" in legacy_body
  assert "moe_layer_logits_buffer_list.clear()" in sep_body
