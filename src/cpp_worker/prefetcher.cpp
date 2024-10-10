#include <omp.h>
#include "prefetcher.hpp"
#include "profiler.hpp"
#include "logging.hpp"

void FetchScheduleWorker::reorder_experts(int layer_idx, int64_t *expert_idxs, size_t num_expert) {
  TRACE_EVENT_GURAD(kFetchScheduler, "reorder_experts");
  LOG_BLOCK(DEBUG, logger, {
    logger << "reordering experts for layer " << layer_idx << " with expert " << array_to_str(expert_idxs, num_expert);
  });
  std::vector<ExpertHandler*> done_experts;    // correctly predicted and already done
  std::vector<ExpertHandler*> going_experts;   // correctly predicted and is current task
  std::vector<ExpertHandler*> partial_experts; // correctly predicted and partially fetched, but is not current task
  std::vector<ExpertHandler*> miss_experts;    // correctly predicted, but not in cache

  for (int i = 0; i < num_expert; i++) {
    auto e = model_loader->get_source(layer_idx, expert_idxs[i]);
    if (e == current_task.expert) {
      going_experts.push_back(e);
    } else {
      if (cache->is_in_cache(e)) {
        if (e->num_ready == metas->num_per_expert_param) {
          done_experts.push_back(e);
        } else {
          partial_experts.push_back(e);
        }
      } else {
        miss_experts.push_back(e);
      }
    }
  }

  num_expert = 0;
  // reorder expert order to let model use expert in the same order of fetching
  for (auto e : done_experts)    { expert_idxs[num_expert++] = e->expert_idx; }
  for (auto e : going_experts)   { expert_idxs[num_expert++] = e->expert_idx; }
  for (auto e : partial_experts) { expert_idxs[num_expert++] = e->expert_idx; }
  for (auto e : miss_experts)    { expert_idxs[num_expert++] = e->expert_idx; }
  LOG_BLOCK(DEBUG, logger, {
    logger << "reordered expert to " << array_to_str(expert_idxs, num_expert);
  });
}

void FetchScheduleWorker::preempt_one_expert(int layer_idx, int64_t expert_idx) {
  TRACE_EVENT_GURAD(kFetchScheduler, "preemot_one_expert");

  auto e = model_loader->get_source(layer_idx, expert_idx);
  auto cur_status = e->expert_status.get();

  if (cache->is_in_cache(e) == false) {
    // a completely missed expert
    add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, 0, metas->num_per_expert_param, true);
  } else if (e->num_ready == metas->num_per_expert_param) {
    // bypass a fully fetched expert, no need to add task
    e->expert_status.transfer(kReady, kLaunching);
    cache_hit(e, true);
  } else if (e == current_task.expert) {
    current_task.is_precise = true;
    if (current_task.stop_mem_buf_idx == metas->num_per_expert_param) {
      // no need to add a redundant task
      // note there will be corresponding fetchdone for this task.
      cache_hit(e, true);
    } else {
      add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, current_task.stop_mem_buf_idx, metas->num_per_expert_param, true);
    }
  } else {
    // partial expert
    add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, e->num_ready, metas->num_per_expert_param, true);
  }
}

void FetchScheduleWorker::preempt_one_layer_without_reorder_(int layer_idx, int64_t *expert_idxs, size_t num_expert) {
  TRACE_EVENT_GURAD(kFetchScheduler, "preempt_one_layer_without_reorder_");
  LOG_BLOCK(DEBUG, logger, {
    logger << "scheduler: preempting one layer " << layer_idx << " with expert " << array_to_str(expert_idxs, num_expert);
  });

  bool preceeding_experts_in_cache = true;

  // examine expert status, classify them, and bypass experts that is already fetched.
  // for ready expert, we need to let cache know we access it and update it's priority
  // for not-ready but in cache expert, it will have corresponding task, and cache->hit will be called in the task impl
  // for not-ready and not in cache expert, it will first be called with cache->miss, then be called with cache->hit, which doesn't hurt.

  // ready, current, partial, miss
  for (int i = 0; i < num_expert; i++) {
    auto e = model_loader->get_source(layer_idx, expert_idxs[i]);
    auto cur_status = e->expert_status.get();

    // a completely missed expert
    if (cache->is_in_cache(e) == false) {
      preceeding_experts_in_cache = false;
      add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, 0, metas->num_per_expert_param, true);
      continue;
    }

    // an expert fully/partially in cache, but it may be evicted by preceeding miss expert, so we need to add redundant task for it
    if (preceeding_experts_in_cache == false) {
      add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, 0, metas->num_per_expert_param, true);
      continue;
    }

    // bypass a fully fetched expert, no need to add task
    if (e->num_ready == metas->num_per_expert_param) {
      e->expert_status.transfer(kReady, kLaunching);
      cache_hit(e, true);
      continue;
    }

    if (e == current_task.expert) {
      current_task.is_precise = true;
      if (current_task.stop_mem_buf_idx == metas->num_per_expert_param) {
        // no need to add a redundant task
        // note there will be corresponding fetchdone for this task.
        cache_hit(e, true);
      } else {
        add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, current_task.stop_mem_buf_idx, metas->num_per_expert_param, true);
      }
      continue;
    }

    // partial expert
    add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, e->num_ready, metas->num_per_expert_param, true);
  }
}
void FetchScheduleWorker::add_single_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue* queue, int starting_mem_buffer, int stop_mem_buffer, bool is_precise) {
  auto expert_handler = model_loader->get_source(layer_idx, expert_idx);
  CopyTask task;
  task.start_mem_buf_idx = starting_mem_buffer;
  task.stop_mem_buf_idx = stop_mem_buffer;
  task.expert = expert_handler;
  task.is_precise = is_precise;
  LOG(TRACE) << "scheduler: add prefetch task for one param " << task.toString();
  queue->push(task);
}

void FetchScheduleWorker::add_separate_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue *queue, int start_mem_buf_idx, int stop_mem_buf_idx, bool is_precise) {
  for (int j = start_mem_buf_idx; j < stop_mem_buf_idx; j++) {
    add_single_tasks_for_one_expert(layer_idx, expert_idx, queue, j, j+1, is_precise);
  }
}

void FetchScheduleWorker::pop_next_task(CopyTask &task, bool &found) {
  found = false;
  if (!precise_job_queue.empty()) {
    task = precise_job_queue.front();
    precise_job_queue.pop();
    found = true;
  } else {
    int i = 0;
    for (; i < metas->num_layer; i++) {
      if (per_layer_job_queues[i].empty()) {
        continue;
      }
      task = per_layer_job_queues[i].front();
      per_layer_job_queues[i].pop();
      found = true;
      break;
    }
  }
}
void PrefetchMngr::init_gpu_mem_buffer() {
  // hack: append a dummy chunk to each host experts
  if (metas->expert_mem_scale != 1.0) {
    model_loader->add_all_dummy_expert_params();
  }

  uint64_t cache_len = 0;
  if (metas->per_layer_cache) {
    // cache_len = round(metas->cache_rate * metas->num_expert) * metas->num_layer;
    cache_len = round(metas->cache_rate * metas->num_layer * metas->num_expert);
  } else {
    cache_len = round(metas->cache_rate * metas->num_layer * metas->num_expert);
  }
  cache->init_gpu_mem_buffer(cache_len);
  model_loader->mem_mngr_ctx->dummy_physical = cache->cache_slots->slots.front().unused_mems.front();
}
void PrefetchMngr::preempt_and_launch_one_layer(int layer_idx, int64_t* experts, int64_t num_expert) {
  PreemptTask preempt_task;
  preempt_task.layer_idx = layer_idx;
  preempt_task.expert_idxs = experts;
  preempt_task.num_expert = num_expert;
  auto handler = fetch_schedule_thread->add_one_task(&preempt_task);
  fetch_schedule_thread->wait_progress(handler);
  // preempt_one_layer_(layer_idx, experts.data_ptr<int64_t>(), experts.size(0));
}

void PrefetchMngr::report_one_layer(int layer_id, torch::Tensor experts) {
  report_one_layer(layer_id, experts.data_ptr<int64_t>(), experts.numel());
}
void PrefetchMngr::report_one_layer(int layer_id, int64_t* experts, int64_t num_expert) {
  TRACE_EVENT_GURAD(kHook, "report_one_layer");
  cache_stats->forward();
  LOG(INFO) << "prefetcher: consume prefetch layer progress at layer " << layer_id;
  int progress_idx = predict_thread->consume_prefetch_layer_progress();
  LOG(INFO) << "prefetcher: consume prefetch layer progress at layer " << layer_id << " done " << progress_idx;

  preempt_and_launch_one_layer(layer_id, experts, num_expert); // handle reorder, launch precise task, clear prefetch queue
  profiler->add(TimeProfiler::kCntActivatedExpert, num_expert);
  precision_profiler->record_activated_experts(layer_id, experts, num_expert);
  record_then_predict_and_prefetch(layer_id, experts, num_expert);
}
void PrefetchMngr::one_moe_layer_done(int layer_id) {
  TRACE_EVENT_GURAD(kHook, "one_moe_layer_done");
  if (metas->early_preempt == false) {
    LOG(INFO) << "prefetcher: one moe layer done, add prefetch layer budget : " << layer_id;
    predict_thread->add_prefetch_layer_budget();
  }
  if (layer_id == metas->num_layer - 1) {
    profiler->push(TimeProfiler::kCntActivatedExpert, 0);
    profiler->push(TimeProfiler::kHitCnt, 0);
    profiler->push(TimeProfiler::kMissCnt, 0);
    profiler->push(TimeProfiler::kReadyCnt, 0);
    profiler->push(TimeProfiler::kUnreadyCnt, 0);
    profiler->push(TimeProfiler::kPrefetchHitCnt, 0);
    profiler->push(TimeProfiler::kPrefetchMissCnt, 0);
    profiler->push(TimeProfiler::kWaitTime, 0);
  }
  if (layer_id == metas->num_layer - 1) {
    predict_thread->on_one_iter_done();
    // predict_thread->add_one_task();
  }
}

void PrefetchMngr::report_one_expert(int layer_id, int expert_id) {
  TRACE_EVENT_GURAD(kHook, "report_one_expert");
  if (metas->early_preempt == false) {
    PreemptOneExpertTask task;
    task.layer_id = layer_id;
    task.expert_id = expert_id;
    auto handler = fetch_schedule_thread->add_one_task(&task);
    fetch_schedule_thread->wait_progress(handler);
  }
  this->wait_expert(layer_id, expert_id);
}
void PrefetchMngr::one_expert_done(int layer_id, int expert_id) {
  TRACE_EVENT_GURAD(kHook, "one_expert_done");
  mark_expert_using(layer_id, expert_id);
}

void PrefetchMngr::wait_expert(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  TRACE_EVENT_GURAD(kHook, "wait:" + expert->toString());
  LOG(TRACE) << "waiting expert " << expert->toString();
  // model_loader->get_source(layer_id, expert_id)->expert_status.wait(kReady, kLaunching);
  auto current_status = expert->expert_status.get();;
  if (current_status == kLaunching) {
    cache_stats->hit();
    profiler->add(TimeProfiler::kReadyCnt, 1);
  } else {
    Timer timer;
    cache_stats->miss();
    profiler->add(TimeProfiler::kUnreadyCnt, 1);
    // todo: add timing of waiting expert ready
    expert->expert_status.wait(kLaunching);
    auto dur = timer.dur_us();
    profiler->add(TimeProfiler::kWaitTime, dur);
  }
  LOG(TRACE) << "waiting expert " << expert->toString() << " success";
}
void PrefetchMngr::mark_expert_using(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  CUDA_CALL(cudaEventRecord(expert->event, (cudaStream_t)(this->compute_stream)));
  expert->expert_status.transfer(kLaunching, kUsing);

  expert_unlocker_thread->add_one_task(expert);
}

void PrefetchMngr::launch_thread() {
  this->reload_env();
  predict_thread->on_one_iter_done();
  predict_thread->on_moe_layer_logits_recorded(metas->num_layer);

  fetch_schedule_thread->launch();
  predict_thread->launch();
  expert_unlocker_thread->launch();
  fetch_thread->launch();
  if (string_is_on(GetEnv("SPARSE_CACHE_THREAD_TO_E_CORE"))) {
    LOG(INFO) << "set cpu affinity";
    fetch_schedule_thread->set_cpu_affinity({30});
    predict_thread->set_cpu_affinity({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    expert_unlocker_thread->set_cpu_affinity({28});
    fetch_thread->set_cpu_affinity({26});
  }
}
PrefetchMngr::PrefetchMngr(std::shared_ptr<ModuleMeta> metas,
                           std::shared_ptr<ModelLoader> model_loader,
                           std::shared_ptr<PredictorBase> predictor,
                           int64_t compute_stream_param,
                           bool create_compute_stream,
                           TimeProfiler* profiler_ptr)
    : metas(metas), model_loader(model_loader), predictor(predictor) {
  this->cache = std::make_shared<CacheMngr>(metas, model_loader);
  predict_thread = std::make_shared<PredictWorker>();
  expert_unlocker_thread = std::make_shared<ExpertUnlockWorker>();
  fetch_thread = std::make_shared<FetchWorker>();
  fetch_schedule_thread = std::make_shared<FetchScheduleWorker>();
  cache_stats = std::make_shared<CacheStatistics>();
  // cache_stats->add_reporter([this, metas = this->metas](CacheStatistics* stats){
  //   auto tensor = stats->to_tensor();
  //   // remove iteration of prefill
  //   tensor = tensor.index({tensor.sum(1) <= metas->num_expert_per_token});
  //   // skip first 10 iteration
  //   tensor = tensor.index({torch::indexing::Slice(metas->num_layer * 10)});
  //   tensor = tensor.mean(0);
  //   std::cout << "legacy_decode_stage_hit_cnt:"  << tensor[0].item<float>() << std::endl;
  //   std::cout << "legacy_decode_stage_miss_cnt:" << tensor[1].item<float>() << std::endl;
  //   std::cout << "legacy_decode_stage_hit_rate:" << tensor[0].item<float>() / (tensor[0].item<float>() + tensor[1].item<float>()) << std::endl;
  // });
  // cache_stats->add_reporter([this, metas = this->metas](CacheStatistics* stats){
  //   auto tensor = stats->to_tensor();
  //   // remove iteration of decode
  //   tensor = tensor.index({tensor.sum(1) > metas->num_expert_per_token});
  //   // skip first 10 iteration
  //   tensor = tensor.index({torch::indexing::Slice(metas->num_layer * 2)});
  //   tensor = tensor.mean(0);
  //   std::cout << "legacy_prefill_stage_hit_cnt:"  << tensor[0].item<float>() << std::endl;
  //   std::cout << "legacy_prefill_stage_miss_cnt:" << tensor[1].item<float>() << std::endl;
  //   std::cout << "legacy_prefill_stage_hit_rate:" << tensor[0].item<float>() / (tensor[0].item<float>() + tensor[1].item<float>()) << std::endl;
  // });
  if (profiler_ptr == nullptr) {
    profiler = std::make_shared<TimeProfiler>();
  } else {
    profiler = profiler_ptr->shared_from_this();
  }
  // profiler = std::make_shared<TimeProfiler>();
  profiler->add_reporter([this, metas = this->metas](TimeProfiler *p){
    auto num_used_expert_tensor = p->to_tensor(TimeProfiler::kCntActivatedExpert);
    // auto idx_is_prefill = num_used_expert_tensor >  (metas->num_expert_per_token * metas->num_layer);
    // auto idx_is_decode  = num_used_expert_tensor <= (metas->num_expert_per_token * metas->num_layer);
    auto idx_is_prefill = p->to_tensor(TimeProfiler::kSeqLen) > 1;
    auto idx_is_decode  = p->to_tensor(TimeProfiler::kSeqLen) <= 1;
    auto smart_slice = [](torch::Tensor tensor, int skip_first) {
      if (skip_first > tensor.size(0)) {
        return tensor.index({torch::indexing::Slice(tensor.size(0))});
      } else {
        return tensor.index({torch::indexing::Slice(skip_first)});
      }
    };
    auto lambda_report_one_pair([this, p, smart_slice](
        TimeProfiler::TimeType on,
        TimeProfiler::TimeType off,
        int skip_first, torch::Tensor idx,
        std::string on_name,
        std::string off_name,
        std::string rate_name) {
      auto on_val  = smart_slice(p->to_tensor(on  ).index({idx}), skip_first).mean(torch::kFloat32).item<float>();
      auto off_val = smart_slice(p->to_tensor(off ).index({idx}), skip_first).mean(torch::kFloat32).item<float>();
      std::cout << on_name   << ":" << on_val  << std::endl;
      std::cout << off_name  << ":" << off_val << std::endl;
      std::cout << rate_name << ":" << on_val / (on_val + off_val) << std::endl;
    });

    lambda_report_one_pair(TimeProfiler::kReadyCnt, TimeProfiler::kUnreadyCnt, 10, idx_is_decode,   "decode_stage_ready_cnt",  "decode_stage_unready_cnt",  "decode_stage_ready_rate");
    lambda_report_one_pair(TimeProfiler::kReadyCnt, TimeProfiler::kUnreadyCnt,  2, idx_is_prefill, "prefill_stage_ready_cnt", "prefill_stage_unready_cnt", "prefill_stage_ready_rate");
    lambda_report_one_pair(TimeProfiler::kHitCnt, TimeProfiler::kMissCnt, 10, idx_is_decode,   "decode_stage_hit_cnt",  "decode_stage_miss_cnt",  "decode_stage_hit_rate");
    lambda_report_one_pair(TimeProfiler::kHitCnt, TimeProfiler::kMissCnt,  2, idx_is_prefill, "prefill_stage_hit_cnt", "prefill_stage_miss_cnt", "prefill_stage_hit_rate");
    lambda_report_one_pair(TimeProfiler::kPrefetchHitCnt, TimeProfiler::kPrefetchMissCnt, 10, idx_is_decode,   "decode_stage_prefetch_hit_cnt",  "decode_stage_prefetch_miss_cnt",  "decode_stage_prefetch_hit_rate");
    lambda_report_one_pair(TimeProfiler::kPrefetchHitCnt, TimeProfiler::kPrefetchMissCnt,  2, idx_is_prefill, "prefill_stage_prefetch_hit_cnt", "prefill_stage_prefetch_miss_cnt", "prefill_stage_prefetch_hit_rate");

    {
      auto time = smart_slice(p->to_tensor(TimeProfiler::kWaitTime).index({idx_is_decode}), 10); // skip first 10 and last 1iteration
      std::cout << "decode_stage_wait_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
    {
      auto time = smart_slice(p->to_tensor(TimeProfiler::kWaitTime).index({idx_is_prefill}), 2); // skip first 2 and last 1iteration
      std::cout << "prefill_stage_wait_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
    {
      auto time = smart_slice(p->to_tensor(TimeProfiler::kModelForward).index({idx_is_decode}), 10); // skip first 10 and last 1iteration
      std::cout << "decode_stage_forward_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
    {
      auto time = smart_slice(p->to_tensor(TimeProfiler::kModelForward).index({idx_is_prefill}), 2); // skip first 2 and last 1iteration
      std::cout << "prefill_stage_forward_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
    {
      auto time = smart_slice(p->to_tensor(TimeProfiler::kPredictTime), 10); // skip first 10 and last 1iteration
      std::cout << "predict_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
  });
  precision_profiler = std::make_shared<PrecisionProfiler>();
  precision_profiler->decode_expert_per_token = metas->num_expert_per_token;

  if (create_compute_stream) {
    CUDA_CALL(cudaStreamCreateWithFlags((cudaStream_t*)(&compute_stream), cudaStreamNonBlocking));
    {
      // set torch compute stream
      at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal((cudaStream_t)compute_stream, model_loader->mem_mngr_ctx->device_id));
      auto blas_handle = at::cuda::getCurrentCUDABlasHandle();
      cublasStatus_t ret = cublasSetStream(blas_handle, at::cuda::getCurrentCUDAStream());
      CHECK(ret == CUBLAS_STATUS_SUCCESS);
    }
    this->set_compute_stream(compute_stream);
  } else {
    this->set_compute_stream(compute_stream_param);
  }

  if (metas->cache_only) {
    copy_stream = compute_stream;
  } else {
    CUDA_CALL(cudaStreamCreateWithFlags((cudaStream_t*)(&copy_stream),    cudaStreamNonBlocking));
  }

  predict_thread->init(fetch_schedule_thread.get(), predictor.get(), cache.get(), metas.get());
  predict_thread->precision_profiler = precision_profiler.get();
  fetch_thread->init(metas.get(), fetch_schedule_thread.get(), model_loader->mem_mngr_ctx.get(), (cudaStream_t)copy_stream);
  fetch_schedule_thread->init(metas.get(), model_loader.get(), this->cache.get(), fetch_thread.get(), predict_thread.get(), cache_stats.get(), profiler.get());

  predictor->profiler = profiler;
}

void PrefetchMngr::set_compute_stream(int64_t stream) {
  compute_stream = stream;
  predictor->compute_stream = (cudaStream_t)stream;
}

void PrefetchMngr::report_moe_attn_logits(int layer_id, torch::Tensor attn_logits) {
  LOG_BLOCK(DEBUG, logger, {
    logger << "prefetch mngr, report_moe_attn_logits " << layer_id << ", " << attn_logits.sizes();
  });
  predictor->record_moe_attn_logits(layer_id, attn_logits);
  predict_thread->on_moe_attn_input_logits_recorded(layer_id);
}

void PrefetchMngr::report_moe_layer_logits(int layer_id, torch::Tensor layer_logits) {
  LOG_BLOCK(INFO, logger, {
    logger << "prefetch mngr, report_moe_layer_logits " << layer_id << ", " << layer_logits.sizes();
  });
  predictor->record_moe_layer_logits(layer_id, layer_logits);
  predict_thread->on_moe_layer_logits_recorded(layer_id);
  if (layer_id == 0) {
    auto seq_len = layer_logits.size(1);
    profiler->push(TimeProfiler::kSeqLen, seq_len);
  }
  if (metas->sleep_on_report_logits_us) {
    cuda_sleep(metas->sleep_on_report_logits_us, compute_stream);
  }
}

void PrefetchMngr::record_then_predict_and_prefetch(int layer_id, int64_t* experts, int64_t num_expert) {
  TRACE_EVENT_GURAD(kHook, "record_then_predict_and_launch");
  // LOG_BLOCK(DEBUG, logger, {
  //   logger << "actual " << layer_id << ":" << tensor_to_str(experts);
  // });
  if (num_expert <= metas->num_expert_per_token) {
    predictor->add_one_layer(layer_id, experts, num_expert);
  } else {
    LOG(TRACE) << "identified prefill iteration, skip adding it to prefill " << num_expert;
    // predictor->clear_access_buffer();
    if (layer_id == 0) {
      predictor->start_of_new_sequence();
    }
  }
  // if (layer_id == metas->num_layer - 1) {
  //   predict_thread->add_one_task();
  // }
}
PrefetchMngr::~PrefetchMngr() {
  // predict_thread->add_one_task(PredictJob());
  predict_thread->add_prefetch_layer_budget();
  fetch_thread->exit();
  predict_thread->exit();
  expert_unlocker_thread->exit();
  fetch_schedule_thread->exit();
  if (TraceEventCollector::globally_enabled) {
    LOG(WARNING) << "dumping trace event to trace.json";
    std::ofstream f("trace.json", std::ios::out | std::ios::trunc);
    f << TraceEventCollector::singleton().dump_json_to_string();
    f.close();
  }
}
void FetchScheduleWorker::do_one_task_impl(FetchScheduleTaskBase *task) {
  switch (task->task_type) {
    case FetchScheduleTaskBase::kPreempt: {
      do_one_task_impl(dynamic_cast<PreemptTask *>(task));
      break;
    }
    case FetchScheduleTaskBase::kFetchDone: {
      do_one_task_impl(dynamic_cast<FetchDoneTask*>(task));
      break;
    }
    case FetchScheduleTaskBase::kIdle: {
      do_one_task_impl(dynamic_cast<IdleTask*>(task));
      break;
    }
    case FetchScheduleTaskBase::kPrefetchLayer: {
      do_one_task_impl(dynamic_cast<PrefetchLayerTask*>(task));
      break;
    }
    case FetchScheduleTaskBase::kPreemptOneExpert: {
      do_one_task_impl(dynamic_cast<PreemptOneExpertTask*>(task));
      break;
    }
    default: {
      CHECK(false) << "unknown task type " << task->task_type;
    }
  }
}
void FetchScheduleWorker::do_one_task_impl(PreemptTask *task) {
  TRACE_EVENT_GURAD(kFetchScheduler, "do preempt");
  if (metas->reorder_experts) {
    this->reorder_experts(task->layer_idx, task->expert_idxs, task->num_expert);
  }
  if (metas->early_preempt) {
    this->preempt_one_layer_without_reorder_(task->layer_idx, task->expert_idxs, task->num_expert);
    predict_thread->add_prefetch_layer_budget();
  }

  if (per_layer_job_queues[task->layer_idx].empty() == false) {
    LOG(TRACE) << "scheduler: do PreemptTask, preempting one layer " << task->layer_idx << ", queue is not empty, current task is " << current_task.toString();
    per_layer_job_queues[task->layer_idx].clear();
  } else {
    LOG(TRACE) << "scheduler: do PreemptTask, preempting one layer " << task->layer_idx << ", queue is empty";
  }
}
void FetchScheduleWorker::do_one_task_impl(PreemptOneExpertTask *task) {
  TRACE_EVENT_GURAD(kFetchScheduler, "do preempt one expert");
  this->preempt_one_expert(task->layer_id, task->expert_id);
}

void FetchScheduleWorker::init(ModuleMeta *metas, ModelLoader *model_loader,
                               CacheMngr *cache, FetchWorker *fetch_thread,
                               PredictWorker *predict_thread, CacheStatistics *cache_stats, TimeProfiler* profiler) {
  this->metas = metas;
  this->model_loader = model_loader;
  this->cache = cache;
  this->fetch_thread = fetch_thread;
  this->predict_thread = predict_thread;
  this->cache_stats = cache_stats;
  this->profiler = profiler;
  per_layer_job_queues.resize(metas->num_layer);
  this->add_one_task(&this->idle_task);
}
void FetchScheduleWorker::do_one_task_impl(FetchDoneTask *_) {
  TRACE_EVENT_GURAD(kFetchScheduler, "one fetch done " + current_task.toString());
  LOG(TRACE) << "scheduler: received one fetch job done " << current_task.toString();
  current_task.expert->num_ready = current_task.stop_mem_buf_idx;
  if (current_task.stop_mem_buf_idx == metas->num_per_expert_param) {
    LOG(TRACE) << "scheduler: all fetch job done for expert " << current_task.toString();

    if (current_task.is_precise && metas->reorder_experts == false) {
      // cache_hit(current_task.expert, current_task.is_precise);
    }
    current_task.expert->expert_status.transfer(kFetching, current_task.is_precise ? kLaunching : kReady);
  }
  current_task.expert = nullptr;
  this->add_one_task(&this->idle_task);
}
void FetchScheduleWorker::do_one_task_impl(IdleTask *idle_task) {
  CHECK(idle_task == &this->idle_task);
  bool found = false, sent = false;
  pop_next_task(current_task, found);
  if (found) {
    sent = send_one_job(&current_task);
  }
  if (!found || !sent) {
    // re add this idle task
    current_task.expert = nullptr;
    this->add_one_task(&this->idle_task);
  }
}

void FetchScheduleWorker::do_one_task_impl(PrefetchLayerTask *task) {
  TRACE_EVENT_GURAD(kFetchScheduler, "add task for layer " + std::to_string(task->layer_idx) + "[" + array_to_str(task->expert_idxs, task->num_expert) + "]");
  LOG(TRACE) << "scheduler: do PrefetchLayerTask, add prefetch task for layer " << task->layer_idx;
  CHECK(per_layer_job_queues[task->layer_idx].empty());
  CHECK(current_task.expert == nullptr || current_task.expert->layer_idx != task->layer_idx);
  // ready, partial, miss
  for (int i = 0; i < task->num_expert; i++) {
    auto expert = model_loader->get_source(task->layer_idx, task->expert_idxs[i]);
    LOG(TRACE) << "scheduler: do PrefetchLayerTask, adding prefetch task " << expert->toString();
    if (metas->promote_hit_in_prefetch && cache->is_in_cache(expert)) { cache_hit(expert, false); }
    // if (expert->num_ready == metas->num_per_expert_param) {
    //   auto cur_status = expert->expert_status.get();
    //   CHECK(cur_status == kReady || cur_status == kUsing) << "expert " << expert->toString() << " must be ready, but is " << cur_status;
    //   LOG(TRACE) << "skip add prefetch task " << expert->toString();
    //   continue;
    // }
    // if (cur_status == kReady || cur_status == kUsing) {
    //   LOG(TRACE) << "skip add prefetch task " << expert->toString();
    //   continue;
    // }
    if (metas->chunk_prefetch) {
      add_separate_tasks_for_one_expert(task->layer_idx, task->expert_idxs[i], &per_layer_job_queues[task->layer_idx], 0, metas->num_per_expert_param, false);
    } else {
      add_single_tasks_for_one_expert(task->layer_idx, task->expert_idxs[i], &per_layer_job_queues[task->layer_idx], 0, metas->num_per_expert_param, false);
    }
  }
}

bool FetchScheduleWorker::send_one_job(CopyTask *task) {
  TRACE_EVENT_GURAD(kFetchScheduler, "send:" + task->toString());
  LOG(TRACE) << "scheduler: send one prefetch task " << task->toString();

  // nullptr and 0: first time task
  // nullptr and >0 : a partial task gets evicted
  // not nullptr and 0: duplicated
  // not nullptr and not 0: normal
  CacheMngr::CacheLineOccupancyWaiter lambda_wait = [](){};
  if (task->expert->gpu_data == nullptr) {
    CHECK(task->start_mem_buf_idx == 0);
    CHECK(task->expert->num_ready == 0);
    // a missed task
    LOG(TRACE) << "scheduler: assigning gpu mem for expert " << task->toString();
    lambda_wait = cache_miss(task->expert, task->is_precise);
    task->expert->expert_status.transfer(kIdle, kFetching);
  } else {
    // handle cache_hit calls
    if (task->is_precise) {
      cache_hit(task->expert, task->is_precise);
    } else if (task->start_mem_buf_idx == 0) {
      cache_hit(task->expert, task->is_precise);
    }

    // handle duplicated task
    if (task->expert->num_ready >= task->stop_mem_buf_idx) {
      LOG(TRACE) << "scheduler: a fully duplicated task, skip it: " << task->toString();
      if (task->is_precise) {
        CHECK(task->expert->num_ready == metas->num_per_expert_param);
        task->expert->expert_status.transfer(kReady, kLaunching, false);
      }
      return false;
    } else if (task->expert->num_ready > task->start_mem_buf_idx) {
      LOG(TRACE) << "scheduler: a duplicated task is partially done, skip duplicated part: " << task->toString();
      task->start_mem_buf_idx = task->expert->num_ready;
    } else {
      CHECK(task->expert->num_ready == task->start_mem_buf_idx);
    }
  }

  {
    current_task = *task;
    current_task.lambda_wait = lambda_wait;
    fetch_thread->add_one_task(&current_task);
  }
  return true;
}
void PrefetchMngr::reload_env() {
  TraceEventCollector::reload_env();
  LogMessage::reload_env();
}
void PrefetchMngr::temp_move_expert_to_gpu(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  auto gpu_data = model_loader->mem_mngr_ctx->dummy_physical;

  // use compute stream to avoid race
  for (int mem_buf_idx = 0; mem_buf_idx < metas->num_per_expert_param; mem_buf_idx++) {
    // LOG(ERROR) << "fetcher: copy from " << task.expert->host_data.ptr(mem_buf_idx) << " to " << task.expert->gpu_data->ptr(mem_buf_idx);
    CUDA_CALL(cudaMemcpyAsync(
        gpu_data->ptr(mem_buf_idx),
        expert->host_data->ptr(mem_buf_idx),
        expert->host_data->nbytes(mem_buf_idx),
        cudaMemcpyHostToDevice, (cudaStream_t)compute_stream));
  }
  expert->reference_to_model_param->unmap();
  expert->reference_to_model_param->map_to(gpu_data, model_loader->mem_mngr_ctx.get());

  CUDA_CALL(cudaStreamSynchronize((cudaStream_t)compute_stream));
}
void PrefetchMngr::temp_move_expert_back_to_host(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  auto gpu_data = model_loader->mem_mngr_ctx->dummy_physical;

  // use compute stream to avoid race
  for (int mem_buf_idx = 0; mem_buf_idx < metas->num_per_expert_param; mem_buf_idx++) {
    // LOG(ERROR) << "fetcher: copy from " << task.expert->host_data.ptr(mem_buf_idx) << " to " << task.expert->gpu_data->ptr(mem_buf_idx);
    CUDA_CALL(cudaMemcpyAsync(
        expert->host_data->ptr(mem_buf_idx),
        gpu_data->ptr(mem_buf_idx),
        expert->host_data->nbytes(mem_buf_idx),
        cudaMemcpyDeviceToHost, (cudaStream_t)compute_stream));
  }

  CUDA_CALL(cudaStreamSynchronize((cudaStream_t)compute_stream));
}
