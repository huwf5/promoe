#include <omp.h>
#include "prefetcher.hpp"
#include "profiler.hpp"
#include "logging.hpp"

void FetchScheduleWorker::preempt_one_layer_(int layer_idx, int64_t *expert_idxs, size_t num_expert) {
  this->reorder_experts(layer_idx, expert_idxs, num_expert);
  this->preempt_one_layer_without_reorder_(layer_idx, expert_idxs, num_expert);
  #ifdef DEAD_CODE
  TRACE_EVENT_GURAD(kHook, "preempt_one_layer_");
  LOG_BLOCK(DEBUG, logger, {
    logger << "preempting one layer " << layer_idx << " with expert " << array_to_str(expert_idxs, num_expert);
  });
  std::vector<ExpertHandler*> done_experts;    // correctly predicted and already done
  std::vector<ExpertHandler*> going_experts;   // correctly predicted and is current task
  std::vector<ExpertHandler*> partial_experts; // correctly predicted and partially fetched, but is not current task
  std::vector<ExpertHandler*> miss_experts;    // correctly predicted, but not in cache

  // examine expert status, classify them, and bypass experts that is already fetched.
  // for ready expert, we need to let cache know we access it and update it's priority
  // for not-ready but in cache expert, it will have corresponding task, and cache->hit will be called in the task impl
  // for not-ready and not in cache expert, it will first be called with cache->miss, then be called with cache->hit, which doesn't hurt.
  for (int i = 0; i < num_expert; i++) {
    auto e = model_loader->get_source(layer_idx, expert_idxs[i]);
    auto cur_status = e->expert_status.get();
    if (e == current_task.expert) {
      CHECK(cur_status == kFetching) << "current task's status must be fetching, but is " << cur_status << " for " << e->toString();
      // Q: how migrate this from ready to launching? A: by setting this task to be precise
      going_experts.push_back(e);
      current_task.is_precise = true;
    } else {
      if (cache->is_in_cache(e)) {
        if (cur_status == kReady) {
          // bypass an already fetched expert
          e->expert_status.transfer(kReady, kLaunching);
          done_experts.push_back(e);
        } else if (cur_status == kFetching) {
          partial_experts.push_back(e);
        } else {
          CHECK(false) << "impossible status " << cur_status << " for preempt expert " << e->toString();
        }
      } else {
        miss_experts.push_back(e);
      }
    }
  }

  profiler->add(TimeProfiler::kCntActivatedExpert, num_expert);

  if (per_layer_job_queues[layer_idx].empty() == false) {
    LOG(TRACE) << "preempting one layer " << layer_idx << ", queue is not empty, current task is " << current_task.toString();
    per_layer_job_queues[layer_idx].clear();
  } else {
    LOG(TRACE) << "preempting one layer " << layer_idx << ", queue is empty";
  }

  if (going_experts.size() > 0) {
    auto e = going_experts[0];
    CHECK(going_experts.size() == 1) << "going_experts.size() must be 1, but is " << going_experts.size();
    LOG(TRACE) << "preempting one layer " << layer_idx << ", add task for " << e->expert_idx;
    if (current_task.mem_buf_idx == metas->num_per_expert_param - 1) {
      // no need to add a redundant task
    } else {
      add_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, current_task.mem_buf_idx + 1, true);
    }
  }
  for (auto e : partial_experts) {
    LOG(TRACE) << "preempting one layer " << layer_idx << ", add task for " << e->expert_idx;
    CHECK(e->num_ready != metas->num_per_expert_param && e->num_ready != 0) << "num_ready must not be " << metas->num_per_expert_param << " and 0 for " << e->toString();
    add_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, e->num_ready, true);
  }
  for (auto e : miss_experts) {
    LOG(TRACE) << "preempting one layer " << layer_idx << ", add task for " << e->expert_idx;
    CHECK(e->num_ready == 0) << "num_ready must not be 0 for " << e->toString();
    add_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, e->num_ready, true);
  }

  num_expert = 0;
  // reorder expert order to let model use expert in the same order of fetching
  for (auto e : done_experts)    { expert_idxs[num_expert++] = e->expert_idx; cache->hit(e); }
  for (auto e : going_experts)   { expert_idxs[num_expert++] = e->expert_idx; cache->hit(e); }
  for (auto e : partial_experts) { expert_idxs[num_expert++] = e->expert_idx; cache->hit(e); }
  for (auto e : miss_experts)    { expert_idxs[num_expert++] = e->expert_idx; }
  LOG_BLOCK(DEBUG, logger, {
    logger << "reordered expert to " << array_to_str(expert_idxs, num_expert);
  });
  #endif
}

void FetchScheduleWorker::reorder_experts(int layer_idx, int64_t *expert_idxs, size_t num_expert) {
  TRACE_EVENT_GURAD(kHook, "reorder_experts");
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
  TRACE_EVENT_GURAD(kHook, "preemot_one_expert");

  auto e = model_loader->get_source(layer_idx, expert_idx);
  auto cur_status = e->expert_status.get();

  if (cache->is_in_cache(e) == false) {
    // a completely missed expert
    add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, 0, metas->num_per_expert_param, true);
  } else if (e->num_ready == metas->num_per_expert_param) {
    // bypass a fully fetched expert, no need to add task
    e->expert_status.transfer(kReady, kLaunching);
    cache->hit(e);
  } else if (e == current_task.expert) {
    current_task.is_precise = true;
    if (current_task.stop_mem_buf_idx == metas->num_per_expert_param) {
      // no need to add a redundant task
      // note there will be corresponding fetchdone for this task.
      cache->hit(e);
    } else {
      add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, current_task.stop_mem_buf_idx, metas->num_per_expert_param, true);
    }
  } else {
    // partial expert
    add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, e->num_ready, metas->num_per_expert_param, true);
  }
}

void FetchScheduleWorker::preempt_one_layer_without_reorder_(int layer_idx, int64_t *expert_idxs, size_t num_expert) {
  TRACE_EVENT_GURAD(kHook, "preempt_one_layer_without_reorder_");
  LOG_BLOCK(DEBUG, logger, {
    logger << "preempting one layer " << layer_idx << " with expert " << array_to_str(expert_idxs, num_expert);
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
      cache->hit(e);
      continue;
    }

    if (e == current_task.expert) {
      current_task.is_precise = true;
      if (current_task.stop_mem_buf_idx == metas->num_per_expert_param) {
        // no need to add a redundant task
        // note there will be corresponding fetchdone for this task.
        cache->hit(e);
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
  LOG(TRACE) << "add prefetch task for one param " << expert_handler->toString() << ", starting from " << starting_mem_buffer;
  CopyTask task;
  task.start_mem_buf_idx = starting_mem_buffer;
  task.stop_mem_buf_idx = stop_mem_buffer;
  task.expert = expert_handler;
  task.is_precise = is_precise;
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
void PrefetchMngr::init_gpu_mem_buffer(size_t num_buffers) {
  cache->init_gpu_mem_buffer(num_buffers);
}
void PrefetchMngr::preempt_and_launch_one_layer(int layer_idx, torch::Tensor experts) {
  PreemptTask preempt_task;
  preempt_task.layer_idx = layer_idx;
  preempt_task.expert_idxs = experts.data_ptr<int64_t>();
  preempt_task.num_expert = experts.numel();
  auto handler = fetch_schedule_thread->add_one_task(&preempt_task);
  fetch_schedule_thread->wait_progress(handler);
  // preempt_one_layer_(layer_idx, experts.data_ptr<int64_t>(), experts.size(0));
}

void PrefetchMngr::report_one_layer(int layer_id, torch::Tensor experts) {
  cache_stats->forward();
  if (layer_id == 0) { profiler->push(TimeProfiler::kCntActivatedExpert, 0); }
  predict_thread->consume_prefetch_layer_progress();
  preempt_and_launch_one_layer(layer_id, experts); // handle reorder, launch precise task, clear prefetch queue
  profiler->add(TimeProfiler::kCntActivatedExpert, experts.numel());
  record_then_predict_and_prefetch(layer_id, experts);
}
void PrefetchMngr::one_moe_layer_done(int layer_id) {
  if (metas->early_preempt == false) {
    predict_thread->add_prefetch_layer_budget();
  }
}

void PrefetchMngr::report_one_expert(int layer_id, int expert_id) {
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
  } else {
    cache_stats->miss();
    // todo: add timing of waiting expert ready
    expert->expert_status.wait(kLaunching, kLaunching);
  }
  LOG(TRACE) << "waiting expert " << expert->toString() << " success";
}
void PrefetchMngr::mark_expert_using(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  CUDA_CALL(cudaEventRecord(expert->event, nullptr));
  expert->expert_status.transfer(kLaunching, kUsing);

  expert_unlocker_thread->add_one_task(expert);
}

void PrefetchMngr::launch_thread() {
  predict_thread->add_one_task();
  fetch_schedule_thread->launch();
  predict_thread->launch();
  expert_unlocker_thread->launch();
  fetch_thread->launch();
}
PrefetchMngr::PrefetchMngr(std::shared_ptr<ModuleMeta> metas,
                           std::shared_ptr<ModelLoader> model_loader,
                           std::shared_ptr<Predictor> predictor)
    : metas(metas), model_loader(model_loader), predictor(predictor) {
  this->cache = std::make_shared<CacheMngr>(metas, model_loader);
  predict_thread = std::make_shared<PredictWorker>();
  expert_unlocker_thread = std::make_shared<ExpertUnlockWorker>();
  fetch_thread = std::make_shared<FetchWorker>();
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  fetch_schedule_thread = std::make_shared<FetchScheduleWorker>();
  cache_stats = std::make_shared<CacheStatistics>();
  cache_stats->add_reporter([this](CacheStatistics* stats){
    auto tensor = stats->to_tensor();
    // remove iteration of prefill
    tensor = tensor.index({tensor.sum(1) <= this->metas->num_expert_per_token});
    // skip first 10 iteration
    tensor = tensor.index({torch::indexing::Slice(this->metas->num_layer * 10)});
    tensor = tensor.mean(0);
    std::cout << "decode_stage_hit_cnt:"  << tensor[0].item<float>() << std::endl;
    std::cout << "decode_stage_miss_cnt:" << tensor[1].item<float>() << std::endl;
    std::cout << "decode_stage_hit_rate:" << tensor[0].item<float>() / (tensor[0].item<float>() + tensor[1].item<float>()) << std::endl;
  });
  cache_stats->add_reporter([this](CacheStatistics* stats){
    auto tensor = stats->to_tensor();
    // remove iteration of decode
    tensor = tensor.index({tensor.sum(1) > this->metas->num_expert_per_token});
    // skip first 10 iteration
    tensor = tensor.index({torch::indexing::Slice(this->metas->num_layer * 2)});
    tensor = tensor.mean(0);
    std::cout << "prefill_stage_hit_cnt:"  << tensor[0].item<float>() << std::endl;
    std::cout << "prefill_stage_miss_cnt:" << tensor[1].item<float>() << std::endl;
    std::cout << "prefill_stage_hit_rate:" << tensor[0].item<float>() / (tensor[0].item<float>() + tensor[1].item<float>()) << std::endl;
  });
  profiler = std::make_shared<TimeProfiler>();
  profiler->add_reporter([this](TimeProfiler *p){
    auto num_used_expert_tensor = p->to_tensor(TimeProfiler::kCntActivatedExpert);
    auto forward_time_tensor    = p->to_tensor(TimeProfiler::kModelForward);
    {
      auto idx = num_used_expert_tensor <= (this->metas->num_expert_per_token * this->metas->num_layer);
      auto time = forward_time_tensor.index({idx}).index({torch::indexing::Slice(10)});
      std::cout << "decode_stage_forward_time:" << time.mean(torch::kFloat32).item() << std::endl;
    }
    {
      auto idx = num_used_expert_tensor > (this->metas->num_expert_per_token * this->metas->num_layer);
      auto time = forward_time_tensor.index({idx}).index({torch::indexing::Slice(2)});
      std::cout << "prefill_stage_forward_time:" << time.mean(torch::kFloat32).item() << std::endl;
    }
  });
  predict_thread->init(fetch_schedule_thread.get(), predictor.get(), cache.get(), metas.get());
  fetch_thread->init(metas.get(), fetch_schedule_thread.get(), stream);
  fetch_schedule_thread->init(metas.get(), model_loader.get(), this->cache.get(), fetch_thread.get(), predict_thread.get(), cache_stats.get(), profiler.get());
}
void PrefetchMngr::record_then_predict_and_prefetch(int layer_id, torch::Tensor experts) {
  TRACE_EVENT_GURAD(kHook, "record_then_predict_and_launch");
  LOG_BLOCK(DEBUG, logger, {
    logger << "actual " << layer_id << ":" << tensor_to_str(experts);
  });
  if (experts.numel() <= metas->num_expert_per_token) {
    predictor->add_one_layer(layer_id, experts);
  } else {
    LOG(DEBUG) << "identified prefill iteration, skip adding it to prefill " << experts.numel();
  }
  if (layer_id == metas->num_layer - 1) {
    predict_thread->add_one_task();
  }
}
#ifdef DEAD_CODE
void FetchScheduleWorker::add_one_layer_task(int layer_idx, torch::Tensor experts) {
  CHECK(false) << "Deprecated";
  add_one_layer_task(layer_idx, experts.data_ptr<int64_t>(), experts.size(0));
}

void FetchScheduleWorker::add_one_layer_task(int layer_idx, int64_t *expert_idxs, size_t num_expert) {
  CHECK(false) << "Deprecated";
  TRACE_EVENT_GURAD(kPredict, "add task for layer " + std::to_string(layer_idx));
  lock_task_queue();
  CHECK(per_layer_job_queues[layer_idx].empty());
  for (int i = 0; i < num_expert; i++) {
    auto expert = model_loader->get_source(layer_idx, expert_idxs[i]);
    LOG(TRACE) << "adding prefetch task " << expert->toString();
    auto cur_status = expert->expert_status.get();
    if (cur_status == kReady || cur_status == kUsing) {
      LOG(TRACE) << "skip add prefetch task " << expert->toString();
      continue;
    }
    add_tasks_for_one_expert(layer_idx, expert_idxs[i], &per_layer_job_queues[layer_idx]);
  }
  unlock_task_queue();
}
#endif
PrefetchMngr::~PrefetchMngr() {
  predict_thread->add_one_task();
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
  if (metas->reorder_experts) {
    this->reorder_experts(task->layer_idx, task->expert_idxs, task->num_expert);
  }
  if (metas->early_preempt) {
    this->preempt_one_layer_without_reorder_(task->layer_idx, task->expert_idxs, task->num_expert);
    predict_thread->add_prefetch_layer_budget();
  }

  if (per_layer_job_queues[task->layer_idx].empty() == false) {
    LOG(TRACE) << "preempting one layer " << task->layer_idx << ", queue is not empty, current task is " << current_task.toString();
    per_layer_job_queues[task->layer_idx].clear();
  } else {
    LOG(TRACE) << "preempting one layer " << task->layer_idx << ", queue is empty";
  }
}
void FetchScheduleWorker::do_one_task_impl(PreemptOneExpertTask *task) {
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
  LOG(TRACE) << "scheduler: received one fetch job done " << current_task.toString();
  current_task.expert->num_ready = current_task.stop_mem_buf_idx;
  if (current_task.stop_mem_buf_idx == metas->num_per_expert_param) {
    LOG(TRACE) << "scheduler: all fetch job done for expert " << current_task.toString();

    if (current_task.is_precise && metas->reorder_experts == false) {
      cache->hit(current_task.expert);
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
  TRACE_EVENT_GURAD(kPredict, "add task for layer " + std::to_string(task->layer_idx));
  LOG(DEBUG) << "add prefetch task for layer " << task->layer_idx;
  CHECK(per_layer_job_queues[task->layer_idx].empty());
  CHECK(current_task.expert == nullptr || current_task.expert->layer_idx != task->layer_idx);
  // ready, partial, miss
  for (int i = 0; i < task->num_expert; i++) {
    auto expert = model_loader->get_source(task->layer_idx, task->expert_idxs[i]);
    LOG(TRACE) << "adding prefetch task " << expert->toString();
    if (metas->promote_hit_in_prefetch && cache->is_in_cache(expert)) { cache->hit(expert); }
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
    add_separate_tasks_for_one_expert(task->layer_idx, task->expert_idxs[i], &per_layer_job_queues[task->layer_idx], 0, metas->num_per_expert_param, false);
  }
}

bool FetchScheduleWorker::send_one_job(CopyTask *task) {
  TRACE_EVENT_GURAD(kPrefetch, "send:" + task->toString());
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
