#include <omp.h>
#include "prefetcher.hpp"
#include "profiler.hpp"
#include "logging.hpp"

void FetchScheduleWorker::preempt_one_layer_(int layer_idx, int64_t *expert_idxs, size_t num_expert) {
  TRACE_EVENT_GURAD(kHook, "preempt_one_layer_");
  LOG_BLOCK(DEBUG, logger, {
    logger << "preempting one layer " << layer_idx << " with expert " << array_to_str(expert_idxs, num_expert);
  });
  std::vector<ExpertHandler*> done_experts;    // correctly predicted and already done
  std::vector<ExpertHandler*> going_experts;   // correctly predicted and is current task
  std::vector<ExpertHandler*> partial_experts; // correctly predicted and partially fetched, but is not current task
  std::vector<ExpertHandler*> miss_experts;    // correctly predicted, but not in cache

  predict_thread->consume_prefetch_layer_progress();

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

  cache_stats->forward();
  if (layer_idx == 0) { profiler->push(TimeProfiler::kCntActivatedExpert, 0); }
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

  predict_thread->add_prefetch_layer_budget();
  num_expert = 0;
  // reorder expert order to let model use expert in the same order of fetching
  for (auto e : done_experts)    { expert_idxs[num_expert++] = e->expert_idx; cache->hit(e); }
  for (auto e : going_experts)   { expert_idxs[num_expert++] = e->expert_idx; cache->hit(e); }
  for (auto e : partial_experts) { expert_idxs[num_expert++] = e->expert_idx; cache->hit(e); }
  for (auto e : miss_experts)    { expert_idxs[num_expert++] = e->expert_idx; }
  LOG_BLOCK(DEBUG, logger, {
    logger << "reordered expert to " << array_to_str(expert_idxs, num_expert);
  });
}
void FetchScheduleWorker::add_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue* queue, int starting_mem_buffer, bool is_precise) {
  auto expert_handler = model_loader->get_source(layer_idx, expert_idx);
  LOG(TRACE) << "add prefetch task for one param " << expert_handler->toString() << ", starting from " << starting_mem_buffer;
  for (int j = starting_mem_buffer; j < expert_handler->host_data.mem_buffers.size(); j++) {
    CopyTask task;
    task.mem_buf_idx = j;
    task.expert = expert_handler;
    task.is_precise = is_precise;
    queue->push(task);
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
void PrefetchMngr::wait_expert(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  TRACE_EVENT_GURAD(kHook, "wait:" + expert->toString());
  LOG(DEBUG) << "waiting expert " << expert->toString();
  // model_loader->get_source(layer_id, expert_id)->expert_status.wait(kReady, kLaunching);
  auto current_status = expert->expert_status.get();;
  if (current_status == kLaunching) {
    cache_stats->hit();
  } else {
    cache_stats->miss();
    // todo: add timing of waiting expert ready
    expert->expert_status.wait(kLaunching, kLaunching);
  }
  LOG(DEBUG) << "waiting expert " << expert->toString() << " success";
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
PrefetchMngr::~PrefetchMngr() {
  predict_thread->add_one_task();
  predict_thread->add_prefetch_layer_budget();
  fetch_thread->exit();
  predict_thread->exit();
  expert_unlocker_thread->exit();
  fetch_schedule_thread->exit();
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
    default: {
      CHECK(false) << "unknown task type " << task->task_type;
    }
  }
}
void FetchScheduleWorker::do_one_task_impl(PreemptTask *task) {
  this->preempt_one_layer_(task->layer_idx, task->expert_idxs, task->num_expert);
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
  LOG(DEBUG) << "scheduler: received one fetch job done " << current_task.toString();
  current_task.expert->num_ready = current_task.mem_buf_idx + 1;
  if (current_task.mem_buf_idx == metas->num_per_expert_param - 1) {
    LOG(DEBUG) << "scheduler: all fetch job done for expert " << current_task.toString();

    // if (current_task.is_precise) {
    //   cache->hit(current_task.expert);
    // }
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
    this->add_one_task(&this->idle_task);
  }
}

void FetchScheduleWorker::do_one_task_impl(PrefetchLayerTask *task) {
  TRACE_EVENT_GURAD(kPredict, "add task for layer " + std::to_string(task->layer_idx));
  CHECK(per_layer_job_queues[task->layer_idx].empty());
  CHECK(current_task.expert == nullptr || current_task.expert->layer_idx != task->layer_idx);
  // ready, partial, miss
  for (int i = 0; i < task->num_expert; i++) {
    auto expert = model_loader->get_source(task->layer_idx, task->expert_idxs[i]);
    LOG(TRACE) << "adding prefetch task " << expert->toString();
    if (cache->is_in_cache(expert)) { cache->hit(expert); }
    if (expert->num_ready == metas->num_per_expert_param) {
      auto cur_status = expert->expert_status.get();
      CHECK(cur_status == kReady) << "expert " << expert->toString() << " must be ready, but is " << cur_status;
      LOG(TRACE) << "skip add prefetch task " << expert->toString();
      continue;
    }
    // if (cur_status == kReady || cur_status == kUsing) {
    //   LOG(TRACE) << "skip add prefetch task " << expert->toString();
    //   continue;
    // }
    add_tasks_for_one_expert(task->layer_idx, task->expert_idxs[i], &per_layer_job_queues[task->layer_idx], expert->num_ready, false);
  }
}

bool FetchScheduleWorker::send_one_job(CopyTask *task) {
  TRACE_EVENT_GURAD(kPrefetch, "send:" + task->toString());
  LOG(DEBUG) << "scheduler: send one prefetch task " << task->toString();

  // nullptr and 0: first time task
  // nullptr and >0 : a partial task gets evicted
  // not nullptr and 0: duplicated
  // not nullptr and not 0: normal
  CacheMngr::CacheLineOccupancyWaiter lambda_wait = [](){};
  if (task->expert->gpu_data == nullptr) {
    if (task->mem_buf_idx != 0) {
      CHECK(task->expert->num_ready == 0);
      // a predicted task gets preempted, but survives during preempt, then be predicted in next iteration, and gets evicted by a higher probability prefetch.
      LOG(ERROR) << "scheduler: a partial task gets evicted " << task->toString();
      CHECK(task->is_precise == false);
      CHECK(task->expert->expert_status.get() == kIdle);
      // task->expert->expert_status.transfer(kIdle, kIdle);
      return false;
    } else {
      // a first time task
      LOG(TRACE) << "scheduler: assigning gpu mem for expert " << task->toString();
      lambda_wait = cache->miss(task->expert);
      // lambda_wait();
      CHECK(task->expert->num_ready == 0);
      task->expert->expert_status.transfer(kIdle, kFetching);
      // LOG(TRACE) << "assigning gpu mem " << task->expert->gpu_data << " for expert " << task->toString();
    }
  }

  auto orig_status = task->expert->expert_status.get();
  CHECK(task->expert->num_ready == task->mem_buf_idx) << "redundant task not allowed: " << task->toString();
  if (task->expert->num_ready > task->mem_buf_idx) {
    CHECK(false);
    LOG(TRACE) << "scheduler: a duplicated partial task, skip it: expert " << task->toString() << ", " << task->expert->num_ready << ">" << task->mem_buf_idx;
    if (task->expert->num_ready == metas->num_per_expert_param) {
      if (task->is_precise) {
        task->expert->expert_status.transfer(kReady, kLaunching, false);
        // cache->hit(task->expert);
      }
      CHECK(orig_status == kReady || orig_status == kUsing || orig_status == kLaunching);
    } else {
      CHECK(orig_status == kFetching);
    }
    return false;
  } else {
    CHECK(task->expert->num_ready == task->mem_buf_idx);
    CHECK(orig_status == kFetching);
  }

  {
    current_task = *task;
    current_task.lambda_wait = lambda_wait;
    fetch_thread->add_one_task(&current_task);
  }
  return true;
}
