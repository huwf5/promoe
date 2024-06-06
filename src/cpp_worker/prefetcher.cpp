#include <omp.h>
#include "prefetcher.hpp"
#include "profiler.hpp"
#include "logging.hpp"

void PrefetchMngr::preempt_one_layer_(int layer_idx, int64_t *expert_idxs,
                                      size_t num_expert) {
  TRACE_EVENT_GURAD(kHook, "preempt_one_layer_");
  LOG_BLOCK(DEBUG, logger, {
    logger << "preempting one layer " << layer_idx << " with expert " << array_to_str(expert_idxs, num_expert);
  });
  // LOG(TRACE) << "preempting one layer " << layer_idx;
  {
    int prev_layer_idx = (layer_idx + metas->num_layer - 1) % metas->num_layer;
    LOG(TRACE) << "preempting one layer " << layer_idx << ", releasing previous layer " << prev_layer_idx << " first";
    // try_release_expert_in_layer(prev_layer_idx);
  }
  // incase of predictor thread adding task for same iteration even after the queue is preempted
  if (layer_idx == 0) { try_wait_pretictor_done(); }
  std::vector<uint64_t> correct_done_experts; // correctly predicted and already done
  std::vector<uint64_t> not_started_experts;  // not predicted
  std::vector<uint64_t> correct_going_experts;   // correctly predicted and started
  lock_task_queue();

  for (int i = 0; i < num_expert; i++) {
    // kIdle: in queue or not in queue. not started
    // kFetching: correctly predicted, maybe in queue, already started fetching (may not be the current task)
    // kReady: correctly predicted, already fetched
    // kUsing: impossible
    // inside this lock, only the status of current/previous task may change:
    //   previous:
    //     - kFetching -> kIdle, a wrong task is preempted
    //   current:
    //     - kIdle -> kFetching, a task just starts
    //     - kFetching -> kReady, a task finishes
    //   other expert may also change, since it may be evicted!!!!!!
    auto e = model_loader->get_source(layer_idx, expert_idxs[i]);
    if (e == current_task.expert) {
      // cur_status must be kIdle, kFetching, kReady
      // fixme: how to avoid this task to be evicted?
      //   by adding a redundant precise task
      auto cur_status = e->expert_status.get();
      CHECK(cur_status == kFetching || cur_status == kReady || cur_status == kIdle);
      correct_going_experts.push_back(expert_idxs[i]);
    } else {
      auto cur_status = e->expert_status.transfer(kReady, kLaunching, false);
      switch (cur_status) {
        // fixme: reorder done experts by order in cache evict policy to improve performance
        case kReady:    { correct_done_experts.push_back(expert_idxs[i]); break; }
        // case kQueue:    { queue_experts.push_back(expert_idxs[i]); break; }
        case kIdle:     { not_started_experts.push_back(expert_idxs[i]); break; }
        case kFetching: { not_started_experts.insert(not_started_experts.begin(), expert_idxs[i]); break; } // previously partially fetched then preempted expert
        // case kUsing:
        default: { CHECK(false); }
      }
    }
  }

  if (per_layer_job_queues[layer_idx].empty() == false) {
    LOG(TRACE) << "preempting one layer " << layer_idx << ", queue is not empty, current task is " << current_task.toString();
    per_layer_job_queues[layer_idx].clear();
  } else {
    LOG(TRACE) << "preempting one layer " << layer_idx << ", queue is empty";
  }
  for (auto eid : correct_going_experts) {
    LOG(TRACE) << "preempting one layer " << layer_idx << ", add task for " << eid;
    add_tasks_for_one_expert(layer_idx, eid, &precise_job_queue, std::min(current_task.mem_buf_idx + 1, metas->num_per_expert_param-1), true);
  }
  for (auto eid : not_started_experts) {
    LOG(TRACE) << "preempting one layer " << layer_idx << ", add task for " << eid;
    add_tasks_for_one_expert(layer_idx, eid, &precise_job_queue, 0, true);
  }
  unlock_task_queue();
  num_expert = 0;
  // reorder expert order to let model use expert in the same order of fetching
  for (auto e : correct_done_experts)   { expert_idxs[num_expert++] = e; }
  for (auto e : correct_going_experts)  { expert_idxs[num_expert++] = e; }
  for (auto e : not_started_experts) { expert_idxs[num_expert++] = e; }
  LOG_BLOCK(DEBUG, logger, {
    logger << "reordered expert to " << array_to_str(expert_idxs, num_expert);
  });
}
void PrefetchMngr::add_one_layer_task_(int layer_idx, int64_t *expert_idxs,
                                       size_t num_expert) {
  CHECK(false) << "Deprecated";
}
void PrefetchMngr::do_one_task(PrefetchTask *task) {
  TRACE_EVENT_GURAD(kPrefetch, "do:" + task->toString());
  LOG(DEBUG) << "do one prefetch task " << task->toString();

  // nullptr and 0: first time task
  // nullptr and >0 : a partial task gets evicted
  // not nullptr and 0: duplicated
  // not nullptr and not 0: normal

  if (task->expert->gpu_data == nullptr) {
    if (task->mem_buf_idx != 0) {
      LOG(ERROR) << "a partial task gets evicted " << task->toString();
      CHECK(task->is_precise == false);
      CHECK(task->expert->expert_status.get() == kIdle);
      // task->expert->expert_status.transfer(kIdle, kIdle);
      return;
    } else {
      // a first time task
      LOG(TRACE) << "assigning gpu mem for expert " << task->toString();
      cache->miss(task->expert);
      CHECK(task->expert->gpu_data->num_ready == 0);
      task->expert->expert_status.transfer(kIdle, kFetching);
      // LOG(TRACE) << "assigning gpu mem " << task->expert->gpu_data << " for expert " << task->toString();
    }
  }

  auto orig_status = task->expert->expert_status.get();
  if (task->expert->gpu_data->num_ready > task->mem_buf_idx) {
    LOG(TRACE) << "a duplicated partial task, skip it: expert " << task->toString() << ", " << task->expert->gpu_data->num_ready << ">" << task->mem_buf_idx;
    if (task->expert->gpu_data->num_ready == metas->num_per_expert_param) {
      if (task->is_precise) {
        task->expert->expert_status.transfer(kReady, kLaunching, false);
      }
      CHECK(orig_status == kReady || orig_status == kUsing || orig_status == kLaunching);
    } else {
      CHECK(orig_status == kFetching);
    }
    return;
  } else {
    CHECK(task->expert->gpu_data->num_ready == task->mem_buf_idx);
    CHECK(orig_status == kFetching);
  }

  CUDA_CALL(cudaMemcpyAsync(
      task->expert->gpu_data->mem_buffers[task->mem_buf_idx].ptr(),
      task->expert->host_data.mem_buffers[task->mem_buf_idx].ptr(),
      task->expert->host_data.mem_buffers[task->mem_buf_idx].len(),
      cudaMemcpyHostToDevice, this->stream));
  {
    auto gpu_tensor = task->expert->gpu_data->mem_buffers[task->mem_buf_idx].get_tensor();
    auto expert_param = task->expert->reference_to_model_param.mem_buffers[task->mem_buf_idx].get_tensor();
    expert_param.set_(gpu_tensor, 0, gpu_tensor.sizes(), gpu_tensor.strides());
  }
  CUDA_CALL(cudaStreamSynchronize(this->stream));
  task->expert->gpu_data->num_ready = task->mem_buf_idx + 1;
  if (task->mem_buf_idx == metas->num_per_expert_param - 1) {
    // for (int i = 0; i < metas->num_per_expert_param; i++) {
    //   task->expert->expert_module->register_parameter(
    //       metas->param_name_list[i],
    //       task->expert->gpu_data->mem_buffers[i].get_tensor());
    // }
    LOG(DEBUG) << "all fetch job done for expert " << task->toString();

    // CUDA_CALL(cudaStreamSynchronize(this->stream));
    task->expert->expert_status.transfer(kFetching, task->is_precise ? kLaunching : kReady);
  }
}
void PrefetchEngine::init_prefetch_worker() {
  prefetch_worker = std::make_shared<PrefetchMngr>(metas, model_loader, predictor);
}
void PrefetchMngr::add_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue* queue,
                                            int starting_mem_buffer, bool is_precise) {
  auto expert_handler = model_loader->get_source(layer_idx, expert_idx);
  LOG(TRACE) << "add prefetch task for one param " << expert_handler->toString() << ", starting from " << starting_mem_buffer;
  for (int j = starting_mem_buffer; j < expert_handler->host_data.mem_buffers.size(); j++) {
    PrefetchTask task;
    task.layer_idx = layer_idx;
    task.expert_idx = expert_idx;
    task.mem_buf_idx = j;
    task.expert = expert_handler;
    task.is_precise = is_precise;
    queue->push(task);
  }
}
void PrefetchMngr::prefetch_thread_func() {
  while (thread_exit_mark == false) {
    bool found = false;
    lock_task_queue();
    if (!precise_job_queue.empty()) {
      previous_task = current_task;
      current_task = precise_job_queue.front();
      precise_job_queue.pop();
      found = true;
    } else {
      int i = 0;
      for (; i < metas->num_layer; i++) {
        if (per_layer_job_queues[i].empty()) {
          continue;
        }
        previous_task = current_task;
        current_task = per_layer_job_queues[i].front();
        per_layer_job_queues[i].pop();
        found = true;
        break;
      }
    }

    if (found) {
      unlock_task_queue();
      do_one_task(&current_task);
    } else {
      unlock_task_queue();
      usleep(10);
    }
  }
}
void PrefetchMngr::init_gpu_mem_buffer(size_t num_buffers) {
  cache->init_gpu_mem_buffer(num_buffers);
}
void PrefetchMngr::add_one_layer_task(int layer_idx, torch::Tensor experts) {
  add_one_layer_task_(layer_idx, experts.data_ptr<int64_t>(), experts.size(0));
}
void PrefetchMngr::preempt_and_launch_one_layer(int layer_idx, torch::Tensor experts) {
  preempt_one_layer_(layer_idx, experts.data_ptr<int64_t>(), experts.size(0));
}
void PrefetchMngr::wait_expert(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  TRACE_EVENT_GURAD(kHook, "wait:" + expert->toString());
  LOG(DEBUG) << "waiting expert " << expert->toString();
  // model_loader->get_source(layer_id, expert_id)->expert_status.wait(kReady, kLaunching);
  expert->expert_status.wait(kLaunching, kLaunching);
  LOG(DEBUG) << "waiting expert " << expert->toString() << " success";
}
void PrefetchMngr::record_cuda_event(int layer_id, int expert_id) {
  CHECK(false) << "Deprecated";
  CUDA_CALL(cudaEventRecord(model_loader->get_source(layer_id, expert_id)->event, nullptr));
}
void PrefetchMngr::mark_expert_using(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  CUDA_CALL(cudaEventRecord(expert->event, nullptr));
  expert->expert_status.transfer(kLaunching, kUsing);

  expert_usage_queue_lock.lock();
  expert_usage_queue.push(expert);
  expert_usage_queue_lock.unlock();
}

void PrefetchMngr::try_release_expert(int layer_id, int expert_id) {
  LOG(TRACE) << "try unlocking expert " << layer_id << "." << expert_id;
  auto expert_handler = model_loader->get_source(layer_id, expert_id);
  auto cur_status = expert_handler->expert_status.transfer(kUsing, kReady, false);
  if (cur_status != kUsing) {
    LOG(TRACE) << "try unlocking expert " << layer_id << "." << expert_id << ": it's " << cur_status;
    return;
  }
}
void PrefetchMngr::try_release_expert_in_layer(int layer_id) {
  TRACE_EVENT_GURAD(kHook, "try_release:" + std::to_string(layer_id));
  for (int expert_id = 0; expert_id < metas->num_expert; expert_id++) {
    try_release_expert(layer_id, expert_id);
  }
}
void PrefetchMngr::launch_thread() {
  try_wait_pretictor_done = [this]() { sem_wait(&predictor_done); };
  prefetch_thread = std::thread([this]() { this->prefetch_thread_func(); });
  predict_thread = std::thread([this]() { this->predict_thread_func(); });
  expert_unlocker_thread = std::thread([this]() { this->expert_unlocker_thread_func(); });
}
void PrefetchEngine::init_predictor(std::string model_path) {
  predictor = std::make_shared<Predictor>(this->metas);
  predictor->load_model(model_path);
}
PrefetchMngr::PrefetchMngr(std::shared_ptr<ModuleMeta> metas,
                           std::shared_ptr<ModelLoader> model_loader,
                           std::shared_ptr<Predictor> predictor)
    : metas(metas), model_loader(model_loader), predictor(predictor) {
  this->cache = std::make_shared<CacheMngr>(metas, model_loader);
  per_layer_job_queues.resize(metas->num_layer);
  try_wait_pretictor_done = [](){};
  sem_init(&predictor_send, 0, 0);
  sem_init(&predictor_done, 0, 1);
  CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}
void PrefetchMngr::record_then_predict_and_prefetch(int layer_id, torch::Tensor experts) {
  TRACE_EVENT_GURAD(kHook, "record_then_predict_and_launch");
  LOG_BLOCK(DEBUG, logger, {
    logger << "actual " << layer_id << ":" << tensor_to_str(experts);
  });
  predictor->add_one_layer(layer_id, experts);
  if (layer_id == metas->num_layer - 1) {
    sem_post(&predictor_send);
    // sem_wait(&predictor_done);
  }
}
void PrefetchMngr::add_multi_layer_task(torch::Tensor experts) {
  TRACE_EVENT_GURAD(kPredict, "add_multi_layer_task");
  size_t per_layer_num_expert = experts.size(1);
  for (int layer_idx = 0; layer_idx < metas->num_layer; layer_idx++) {
    lock_task_queue();
    CHECK(per_layer_job_queues[layer_idx].empty());
    int64_t* expert_idxs = experts[layer_idx].data_ptr<int64_t>();
    for (int i = 0; i < per_layer_num_expert; i++) {
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
}
PrefetchMngr::~PrefetchMngr() {
  // for (int l = 0; l < metas->num_layer; l++) {
  //   try_release_expert_in_layer(l);
  // }
  thread_exit_mark = true;
  sem_post(&predictor_send);
  if (prefetch_thread.joinable()) { prefetch_thread.join(); }
  if (predict_thread.joinable()) { predict_thread.join(); }
  if (expert_unlocker_thread.joinable()) { expert_unlocker_thread.join(); }
}
void PrefetchMngr::predict_thread_func() {
  while (true) {
    sem_wait(&predictor_send);
    if (thread_exit_mark) { break; }
    TRACE_EVENT_GURAD(kPredict, "predict thread");
    auto prob = predictor->predict().reshape({metas->num_layer, metas->num_expert});
    auto sorted = prob.sort(-1, true);
    // auto predicted_expert_prob = std::get<0>(sorted).slice(1, 0, metas->num_predict_expert_per_layer);
    auto predicted_expert = std::get<1>(sorted).slice(1, 0, metas->num_predict_expert_per_layer);

    LOG_BLOCK(DEBUG, logger, {
      for (int l = 0; l < metas->num_layer; l++) {
        logger << "predicted expert" << l << ":" << tensor_to_str(predicted_expert[l]);
      }
    });
    this->add_multi_layer_task(predicted_expert);
    predictor->clear_access_buffer();
    sem_post(&predictor_done);
  }
}
void PrefetchMngr::expert_unlocker_thread_func() {
  while (true) {
    if (thread_exit_mark) {
      break;
    }
    ExpertHandler* expert = nullptr;
    expert_usage_queue_lock.lock();
    if (expert_usage_queue.empty() == false) {
      expert = expert_usage_queue.front();
      expert_usage_queue.pop();
      CHECK(expert != nullptr);
    }
    expert_usage_queue_lock.unlock();

    if (expert == nullptr) {
      usleep(10);
      continue;
    } else {
      expert->expert_status.wait(kUsing, kUsing);
      CUDA_CALL(cudaEventSynchronize(expert->event));
      expert->expert_status.transfer(kUsing, kReady);
    }
  }
}
