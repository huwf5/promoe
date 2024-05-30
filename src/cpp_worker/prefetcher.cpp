#include "prefetcher.hpp"
#include "profiler.hpp"
#include "logging.hpp"

void PrefetchMngr::preempt_one_layer_(int layer_idx, int64_t *expert_idxs,
                                      size_t num_expert) {
  TRACE_EVENT_GURAD(kCacheLib, "preempt_one_layer_");
  LOG_BLOCK(DEBUG, logger, {
    logger << "preempting one layer " << layer_idx << " with expert " << array_to_str(expert_idxs, num_expert);
  });
  // LOG(TRACE) << "preempting one layer " << layer_idx;
  {
    int prev_layer_idx = (layer_idx + metas->num_layer - 1) % metas->num_layer;
    LOG(TRACE) << "preempting one layer " << layer_idx << ", releasing previous layer " << prev_layer_idx << " first";
    for (int i = 0; i < metas->num_expert; i++) {
      try_release_expert(prev_layer_idx, i);
    }
  }
  std::unordered_set<int> expert_idxs_set;
  std::vector<int64_t> correct_experts, wrong_experts;
  correct_experts.reserve(num_expert);
  wrong_experts.reserve(num_expert);
  for (int i = 0; i < num_expert; i++) {
    expert_idxs_set.insert(expert_idxs[i]);
  }
  lock_queue();
  //// empty queue, insert all missing experts
  if (per_layer_job_queues[layer_idx].empty()) {
    LOG(TRACE) << "preempting one layer " << layer_idx << ", queue is empty";
    for (int i = 0; i < num_expert; i++) {
      if (prefetched_experts[layer_idx].find(expert_idxs[i]) != prefetched_experts[layer_idx].end()) {
        LOG(TRACE) << "preempting one layer " << layer_idx << ", skipping [" << i << "]=" << expert_idxs[i] << " since it's in gpu";
        correct_experts.push_back(expert_idxs[i]);
        continue;
      }
      LOG(TRACE) << "preempting one layer " << layer_idx << ", add task for [" << i << "]=" << expert_idxs[i];
      add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue);
      wrong_experts.push_back(expert_idxs[i]);
    }
  } else {
    auto first_task = per_layer_job_queues[layer_idx].front();
    LOG(TRACE) << "preempting one layer " << layer_idx << ", queue is not empty, first task is " << first_task.layer_idx << "." << first_task.expert_idx << "." << first_task.mem_buf_idx;
    per_layer_job_queues[layer_idx].clear();
    //// queue with pending tasks, but the first task is wrongly predicted
    if (expert_idxs_set.find(first_task.expert_idx) == expert_idxs_set.end() || first_task.mem_buf_idx == 0) {
      LOG(TRACE) << "preempting one layer " << layer_idx << ", first task is miss predicted or not started. just clear entire queue.";
      for (int i = 0; i < num_expert; i++) {
        if (prefetched_experts[layer_idx].find(expert_idxs[i]) != prefetched_experts[layer_idx].end()) {
          correct_experts.push_back(expert_idxs[i]);
          continue;
        }
        add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue);
        wrong_experts.push_back(expert_idxs[i]);
      }
    } else {
      LOG(TRACE) << "preempting one layer " << layer_idx << ", first task is correctly predicted and started";
      int i = 0;
      for (; i < num_expert && expert_idxs[i] < first_task.expert_idx; i++) {
        if (prefetched_experts[layer_idx].find(expert_idxs[i]) != prefetched_experts[layer_idx].end()) {
          correct_experts.push_back(expert_idxs[i]);
          continue;
        }
        add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue);
        wrong_experts.push_back(expert_idxs[i]);
      }

      CHECK(first_task.expert_idx == expert_idxs[i]);
      add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue, first_task.mem_buf_idx);
      wrong_experts.push_back(expert_idxs[i]);
      i++;

      for (; i < num_expert; i++) {
        if (prefetched_experts[layer_idx].find(expert_idxs[i]) != prefetched_experts[layer_idx].end()) {
          correct_experts.push_back(expert_idxs[i]);
          continue;
        }
        add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue);
        wrong_experts.push_back(expert_idxs[i]);
      }
    }
  }
  unlock_queue();
  memcpy(expert_idxs,                          correct_experts.data(), correct_experts.size() * sizeof(*expert_idxs));
  memcpy(expert_idxs + correct_experts.size(),   wrong_experts.data(),   wrong_experts.size() * sizeof(*expert_idxs));
  LOG_BLOCK(DEBUG, logger, {
    logger << "reordered expert to " << array_to_str(expert_idxs, num_expert);
  });
}
void PrefetchMngr::add_one_layer_task_(int layer_idx, int64_t *expert_idxs,
                                       size_t num_expert) {
  TRACE_EVENT_GURAD(kCacheLib, "add_one_layer_task_");
  CHECK(per_layer_job_queues[layer_idx].empty());
  for (int i = 0; i < num_expert; i++) {
    LOG(TRACE) << "adding prefetch task " << layer_idx << "," << expert_idxs[i];
    lock_queue();
    if (prefetched_experts[layer_idx].find(expert_idxs[i]) != prefetched_experts[layer_idx].end()) {
      LOG(TRACE) << "skip add prefetch task " << layer_idx << "," << expert_idxs[i];
      unlock_queue();
      continue;
    }
    add_tasks_for_one_expert(layer_idx, expert_idxs[i], &per_layer_job_queues[layer_idx]);
    unlock_queue();
  }
}
void PrefetchMngr::do_one_task(PrefetchTask *task) {
  TRACE_EVENT_GURAD(kPrefetch, "do:" + task->toString());
  LOG(DEBUG) << "do one prefetch task " << task->layer_idx << "," << task->expert_idx << "," << task->mem_buf_idx << ", precise " << task->is_precise;
  if (previous_task.expert != nullptr && previous_task.expert != task->expert &&
      previous_task.mem_buf_idx != metas->num_per_expert_param-1) {
    lock_queue();
    LOG(DEBUG) << "removing partially fetched expert " << previous_task.layer_idx << "," << previous_task.expert_idx;
    prefetched_experts[previous_task.layer_idx].erase(previous_task.expert_idx);
    unlock_queue();
    unused_mems_lock.lock();
    CHECK(previous_task.expert->gpu_data != nullptr);
    unused_mems.push_back(previous_task.expert->gpu_data);
    previous_task.expert->gpu_data = nullptr;
    unused_mems_lock.unlock();
  }
  previous_task = *task;
  if (task->expert->expert_status.is_locked(kFetching) == false) {
    LOG(TRACE) << "a duplicated task, skip it: expert " << task->layer_idx << "," << task->expert_idx << "," << task->mem_buf_idx;
    return;
  }
  if (task->expert->gpu_data == nullptr) {
    LOG(TRACE) << "assigning gpu mem for expert " << task->layer_idx << "," << task->expert_idx << "," << task->mem_buf_idx;
    unused_mems_lock.lock();
    if (unused_mems.size() > 0) {
      task->expert->gpu_data = unused_mems.back();
      CHECK(task->expert->gpu_data != nullptr);
      LOG(TRACE) << "assigning gpu mem " << task->expert->gpu_data << " for expert " << task->layer_idx << "," << task->expert_idx << "," << task->mem_buf_idx;
      unused_mems.pop_back();
    } else {
      CHECK(false) << "no remaining mem buffer";
    }
    unused_mems_lock.unlock();
    // fixme: find one loc from cache
    lock_queue();
    prefetched_experts[task->layer_idx][task->expert_idx] = task->expert;
    unlock_queue();
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
  if (task->mem_buf_idx == metas->num_per_expert_param - 1) {
    // for (int i = 0; i < metas->num_per_expert_param; i++) {
    //   task->expert->expert_module->register_parameter(
    //       metas->param_name_list[i],
    //       task->expert->gpu_data->mem_buffers[i].get_tensor());
    // }
    LOG(DEBUG) << "all fetch job done for expert " << task->layer_idx << "," << task->expert_idx << "," << task->mem_buf_idx;

    CUDA_CALL(cudaStreamSynchronize(this->stream));
    task->expert->expert_status.unlock(kFetching, kReady);
  }
}
void PrefetchEngine::init_prefetch_worker() {
  prefetch_worker = std::make_shared<PrefetchMngr>(metas, model_loader, predictor);
}
void PrefetchMngr::add_tasks_for_one_expert(int layer_idx, int expert_idx, Queue* queue,
                                            int starting_mem_buffer) {
  auto expert_handler = model_loader->get_source(layer_idx, expert_idx);
  for (int j = starting_mem_buffer; j < expert_handler->host_data.mem_buffers.size(); j++) {
    PrefetchTask task;
    task.layer_idx = layer_idx;
    task.expert_idx = expert_idx;
    task.mem_buf_idx = j;
    task.expert = expert_handler;
    queue->push(task);
    LOG(TRACE) << "add prefetch task for one param " << layer_idx << "," << expert_idx << "," << j;
  }
}
void PrefetchMngr::thread_func() {
  while (thread_exit_mark == false) {
    PrefetchTask task;
    bool found = false;
    lock_queue();
    if (!precise_job_queue.empty()) {
      task = precise_job_queue.front();
      task.is_precise = true;
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

    if (found) {
      unlock_queue();
      do_one_task(&task);
    } else {
      unlock_queue();
      usleep(10);
    }
  }
}
void PrefetchMngr::init_gpu_mem_buffer(size_t num_buffers) {
  unused_mems.resize(num_buffers, nullptr);
  auto &mem_example = model_loader->get_source(0, 0)->host_data.mem_buffers;
  for (int i = 0; i < num_buffers; i++) {
    unused_mems[i] = new ExpertMemHanlder;
    unused_mems[i]->mem_buffers.resize(mem_example.size());
    for (int j = 0; j < mem_example.size(); j++) {
      unused_mems[i]->mem_buffers[j].set_tensor(
          torch::empty_like(mem_example[j].get_tensor(),
                            torch::TensorOptions().device(torch::kCUDA, 0)));
    }
  }
}
void PrefetchMngr::add_one_layer_task(int layer_idx, torch::Tensor experts) {
  add_one_layer_task_(layer_idx, experts.data_ptr<int64_t>(), experts.size(0));
}
void PrefetchMngr::preempt_one_layer(int layer_idx, torch::Tensor experts) {
  preempt_one_layer_(layer_idx, experts.data_ptr<int64_t>(), experts.size(0));
}
void PrefetchMngr::wait_and_lock_expert(int layer_id, int expert_id) {
  TRACE_EVENT_GURAD(kCacheLib, "wait_and_lock_expert");
  LOG(DEBUG) << "waiting expert " << layer_id << "." << expert_id;
  model_loader->get_source(layer_id, expert_id)->expert_status.lock(kReady, kUsing);
  LOG(DEBUG) << "waiting expert " << layer_id << "." << expert_id << " success";
}
void PrefetchMngr::try_release_expert(int layer_id, int expert_id) {
  // TRACE_EVENT_GURAD(kCacheLib, "try_release:" + std::to_string(layer_id) + "." + std::to_string(expert_id));
  LOG(TRACE) << "try unlocking expert " << layer_id << "." << expert_id;
  auto expert_handler = model_loader->get_source(layer_id, expert_id);
  if (expert_handler->expert_status.is_locked(kUsing) == false) {
    LOG(TRACE) << "try unlocking expert " << layer_id << "." << expert_id << ": it's not locked";
    return;
  }
  {
    {
      TRACE_EVENT_GURAD(kCacheLib, "try_release_expert.lock_queue");
      lock_queue();
    }
    prefetched_experts[layer_id].erase(expert_id);
    unlock_queue();
  }
  // fixme: the memory may should not be released here. add a cache module
  if (expert_handler->gpu_data != nullptr) {
    LOG(TRACE) << "try unlocking expert " << layer_id << "." << expert_id << ": returning it's gpu memory " << expert_handler->gpu_data;
    {
      TRACE_EVENT_GURAD(kCacheLib, "try_release_expert.lock_unused_mems");
      unused_mems_lock.lock();
    }
    unused_mems.push_back(expert_handler->gpu_data);
    unused_mems_lock.unlock();
    expert_handler->gpu_data = nullptr;
  }
  auto unlock_success = expert_handler->expert_status.try_unlock(kUsing, kFetching);
  LOG(TRACE) << "try unlocking expert " << layer_id << "." << expert_id << " success:" << unlock_success;
}
void PrefetchMngr::try_release_expert_in_layer(int layer_id) {
  TRACE_EVENT_GURAD(kCacheLib, "try_release:" + std::to_string(layer_id));
  for (int expert_id = 0; expert_id < metas->num_expert; expert_id++) {
    try_release_expert(layer_id, expert_id);
  }
}
void PrefetchMngr::launch_prefetch_thread() {
  prefetch_thread = std::thread([this]() { this->thread_func(); });
}
void PrefetchEngine::init_predictor(std::string model_path) {
  predictor = std::make_shared<Predictor>(this->metas);
  predictor->load_model(model_path);
}
PrefetchMngr::PrefetchMngr(std::shared_ptr<ModuleMeta> metas,
                           std::shared_ptr<ModelLoader> model_loader,
                           std::shared_ptr<Predictor> predictor)
    : metas(metas), model_loader(model_loader), predictor(predictor) {
  per_layer_job_queues.resize(metas->num_layer);
  prefetched_experts.resize(metas->num_layer);
  CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}
void PrefetchMngr::record_then_predict_and_launch(int layer_id, torch::Tensor experts) {
  TRACE_EVENT_GURAD(kCacheLib, "record_then_predict_and_launch");
  LOG_BLOCK(DEBUG, logger, {
    logger << "actual " << layer_id << ":" << tensor_to_str(experts);
  });
  predictor->add_one_layer(layer_id, experts);
  if (layer_id == metas->num_layer - 1) {
    auto prob = predictor->predict().reshape({metas->num_layer, metas->num_expert});
    auto sorted = prob.sort(-1, true);
    auto predicted_expert_prob = std::get<0>(sorted).slice(1, 0, metas->num_predict_expert_per_layer);
    auto predicted_expert = std::get<1>(sorted).slice(1, 0, metas->num_predict_expert_per_layer);

    // for (int l = 0; l < metas->num_layer; l++) {
    //   // auto cur_layer_predicted_expert = predicted_expert[l].slice(0, 0, metas->num_predict_expert_per_layer);
    //   auto cur_layer_predicted_expert = predicted_expert[l];
    //   LOG(DEBUG) << "predicted expert" << l << ":" << tensor_to_str(cur_layer_predicted_expert);
    //   // LOG(DEBUG) << "predicted prob  " << l << ":" << tensor_to_str(predicted_expert_prob[l]);
    //   this->add_one_layer_task(l, cur_layer_predicted_expert);
    // }

    LOG_BLOCK(DEBUG, logger, {
      for (int l = 0; l < metas->num_layer; l++) {
        logger << "predicted expert" << l << ":" << tensor_to_str(predicted_expert[l]);
      }
    });
    this->add_multi_layer_task(predicted_expert);

    predictor->clear_access_buffer();
  }
}
void PrefetchMngr::add_multi_layer_task(torch::Tensor experts) {
  TRACE_EVENT_GURAD(kCacheLib, "add_multi_layer_task");
  size_t per_layer_num_expert = experts.size(1);
  lock_queue();
  for (int layer_idx = 0; layer_idx < metas->num_layer; layer_idx++) {
    CHECK(per_layer_job_queues[layer_idx].empty());
    int64_t* expert_idxs = experts[layer_idx].data_ptr<int64_t>();
    for (int i = 0; i < per_layer_num_expert; i++) {
      LOG(TRACE) << "adding prefetch task " << layer_idx << "," << expert_idxs[i];
      if (prefetched_experts[layer_idx].find(expert_idxs[i]) != prefetched_experts[layer_idx].end()) {
        LOG(TRACE) << "skip add prefetch task " << layer_idx << "," << expert_idxs[i];
        continue;
      }
      add_tasks_for_one_expert(layer_idx, expert_idxs[i], &per_layer_job_queues[layer_idx]);
    }
  }
  unlock_queue();
}
