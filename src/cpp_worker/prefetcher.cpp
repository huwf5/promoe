#include "prefetcher.hpp"
#include "logging.hpp"

void PrefetchMngr::preempt_one_layer_(int layer_idx, int64_t *expert_idxs,
                                      size_t num_expert) {
  LOG(DEBUG) << "preempting one layer " << layer_idx;
  {
    int prev_layer_idx = (layer_idx + metas->num_layer - 1) % metas->num_layer;
    LOG(DEBUG) << "preempting one layer " << layer_idx << ", releasing previous layer " << prev_layer_idx << " first";
    for (int i = 0; i < metas->num_expert; i++) {
      try_release_expert(prev_layer_idx, i);
    }
  }
  std::unordered_set<int> expert_idxs_set;
  for (int i = 0; i < num_expert; i++) {
    expert_idxs_set.insert(expert_idxs[i]);
  }
  lock_queue();
  //// empty queue, insert all missing experts
  if (per_layer_job_queues[layer_idx].empty()) {
    LOG(DEBUG) << "preempting one layer " << layer_idx << ", queue is empty";
    for (int i = 0; i < num_expert; i++) {
      if (prefetched_experts[layer_idx].find(expert_idxs[i]) !=
          prefetched_experts[layer_idx].end()) {
        LOG(DEBUG) << "preempting one layer " << layer_idx << ", skipping [" << i << "]=" << expert_idxs[i] << " since it's in gpu";
        continue;
      }
      LOG(DEBUG) << "preempting one layer " << layer_idx << ", add task for [" << i << "]=" << expert_idxs[i];
      add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue);
    }
  } else {
    auto first_task = per_layer_job_queues[layer_idx].front();
    LOG(DEBUG) << "preempting one layer " << layer_idx << ", queue is not empty, first task is " << first_task.layer_idx << "." << first_task.expert_idx << "." << first_task.mem_buf_idx;
    per_layer_job_queues[layer_idx].clear();
    //// queue with pending tasks, but the first task is wrongly predicted
    if (expert_idxs_set.find(first_task.expert_idx) == expert_idxs_set.end() ||
        first_task.mem_buf_idx == 0) {
      LOG(DEBUG) << "preempting one layer " << layer_idx << ", first task is miss predicted or not started. just clear entire queue.";
      for (int i = 0; i < num_expert; i++) {
        if (prefetched_experts[layer_idx].find(expert_idxs[i]) !=
            prefetched_experts[layer_idx].end()) {
          continue;
        }
        add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue);
      }
    } else {
      LOG(DEBUG) << "preempting one layer " << layer_idx << ", first task is correctly predicted and started";
      int i = 0;
      for (; i < num_expert && expert_idxs[i] < first_task.expert_idx; i++) {
        if (prefetched_experts[layer_idx].find(expert_idxs[i]) !=
            prefetched_experts[layer_idx].end()) {
          continue;
        }
        add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue);
      }

      CHECK(first_task.expert_idx == expert_idxs[i]);
      add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue,
                               first_task.mem_buf_idx);
      i++;

      for (; i < num_expert; i++) {
        if (prefetched_experts[layer_idx].find(expert_idxs[i]) !=
            prefetched_experts[layer_idx].end()) {
          continue;
        }
        add_tasks_for_one_expert(layer_idx, expert_idxs[i], &precise_job_queue);
      }
    }
  }
  unlock_queue();
}
void PrefetchMngr::add_one_layer_task_(int layer_idx, int64_t *expert_idxs,
                                       size_t num_expert) {
  CHECK(per_layer_job_queues[layer_idx].empty());
  for (int i = 0; i < num_expert; i++) {
    LOG(DEBUG) << "adding prefetch task " << layer_idx << "," << expert_idxs[i];
    lock_queue();
    if (prefetched_experts[layer_idx].find(expert_idxs[i]) !=
        prefetched_experts[layer_idx].end()) {
      LOG(DEBUG) << "skip add prefetch task " << layer_idx << "," << expert_idxs[i];
      unlock_queue();
      continue;
    }
    add_tasks_for_one_expert(layer_idx, expert_idxs[i], &per_layer_job_queues[layer_idx]);
    unlock_queue();
  }
}
void PrefetchMngr::do_one_task(PrefetchTask *task) {
  LOG(DEBUG) << "do one prefetch task " << task->layer_idx << "," << task->expert_idx << "," << task->mem_buf_idx;
  if (previous_task.expert != nullptr && previous_task.expert != task->expert &&
      previous_task.mem_buf_idx != metas->num_per_expert_param-1) {
    LOG(DEBUG) << "removing partially fetched expert " << previous_task.layer_idx << "," << previous_task.expert_idx;
    prefetched_experts[previous_task.layer_idx].erase(
        previous_task.expert_idx);
    unused_mems_lock.lock();
    CHECK(previous_task.expert->gpu_data != nullptr);
    unused_mems.push_back(previous_task.expert->gpu_data);
    previous_task.expert->gpu_data = nullptr;
    unused_mems_lock.unlock();
  }
  previous_task = *task;
  if (task->expert->gpu_data == nullptr) {
    LOG(DEBUG) << "assigning gpu mem for expert " << task->layer_idx << "," << task->expert_idx << "," << task->mem_buf_idx;
    unused_mems_lock.lock();
    if (unused_mems.size() > 0) {
      task->expert->gpu_data = unused_mems.back();
      CHECK(task->expert->gpu_data != nullptr);
      LOG(DEBUG) << "assigning gpu mem " << task->expert->gpu_data << " for expert " << task->layer_idx << "," << task->expert_idx << "," << task->mem_buf_idx;
      unused_mems.pop_back();
    } else {
      CHECK(false) << "no remaining mem buffer";
    }
    unused_mems_lock.unlock();
    // fixme: find one loc from cache
    prefetched_experts[task->layer_idx][task->expert_idx] = task->expert;
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
  prefetch_worker = std::make_shared<PrefetchMngr>(metas, model_loader);
}
void PrefetchMngr::add_tasks_for_one_expert(int layer_idx, int expert_idx, Queue* queue,
                                            int starting_mem_buffer) {
  auto expert_handler = model_loader->get_source(layer_idx, expert_idx);
  for (int j = starting_mem_buffer;
       j < expert_handler->host_data.mem_buffers.size(); j++) {
    PrefetchTask task;
    task.layer_idx = layer_idx;
    task.expert_idx = expert_idx;
    task.mem_buf_idx = j;
    task.expert = expert_handler;
    queue->push(task);
    LOG(DEBUG) << "add prefetch task for one param " << layer_idx << "," << expert_idx << "," << j;
  }
}
void PrefetchMngr::thread_func() {
  while (true) {
    PrefetchTask task;
    bool found = false;
    lock_queue();
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

    if (found) {
      do_one_task(&task);
      unlock_queue();
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
  LOG(DEBUG) << "waiting expert " << layer_id << "." << expert_id;
  model_loader->get_source(layer_id, expert_id)
      ->expert_status.lock(kReady, kUsing);
  LOG(DEBUG) << "waiting expert " << layer_id << "." << expert_id << " success";
}
void PrefetchMngr::try_release_expert(int layer_id, int expert_id) {
  LOG(DEBUG) << "try unlocking expert " << layer_id << "." << expert_id;
  auto expert_handler = model_loader->get_source(layer_id, expert_id);
  if (expert_handler->expert_status.is_locked(kUsing) == false) {
    LOG(DEBUG) << "try unlocking expert " << layer_id << "." << expert_id << ": it's not locked";
    return;
  }
  lock_queue();
  prefetched_experts[layer_id].erase(expert_id);
  unlock_queue();
  // fixme: the memory may should not be released here. add a cache module
  if (expert_handler->gpu_data != nullptr) {
    LOG(DEBUG) << "try unlocking expert " << layer_id << "." << expert_id << ": returning it's gpu memory " << expert_handler->gpu_data;
    unused_mems_lock.lock();
    unused_mems.push_back(expert_handler->gpu_data);
    unused_mems_lock.unlock();
    expert_handler->gpu_data = nullptr;
  }
  auto unlock_success = expert_handler->expert_status.try_unlock(kUsing, kFetching);
  LOG(DEBUG) << "try unlocking expert " << layer_id << "." << expert_id << " success:" << unlock_success;
}
void PrefetchMngr::try_release_expert_in_layer(int layer_id) {
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
                           std::shared_ptr<ModelLoader> model_loader)
    : metas(metas), model_loader(model_loader) {
  per_layer_job_queues.resize(metas->num_layer);
  prefetched_experts.resize(metas->num_layer);
  CUDA_CALL(cudaStreamCreate(&stream))
}
