#pragma once
#include <bitset>
#include <unistd.h>
#include <unordered_map>
#include <string>
#include <unordered_set>
#include <vector>
#include <queue>
#include <cuda_runtime.h>

#include "utils.hpp"
#include "model_loader.hpp"
#include "predictor.hpp"

class PrefetchTask {
 public:
  int layer_idx, expert_idx, mem_buf_idx;
  ExpertHandler* expert = nullptr;
};

class Queue {
  std::queue<PrefetchTask> queue;
 public:
  void clear() {
    auto empty_queue = std::queue<PrefetchTask>();
    queue.swap(empty_queue);
  }
  void push(PrefetchTask task) {
    queue.push(task);
  }
  bool empty() {
    return queue.empty();
  }
  PrefetchTask front() {
    return queue.front();
  }
  void pop() {
    queue.pop();
  }

};

class ExpertHandler;

class PrefetchMngr {
  cudaStream_t stream;
  std::shared_ptr<ModuleMeta> metas;
  std::shared_ptr<ModelLoader> model_loader;
  std::vector<Queue> per_layer_job_queues; // the fetching thread takes out the first task from queue, then execute it.
  std::vector<std::unordered_map<int, ExpertHandler*>> prefetched_experts; // the ongoing job also lives in here.
  std::thread prefetch_thread;
  std::vector<ExpertMemHanlder*> unused_mems;
  AtomicLock unused_mems_lock;
  AtomicLock queue_lock;

  ExpertHandler * previous_task = nullptr;

  void add_tasks_for_one_expert(int layer_idx, int exper_idx,
                                int starting_mem_buffer = 0);

  void thread_func();
  void do_one_task(PrefetchTask *task);

  inline void lock_queue() {
    queue_lock.lock();
  }
  inline void unlock_queue() {
    queue_lock.unlock();
  }

  void preempt_one_layer_(int layer_idx, int *expert_idxs, size_t num_expert);

  void add_one_layer_task_(int layer_idx, int *expert_idxs, size_t num_expert);

public:
  PrefetchMngr(std::shared_ptr<ModuleMeta> metas,
               std::shared_ptr<ModelLoader> model_loader);
  void init_gpu_mem_buffer(size_t num_buffers);

  void add_one_layer_task(int layer_idx, torch::Tensor experts);
  void preempt_one_layer(int layer_idx, torch::Tensor experts);

  void wait_and_lock_expert(int layer_id, int expert_id);
  void try_release_expert(int layer_id, int expert_id);

  void launch_prefetch_thread();
};

class PrefetchEngine {
 public:
  std::shared_ptr<PrefetchMngr> prefetch_worker;
  std::shared_ptr<ModuleMeta> metas;
  std::shared_ptr<ModelLoader> model_loader;
  std::shared_ptr<Predictor> predictor;
  PrefetchEngine() {}
  void init_predictor(std::string model_path);
  void init_meta(int num_layer, int num_expert) {
    metas = std::make_shared<ModuleMeta>(num_layer, num_expert);
  }
  void init_model_loader() {
    model_loader = std::make_shared<ModelLoader>(metas);
  }
  void init_prefetch_worker();

  std::shared_ptr<ModelLoader> get_model_loader() {
    return model_loader;
  }

};