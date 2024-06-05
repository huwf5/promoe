#pragma once
#include <bitset>
#include <unistd.h>
#include <unordered_map>
#include <string>
#include <unordered_set>
#include <vector>
#include <queue>
#include <cuda_runtime.h>
#include <semaphore.h>

#include "utils.hpp"
#include "model_loader.hpp"
#include "predictor.hpp"
#include "cache.hpp"

class PrefetchTask {
 public:
  int layer_idx, expert_idx, mem_buf_idx;
  bool is_precise = false;
  ExpertHandler* expert = nullptr;
  std::string toString() const {
    std::stringstream ss;
    ss << layer_idx << "." << expert_idx << "." << mem_buf_idx << ", precise " << (is_precise?"true":"false");
    return ss.str();
  }
};

class Queue {
  std::vector<PrefetchTask> queue_buffer;
  int start, stop;
  void extend() {
    auto orig_size = queue_buffer.size();
    queue_buffer.resize(queue_buffer.size() * 2);
    if (start < stop) { return; }
    std::copy(queue_buffer.begin(), queue_buffer.begin() + stop, queue_buffer.begin() + orig_size);
    stop = orig_size + stop;
    // {
    //   std::vector<PrefetchTask> new_queue;
    //   new_queue.resize(queue_buffer.size() * 2);
    //   auto new_tail = new_queue.begin();
    //   if (start < stop) {
    //     new_tail = std::copy(queue_buffer.begin() + start, queue_buffer.begin() + stop, new_tail);
    //   } else {
    //     new_tail = std::copy(queue_buffer.begin() + start, queue_buffer.end(), new_tail);
    //     new_tail = std::copy(queue_buffer.begin(), queue_buffer.begin() + stop, new_tail);
    //   }
    //   start = 0;
    //   stop = new_tail - new_queue.begin();
    //   queue_buffer.swap(new_queue);
    // }
  }
  int next(int a) { return (a + 1) % queue_buffer.size(); }
 public:
  std::unordered_map<int, int> expert_to_remaining_task;
  Queue() : queue_buffer(10), start(0), stop(0) {}
  void clear() {
    expert_to_remaining_task.clear();
    start = stop;
  }
  bool empty() {
    return start == stop;
  }
  PrefetchTask front() {
    CHECK(empty() == false);
    return queue_buffer[start];
  }
  void pop() {
    CHECK(empty() == false);
    expert_to_remaining_task[queue_buffer[start].expert_idx] -= 1;
    start = next(start);
  }
  void push(PrefetchTask task) {
    if (next(stop) == start) {
      extend();
    }
    queue_buffer[stop] = task;
    stop = next(stop);
    if (expert_to_remaining_task.find(task.expert_idx) == expert_to_remaining_task.end()) {
      expert_to_remaining_task[task.expert_idx] = 0;
    }
    expert_to_remaining_task[task.expert_idx] += 1;
  }
  int remaining_task(int expert_id) {
    auto iter = expert_to_remaining_task.find(expert_id);
    if (iter == expert_to_remaining_task.end()) { return 0; }
    return iter->second;
  }
};

#ifdef DEAD_CODE
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
#endif

class ExpertHandler;

class PrefetchMngr {
  cudaStream_t stream;
  std::shared_ptr<ModuleMeta> metas;
  std::shared_ptr<ModelLoader> model_loader;
  std::shared_ptr<Predictor> predictor;
  std::shared_ptr<CacheMngr> cache;
  /**
   * protected by queue_lock
   */
  std::vector<Queue> per_layer_job_queues; // the fetching thread takes out the first task from queue, then execute it.
  /**
   * protected by queue_lock
   */
  Queue precise_job_queue;
  std::thread prefetch_thread;
  std::thread predict_thread;
  sem_t predictor_send, predictor_done;
  std::function<void()> try_wait_pretictor_done;
  volatile bool thread_exit_mark = false;
  AtomicQueueLock queue_lock;

  /**
   * protected by queue_lock
   */
  PrefetchTask previous_task;
  /**
   * protected by queue_lock
   */
  PrefetchTask current_task;
  // ExpertHandler * previous_task = nullptr;

  void add_tasks_for_one_expert(int layer_idx, int expert_idx, Queue* queue,
                                int starting_mem_buffer = 0);

  void prefetch_thread_func();
  void predict_thread_func();
  void do_one_task(PrefetchTask *task);

  inline void lock_queue() {
    queue_lock.lock();
  }
  inline void unlock_queue() {
    queue_lock.unlock();
  }

  void preempt_one_layer_(int layer_idx, int64_t *expert_idxs, size_t num_expert);

  void add_one_layer_task_(int layer_idx, int64_t *expert_idxs, size_t num_expert);

public:
  ~PrefetchMngr();

  PrefetchMngr(std::shared_ptr<ModuleMeta> metas,
               std::shared_ptr<ModelLoader> model_loader,
               std::shared_ptr<Predictor> predictor);
  void init_gpu_mem_buffer(size_t num_buffers);

  void add_one_layer_task(int layer_idx, torch::Tensor experts);
  void add_multi_layer_task(torch::Tensor experts);
  void preempt_one_layer(int layer_idx, torch::Tensor experts);

  void record_then_predict_and_launch(int layer_id, torch::Tensor experts);

  void wait_and_lock_expert(int layer_id, int expert_id);
  void try_release_expert(int layer_id, int expert_id);
  void try_release_expert_in_layer(int layer_id);

  void launch_thread();
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