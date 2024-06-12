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

#include "worker.hpp"
#include "utils.hpp"
#include "model_loader.hpp"
#include "predictor.hpp"
#include "cache.hpp"

class ExpertHandler;

class FetchScheduleTaskBase {
 public:
  enum TaskType {
    kIdle,
    kPreempt,
    kFetchDone,
  };
  TaskType task_type;
  FetchScheduleTaskBase(TaskType task_type) : task_type(task_type) {}
  virtual ~FetchScheduleTaskBase() {}
};
class IdleTask : public FetchScheduleTaskBase {
 public:
  IdleTask() : FetchScheduleTaskBase(kIdle) {}
};
class PreemptTask : public FetchScheduleTaskBase {
 public:
  PreemptTask() : FetchScheduleTaskBase(kPreempt) {}
  int layer_idx;
  int64_t* expert_idxs;
  size_t num_expert;
};
class FetchDoneTask : public FetchScheduleTaskBase, public CopyTask {
 public:
  FetchDoneTask() : FetchScheduleTaskBase(kFetchDone) {}
  void init(CopyTask* task) {
    this->mem_buf_idx = task->mem_buf_idx;
    this->expert = task->expert;
    this->is_precise = task->is_precise;
  }
};

class FetchScheduleWorker : public WorkerThread<FetchScheduleTaskBase*> {
  ModuleMeta* metas;
  CacheMngr* cache;
  PrefetchMngr* prefetcher;

  IdleTask idle_task;
  CopyTask copy_task; // we allow only one ongoing copy task
  friend class FetchWorker;
  FetchDoneTask copy_done_task;

  bool send_one_job(CopyTask *task);
  void do_one_task_impl(IdleTask *task);
  void do_one_task_impl(PreemptTask *task);
  void do_one_task_impl(FetchDoneTask *task);

public:
  void init(ModuleMeta* metas, CacheMngr *cache, PrefetchMngr *prefetcher);
  void do_one_task_impl(FetchScheduleTaskBase *task);
};

class PrefetchMngr {
  friend class FetchScheduleWorker;
  friend class FetchWorker;
  using TaskQueue = Queue<CopyTask>;
  cudaStream_t stream;
  std::shared_ptr<ModuleMeta> metas;
  std::shared_ptr<ModelLoader> model_loader;
  std::shared_ptr<Predictor> predictor;
  std::shared_ptr<CacheMngr> cache;
  /**
   * protected by queue_lock
   */
  std::vector<TaskQueue> per_layer_job_queues; // the fetching thread takes out the first task from queue, then execute it.
  /**
   * protected by queue_lock
   */
  TaskQueue precise_job_queue;
  std::shared_ptr<FetchScheduleWorker> fetch_schedule_thread;
  std::shared_ptr<FetchWorker>         fetch_thread;
  std::shared_ptr<PredictWorker>       predict_thread;
  std::shared_ptr<ExpertUnlockWorker>  expert_unlocker_thread;

  AtomicQueueLock task_queue_lock;

  /**
   * protected by queue_lock
   */
  CopyTask current_task;

  void add_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue* queue,
                                int starting_mem_buffer = 0, bool is_precise = false);

  void pop_next_task(CopyTask &task, bool &found);

  inline void lock_task_queue() {
    task_queue_lock.lock();
  }
  inline void unlock_task_queue() {
    task_queue_lock.unlock();
  }

  void preempt_one_layer_(int layer_idx, int64_t *expert_idxs, size_t num_expert);


public:
  ~PrefetchMngr();

  PrefetchMngr(std::shared_ptr<ModuleMeta> metas,
               std::shared_ptr<ModelLoader> model_loader,
               std::shared_ptr<Predictor> predictor);
  void init_gpu_mem_buffer(size_t num_buffers);

  void add_one_layer_task(int layer_idx, torch::Tensor experts);
  // void add_multi_layer_task(torch::Tensor experts);
  void add_one_layer_task(int layer_idx, int64_t *expert_idxs, size_t num_expert);

  /**
   * for already in cache, directly lock it
   * for fetching, lock it after fetching is done
   */
  void preempt_and_launch_one_layer(int layer_idx, torch::Tensor experts);

  void record_then_predict_and_prefetch(int layer_id, torch::Tensor experts);

  void wait_expert(int layer_id, int expert_id);
  void mark_expert_using(int layer_id, int expert_id);

  void launch_thread();
};
