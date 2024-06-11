#pragma once
#include <semaphore.h>
#include "cache.hpp"
#include "utils.hpp"
#include "model_loader.hpp"
#include "predictor.hpp"

template<typename TASK_T>
class WorkerThread {
  Queue<TASK_T> queue;
  TASK_T current_task;
  AtomicQueueLock queue_lock;
  std::thread worker_thread;
  volatile bool exit_mark = false;
 protected:
  bool should_exit() {
    return exit_mark;
  }
 public:
  virtual void exit() {
    exit_mark = true;
    if (worker_thread.joinable()) { worker_thread.join(); }
  }
  void launch() {
    worker_thread = std::thread([this](){
      thread_func();
    });
  }
  void thread_func() {
    while (should_exit() == false) {
      queue_lock.lock();
      if (queue.empty()) {
        queue_lock.unlock();
        usleep(10);
      } else {
        current_task = queue.front();
        queue.pop();
        queue_lock.unlock();
        do_one_task(current_task);
      }
    }
  }
  virtual void add_one_task(TASK_T task) {
    queue_lock.lock();
    queue.push(task);
    queue_lock.unlock();
  }
  virtual void do_one_task(TASK_T task) {}
  virtual ~WorkerThread() {}
};

class BaseTask {
  public:
};

/**
 * Param Fetcher
 */

class CopyTask : public BaseTask {
 public:
  int mem_buf_idx;
  bool is_precise = false;
  ExpertHandler *expert = nullptr;
  ExpertMemHanlder *dst = nullptr;
  std::function<void()> lambda_wait = [](){};
  std::string toString() const {
    std::stringstream ss;
    ss << expert->toString() << "." << mem_buf_idx << ", precise " << (is_precise?"true":"false");
    return ss.str();
  }
};

class FetchWorker : public WorkerThread<CopyTask*> {
  std::shared_ptr<ModuleMeta> metas;
  cudaStream_t stream;
  friend class PrefetchMngr;
 public:
  FetchWorker(std::shared_ptr<ModuleMeta> metas) : WorkerThread<CopyTask*>(), metas(metas) {}
  void do_one_task(CopyTask *task) override;
};

/**
 * Expert Unlocker, which tracks expert forward progress in python
 */
class ExpertUnlockWorker : public WorkerThread<ExpertHandler*> {
  AtomicQueueLock expert_usage_queue_lock;
 public:
   void do_one_task(ExpertHandler *task) override;
};

/**
 * Predict Worker
 */
class PrefetchMngr;
class PredictWorker : public WorkerThread<BaseTask*> {
  PrefetchMngr  * prefetcher;
  Predictor  * predictor;
  CacheMngr  * cache;
  ModuleMeta * metas;
  sem_t prefetch_layer_budget, prefetch_layer_progress;
  friend class PrefetchMngr;
 public:
  PredictWorker() : WorkerThread<BaseTask*>() {}
  void init(PrefetchMngr* prefetcher, Predictor* predictor, CacheMngr* cache, ModuleMeta* metas) {
    this->prefetcher = prefetcher;
    this->predictor = predictor;
    this->cache = cache;
    this->metas = metas;
    sem_init(&prefetch_layer_budget, 0, metas->max_prefetch_layer_distance);
    sem_init(&prefetch_layer_progress, 0, 0); 
  }
  void add_prefetch_layer_budget() {
    sem_post(&prefetch_layer_budget);
  }
  void consume_prefetch_layer_progress() {
    sem_wait(&prefetch_layer_progress);
  }
  void do_one_task(BaseTask *_) override;
};

class PreemptTask {
  public:
};