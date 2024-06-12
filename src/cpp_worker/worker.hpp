#pragma once
#include <semaphore.h>
#include "cache.hpp"
#include "utils.hpp"
#include "model_loader.hpp"
#include "predictor.hpp"

template<typename TASK_T>
class WorkerThread {
  Queue<TASK_T> queue;
  // TASK_T current_task;
  AtomicQueueLock queue_lock;
  std::thread worker_thread;
  volatile bool exit_mark = false;
 protected:
  using progress_handler_t = int64_t;
 private:
  progress_handler_t queued = 0;
  std::atomic<progress_handler_t> progress{0};
 protected:
  bool should_exit() {
    return exit_mark;
  }
  virtual void do_one_task_impl(TASK_T task) {}
  inline void do_one_task(TASK_T task) { 
    do_one_task_impl(task);
    progress.fetch_add(1);
  }
 public:
  WorkerThread() : progress(0) {}
  virtual void exit() {
    exit_mark = true;
    if (worker_thread.joinable()) { worker_thread.join(); }
  }
  void launch() {
    worker_thread = std::thread([this](){
      thread_func();
    });
  }
 private:
  void thread_func() {
    while (should_exit() == false) {
      queue_lock.lock();
      if (queue.empty()) {
        queue_lock.unlock();
        // usleep(10);
      } else {
        auto current_task = queue.front();
        queue.pop();
        queue_lock.unlock();
        do_one_task(current_task);
      }
    }
  }
 public:
  progress_handler_t add_one_task(TASK_T task) {
    queue_lock.lock();
    auto ret = queued++;
    queue.push(task);
    queue_lock.unlock();
    return ret;
  }
  void wait_progress(progress_handler_t handle) {
    // todo: handle overflow
    while(progress.load() <= handle) {};
  }
  virtual ~WorkerThread() {}
};

template<>
class WorkerThread<void> : public WorkerThread<DummyStruct> {
  using progress_handler_t = WorkerThread<DummyStruct>::progress_handler_t;
 protected:
  virtual void do_one_task_impl() = 0;
  void do_one_task_impl(DummyStruct task) override {
    do_one_task_impl();
  }
  progress_handler_t add_one_task() {
    return WorkerThread<DummyStruct>::add_one_task(DummyStruct());
  }
};

class BaseTask {
  public:
};

class PrefetchMngr;
/**
 * Param Fetcher
 */
class CopyTask : public BaseTask {
 public:
  int mem_buf_idx;
  bool is_precise = false;
  ExpertHandler *expert = nullptr;
  std::function<void()> lambda_wait = [](){};
  std::string toString() const {
    std::stringstream ss;
    if (expert) {
      ss << expert->toString() << "." << mem_buf_idx << ", precise " << (is_precise?"true":"false");
    } else {
      ss << "null";
    }
    return ss.str();
  }
};

class FetchScheduleWorker;

class FetchWorker : public WorkerThread<CopyTask*> {
  ModuleMeta* metas;
  FetchScheduleWorker* fetch_schedule_thread;
  cudaStream_t stream;
  friend class PrefetchMngr;
 public:
  void init(ModuleMeta* metas, FetchScheduleWorker* fetch_schedule_thread, cudaStream_t stream) {
    this->metas = metas;
    this->fetch_schedule_thread = fetch_schedule_thread;
    this->stream = stream;
  }
 protected:
  void do_one_task_impl(CopyTask *task) override;
};

/**
 * Expert Unlocker, which tracks expert forward progress in python
 */
class ExpertUnlockWorker : public WorkerThread<ExpertHandler*> {
  AtomicQueueLock expert_usage_queue_lock;
 protected:
  void do_one_task_impl(ExpertHandler *task) override;
};

/**
 * Predict Worker
 */
class PredictWorker : public WorkerThread<void> {
  FetchScheduleWorker* fetch_schedule_thread;
  Predictor  * predictor;
  CacheMngr  * cache;
  ModuleMeta * metas;
  sem_t prefetch_layer_budget, prefetch_layer_progress;
  friend class PrefetchMngr;
 public:
  PredictWorker() : WorkerThread<void>() {}
  void init(FetchScheduleWorker* fetch_schedule_thread, Predictor* predictor, CacheMngr* cache, ModuleMeta* metas) {
    this->fetch_schedule_thread = fetch_schedule_thread;
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
 protected:
  void do_one_task_impl() override;
};
