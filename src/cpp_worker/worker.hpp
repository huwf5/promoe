#pragma once
#include <semaphore.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <vector>
#include "cache.hpp"
#include "utils.hpp"
#include "model_loader.hpp"
#include "predictor.hpp"
#include "erpp_encoder_predictor.hpp"
#include "profiler.hpp"
#include <pthread.h>

template<typename TASK_T>
class WorkerThreadBase {
 protected:
  Queue<TASK_T> queue;
  std::thread worker_thread;
  std::atomic<bool> exit_mark_atomic{false};
  std::atomic<bool> active{false};
  using progress_handler_t = int64_t;
  progress_handler_t queued = 0;
  std::atomic<progress_handler_t> progress{0};
 protected:
  bool should_exit() {
    return std::atomic_load_explicit(&exit_mark_atomic, std::memory_order_relaxed);
  }
  virtual void do_one_task_impl(TASK_T task) {}
  inline void do_one_task(TASK_T task) {
    try {
      do_one_task_impl(task);
      progress.fetch_add(1);
      active.store(false, std::memory_order_release);
    } catch (...) {
      active.store(false, std::memory_order_release);
      throw;
    }
  }
 public:
  WorkerThreadBase() : progress(0) {}
  virtual void exit() = 0;
  void launch() {
    worker_thread = std::thread([this](){
      thread_func();
    });
  }
 protected:
  virtual void thread_func() = 0;
 public:
  void set_cpu_affinity(std::vector<int> cpu_ids) {
    if (worker_thread.native_handle()) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      for (auto cpu_id : cpu_ids) {
        CPU_SET(cpu_id, &cpuset);
      }
      int result = pthread_setaffinity_np(worker_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
      if (result != 0) {
        std::cerr << "Error setting thread affinity: " << std::strerror(result) << std::endl;
      }
    }
  }
  virtual progress_handler_t add_one_task(TASK_T task) = 0;
  bool is_active() const { return active.load(std::memory_order_acquire); }
  virtual bool queue_empty() = 0;
  virtual void clear_pending_tasks() = 0;
  void wait_until_idle() {
    while (is_active() || !queue_empty()) {}
  }
  void wait_progress(progress_handler_t handle) {
    // todo: handle overflow
    while(progress.load() <= handle) {};
    // while(progress.load() <= handle) { std::this_thread::yield(); }
  }
  virtual ~WorkerThreadBase() {}
};

template<typename TASK_T>
class WorkerThreadMutex : public WorkerThreadBase<TASK_T> {
  std::mutex queue_mutex;
  std::condition_variable cv;
 public:
  WorkerThreadMutex() : WorkerThreadBase<TASK_T>() {}
  void exit() override {
    std::atomic_store_explicit(&this->exit_mark_atomic, true, std::memory_order_relaxed);
    cv.notify_one();
    if (this->worker_thread.joinable()) { this->worker_thread.join(); }
  }
 protected:
  void thread_func() override {
    while (!this->should_exit()) {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cv.wait(lock, [this]{ return !this->queue.empty() || this->should_exit(); });

      if (this->should_exit()) {
        break;
      }

      auto current_task = this->queue.front();
      this->queue.pop();
      this->active.store(true, std::memory_order_release);
      lock.unlock();

      this->do_one_task(current_task);
    }
  }
 public:
  using progress_handler_t = typename WorkerThreadBase<TASK_T>::progress_handler_t;
  progress_handler_t add_one_task(TASK_T task) override {
    std::lock_guard<std::mutex> lock(queue_mutex);
    auto ret = this->queued++;
    this->queue.push(task);
    cv.notify_one();
    return ret;
  }
  bool queue_empty() override {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return this->queue.empty() && !this->active.load(std::memory_order_acquire);
  }
  void clear_pending_tasks() override {
    std::lock_guard<std::mutex> lock(queue_mutex);
    while (!this->queue.empty()) {
      this->queue.pop();
    }
  }
};



template<typename TASK_T>
class WorkerThreadSpin : public WorkerThreadBase<TASK_T> {
  AtomicQueueLock queue_lock;
 public:
  WorkerThreadSpin() : WorkerThreadBase<TASK_T>() {}
  void exit() override {
    std::atomic_store_explicit(&this->exit_mark_atomic, true, std::memory_order_relaxed);
    if (this->worker_thread.joinable()) { this->worker_thread.join(); }
  }
 protected:
  void thread_func() override {
    while (!this->should_exit()) {
      queue_lock.lock();
      if (this->queue.empty()) {
        queue_lock.unlock();
        // usleep(10);
      } else {
        auto current_task = this->queue.front();
        this->queue.pop();
        this->active.store(true, std::memory_order_release);
        queue_lock.unlock();
        this->do_one_task(current_task);
      }
    }
  }
 public:
  using progress_handler_t = typename WorkerThreadBase<TASK_T>::progress_handler_t;
  progress_handler_t add_one_task(TASK_T task) override {
    queue_lock.lock();
    auto ret = this->queued++;
    this->queue.push(task);
    queue_lock.unlock();
    return ret;
  }
  bool queue_empty() override {
    queue_lock.lock();
    bool empty = this->queue.empty() && !this->active.load(std::memory_order_acquire);
    queue_lock.unlock();
    return empty;
  }
  void clear_pending_tasks() override {
    queue_lock.lock();
    while (!this->queue.empty()) {
      this->queue.pop();
    }
    queue_lock.unlock();
  }
};

template<typename TASK_T>
using WorkerThread = WorkerThreadSpin<TASK_T>;

class BaseTask {
  public:
};

class PrefetchMngr;
/**
 * Param Fetcher
 */
class CopyTask : public BaseTask {
 public:
  int start_mem_buf_idx, stop_mem_buf_idx;
  bool is_precise = false;
  int64_t forward_epoch = 0;
  int64_t generate_epoch = 0;
  CacheRequestType request_type = kCacheRequestDecoderPredictorPrefetch;
  ExpertHandler *expert = nullptr;
  CacheMngr::CacheLineOccupancyWaiter lambda_wait = [](){};
  std::string toString() const {
    std::stringstream ss;
    if (expert) {
      ss << expert->toString() << ".[" << start_mem_buf_idx << "," << stop_mem_buf_idx
         << "), precise " << (is_precise ? "true" : "false")
         << ", forward_epoch=" << forward_epoch
         << ", generate_epoch=" << generate_epoch
         << ", request_type " << request_type;
    } else {
      ss << "null";
    }
    return ss.str();
  }
};

class FetchScheduleWorker;
class ErppEncoderPredictor;

class FetchWorker : public WorkerThread<CopyTask*> {
  ModuleMeta* metas;
  FetchScheduleWorker* fetch_schedule_thread;
  MemMngrCtx* mem_mngr_ctx;
  cudaStream_t stream;
  friend class PrefetchMngr;
 public:
  void init(ModuleMeta* metas, FetchScheduleWorker* fetch_schedule_thread, MemMngrCtx* mem_mngr_ctx, cudaStream_t stream) {
    this->metas = metas;
    this->fetch_schedule_thread = fetch_schedule_thread;
    this->stream = stream;
    this->mem_mngr_ctx = mem_mngr_ctx;
  }
 protected:
  void do_one_task_impl(CopyTask *task) override;
};

/**
 * Expert Unlocker, which tracks expert forward progress in python
 */
class ExpertUnlockWorker : public WorkerThread<ExpertHandler*> {
 protected:
  void do_one_task_impl(ExpertHandler *task) override;
};

/**
 * Predict Worker
 */
struct PredictJob {
  int input_layer_id = 0;
  int64_t forward_epoch = 0;
  int64_t generate_epoch = 0;
  PredictJob() {}
  PredictJob(int input_layer_id, int64_t forward_epoch = 0, int64_t generate_epoch = 0)
      : input_layer_id(input_layer_id), forward_epoch(forward_epoch), generate_epoch(generate_epoch) {}
};
class AtomicQueue {
 public:
  std::queue<int> queue_;
  AtomicQueueLock lock_;
  void init(int init_val = 0) {
    for (int i = 0; i < init_val; i++) {
      queue_.push(i);
    }
  }
  void reset(int init_val = 0) {
    lock();
    std::queue<int> empty;
    std::swap(queue_, empty);
    for (int i = 0; i < init_val; i++) {
      queue_.push(i);
    }
    unlock();
  }
  void lock() {
    lock_.lock();
  }
  void unlock() {
    lock_.unlock();
  }
  void push(int task) {
    lock();
    queue_.push(task);
    unlock();
  }
  int pop(bool return_size = false) {
    while (true) {
      lock();
      if (queue_.empty()) {
        unlock();
      } else {
        int task = queue_.front();
        queue_.pop();
        int size = queue_.size();
        unlock();
        if (return_size) {
          return size;
        }
        return task;
      }
    }
  }
  int try_pop(bool return_size = false) {
    lock();
    if (queue_.empty()) {
      unlock();
      return -1;
    }
    int task = queue_.front();
    queue_.pop();
    int size = queue_.size();
    unlock();
    if (return_size) {
      return size;
    }
    return task;
  }
};

class SemQueue {
 public:
  sem_t sem_;
  SemQueue() {}
  void init(int init_val = 0) {
    sem_init(&sem_, 0, init_val);
  }
  void reset(int init_val = 0) {
    sem_destroy(&sem_);
    sem_init(&sem_, 0, init_val);
  }
  ~SemQueue() {
    sem_destroy(&sem_);
  }

  void push(int task) {
    sem_post(&sem_);
  }
  int pop(bool return_size = false) {
    sem_wait(&sem_);
    return 0;
  }
  int try_pop(bool return_size = false) {
    if (sem_trywait(&sem_) == 0) {
      // success
      return 0;
    }
    // failed
    return -1;
  }
};

class PredictWorker : public WorkerThread<PredictJob> {
  FetchScheduleWorker* fetch_schedule_thread;
  PredictorBase  * predictor;
  CacheMngr  * cache;
  ModuleMeta * metas;
  std::atomic<int64_t> current_generate_epoch{0};

  PrecisionProfiler * precision_profiler;

  SemQueue    prefetch_layer_budget;
  AtomicQueue prefetch_layer_progress;
  // AtomicQueue prefetch_layer_budget;

  friend class PrefetchMngr;
 public:
  PredictWorker() : WorkerThread<PredictJob>() {}
  void init(FetchScheduleWorker* fetch_schedule_thread, PredictorBase* predictor, CacheMngr* cache, ModuleMeta* metas) {
    this->fetch_schedule_thread = fetch_schedule_thread;
    this->predictor = predictor;
    this->cache = cache;
    this->metas = metas;

    prefetch_layer_budget.init(metas->max_prefetch_layer_distance);
    prefetch_layer_progress.init(0); 
  }
  void add_prefetch_layer_budget();
  int  consume_prefetch_layer_progress() {
    return prefetch_layer_progress.pop();
  }
  void release_prefetch_layer_budget_for_reset() {
    int release_count = metas->num_layer * metas->num_layer;
    if (release_count < 1) {
      release_count = 1;
    }
    for (int i = 0; i < release_count; i++) {
      prefetch_layer_budget.push(0);
    }
  }
  void begin_reset_for_generate() {
    clear_pending_tasks();
    release_prefetch_layer_budget_for_reset();
  }
  void reset_for_generate(int64_t next_generate_epoch) {
    wait_until_idle();
    clear_pending_tasks();
    prefetch_layer_budget.reset(metas->max_prefetch_layer_distance);
    prefetch_layer_progress.reset(0);
    current_generate_epoch.store(next_generate_epoch, std::memory_order_release);
  }
  void on_one_iter_done(int64_t forward_epoch, int64_t generate_epoch);
  void on_moe_attn_input_logits_recorded(int layer_id, int64_t forward_epoch, int64_t generate_epoch);
  void on_moe_layer_logits_recorded(int layer_id, int64_t forward_epoch, int64_t generate_epoch);

protected:
  void do_one_task_impl(PredictJob job) override;
};
struct ErppEncoderPredictJob {
  int64_t forward_epoch = 0;
  int64_t generate_epoch = 0;
  ErppEncoderPredictJob(int64_t forward_epoch = 0, int64_t generate_epoch = 0)
      : forward_epoch(forward_epoch), generate_epoch(generate_epoch) {}
};

class ErppEncoderPredictWorker : public WorkerThread<ErppEncoderPredictJob> {
  FetchScheduleWorker* fetch_schedule_thread = nullptr;
  ErppEncoderPredictor* erpp_encoder_predictor = nullptr;
  ModuleMeta* metas = nullptr;
  std::atomic<int64_t> current_generate_epoch{0};

 public:
  ErppEncoderPredictWorker() : WorkerThread<ErppEncoderPredictJob>() {}

  void init(FetchScheduleWorker* fetch_schedule_thread, ErppEncoderPredictor* erpp_encoder_predictor, ModuleMeta* metas){
    this->fetch_schedule_thread = fetch_schedule_thread;
    this->erpp_encoder_predictor = erpp_encoder_predictor;
    this->metas = metas;
  }

  void begin_reset_for_generate(){
    // advance generate_epoch to trigger stale job detection
    current_generate_epoch.fetch_add(1, std::memory_order_acq_rel);
    clear_pending_tasks();
  }
  void reset_for_generate(int64_t next_generate_epoch){
    wait_until_idle();
    clear_pending_tasks();
    current_generate_epoch.store(next_generate_epoch, std::memory_order_release);
  }

  void on_encoder_layer0_recorded(int64_t forward_epoch, int64_t generate_epoch);

 protected:
  void do_one_task_impl(ErppEncoderPredictJob job) override;
};
