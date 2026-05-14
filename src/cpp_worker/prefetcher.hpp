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
#include <ATen/cuda/CUDAContext.h>

#include "worker.hpp"
#include "utils.hpp"
#include "model_loader.hpp"
#include "predictor.hpp"
#include "cache.hpp"
#include "profiler.hpp"

class ExpertHandler;

class FetchScheduleTaskBase {
 public:
  enum TaskType {
    kIdle,
    kPreempt,
    kFetchDone,
    kPrefetchLayer,
    kPreemptOneExpert,
    kGenerationStart,
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
  int64_t generation = 0;
  int64_t* expert_idxs;
  size_t num_expert;
};
class PreemptOneExpertTask : public FetchScheduleTaskBase {
 public:
  PreemptOneExpertTask() : FetchScheduleTaskBase(kPreemptOneExpert) {}
  int layer_id;
  int64_t expert_id;
};
class PrefetchLayerTask : public FetchScheduleTaskBase {
 public:
  PrefetchLayerTask() : FetchScheduleTaskBase(kPrefetchLayer) {}
  int layer_idx;
  int64_t generation = 0;
  int64_t* expert_idxs;
  size_t num_expert;
};
class FetchDoneTask : public FetchScheduleTaskBase {
 public:
  FetchDoneTask() : FetchScheduleTaskBase(kFetchDone) {}
};
enum class DecoderWarmupAction {
  kPreserve = 0,
  kClear,
  kRebuildForGenerateStart,
};
class GenerationStartTask : public FetchScheduleTaskBase {
 public:
  GenerationStartTask() : FetchScheduleTaskBase(kGenerationStart) {}
  int64_t generation = 0;
  DecoderWarmupAction decoder_warmup_action = DecoderWarmupAction::kPreserve;
};

class FetchScheduleWorker : public WorkerThread<FetchScheduleTaskBase*> {
  using TaskQueue = Queue<CopyTask>;

  ModuleMeta*      metas;
  ModelLoader*     model_loader;
  CacheMngr*       cache;

  void cache_hit(ExpertHandler* e, bool is_precise) {
    profiler->add(is_precise ? TimeProfiler::kHitCnt : TimeProfiler::kPrefetchHitCnt, 1);
    cache->hit(e, is_precise);
  }
  CacheMngr::CacheLineOccupancyWaiter cache_miss(
      ExpertHandler* e,
      bool is_precise,
      CacheRequestType request_type) {
    profiler->add(is_precise ? TimeProfiler::kMissCnt : TimeProfiler::kPrefetchMissCnt, 1);
    return cache->miss(e, is_precise, request_type);
  }

  CacheStatistics* cache_stats;
  TimeProfiler*    profiler;

  FetchWorker*         fetch_thread;
  PredictWorker*       predict_thread;
  int64_t current_generation = 0;
  int current_layer = -1;

  IdleTask idle_task;
  CopyTask current_task; // we allow only one ongoing copy task
  friend class FetchWorker;
  friend class PrefetchMngr;
  FetchDoneTask copy_done_task;
  GenerationStartTask generation_start_task;

  #ifdef DEAD_CODE
  /** protected by queue_lock */
  AtomicQueueLock task_queue_lock;
  #endif
  std::vector<TaskQueue> per_layer_job_queues; // the fetching thread takes out the first task from queue, then execute it.
  /** no lock requried */
  TaskQueue precise_job_queue;
  enum SchedulerPhase {
    kEncoderPhase = 0,
    kDecoderPredictorPhase,
  };
  SchedulerPhase phase = kEncoderPhase;
  std::queue<std::pair<int, int>> decoder_warmup_queue;
  std::unordered_set<int64_t> decoder_warmup_seen;

  #ifdef DEAD_CODE
  inline void lock_task_queue() { task_queue_lock.lock(); }
  inline void unlock_task_queue() { task_queue_lock.unlock(); }
  #endif

  bool send_one_job(CopyTask *task);
  void do_one_task_impl(IdleTask *task);
  void do_one_task_impl(PreemptTask *task);
  void do_one_task_impl(PreemptOneExpertTask *task);
  void do_one_task_impl(FetchDoneTask *task);
  void do_one_task_impl(PrefetchLayerTask *task);
  void do_one_task_impl(GenerationStartTask *task);

  void pop_next_task(CopyTask &task, bool &found);

  void add_single_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue* queue, int start_mem_buf_idx, int stop_mem_buf_idx, bool is_precise, int64_t generation, CacheRequestType request_type);
  void add_separate_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue *queue, int start_mem_buf_idx, int stop_mem_buf_idx, bool is_precise, int64_t generation, CacheRequestType request_type);

  void reorder_experts(int layer_idx, int64_t *expert_idxs, size_t num_expert);
  #ifdef DEAD_CODE
  void preempt_one_layer_(int layer_idx, int64_t *expert_idxs, size_t num_expert);
  #endif
  void preempt_one_layer_without_reorder_(int layer_idx, int64_t *expert_idxs, size_t num_expert);
  void preempt_one_expert(int layer_idx, int64_t expert_idx);
  bool is_stale_prefetch(int64_t generation, int layer_idx) const;
  void start_generation(int64_t generation, DecoderWarmupAction decoder_warmup_action);
  void advance_actual_layer(int64_t generation, int layer_idx);
  void clear_prefetch_queues_up_to_layer(int layer_idx);
  void clear_all_prefetch_queues();
  void clear_all_job_queues();
  void set_phase(SchedulerPhase next_phase);
  void clear_decoder_warmup_queue();
  void rebuild_decoder_warmup_queue();
  bool parse_layer_expert_plan(const std::string& plan, std::vector<std::pair<int, int>>& out);
  bool pop_next_normal_prefetch(CopyTask& task);
  bool pop_next_decoder_warmup(CopyTask& task);
  int64_t flatten_expert(int layer_idx, int expert_idx) const;
  bool is_idle();

 public:
  #ifdef DEAD_CODE
  void add_one_layer_task(int layer_idx, int64_t *expert_idxs, size_t num_expert);
  void add_one_layer_task(int layer_idx, torch::Tensor experts);
  #endif
  void init(ModuleMeta *metas, ModelLoader *model_loader, CacheMngr *cache, FetchWorker *fetch_thread, PredictWorker *predict_thread, CacheStatistics *cache_stats, TimeProfiler* profiler);

protected:
  void do_one_task_impl(FetchScheduleTaskBase *task);
};

class PrefetchMngr : public std::enable_shared_from_this<PrefetchMngr> {
  friend class FetchScheduleWorker;
  friend class FetchWorker;


  std::shared_ptr<FetchScheduleWorker> fetch_schedule_thread;
  std::shared_ptr<FetchWorker>         fetch_thread;
  std::shared_ptr<PredictWorker>       predict_thread;
  std::shared_ptr<ExpertUnlockWorker>  expert_unlocker_thread;

  /**
   * for already in cache, directly lock it
   * for fetching, lock it after fetching is done
   */
  void preempt_and_launch_one_layer(int layer_idx, int64_t* experts, int64_t num_expert);
  void record_then_predict_and_prefetch(int layer_id, int64_t* experts, int64_t num_expert);

  void wait_expert(int layer_id, int expert_id);
  void mark_expert_using(int layer_id, int expert_id);

public:
  std::shared_ptr<ModuleMeta>      metas;
  std::shared_ptr<ModelLoader>     model_loader;
  std::shared_ptr<PredictorBase>   predictor;
  std::shared_ptr<CacheStatistics> cache_stats;
  std::shared_ptr<TimeProfiler>    profiler;
  std::shared_ptr<CacheMngr>       cache;

  std::shared_ptr<PrecisionProfiler> precision_profiler;

  int64_t compute_stream = 0, copy_stream = 0;
  int64_t prefetch_generation = 0;
  // cudaStream_t compute_stream = nullptr, copy_stream = nullptr;
  // at::cuda::CUDAStream compute_stream, copy_stream;

  PrefetchMngr(std::shared_ptr<ModuleMeta> metas,
               std::shared_ptr<ModelLoader> model_loader,
               std::shared_ptr<PredictorBase> predictor,
               int64_t compute_stream = 0,
               bool create_compute_stream = true,
               TimeProfiler* profiler = nullptr);
  ~PrefetchMngr();
  void init_gpu_mem_buffer();
  void reset_and_load_initial_cache();

  void report_one_layer(int layer_id, torch::Tensor experts);
  void report_one_layer(int layer_id, int64_t* experts, int64_t num_expert);
  void one_moe_layer_done(int layer_id);

  void report_one_expert(int layer_id, int expert_id);
  void one_expert_done(int layer_id, int expert_id);

  void report_moe_attn_logits(int layer_id, torch::Tensor attn_logits);

  void report_moe_layer_logits(int layer_id, torch::Tensor layer_logits);

  void launch_thread();
  TimerGuard build_timer() { return TimerGuard(this->profiler.get()); }
  void reload_env();

  void set_compute_stream(int64_t stream);

  void temp_move_expert_to_gpu(int layer_id, int expert_id);
  void temp_move_expert_back_to_host(int layer_id, int expert_id);
};
