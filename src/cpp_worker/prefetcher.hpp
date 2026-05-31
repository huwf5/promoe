#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <string>
#include <unordered_set>
#include <vector>
#include <queue>
#include <sstream>
#include <ATen/cuda/CUDAContext.h>

#include "worker.hpp"
#include "utils.hpp"
#include "model_loader.hpp"
#include "predictor.hpp"
#include "erpp_encoder_predictor.hpp"
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
    kForwardEpochStart,
    kReset,
    kErppEncoderJitRankings,
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
  int64_t forward_epoch = 0;
  int64_t generate_epoch = 0;
  int64_t* expert_idxs;
  size_t num_expert;
};
class PreemptOneExpertTask : public FetchScheduleTaskBase {
 public:
  PreemptOneExpertTask() : FetchScheduleTaskBase(kPreemptOneExpert) {}
  int layer_id;
  int64_t generate_epoch = 0;
  int64_t expert_id;
};
class PrefetchLayerTask : public FetchScheduleTaskBase {
 public:
  PrefetchLayerTask() : FetchScheduleTaskBase(kPrefetchLayer) {}
  int layer_idx;
  int64_t forward_epoch = 0;
  int64_t generate_epoch = 0;
  int64_t* expert_idxs;
  size_t num_expert;
  CacheRequestType request_type = kCacheRequestDecoderPredictorPrefetch;
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
class ForwardEpochStartTask : public FetchScheduleTaskBase {
 public:
  ForwardEpochStartTask() : FetchScheduleTaskBase(kForwardEpochStart) {}
  int64_t forward_epoch = 0;
  int64_t generate_epoch = 0;
  DecoderWarmupAction decoder_warmup_action = DecoderWarmupAction::kPreserve;
};
class ResetTask : public FetchScheduleTaskBase {
 public:
  ResetTask() : FetchScheduleTaskBase(kReset) {}
  int64_t next_generate_epoch = 0;
  int64_t next_forward_epoch = 0;
  DecoderWarmupAction action = DecoderWarmupAction::kClear;
};

class ErppEncoderJitRankingsTask : public FetchScheduleTaskBase {
 public:
  ErppEncoderJitRankingsTask() : FetchScheduleTaskBase(kErppEncoderJitRankings) {}
  int64_t forward_epoch = 0;
  int64_t generate_epoch = 0;
  std::vector<std::vector<int64_t>> rankings;
  std::vector<int> budgets;
};

class FetchScheduleWorker : public WorkerThread<FetchScheduleTaskBase*> {
  using TaskQueue = Queue<CopyTask>;
  enum class PrefetchClass {
    kEncoderPredictor = 0,
    kDecoderPredictor,
    kDecoderWarmup,
  };
  struct PrefetchQueueSet {
    std::vector<TaskQueue> encoder_predictor_by_layer;
    std::vector<TaskQueue> decoder_predictor_by_layer;
    TaskQueue decoder_warmup_plan_queue;
  };

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
  int64_t current_forward_epoch = 0;
  int64_t current_generate_epoch = 0;
  int current_layer = -1;

  IdleTask idle_task;
  CopyTask current_task; // we allow only one ongoing copy task
  friend class FetchWorker;
  friend class PrefetchMngr;
  FetchDoneTask copy_done_task;
  ForwardEpochStartTask forward_epoch_start_task;
  ResetTask reset_task;
  std::atomic<bool> reset_requested{false};

  #ifdef DEAD_CODE
  /** protected by queue_lock */
  AtomicQueueLock task_queue_lock;
  #endif
  PrefetchQueueSet prefetch_queues;
  /** no lock requried */
  TaskQueue precise_job_queue;
  enum SchedulerPhase {
    kEncoderPhase = 0,
    kDecoderPredictorPhase,
  };
  SchedulerPhase phase = kEncoderPhase;
  std::unordered_set<int64_t> decoder_warmup_seen;
  struct PendingReclaimableUpdate {
    enum Mode {
      kNone = 0,
      kSomeExperts,
      kLayerExcept,
      kLayerAll,
    };

    Mode mode = kNone;
    std::vector<uint8_t> expert_mask;
    std::vector<uint8_t> needed_mask;
  };
  AtomicQueueLock reclaimable_update_lock;
  std::vector<PendingReclaimableUpdate> pending_reclaimable_updates;
  std::vector<int> pending_reclaimable_layers;
  std::vector<uint8_t> pending_reclaimable_layer_mask;
  std::atomic<bool> has_pending_reclaimable_updates{false};
  std::vector<std::vector<int64_t>> encoder_jit_rankings;
  std::vector<int> encoder_jit_budgets;
  std::vector<std::vector<uint8_t>> encoder_jit_submitted_mask;
  std::vector<uint8_t> encoder_jit_enabled_layer_mask;
  int64_t encoder_jit_forward_epoch = -1;
  int64_t encoder_jit_generate_epoch = -1;


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
  void do_one_task_impl(ForwardEpochStartTask *task);
  void do_one_task_impl(ResetTask *task);
  void do_one_task_impl(ErppEncoderJitRankingsTask *task);

  void pop_next_task(CopyTask &task, bool &found);

  void add_single_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue* queue, int start_mem_buf_idx, int stop_mem_buf_idx, bool is_precise, int64_t forward_epoch, CacheRequestType request_type);
  void add_separate_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue *queue, int start_mem_buf_idx, int stop_mem_buf_idx, bool is_precise, int64_t forward_epoch, CacheRequestType request_type);

  void reorder_experts(int layer_idx, int64_t *expert_idxs, size_t num_expert);
  #ifdef DEAD_CODE
  void preempt_one_layer_(int layer_idx, int64_t *expert_idxs, size_t num_expert);
  #endif
  void preempt_one_layer_without_reorder_(int layer_idx, int64_t *expert_idxs, size_t num_expert);
  void preempt_one_expert(int layer_idx, int64_t expert_idx);
  bool is_stale_prefetch(int64_t forward_epoch, int layer_idx) const;
  void start_forward_epoch(int64_t forward_epoch, DecoderWarmupAction decoder_warmup_action);
  void advance_actual_layer(int64_t forward_epoch, int layer_idx);
  void clear_prefetch_queues_up_to_layer(int layer_idx);
  void clear_all_prefetch_queues();
  void clear_stale_prefetch_queues_before_epoch(int64_t min_forward_epoch);
  void clear_all_job_queues();
  void set_phase(SchedulerPhase next_phase);
  void clear_decoder_warmup_plan_queue();
  void rebuild_decoder_warmup_queue();
  bool parse_layer_expert_plan(const std::string& plan, std::vector<std::pair<int, int>>& out);
  PrefetchClass prefetch_class_for_request(CacheRequestType request_type) const;
  CacheRequestType request_type_for_prefetch_class(PrefetchClass cls) const;
  bool requires_encoder_phase(PrefetchClass cls) const;
  bool requires_reclaimable_encoder(PrefetchClass cls) const;
  bool blocks_lower_priority_when_pending(PrefetchClass cls) const;
  bool replace_same_layer_on_enqueue(PrefetchClass cls) const;
  bool pop_next_prefetch_for_class(PrefetchClass cls, CopyTask& task, bool* blocked_lower_priority = nullptr);
  bool has_pending_prefetch_for_class(PrefetchClass cls);
  void prune_prefetch_class(PrefetchClass cls);
  TaskQueue* queue_for_class_and_layer(PrefetchClass cls, int layer_idx);
  const TaskQueue* queue_for_class_and_layer(PrefetchClass cls, int layer_idx) const;
  void clear_prefetch_class(PrefetchClass cls);
  void clear_prefetch_class_up_to_layer(PrefetchClass cls, int layer_idx);
  void clear_prefetch_class_for_layer(PrefetchClass cls, int64_t forward_epoch, int layer_idx);
  int64_t flatten_expert(int layer_idx, int expert_idx) const;
  void ensure_reclaimable_pending_initialized();
  void reset_pending_reclaimable_updates();
  void note_pending_reclaimable_layer_locked(int layer_idx);
  bool is_idle();
  void store_erpp_encoder_jit_rankings(const ErppEncoderJitRankingsTask& task);
  void clear_encoder_jit_state() {
    encoder_jit_rankings.clear();
    encoder_jit_budgets.clear();
    encoder_jit_submitted_mask.clear();
    encoder_jit_forward_epoch = -1;
    encoder_jit_generate_epoch = -1;
  }
  void initialize_encoder_jit_enabled_layer_mask() {
    encoder_jit_enabled_layer_mask.assign(metas->num_encoder_moe_layer, 0);
    const std::string& spec = metas->erpp_encoder_jit_refill_layers;
    if (spec == "all") {
      std::fill(encoder_jit_enabled_layer_mask.begin(),
                encoder_jit_enabled_layer_mask.end(),
                1);
      return;
    }

    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (item.empty()) {
        continue;
      }
      int parsed = std::stoi(item);
      if (parsed < 0) {
        parsed += metas->num_encoder_moe_layer;
      }
      if (parsed >= 0 && parsed < metas->num_encoder_moe_layer) {
        encoder_jit_enabled_layer_mask[parsed] = 1;
      }
    }
  }
  void mark_encoder_jit_submitted(int layer_idx, int expert_idx) {
    CHECK(layer_idx >= 0 && layer_idx < static_cast<int>(encoder_jit_submitted_mask.size())) << "layer_idx is out of range";
    CHECK(expert_idx >= 0 && expert_idx < static_cast<int>(encoder_jit_submitted_mask[layer_idx].size())) << "expert_idx is out of range";
    encoder_jit_submitted_mask[layer_idx][expert_idx] = 1;
  }

  bool encoder_jit_is_submitted(int layer_idx, int expert_idx) const {
    return encoder_jit_submitted_mask[layer_idx][expert_idx];
  }

  // Per-layer minimum expert count (cache + in-flight refill) for encoder JIT refill.
  // When occupancy drops below low_watermark (floor * ratio), refill ranks experts up to floor.
  // Also passed to cache eviction as a per-layer retention hint (see encoder_jit_can_dispatch).
  int encoder_jit_floor() const {
    int floor_value = 1;
    if (metas->erpp_encoder_jit_refill_floor_mode == "fixed" &&
        metas->erpp_encoder_jit_refill_floor_value > 0) {
      // Explicit cap from --erpp_encoder_jit_refill_floor_value.
      floor_value = metas->erpp_encoder_jit_refill_floor_value;
    } else {
      // "avg": split total GPU expert slots evenly across encoder MoE layers.
      const int total_slots = cache != nullptr && cache->cache_len > 0
          ? static_cast<int>(cache->cache_len)
          : int(std::floor(metas->cache_rate * metas->num_layer * metas->num_expert));
      floor_value = metas->num_encoder_moe_layer > 0
          ? total_slots / metas->num_encoder_moe_layer
          : 1;
    }
    return std::max(1, std::min(metas->num_expert, floor_value));
  }

  // Per-layer minimum expert count (cache + in-flight refill) for encoder JIT refill.
  // lower threshold for refill
  int encoder_jit_low_watermark() const {
    const int floor_value = encoder_jit_floor();
    return std::max(1, int(std::floor(floor_value * metas->erpp_encoder_jit_refill_low_watermark_ratio))); // clamp to [1, num_expert]
  }

  bool encoder_jit_layer_enabled(int layer_idx) const{
    CHECK(layer_idx >= 0 && layer_idx < static_cast<int>(encoder_jit_enabled_layer_mask.size())) << "layer_idx is out of range";
    return encoder_jit_enabled_layer_mask[layer_idx];
  }
  bool encoder_jit_is_missing(int layer_idx, int expert_idx) const;
  bool encoder_jit_target_in_window(int layer_idx) const {
    return metas->is_encoder_layer(layer_idx) &&
           layer_idx > current_layer &&
           layer_idx <= current_layer + metas->erpp_encoder_jit_refill_window;
  }
  std::vector<int> build_encoder_jit_required_experts(
      int layer_idx, int occupancy, int floor_value, int low_watermark, int budget);
  bool encoder_jit_can_dispatch(const CopyTask& task) const;
  void maybe_enqueue_encoder_jit_refill();

 public:
  #ifdef DEAD_CODE
  void add_one_layer_task(int layer_idx, int64_t *expert_idxs, size_t num_expert);
  void add_one_layer_task(int layer_idx, torch::Tensor experts);
  #endif
  void enqueue_layer_reclaimable_except(int layer_idx, const std::vector<uint8_t>& needed_mask);
  void enqueue_expert_reclaimable(int layer_idx, int expert_idx);
  void enqueue_layer_reclaimable(int layer_idx);
  void drain_reclaimable_updates(int max_updates = -1);
  void init(ModuleMeta *metas, ModelLoader *model_loader, CacheMngr *cache, FetchWorker *fetch_thread, PredictWorker *predict_thread, CacheStatistics *cache_stats, TimeProfiler* profiler);
  void begin_reset_for_generate() {
    reset_requested.store(true, std::memory_order_release);
  }
  void reset_for_generate(int64_t next_generate_epoch, int64_t next_forward_epoch, DecoderWarmupAction action);

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
  std::shared_ptr<ErppEncoderPredictor> erpp_encoder_predictor;
  std::shared_ptr<ErppEncoderPredictWorker> erpp_encoder_predict_thread;
  struct EncoderLayerStats {
    int64_t forward_epoch = -1;
    int64_t generate_epoch = -1;
    int64_t needed = 0;
    int64_t entry_hit = 0;
    int64_t entry_miss = 0;
    int64_t actual_hit = 0;
    int64_t actual_miss = 0;
    int64_t waited = 0;
    int64_t wait_us_total = 0;
    int64_t wait_us_max = 0;
  };
  std::vector<EncoderLayerStats> encoder_layer_stats;
  void ensure_encoder_layer_stats_size();
  bool encoder_layer_expert_entry_hit(int layer_id, int expert_id) const;
  void log_encoder_layer_entry_stats(int layer_id, int64_t* experts, int64_t num_expert);
  void log_encoder_layer_use_stats(
      int layer_id,
      int expert_id,
      bool hit,
      bool waited,
      uint64_t wait_us,
      int status_before);
  void log_encoder_layer_done_stats(int layer_id);

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
  int64_t forward_epoch = 0;
  int64_t generate_epoch = 0;
  std::atomic<bool> reset_in_progress{false};
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
  void reset_for_generate();
  void reset_and_load_initial_cache() { reset_for_generate(); }

  void report_one_layer(int layer_id, torch::Tensor experts);
  void report_one_layer(int layer_id, int64_t* experts, int64_t num_expert);
  void one_moe_layer_done(int layer_id);

  void report_one_expert(int layer_id, int expert_id);
  void one_expert_done(int layer_id, int expert_id);

  void report_moe_attn_logits(int layer_id, torch::Tensor attn_logits);

  void report_moe_layer_logits(int layer_id, torch::Tensor layer_logits);
  void report_erpp_encoder_layer0(torch::Tensor hidden, torch::Tensor attention_mask);

  void launch_thread();
  TimerGuard build_timer() { return TimerGuard(this->profiler.get()); }
  void reload_env();

  void set_compute_stream(int64_t stream);

  void temp_move_expert_to_gpu(int layer_id, int expert_id);
  void temp_move_expert_back_to_host(int layer_id, int expert_id);
};
