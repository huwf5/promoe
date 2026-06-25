#include <omp.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <tuple>
#include <cuda_runtime.h>
#include "prefetcher.hpp"
#include "profiler.hpp"
#include "logging.hpp"
#include "nvtx_utils.hpp"

namespace {
class AtomicQueueLockGuard {
  AtomicQueueLock& lock_;

 public:
  explicit AtomicQueueLockGuard(AtomicQueueLock& lock) : lock_(lock) {
    lock_.lock();
  }
  ~AtomicQueueLockGuard() {
    lock_.unlock();
  }
  AtomicQueueLockGuard(const AtomicQueueLockGuard&) = delete;
  AtomicQueueLockGuard& operator=(const AtomicQueueLockGuard&) = delete;
};

// Set SPARSE_CACHE_LOG_ENCODER_EXPERTS_ON_DECODER_ENTRY=1 to print encoder-side
// cache occupancy when the scheduler first enters the decoder predictor phase.
bool log_prefetch_decision_enabled() {
  const char* flag = std::getenv("SPARSE_CACHE_LOG_PREFETCH_DECISION");
  return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}

bool log_encoder_layer_stats_enabled() {
  const char* flag = std::getenv("SPARSE_CACHE_LOG_ENCODER_LAYER_STATS");
  return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}

bool log_erpp_encoder_diagnostics_enabled() {
  const char* flag = std::getenv("SPARSE_CACHE_LOG_ERPP_ENCODER_DIAGNOSTICS");
  return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}

void log_encoder_experts_in_cache_on_decoder_entry(
    ModuleMeta* metas,
    ModelLoader* model_loader,
    CacheMngr* cache) {
  if (metas == nullptr || model_loader == nullptr || cache == nullptr) {
    return;
  }
  const char* flag = std::getenv("SPARSE_CACHE_LOG_ENCODER_EXPERTS_ON_DECODER_ENTRY");
  if (flag == nullptr || flag[0] == '\0' || flag[0] == '0') {
    return;
  }
  const int enc_layers = metas->num_encoder_moe_layer;
  if (enc_layers <= 0) {
    LOG(INFO) << "decoder phase entry: num_encoder_moe_layer=0, skip encoder cache scan";
    return;
  }
  int in_cache = 0;
  int fully_ready = 0;
  for (int layer = 0; layer < enc_layers; layer++) {
    for (int expert = 0; expert < metas->num_expert; expert++) {
      ExpertHandler* e = model_loader->get_source(layer, expert);
      if (!cache->is_in_cache(e)) {
        continue;
      }
      in_cache++;
      if (e->num_ready >= metas->num_per_expert_param) {
        fully_ready++;
      }
    }
  }
  const int total = enc_layers * metas->num_expert;
  LOG(INFO) << "decoder phase entry: encoder experts in cache (gpu slot) " << in_cache << "/" << total
            << ", fully_ready(num_ready>=" << metas->num_per_expert_param << ") " << fully_ready;
}
} // namespace

void FetchScheduleWorker::reorder_experts(int layer_idx, int64_t *expert_idxs, size_t num_expert) {
  TRACE_EVENT_GURAD(kFetchScheduler, "reorder_experts");
  LOG_BLOCK(DEBUG, logger, {
    logger << "reordering experts for layer " << layer_idx << " with expert " << array_to_str(expert_idxs, num_expert);
  });
  std::vector<ExpertHandler*> done_experts;    // correctly predicted and already done
  std::vector<ExpertHandler*> going_experts;   // correctly predicted and is current task
  std::vector<ExpertHandler*> partial_experts; // correctly predicted and partially fetched, but is not current task
  std::vector<ExpertHandler*> miss_experts;    // correctly predicted, but not in cache

  for (int i = 0; i < num_expert; i++) {
    auto e = model_loader->get_source(layer_idx, expert_idxs[i]);
    if (e == current_task.expert) {
      going_experts.push_back(e);
    } else {
      if (cache->is_in_cache(e)) {
        if (e->num_ready == metas->num_per_expert_param) {
          done_experts.push_back(e);
        } else {
          partial_experts.push_back(e);
        }
      } else {
        miss_experts.push_back(e);
      }
    }
  }

  num_expert = 0;
  // reorder expert order to let model use expert in the same order of fetching
  for (auto e : done_experts)    { expert_idxs[num_expert++] = e->expert_idx; }
  for (auto e : going_experts)   { expert_idxs[num_expert++] = e->expert_idx; }
  for (auto e : partial_experts) { expert_idxs[num_expert++] = e->expert_idx; }
  for (auto e : miss_experts)    { expert_idxs[num_expert++] = e->expert_idx; }
  LOG_BLOCK(DEBUG, logger, {
    logger << "reordered expert to " << array_to_str(expert_idxs, num_expert);
  });
}

bool FetchScheduleWorker::is_stale_prefetch(int64_t forward_epoch, int layer_idx) const {
  if (forward_epoch < current_forward_epoch) {
    return true;
  }
  if (forward_epoch == current_forward_epoch && layer_idx <= current_layer) {
    return true;
  }
  return false;
}

void FetchScheduleWorker::clear_all_prefetch_queues() {
  clear_encoder_jit_state();
  for (auto& queue : prefetch_queues.encoder_predictor_by_layer) {
    queue.clear();
  }
  for (auto& queue : prefetch_queues.decoder_predictor_by_layer) {
    queue.clear();
  }
  prefetch_queues.decoder_warmup_plan_queue.clear();
}

void FetchScheduleWorker::clear_stale_prefetch_queues_before_epoch(
    int64_t min_forward_epoch) {
  auto keep_current_epoch = [min_forward_epoch](TaskQueue& queue) {
    TaskQueue kept;
    while (!queue.empty()) {
      CopyTask task = queue.front();
      queue.pop();
      if (task.forward_epoch >= min_forward_epoch) {
        kept.push(task);
      }
    }
    while (!kept.empty()) {
      CopyTask task = kept.front();
      kept.pop();
      queue.push(task);
    }
  };

  for (auto& queue : prefetch_queues.encoder_predictor_by_layer) {
    keep_current_epoch(queue);
  }
  for (auto& queue : prefetch_queues.decoder_predictor_by_layer) {
    keep_current_epoch(queue);
  }
}

void FetchScheduleWorker::clear_all_job_queues() {
  clear_all_prefetch_queues();
  precise_job_queue.clear();
}

bool FetchScheduleWorker::is_idle() {
  // Prefetch work is opportunistic; reset clears stale pending entries.
  if (current_task.expert != nullptr) {
    return false;
  }
  if (!precise_job_queue.empty()) {
    return false;
  }
  return true;
}

int64_t FetchScheduleWorker::flatten_expert(int layer_idx, int expert_idx) const {
  return int64_t(layer_idx) * int64_t(metas->num_expert) + int64_t(expert_idx);
}

void FetchScheduleWorker::ensure_reclaimable_pending_initialized() {
  if (pending_reclaimable_updates.size() != static_cast<size_t>(metas->num_layer)) {
    pending_reclaimable_updates.resize(metas->num_layer);
  }
  if (pending_reclaimable_layer_mask.size() != static_cast<size_t>(metas->num_layer)) {
    pending_reclaimable_layer_mask.assign(metas->num_layer, 0);
  }
}

void FetchScheduleWorker::reset_pending_reclaimable_updates() {
  AtomicQueueLockGuard guard(reclaimable_update_lock);
  pending_reclaimable_updates.assign(metas->num_layer, PendingReclaimableUpdate());
  pending_reclaimable_layers.clear();
  pending_reclaimable_layer_mask.assign(metas->num_layer, 0);
  has_pending_reclaimable_updates.store(false, std::memory_order_release);
}

void FetchScheduleWorker::note_pending_reclaimable_layer_locked(int layer_idx) {
  if (!pending_reclaimable_layer_mask[layer_idx]) {
    pending_reclaimable_layer_mask[layer_idx] = 1;
    pending_reclaimable_layers.push_back(layer_idx);
  }
}

void FetchScheduleWorker::enqueue_layer_reclaimable_except(
    int layer_idx,
    const std::vector<uint8_t>& needed_mask) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  AtomicQueueLockGuard guard(reclaimable_update_lock);
  ensure_reclaimable_pending_initialized();
  auto& pending = pending_reclaimable_updates[layer_idx];
  if (pending.mode == PendingReclaimableUpdate::kLayerAll) {
    return;
  }

  std::vector<uint8_t> merged = needed_mask;
  if (merged.size() != static_cast<size_t>(metas->num_expert)) {
    merged.resize(metas->num_expert, 0);
  }
  if (pending.mode == PendingReclaimableUpdate::kSomeExperts) {
    if (pending.expert_mask.size() != static_cast<size_t>(metas->num_expert)) {
      pending.expert_mask.resize(metas->num_expert, 0);
    }
    for (int expert_idx = 0; expert_idx < metas->num_expert; expert_idx++) {
      if (pending.expert_mask[expert_idx]) {
        merged[expert_idx] = 0;
      }
    }
  } else if (pending.mode == PendingReclaimableUpdate::kLayerExcept) {
    if (pending.needed_mask.size() != static_cast<size_t>(metas->num_expert)) {
      pending.needed_mask.resize(metas->num_expert, 0);
    }
    for (int expert_idx = 0; expert_idx < metas->num_expert; expert_idx++) {
      merged[expert_idx] = merged[expert_idx] && pending.needed_mask[expert_idx];
    }
  }

  pending.mode = PendingReclaimableUpdate::kLayerExcept;
  pending.needed_mask = std::move(merged);
  pending.expert_mask.clear();
  note_pending_reclaimable_layer_locked(layer_idx);
  has_pending_reclaimable_updates.store(true, std::memory_order_release);
}

void FetchScheduleWorker::enqueue_expert_reclaimable(int layer_idx, int expert_idx) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  CHECK(expert_idx >= 0 && expert_idx < metas->num_expert)
      << "expert index out of range for reclaimable update: " << expert_idx;
  AtomicQueueLockGuard guard(reclaimable_update_lock);
  ensure_reclaimable_pending_initialized();
  auto& pending = pending_reclaimable_updates[layer_idx];
  if (pending.mode == PendingReclaimableUpdate::kLayerAll) {
    return;
  }
  if (pending.mode == PendingReclaimableUpdate::kLayerExcept) {
    if (expert_idx >= 0 && expert_idx < static_cast<int>(pending.needed_mask.size())) {
      pending.needed_mask[expert_idx] = 0;
    }
  } else {
    if (pending.expert_mask.empty()) {
      pending.expert_mask.assign(metas->num_expert, 0);
    }
    pending.mode = PendingReclaimableUpdate::kSomeExperts;
    pending.expert_mask[expert_idx] = 1;
  }
  note_pending_reclaimable_layer_locked(layer_idx);
  has_pending_reclaimable_updates.store(true, std::memory_order_release);
}

void FetchScheduleWorker::enqueue_layer_reclaimable(int layer_idx) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  AtomicQueueLockGuard guard(reclaimable_update_lock);
  ensure_reclaimable_pending_initialized();
  auto& pending = pending_reclaimable_updates[layer_idx];
  pending.mode = PendingReclaimableUpdate::kLayerAll;
  pending.expert_mask.clear();
  pending.needed_mask.clear();
  note_pending_reclaimable_layer_locked(layer_idx);
  has_pending_reclaimable_updates.store(true, std::memory_order_release);
}

void FetchScheduleWorker::drain_reclaimable_updates(int max_updates) {
  if (!has_pending_reclaimable_updates.load(std::memory_order_acquire)) {
    return;
  }

  std::vector<PendingReclaimableUpdate> local_updates;
  std::vector<int> local_layers;

  {
    AtomicQueueLockGuard guard(reclaimable_update_lock);
    ensure_reclaimable_pending_initialized();

    int drained = 0;
    std::vector<int> remaining_layers;
    for (int layer_idx : pending_reclaimable_layers) {
      if (max_updates >= 0 && drained >= max_updates) {
        remaining_layers.push_back(layer_idx);
        continue;
      }
      local_layers.push_back(layer_idx);
      local_updates.push_back(std::move(pending_reclaimable_updates[layer_idx]));
      pending_reclaimable_updates[layer_idx] = PendingReclaimableUpdate();
      pending_reclaimable_layer_mask[layer_idx] = 0;
      drained += 1;
    }

    pending_reclaimable_layers.swap(remaining_layers);
    has_pending_reclaimable_updates.store(!pending_reclaimable_layers.empty(),
                                          std::memory_order_release);
  }

  for (size_t i = 0; i < local_layers.size(); i++) {
    const int layer_idx = local_layers[i];
    const auto& update = local_updates[i];
    switch (update.mode) {
      case PendingReclaimableUpdate::kSomeExperts: {
        for (int expert_idx = 0; expert_idx < metas->num_expert; expert_idx++) {
          if (expert_idx < static_cast<int>(update.expert_mask.size()) &&
              update.expert_mask[expert_idx]) {
            cache->mark_reclaimable(layer_idx, expert_idx);
          }
        }
        break;
      }
      case PendingReclaimableUpdate::kLayerExcept: {
        cache->mark_layer_reclaimable_except(layer_idx, update.needed_mask);
        break;
      }
      case PendingReclaimableUpdate::kLayerAll: {
        cache->mark_layer_reclaimable(layer_idx);
        break;
      }
      case PendingReclaimableUpdate::kNone: {
        break;
      }
    }
  }
}

bool FetchScheduleWorker::encoder_jit_is_missing(int layer_idx, int expert_idx) const {
  if (encoder_jit_is_submitted(layer_idx, expert_idx)) {
    return false;
  }
  ExpertHandler* expert = model_loader->get_source(layer_idx, expert_idx);
  if (cache->is_in_cache(expert) || current_task.expert == expert) {
    return false;
  }
  return true;
}

bool FetchScheduleWorker::encoder_jit_can_dispatch(const CopyTask& task) const {
  if (task.request_type != kCacheRequestEncoderJitRefill || task.expert == nullptr) {
    return true;
  }
  if (task.expert->gpu_data != nullptr || cache->has_unused_slot_for(task.expert)) {
    return true;
  }
  return cache->has_reclaimable_encoder();
}

std::vector<int> FetchScheduleWorker::build_encoder_jit_required_experts(
    int layer_idx, int occupancy, int floor_value, int low_watermark, int budget) {
  std::vector<int> required;
  if (layer_idx < 0 || layer_idx >= static_cast<int>(encoder_jit_rankings.size())) {
    return required;
  }
  const auto& ranking = encoder_jit_rankings[layer_idx];
  std::vector<uint8_t> selected(metas->num_expert, 0);
  required.reserve(std::min(
      metas->num_expert,
      std::max(0, floor_value - occupancy) + std::max(0, budget)));
  const bool log_enabled = log_prefetch_decision_enabled();

  auto add_missing = [&](int expert_idx, const char* reason) {
    if (selected[expert_idx]) {
      return false;
    }
    if (!encoder_jit_is_missing(layer_idx, expert_idx)) {
      if (log_enabled) {
        LOG(INFO) << "erpp_encoder_jit_refill: skip target=L" << layer_idx
                  << " expert=" << expert_idx
                  << " reason=already_cache_or_pending";
      }
      return false;
    }
    selected[expert_idx] = 1;
    required.push_back(expert_idx);
    if (log_enabled) {
      LOG(INFO) << "erpp_encoder_jit_refill: select target=L" << layer_idx
                << " expert=" << expert_idx
                << " " << reason;
    }
    return true;
  };

  if (occupancy < low_watermark) {
    int floor_deficit = std::max(0, floor_value - occupancy);
    for (int expert_idx : ranking) {
      if (floor_deficit <= 0) {
        break;
      }
      if (add_missing(expert_idx, "reason=floor_deficit")) {
        floor_deficit -= 1;
      }
    }
  }

  if (metas->enable_erpp_encoder_jit_topk_cover && budget > 0) {
    const int limit = std::min<int>(budget, ranking.size());
    for (int rank = 0; rank < limit; rank++) {
      add_missing(ranking[rank], "reason=topk_cover");
    }
  }

  return required;
}

void FetchScheduleWorker::ensure_encoder_jit_wait_penalty() {
  if (encoder_jit_wait_penalty_ema_us.size() !=
      static_cast<size_t>(metas->num_encoder_moe_layer)) {
    encoder_jit_wait_penalty_ema_us.assign(metas->num_encoder_moe_layer, 1000.0);
  }
}

void FetchScheduleWorker::update_encoder_jit_wait_penalty(
    int layer_idx, int64_t actual_miss, int64_t wait_us_total) {
  if (layer_idx < 0 || layer_idx >= metas->num_encoder_moe_layer) {
    return;
  }
  ensure_encoder_jit_wait_penalty();
  double& ema = encoder_jit_wait_penalty_ema_us[layer_idx];
  if (actual_miss > 0 && wait_us_total > 0) {
    const double observed = static_cast<double>(wait_us_total) /
        static_cast<double>(actual_miss);
    ema = ema <= 0.0 ? observed : ema * 0.80 + observed * 0.20;
  } else {
    ema = std::max(100.0, ema * 0.95);
  }
}

FetchScheduleWorker::EncoderJitRefillCandidate
FetchScheduleWorker::build_encoder_jit_refill_candidate(int layer_idx) {
  EncoderJitRefillCandidate candidate;
  candidate.layer_idx = layer_idx;
  if (layer_idx < 0 || layer_idx >= metas->num_encoder_moe_layer ||
      layer_idx >= static_cast<int>(encoder_jit_rankings.size())) {
    return candidate;
  }
  candidate.occupancy = cache->encoder_layer_cache_occupancy(layer_idx);
  candidate.budget = layer_idx < static_cast<int>(encoder_jit_budgets.size())
      ? encoder_jit_budgets[layer_idx]
      : 0;
  candidate.floor_value = encoder_jit_auto_floor(layer_idx);
  if (metas->erpp_encoder_jit_refill_floor_mode == "budget" &&
      candidate.budget > candidate.floor_value) {
    // In auto mode, the predicted top-k cover width must obey the same
    // capacity-aware cap as the floor. Otherwise a 4GB run keeps chasing
    // ~55 predicted experts per layer even when the layer can only retain ~32.
    candidate.budget = candidate.floor_value;
  }
  candidate.low_watermark = encoder_jit_low_watermark_for_floor(candidate.floor_value);
  candidate.occupancy_gap = std::max(0, candidate.floor_value - candidate.occupancy);
  candidate.distance = layer_idx - current_layer;

  const auto& ranking = encoder_jit_rankings[layer_idx];
  candidate.predicted_need = candidate.budget > 0
      ? std::min<int>(candidate.budget, ranking.size())
      : 0;
  if (candidate.predicted_need > 0) {
    for (int rank = 0; rank < candidate.predicted_need; rank++) {
      if (encoder_jit_is_missing(layer_idx, ranking[rank])) {
        candidate.predicted_missing += 1;
      }
    }
    candidate.scarcity = static_cast<double>(candidate.predicted_missing) /
        static_cast<double>(candidate.predicted_need);
  }
  candidate.required = build_encoder_jit_required_experts(
      layer_idx, candidate.occupancy, candidate.floor_value,
      candidate.low_watermark, candidate.budget);
  candidate.demand_weight = 1.0;
  if (candidate.predicted_need > 0 && candidate.predicted_missing > 0) {
    const double missing = static_cast<double>(candidate.predicted_missing);
    const double pressure = 0.0;  // Phase 1A pure scarcity; no cache-rate branch.
    const double scarcity_power = 1.0 + pressure;
    candidate.demand_weight = 1.0 + missing *
        std::pow(candidate.scarcity, scarcity_power);
  }
  candidate.score = score_encoder_jit_refill_candidate(candidate);
  return candidate;
}

double FetchScheduleWorker::score_encoder_jit_refill_candidate(
    const EncoderJitRefillCandidate& candidate) const {
  if (candidate.layer_idx < 0 || candidate.required.empty() ||
      candidate.distance <= 0) {
    return 0.0;
  }

  if (candidate.occupancy >= candidate.low_watermark &&
      candidate.occupancy_gap == 0) {
    // Above-watermark top-k cover can still help, but Phase 1A makes it a
    // secondary choice instead of rewarding raw empty capacity.
    return 0.25;
  }

  return candidate.demand_weight;
}


void FetchScheduleWorker::log_encoder_jit_layer_entry_diagnostics(
    int layer_idx, const int64_t* expert_idxs, size_t num_expert) {
  if (metas == nullptr || cache == nullptr || model_loader == nullptr) {
    return;
  }
  if (!metas->enable_erpp_encoder_jit_refill || !metas->is_encoder_layer(layer_idx)) {
    return;
  }
  if (!log_encoder_layer_stats_enabled() &&
      !log_prefetch_decision_enabled() &&
      !log_erpp_encoder_diagnostics_enabled()) {
    return;
  }

  const bool ranking_available =
      !encoder_jit_rankings.empty() &&
      encoder_jit_generate_epoch == current_generate_epoch &&
      encoder_jit_forward_epoch == current_forward_epoch &&
      layer_idx >= 0 &&
      layer_idx < static_cast<int>(encoder_jit_rankings.size());
  const int budget = layer_idx >= 0 && layer_idx < static_cast<int>(encoder_jit_budgets.size())
      ? encoder_jit_budgets[layer_idx]
      : 0;
  const int occupancy = cache->encoder_layer_cache_occupancy(layer_idx);
  const int floor_value = encoder_jit_floor(layer_idx);
  const int low_watermark = encoder_jit_low_watermark(layer_idx);

  int predicted_cover = 0;
  int ready_cover = 0;
  int submitted_not_ready = 0;
  int predicted_not_ready = 0;
  int outside_ranking = 0;

  struct DemandDiagnostic {
    int expert_idx = -1;
    int rank = -1;
    bool within_budget = false;
    bool submitted = false;
    bool in_cache = false;
    bool ready = false;
    bool is_current_copy = false;
    int num_ready = 0;
    int status = 0;
  };
  std::vector<DemandDiagnostic> diagnostics;
  diagnostics.reserve(num_expert);

  for (size_t i = 0; i < num_expert; i++) {
    DemandDiagnostic diag;
    diag.expert_idx = static_cast<int>(expert_idxs[i]);
    if (diag.expert_idx < 0 || diag.expert_idx >= metas->num_expert) {
      outside_ranking += 1;
      diagnostics.push_back(diag);
      continue;
    }

    if (ranking_available) {
      const auto& ranking = encoder_jit_rankings[layer_idx];
      auto it = std::find(ranking.begin(), ranking.end(), diag.expert_idx);
      if (it != ranking.end()) {
        diag.rank = static_cast<int>(std::distance(ranking.begin(), it));
      }
    }
    diag.within_budget = diag.rank >= 0 && diag.rank < budget;
    if (diag.within_budget) {
      predicted_cover += 1;
    } else if (diag.rank < 0) {
      outside_ranking += 1;
    }

    if (layer_idx >= 0 && layer_idx < static_cast<int>(encoder_jit_submitted_mask.size()) &&
        diag.expert_idx < static_cast<int>(encoder_jit_submitted_mask[layer_idx].size())) {
      diag.submitted = encoder_jit_submitted_mask[layer_idx][diag.expert_idx] != 0;
    }

    ExpertHandler* expert = model_loader->get_source(layer_idx, diag.expert_idx);
    diag.in_cache = cache->is_in_cache(expert);
    diag.is_current_copy = current_task.expert == expert;
    diag.num_ready = expert->num_ready;
    diag.status = int(expert->expert_status.get());
    diag.ready = diag.status == int(kReady) || diag.status == int(kLaunching);

    if (diag.within_budget && diag.ready) {
      ready_cover += 1;
    }
    if (diag.submitted && !diag.ready) {
      submitted_not_ready += 1;
    }
    if (diag.rank >= 0 && !diag.ready) {
      predicted_not_ready += 1;
    }
    diagnostics.push_back(diag);
  }

  LOG(INFO) << "erpp_encoder_jit_refill: demand_summary"
            << " current=L" << current_layer
            << " target=L" << layer_idx
            << " needed=" << num_expert
            << " predicted_cover=" << predicted_cover
            << " ready_cover=" << ready_cover
            << " submitted_not_ready=" << submitted_not_ready
            << " predicted_not_ready=" << predicted_not_ready
            << " outside_ranking=" << outside_ranking
            << " ranking_available=" << ranking_available
            << " budget=" << budget
            << " occupancy=" << occupancy
            << " floor=" << floor_value
            << " low=" << low_watermark
            << " forward_epoch=" << current_forward_epoch
            << " generate_epoch=" << current_generate_epoch
            << " ranking_forward_epoch=" << encoder_jit_forward_epoch
            << " ranking_generate_epoch=" << encoder_jit_generate_epoch;

  if (log_erpp_encoder_diagnostics_enabled()) {
    LOG(INFO) << "erpp_encoder_diagnostics: demand_coverage"
              << " current_layer=" << current_layer
              << " target_layer=" << layer_idx
              << " needed=" << num_expert
              << " predicted_cover=" << predicted_cover
              << " ready_cover=" << ready_cover
              << " submitted_not_ready=" << submitted_not_ready
              << " predicted_not_ready=" << predicted_not_ready
              << " outside_ranking=" << outside_ranking
              << " ranking_available=" << ranking_available
              << " budget=" << budget
              << " occupancy=" << occupancy
              << " floor=" << floor_value
              << " low=" << low_watermark
              << " forward_epoch=" << current_forward_epoch
              << " generate_epoch=" << current_generate_epoch;
  }

  for (const auto& diag : diagnostics) {
    LOG(INFO) << "erpp_encoder_jit_refill: demand_detail"
              << " current=L" << current_layer
              << " target=L" << layer_idx
              << " expert=" << diag.expert_idx
              << " rank=" << diag.rank
              << " within_budget=" << diag.within_budget
              << " submitted=" << diag.submitted
              << " in_cache=" << diag.in_cache
              << " ready=" << diag.ready
              << " is_current_copy=" << diag.is_current_copy
              << " num_ready=" << diag.num_ready
              << " status=" << diag.status
              << " forward_epoch=" << current_forward_epoch
              << " generate_epoch=" << current_generate_epoch;
  }
}

void FetchScheduleWorker::maybe_enqueue_encoder_jit_refill() {
  if (!metas->enable_erpp_encoder_jit_refill || phase != kEncoderPhase) {
    return;
  }
  const bool log_enabled = log_prefetch_decision_enabled();
  const bool diagnostics_enabled = log_erpp_encoder_diagnostics_enabled();
  const bool auto_refill = encoder_jit_auto_refill_enabled();
  const int effective_per_idle_limit = metas->erpp_encoder_jit_refill_per_idle;
  ensure_encoder_jit_wait_penalty();
  int inspected_layers = 0;
  int required_experts = 0;
  int enqueued_experts = 0;
  int skipped_disabled_layers = 0;
  int per_idle_limited = 0;
  auto log_jit_enqueue_summary = [&](const char* stop_reason) {
    if (!diagnostics_enabled) {
      return;
    }
    LOG(INFO) << "erpp_encoder_diagnostics: jit_enqueue_summary"
              << " reason=" << stop_reason
              << " mode=" << (auto_refill ? "auto" : "ordered")
              << " current_layer=" << current_layer
              << " inspected_layers=" << inspected_layers
              << " required_experts=" << required_experts
              << " enqueued_experts=" << enqueued_experts
              << " skipped_disabled_layers=" << skipped_disabled_layers
              << " per_idle_limited=" << per_idle_limited
              << " effective_per_idle_limit=" << effective_per_idle_limit
              << " pending_encoder_prefetch="
              << has_pending_prefetch_for_class(PrefetchClass::kEncoderPredictor)
              << " has_reclaimable_encoder=" << cache->has_reclaimable_encoder()
              << " forward_epoch=" << current_forward_epoch
              << " generate_epoch=" << current_generate_epoch;
  };
  if (encoder_jit_rankings.empty()) {
    if (log_enabled) {
      LOG(INFO) << "erpp_encoder_jit_refill: skip reason=no_ranking"
                << " current=L" << current_layer
                << " forward_epoch=" << current_forward_epoch;
    }
    log_jit_enqueue_summary("no_ranking");
    return;
  }
  if (encoder_jit_generate_epoch != current_generate_epoch ||
      encoder_jit_forward_epoch != current_forward_epoch) {
        CHECK(false) << "erpp_encoder_jit_refill: skip reason=stale_ranking";
    log_jit_enqueue_summary("stale_ranking");
    return;
  }

  const int begin_layer = std::max(0, current_layer + 1);
  const int end_layer = std::min(metas->num_encoder_moe_layer, current_layer + metas->erpp_encoder_jit_refill_window + 1);
  if (begin_layer >= end_layer) {
    if (log_enabled) {
      LOG(INFO) << "erpp_encoder_jit_refill: skip reason=no_future_layer_in_window"
                << " current=L" << current_layer
                << " window=" << metas->erpp_encoder_jit_refill_window;
    }
    log_jit_enqueue_summary("no_future_layer_in_window");
    return;
  }

  auto enqueue_one = [&](int layer_idx, int expert_idx, const char* mode,
                         double score, int occupancy, int floor_value,
                         int low_watermark, int predicted_missing,
                         int occupancy_gap, int distance) {
    TaskQueue* queue = queue_for_class_and_layer(PrefetchClass::kEncoderPredictor, layer_idx);
    if (queue == nullptr) {
      return false;
    }
    if (metas->chunk_prefetch) {
      add_separate_tasks_for_one_expert(layer_idx, expert_idx, queue, 0,
                                        metas->num_per_expert_param, false,
                                        current_forward_epoch,
                                        kCacheRequestEncoderJitRefill);
    } else {
      add_single_tasks_for_one_expert(layer_idx, expert_idx, queue, 0,
                                      metas->num_per_expert_param, false,
                                      current_forward_epoch,
                                      kCacheRequestEncoderJitRefill);
    }
    mark_encoder_jit_submitted(layer_idx, expert_idx);
    enqueued_experts += 1;
    if (log_enabled) {
      int rank = -1;
      if (layer_idx >= 0 && layer_idx < static_cast<int>(encoder_jit_rankings.size())) {
        const auto& ranking = encoder_jit_rankings[layer_idx];
        auto it = std::find(ranking.begin(), ranking.end(), expert_idx);
        if (it != ranking.end()) {
          rank = static_cast<int>(std::distance(ranking.begin(), it));
        }
      }
      const int budget = layer_idx < static_cast<int>(encoder_jit_budgets.size())
          ? encoder_jit_budgets[layer_idx]
          : 0;
      const char* reason = metas->enable_erpp_encoder_jit_topk_cover &&
              rank >= 0 && rank < budget
          ? "topk_cover"
          : "floor_deficit";
      LOG(INFO) << "erpp_encoder_jit_refill: enqueue target=L" << layer_idx
                << " expert=" << expert_idx
                << " rank=" << rank
                << " reason=" << reason
                << " mode=" << mode
                << " score=" << score
                << " occupancy=" << occupancy
                << " floor=" << floor_value
                << " low=" << low_watermark
                << " predicted_missing=" << predicted_missing
                << " occupancy_gap=" << occupancy_gap
                << " distance=" << distance
                << " forward_epoch=" << current_forward_epoch;
    }
    return true;
  };

  int enqueued = 0;
  if (auto_refill) {
    while (effective_per_idle_limit < 0 || enqueued < effective_per_idle_limit) {
      EncoderJitRefillCandidate best;
      for (int layer_idx = begin_layer; layer_idx < end_layer; layer_idx++) {
        if (!encoder_jit_layer_enabled(layer_idx)) {
          skipped_disabled_layers += 1;
          if (log_enabled) {
            LOG(INFO) << "erpp_encoder_jit_refill: skip target=L" << layer_idx
                      << " reason=layer_disabled"
                      << " mode=auto";
          }
          continue;
        }
        inspected_layers += 1;
        auto candidate = build_encoder_jit_refill_candidate(layer_idx);
        required_experts += static_cast<int>(candidate.required.size());
        if (log_enabled) {
          LOG(INFO) << "erpp_encoder_jit_refill: inspect current=L" << current_layer
                    << " target=L" << layer_idx
                    << " mode=auto"
                    << " occupancy=" << candidate.occupancy
                    << " floor=" << candidate.floor_value
                    << " low=" << candidate.low_watermark
                    << " budget=" << candidate.budget
                    << " predicted_need=" << candidate.predicted_need
                    << " predicted_missing=" << candidate.predicted_missing
                    << " scarcity=" << candidate.scarcity
                    << " demand_weight=" << candidate.demand_weight
                    << " occupancy_gap=" << candidate.occupancy_gap
                    << " distance=" << candidate.distance
                    << " required=" << candidate.required.size()
                    << " score=" << candidate.score
                    << " wait_penalty_us="
                    << (layer_idx < static_cast<int>(encoder_jit_wait_penalty_ema_us.size())
                        ? encoder_jit_wait_penalty_ema_us[layer_idx]
                        : 0.0);
        }
        if (!candidate.required.empty() && candidate.score > best.score) {
          best = std::move(candidate);
        }
      }
      if (best.layer_idx < 0 || best.required.empty() || best.score <= 0.0) {
        log_jit_enqueue_summary(enqueued > 0 ? "complete" : "no_candidate");
        return;
      }
      if (!enqueue_one(best.layer_idx, best.required.front(), "auto",
                       best.score, best.occupancy, best.floor_value,
                       best.low_watermark, best.predicted_missing,
                       best.occupancy_gap, best.distance)) {
        log_jit_enqueue_summary("queue_unavailable");
        return;
      }
      enqueued += 1;
    }
    per_idle_limited += 1;
    log_jit_enqueue_summary("per_idle_limit");
    return;
  }

  for (int layer_idx = begin_layer; layer_idx < end_layer; layer_idx++) {
    if (!encoder_jit_layer_enabled(layer_idx)) {
      skipped_disabled_layers += 1;
      if (log_enabled) {
        LOG(INFO) << "erpp_encoder_jit_refill: skip target=L" << layer_idx
                  << " reason=layer_disabled";
      }
      continue;
    }
    inspected_layers += 1;

    const int occupancy = cache->encoder_layer_cache_occupancy(layer_idx);
    const int budget = layer_idx < static_cast<int>(encoder_jit_budgets.size())
        ? encoder_jit_budgets[layer_idx]
        : 0;
    const int floor_value = encoder_jit_floor(layer_idx);
    const int low_watermark = encoder_jit_low_watermark(layer_idx);
    if (log_enabled) {
      LOG(INFO) << "erpp_encoder_jit_refill: inspect current=L" << current_layer
                << " target=L" << layer_idx
                << " occupancy=" << occupancy
                << " floor=" << floor_value
                << " low=" << low_watermark
                << " budget=" << budget
                << " topk_cover=" << metas->enable_erpp_encoder_jit_topk_cover;
    }
    auto required = build_encoder_jit_required_experts(
        layer_idx, occupancy, floor_value, low_watermark, budget);
    required_experts += static_cast<int>(required.size());
    for (int expert_idx : required) {
      if (metas->erpp_encoder_jit_refill_per_idle > 0 &&
          enqueued >= metas->erpp_encoder_jit_refill_per_idle) {
        per_idle_limited += 1;
        if (log_enabled) {
          LOG(INFO) << "erpp_encoder_jit_refill: skip target=L" << layer_idx
                    << " expert=" << expert_idx
                    << " reason=per_idle_limit"
                    << " limit=" << metas->erpp_encoder_jit_refill_per_idle;
        }
        log_jit_enqueue_summary("per_idle_limit");
        return;
      }
      if (!enqueue_one(layer_idx, expert_idx, "ordered", 0.0, occupancy,
                       floor_value, low_watermark, 0,
                       std::max(0, floor_value - occupancy),
                       layer_idx - current_layer)) {
        log_jit_enqueue_summary("queue_unavailable");
        return;
      }
      enqueued += 1;
    }
  }

  log_jit_enqueue_summary("complete");
}

void FetchScheduleWorker::store_erpp_encoder_jit_rankings(
    const ErppEncoderJitRankingsTask& task) {
  if (!metas->enable_erpp_encoder_jit_refill) {
    return;
  }
  if (task.generate_epoch != current_generate_epoch ||
      task.forward_epoch < current_forward_epoch) {
    if (log_prefetch_decision_enabled()) {
      LOG(INFO) << "erpp_encoder_jit_refill: drop stale rankings"
                << " task_forward_epoch=" << task.forward_epoch
                << " current_forward_epoch=" << current_forward_epoch
                << " task_generate_epoch=" << task.generate_epoch
                << " current_generate_epoch=" << current_generate_epoch;
    }
    return;
  }
  encoder_jit_rankings = task.rankings;
  encoder_jit_budgets = task.budgets;
  encoder_jit_forward_epoch = task.forward_epoch;
  encoder_jit_generate_epoch = task.generate_epoch;
  encoder_jit_submitted_mask.assign(
      metas->num_encoder_moe_layer,
      std::vector<uint8_t>(metas->num_expert, 0));
  if (log_prefetch_decision_enabled()) {
    LOG(INFO) << "erpp_encoder_jit_refill: store rankings"
              << " forward_epoch=" << task.forward_epoch
              << " generate_epoch=" << task.generate_epoch
              << " layers=" << encoder_jit_rankings.size();
  }
  if (current_layer >= 0) {
    maybe_enqueue_encoder_jit_refill();
  }
}

void FetchScheduleWorker::do_one_task_impl(ErppEncoderJitRankingsTask *task) {
  store_erpp_encoder_jit_rankings(*task);
}

void FetchScheduleWorker::set_phase(SchedulerPhase next_phase) {
  if (phase == next_phase) {
    return;
  }
  phase = next_phase;
  if (phase == kDecoderPredictorPhase) {
    log_encoder_experts_in_cache_on_decoder_entry(metas, model_loader, cache);
    clear_decoder_warmup_plan_queue();
  }
}

void FetchScheduleWorker::clear_decoder_warmup_plan_queue() {
  prefetch_queues.decoder_warmup_plan_queue.clear();
  decoder_warmup_seen.clear();
}

void FetchScheduleWorker::reset_for_generate(
    int64_t next_generate_epoch,
    int64_t next_forward_epoch,
    DecoderWarmupAction action) {
  reset_task.next_generate_epoch = next_generate_epoch;
  reset_task.next_forward_epoch = next_forward_epoch;
  reset_task.action = action;
  auto handle = WorkerThread<FetchScheduleTaskBase*>::add_one_task(&reset_task);
  WorkerThread<FetchScheduleTaskBase*>::wait_progress(handle);
}

bool FetchScheduleWorker::parse_layer_expert_plan(
    const std::string& plan,
    std::vector<std::pair<int, int>>& out) {
  out.clear();
  if (plan.empty()) {
    return true;
  }
  std::stringstream ss(plan);
  std::string entry;
  while (std::getline(ss, entry, ',')) {
    auto sep = entry.find(':');
    CHECK(!entry.empty() && sep != std::string::npos)
        << "invalid decoder_warmup_expert_plan entry: " << entry
        << ", plan=" << plan;
    int layer_idx = std::stoi(entry.substr(0, sep));
    int expert_idx = std::stoi(entry.substr(sep + 1));
    CHECK(metas->is_decoder_layer(layer_idx))
        << "decoder_warmup_expert_plan contains non-decoder layer: " << layer_idx;
    CHECK(expert_idx >= 0 && expert_idx < metas->num_expert)
        << "decoder_warmup_expert_plan expert out of range: " << expert_idx;
    out.push_back({layer_idx, expert_idx});
  }
  return true;
}

void FetchScheduleWorker::rebuild_decoder_warmup_queue() {
  clear_decoder_warmup_plan_queue();
  if (!metas->enable_decoder_warmup_overlap) {
    return;
  }
  std::vector<std::pair<int, int>> parsed;
  parse_layer_expert_plan(metas->decoder_warmup_expert_plan, parsed);
  for (auto [layer_idx, expert_idx] : parsed) {
    auto gid = flatten_expert(layer_idx, expert_idx);
    if (decoder_warmup_seen.insert(gid).second) {
      const int num_chunks = metas->chunk_prefetch ? metas->num_per_expert_param : 1;
      for (int j = 0; j < num_chunks; j++) {
        CopyTask task;
        task.start_mem_buf_idx = metas->chunk_prefetch ? j : 0;
        task.stop_mem_buf_idx = metas->chunk_prefetch ? j + 1 : metas->num_per_expert_param;
        task.expert = model_loader->get_source(layer_idx, expert_idx);
        task.is_precise = false;
        task.forward_epoch = current_forward_epoch;
        task.generate_epoch = current_generate_epoch;
        task.request_type = kCacheRequestDecoderWarmupPrefetch;
        NVTX_DETAIL_MARK(std::string("sched/enqueue_decoder_warmup_prefetch") + " L" +
                        std::to_string(layer_idx) +
                        " E" + std::to_string(expert_idx) +
                        " P" + std::to_string(task.start_mem_buf_idx) +
                        "-" + std::to_string(task.stop_mem_buf_idx) +
                        " forward_epoch=" + std::to_string(task.forward_epoch));
        prefetch_queues.decoder_warmup_plan_queue.push(task);
      }
    }
  }
}

void FetchScheduleWorker::clear_prefetch_queues_up_to_layer(int layer_idx) {
  if (layer_idx < 0) {
    return;
  }
  clear_prefetch_class_up_to_layer(PrefetchClass::kEncoderPredictor, layer_idx);
  clear_prefetch_class_up_to_layer(PrefetchClass::kDecoderPredictor, layer_idx);
}

void FetchScheduleWorker::start_forward_epoch(
    int64_t forward_epoch,
    DecoderWarmupAction decoder_warmup_action) {
  if (forward_epoch > current_forward_epoch) {
    current_forward_epoch = forward_epoch;
    current_layer = -1;
    clear_encoder_jit_state();
    clear_encoder_prefetch_metrics();
    clear_stale_prefetch_queues_before_epoch(current_forward_epoch);
    precise_job_queue.clear();
  }
  switch (decoder_warmup_action) {
    case DecoderWarmupAction::kPreserve: {
      break;
    }
    case DecoderWarmupAction::kClear: {
      clear_decoder_warmup_plan_queue();
      break;
    }
    case DecoderWarmupAction::kRebuildForGenerateStart: {
      set_phase(kEncoderPhase);
      rebuild_decoder_warmup_queue();
      break;
    }
  }
}

void FetchScheduleWorker::advance_actual_layer(int64_t forward_epoch, int layer_idx) {
  if (forward_epoch > current_forward_epoch) {
    current_forward_epoch = forward_epoch;
    current_layer = -1;
    clear_encoder_jit_state();
    clear_encoder_prefetch_metrics();
    clear_stale_prefetch_queues_before_epoch(current_forward_epoch);
    precise_job_queue.clear();
  }
  if (forward_epoch == current_forward_epoch && layer_idx > current_layer) {
    current_layer = layer_idx;
  }
  if (metas->is_decoder_layer(layer_idx)) {
    set_phase(kDecoderPredictorPhase);
  }
  clear_prefetch_queues_up_to_layer(current_layer);
}

void FetchScheduleWorker::preempt_one_expert(int layer_idx, int64_t expert_idx) {
  TRACE_EVENT_GURAD(kFetchScheduler, "preemot_one_expert");

  auto e = model_loader->get_source(layer_idx, expert_idx);

  if (cache->is_in_cache(e) == false) {
    // a completely missed expert
    add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, 0, metas->num_per_expert_param, true, current_forward_epoch, kCacheRequestDemand);
  } else if (e->num_ready == metas->num_per_expert_param) {
    // bypass a fully fetched expert, no need to add task
    auto launch_status = e->expert_status.transfer(kReady, kLaunching, false);
    if (launch_status == kReady) {
      cache_hit(e, true);
    } else if (launch_status == kLaunching) {
      LOG(TRACE) << "scheduler: skip redundant launch for " << e->toString();
    } else if (launch_status == kUsing) {
      LOG(TRACE) << "scheduler: wait active expert before demand launch " << e->toString();
      e->expert_status.wait(kReady, kLaunching);
      cache_hit(e, true);
    } else {
      CHECK(launch_status == kReady || launch_status == kLaunching || launch_status == kUsing)
          << "launch ready expert " << e->toString()
          << " but current status is " << launch_status;
    }
  } else if (e == current_task.expert) {
    current_task.is_precise = true;
    if (current_task.stop_mem_buf_idx == metas->num_per_expert_param) {
      // no need to add a redundant task
      // note there will be corresponding fetchdone for this task.
      cache_hit(e, true);
    } else {
      add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, current_task.stop_mem_buf_idx, metas->num_per_expert_param, true, current_forward_epoch, kCacheRequestDemand);
    }
  } else {
    // partial expert
    add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, e->num_ready, metas->num_per_expert_param, true, current_forward_epoch, kCacheRequestDemand);
  }
}

void FetchScheduleWorker::preempt_one_layer_without_reorder_(int layer_idx, int64_t *expert_idxs, size_t num_expert) {
  TRACE_EVENT_GURAD(kFetchScheduler, "preempt_one_layer_without_reorder_");
  LOG_BLOCK(DEBUG, logger, {
    logger << "scheduler: preempting one layer " << layer_idx << " with expert " << array_to_str(expert_idxs, num_expert);
  });

  bool preceeding_experts_in_cache = true;
  std::unordered_set<int64_t> seen_demand_experts;

  // examine expert status, classify them, and bypass experts that is already fetched.
  // for ready expert, we need to let cache know we access it and update it's priority
  // for not-ready but in cache expert, it will have corresponding task, and cache->hit will be called in the task impl
  // for not-ready and not in cache expert, it will first be called with cache->miss, then be called with cache->hit, which doesn't hurt.

  // ready, current, partial, miss
  for (int i = 0; i < num_expert; i++) {
    auto e = model_loader->get_source(layer_idx, expert_idxs[i]);
    auto demand_gid = flatten_expert(layer_idx, e->expert_idx);
    if (!seen_demand_experts.insert(demand_gid).second) {
      LOG(TRACE) << "scheduler: skip duplicate demand expert " << e->toString();
      continue;
    }

    // a completely missed expert
    if (cache->is_in_cache(e) == false) {
      preceeding_experts_in_cache = false;
      add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, 0, metas->num_per_expert_param, true, current_forward_epoch, kCacheRequestDemand);
      continue;
    }

    // an expert fully/partially in cache, but it may be evicted by preceeding miss expert, so we need to add redundant task for it
    if (preceeding_experts_in_cache == false) {
      add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, 0, metas->num_per_expert_param, true, current_forward_epoch, kCacheRequestDemand);
      continue;
    }

    // bypass a fully fetched expert, no need to add task
    if (e->num_ready == metas->num_per_expert_param) {
      auto launch_status = e->expert_status.transfer(kReady, kLaunching, false);
      if (launch_status == kReady) {
        cache_hit(e, true);
      } else if (launch_status == kLaunching) {
        LOG(TRACE) << "scheduler: skip redundant launch for " << e->toString();
      } else if (launch_status == kUsing) {
        LOG(TRACE) << "scheduler: wait active expert before demand launch " << e->toString();
        e->expert_status.wait(kReady, kLaunching);
        cache_hit(e, true);
      } else {
        CHECK(launch_status == kReady || launch_status == kLaunching || launch_status == kUsing)
            << "launch ready expert " << e->toString()
            << " but current status is " << launch_status;
      }
      continue;
    }

    if (e == current_task.expert) {
      current_task.is_precise = true;
      if (current_task.stop_mem_buf_idx == metas->num_per_expert_param) {
        // no need to add a redundant task
        // note there will be corresponding fetchdone for this task.
        cache_hit(e, true);
      } else {
        add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, current_task.stop_mem_buf_idx, metas->num_per_expert_param, true, current_forward_epoch, kCacheRequestDemand);
      }
      continue;
    }

    // partial expert
    add_single_tasks_for_one_expert(layer_idx, e->expert_idx, &precise_job_queue, e->num_ready, metas->num_per_expert_param, true, current_forward_epoch, kCacheRequestDemand);
  }
}
void FetchScheduleWorker::add_single_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue* queue, int starting_mem_buffer, int stop_mem_buffer, bool is_precise, int64_t forward_epoch, CacheRequestType request_type) {
  auto expert_handler = model_loader->get_source(layer_idx, expert_idx);
  std::string request_label = "decoder_predictor_prefetch";
  if (request_type == kCacheRequestEncoderPredictorPrefetch) {
    request_label = "encoder_predictor_prefetch";
  } else if (request_type == kCacheRequestEncoderJitRefill) {
    request_label = "encoder_jit_refill";
  } else if (request_type == kCacheRequestDecoderWarmupPrefetch) {
    request_label = "decoder_warmup_prefetch";
  } else if (request_type == kCacheRequestDecoderPredictorPrefetch) {
    request_label = "decoder_predictor_prefetch";
  } else if (request_type == kCacheRequestDemand) {
    request_label = "demand";
  }
  NVTX_RANGE(std::string("submit/") + request_label + "_io " +
             "L" + std::to_string(layer_idx) +
             " E" + std::to_string(expert_idx) +
             " P" + std::to_string(starting_mem_buffer) +
             "-" + std::to_string(stop_mem_buffer) +
             " forward_epoch=" + std::to_string(forward_epoch));
  CopyTask task;
  task.start_mem_buf_idx = starting_mem_buffer;
  task.stop_mem_buf_idx = stop_mem_buffer;
  task.expert = expert_handler;
  task.is_precise = is_precise;
  task.forward_epoch = forward_epoch;
  task.generate_epoch = current_generate_epoch;
  task.request_type = request_type;
  LOG(TRACE) << "scheduler: add prefetch task for one param " << task.toString();
  queue->push(task);
  if (!is_precise && is_encoder_prefetch_request(request_type)) {
    mark_encoder_prefetch_enqueued(layer_idx, expert_idx, stop_mem_buffer - starting_mem_buffer);
  }
}

void FetchScheduleWorker::add_separate_tasks_for_one_expert(int layer_idx, int expert_idx, TaskQueue *queue, int start_mem_buf_idx, int stop_mem_buf_idx, bool is_precise, int64_t forward_epoch, CacheRequestType request_type) {
  for (int j = start_mem_buf_idx; j < stop_mem_buf_idx; j++) {
    add_single_tasks_for_one_expert(layer_idx, expert_idx, queue, j, j+1, is_precise, forward_epoch, request_type);
  }
}

void FetchScheduleWorker::pop_next_task(CopyTask &task, bool &found) {
  found = false;
  // Runtime priority: demand > ERPP encoder prefetch > normal predictor
  // prefetch > decoder warmup overlap.
  if (!precise_job_queue.empty()) {
    task = precise_job_queue.front();
    precise_job_queue.pop();
    found = true;
    if (log_prefetch_decision_enabled()) {
      LOG(INFO) << "prefetch_decision: choose demand L"
                << task.expert->layer_idx << " E" << task.expert->expert_idx
                << " current_layer=" << current_layer
                << " forward_epoch=" << current_forward_epoch;
    }
    return;
  }
  drain_reclaimable_updates();
  if (log_prefetch_decision_enabled()) {
    LOG(INFO) << "prefetch_decision: scan phase=" << phase
              << " current_layer=" << current_layer
              << " forward_epoch=" << current_forward_epoch
              << " encoder_predictor_prefetch_pending=" << has_pending_prefetch_for_class(PrefetchClass::kEncoderPredictor)
              << " decoder_predictor_prefetch_pending=" << has_pending_prefetch_for_class(PrefetchClass::kDecoderPredictor)
              << " decoder_warmup_pending=" << has_pending_prefetch_for_class(PrefetchClass::kDecoderWarmup)
              << " has_reclaimable_encoder=" << cache->has_reclaimable_encoder();
  }
  bool block_lower_priority_for_encoder = false;
  if (pop_next_prefetch_for_class(PrefetchClass::kEncoderPredictor, task, &block_lower_priority_for_encoder)) {
    found = true;
    return;
  }
  if (block_lower_priority_for_encoder) {
    if (log_prefetch_decision_enabled()) {
      LOG(INFO) << "prefetch_decision: hold lower priority prefetches because ERPP encoder queue is waiting reclaimable"
                << " current_layer=" << current_layer
                << " forward_epoch=" << current_forward_epoch;
    }
    return;
  }
  if (pop_next_prefetch_for_class(PrefetchClass::kDecoderPredictor, task)) {
    found = true;
    return;
  }
  if (pop_next_prefetch_for_class(PrefetchClass::kDecoderWarmup, task)) {
    found = true;
    return;
  }
}

FetchScheduleWorker::PrefetchClass FetchScheduleWorker::prefetch_class_for_request(CacheRequestType request_type) const {
  if (request_type == kCacheRequestEncoderPredictorPrefetch ||
      request_type == kCacheRequestEncoderJitRefill) {
    return PrefetchClass::kEncoderPredictor;
  }
  if (request_type == kCacheRequestDecoderWarmupPrefetch) {
    return PrefetchClass::kDecoderWarmup;
  }
  return PrefetchClass::kDecoderPredictor;
}

CacheRequestType FetchScheduleWorker::request_type_for_prefetch_class(PrefetchClass cls) const {
  switch (cls) {
    case PrefetchClass::kEncoderPredictor: return kCacheRequestEncoderPredictorPrefetch;
    case PrefetchClass::kDecoderPredictor: return kCacheRequestDecoderPredictorPrefetch;
    case PrefetchClass::kDecoderWarmup: return kCacheRequestDecoderWarmupPrefetch;
  }
  CHECK(false) << "unknown prefetch class";
  return kCacheRequestDecoderPredictorPrefetch;
}

bool FetchScheduleWorker::requires_encoder_phase(PrefetchClass cls) const {
  return cls == PrefetchClass::kEncoderPredictor;
}

bool FetchScheduleWorker::requires_reclaimable_encoder(PrefetchClass cls) const {
  return cls == PrefetchClass::kEncoderPredictor || cls == PrefetchClass::kDecoderWarmup;
}

bool FetchScheduleWorker::blocks_lower_priority_when_pending(PrefetchClass cls) const {
  return cls == PrefetchClass::kEncoderPredictor;
}

bool FetchScheduleWorker::replace_same_layer_on_enqueue(PrefetchClass cls) const {
  return cls == PrefetchClass::kEncoderPredictor || cls == PrefetchClass::kDecoderPredictor;
}

FetchScheduleWorker::TaskQueue* FetchScheduleWorker::queue_for_class_and_layer(PrefetchClass cls, int layer_idx) {
  switch (cls) {
    case PrefetchClass::kEncoderPredictor:
      CHECK(layer_idx >= 0 && layer_idx < static_cast<int>(prefetch_queues.encoder_predictor_by_layer.size()));
      return &prefetch_queues.encoder_predictor_by_layer[layer_idx];
    case PrefetchClass::kDecoderPredictor:
      CHECK(layer_idx >= 0 && layer_idx < static_cast<int>(prefetch_queues.decoder_predictor_by_layer.size()));
      return &prefetch_queues.decoder_predictor_by_layer[layer_idx];
    case PrefetchClass::kDecoderWarmup:
      return &prefetch_queues.decoder_warmup_plan_queue;
  }
  CHECK(false) << "unknown prefetch class";
  return nullptr;
}

const FetchScheduleWorker::TaskQueue* FetchScheduleWorker::queue_for_class_and_layer(PrefetchClass cls, int layer_idx) const {
  switch (cls) {
    case PrefetchClass::kEncoderPredictor:
      CHECK(layer_idx >= 0 && layer_idx < static_cast<int>(prefetch_queues.encoder_predictor_by_layer.size()));
      return &prefetch_queues.encoder_predictor_by_layer[layer_idx];
    case PrefetchClass::kDecoderPredictor:
      CHECK(layer_idx >= 0 && layer_idx < static_cast<int>(prefetch_queues.decoder_predictor_by_layer.size()));
      return &prefetch_queues.decoder_predictor_by_layer[layer_idx];
    case PrefetchClass::kDecoderWarmup:
      return &prefetch_queues.decoder_warmup_plan_queue;
  }
  CHECK(false) << "unknown prefetch class";
  return nullptr;
}

bool FetchScheduleWorker::has_pending_prefetch_for_class(PrefetchClass cls) {
  if (cls == PrefetchClass::kEncoderPredictor) {
    for (auto& queue : prefetch_queues.encoder_predictor_by_layer) {
      if (!queue.empty()) { return true; }
    }
    return false;
  }
  if (cls == PrefetchClass::kDecoderPredictor) {
    for (auto& queue : prefetch_queues.decoder_predictor_by_layer) {
      if (!queue.empty()) { return true; }
    }
    return false;
  }
  return !prefetch_queues.decoder_warmup_plan_queue.empty();
}

void FetchScheduleWorker::clear_prefetch_class(PrefetchClass cls) {
  if (cls == PrefetchClass::kEncoderPredictor) {
    clear_encoder_jit_state();
    for (auto& queue : prefetch_queues.encoder_predictor_by_layer) { queue.clear(); }
    return;
  }
  if (cls == PrefetchClass::kDecoderPredictor) {
    for (auto& queue : prefetch_queues.decoder_predictor_by_layer) { queue.clear(); }
    return;
  }
  prefetch_queues.decoder_warmup_plan_queue.clear();
}

void FetchScheduleWorker::clear_prefetch_class_up_to_layer(PrefetchClass cls, int layer_idx) {
  if (layer_idx < 0) { return; }
  auto clear_by_layer = [layer_idx](std::vector<TaskQueue>& queues) {
    if (queues.empty()) { return; }
    int stop_layer = std::min<int>(layer_idx, queues.size() - 1);
    for (int l = 0; l <= stop_layer; l++) { queues[l].clear(); }
  };
  if (cls == PrefetchClass::kEncoderPredictor) {
    if (log_prefetch_decision_enabled()) {
      LOG(INFO) << "prefetch_decision: clear ERPP encoder prefetch queues up to layer " << layer_idx;
    }
    clear_by_layer(prefetch_queues.encoder_predictor_by_layer);
    if (!encoder_jit_submitted_mask.empty()) {
      int stop_layer = std::min<int>(layer_idx, encoder_jit_submitted_mask.size() - 1);
      for (int l = 0; l <= stop_layer; l++) {
        std::fill(encoder_jit_submitted_mask[l].begin(), encoder_jit_submitted_mask[l].end(), 0);
      }
    }
  } else if (cls == PrefetchClass::kDecoderPredictor) {
    clear_by_layer(prefetch_queues.decoder_predictor_by_layer);
  }
}

void FetchScheduleWorker::clear_prefetch_class_for_layer(PrefetchClass cls, int64_t forward_epoch, int layer_idx) {
  TaskQueue* queue = queue_for_class_and_layer(cls, layer_idx);
  TaskQueue kept;
  while (!queue->empty()) {
    CopyTask task = queue->front();
    queue->pop();
    if (task.expert != nullptr && task.forward_epoch == forward_epoch && task.expert->layer_idx == layer_idx) {
      continue;
    }
    kept.push(task);
  }
  while (!kept.empty()) {
    CopyTask task = kept.front();
    kept.pop();
    queue->push(task);
  }
}

void FetchScheduleWorker::prune_prefetch_class(PrefetchClass cls) {
  auto prune_queue = [this](TaskQueue& queue, const char* stale_label) {
    TaskQueue kept;
    while (!queue.empty()) {
      CopyTask task = queue.front();
      queue.pop();
      if (task.expert == nullptr) {
        continue;
      }
      if (is_stale_prefetch(task.forward_epoch, task.expert->layer_idx)) {
        NVTX_DETAIL_MARK(std::string(stale_label) + " L" +
                         std::to_string(task.expert->layer_idx) +
                         " E" + std::to_string(task.expert->expert_idx) +
                         " forward_epoch=" + std::to_string(task.forward_epoch) +
                         " current_layer=" + std::to_string(current_layer));
        continue;
      }
      if (task.expert->num_ready >= task.stop_mem_buf_idx) {
        continue;
      }
      kept.push(task);
    }
    while (!kept.empty()) {
      CopyTask task = kept.front();
      kept.pop();
      queue.push(task);
    }
  };

  auto prune_decoder_warmup_queue = [this]() {
    TaskQueue kept;
    auto& queue = prefetch_queues.decoder_warmup_plan_queue;
    while (!queue.empty()) {
      CopyTask task = queue.front();
      queue.pop();
      if (task.expert == nullptr) {
        continue;
      }
      if (task.expert->num_ready >= task.stop_mem_buf_idx) {
        continue;
      }
      kept.push(task);
    }
    while (!kept.empty()) {
      CopyTask task = kept.front();
      kept.pop();
      queue.push(task);
    }
  };

  if (cls == PrefetchClass::kEncoderPredictor) {
    for (auto& queue : prefetch_queues.encoder_predictor_by_layer) {
      prune_queue(queue, "sched/drop_stale_encoder_predictor_prefetch");
    }
  } else if (cls == PrefetchClass::kDecoderPredictor) {
    for (auto& queue : prefetch_queues.decoder_predictor_by_layer) {
      prune_queue(queue, "sched/drop_stale_decoder_predictor_prefetch");
    }
  } else {
    prune_decoder_warmup_queue();
  }
}

bool FetchScheduleWorker::pop_next_prefetch_for_class(PrefetchClass cls, CopyTask& task, bool* blocked_lower_priority) {
  if (requires_encoder_phase(cls) && phase != kEncoderPhase) {
    if (log_prefetch_decision_enabled() && has_pending_prefetch_for_class(cls)) {
      LOG(INFO) << "prefetch_decision: clear ERPP encoder prefetch because scheduler left encoder phase"
                << " phase=" << phase
                << " current_layer=" << current_layer
                << " forward_epoch=" << current_forward_epoch;
    }
    clear_prefetch_class(cls);
    return false;
  }
  if (cls == PrefetchClass::kDecoderWarmup && !metas->enable_decoder_warmup_overlap) {
    return false;
  }

  if (cls == PrefetchClass::kEncoderPredictor || cls == PrefetchClass::kDecoderWarmup) {
    drain_reclaimable_updates();
  }
  prune_prefetch_class(cls);
  if (!has_pending_prefetch_for_class(cls)) {
    if (log_prefetch_decision_enabled() && cls == PrefetchClass::kEncoderPredictor) {
      LOG(INFO) << "prefetch_decision: no ERPP encoder prefetch candidate after prune"
                << " current_layer=" << current_layer
                << " forward_epoch=" << current_forward_epoch;
    }
    return false;
  }
  if (cls != PrefetchClass::kEncoderPredictor &&
      requires_reclaimable_encoder(cls) && !cache->has_reclaimable_encoder()) {
    NVTX_DETAIL_MARK("sched/wait_decoder_warmup_prefetch_reclaimable");
    return false;
  }

  if (cls == PrefetchClass::kEncoderPredictor) {
    bool legacy_waiting_for_reclaimable = false;
    for (auto& queue : prefetch_queues.encoder_predictor_by_layer) {
      while (!queue.empty()) {
        CopyTask candidate = queue.front();
        queue.pop();
        if (candidate.expert == nullptr) {
          if (log_prefetch_decision_enabled()) {
            LOG(INFO) << "prefetch_decision: skip ERPP encoder prefetch with null expert";
          }
          continue;
        }
        if (is_stale_prefetch(candidate.forward_epoch, candidate.expert->layer_idx)) {
          if (log_prefetch_decision_enabled()) {
            LOG(INFO) << "prefetch_decision: skip stale ERPP encoder prefetch L"
                      << candidate.expert->layer_idx << " E" << candidate.expert->expert_idx
                      << " P" << candidate.start_mem_buf_idx << "-" << candidate.stop_mem_buf_idx
                      << " task_forward_epoch=" << candidate.forward_epoch
                      << " current_layer=" << current_layer
                      << " current_forward_epoch=" << current_forward_epoch;
          }
          continue;
        }
        if (candidate.expert->num_ready >= candidate.stop_mem_buf_idx) {
          if (log_prefetch_decision_enabled()) {
            LOG(INFO) << "prefetch_decision: skip ready ERPP encoder prefetch L"
                      << candidate.expert->layer_idx << " E" << candidate.expert->expert_idx
                      << " P" << candidate.start_mem_buf_idx << "-" << candidate.stop_mem_buf_idx
                      << " num_ready=" << candidate.expert->num_ready;
          }
          continue;
        }
        if (candidate.request_type == kCacheRequestEncoderPredictorPrefetch &&
            !cache->has_reclaimable_encoder()) {
          legacy_waiting_for_reclaimable = true;
          queue.push(candidate);
          if (log_prefetch_decision_enabled()) {
            LOG(INFO) << "prefetch_decision: ERPP encoder legacy prefetch waits reclaimable victim L"
                      << candidate.expert->layer_idx << " E" << candidate.expert->expert_idx
                      << " current_layer=" << current_layer
                      << " forward_epoch=" << current_forward_epoch;
          }
          if (log_erpp_encoder_diagnostics_enabled()) {
            LOG(INFO) << "erpp_encoder_diagnostics: scheduler_block"
                      << " reason=no_reclaimable_encoder"
                      << " request=encoder_predictor_prefetch"
                      << " target_layer=" << candidate.expert->layer_idx
                      << " expert=" << candidate.expert->expert_idx
                      << " current_layer=" << current_layer
                      << " forward_epoch=" << current_forward_epoch
                      << " generate_epoch=" << current_generate_epoch;
          }
          break;
        }
        if (candidate.request_type == kCacheRequestEncoderJitRefill &&
            !encoder_jit_can_dispatch(candidate)) {
          queue.push(candidate);
          if (log_prefetch_decision_enabled()) {
            LOG(INFO) << "erpp_encoder_jit_refill: skip target=L"
                      << candidate.expert->layer_idx
                      << " expert=" << candidate.expert->expert_idx
                      << " reason=no_safe_victim"
                      << " current=L" << current_layer
                      << " floor=" << encoder_jit_floor(candidate.expert->layer_idx);
          }
          if (log_erpp_encoder_diagnostics_enabled()) {
            LOG(INFO) << "erpp_encoder_diagnostics: scheduler_block"
                      << " reason=no_safe_victim"
                      << " request=encoder_jit_refill"
                      << " target_layer=" << candidate.expert->layer_idx
                      << " expert=" << candidate.expert->expert_idx
                      << " current_layer=" << current_layer
                      << " floor=" << encoder_jit_floor(candidate.expert->layer_idx)
                      << " has_reclaimable_encoder=" << cache->has_reclaimable_encoder()
                      << " forward_epoch=" << current_forward_epoch
                      << " generate_epoch=" << current_generate_epoch;
          }
          ensure_encoder_prefetch_metrics();
          if (candidate.expert->layer_idx >= 0 &&
              candidate.expert->layer_idx < static_cast<int>(encoder_prefetch_metrics.size())) {
            encoder_prefetch_metrics[candidate.expert->layer_idx].no_victim_blocks += 1;
          }
          break;
        }
        NVTX_DETAIL_MARK(std::string(candidate.request_type == kCacheRequestEncoderJitRefill
                             ? "sched/pop_encoder_jit_refill"
                             : "sched/pop_encoder_predictor_prefetch") + " L" +
                         std::to_string(candidate.expert->layer_idx) +
                         " E" + std::to_string(candidate.expert->expert_idx) +
                         " P" + std::to_string(candidate.start_mem_buf_idx) +
                         "-" + std::to_string(candidate.stop_mem_buf_idx) +
                         " forward_epoch=" + std::to_string(candidate.forward_epoch));
        if (log_prefetch_decision_enabled()) {
          LOG(INFO) << "prefetch_decision: choose ERPP encoder prefetch L"
                    << candidate.expert->layer_idx << " E" << candidate.expert->expert_idx
                    << " P" << candidate.start_mem_buf_idx << "-" << candidate.stop_mem_buf_idx
                    << " current_layer=" << current_layer
                    << " forward_epoch=" << current_forward_epoch;
        }
        if (log_erpp_encoder_diagnostics_enabled()) {
          LOG(INFO) << "erpp_encoder_diagnostics: scheduler_dispatch"
                    << " request="
                    << (candidate.request_type == kCacheRequestEncoderJitRefill
                            ? "encoder_jit_refill"
                            : "encoder_predictor_prefetch")
                    << " target_layer=" << candidate.expert->layer_idx
                    << " expert=" << candidate.expert->expert_idx
                    << " part=" << candidate.start_mem_buf_idx << "-" << candidate.stop_mem_buf_idx
                    << " current_layer=" << current_layer
                    << " forward_epoch=" << current_forward_epoch
                    << " generate_epoch=" << current_generate_epoch;
        }
        task = candidate;
        return true;
      }
    }
    if (legacy_waiting_for_reclaimable) {
      if (blocked_lower_priority != nullptr) {
        *blocked_lower_priority = true;
      }
      NVTX_DETAIL_MARK("sched/wait_encoder_predictor_prefetch_reclaimable");
      if (log_prefetch_decision_enabled()) {
        LOG(INFO) << "prefetch_decision: hold lower priority prefetches because legacy ERPP encoder queue is waiting reclaimable"
                  << " current_layer=" << current_layer
                  << " forward_epoch=" << current_forward_epoch;
      }
      return false;
    }
    if (log_prefetch_decision_enabled()) {
      LOG(INFO) << "prefetch_decision: ERPP encoder queue exhausted without dispatch"
                << " current_layer=" << current_layer
                << " forward_epoch=" << current_forward_epoch;
    }
    return false;
  }

  if (cls == PrefetchClass::kDecoderPredictor) {
    int best_layer = -1;
    std::tuple<int, int, int> best_key{INT_MAX, INT_MAX, INT_MAX};
    for (int layer_idx = 0; layer_idx < int(prefetch_queues.decoder_predictor_by_layer.size()); layer_idx++) {
      if (prefetch_queues.decoder_predictor_by_layer[layer_idx].empty()) {
        continue;
      }
      int bucket = 0;
      int distance = 0;
      if (current_layer < 0) {
        bucket = 1;
        distance = layer_idx;
      } else if (layer_idx == current_layer) {
        bucket = 0;
        distance = 0;
      } else if (layer_idx > current_layer) {
        bucket = 1;
        distance = layer_idx - current_layer;
      } else {
        bucket = 2;
        distance = current_layer - layer_idx;
      }
      auto key = std::make_tuple(bucket, distance, layer_idx);
      if (key < best_key) {
        best_key = key;
        best_layer = layer_idx;
      }
    }
    if (best_layer < 0) {
      return false;
    }
    auto& queue = prefetch_queues.decoder_predictor_by_layer[best_layer];
    task = queue.front();
    queue.pop();
    NVTX_DETAIL_MARK(std::string("sched/pop_decoder_predictor_prefetch") + " L" +
                     std::to_string(task.expert->layer_idx) +
                     " E" + std::to_string(task.expert->expert_idx) +
                     " P" + std::to_string(task.start_mem_buf_idx) +
                     "-" + std::to_string(task.stop_mem_buf_idx) +
                     " forward_epoch=" + std::to_string(task.forward_epoch));
    return true;
  }

  while (!prefetch_queues.decoder_warmup_plan_queue.empty()) {
    CopyTask candidate = prefetch_queues.decoder_warmup_plan_queue.front();
    prefetch_queues.decoder_warmup_plan_queue.pop();
    if (candidate.expert == nullptr ||
        candidate.expert->num_ready >= candidate.stop_mem_buf_idx) {
      continue;
    }
    NVTX_DETAIL_MARK(std::string("sched/pop_decoder_warmup_prefetch") + " L" +
                     std::to_string(candidate.expert->layer_idx) +
                     " E" + std::to_string(candidate.expert->expert_idx) +
                     " P" + std::to_string(candidate.start_mem_buf_idx) +
                     "-" + std::to_string(candidate.stop_mem_buf_idx) +
                     " forward_epoch=" + std::to_string(candidate.forward_epoch));
    if (log_prefetch_decision_enabled()) {
      LOG(INFO) << "prefetch_decision: choose decoder warmup L"
                << candidate.expert->layer_idx << " E" << candidate.expert->expert_idx
                << " P" << candidate.start_mem_buf_idx << "-" << candidate.stop_mem_buf_idx
                << " current_layer=" << current_layer
                << " forward_epoch=" << current_forward_epoch;
    }
    task = candidate;
    return true;
  }
  return false;
}
void PrefetchMngr::init_gpu_mem_buffer() {
  // hack: append a dummy chunk to each host experts
  if (metas->expert_mem_scale != 1.0) {
    model_loader->add_all_dummy_expert_params();
  }

  uint64_t cache_len = 0;
  if (metas->per_layer_cache) {
    // cache_len = round(metas->cache_rate * metas->num_expert) * metas->num_layer;
    cache_len = round(metas->cache_rate * metas->num_layer * metas->num_expert);
  } else {
    cache_len = round(metas->cache_rate * metas->num_layer * metas->num_expert);
  }
  cache->init_gpu_mem_buffer(cache_len);
  model_loader->mem_mngr_ctx->dummy_physical = cache->cache_slots->slots.front().unused_mems.front();
}

void PrefetchMngr::reset_for_generate() {
  if (!metas->reset_cache_on_generate_start) {
    return;
  }
  bool reset_finished = false;
  struct ResetInProgressGuard {
    std::atomic<bool>& reset_in_progress;
    bool& reset_finished;
    ~ResetInProgressGuard() {
      if (!reset_finished) {
        reset_in_progress.store(false, std::memory_order_release);
      }
    }
  } reset_guard{reset_in_progress, reset_finished};

  NVTX_RANGE("cache_init/reset_for_generate");
  Timer total_timer;
  uint64_t reset_us = 0;
  uint64_t load_us = 0;
  uint64_t rebuild_us = 0;
  CHECK(!reset_in_progress.exchange(true, std::memory_order_acq_rel));
  generate_epoch += 1;
  forward_epoch += 1;
  LOG(INFO) << "reset_for_generate: begin generate_epoch=" << generate_epoch
            << " forward_epoch=" << forward_epoch;
  LOG(INFO) << "reset_for_generate: begin predict reset barrier";
  if (erpp_encoder_predict_thread != nullptr) {
    erpp_encoder_predict_thread->begin_reset_for_generate();
  }
  predict_thread->begin_reset_for_generate();
  predict_thread->wait_until_idle();
  if (erpp_encoder_predict_thread != nullptr) {
    erpp_encoder_predict_thread->wait_until_idle();
  }
  LOG(INFO) << "reset_for_generate: predict reset barrier done";
  LOG(INFO) << "reset_for_generate: begin fetch scheduler barrier";
  fetch_schedule_thread->begin_reset_for_generate();
  fetch_schedule_thread->wait_until_idle();
  LOG(INFO) << "reset_for_generate: fetch scheduler barrier done";
  LOG(INFO) << "reset_for_generate: begin fetch worker barrier";
  fetch_thread->wait_until_idle();
  LOG(INFO) << "reset_for_generate: fetch worker barrier done";
  // Wait for FetchDoneTask to consume current_task before ResetTask clears it.
  while (!fetch_schedule_thread->is_idle()) {}
  LOG(INFO) << "reset_for_generate: fetch done barrier done";
  // Wait for compute events to release experts before cache slots are reset/reloaded.
  expert_unlocker_thread->wait_until_idle();
  LOG(INFO) << "reset_for_generate: expert unlocker barrier done";
  // Reset rewrites cache slots for the next sequence; no previous CUDA work may
  // still reference the old slot contents at that point.
  CUDA_CALL(cudaDeviceSynchronize());
  LOG(INFO) << "reset_for_generate: device synchronize done";
  predict_thread->reset_for_generate(generate_epoch);
  LOG(INFO) << "reset_for_generate: predict worker reset done";
  if (erpp_encoder_predict_thread != nullptr) {
    erpp_encoder_predict_thread->reset_for_generate(generate_epoch);
    LOG(INFO) << "reset_for_generate: ERPP encoder predict worker reset done";
  }
  predictor->reset_sequence_state();
  LOG(INFO) << "reset_for_generate: predictor sequence reset done";
  if (erpp_encoder_predictor != nullptr) {
    erpp_encoder_predictor->reset_sequence_state();
    LOG(INFO) << "reset_for_generate: ERPP encoder predictor sequence reset done";
  }
  fetch_schedule_thread->reset_for_generate(
      generate_epoch, forward_epoch, DecoderWarmupAction::kClear);
  LOG(INFO) << "reset_for_generate: scheduler clear reset done";
  {
    NVTX_RANGE("cache_init/reset_cache_contents");
    Timer timer;
    LOG(INFO) << "reset_for_generate: reset cache contents begin";
    cache->reset_cache_contents();
    reset_us = timer.dur_us();
    LOG(INFO) << "reset_for_generate: reset cache contents done";
  }
  {
    NVTX_RANGE("cache_init/load_initial_plan_sync");
    Timer timer;
    LOG(INFO) << "reset_for_generate: load initial plan begin";
    cache->load_initial_plan_sync((cudaStream_t)copy_stream);
    load_us = timer.dur_us();
    LOG(INFO) << "reset_for_generate: load initial plan done";
  }
  {
    NVTX_RANGE("cache_init/scheduler_rebuild_for_generate_start");
    Timer timer;
    LOG(INFO) << "reset_for_generate: scheduler rebuild begin";
    fetch_schedule_thread->reset_for_generate(
        generate_epoch, forward_epoch, DecoderWarmupAction::kRebuildForGenerateStart);
    rebuild_us = timer.dur_us();
    LOG(INFO) << "reset_for_generate: scheduler rebuild done";
  }
  LOG(INFO) << "ttft_breakdown_cache_init_us total=" << total_timer.dur_us()
            << " reset_cache_contents=" << reset_us
            << " load_initial_plan_sync=" << load_us
            << " scheduler_rebuild=" << rebuild_us
            << " generate_epoch=" << generate_epoch
            << " forward_epoch=" << forward_epoch;
  reset_in_progress.store(false, std::memory_order_release);
  reset_finished = true;
}

void PrefetchMngr::preempt_and_launch_one_layer(int layer_idx, int64_t* experts, int64_t num_expert) {
  PreemptTask preempt_task;
  preempt_task.layer_idx = layer_idx;
  preempt_task.forward_epoch = forward_epoch;
  preempt_task.generate_epoch = generate_epoch;
  preempt_task.expert_idxs = experts;
  preempt_task.num_expert = num_expert;
  auto handler = fetch_schedule_thread->add_one_task(&preempt_task);
  fetch_schedule_thread->wait_progress(handler);
  // preempt_one_layer_(layer_idx, experts.data_ptr<int64_t>(), experts.size(0));
}

void PrefetchMngr::ensure_encoder_layer_stats_size() {
  if (encoder_layer_stats.size() != static_cast<size_t>(metas->num_layer)) {
    encoder_layer_stats.assign(metas->num_layer, EncoderLayerStats());
  }
}

bool PrefetchMngr::encoder_layer_expert_entry_hit(int layer_id, int expert_id) const {
  auto expert = model_loader->get_source(layer_id, expert_id);
  auto status = expert->expert_status.get();
  return status == kReady || status == kLaunching;
}

void PrefetchMngr::log_encoder_layer_entry_stats(
    int layer_id,
    int64_t* experts,
    int64_t num_expert) {
  if (!log_encoder_layer_stats_enabled() &&
      !log_erpp_encoder_diagnostics_enabled()) {
    return;
  }
  if (!metas->is_encoder_layer(layer_id)) {
    return;
  }
  ensure_encoder_layer_stats_size();
  auto& stats = encoder_layer_stats[layer_id];
  stats = EncoderLayerStats();
  stats.forward_epoch = forward_epoch;
  stats.generate_epoch = generate_epoch;
  stats.needed = num_expert;
  if (fetch_schedule_thread != nullptr) {
    fetch_schedule_thread->ensure_encoder_prefetch_metrics();
    if (layer_id >= 0 && layer_id < static_cast<int>(fetch_schedule_thread->encoder_prefetch_metrics.size())) {
      const auto& metrics = fetch_schedule_thread->encoder_prefetch_metrics[layer_id];
      stats.occupancy_at_entry = cache->encoder_layer_cache_occupancy(layer_id);
      stats.prefetch_enqueued_experts = metrics.enqueued_experts;
      stats.prefetch_enqueued_chunks = metrics.enqueued_chunks;
      stats.prefetch_dispatched_experts = metrics.dispatched_experts;
      stats.prefetch_dispatched_chunks = metrics.dispatched_chunks;
      stats.prefetch_completed_experts = metrics.completed_experts;
      stats.prefetch_completed_chunks = metrics.completed_chunks;
      stats.prefetch_completed_after_entry_experts = metrics.completed_after_entry_experts;
      stats.prefetch_completed_after_entry_chunks = metrics.completed_after_entry_chunks;
      stats.prefetch_no_victim_blocks = metrics.no_victim_blocks;
      fetch_schedule_thread->encoder_layer_entered_mask[layer_id] = 1;
    }
  }
  std::vector<uint8_t> seen(metas->num_expert, 0);
  for (int64_t i = 0; i < num_expert; i++) {
    int expert_id = int(experts[i]);
    if (expert_id < 0 || expert_id >= metas->num_expert) {
      stats.entry_miss += 1;
      continue;
    }
    const bool entry_hit = encoder_layer_expert_entry_hit(layer_id, expert_id);
    if (entry_hit) {
      stats.entry_hit += 1;
    } else {
      stats.entry_miss += 1;
    }
    if (seen[expert_id]) {
      continue;
    }
    seen[expert_id] = 1;
    if (fetch_schedule_thread != nullptr &&
        layer_id >= 0 &&
        layer_id < static_cast<int>(fetch_schedule_thread->encoder_prefetch_completed_mask.size())) {
      const bool completed = fetch_schedule_thread->encoder_prefetch_completed_mask[layer_id][expert_id] != 0;
      const bool dispatched = fetch_schedule_thread->encoder_prefetch_dispatched_mask[layer_id][expert_id] != 0;
      if (completed && entry_hit) {
        stats.needed_prefetch_completed_at_entry += 1;
      } else if (completed && !entry_hit) {
        stats.needed_prefetch_evicted_at_entry += 1;
      } else if (dispatched && !entry_hit) {
        stats.needed_prefetch_pending_at_entry += 1;
      }
    }
  }
  if (log_encoder_layer_stats_enabled()) {
    LOG(INFO) << "encoder_layer_stats: phase=entry"
              << " L=" << layer_id
              << " needed=" << stats.needed
              << " hit=" << stats.entry_hit
              << " miss=" << stats.entry_miss
              << " entry_hit=" << stats.entry_hit
              << " entry_miss=" << stats.entry_miss
              << " occupancy_at_entry=" << stats.occupancy_at_entry
              << " prefetch_enqueued_experts=" << stats.prefetch_enqueued_experts
              << " prefetch_enqueued_chunks=" << stats.prefetch_enqueued_chunks
              << " prefetch_dispatched_experts=" << stats.prefetch_dispatched_experts
              << " prefetch_dispatched_chunks=" << stats.prefetch_dispatched_chunks
              << " prefetch_completed_experts=" << stats.prefetch_completed_experts
              << " prefetch_completed_chunks=" << stats.prefetch_completed_chunks
              << " needed_prefetch_completed_at_entry=" << stats.needed_prefetch_completed_at_entry
              << " needed_prefetch_pending_at_entry=" << stats.needed_prefetch_pending_at_entry
              << " needed_prefetch_evicted_at_entry=" << stats.needed_prefetch_evicted_at_entry
              << " prefetch_no_victim_blocks=" << stats.prefetch_no_victim_blocks
              << " forward_epoch=" << stats.forward_epoch
              << " generate_epoch=" << stats.generate_epoch;
  }
}

void PrefetchMngr::log_encoder_layer_use_stats(
    int layer_id,
    int expert_id,
    bool hit,
    bool waited,
    uint64_t wait_us,
    int status_before) {
  if (!log_encoder_layer_stats_enabled() &&
      !log_erpp_encoder_diagnostics_enabled()) {
    return;
  }
  if (!metas->is_encoder_layer(layer_id)) {
    return;
  }
  ensure_encoder_layer_stats_size();
  auto& stats = encoder_layer_stats[layer_id];
  if (stats.forward_epoch != forward_epoch || stats.generate_epoch != generate_epoch) {
    stats = EncoderLayerStats();
    stats.forward_epoch = forward_epoch;
    stats.generate_epoch = generate_epoch;
  }
  bool completed_by_prefetch = false;
  bool dispatched_by_prefetch = false;
  if (fetch_schedule_thread != nullptr &&
      layer_id >= 0 && layer_id < metas->num_encoder_moe_layer &&
      expert_id >= 0 && expert_id < metas->num_expert) {
    fetch_schedule_thread->ensure_encoder_prefetch_metrics();
    fetch_schedule_thread->encoder_prefetch_used_mask[layer_id][expert_id] = 1;
    completed_by_prefetch = fetch_schedule_thread->encoder_prefetch_completed_mask[layer_id][expert_id] != 0;
    dispatched_by_prefetch = fetch_schedule_thread->encoder_prefetch_dispatched_mask[layer_id][expert_id] != 0;
  }
  if (hit) {
    stats.actual_hit += 1;
    if (completed_by_prefetch) {
      stats.prefetch_completed_before_use += 1;
    }
  } else {
    stats.actual_miss += 1;
    if (completed_by_prefetch) {
      stats.prefetch_evicted_before_use += 1;
    } else if (dispatched_by_prefetch) {
      stats.prefetch_late_on_use += 1;
    }
  }
  if (waited) {
    stats.waited += 1;
    stats.wait_us_total += static_cast<int64_t>(wait_us);
    stats.wait_us_max = std::max<int64_t>(stats.wait_us_max, static_cast<int64_t>(wait_us));
  }
  if (log_encoder_layer_stats_enabled()) {
    LOG(INFO) << "encoder_layer_stats: phase=use"
              << " L=" << layer_id
              << " E=" << expert_id
              << " hit=" << (hit ? 1 : 0)
              << " miss=" << (hit ? 0 : 1)
              << " waited=" << (waited ? 1 : 0)
              << " wait_us=" << wait_us
              << " status_before=" << status_before
              << " actual_hit=" << stats.actual_hit
              << " actual_miss=" << stats.actual_miss
              << " wait_us_total=" << stats.wait_us_total
              << " prefetch_completed_before_use=" << stats.prefetch_completed_before_use
              << " prefetch_late_on_use=" << stats.prefetch_late_on_use
              << " prefetch_evicted_before_use=" << stats.prefetch_evicted_before_use
              << " forward_epoch=" << stats.forward_epoch
              << " generate_epoch=" << stats.generate_epoch;
  }
}

void PrefetchMngr::log_encoder_layer_done_stats(int layer_id) {
  if (!log_encoder_layer_stats_enabled() &&
      !log_erpp_encoder_diagnostics_enabled()) {
    return;
  }
  if (!metas->is_encoder_layer(layer_id)) {
    return;
  }
  ensure_encoder_layer_stats_size();
  auto& stats = encoder_layer_stats[layer_id];
  if (stats.done_logged) {
    return;
  }
  stats.prefetch_unused_completed = 0;
  if (fetch_schedule_thread != nullptr &&
      layer_id >= 0 && layer_id < metas->num_encoder_moe_layer) {
    fetch_schedule_thread->ensure_encoder_prefetch_metrics();
    if (layer_id < static_cast<int>(fetch_schedule_thread->encoder_prefetch_completed_mask.size())) {
      for (int expert_id = 0; expert_id < metas->num_expert; expert_id++) {
        if (fetch_schedule_thread->encoder_prefetch_completed_mask[layer_id][expert_id] &&
            !fetch_schedule_thread->encoder_prefetch_used_mask[layer_id][expert_id]) {
          stats.prefetch_unused_completed += 1;
        }
      }
      const auto& metrics = fetch_schedule_thread->encoder_prefetch_metrics[layer_id];
      stats.prefetch_completed_after_entry_experts = metrics.completed_after_entry_experts;
      stats.prefetch_completed_after_entry_chunks = metrics.completed_after_entry_chunks;
      stats.prefetch_no_victim_blocks = metrics.no_victim_blocks;
    }
  }
  if (fetch_schedule_thread != nullptr) {
    fetch_schedule_thread->update_encoder_jit_wait_penalty(
        layer_id, stats.actual_miss, stats.wait_us_total);
  }
  stats.done_logged = true;
  if (log_encoder_layer_stats_enabled()) {
    LOG(INFO) << "encoder_layer_stats: phase=done"
              << " L=" << layer_id
              << " needed=" << stats.needed
              << " entry_hit=" << stats.entry_hit
              << " entry_miss=" << stats.entry_miss
              << " actual_hit=" << stats.actual_hit
              << " actual_miss=" << stats.actual_miss
              << " waited=" << stats.waited
              << " wait_us_total=" << stats.wait_us_total
              << " wait_us_max=" << stats.wait_us_max
              << " occupancy_at_entry=" << stats.occupancy_at_entry
              << " prefetch_enqueued_experts=" << stats.prefetch_enqueued_experts
              << " prefetch_enqueued_chunks=" << stats.prefetch_enqueued_chunks
              << " prefetch_dispatched_experts=" << stats.prefetch_dispatched_experts
              << " prefetch_dispatched_chunks=" << stats.prefetch_dispatched_chunks
              << " prefetch_completed_experts=" << stats.prefetch_completed_experts
              << " prefetch_completed_chunks=" << stats.prefetch_completed_chunks
              << " prefetch_completed_after_entry_experts=" << stats.prefetch_completed_after_entry_experts
              << " prefetch_completed_after_entry_chunks=" << stats.prefetch_completed_after_entry_chunks
              << " needed_prefetch_completed_at_entry=" << stats.needed_prefetch_completed_at_entry
              << " needed_prefetch_pending_at_entry=" << stats.needed_prefetch_pending_at_entry
              << " needed_prefetch_evicted_at_entry=" << stats.needed_prefetch_evicted_at_entry
              << " prefetch_completed_before_use=" << stats.prefetch_completed_before_use
              << " prefetch_late_on_use=" << stats.prefetch_late_on_use
              << " prefetch_evicted_before_use=" << stats.prefetch_evicted_before_use
              << " prefetch_unused_completed=" << stats.prefetch_unused_completed
              << " prefetch_no_victim_blocks=" << stats.prefetch_no_victim_blocks
              << " forward_epoch=" << stats.forward_epoch
              << " generate_epoch=" << stats.generate_epoch;
  }
}

void PrefetchMngr::report_one_layer(int layer_id, torch::Tensor experts) {
  report_one_layer(layer_id, experts.data_ptr<int64_t>(), experts.numel());
}
void PrefetchMngr::report_one_layer(int layer_id, int64_t* experts, int64_t num_expert) {
  TRACE_EVENT_GURAD(kHook, "report_one_layer");
  NVTX_RANGE("hook/report_one_layer L" + std::to_string(layer_id) + " N" + std::to_string(num_expert));
  NVTX_DETAIL_MARK("demand/layer_experts L" + std::to_string(layer_id) +
                   " experts=[" + array_to_str(experts, num_expert) + "]");
  cache_stats->forward();
  const int previous_layer = layer_id - 1;
  if (previous_layer >= 0 && metas->is_encoder_layer(previous_layer)) {
    log_encoder_layer_done_stats(previous_layer);
  }
  log_encoder_layer_entry_stats(layer_id, experts, num_expert);
  const bool should_wait_decoder_prefetch =
      metas->is_decoder_layer(layer_id) &&
      metas->num_predict_expert_per_layer > 0 &&
      metas->predict_input_mode != kNoPredict;
  if (should_wait_decoder_prefetch) {
    LOG(INFO) << "prefetcher: consume prefetch layer progress at layer " << layer_id;
    int progress_idx = predict_thread->consume_prefetch_layer_progress();
    LOG(INFO) << "prefetcher: consume prefetch layer progress at layer " << layer_id << " done " << progress_idx;
  }

  if (metas->is_encoder_layer(layer_id)) {
    std::vector<uint8_t> needed_mask(metas->num_expert, 0);
    for (int64_t i = 0; i < num_expert; i++) {
      const int expert_idx = int(experts[i]);
      if (expert_idx >= 0 && expert_idx < metas->num_expert) {
        needed_mask[expert_idx] = 1;
      }
    }
    fetch_schedule_thread->enqueue_layer_reclaimable_except(layer_id, needed_mask);
  }
  preempt_and_launch_one_layer(layer_id, experts, num_expert); // handle reorder, launch precise task, clear prefetch queue
  profiler->add(TimeProfiler::kCntActivatedExpert, num_expert);
  precision_profiler->record_activated_experts(layer_id, experts, num_expert);
  record_then_predict_and_prefetch(layer_id, experts, num_expert);
}
void PrefetchMngr::one_moe_layer_done(int layer_id) {
  TRACE_EVENT_GURAD(kHook, "one_moe_layer_done");
  NVTX_RANGE("hook/one_moe_layer_done L" + std::to_string(layer_id));
  if (metas->is_encoder_layer(layer_id)) {
    log_encoder_layer_done_stats(layer_id);
    fetch_schedule_thread->enqueue_layer_reclaimable(layer_id);
  }
  if (metas->early_preempt == false) {
    LOG(INFO) << "prefetcher: one moe layer done, add prefetch layer budget : " << layer_id;
    predict_thread->add_prefetch_layer_budget();
  }
  if (layer_id == metas->num_layer - 1) {
    profiler->push(TimeProfiler::kCntActivatedExpert, 0);
    profiler->push(TimeProfiler::kHitCnt, 0);
    profiler->push(TimeProfiler::kMissCnt, 0);
    profiler->push(TimeProfiler::kReadyCnt, 0);
    profiler->push(TimeProfiler::kUnreadyCnt, 0);
    profiler->push(TimeProfiler::kPrefetchHitCnt, 0);
    profiler->push(TimeProfiler::kPrefetchMissCnt, 0);
    profiler->push(TimeProfiler::kWaitTime, 0);
  }
  if (layer_id == metas->num_layer - 1) {
    switch (metas->predict_input_mode) {
      case kNoPredict:
      case kOneToken:
      case kDecodeCumsum:
      case kLastUseDistance:
      case kWeighedDecodeCumsum: {
        forward_epoch += 1;
        fetch_schedule_thread->forward_epoch_start_task.forward_epoch = forward_epoch;
        fetch_schedule_thread->forward_epoch_start_task.generate_epoch = generate_epoch;
        fetch_schedule_thread->forward_epoch_start_task.decoder_warmup_action = DecoderWarmupAction::kPreserve;
        auto handler = fetch_schedule_thread->add_one_task(&fetch_schedule_thread->forward_epoch_start_task);
        fetch_schedule_thread->wait_progress(handler);
        break;
      }
      default: { break; }
    }
    predict_thread->on_one_iter_done(forward_epoch, generate_epoch);
    // predict_thread->add_one_task();
  }
}

void PrefetchMngr::report_one_expert(int layer_id, int expert_id) {
  TRACE_EVENT_GURAD(kHook, "report_one_expert");
  NVTX_RANGE("hook/report_one_expert L" + std::to_string(layer_id) + " E" + std::to_string(expert_id));
  NVTX_MARK("demand/need_expert L" + std::to_string(layer_id) + " E" + std::to_string(expert_id));
  auto expert = model_loader->get_source(layer_id, expert_id);
  auto current_status = expert->expert_status.get();
  if (metas->early_preempt == false || current_status != kLaunching) {
    PreemptOneExpertTask task;
    task.layer_id = layer_id;
    task.generate_epoch = generate_epoch;
    task.expert_id = expert_id;
    auto handler = fetch_schedule_thread->add_one_task(&task);
    fetch_schedule_thread->wait_progress(handler);
  }
  this->wait_expert(layer_id, expert_id);
}
void PrefetchMngr::one_expert_done(int layer_id, int expert_id) {
  TRACE_EVENT_GURAD(kHook, "one_expert_done");
  NVTX_RANGE("hook/one_expert_done L" + std::to_string(layer_id) + " E" + std::to_string(expert_id));
  mark_expert_using(layer_id, expert_id);
  if (metas->is_encoder_layer(layer_id)) {
    if (log_prefetch_decision_enabled()) {
      LOG(INFO) << "erpp_encoder_prefetch: encoder expert reclaimable L"
                << layer_id << " E" << expert_id;
    }
    fetch_schedule_thread->enqueue_expert_reclaimable(layer_id, expert_id);
  }
}

void PrefetchMngr::wait_expert(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  TRACE_EVENT_GURAD(kHook, "wait:" + expert->toString());
  LOG(TRACE) << "waiting expert " << expert->toString();
  // model_loader->get_source(layer_id, expert_id)->expert_status.wait(kReady, kLaunching);
  auto current_status = expert->expert_status.get();;
  if (current_status == kLaunching) {
    cache_stats->hit();
    profiler->add(TimeProfiler::kReadyCnt, 1);
    log_encoder_layer_use_stats(layer_id, expert_id, true, false, 0, int(current_status));
  } else {
    NVTX_RANGE("hook/wait_expert L" + std::to_string(layer_id) + " E" + std::to_string(expert_id));
    Timer timer;
    cache_stats->miss();
    profiler->add(TimeProfiler::kUnreadyCnt, 1);
    // todo: add timing of waiting expert ready
    expert->expert_status.wait(kLaunching);
    auto dur = timer.dur_us();
    profiler->add(TimeProfiler::kWaitTime, dur);
    log_encoder_layer_use_stats(layer_id, expert_id, false, true, dur, int(current_status));
  }
  LOG(TRACE) << "waiting expert " << expert->toString() << " success";
}
void PrefetchMngr::mark_expert_using(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  CUDA_CALL(cudaEventRecord(expert->event, (cudaStream_t)(this->compute_stream)));
  expert->expert_status.transfer(kLaunching, kUsing);

  expert_unlocker_thread->add_one_task(expert);
}

void PrefetchMngr::report_erpp_encoder_layer0(torch::Tensor hidden, torch::Tensor attention_mask) {
  if (!metas->enable_erpp_encoder_prefetch ||
      erpp_encoder_predictor == nullptr ||
      erpp_encoder_predict_thread == nullptr) {
    if (log_prefetch_decision_enabled()) {
      LOG(INFO) << "erpp_encoder_prefetch: skip report"
                << " enabled=" << metas->enable_erpp_encoder_prefetch
                << " predictor_null=" << (erpp_encoder_predictor == nullptr)
                << " worker_null=" << (erpp_encoder_predict_thread == nullptr);
    }
    return;
  }
  const int64_t report_forward_epoch = forward_epoch;
  const int64_t report_generate_epoch = generate_epoch;
  erpp_encoder_predictor->record_encoder_layer0(
      hidden,
      attention_mask,
      (cudaStream_t)compute_stream,
      report_forward_epoch,
      report_generate_epoch);
  if (log_prefetch_decision_enabled()) {
    LOG(INFO) << "erpp_encoder_prefetch: report recorded"
              << " forward_epoch=" << report_forward_epoch
              << " generate_epoch=" << report_generate_epoch;
  }
  erpp_encoder_predict_thread->on_encoder_layer0_recorded(
      report_forward_epoch, report_generate_epoch);
}

void PrefetchMngr::launch_thread() {
  this->reload_env();
  predict_thread->on_one_iter_done(forward_epoch, generate_epoch);

  fetch_schedule_thread->launch();
  predict_thread->launch();
  if (erpp_encoder_predict_thread != nullptr) {
    erpp_encoder_predict_thread->launch();
  }
  expert_unlocker_thread->launch();
  fetch_thread->launch();
  if (string_is_on(GetEnv("SPARSE_CACHE_THREAD_TO_E_CORE"))) {
    LOG(INFO) << "set cpu affinity";
    fetch_schedule_thread->set_cpu_affinity({30});
    predict_thread->set_cpu_affinity({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    expert_unlocker_thread->set_cpu_affinity({28});
    fetch_thread->set_cpu_affinity({26});
  }
}
PrefetchMngr::PrefetchMngr(std::shared_ptr<ModuleMeta> metas,
                           std::shared_ptr<ModelLoader> model_loader,
                           std::shared_ptr<PredictorBase> predictor,
                           int64_t compute_stream_param,
                           bool create_compute_stream,
                           TimeProfiler* profiler_ptr)
    : metas(metas), model_loader(model_loader), predictor(predictor) {
  this->cache = std::make_shared<CacheMngr>(metas, model_loader);
  predict_thread = std::make_shared<PredictWorker>();
  expert_unlocker_thread = std::make_shared<ExpertUnlockWorker>();
  fetch_thread = std::make_shared<FetchWorker>();
  fetch_schedule_thread = std::make_shared<FetchScheduleWorker>();
  if (metas->enable_erpp_encoder_prefetch) {
    erpp_encoder_predictor = std::make_shared<ErppEncoderPredictor>(metas.get());
    erpp_encoder_predictor->load_model_from(metas->erpp_encoder_model_path);
    erpp_encoder_predict_thread = std::make_shared<ErppEncoderPredictWorker>();
  }
  cache_stats = std::make_shared<CacheStatistics>();
  // cache_stats->add_reporter([this, metas = this->metas](CacheStatistics* stats){
  //   auto tensor = stats->to_tensor();
  //   // remove iteration of prefill
  //   tensor = tensor.index({tensor.sum(1) <= metas->num_expert_per_token});
  //   // skip first 10 iteration
  //   tensor = tensor.index({torch::indexing::Slice(metas->num_layer * 10)});
  //   tensor = tensor.mean(0);
  //   std::cout << "legacy_decode_stage_hit_cnt:"  << tensor[0].item<float>() << std::endl;
  //   std::cout << "legacy_decode_stage_miss_cnt:" << tensor[1].item<float>() << std::endl;
  //   std::cout << "legacy_decode_stage_hit_rate:" << tensor[0].item<float>() / (tensor[0].item<float>() + tensor[1].item<float>()) << std::endl;
  // });
  // cache_stats->add_reporter([this, metas = this->metas](CacheStatistics* stats){
  //   auto tensor = stats->to_tensor();
  //   // remove iteration of decode
  //   tensor = tensor.index({tensor.sum(1) > metas->num_expert_per_token});
  //   // skip first 10 iteration
  //   tensor = tensor.index({torch::indexing::Slice(metas->num_layer * 2)});
  //   tensor = tensor.mean(0);
  //   std::cout << "legacy_prefill_stage_hit_cnt:"  << tensor[0].item<float>() << std::endl;
  //   std::cout << "legacy_prefill_stage_miss_cnt:" << tensor[1].item<float>() << std::endl;
  //   std::cout << "legacy_prefill_stage_hit_rate:" << tensor[0].item<float>() / (tensor[0].item<float>() + tensor[1].item<float>()) << std::endl;
  // });
  if (profiler_ptr == nullptr) {
    profiler = std::make_shared<TimeProfiler>();
  } else {
    profiler = profiler_ptr->shared_from_this();
  }
  // profiler = std::make_shared<TimeProfiler>();
  profiler->add_reporter([this, metas = this->metas](TimeProfiler *p){
    auto num_used_expert_tensor = p->to_tensor(TimeProfiler::kCntActivatedExpert);
    // auto idx_is_prefill = num_used_expert_tensor >  (metas->num_expert_per_token * metas->num_layer);
    // auto idx_is_decode  = num_used_expert_tensor <= (metas->num_expert_per_token * metas->num_layer);
    auto seq_len_tensor = p->to_tensor(TimeProfiler::kSeqLen);
    if (seq_len_tensor.numel() == 0) {
      return;
    }
    auto idx_is_prefill = seq_len_tensor > 1;
    auto idx_is_decode  = seq_len_tensor <= 1;
    auto prefill_idxs = torch::nonzero(idx_is_prefill).flatten();
    int num_first_prompts_to_skip = 0;
    if (prefill_idxs.numel() >= 3 &&
        (prefill_idxs[1].item<int>() == 1) && (prefill_idxs[2].item<int>() == 2)) {
      // we are in llama.cpp, where it includes a system prompt, warm with <bos><eos>, and 3 warm requests
      num_first_prompts_to_skip = 5;
    } else {
      // we are in transformers, it has 4 warm up requests
      num_first_prompts_to_skip = 4;
    }
    int starting_point = 0;
    if (prefill_idxs.size(0) <= num_first_prompts_to_skip) {
      starting_point = idx_is_prefill.size(0);
    } else {
      starting_point = prefill_idxs[num_first_prompts_to_skip].item<int>();
    }
    idx_is_prefill.index_put_({torch::indexing::Slice(0, starting_point)}, false);
    idx_is_decode.index_put_({torch::indexing::Slice(0, starting_point)}, false);
    auto smart_slice = [](torch::Tensor tensor, int skip_first) {
      if (skip_first > tensor.size(0)) {
        return tensor.index({torch::indexing::Slice(tensor.size(0))});
      } else {
        return tensor.index({torch::indexing::Slice(skip_first)});
      }
    };
    auto lambda_report_one_pair([this, p](
        TimeProfiler::TimeType on,
        TimeProfiler::TimeType off,
        torch::Tensor idx,
        std::string on_name,
        std::string off_name,
        std::string rate_name) {
      auto on_val  = p->to_tensor(on  ).index({idx}).mean(torch::kFloat32).item<float>();
      auto off_val = p->to_tensor(off ).index({idx}).mean(torch::kFloat32).item<float>();
      std::cout << on_name   << ":" << on_val  << std::endl;
      std::cout << off_name  << ":" << off_val << std::endl;
      std::cout << rate_name << ":" << on_val / (on_val + off_val) << std::endl;
    });

    lambda_report_one_pair(TimeProfiler::kReadyCnt, TimeProfiler::kUnreadyCnt, idx_is_decode,   "decode_stage_ready_cnt",  "decode_stage_unready_cnt",  "decode_stage_ready_rate");
    lambda_report_one_pair(TimeProfiler::kReadyCnt, TimeProfiler::kUnreadyCnt, idx_is_prefill, "prefill_stage_ready_cnt", "prefill_stage_unready_cnt", "prefill_stage_ready_rate");
    lambda_report_one_pair(TimeProfiler::kHitCnt, TimeProfiler::kMissCnt, idx_is_decode,   "decode_stage_hit_cnt",  "decode_stage_miss_cnt",  "decode_stage_hit_rate");
    lambda_report_one_pair(TimeProfiler::kHitCnt, TimeProfiler::kMissCnt, idx_is_prefill, "prefill_stage_hit_cnt", "prefill_stage_miss_cnt", "prefill_stage_hit_rate");
    lambda_report_one_pair(TimeProfiler::kPrefetchHitCnt, TimeProfiler::kPrefetchMissCnt, idx_is_decode,   "decode_stage_prefetch_hit_cnt",  "decode_stage_prefetch_miss_cnt",  "decode_stage_prefetch_hit_rate");
    lambda_report_one_pair(TimeProfiler::kPrefetchHitCnt, TimeProfiler::kPrefetchMissCnt, idx_is_prefill, "prefill_stage_prefetch_hit_cnt", "prefill_stage_prefetch_miss_cnt", "prefill_stage_prefetch_hit_rate");

    {
      auto time = p->to_tensor(TimeProfiler::kWaitTime).index({idx_is_decode});
      std::cout << "decode_stage_wait_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
    {
      auto time = p->to_tensor(TimeProfiler::kWaitTime).index({idx_is_prefill});
      std::cout << "prefill_stage_wait_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
    {
      auto time = p->to_tensor(TimeProfiler::kModelForward).index({idx_is_decode});
      std::cout << "decode_stage_forward_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
    {
      auto time = p->to_tensor(TimeProfiler::kModelForward).index({idx_is_prefill});
      std::cout << "prefill_stage_forward_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
    {
      std::cout << "num_prefill_iter:" << torch::nonzero(idx_is_prefill).squeeze().numel() << std::endl;
      std::cout << "num_decode_iter:" << torch::nonzero(idx_is_decode).squeeze().numel() << std::endl;
    }
    {
      auto time = smart_slice(p->to_tensor(TimeProfiler::kPredictTime), 10); // skip first 10 and last 1iteration
      std::cout << "predict_time:" << time.mean(torch::kFloat32).item().to<double>() / 1000.0 << std::endl;
    }
  });
  precision_profiler = std::make_shared<PrecisionProfiler>();
  precision_profiler->decode_expert_per_token = metas->num_expert_per_token;

  if (create_compute_stream) {
    CUDA_CALL(cudaStreamCreateWithFlags((cudaStream_t*)(&compute_stream), cudaStreamNonBlocking));
    {
      // set torch compute stream
      at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal((cudaStream_t)compute_stream, model_loader->mem_mngr_ctx->device_id));
      auto blas_handle = at::cuda::getCurrentCUDABlasHandle();
      cublasStatus_t ret = cublasSetStream(blas_handle, at::cuda::getCurrentCUDAStream());
      CHECK(ret == CUBLAS_STATUS_SUCCESS);
    }
    this->set_compute_stream(compute_stream);
  } else {
    this->set_compute_stream(compute_stream_param);
  }

  if (metas->cache_only) {
    copy_stream = compute_stream;
  } else {
    CUDA_CALL(cudaStreamCreateWithFlags((cudaStream_t*)(&copy_stream),    cudaStreamNonBlocking));
  }

  predict_thread->init(fetch_schedule_thread.get(), predictor.get(), cache.get(), metas.get());
  predict_thread->precision_profiler = precision_profiler.get();
  fetch_thread->init(metas.get(), fetch_schedule_thread.get(), model_loader->mem_mngr_ctx.get(), (cudaStream_t)copy_stream);
  fetch_schedule_thread->init(metas.get(), model_loader.get(), this->cache.get(), fetch_thread.get(), predict_thread.get(), cache_stats.get(), profiler.get());

  if (erpp_encoder_predict_thread != nullptr) {
    erpp_encoder_predict_thread->init(
        fetch_schedule_thread.get(),
        erpp_encoder_predictor.get(),
        metas.get());
  }

  predictor->profiler = profiler;
}

void PrefetchMngr::set_compute_stream(int64_t stream) {
  compute_stream = stream;
  predictor->compute_stream = (cudaStream_t)stream;
}

void PrefetchMngr::report_moe_attn_logits(int layer_id, torch::Tensor attn_logits) {
  NVTX_RANGE("hook/report_moe_attn_logits L" + std::to_string(layer_id));
  if (layer_id == metas->first_decoder_layer() &&
      (metas->predict_input_mode == kFirstMoeAttnInputLogits ||
       metas->predict_input_mode == kMoeAttnInputLogits)) {
    forward_epoch += 1;
    fetch_schedule_thread->forward_epoch_start_task.forward_epoch = forward_epoch;
    fetch_schedule_thread->forward_epoch_start_task.generate_epoch = generate_epoch;
    fetch_schedule_thread->forward_epoch_start_task.decoder_warmup_action = DecoderWarmupAction::kPreserve;
    auto handler = fetch_schedule_thread->add_one_task(&fetch_schedule_thread->forward_epoch_start_task);
    fetch_schedule_thread->wait_progress(handler);
  }
  LOG_BLOCK(DEBUG, logger, {
    logger << "prefetch mngr, report_moe_attn_logits " << layer_id << ", " << attn_logits.sizes();
  });
  predictor->record_moe_attn_logits(layer_id, attn_logits);
  predict_thread->on_moe_attn_input_logits_recorded(layer_id, forward_epoch, generate_epoch);
}

void PrefetchMngr::report_moe_layer_logits(int layer_id, torch::Tensor layer_logits) {
  NVTX_RANGE("hook/report_moe_layer_logits L" + std::to_string(layer_id));
  int64_t predict_forward_epoch = forward_epoch;
  if (layer_id == metas->first_decoder_layer() && metas->predict_input_mode == kMoeLayerLogits) {
    forward_epoch += 1;
    predict_forward_epoch = forward_epoch;
    fetch_schedule_thread->forward_epoch_start_task.forward_epoch = forward_epoch;
    fetch_schedule_thread->forward_epoch_start_task.generate_epoch = generate_epoch;
    fetch_schedule_thread->forward_epoch_start_task.decoder_warmup_action = DecoderWarmupAction::kPreserve;
    auto handler = fetch_schedule_thread->add_one_task(&fetch_schedule_thread->forward_epoch_start_task);
    fetch_schedule_thread->wait_progress(handler);
  } else if (layer_id == metas->num_layer && metas->predict_input_mode == kMoeLayerLogits) {
    // we are at the last layer, so we need to advance the prediction epoch
    predict_forward_epoch = forward_epoch + 1;
  }
  LOG_BLOCK(INFO, logger, {
    logger << "prefetch mngr, report_moe_layer_logits " << layer_id << ", " << layer_logits.sizes();
  });
  predictor->record_moe_layer_logits(layer_id, layer_logits);
  predict_thread->on_moe_layer_logits_recorded(layer_id, predict_forward_epoch, generate_epoch);
  if (layer_id == metas->first_decoder_layer()) {
    auto seq_len = layer_logits.size(1);
    profiler->push(TimeProfiler::kSeqLen, seq_len);
  }
  if (metas->sleep_on_report_logits_us) {
    cuda_sleep(metas->sleep_on_report_logits_us, compute_stream);
  }
}

void PrefetchMngr::record_then_predict_and_prefetch(int layer_id, int64_t* experts, int64_t num_expert) {
  TRACE_EVENT_GURAD(kHook, "record_then_predict_and_launch");
  // LOG_BLOCK(DEBUG, logger, {
  //   logger << "actual " << layer_id << ":" << tensor_to_str(experts);
  // });
  if (num_expert <= metas->num_expert_per_token) {
    predictor->add_one_layer(layer_id, experts, num_expert);
  } else {
    LOG(TRACE) << "identified prefill iteration, skip adding it to prefill " << num_expert;
    // predictor->clear_access_buffer();
    if (layer_id == 0) {
      predictor->start_of_new_sequence();
    }
  }
  // if (layer_id == metas->num_layer - 1) {
  //   predict_thread->add_one_task();
  // }
}
PrefetchMngr::~PrefetchMngr() {
  // predict_thread->add_one_task(PredictJob());
  predict_thread->add_prefetch_layer_budget();
  if (erpp_encoder_predict_thread != nullptr) {
    erpp_encoder_predict_thread->begin_reset_for_generate();
    erpp_encoder_predict_thread->wait_until_idle();
    erpp_encoder_predict_thread->exit();
  }
  fetch_thread->exit();
  predict_thread->exit();
  expert_unlocker_thread->exit();
  fetch_schedule_thread->exit();
  model_loader->release_logical_expert_param_refs();
  profiler->clear_reporters();
  profiler.reset();
  if (TraceEventCollector::globally_enabled) {
    LOG(WARNING) << "dumping trace event to trace.json";
    std::ofstream f("trace.json", std::ios::out | std::ios::trunc);
    f << TraceEventCollector::singleton().dump_json_to_string();
    f.close();
  }
}
void FetchScheduleWorker::do_one_task_impl(FetchScheduleTaskBase *task) {
  switch (task->task_type) {
    case FetchScheduleTaskBase::kPreempt: {
      do_one_task_impl(dynamic_cast<PreemptTask *>(task));
      break;
    }
    case FetchScheduleTaskBase::kForwardEpochStart: {
      do_one_task_impl(dynamic_cast<ForwardEpochStartTask *>(task));
      break;
    }
    case FetchScheduleTaskBase::kFetchDone: {
      do_one_task_impl(dynamic_cast<FetchDoneTask*>(task));
      break;
    }
    case FetchScheduleTaskBase::kIdle: {
      do_one_task_impl(dynamic_cast<IdleTask*>(task));
      break;
    }
    case FetchScheduleTaskBase::kPrefetchLayer: {
      do_one_task_impl(dynamic_cast<PrefetchLayerTask*>(task));
      break;
    }
    case FetchScheduleTaskBase::kPreemptOneExpert: {
      do_one_task_impl(dynamic_cast<PreemptOneExpertTask*>(task));
      break;
    }
    case FetchScheduleTaskBase::kReset: {
      do_one_task_impl(dynamic_cast<ResetTask*>(task));
      break;
    }
    case FetchScheduleTaskBase::kErppEncoderJitRankings: {
      do_one_task_impl(dynamic_cast<ErppEncoderJitRankingsTask*>(task));
      break;
    }
    default: {
      CHECK(false) << "unknown task type " << task->task_type;
    }
  }
}
void FetchScheduleWorker::do_one_task_impl(PreemptTask *task) {
  CHECK(!(task->generate_epoch != current_generate_epoch))
      << "stale generate task: task_generate_epoch=" << task->generate_epoch
      << ", current_generate_epoch=" << current_generate_epoch;
  TRACE_EVENT_GURAD(kFetchScheduler, "do preempt");
  NVTX_RANGE("sched/preempt_layer L" + std::to_string(task->layer_idx) + " N" + std::to_string(task->num_expert));
  advance_actual_layer(task->forward_epoch, task->layer_idx);
  log_encoder_jit_layer_entry_diagnostics(task->layer_idx, task->expert_idxs, task->num_expert);
  if (metas->reorder_experts) {
    this->reorder_experts(task->layer_idx, task->expert_idxs, task->num_expert);
  }
  if (metas->early_preempt) {
    this->preempt_one_layer_without_reorder_(task->layer_idx, task->expert_idxs, task->num_expert);
    predict_thread->add_prefetch_layer_budget();
  }

  clear_prefetch_class_for_layer(PrefetchClass::kDecoderPredictor, task->forward_epoch, task->layer_idx);
  drain_reclaimable_updates();
  maybe_enqueue_encoder_jit_refill();
}
void FetchScheduleWorker::do_one_task_impl(ForwardEpochStartTask *task) {
  CHECK(task == &this->forward_epoch_start_task);
  current_generate_epoch = task->generate_epoch;
  start_forward_epoch(task->forward_epoch, task->decoder_warmup_action);
}
void FetchScheduleWorker::do_one_task_impl(ResetTask *task) {
  CHECK(task == &this->reset_task);
  reset_requested.store(true, std::memory_order_release);
  current_task.expert = nullptr;
  clear_all_job_queues();
  clear_decoder_warmup_plan_queue();
  clear_encoder_jit_state();
  clear_encoder_prefetch_metrics();
  reset_pending_reclaimable_updates();
  current_layer = -1;
  current_forward_epoch = task->next_forward_epoch;
  current_generate_epoch = task->next_generate_epoch;
  set_phase(kEncoderPhase);
  if (task->action == DecoderWarmupAction::kRebuildForGenerateStart) {
    rebuild_decoder_warmup_queue();
    reset_requested.store(false, std::memory_order_release);
    add_one_task(&idle_task);
  }
}
void FetchScheduleWorker::do_one_task_impl(PreemptOneExpertTask *task) {
  CHECK(!(task->generate_epoch != current_generate_epoch))
      << "stale generate task: task_generate_epoch=" << task->generate_epoch
      << ", current_generate_epoch=" << current_generate_epoch;
  TRACE_EVENT_GURAD(kFetchScheduler, "do preempt one expert");
  NVTX_RANGE("sched/preempt_one L" + std::to_string(task->layer_id) + " E" + std::to_string(task->expert_id));
  this->preempt_one_expert(task->layer_id, task->expert_id);
  drain_reclaimable_updates();
}

void FetchScheduleWorker::init(ModuleMeta *metas, ModelLoader *model_loader,
                               CacheMngr *cache, FetchWorker *fetch_thread,
                               PredictWorker *predict_thread, CacheStatistics *cache_stats, TimeProfiler* profiler) {
  this->metas = metas;
  this->model_loader = model_loader;
  this->cache = cache;
  this->fetch_thread = fetch_thread;
  this->predict_thread = predict_thread;
  this->cache_stats = cache_stats;
  this->profiler = profiler;
  prefetch_queues.encoder_predictor_by_layer.resize(metas->num_layer);
  prefetch_queues.decoder_predictor_by_layer.resize(metas->num_layer);
  initialize_encoder_jit_enabled_layer_mask();
  clear_encoder_prefetch_metrics();
  pending_reclaimable_updates.assign(metas->num_layer, PendingReclaimableUpdate());
  pending_reclaimable_layers.clear();
  pending_reclaimable_layer_mask.assign(metas->num_layer, 0);
  has_pending_reclaimable_updates.store(false, std::memory_order_release);
  this->add_one_task(&this->idle_task);
}
void FetchScheduleWorker::do_one_task_impl(FetchDoneTask *_) {
  CHECK(!(current_task.generate_epoch != current_generate_epoch))
      << "stale generate task: task_generate_epoch=" << current_task.generate_epoch
      << ", current_generate_epoch=" << current_generate_epoch;
  TRACE_EVENT_GURAD(kFetchScheduler, "one fetch done " + current_task.toString());
  LOG(TRACE) << "scheduler: received one fetch job done " << current_task.toString();
  current_task.expert->num_ready = current_task.stop_mem_buf_idx;
  if (!current_task.is_precise && is_encoder_prefetch_request(current_task.request_type)) {
    mark_encoder_prefetch_completed(
        current_task.expert->layer_idx,
        current_task.expert->expert_idx,
        current_task.stop_mem_buf_idx - current_task.start_mem_buf_idx,
        current_task.stop_mem_buf_idx == metas->num_per_expert_param);
  }
  if (current_task.stop_mem_buf_idx == metas->num_per_expert_param) {
    LOG(TRACE) << "scheduler: all fetch job done for expert " << current_task.toString();

    if (current_task.is_precise && metas->reorder_experts == false) {
      // cache_hit(current_task.expert, current_task.is_precise);
    }
    current_task.expert->expert_status.transfer(kFetching, current_task.is_precise ? kLaunching : kReady);
  }
  current_task.expert = nullptr;
  this->add_one_task(&this->idle_task);
}
void FetchScheduleWorker::do_one_task_impl(IdleTask *idle_task) {
  CHECK(idle_task == &this->idle_task);
  if (reset_requested.load(std::memory_order_acquire)) {
    return;
  }
  if (current_task.expert != nullptr) {
    this->add_one_task(&this->idle_task);
    return;
  }
  bool found = false, sent = false;
  pop_next_task(current_task, found);
  if (found) {
    sent = send_one_job(&current_task);
  }
  if (!found || !sent) {
    // Re-add the idle task. ERPP encoder prefetch can be valid but blocked until
    // reclaimable experts appear; keep the original scheduler polling behavior.
    current_task.expert = nullptr;
    this->add_one_task(&this->idle_task);
  }
}

void FetchScheduleWorker::do_one_task_impl(PrefetchLayerTask *task) {
  CHECK(!(task->generate_epoch != current_generate_epoch))
      << "stale generate task: task_generate_epoch=" << task->generate_epoch
      << ", current_generate_epoch=" << current_generate_epoch;
  TRACE_EVENT_GURAD(kFetchScheduler, "add task for layer " + std::to_string(task->layer_idx) + "[" + array_to_str(task->expert_idxs, task->num_expert) + "]");
  NVTX_RANGE("sched/prefetch_layer L" + std::to_string(task->layer_idx) +
             " N" + std::to_string(task->num_expert) +
             " forward_epoch=" + std::to_string(task->forward_epoch));
  LOG(TRACE) << "scheduler: do PrefetchLayerTask, add prefetch task for layer " << task->layer_idx;
  if (is_stale_prefetch(task->forward_epoch, task->layer_idx)) {
    LOG(TRACE) << "scheduler: drop stale prefetch layer " << task->layer_idx
               << ", task forward_epoch " << task->forward_epoch
               << ", current forward_epoch " << current_forward_epoch
               << ", current layer " << current_layer;
    return;
  }

  PrefetchClass cls = prefetch_class_for_request(task->request_type);
  TaskQueue* target_queue = queue_for_class_and_layer(cls, task->layer_idx);
  if (replace_same_layer_on_enqueue(cls)) {
    clear_prefetch_class_for_layer(cls, task->forward_epoch, task->layer_idx);
  }
  if (cls == PrefetchClass::kEncoderPredictor) {
    if (log_prefetch_decision_enabled()) {
      LOG(INFO) << "prefetch_decision: enqueue ERPP encoder prefetch L"
                << task->layer_idx
                << " num_expert=" << task->num_expert
                << " experts=[" << array_to_str(task->expert_idxs, task->num_expert) << "]"
                << " forward_epoch=" << task->forward_epoch
                << " generate_epoch=" << task->generate_epoch;
    }
    NVTX_DETAIL_MARK(std::string("sched/enqueue_encoder_predictor_prefetch") + " L" +
                     std::to_string(task->layer_idx) +
                     " N=" + std::to_string(task->num_expert) +
                     " forward_epoch=" + std::to_string(task->forward_epoch));
  } else if (cls == PrefetchClass::kDecoderPredictor) {
    NVTX_DETAIL_MARK(std::string("sched/enqueue_decoder_predictor_prefetch") + " L" +
                     std::to_string(task->layer_idx) +
                     " N=" + std::to_string(task->num_expert) +
                     " forward_epoch=" + std::to_string(task->forward_epoch));
  } else if (cls == PrefetchClass::kDecoderWarmup) {
    NVTX_DETAIL_MARK(std::string("sched/enqueue_decoder_warmup_prefetch") + " L" +
                     std::to_string(task->layer_idx) +
                     " N=" + std::to_string(task->num_expert) +
                     " forward_epoch=" + std::to_string(task->forward_epoch));
  }
  // ready, partial, miss
  for (int i = 0; i < task->num_expert; i++) {
    auto expert = model_loader->get_source(task->layer_idx, task->expert_idxs[i]);
    LOG(TRACE) << "scheduler: do PrefetchLayerTask, adding prefetch task " << expert->toString();
    if (metas->promote_hit_in_prefetch && cache->is_in_cache(expert)) { cache_hit(expert, false); }
    // if (expert->num_ready == metas->num_per_expert_param) {
    //   auto cur_status = expert->expert_status.get();
    //   CHECK(cur_status == kReady || cur_status == kUsing) << "expert " << expert->toString() << " must be ready, but is " << cur_status;
    //   LOG(TRACE) << "skip add prefetch task " << expert->toString();
    //   continue;
    // }
    // if (cur_status == kReady || cur_status == kUsing) {
    //   LOG(TRACE) << "skip add prefetch task " << expert->toString();
    //   continue;
    // }
    if (metas->chunk_prefetch) {
      add_separate_tasks_for_one_expert(task->layer_idx, task->expert_idxs[i], target_queue, 0, metas->num_per_expert_param, false, task->forward_epoch, task->request_type);
    } else {
      add_single_tasks_for_one_expert(task->layer_idx, task->expert_idxs[i], target_queue, 0, metas->num_per_expert_param, false, task->forward_epoch, task->request_type);
    }
  }
}

bool FetchScheduleWorker::send_one_job(CopyTask *task) {
  CHECK(!(task->generate_epoch != current_generate_epoch))
      << "stale generate task: task_generate_epoch=" << task->generate_epoch
      << ", current_generate_epoch=" << current_generate_epoch;
  TRACE_EVENT_GURAD(kFetchScheduler, "send:" + task->toString());
  NVTX_RANGE("sched/send L" + std::to_string(task->expert->layer_idx) +
             " E" + std::to_string(task->expert->expert_idx) +
             " P" + std::to_string(task->start_mem_buf_idx) +
             "-" + std::to_string(task->stop_mem_buf_idx) +
             (task->is_precise ? " precise" : " prefetch"));
  std::string request_label = "decoder_predictor_prefetch";
  if (task->request_type == kCacheRequestEncoderPredictorPrefetch) {
    request_label = "encoder_predictor_prefetch";
  } else if (task->request_type == kCacheRequestEncoderJitRefill) {
    request_label = "encoder_jit_refill";
  } else if (task->request_type == kCacheRequestDecoderWarmupPrefetch) {
    request_label = "decoder_warmup_prefetch";
  } else if (task->request_type == kCacheRequestDecoderPredictorPrefetch) {
    request_label = "decoder_predictor_prefetch";
  } else if (task->request_type == kCacheRequestDemand) {
    request_label = "demand";
  }
  NVTX_RANGE(std::string("dispatch/") + request_label + "_io " +
             "L" + std::to_string(task->expert->layer_idx) +
             " E" + std::to_string(task->expert->expert_idx) +
             " P" + std::to_string(task->start_mem_buf_idx) +
             "-" + std::to_string(task->stop_mem_buf_idx) +
             " forward_epoch=" + std::to_string(task->forward_epoch));
  LOG(TRACE) << "scheduler: send one prefetch task " << task->toString();

  if (task->is_precise == false &&
      task->request_type != kCacheRequestDecoderWarmupPrefetch &&
      task->expert != nullptr &&
      is_stale_prefetch(task->forward_epoch, task->expert->layer_idx)) {
    LOG(TRACE) << "scheduler: skip stale prefetch copy " << task->toString()
               << ", current forward_epoch " << current_forward_epoch
               << ", current layer " << current_layer;
    return false;
  }

  // nullptr and 0: first time task
  // nullptr and >0 : a partial task gets evicted
  // not nullptr and 0: duplicated
  // not nullptr and not 0: normal
  CacheMngr::CacheLineOccupancyWaiter lambda_wait = [](){};
  if (task->expert->gpu_data == nullptr) {
    if ((task->request_type == kCacheRequestEncoderPredictorPrefetch ||
         task->request_type == kCacheRequestEncoderJitRefill ||
         task->request_type == kCacheRequestDecoderWarmupPrefetch) &&
        task->start_mem_buf_idx > 0) {
      return false;
    }
    CHECK(task->start_mem_buf_idx == 0);
    CHECK(task->expert->num_ready == 0);
    // a missed task
    LOG(TRACE) << "scheduler: assigning gpu mem for expert " << task->toString();
    lambda_wait = cache_miss(task->expert, task->is_precise, task->request_type);
    if ((task->request_type == kCacheRequestEncoderPredictorPrefetch ||
         task->request_type == kCacheRequestEncoderJitRefill ||
         task->request_type == kCacheRequestDecoderWarmupPrefetch) &&
        task->expert->gpu_data == nullptr) {
      if (log_prefetch_decision_enabled()) {
        LOG(INFO) << "prefetch_decision: skip reclaimable-only request without victim "
                  << request_label << " L" << task->expert->layer_idx
                  << " E" << task->expert->expert_idx
                  << " P" << task->start_mem_buf_idx << "-" << task->stop_mem_buf_idx
                  << " current_layer=" << current_layer
                  << " forward_epoch=" << current_forward_epoch;
      }
      if (is_encoder_prefetch_request(task->request_type)) {
        ensure_encoder_prefetch_metrics();
        if (task->expert->layer_idx >= 0 &&
            task->expert->layer_idx < static_cast<int>(encoder_prefetch_metrics.size())) {
          encoder_prefetch_metrics[task->expert->layer_idx].no_victim_blocks += 1;
        }
      }
      return false;
    }
    task->expert->expert_status.transfer(kIdle, kFetching);
  } else {
    // handle cache_hit calls
    if (task->is_precise) {
      cache_hit(task->expert, task->is_precise);
    } else if (task->start_mem_buf_idx == 0) {
      cache_hit(task->expert, task->is_precise);
    }

    // handle duplicated task
    if (task->expert->num_ready >= task->stop_mem_buf_idx) {
      LOG(TRACE) << "scheduler: a fully duplicated task, skip it: " << task->toString();
      if (task->is_precise) {
        CHECK(task->expert->num_ready == metas->num_per_expert_param);
        task->expert->expert_status.transfer(kReady, kLaunching, false);
      }
      return false;
    } else if (task->expert->num_ready > task->start_mem_buf_idx) {
      LOG(TRACE) << "scheduler: a duplicated task is partially done, skip duplicated part: " << task->toString();
      task->start_mem_buf_idx = task->expert->num_ready;
    } else if (task->expert->num_ready < task->start_mem_buf_idx) {
      LOG(TRACE) << "scheduler: a stale partial task was reset before launch: " << task->toString();
      if (task->is_precise) {
        task->start_mem_buf_idx = task->expert->num_ready;
      } else {
        return false;
      }
    } else {
      CHECK(task->expert->num_ready == task->start_mem_buf_idx);
    }
  }

  {
    if (log_prefetch_decision_enabled()) {
      if (task->request_type == kCacheRequestEncoderPredictorPrefetch) {
        LOG(INFO) << "prefetch_decision: dispatch encoder_predictor_prefetch"
                  << " L" << task->expert->layer_idx
                  << " E" << task->expert->expert_idx
                  << " P" << task->start_mem_buf_idx << "-" << task->stop_mem_buf_idx
                  << " precise=" << task->is_precise
                  << " forward_epoch=" << task->forward_epoch
                  << " generate_epoch=" << task->generate_epoch;
      } else {
        LOG(INFO) << "prefetch_decision: dispatch " << request_label
                  << " L" << task->expert->layer_idx
                  << " E" << task->expert->expert_idx
                  << " P" << task->start_mem_buf_idx << "-" << task->stop_mem_buf_idx
                  << " precise=" << task->is_precise
                  << " forward_epoch=" << task->forward_epoch
                  << " generate_epoch=" << task->generate_epoch;
      }
    }
    current_task = *task;
    current_task.lambda_wait = lambda_wait;
    if (!current_task.is_precise && is_encoder_prefetch_request(current_task.request_type)) {
      mark_encoder_prefetch_dispatched(
          current_task.expert->layer_idx,
          current_task.expert->expert_idx,
          current_task.stop_mem_buf_idx - current_task.start_mem_buf_idx);
    }
    fetch_thread->add_one_task(&current_task);
  }
  return true;
}
void PrefetchMngr::reload_env() {
  TraceEventCollector::reload_env();
  LogMessage::reload_env();
}
void PrefetchMngr::temp_move_expert_to_gpu(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  auto gpu_data = model_loader->mem_mngr_ctx->dummy_physical;

  // use compute stream to avoid race
  for (int mem_buf_idx = 0; mem_buf_idx < metas->num_per_expert_param; mem_buf_idx++) {
    // LOG(ERROR) << "fetcher: copy from " << task.expert->host_data.ptr(mem_buf_idx) << " to " << task.expert->gpu_data->ptr(mem_buf_idx);
    CUDA_CALL(cudaMemcpyAsync(
        gpu_data->ptr(mem_buf_idx),
        expert->host_data->ptr(mem_buf_idx),
        expert->host_data->nbytes(mem_buf_idx),
        cudaMemcpyHostToDevice, (cudaStream_t)compute_stream));
  }
  expert->reference_to_model_param->unmap();
  expert->reference_to_model_param->map_to(gpu_data, model_loader->mem_mngr_ctx.get());

  CUDA_CALL(cudaStreamSynchronize((cudaStream_t)compute_stream));
}
void PrefetchMngr::temp_move_expert_back_to_host(int layer_id, int expert_id) {
  auto expert = model_loader->get_source(layer_id, expert_id);
  auto gpu_data = model_loader->mem_mngr_ctx->dummy_physical;

  // use compute stream to avoid race
  for (int mem_buf_idx = 0; mem_buf_idx < metas->num_per_expert_param; mem_buf_idx++) {
    // LOG(ERROR) << "fetcher: copy from " << task.expert->host_data.ptr(mem_buf_idx) << " to " << task.expert->gpu_data->ptr(mem_buf_idx);
    CUDA_CALL(cudaMemcpyAsync(
        expert->host_data->ptr(mem_buf_idx),
        gpu_data->ptr(mem_buf_idx),
        expert->host_data->nbytes(mem_buf_idx),
        cudaMemcpyDeviceToHost, (cudaStream_t)compute_stream));
  }

  CUDA_CALL(cudaStreamSynchronize((cudaStream_t)compute_stream));
}
