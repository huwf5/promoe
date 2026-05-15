#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include "cache.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "nvtx_utils.hpp"

namespace {

std::vector<std::shared_ptr<CachePolicy>>& retired_scheduler_aware_policies() {
  static auto* policies = new std::vector<std::shared_ptr<CachePolicy>>();
  return *policies;
}

int parse_initial_plan_int(const std::string &value,
                           const std::string &entry,
                           const std::string &initial_plan_config) {
  CHECK(!value.empty())
      << "invalid initial cache plan entry: " << entry
      << ", initial plan config=" << initial_plan_config;
  size_t pos = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &pos);
  } catch (const std::invalid_argument&) {
    CHECK(false) << "invalid initial cache plan entry: " << entry
                 << ", initial plan config=" << initial_plan_config;
  } catch (const std::out_of_range&) {
    CHECK(false) << "invalid initial cache plan entry: " << entry
                 << ", initial plan config=" << initial_plan_config;
  }
  CHECK(pos == value.size())
      << "invalid initial cache plan entry: " << entry
      << ", initial plan config=" << initial_plan_config;
  return parsed;
}

void bi_traverse(int* array, int begin, int end, std::vector<int>& ret) {
  if (begin == end) {
    return;
  }
  std::queue<std::pair<int,int>> q;
  q.push({begin, end});
  
  while (!q.empty()) {
    auto [l, r] = q.front();
    q.pop();
    if (l == r) continue;

    if (r - l == 1) {
      if (q.empty()) {
        ret.push_back(array[l]);
        return;
      } else {
        // collect all remaining elements in queue to a new array, and call bi_traverse on it
        std::vector<int> remaining;
        for (int i = l; i < r; i++) {
          remaining.push_back(array[i]);
        }
        while (!q.empty()) {
          auto [l, r] = q.front();
          q.pop();
          for (int i = l; i < r; i++) {
            remaining.push_back(array[i]);
          }
        }
        bi_traverse(remaining.data(), 0, remaining.size(), ret);
        return;
      }
    }
    
    int mid = (l + r) / 2;
    ret.push_back(array[mid]);
    
    if (l < mid) {
      q.push({l, mid});
    }
    if (mid + 1 < r) {
      q.push({mid + 1, r}); 
    }
  }
}

};

void CacheMngr::init_gpu_mem_buffer(size_t num_buffers) {
  cache_len = num_buffers;
  size_t num_cache_slot = cache_slots->slots.size();
  // CHECK(num_buffers % num_cache_slot == 0);
  // size_t per_layer_cache_len = num_buffers / num_cache_slot;
  std::vector<size_t> per_slot_cache_len(num_cache_slot, 0);
  {
    // Distribute num_buffers across num_cache_slot as evenly as possible
    size_t base_size = num_buffers / num_cache_slot;
    size_t remainder = num_buffers % num_cache_slot;

    std::vector<int> priority_to_get_remainder;
    std::vector<int> candidates(num_cache_slot, 0);
    for (int i = 0; i < num_cache_slot; i++) {
      candidates[i] = i;
    }
    if (num_cache_slot > 2) {
      priority_to_get_remainder.push_back(candidates[0]);
      priority_to_get_remainder.push_back(candidates[1]);
      bi_traverse(candidates.data(), 2, num_cache_slot, priority_to_get_remainder);
    } else {
      priority_to_get_remainder = candidates;
    }

    for (size_t i = 0; i < num_cache_slot; ++i) {
      auto slot_idx = priority_to_get_remainder[i];
      per_slot_cache_len[slot_idx] = base_size + (i < remainder ? 1 : 0);
    }
    for (size_t i = 0; i < num_cache_slot; ++i) {
      LOG(ERROR) << "layer " << i << " has " << per_slot_cache_len[i] << " buffers";
    }

    // Verify that max difference is <= 1
    size_t min_size = *std::min_element(per_slot_cache_len.begin(), per_slot_cache_len.end());
    size_t max_size = *std::max_element(per_slot_cache_len.begin(), per_slot_cache_len.end());
    CHECK(max_size - min_size <= 1);

    LOG(INFO) << "Cache slots distribution: " << nlohmann::json(per_slot_cache_len).dump();
  }

  size_t max_per_expert_nbytes = 0;
  auto max_expert_example = model_loader->get_source(0, 0)->host_data.get();
  // find the expert with the max memory usage
  for (int l = 0; l < metas->num_layer; l++) {
    for (int e = 0; e < metas->num_expert; e++) {
      auto host_data = model_loader->get_source(l, e)->host_data.get();
      if (host_data->total_alloc_nbytes() > max_per_expert_nbytes) {
        max_per_expert_nbytes = host_data->total_alloc_nbytes();
        max_expert_example = host_data;
      }
    }
  }

  size_t total_nbytes = 0;

  if (metas->physical_mem_impl == "tensor_global_unified") {
    model_loader->mem_mngr_ctx->global_unified_mem_size = max_per_expert_nbytes * num_buffers;
    model_loader->mem_mngr_ctx->global_unified_mem_offset = 0;
    CUDA_CALL(cudaMalloc(&model_loader->mem_mngr_ctx->global_unified_mem, model_loader->mem_mngr_ctx->global_unified_mem_size));
    CUDA_CALL(cudaMemset(model_loader->mem_mngr_ctx->global_unified_mem, 0, model_loader->mem_mngr_ctx->global_unified_mem_size));
  }

  for (int i = 0; i < num_cache_slot; i++) {
    auto & cache_slot = cache_slots->slots[i];
    auto per_layer_cache_len = per_slot_cache_len[i];
    cache_slot.full_len = per_layer_cache_len;
    cache_slot.unused_mems.resize(per_layer_cache_len, nullptr);
    for (auto & cache_line : cache_slot.unused_mems) {
      cache_line = ExpertMemParamFactory::get().create_physical(metas->physical_mem_impl);
      cache_line->allocate_like(max_expert_example, model_loader->mem_mngr_ctx.get());
      cache_slot.all_mems.push_back(cache_line);
      total_nbytes += cache_line->get_allocation_nbytes();
    }
  }
  LOG(ERROR) << "cache allocated " << total_nbytes / 1024.0 / 1024.0 << " MiB";
  LOG(ERROR) << "changing num_predict from " << metas->num_predict_expert_per_layer << " to min(" << metas->num_predict_expert_per_layer << ", " << query_per_layer_cache_len() << ")";
  metas->num_predict_expert_per_layer = std::min<int>(metas->num_predict_expert_per_layer, query_per_layer_cache_len());
}

std::vector<CacheMngr::InitialExpert> CacheMngr::build_manual_initial_plan() const {
  CHECK(metas->initial_cache_policy == "manual")
      << "only initial_cache_policy=manual is supported";

  std::vector<InitialExpert> plan;

  if (!metas->initial_expert_plan.empty()) {
    CHECK(metas->initial_expert_plan.back() != ',')
        << "invalid initial_expert_plan entry: "
        << ", initial_expert_plan=" << metas->initial_expert_plan;

    std::unordered_set<int64_t> seen_experts;
    std::stringstream explicit_plan(metas->initial_expert_plan);
    std::string entry;
    while (std::getline(explicit_plan, entry, ',')) {
      auto sep = entry.find(':');
      CHECK(!entry.empty() && sep != std::string::npos && entry.find(':', sep + 1) == std::string::npos)
          << "invalid initial_expert_plan entry: " << entry
          << ", initial_expert_plan=" << metas->initial_expert_plan;
      int layer_idx = parse_initial_plan_int(entry.substr(0, sep),
                                             entry,
                                             metas->initial_expert_plan);
      int expert_idx = parse_initial_plan_int(entry.substr(sep + 1),
                                              entry,
                                              metas->initial_expert_plan);

      CHECK(layer_idx >= 0 && layer_idx < metas->num_layer)
          << "initial_expert_plan layer out of range: layer=" << layer_idx
          << ", num_layer=" << metas->num_layer
          << ", initial_expert_plan=" << metas->initial_expert_plan;
      CHECK(expert_idx >= 0 && expert_idx < metas->num_expert)
          << "initial_expert_plan expert out of range: layer=" << layer_idx
          << ", expert=" << expert_idx
          << ", num_expert=" << metas->num_expert
          << ", initial_expert_plan=" << metas->initial_expert_plan;
      int64_t gid = int64_t(layer_idx) * int64_t(metas->num_expert) + int64_t(expert_idx);
      CHECK(seen_experts.insert(gid).second)
          << "duplicate expert in initial_expert_plan: layer=" << layer_idx
          << ", expert=" << expert_idx
          << ", initial_expert_plan=" << metas->initial_expert_plan;
      plan.push_back({layer_idx, expert_idx});
    }

    validate_initial_plan_size(plan);
    return plan;
  }

  CHECK(!metas->initial_layer_budgets.empty())
      << "initial_layer_budgets must be non-empty";
  CHECK(metas->initial_layer_budgets.back() != ',')
      << "invalid initial_layer_budgets entry: "
      << ", initial_layer_budgets=" << metas->initial_layer_budgets;

  std::unordered_set<int> seen_layers;
  std::stringstream budgets(metas->initial_layer_budgets);
  std::string entry;
  while (std::getline(budgets, entry, ',')) {
    auto sep = entry.find(':');
    CHECK(!entry.empty() && sep != std::string::npos && entry.find(':', sep + 1) == std::string::npos)
        << "invalid initial_layer_budgets entry: " << entry
        << ", initial_layer_budgets=" << metas->initial_layer_budgets;
    int layer_idx = parse_initial_plan_int(entry.substr(0, sep),
                                           entry,
                                           metas->initial_layer_budgets);
    int budget = parse_initial_plan_int(entry.substr(sep + 1),
                                        entry,
                                        metas->initial_layer_budgets);

    CHECK(layer_idx >= 0 && layer_idx < metas->num_layer)
        << "initial_layer_budgets layer out of range: layer=" << layer_idx
        << ", num_layer=" << metas->num_layer
        << ", initial_layer_budgets=" << metas->initial_layer_budgets;
    CHECK(budget >= 0 && budget <= metas->num_expert)
        << "initial_layer_budgets budget out of range: layer=" << layer_idx
        << ", budget=" << budget
        << ", num_expert=" << metas->num_expert
        << ", initial_layer_budgets=" << metas->initial_layer_budgets;
    CHECK(seen_layers.insert(layer_idx).second)
        << "duplicate layer in initial_layer_budgets: layer=" << layer_idx
        << ", initial_layer_budgets=" << metas->initial_layer_budgets;

    for (int expert_idx = 0; expert_idx < budget; expert_idx++) {
      plan.push_back({layer_idx, expert_idx});
    }
  }

  validate_initial_plan_size(plan);
  return plan;
}

void CacheMngr::validate_initial_plan_size(const std::vector<InitialExpert>& plan) const {
  size_t plan_size = plan.size();
  if (plan_size < cache_len) {
    CHECK(false) << "initial cache plan smaller than cache_size"
                 << ", cache_size=" << cache_len
                 << ", plan_size=" << plan_size
                 << ", missing=" << (cache_len - plan_size)
                 << ", initial_expert_plan=" << metas->initial_expert_plan
                 << ", initial_layer_budgets=" << metas->initial_layer_budgets;
  }
  if (plan_size > cache_len) {
    CHECK(false) << "initial cache plan larger than cache_size"
                 << ", cache_size=" << cache_len
                 << ", plan_size=" << plan_size
                 << ", overflow=" << (plan_size - cache_len)
                 << ", initial_expert_plan=" << metas->initial_expert_plan
                 << ", initial_layer_budgets=" << metas->initial_layer_budgets;
  }
}

void CacheMngr::reset_cache_contents() {
  NVTX_RANGE("cache/reset_cache_contents");
  CHECK(metas->per_layer_cache == false)
      << "deterministic initial cache requires global cache";
  CHECK(cache_slots->slots.size() == 1)
      << "deterministic initial cache requires global cache";

  LOG(INFO) << "cache/reset_cache_contents: begin prefetched_experts="
            << prefetched_experts.size()
            << " slots=" << cache_slots->slots.size();
  size_t reset_idx = 0;
  size_t ready_count = 0;
  size_t idle_count = 0;
  size_t fetching_count = 0;
  for (auto &pair : prefetched_experts) {
    auto expert = pair.first;
    CHECK(expert != nullptr) << "prefetched_experts contains null expert";
    auto status = expert->expert_status.get();
    if ((reset_idx % 64) == 0) {
      LOG(INFO) << "cache/reset_cache_contents: resetting entry idx="
                << reset_idx
                << " expert=" << expert->layer_idx << "." << expert->expert_idx
                << " status=" << status
                << " num_ready=" << expert->num_ready
                << " gpu_ptr=" << pair.second;
    }
    CHECK(status == kReady || status == kIdle || status == kFetching)
        << "cannot reset active expert " << expert->toString()
        << " with status " << status;
    if (status == kReady) {
      expert->expert_status.transfer(kReady, kIdle);
      ready_count += 1;
    } else if (status == kFetching) {
      expert->expert_status.transfer(kFetching, kIdle);
      fetching_count += 1;
    } else {
      idle_count += 1;
    }
    expert->gpu_data = nullptr;
    expert->num_ready = 0;
    reset_idx += 1;
  }
  LOG(INFO) << "cache/reset_cache_contents: reset entries done total="
            << reset_idx
            << " ready=" << ready_count
            << " idle=" << idle_count
            << " fetching=" << fetching_count;
  LOG(INFO) << "cache/reset_cache_contents: clear prefetched_experts begin";
  prefetched_experts.clear();
  LOG(INFO) << "cache/reset_cache_contents: clear prefetched_experts done";
  for (auto &cache_slot : cache_slots->slots) {
    LOG(INFO) << "cache/reset_cache_contents: reset slot begin all_mems="
              << cache_slot.all_mems.size()
              << " unused_before=" << cache_slot.unused_mems.size();
    cache_slot.unused_mems = cache_slot.all_mems;
    LOG(INFO) << "cache/reset_cache_contents: reset slot unused_mems done";
    if (metas->cache_policy == "scheduler_aware" &&
        cache_slot.policy != nullptr) {
      retired_scheduler_aware_policies().push_back(cache_slot.policy);
      cache_slot.policy = policy_factory.create_policy(metas->cache_policy);
      LOG(INFO) << "cache/reset_cache_contents: reset slot policy retired";
    } else {
      cache_slot.policy = policy_factory.create_policy(metas->cache_policy);
      LOG(INFO) << "cache/reset_cache_contents: reset slot policy recreated";
    }
  }
  LOG(INFO) << "cache/reset_cache_contents: reset slots done";
  max_priority = std::numeric_limits<float>::min();
  LOG(INFO) << "cache/reset_cache_contents: max priority reset done";
  if (metas->cache_policy == "nn") {
    priority.zero_();
  }
  LOG(INFO) << "cache/reset_cache_contents: priority tensor reset done";

  LOG(INFO) << "cache/reset_cache_contents: final unused_mems="
            << cache_slots->slots[0].unused_mems.size()
            << " cache_len=" << cache_len;
  CHECK(cache_slots->slots[0].unused_mems.size() == cache_len)
      << "cache reset did not restore all cache lines: restored="
      << cache_slots->slots[0].unused_mems.size()
      << ", cache_len=" << cache_len;
  LOG(INFO) << "cache/reset_cache_contents: final check done";
}

void CacheMngr::load_initial_plan_sync(cudaStream_t stream) {
  NVTX_RANGE("cache/load_initial_plan_sync");
  Timer total_timer;
  uint64_t build_plan_us = 0;
  uint64_t miss_wait_us = 0;
  uint64_t h2d_enqueue_us = 0;
  uint64_t remap_us = 0;
  uint64_t stream_sync_us = 0;
  uint64_t status_us = 0;
  std::vector<InitialExpert> plan;
  {
    NVTX_RANGE("cache/load_initial_plan/build_plan");
    Timer timer;
    plan = build_manual_initial_plan();
    build_plan_us = timer.dur_us();
  }
  for (auto [layer_idx, expert_idx] : plan) {
    NVTX_RANGE("cache/load_initial_plan/expert");
    auto expert = model_loader->get_source(layer_idx, expert_idx);
    CHECK(!is_in_cache(expert));
    {
      NVTX_RANGE("cache/load_initial_plan/miss_and_wait");
      Timer timer;
      auto waiter = miss(expert, false);
      waiter();
      miss_wait_us += timer.dur_us();
    }
    {
      Timer timer;
      expert->expert_status.transfer(kIdle, kFetching);
      status_us += timer.dur_us();
    }

    {
      NVTX_RANGE("cache/load_initial_plan/h2d_enqueue");
      Timer timer;
      for (int mem_buf_idx = 0; mem_buf_idx < metas->num_per_expert_param; mem_buf_idx++) {
        CUDA_CALL(cudaMemcpyAsync(
            expert->gpu_data->ptr(mem_buf_idx),
            expert->host_data->ptr(mem_buf_idx),
            expert->host_data->nbytes(mem_buf_idx),
            cudaMemcpyHostToDevice, stream));
      }
      h2d_enqueue_us += timer.dur_us();
    }
    {
      NVTX_RANGE("cache/load_initial_plan/remap_model_param");
      Timer timer;
      expert->reference_to_model_param->unmap();
      expert->reference_to_model_param->map_to(expert->gpu_data,
                                               model_loader->mem_mngr_ctx.get());
      remap_us += timer.dur_us();
    }
    {
      NVTX_RANGE("cache/load_initial_plan/stream_sync");
      Timer timer;
      CUDA_CALL(cudaStreamSynchronize(stream));
      stream_sync_us += timer.dur_us();
    }
    {
      Timer timer;
      expert->num_ready = metas->num_per_expert_param;
      expert->expert_status.transfer(kFetching, kReady);
      status_us += timer.dur_us();
    }
  }
  {
    NVTX_RANGE("cache/load_initial_plan/verify_ready");
    for (auto [layer_idx, expert_idx] : plan) {
      auto expert = model_loader->get_source(layer_idx, expert_idx);
      CHECK(expert->num_ready == metas->num_per_expert_param);
      CHECK(expert->expert_status.get() == kReady);
    }
  }
  LOG(INFO) << "ttft_breakdown_load_initial_plan_us total=" << total_timer.dur_us()
            << " build_plan=" << build_plan_us
            << " miss_wait=" << miss_wait_us
            << " h2d_enqueue=" << h2d_enqueue_us
            << " remap=" << remap_us
            << " stream_sync=" << stream_sync_us
            << " status=" << status_us
            << " experts=" << plan.size()
            << " num_per_expert_param=" << metas->num_per_expert_param;
}
CacheMngr::~CacheMngr() {
  size_t used_mem_cnt = prefetched_experts.size();
  // for (auto &l : prefetched_experts) {
  //   used_mem_cnt += l.size();
  // }
  size_t unused_mem_cnt = 0;
  for (auto &l : cache_slots->slots) {
    unused_mem_cnt += l.unused_mems.size();
    if (metas->cache_policy == "scheduler_aware" && l.policy != nullptr) {
      retired_scheduler_aware_policies().push_back(l.policy);
      l.policy.reset();
    }
  }
  LOG(ERROR) << unused_mem_cnt << "+" << used_mem_cnt << "=" << unused_mem_cnt + used_mem_cnt;
}
CacheMngr::CacheMngr(std::shared_ptr<ModuleMeta> metas,
                     std::shared_ptr<ModelLoader> model_loader)
    : metas(metas), model_loader(model_loader), policy_factory() {
  // prefetched_experts.resize(metas->num_layer);

  if (metas->cache_policy == "nn") {
    this->priority_get_fn = [this](ExpertHandler* e) ->float { return this->priority.index({e->layer_idx, e->expert_idx}).item<float>(); };
    this->priority_set_fn = [this](ExpertHandler* e, float p)  { this->priority.index_put_({e->layer_idx, e->expert_idx}, p); };
    this->priority = torch::zeros({metas->num_layer, metas->num_expert}, torch::kFloat32);
  }
  if (metas->cache_policy == "min") {
    this->cache_oracle = std::make_shared<CacheOracle>();
    cache_oracle->init(metas.get(), model_loader.get());
  }

  policy_factory.register_policy("fifo", [this]() -> std::shared_ptr<CachePolicy>{ return std::make_shared<CachePolicyFIFO>(this); });
  policy_factory.register_policy("lru",  [this]() -> std::shared_ptr<CachePolicy>{ return std::make_shared<CachePolicyLRU>(this);  });
  policy_factory.register_policy("static-1",  [this]() -> std::shared_ptr<CachePolicy>{ return std::make_shared<CachePolicyStatic>(this, 1);  });
  policy_factory.register_policy("static-2",  [this]() -> std::shared_ptr<CachePolicy>{ return std::make_shared<CachePolicyStatic>(this, 2);  });
  policy_factory.register_policy("nn",   [this]() -> std::shared_ptr<CachePolicy>{ 
    auto ret = std::make_shared<CachePolicyNN>(this);
    ret->priority_fn = this->priority_get_fn;
    return ret;
  });
  policy_factory.register_policy("min", [this]()->std::shared_ptr<CachePolicy>{
    auto ret = std::make_shared<CachePolicyMIN>(this);
    ret->oracle = this->cache_oracle.get();
    return ret;
  });
  policy_factory.register_policy("scheduler_aware", [this]() -> std::shared_ptr<CachePolicy>{
    return std::make_shared<CachePolicySchedulerAware>(this);
  });


  if (metas->per_layer_cache) {
    cache_slots = std::make_shared<SlotMapperPerLayer>();
    cache_slots->slots.resize(metas->num_layer);
  } else {
    cache_slots = std::make_shared<SlotMapper>();
    cache_slots->slots.resize(1);
  }

  for (auto & cache_slot : cache_slots->slots) {
    cache_slot.policy = policy_factory.create_policy(metas->cache_policy);
  }

}

void CacheMngr::handle_hit(ExpertHandler *expert) {}
void CacheMngr::handle_miss(ExpertHandler *expert) {
  CHECK(false) << "Deprecated";
}
ExpertMemHanlderBase* CacheMngr::evict(ExpertHandler *e_to_evict, ExpertHandler *incoming_e, bool reserve_mem) {
  CHECK(false) << "Deprecated";
  TRACE_EVENT_GURAD(kCache, "evict " + e_to_evict->toString());
  LOG(TRACE) << "cache evict " << e_to_evict->toString();
  CHECK(e_to_evict != incoming_e);
  // kIdle: ?
  // kFetching: ?
  // kReady: ?
  // kLaunching: wait till using, then wait for event, set back to idle
  // kUsing: wait for event, then set back to idle
  // e_to_evict->expert_status.wait(kReady, ExpertStatus to)

  auto orig_status = e_to_evict->expert_status.transfer(kReady, kIdle, false);
  if (orig_status == kUsing || orig_status == kLaunching) {
    TRACE_EVENT_GURAD(kCache, "waiting " + e_to_evict->toString());
    e_to_evict->expert_status.wait(kReady, kIdle);
  } else if (orig_status == kReady) {
    // successfully locked the e_to_evict to evict
  } else {
    e_to_evict->expert_status.transfer(kFetching, kIdle);// evict a partially fetched e_to_evict
  }

  auto cache_slot = cache_slots->to_slot(e_to_evict);

  cache_slot->policy->evict(e_to_evict);
  auto ret = prefetched_experts[e_to_evict];
  prefetched_experts.erase(e_to_evict);
  CHECK(ret != nullptr);
  e_to_evict->gpu_data = nullptr;
  if (!reserve_mem) {
    CHECK(false) << "Unimplemented";
    cache_slot->unused_mems.push_back(ret);
    ret = nullptr;
  }
  return ret;
}
void CacheMngr::access(ExpertHandler *expert, bool is_precise) {
  if (is_in_cache(expert)) {
    hit(expert, is_precise);
  } else {
    miss(expert, is_precise);
  }
}
void CacheMngr::hit(ExpertHandler *expert, bool is_precise) {
  auto cache_slot = cache_slots->to_slot(expert);
  handle_hit(expert);
  cache_slot->policy->access_on_hit(expert, is_precise);
  // if (is_precise && metas->predict_input_mode == kOneToken) {
  if (is_precise) {
    max_priority += 1;
    cache_slot->policy->update_priority(expert, max_priority);
    this->priority_set_fn(expert, max_priority);
  }
}

CacheMngr::CacheLineOccupancyWaiter CacheMngr::miss(ExpertHandler *incoming_e, bool is_precise) {
  return miss(incoming_e, is_precise,
              is_precise ? kCacheRequestDemand : kCacheRequestPrefetch);
}

CacheMngr::CacheLineOccupancyWaiter CacheMngr::miss(
    ExpertHandler *incoming_e,
    bool is_precise,
    CacheRequestType request_type) {
  TRACE_EVENT_GURAD(kCache, "miss:" + incoming_e->toString());
  NVTX_RANGE("cache/miss L" + std::to_string(incoming_e->layer_idx) +
             " E" + std::to_string(incoming_e->expert_idx) +
             (is_precise ? " precise" : " prefetch"));
  LOG(TRACE) << "cache miss " << incoming_e->toString();
  auto cache_slot = cache_slots->to_slot(incoming_e);
  // if (is_precise && metas->predict_input_mode == kOneToken) {
  if (is_precise) {
    max_priority += 1;
    this->priority_set_fn(incoming_e, max_priority);
  }
  CacheLineOccupancyWaiter lambda_to_wait_expert_occupancy = [](){};
  if (cache_slot->unused_mems.size() > 0 &&
      request_type != kCacheRequestDecoderWarmupOverlap) {
    auto gpu_data = cache_slot->unused_mems.back();
    incoming_e->gpu_data = gpu_data;
    cache_slot->unused_mems.pop_back();
    cache_slot->policy->access_on_miss(incoming_e, is_precise);
    prefetched_experts[incoming_e] = gpu_data;
  } else {
    auto e_to_evict = cache_slot->policy->select_for_evict(incoming_e, request_type);
    if (e_to_evict == nullptr) {
      CHECK(request_type == kCacheRequestDecoderWarmupOverlap)
          << "only decoder warmup overlap may skip eviction";
      return [](){};
    }
    // incoming_e->gpu_data = evict(e_to_evict, incoming_e, true);
    {
      TRACE_EVENT_GURAD(kCache, "evict " + e_to_evict->toString());
      NVTX_RANGE("cache/evict oldL" + std::to_string(e_to_evict->layer_idx) +
                 " oldE" + std::to_string(e_to_evict->expert_idx) +
                 " newL" + std::to_string(incoming_e->layer_idx) +
                 " newE" + std::to_string(incoming_e->expert_idx));
      LOG_BLOCK(DEBUG, logger, {
        logger << "evict " << e_to_evict->toString() << ", policy state is " << cache_slot->policy->toString();
      });
      LOG(TRACE) << "cache evict " << e_to_evict->toString();
      CHECK(e_to_evict != incoming_e);

      cache_slot->policy->evict(e_to_evict);
      cache_slot->policy->access_on_miss(incoming_e, is_precise);
      LOG_BLOCK(DEBUG, logger, {
        logger << "after evict, policy state is " << cache_slot->policy->toString();
      });
      auto gpu_data = prefetched_experts[e_to_evict];
      prefetched_experts.erase(e_to_evict);
      prefetched_experts[incoming_e] = gpu_data;
      // kIdle: ?
      // kFetching: ?
      // kReady: ?
      // kLaunching: wait till using, then wait for event, set back to idle
      // kUsing: wait for event, then set back to idle
      // e_to_evict->expert_status.wait(kReady, ExpertStatus to)


      auto orig_status = e_to_evict->expert_status.transfer(kReady, kIdle, false);
      if (orig_status == kUsing || orig_status == kLaunching) {
        lambda_to_wait_expert_occupancy = [e_to_evict]() {
          TRACE_EVENT_GURAD(kFetcher, "waiting " + e_to_evict->toString());
          NVTX_RANGE("cache/wait_evict oldL" + std::to_string(e_to_evict->layer_idx) +
                     " oldE" + std::to_string(e_to_evict->expert_idx));
          e_to_evict->expert_status.wait(kReady, kIdle);
        };
        // lambda_to_wait_expert_occupancy();
      } else if (orig_status == kReady) {
        // successfully locked the e_to_evict to evict
      } else {
        e_to_evict->expert_status.transfer(kFetching, kIdle);// evict a partially fetched e_to_evict
      }

      CHECK(e_to_evict->gpu_data == gpu_data);
      e_to_evict->gpu_data = nullptr;
      e_to_evict->num_ready = 0;
      incoming_e->gpu_data = gpu_data;
      // incoming_e->num_ready = 0;
    }
  }
  return lambda_to_wait_expert_occupancy;
}

void CacheMngr::mark_reclaimable(int layer_idx, int expert_idx) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  auto expert = model_loader->get_source(layer_idx, expert_idx);
  if (!is_in_cache(expert)) {
    return;
  }
  cache_slots->to_slot(expert)->policy->mark_reclaimable(expert);
}

void CacheMngr::mark_layer_reclaimable(int layer_idx) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  for (int expert_idx = 0; expert_idx < metas->num_expert; expert_idx++) {
    mark_reclaimable(layer_idx, expert_idx);
  }
}

void CacheMngr::mark_layer_reclaimable_except(
    int layer_idx,
    const std::unordered_set<int>& needed_eids) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  for (int expert_idx = 0; expert_idx < metas->num_expert; expert_idx++) {
    if (needed_eids.find(expert_idx) == needed_eids.end()) {
      mark_reclaimable(layer_idx, expert_idx);
    }
  }
}

void CacheMngr::mark_layer_reclaimable_except(
    int layer_idx,
    const std::vector<uint8_t>& needed_mask) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  for (int expert_idx = 0; expert_idx < metas->num_expert; expert_idx++) {
    const bool needed =
        expert_idx < static_cast<int>(needed_mask.size()) && needed_mask[expert_idx];
    if (!needed) {
      mark_reclaimable(layer_idx, expert_idx);
    }
  }
}

bool CacheMngr::has_reclaimable_encoder() const {
  for (auto &slot : cache_slots->slots) {
    if (slot.policy->has_reclaimable_encoder()) {
      return true;
    }
  }
  return false;
}

CachePolicySchedulerAware::~CachePolicySchedulerAware() {
  global_map.clear();
  encoder_map.clear();
  decoder_map.clear();
  reclaimable_map.clear();
  node_free_buffer.clear();

  auto reset_list = [](LL& list) {
    list.guard_head.prev = nullptr;
    list.guard_head.next = &list.guard_tail;
    list.guard_tail.prev = &list.guard_head;
    list.guard_tail.next = nullptr;
    list.len = 0;
  };
  reset_list(global_lru);
  reset_list(encoder_lru);
  reset_list(decoder_lru);
  reset_list(reclaimable_encoder_lru);
}

bool CachePolicySchedulerAware::reset_for_cache_reset() {
  global_map.clear();
  encoder_map.clear();
  decoder_map.clear();
  reclaimable_map.clear();
  node_free_buffer.clear();

  auto reset_list = [](LL& list) {
    list.guard_head.prev = nullptr;
    list.guard_head.next = &list.guard_tail;
    list.guard_tail.prev = &list.guard_head;
    list.guard_tail.next = nullptr;
    list.len = 0;
  };
  reset_list(global_lru);
  reset_list(encoder_lru);
  reset_list(decoder_lru);
  reset_list(reclaimable_encoder_lru);
  return true;
}

CachePolicySchedulerAware::LL::Node* CachePolicySchedulerAware::new_node(ExpertHandler* expert) {
  LL::Node* node = nullptr;
  if (node_free_buffer.empty()) {
    node = new LL::Node;
  } else {
    node = node_free_buffer.back();
    node_free_buffer.pop_back();
  }
  node->data = expert;
  return node;
}

void CachePolicySchedulerAware::recycle_node(LL::Node* node) {
  node_free_buffer.push_back(node);
}

void CachePolicySchedulerAware::touch(
    std::unordered_map<ExpertHandler*, LL::Node*>& map,
    LL& list,
    ExpertHandler* expert) {
  auto it = map.find(expert);
  if (it != map.end()) {
    auto node = list.remove(it->second);
    list.push_back(node);
    return;
  }
  auto node = new_node(expert);
  map[expert] = node;
  list.push_back(node);
}

void CachePolicySchedulerAware::access_on_hit(ExpertHandler* expert) {
  auto reclaimable_it = reclaimable_map.find(expert);
  if (reclaimable_it != reclaimable_map.end()) {
    auto node = reclaimable_encoder_lru.remove(reclaimable_it->second);
    recycle_node(node);
    reclaimable_map.erase(reclaimable_it);
  }
  touch(global_map, global_lru, expert);
  if (cache->metas->is_encoder_layer(expert->layer_idx)) {
    touch(encoder_map, encoder_lru, expert);
  } else if (cache->metas->is_decoder_layer(expert->layer_idx)) {
    touch(decoder_map, decoder_lru, expert);
  }
}

void CachePolicySchedulerAware::access_on_miss(ExpertHandler* expert) {
  access_on_hit(expert);
}

void CachePolicySchedulerAware::mark_reclaimable(ExpertHandler* expert) {
  if (!cache->metas->is_encoder_layer(expert->layer_idx)) {
    return;
  }
  if (!cache->is_in_cache_ptr(expert)) {
    return;
  }
  touch(reclaimable_map, reclaimable_encoder_lru, expert);
}

void CachePolicySchedulerAware::evict(ExpertHandler* expert) {
  auto erase_from = [this, expert](std::unordered_map<ExpertHandler*, LL::Node*>& map, LL& list) {
    auto it = map.find(expert);
    if (it == map.end()) {
      return;
    }
    auto node = list.remove(it->second);
    recycle_node(node);
    map.erase(it);
  };
  erase_from(global_map, global_lru);
  erase_from(encoder_map, encoder_lru);
  erase_from(decoder_map, decoder_lru);
  erase_from(reclaimable_map, reclaimable_encoder_lru);
}

ExpertHandler* CachePolicySchedulerAware::first_loaded_candidate(
    std::unordered_map<ExpertHandler*, LL::Node*>& map,
    LL& list) {
  for (auto node = list.front(); node != &list.guard_tail; node = node->next) {
    auto expert = node->data;
    if (map.find(expert) != map.end() && cache->is_in_cache_ptr(expert)) {
      return expert;
    }
  }
  return nullptr;
}

bool CachePolicySchedulerAware::has_reclaimable_encoder() const {
  for (auto& pair : reclaimable_map) {
    if (cache->is_in_cache_ptr(pair.first)) {
      return true;
    }
  }
  return false;
}

ExpertHandler* CachePolicySchedulerAware::select_for_evict(ExpertHandler* incoming) {
  return select_for_evict(incoming, kCacheRequestPrefetch);
}

ExpertHandler* CachePolicySchedulerAware::select_for_evict(
    ExpertHandler* incoming,
    CacheRequestType request_type) {
  if (auto victim = first_loaded_candidate(reclaimable_map, reclaimable_encoder_lru)) {
    return victim;
  }
  if (request_type == kCacheRequestDecoderWarmupOverlap) {
    return nullptr;
  }
  if (auto victim = first_loaded_candidate(encoder_map, encoder_lru)) {
    return victim;
  }
  if (auto victim = first_loaded_candidate(decoder_map, decoder_lru)) {
    return victim;
  }
  if (auto victim = first_loaded_candidate(global_map, global_lru)) {
    return victim;
  }
  CHECK(false) << "scheduler_aware policy could not choose victim for incoming "
               << incoming->toString();
  return nullptr;
}

std::string CachePolicySchedulerAware::toString() {
  std::stringstream ss;
  ss << "scheduler_aware(global=" << global_map.size()
     << ",encoder=" << encoder_map.size()
     << ",decoder=" << decoder_map.size()
     << ",reclaimable=" << reclaimable_map.size()
     << ")";
  return ss.str();
}
void CachePolicyFIFO::evict(ExpertHandler *e) {
  CHECK(e == select_for_evict(nullptr));
  fifo_queue.pop();
}
void CachePolicyFIFO::access_on_miss(ExpertHandler *e) { fifo_queue.push(e); }
ExpertHandler *CachePolicyFIFO::select_for_evict(ExpertHandler *) {
  return fifo_queue.front();
}
void CachePolicyLRU::access_on_hit(ExpertHandler *e) {
  CHECK(map.find(e) != map.end());
  auto n = map[e];
  linked_list.remove(n);
  linked_list.push_back(n);
}
void CachePolicyLRU::access_on_miss(ExpertHandler *e) {
  CHECK(map.find(e) == map.end());
  LL::Node *n = nullptr;
  if (linked_list_node_free_buffer.empty()) {
    n = new LL::Node;
  } else {
    n = linked_list_node_free_buffer.back();
    linked_list_node_free_buffer.pop_back();
  }
  map[e] = n;
  n->data = e;
  linked_list.push_back(n);
}
void CachePolicyStatic::access_on_hit(ExpertHandler *e) {
  CHECK(map.find(e) != map.end());
  // auto n = map[e];
  // linked_list.remove(n);
  // linked_list.push_back(n);
}
void CachePolicyStatic::access_on_miss(ExpertHandler *e) {
  CHECK(map.find(e) == map.end());

  if (linked_list.size() == this->max_alternative_buffer_len) {
    // remove one from linked_list to persisted_experts
    auto first_node = linked_list.pop_front();
    auto expert_to_persist = first_node->data;
    linked_list_node_free_buffer.push_back(first_node);
    first_node = nullptr;
    persisted_experts.insert(expert_to_persist);
    map[expert_to_persist] = nullptr;
  }

  CHECK(linked_list.size() < this->max_alternative_buffer_len);

  LL::Node *n = nullptr;
  if (linked_list_node_free_buffer.empty()) {
    n = new LL::Node;
  } else {
    n = linked_list_node_free_buffer.back();
    linked_list_node_free_buffer.pop_back();
  }

  map[e] = n;
  n->data = e;
  linked_list.push_back(n);
}

void CachePolicyStatic::evict(ExpertHandler *e) {
  CHECK(map.find(e) != map.end());
  CHECK(map[e] != nullptr);
  CHECK(persisted_experts.find(e) == persisted_experts.end());
  auto n = linked_list.remove(map[e]);
  linked_list_node_free_buffer.push_back(n);
  map.erase(e);
}

void CachePolicyNN::access_on_hit(ExpertHandler *e) {
  CHECK(map.find(e) != map.end());
}
void CachePolicyNN::access_on_miss(ExpertHandler *e) {
  CHECK(map.find(e) == map.end());
  Heap::Node *n = nullptr;
  if (heap_node_free_buffer.empty()) {
    n = new Heap::Node;
  } else {
    n = heap_node_free_buffer.back();
    heap_node_free_buffer.pop_back();
  }
  map[e] = n;
  n->data = e;
  n->priority = priority_fn(e);
  heap.push(n);
}
void CacheMngr::update_all_priority(torch::Tensor p) {
  priority = p.clone();
  max_priority = priority.max().item<float>();
  for (auto &slot : cache_slots->slots) {
    reinterpret_cast<CachePolicyNN *>(slot.policy.get())->update_all_priority();
  }
  LOG_BLOCK(DEBUG, logger, {
    logger << "update priority, result: \n";
    for (int s = 0; s < cache_slots->slots.size(); s++) {
      logger << "slot " << s << ":" << cache_slots->slots[s].policy->toString() << "\n";
    }
  });
}
void CacheMngr::update_some_priority(torch::Tensor p, int starting_layer) {
  CHECK(starting_layer + p.size(0) <= metas->num_layer);
  priority.index_put_({torch::indexing::Slice{starting_layer, starting_layer + p.size(0)}}, p);
  max_priority = priority.max().item<float>();
  std::unordered_set<CacheSlot*> deduped_slot;
  for (int i = 0; i < p.size(0); i++) {
    deduped_slot.insert(cache_slots->to_slot(i + starting_layer));
  }
  for (auto s : deduped_slot) {
    reinterpret_cast<CachePolicyNN *>(s->policy.get())->update_all_priority();
  }
  LOG_BLOCK(DEBUG, logger, {
    logger << "update priority, result: \n";
    for (int s = 0; s < cache_slots->slots.size(); s++) {
      logger << "slot " << s << ":" << cache_slots->slots[s].policy->toString() << "\n";
    }
  });
}
void CacheOracle::load_from_file(std::string file_path) {
  std::ifstream trace_file(file_path);
  nlohmann::json all_traces = nlohmann::json::parse(trace_file);
  for (auto &el : all_traces.items()) {
    uint64_t seq_id = std::stoull(el.key());
    auto & seq_trace = el.value();
    this->sequence_oracles[seq_id] = SequenceOracle();
    auto & seq_oracle = this->sequence_oracles[seq_id];


    for (int l = 0; l < metas->num_layer; l++) {
      for (int e = 0; e < metas->num_expert; e++) {
        auto expert = model_loader->get_source(l, e);
        seq_oracle.expert_oracles[expert] = ExpertOracle();
      }
    }

    uint64_t prompt_len = seq_trace["prompt_len"].get<uint64_t>();
    uint64_t reply_len = seq_trace["0"].size() - prompt_len;
    // prompt
    int64_t time = 0;
    for (int layer_idx = 0; layer_idx < metas->num_layer; layer_idx++) {
      auto &layer_trace = seq_trace[std::to_string(layer_idx)];
      std::set<int> expert_idx_set;
      for (int prompt_token_idx = 0; prompt_token_idx < prompt_len; prompt_token_idx++) {
        auto &expert_idx_list = layer_trace[std::to_string(prompt_token_idx)];
        for (auto &expert_idx : expert_idx_list) {
          auto expert_idx_int = expert_idx.get<int>();
          expert_idx_set.insert(expert_idx_int);
        }
      }
      for (auto &expert_idx : expert_idx_set) {
        auto e = model_loader->get_source(layer_idx, expert_idx);
        seq_oracle.expert_oracles[e].use_times.push_back(time);
      }
      time += 1;
    }
    // reply
    for (int reply_token_idx = 0; reply_token_idx < reply_len; reply_token_idx++) {
      for (int layer_idx = 0; layer_idx < metas->num_layer; layer_idx++) {
        auto &layer_trace = seq_trace[std::to_string(layer_idx)];
        auto &expert_idx_list = layer_trace[std::to_string(reply_token_idx + prompt_len)];

        for (auto &expert_idx : expert_idx_list) {
          auto expert_idx_int = expert_idx.get<int>();
          auto e = model_loader->get_source(layer_idx, expert_idx_int);
          seq_oracle.expert_oracles[e].use_times.push_back(time);
        }
        time += 1;
      }
    }
    for (int l = 0; l < metas->num_layer; l++) {
      for (int e = 0; e < metas->num_expert; e++) {
        auto expert = model_loader->get_source(l, e);
        seq_oracle.expert_oracles[expert].use_times.push_back(std::numeric_limits<int64_t>::max());
      }
    }
  }
}
void CachePolicyMIN::access_on_hit(ExpertHandler *e) {
  // LOG(ERROR) << "min access on hit " << e->toString();
  auto &oracle = current_sequence->expert_oracles[e];
  if (next_use_time_idx.find(e) == next_use_time_idx.end()) {
    next_use_time_idx[e] = 0;
  }
  auto &use_time_idx = next_use_time_idx[e];
  CHECK(use_time_idx < oracle.use_times.size());
  CHECK(current_time <= oracle.use_times[use_time_idx]) << current_time << " > " << oracle.use_times[use_time_idx];
  LOG(TRACE) << "current time from " << current_time << " to " << oracle.use_times[use_time_idx];
  current_time = oracle.use_times[use_time_idx];
  use_time_idx++;

  auto priority = -oracle.use_times[use_time_idx];
  heap.update_priority(map[e], priority);
}
void CachePolicyMIN::access_on_hit(ExpertHandler *e, bool is_precise) {
  // LOG(ERROR) << "min access on hit " << e->toString() << ", is_precise:" << is_precise;
  if (is_precise) {
    return access_on_hit(e);
  }
  auto &oracle = current_sequence->expert_oracles[e];
  if (next_use_time_idx.find(e) == next_use_time_idx.end()) {
    next_use_time_idx[e] = 0;
  }
  auto &use_time_idx = next_use_time_idx[e];
  // CHECK(use_time_idx < oracle.use_times.size());
  // CHECK(current_time <= oracle.use_times[use_time_idx]);
  // LOG(TRACE) << "current time from " << current_time << " to " << oracle.use_times[use_time_idx];
  // current_time = oracle.use_times[use_time_idx];
  // use_time_idx++;

  auto priority = -oracle.use_times[use_time_idx];
  heap.update_priority(map[e], priority);
}
void CachePolicyMIN::access_on_miss(ExpertHandler *e) {
  // LOG(ERROR) << "min access on miss " << e->toString();
  auto &oracle = current_sequence->expert_oracles[e];
  if (next_use_time_idx.find(e) == next_use_time_idx.end()) {
    next_use_time_idx[e] = 0;
  }
  auto &use_time_idx = next_use_time_idx[e];
  CHECK(use_time_idx < oracle.use_times.size());
  CHECK(current_time <= oracle.use_times[use_time_idx]) << current_time << " > " << oracle.use_times[use_time_idx];
  LOG(TRACE) << "current time from " << current_time << " to " << oracle.use_times[use_time_idx];
  current_time = oracle.use_times[use_time_idx];
  use_time_idx++;

  auto priority = -oracle.use_times[use_time_idx];

  CHECK(map.find(e) == map.end());
  Heap::Node *n = nullptr;
  if (heap_node_free_buffer.empty()) {
    n = new Heap::Node;
  } else {
    n = heap_node_free_buffer.back();
    heap_node_free_buffer.pop_back();
  }
  map[e] = n;
  n->data = e;
  n->priority = priority;
  heap.push(n);
}
void CachePolicyMIN::access_on_miss(ExpertHandler *e, bool is_precise) {
  // LOG(ERROR) << "min access on miss " << e->toString() << ", is_precise:" << is_precise;
  if (is_precise) {
    return access_on_miss(e);
  }
  auto &oracle = current_sequence->expert_oracles[e];
  if (next_use_time_idx.find(e) == next_use_time_idx.end()) {
    next_use_time_idx[e] = 0;
  }
  auto &use_time_idx = next_use_time_idx[e];
  // CHECK(use_time_idx < oracle.use_times.size());
  // CHECK(current_time <= oracle.use_times[use_time_idx]);
  // LOG(TRACE) << "current time from " << current_time << " to " << oracle.use_times[use_time_idx];
  // current_time = oracle.use_times[use_time_idx];
  // use_time_idx++;

  auto priority = -oracle.use_times[use_time_idx];

  CHECK(map.find(e) == map.end());
  Heap::Node *n = nullptr;
  if (heap_node_free_buffer.empty()) {
    n = new Heap::Node;
  } else {
    n = heap_node_free_buffer.back();
    heap_node_free_buffer.pop_back();
  }
  map[e] = n;
  n->data = e;
  n->priority = priority;
  heap.push(n);
}
void CacheMngr::update_priority(torch::Tensor p, int starting_layer) {
  if (p.size(0) == metas->num_layer) {
    CHECK(starting_layer == 0);
    update_all_priority(p);
  } else {
    update_some_priority(p, starting_layer);
  }
}
void CacheOracle::load_from_tensor(torch::Tensor entry_metas, torch::Tensor prefill_expert_len, torch::Tensor prefill_expert_selection, torch::Tensor decode_expert_selection) {
  auto num_seq = prefill_expert_len.size(0);  
  auto per_seq_rply_len = entry_metas.index({torch::indexing::Slice{}, 0}).bincount();
  uint64_t cur_seq_entry_begin = 0;
  for (uint64_t seq_id = 0; seq_id < num_seq; seq_id++) {
    this->sequence_oracles[seq_id] = SequenceOracle();

    auto & seq_oracle = this->sequence_oracles[seq_id];

    for (int l = 0; l < metas->num_layer; l++) {
      for (int e = 0; e < metas->num_expert; e++) {
        auto expert = model_loader->get_source(l, e);
        seq_oracle.expert_oracles[expert] = ExpertOracle();
      }
    }
    uint64_t reply_len = per_seq_rply_len[seq_id].item<int64_t>();

    LOG(ERROR) << "seq_id:" << seq_id << ", reply_len:" << reply_len;

    // prompt
    int64_t time = 0;
    for (int layer_idx = 0; layer_idx < metas->num_layer; layer_idx++) {
      uint32_t cur_layer_prefill_expert_len = prefill_expert_len[seq_id][layer_idx].item<int32_t>();
      for (uint32_t offset = 0; offset < cur_layer_prefill_expert_len; offset++) {
        uint32_t expert_idx = prefill_expert_selection[seq_id][layer_idx][offset].item<int32_t>();
        auto e = model_loader->get_source(layer_idx, expert_idx);
        seq_oracle.expert_oracles[e].use_times.push_back(time);
      }
      time += 1;
    }

    // reply
    for (int reply_token_idx = 0; reply_token_idx < reply_len; reply_token_idx++) {
      CHECK(entry_metas[cur_seq_entry_begin + reply_token_idx][0].item<int64_t>() == seq_id)          << cur_seq_entry_begin << "," << reply_token_idx << ":" << entry_metas[cur_seq_entry_begin + reply_token_idx][0].item<int64_t>() << "," << seq_id;
      CHECK(entry_metas[cur_seq_entry_begin + reply_token_idx][1].item<int64_t>() == reply_token_idx) << cur_seq_entry_begin << "," << reply_token_idx << ":" << entry_metas[cur_seq_entry_begin + reply_token_idx][1].item<int64_t>() << "," << reply_token_idx;
      for (int layer_idx = 0; layer_idx < metas->num_layer; layer_idx++) {
        for (uint32_t offset = 0; offset < decode_expert_selection.size(2); offset++) {
          uint32_t expert_idx = decode_expert_selection[cur_seq_entry_begin + reply_token_idx][layer_idx][offset].item<int32_t>();
          auto e = model_loader->get_source(layer_idx, expert_idx);
          seq_oracle.expert_oracles[e].use_times.push_back(time);
        }
        time += 1;
      }
    }
    cur_seq_entry_begin += reply_len;
    for (int l = 0; l < metas->num_layer; l++) {
      for (int e = 0; e < metas->num_expert; e++) {
        auto expert = model_loader->get_source(l, e);
        seq_oracle.expert_oracles[expert].use_times.push_back(std::numeric_limits<int64_t>::max());
      }
    }
  }
}
