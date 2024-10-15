#include <nlohmann/json.hpp>
#include <fstream>
#include "cache.hpp"
#include "logging.hpp"
#include "profiler.hpp"

void CacheMngr::init_gpu_mem_buffer(size_t num_buffers) {
  size_t num_cache_slot = cache_slots->slots.size();
  // CHECK(num_buffers % num_cache_slot == 0);
  // size_t per_layer_cache_len = num_buffers / num_cache_slot;
  std::vector<size_t> per_slot_cache_len(num_cache_slot, 0);
  {
    // Distribute num_buffers across num_cache_slot as evenly as possible
    size_t base_size = num_buffers / num_cache_slot;
    size_t remainder = num_buffers % num_cache_slot;

    LOG(ERROR) << "for layer <  " << remainder << ", cache " << base_size+1 << "/" << metas->num_expert << " experts";
    LOG(ERROR) << "for layer >= " << remainder << ", cache " << base_size   << "/" << metas->num_expert << " experts";

    for (size_t i = 0; i < num_cache_slot; ++i) {
      per_slot_cache_len[i] = base_size + (i < remainder ? 1 : 0);
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
      total_nbytes += cache_line->get_allocation_nbytes();
    }
  }
  LOG(ERROR) << "cache allocated " << total_nbytes / 1024.0 / 1024.0 << " MiB";
  LOG(ERROR) << "changing num_predict from " << metas->num_predict_expert_per_layer << " to min(" << metas->num_predict_expert_per_layer << ", " << query_per_layer_cache_len() << ")";
  metas->num_predict_expert_per_layer = std::min<int>(metas->num_predict_expert_per_layer, query_per_layer_cache_len());
}
CacheMngr::~CacheMngr() {
  size_t used_mem_cnt = prefetched_experts.size();
  // for (auto &l : prefetched_experts) {
  //   used_mem_cnt += l.size();
  // }
  size_t unused_mem_cnt = 0;
  for (auto &l : cache_slots->slots) {
    unused_mem_cnt += l.unused_mems.size();
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
  TRACE_EVENT_GURAD(kCache, "miss:" + incoming_e->toString());
  LOG(TRACE) << "cache miss " << incoming_e->toString();
  auto cache_slot = cache_slots->to_slot(incoming_e);
  // if (is_precise && metas->predict_input_mode == kOneToken) {
  if (is_precise) {
    max_priority += 1;
    this->priority_set_fn(incoming_e, max_priority);
  }
  CacheLineOccupancyWaiter lambda_to_wait_expert_occupancy = [](){};
  if (cache_slot->unused_mems.size() > 0) {
    auto gpu_data = cache_slot->unused_mems.back();
    incoming_e->gpu_data = gpu_data;
    cache_slot->unused_mems.pop_back();
    cache_slot->policy->access_on_miss(incoming_e, is_precise);
    prefetched_experts[incoming_e] = gpu_data;
  } else {
    auto e_to_evict = cache_slot->policy->select_for_evict(incoming_e);
    // incoming_e->gpu_data = evict(e_to_evict, incoming_e, true);
    {
      TRACE_EVENT_GURAD(kCache, "evict " + e_to_evict->toString());
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
