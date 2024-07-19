#include <nlohmann/json.hpp>
#include <fstream>
#include "cache.hpp"
#include "logging.hpp"
#include "profiler.hpp"

void CacheMngr::init_gpu_mem_buffer(size_t num_buffers) {
  size_t num_cache_slot = cache_slots->slots.size();
  CHECK(num_buffers % num_cache_slot == 0);
  size_t per_layer_cache_len = num_buffers / num_cache_slot;

  auto &mem_example = model_loader->get_source(0, 0)->host_data.mem_buffers;

  for (auto & cache_slot : cache_slots->slots) {
    cache_slot.full_len = per_layer_cache_len;
    cache_slot.unused_mems.resize(per_layer_cache_len, nullptr);
    for (auto & cache_line : cache_slot.unused_mems) {
      cache_line = new ExpertMemHanlder;
      cache_line->mem_buffers.resize(mem_example.size());
      for (int j = 0; j < mem_example.size(); j++) {
        cache_line->mem_buffers[j].allocate_like(mem_example[j], torch::TensorOptions().device(torch::kCUDA, 0));
      }
    }
  }
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
ExpertMemHanlder* CacheMngr::evict(ExpertHandler *e_to_evict, ExpertHandler *incoming_e, bool reserve_mem) {
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
  cache_slot->policy->access_on_hit(expert);
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
    cache_slot->policy->access_on_miss(incoming_e);
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
      cache_slot->policy->access_on_miss(incoming_e);
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
  auto &oracle = current_sequence->expert_oracles[e];
  if (next_use_time_idx.find(e) == next_use_time_idx.end()) {
    next_use_time_idx[e] = 0;
  }
  auto &use_time_idx = next_use_time_idx[e];
  CHECK(use_time_idx < oracle.use_times.size());
  CHECK(current_time <= oracle.use_times[use_time_idx]);
  LOG(TRACE) << "current time from " << current_time << " to " << oracle.use_times[use_time_idx];
  current_time = oracle.use_times[use_time_idx];
  use_time_idx++;

  auto priority = -oracle.use_times[use_time_idx];
  heap.update_priority(map[e], priority);
}
void CachePolicyMIN::access_on_miss(ExpertHandler *e) {
  auto &oracle = current_sequence->expert_oracles[e];
  if (next_use_time_idx.find(e) == next_use_time_idx.end()) {
    next_use_time_idx[e] = 0;
  }
  auto &use_time_idx = next_use_time_idx[e];
  CHECK(use_time_idx < oracle.use_times.size());
  CHECK(current_time <= oracle.use_times[use_time_idx]);
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
void CacheMngr::update_priority(torch::Tensor p, int starting_layer) {
  if (p.size(0) == metas->num_layer) {
    CHECK(starting_layer == 0);
    update_all_priority(p);
  } else {
    update_some_priority(p, starting_layer);
  }
}
