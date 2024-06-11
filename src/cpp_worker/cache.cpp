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
        cache_line->mem_buffers[j].set_tensor(torch::empty_like(mem_example[j].get_tensor(), torch::TensorOptions().device(torch::kCUDA, 0)));
      }
    }
  }
}
CacheMngr::~CacheMngr() {
  size_t used_mem_cnt = 0;
  for (auto &l : prefetched_experts) {
    used_mem_cnt += l.size();
  }
  size_t unused_mem_cnt = 0;
  for (auto &l : cache_slots->slots) {
    unused_mem_cnt += l.unused_mems.size();
  }
  LOG(ERROR) << unused_mem_cnt << "+" << used_mem_cnt << "=" << unused_mem_cnt + used_mem_cnt;
}
CacheMngr::CacheMngr(std::shared_ptr<ModuleMeta> metas,
                     std::shared_ptr<ModelLoader> model_loader)
    : metas(metas), model_loader(model_loader), policy_factory() {
  prefetched_experts.resize(metas->num_layer);

  policy_factory.register_policy("fifo", [this]() -> std::shared_ptr<CachePolicy>{ return std::make_shared<CachePolicyFIFO>(this); });
  policy_factory.register_policy("lru",  [this]() -> std::shared_ptr<CachePolicy>{ return std::make_shared<CachePolicyLRU>(this);  });


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
  prefetched_experts[e_to_evict->layer_idx].erase(e_to_evict->expert_idx);
  auto ret = e_to_evict->gpu_data;
  CHECK(ret != nullptr);
  e_to_evict->gpu_data = nullptr;
  ret->num_ready = 0;
  if (!reserve_mem) {
    CHECK(false) << "Unimplemented";
    cache_slot->unused_mems.push_back(e_to_evict->gpu_data);
    ret = nullptr;
  }
  return ret;
}
void CacheMngr::access(ExpertHandler *expert) {
  if (is_in_cache(expert)) {
    hit(expert);
  } else {
    miss(expert);
  }
}
void CacheMngr::hit(ExpertHandler *expert) {
  handle_hit(expert);
  cache_slots->to_slot(expert)->policy->access_on_hit(expert);
}

std::function<void()> CacheMngr::miss(ExpertHandler *incoming_e) {
  TRACE_EVENT_GURAD(kCache, "miss:" + incoming_e->toString());
  LOG(TRACE) << "cache miss " << incoming_e->toString();
  auto cache_slot = cache_slots->to_slot(incoming_e);
  std::function<void()> lambda_to_wait_expert_occupancy = [](){};
  if (cache_slot->unused_mems.size() > 0) {
    incoming_e->gpu_data = cache_slot->unused_mems.back();
    cache_slot->unused_mems.pop_back();
    cache_slots->to_slot(incoming_e)->policy->access_on_miss(incoming_e);
    prefetched_experts[incoming_e->layer_idx][incoming_e->expert_idx] = incoming_e;
  } else {
    auto e_to_evict = cache_slot->policy->select_for_evict(incoming_e);
    // incoming_e->gpu_data = evict(e_to_evict, incoming_e, true);
    {
      TRACE_EVENT_GURAD(kCache, "evict " + e_to_evict->toString());
      LOG(TRACE) << "cache evict " << e_to_evict->toString();
      CHECK(e_to_evict != incoming_e);

      cache_slot->policy->evict(e_to_evict);
      cache_slots->to_slot(incoming_e)->policy->access_on_miss(incoming_e);
      prefetched_experts[e_to_evict->layer_idx].erase(e_to_evict->expert_idx);
      prefetched_experts[incoming_e->layer_idx][incoming_e->expert_idx] = incoming_e;
      // kIdle: ?
      // kFetching: ?
      // kReady: ?
      // kLaunching: wait till using, then wait for event, set back to idle
      // kUsing: wait for event, then set back to idle
      // e_to_evict->expert_status.wait(kReady, ExpertStatus to)


      auto orig_status = e_to_evict->expert_status.transfer(kReady, kIdle, false);
      if (orig_status == kUsing || orig_status == kLaunching) {
        TRACE_EVENT_GURAD(kCache, "waiting " + e_to_evict->toString());
        lambda_to_wait_expert_occupancy = [e_to_evict]() {
          e_to_evict->expert_status.wait(kReady, kIdle);
        };
        // lambda_to_wait_expert_occupancy();
      } else if (orig_status == kReady) {
        // successfully locked the e_to_evict to evict
      } else {
        e_to_evict->expert_status.transfer(kFetching, kIdle);// evict a partially fetched e_to_evict
      }

      CHECK(e_to_evict->gpu_data != nullptr);
      incoming_e->gpu_data = e_to_evict->gpu_data;
      e_to_evict->gpu_data = nullptr;
      incoming_e->gpu_data->num_ready = 0;
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
