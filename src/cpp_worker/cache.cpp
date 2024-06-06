#include "cache.hpp"
#include "logging.hpp"
#include "profiler.hpp"

ExpertMemHanlder *CacheMngr::allocate_from_free_buffer() {
  ExpertMemHanlder *ret = nullptr;
  unused_mems_lock.lock();
  if (unused_mems.size() > 0) {
    ret = unused_mems.back();
    CHECK(ret != nullptr);
    unused_mems.pop_back();
  } else {
    CHECK(false) << "no remaining mem buffer";
  }
  unused_mems_lock.unlock();
  return ret;
}
void CacheMngr::init_gpu_mem_buffer(size_t num_buffers) {
  cache_len = num_buffers;
  unused_mems.resize(num_buffers, nullptr);
  auto &mem_example = model_loader->get_source(0, 0)->host_data.mem_buffers;
  for (int i = 0; i < num_buffers; i++) {
    unused_mems[i] = new ExpertMemHanlder;
    unused_mems[i]->mem_buffers.resize(mem_example.size());
    for (int j = 0; j < mem_example.size(); j++) {
      unused_mems[i]->mem_buffers[j].set_tensor(torch::empty_like(mem_example[j].get_tensor(), torch::TensorOptions().device(torch::kCUDA, 0)));
    }
  }
}
CacheMngr::~CacheMngr() {
  size_t used_mem_cnt = 0;
  for (auto &l : prefetched_experts) {
    used_mem_cnt += l.size();
  }
  LOG(ERROR) << unused_mems.size() << "+" << used_mem_cnt << "=" << unused_mems.size() + used_mem_cnt;
}
CacheMngr::CacheMngr(std::shared_ptr<ModuleMeta> metas,
                     std::shared_ptr<ModelLoader> model_loader)
    : metas(metas), model_loader(model_loader), policy_factory() {
  prefetched_experts.resize(metas->num_layer);
  policy_factory.register_policy("fifo", [this]() -> std::shared_ptr<CachePolicy>{
    return std::make_shared<CachePolicyFIFO>(this);
  });
  this->policy = policy_factory.create_policy("fifo");
}

void CacheMngr::handle_hit(ExpertHandler *expert) {}
void CacheMngr::handle_miss(ExpertHandler *expert) {
  if (unused_mems.size() > 0) {
    expert->gpu_data = unused_mems.back();
    unused_mems.pop_back();
  } else {
    auto e_to_evict = policy->select_for_evict(expert);
    expert->gpu_data = evict(e_to_evict, expert, true);
  }
  prefetched_experts[expert->layer_idx][expert->expert_idx] = expert;
}
ExpertMemHanlder* CacheMngr::evict(ExpertHandler *expert, ExpertHandler *incoming_e, bool reserve_mem) {
  TRACE_EVENT_GURAD(kCache, "evict " + expert->toString());
  LOG(TRACE) << "cache evict " << expert->toString();
  CHECK(expert != incoming_e);
  // kIdle: ?
  // kFetching: ?
  // kReady: ?
  // kLaunching: wait till using, then wait for event, set back to idle
  // kUsing: wait for event, then set back to idle
  // expert->expert_status.wait(kReady, ExpertStatus to)

  auto orig_status = expert->expert_status.transfer(kReady, kIdle, false);
  if (orig_status == kUsing || orig_status == kLaunching) {
    TRACE_EVENT_GURAD(kCache, "waiting " + expert->toString());
    expert->expert_status.wait(kReady, kIdle);
  } else if (orig_status == kReady) {
    // successfully locked the expert to evict
  } else {
    expert->expert_status.transfer(kFetching, kIdle);// evict a partially fetched expert
  }

  policy->evict(expert);
  prefetched_experts[expert->layer_idx].erase(expert->expert_idx);
  auto ret = expert->gpu_data;
  CHECK(ret != nullptr);
  expert->gpu_data = nullptr;
  ret->num_ready = 0;
  if (!reserve_mem) {
    unused_mems.push_back(expert->gpu_data);
    ret = nullptr;
  }
  return ret;
  // fixme: remove from cache map
}
void CacheMngr::access(ExpertHandler *expert) {
  if (is_in_cache(expert)) {
    handle_hit(expert);
    policy->access_on_hit(expert);
  } else {
    miss(expert);
  }
}
void CacheMngr::miss(ExpertHandler *expert) {
  TRACE_EVENT_GURAD(kCache, "miss:" + expert->toString());
  LOG(TRACE) << "cache miss " << expert->toString();
  handle_miss(expert);
  policy->access_on_miss(expert);
}
void CachePolicyFIFO::evict(ExpertHandler *e) {
  CHECK(e == select_for_evict(nullptr));
  fifo_queue.pop();
}
void CachePolicyFIFO::access_on_miss(ExpertHandler *e) { fifo_queue.push(e); }
ExpertHandler *CachePolicyFIFO::select_for_evict(ExpertHandler *) {
  return fifo_queue.front();
}
