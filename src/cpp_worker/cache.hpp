#pragma once
#include "utils.hpp"
#include "model_loader.hpp"

class CachePolicy {

};

class CacheMngr {
 public:
  std::shared_ptr<ModuleMeta> metas;
  std::shared_ptr<ModelLoader> model_loader;

  std::vector<std::unordered_map<int, ExpertHandler*>> prefetched_experts; // the ongoing job also lives in here.

  std::vector<ExpertMemHanlder*> unused_mems;
  AtomicQueueLock unused_mems_lock;

  CacheMngr(std::shared_ptr<ModuleMeta> metas,
            std::shared_ptr<ModelLoader> model_loader);
  ~CacheMngr();

  bool is_in_cache(int layer_id, int expert_id) {
    return (prefetched_experts[layer_id].find(expert_id) != prefetched_experts[layer_id].end());
  }
  void erase(int layer_id, int expert_id) {
    prefetched_experts[layer_id].erase(expert_id);
  }
  void add_free_buffer(ExpertMemHanlder * ptr) {
    unused_mems_lock.lock();
    unused_mems.push_back(ptr);
    unused_mems_lock.unlock();
  }
  ExpertMemHanlder *allocate_from_free_buffer();
  void register_cached(ExpertHandler* expert) {
    prefetched_experts[expert->layer_idx][expert->expert_idx] = expert;
  }
  void init_gpu_mem_buffer(size_t num_buffers);
};