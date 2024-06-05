#pragma once
#include "utils.hpp"
#include "model_loader.hpp"

class CacheMngr;
class CachePolicy {

  CacheMngr* cache;
 public:
  CachePolicy(CacheMngr* cache) : cache(cache) {}
  virtual ~CachePolicy() = default;
  virtual ExpertHandler* select_for_evict(ExpertHandler*) { return nullptr; }
  virtual void evict(ExpertHandler*) {}
  virtual void access_on_hit(ExpertHandler*) {}
  virtual void access_on_miss(ExpertHandler*) {}
};

class CachePolicyFIFO : public CachePolicy {
 public:
  std::queue<ExpertHandler*> fifo_queue;
  CachePolicyFIFO(CacheMngr* cache) : CachePolicy(cache) {}
  ExpertHandler *select_for_evict(ExpertHandler *) override;
  void evict(ExpertHandler *e) override;
  void access_on_miss(ExpertHandler *e) override;
};

class CachePolicyFactory {
  std::map<std::string, std::function<std::shared_ptr<CachePolicy>()>> registry;
  // CacheMngr* mngr;
 public:
  // CachePolicyFactory(CacheMngr* mngr) : mngr(mngr) {}
  CachePolicyFactory() {}
  void register_policy(std::string name, std::function<std::shared_ptr<CachePolicy>()> constructor) {
    registry[name] = constructor;
  }
  std::shared_ptr<CachePolicy> create_policy(std::string name) {
    return registry[name]();
  }
};

class CacheMngr {
  void handle_hit(ExpertHandler *expert);
  void handle_miss(ExpertHandler *expert);
  CachePolicyFactory policy_factory;
 public:
  std::shared_ptr<ModuleMeta> metas;
  std::shared_ptr<ModelLoader> model_loader;
  std::shared_ptr<CachePolicy> policy;

  std::vector<std::unordered_map<int, ExpertHandler*>> prefetched_experts; // the ongoing job also lives in here.

  std::vector<ExpertMemHanlder*> unused_mems;
  AtomicQueueLock unused_mems_lock;

  CacheMngr(std::shared_ptr<ModuleMeta> metas,
            std::shared_ptr<ModelLoader> model_loader);
  ~CacheMngr();

  bool is_in_cache(ExpertHandler* expert) { return is_in_cache(expert->layer_idx, expert->expert_idx); }

  // legacy methods
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


  // formal methods
  ExpertMemHanlder* evict(ExpertHandler *expert, bool reserve_mem = false);
  void access(ExpertHandler *expert);
  void miss(ExpertHandler *expert);
};