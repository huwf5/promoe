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


struct CacheSlot {
  std::shared_ptr<CachePolicy> policy;
  std::vector<ExpertMemHanlder*> unused_mems;
  size_t full_len;
};

class SlotMapper {
 public:
  std::vector<CacheSlot> slots;
  virtual int to_slot_idx(int layer_id) { return 0; };
  inline CacheSlot* to_slot(int layer_id) { return &slots[to_slot_idx(layer_id)]; }
  inline CacheSlot* to_slot(ExpertHandler* expert) { return to_slot(expert->layer_idx); }
  virtual ~SlotMapper() {}
};

class SlotMapperPerLayer : public SlotMapper {
 public:
  int to_slot_idx(int layer_id) override { return layer_id; }
};

class CacheMngr {
  void handle_hit(ExpertHandler *expert);
  void handle_miss(ExpertHandler *expert);
  CachePolicyFactory policy_factory;
  friend class SlotMapper;
 public:
  size_t cache_len = 0;
  std::shared_ptr<ModuleMeta> metas;
  std::shared_ptr<ModelLoader> model_loader;

  std::vector<std::unordered_map<int, ExpertHandler*>> prefetched_experts; // the ongoing job also lives in here.

  std::shared_ptr<SlotMapper> cache_slots;
  AtomicQueueLock unused_mems_lock;

  CacheMngr(std::shared_ptr<ModuleMeta> metas,
            std::shared_ptr<ModelLoader> model_loader);
  ~CacheMngr();

  bool is_in_cache(ExpertHandler* expert) { return is_in_cache(expert->layer_idx, expert->expert_idx); }

  // legacy methods
  bool is_in_cache(int layer_id, int expert_id) {
    return (prefetched_experts[layer_id].find(expert_id) != prefetched_experts[layer_id].end());
  }
  void init_gpu_mem_buffer(size_t num_buffers);


  // formal methods
  size_t query_per_layer_cache_len(int layer_idx = 0) {
    return cache_slots->to_slot(layer_idx)->full_len;
  }
  ExpertMemHanlder* evict(ExpertHandler *evict_e, ExpertHandler *incoming_e=nullptr, bool reserve_mem = false);
  void access(ExpertHandler *expert);
  void miss(ExpertHandler *expert);
};