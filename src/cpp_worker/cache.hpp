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
  virtual void update_priority() {}
  virtual void update_priority(ExpertHandler* e, float p) {}
};

class CachePolicyFIFO : public CachePolicy {
 public:
  std::queue<ExpertHandler*> fifo_queue;
  using CachePolicy::CachePolicy;
  ExpertHandler *select_for_evict(ExpertHandler *) override;
  void evict(ExpertHandler *e) override;
  void access_on_miss(ExpertHandler *e) override;
};

class CachePolicyLRU : public CachePolicy {
  using LL = DoubleLinkedList<ExpertHandler*>;
  std::vector<LL::Node*> linked_list_node_free_buffer;
  LL linked_list;
  std::unordered_map<ExpertHandler*, LL::Node*> map;
 public:
  using CachePolicy::CachePolicy;
  ~CachePolicyLRU() {
    while (linked_list_node_free_buffer.empty() == false) {
      delete linked_list_node_free_buffer.back();
      linked_list_node_free_buffer.pop_back();
    }
  }
  ExpertHandler *select_for_evict(ExpertHandler *) override {
    return linked_list.front()->data;
  }
  void evict(ExpertHandler *e) override {
    auto n = linked_list.remove(map[e]);
    linked_list_node_free_buffer.push_back(n);
    map.erase(e);
  }
  void access_on_hit(ExpertHandler *e) override;
  void access_on_miss(ExpertHandler *e) override;
};

class CachePolicyNN : public CachePolicy {
  using Heap = MinHeap<ExpertHandler*>;
  std::vector<Heap::Node*> heap_node_free_buffer;
  Heap heap;
  std::unordered_map<ExpertHandler*, Heap::Node*> map;
  // torch::Tensor priority;
 public:
  std::function<float(ExpertHandler*)> priority_fn;
  using CachePolicy::CachePolicy;
  ~CachePolicyNN() {
    while (heap_node_free_buffer.empty() == false) {
      delete heap_node_free_buffer.back();
      heap_node_free_buffer.pop_back();
    }
  }
  ExpertHandler *select_for_evict(ExpertHandler *) override {
    return heap.front()->data;
  }
  void evict(ExpertHandler *e) override {
    auto n = heap.remove(map[e]);
    heap_node_free_buffer.push_back(n);
    map.erase(e);
  }
  void access_on_hit(ExpertHandler *e) override;
  void access_on_miss(ExpertHandler *e) override;
  void update_priority() override {
    for (auto &pair : map) {
      pair.second->priority = priority_fn(pair.first);
    }
    heap.rebuild();
  }
  void update_priority(ExpertHandler* e, float p) override {
    auto orig_p = map[e]->priority;
    map[e]->priority = p;
    if (p < orig_p) {
      heap.heapify_up(map[e]->current_idx);
    } else {
      heap.heapify_down(map[e]->current_idx);
    }
  }
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
  torch::Tensor priority;
 public:
  using CacheLineOccupancyWaiter = std::function<void()>;
  size_t cache_len = 0;
  std::shared_ptr<ModuleMeta> metas;
  std::shared_ptr<ModelLoader> model_loader;

  std::unordered_map<ExpertHandler*, ExpertMemHanlder*> prefetched_experts; // the ongoing job also lives in here.

  std::shared_ptr<SlotMapper> cache_slots;

  CacheMngr(std::shared_ptr<ModuleMeta> metas,
            std::shared_ptr<ModelLoader> model_loader);
  ~CacheMngr();

  bool is_in_cache(ExpertHandler* expert) {
    return prefetched_experts.find(expert) != prefetched_experts.end();
  }

  // legacy methods
  bool is_in_cache(int layer_id, int expert_id) {
    return is_in_cache(model_loader->get_source(layer_id, expert_id));
  }
  void init_gpu_mem_buffer(size_t num_buffers);


  // formal methods
  size_t query_per_layer_cache_len(int layer_idx = 0) {
    return cache_slots->to_slot(layer_idx)->full_len;
  }
  ExpertMemHanlder* evict(ExpertHandler *evict_e, ExpertHandler *incoming_e=nullptr, bool reserve_mem = false);
  void access(ExpertHandler *expert, bool is_precise);
  void hit(ExpertHandler *expert, bool is_precise);
  CacheLineOccupancyWaiter miss(ExpertHandler *expert, bool is_precise);

  void update_priority(torch::Tensor p) {
    priority = p.clone();
    for (auto & slot : cache_slots->slots) {
      reinterpret_cast<CachePolicyNN*>(slot.policy.get())->update_priority();
    }
  }
};