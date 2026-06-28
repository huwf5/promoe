#pragma once
#include "utils.hpp"
#include "model_loader.hpp"
#include <unordered_set>

class CacheMngr;
enum CacheRequestType {
  kCacheRequestDemand = 0,
  kCacheRequestEncoderPredictorPrefetch,
  kCacheRequestEncoderJitRefill,
  kCacheRequestDecoderPredictorPrefetch,
  kCacheRequestDecoderWarmupPrefetch,
  kCacheRequestInitialLoad,
};

class CachePolicy {

 protected:
  CacheMngr* cache;
 public:
  CachePolicy(CacheMngr* cache) : cache(cache) {}
  virtual ~CachePolicy() = default;
  virtual ExpertHandler* select_for_evict(ExpertHandler*) { return nullptr; }
  virtual ExpertHandler* select_for_evict(ExpertHandler* incoming, CacheRequestType) {
    return select_for_evict(incoming);
  }
  virtual void evict(ExpertHandler*) {}
  virtual void access_on_hit(ExpertHandler*) {}
  virtual void access_on_hit(ExpertHandler* e, bool is_precise) { return access_on_hit(e); }
  virtual void access_on_miss(ExpertHandler*) {}
  virtual void access_on_miss(ExpertHandler* e, bool is_precise) { return access_on_miss(e); }
  virtual void mark_reclaimable(ExpertHandler*) {}
  virtual void mark_demand_protected(ExpertHandler*) {}
  virtual void clear_demand_protected(ExpertHandler*) {}
  virtual void clear_demand_protected_for_layer(int) {}
  virtual void clear_all_demand_protected() {}
  virtual bool is_demand_protected(ExpertHandler*) const { return false; }
  virtual bool has_reclaimable_encoder() const { return false; }
  virtual bool reset_for_cache_reset() { return false; }
  virtual void update_all_priority() {}
  virtual void update_priority(ExpertHandler* e, float p) {}
  virtual void set_cur_seq(uint64_t seq_id) {}
  virtual std::string toString () { return ""; }
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
  std::unordered_set<ExpertHandler*> demand_protected_experts;
 public:
  using CachePolicy::CachePolicy;
  ~CachePolicyLRU() {
    while (linked_list_node_free_buffer.empty() == false) {
      delete linked_list_node_free_buffer.back();
      linked_list_node_free_buffer.pop_back();
    }
  }
  ExpertHandler *select_for_evict(ExpertHandler *) override;
  void evict(ExpertHandler *e) override;
  void access_on_hit(ExpertHandler *e) override;
  void access_on_miss(ExpertHandler *e) override;
  void mark_demand_protected(ExpertHandler* expert) override;
  void clear_demand_protected(ExpertHandler* expert) override;
  void clear_demand_protected_for_layer(int layer_idx) override;
  void clear_all_demand_protected() override;
  bool is_demand_protected(ExpertHandler* expert) const override;
};
class CachePolicyStatic : public CachePolicy {
  using LL = DoubleLinkedList<ExpertHandler*>;
  std::vector<LL::Node*> linked_list_node_free_buffer;
  LL linked_list;
  std::unordered_map<ExpertHandler*, LL::Node*> map;
  std::unordered_set<ExpertHandler*> persisted_experts;
  int max_alternative_buffer_len = 2;
 public:
  using CachePolicy::CachePolicy;
  CachePolicyStatic(CacheMngr* cache, int max_alternative_buffer_len = 2) : CachePolicy(cache), max_alternative_buffer_len(max_alternative_buffer_len) {}
  ~CachePolicyStatic() {
    while (linked_list_node_free_buffer.empty() == false) {
      delete linked_list_node_free_buffer.back();
      linked_list_node_free_buffer.pop_back();
    }
  }
  ExpertHandler *select_for_evict(ExpertHandler *) override {
    return linked_list.front()->data;
  }
  void evict(ExpertHandler *e) override;
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
  void update_all_priority() override {
    for (auto &pair : map) {
      pair.second->priority = priority_fn(pair.first);
    }
    heap.rebuild();
  }
  void update_priority(ExpertHandler* e, float p) override {
    heap.update_priority(map[e], p);
  }
  std::string toString () override {
    auto v = heap.heap_buffer;
    std::sort(v.begin(), v.end(), [](Heap::Node* a, Heap::Node* b) {
      return a->priority > b->priority;
    });
    std::stringstream ss;
    for (auto n : v) {
      ss << "(" << n->priority << "," << n->data->expert_idx << "),";
    }
    return ss.str();
  }
};


class CacheOracle {
 public:
  struct ExpertOracle {
    std::vector<int64_t> use_times;
  };
  struct SequenceOracle {
    std::unordered_map<ExpertHandler*, ExpertOracle> expert_oracles;
  };
  ModuleMeta* metas;
  ModelLoader* model_loader;
  std::unordered_map<uint64_t, SequenceOracle> sequence_oracles;
  void load_from_file(std::string file_path);
  void load_from_tensor(torch::Tensor entry_metas, torch::Tensor prefill_expert_len, torch::Tensor prefill_expert_selection, torch::Tensor decode_expert_selection);
  void init(ModuleMeta* metas, ModelLoader* model_loader) {
    this->metas = metas;
    this->model_loader = model_loader;
  }
};

class CachePolicyMIN: public CachePolicy {
  CacheOracle::SequenceOracle* current_sequence = nullptr;
  int64_t current_time = -1;

  std::unordered_map<ExpertHandler*, size_t> next_use_time_idx;

  using Heap = MinHeap<ExpertHandler*>;
  std::vector<Heap::Node*> heap_node_free_buffer;
  Heap heap;
  std::unordered_map<ExpertHandler*, Heap::Node*> map;

 public:
  CacheOracle* oracle;
  using CachePolicy::CachePolicy;
  ~CachePolicyMIN() {
    while (heap_node_free_buffer.empty() == false) {
      delete heap_node_free_buffer.back();
      heap_node_free_buffer.pop_back();
    }
  }

  void set_cur_seq(uint64_t seq_id) override {
    current_sequence = &oracle->sequence_oracles[seq_id];
    current_time = -1;

    for (auto &pair : next_use_time_idx) {
      pair.second = 0;
    }

    for (auto &pair : map) {
      auto e = pair.first;
      auto n = pair.second;
      auto& oracle = current_sequence->expert_oracles[e];
      CHECK(next_use_time_idx.find(e) != next_use_time_idx.end());

      auto priority = -oracle.use_times[0];
      pair.second->priority = priority;
    }
    heap.rebuild();
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
  void access_on_hit(ExpertHandler *e, bool is_precise) override;
  void access_on_miss(ExpertHandler *e) override;
  void access_on_miss(ExpertHandler *e, bool is_precise) override;
  // void update_all_priority() override {
  //   // for (auto &pair : map) {
  //   //   pair.second->priority = priority_fn(pair.first);
  //   // }
  //   // heap.rebuild();
  // }
  // void update_priority(ExpertHandler* e, float p) override {
  //   // heap.update_priority(map[e], p);
  // }
  // std::string toString () override {
  //   return "";
  // }

  std::string toString () override {
    auto v = heap.heap_buffer;
    std::sort(v.begin(), v.end(), [](Heap::Node* a, Heap::Node* b) {
      return a->priority > b->priority;
    });
    std::stringstream ss;
    for (auto n : v) {
      ss << "(" << (-n->priority) << "," << n->data->expert_idx << "),";
    }
    return ss.str();
  }
};

class CachePolicySchedulerAware : public CachePolicy {
  using LL = DoubleLinkedList<ExpertHandler*>;
  std::vector<LL::Node*> node_free_buffer;
  LL global_lru;
  LL encoder_lru;
  LL decoder_lru;
  LL reclaimable_encoder_lru;
  std::unordered_map<ExpertHandler*, LL::Node*> global_map;
  std::unordered_map<ExpertHandler*, LL::Node*> encoder_map;
  std::unordered_map<ExpertHandler*, LL::Node*> decoder_map;
  std::unordered_map<ExpertHandler*, LL::Node*> reclaimable_map;
  LL demand_protected_lru;
  std::unordered_map<ExpertHandler*, LL::Node*> demand_protected_map;

  LL::Node* new_node(ExpertHandler* expert);
  void recycle_node(LL::Node* node);
  void touch(std::unordered_map<ExpertHandler*, LL::Node*>& map, LL& list, ExpertHandler* expert);
  ExpertHandler* first_loaded_candidate(std::unordered_map<ExpertHandler*, LL::Node*>& map, LL& list);
 public:
  using CachePolicy::CachePolicy;
  ~CachePolicySchedulerAware();
  ExpertHandler* select_for_evict(ExpertHandler* incoming) override;
  ExpertHandler* select_for_evict(ExpertHandler* incoming, CacheRequestType request_type) override;
  void evict(ExpertHandler* expert) override;
  void access_on_hit(ExpertHandler* expert) override;
  void access_on_miss(ExpertHandler* expert) override;
  void mark_reclaimable(ExpertHandler* expert) override;
  void mark_demand_protected(ExpertHandler* expert) override;
  void clear_demand_protected(ExpertHandler* expert) override;
  void clear_demand_protected_for_layer(int layer_idx) override;
  void clear_all_demand_protected() override;
  bool is_demand_protected(ExpertHandler* expert) const override;
  bool has_reclaimable_encoder() const override;
  bool reset_for_cache_reset() override;
  std::string toString() override;
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
  std::vector<ExpertMemHanlderBase*> unused_mems;
  std::vector<ExpertMemHanlderBase*> all_mems;
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
  float max_priority = std::numeric_limits<float>::min();
  std::function<float(ExpertHandler*)> priority_get_fn = [](ExpertHandler* e) ->float { return 0; };
  std::function<void(ExpertHandler*, float)> priority_set_fn = [](ExpertHandler* e, float p) {};
  std::vector<int> encoder_layer_cached_count;
  void increment_encoder_layer_cached_count(ExpertHandler* expert);
  void decrement_encoder_layer_cached_count(ExpertHandler* expert);

 public:
  std::shared_ptr<CacheOracle> cache_oracle;
  using CacheLineOccupancyWaiter = std::function<void()>;
  using InitialExpert = std::pair<int, int>;
  size_t cache_len = 0;
  std::shared_ptr<ModuleMeta> metas;
  std::shared_ptr<ModelLoader> model_loader;

  std::unordered_map<ExpertHandler*, ExpertMemHanlderBase*> prefetched_experts; // the ongoing job also lives in here.

  std::shared_ptr<SlotMapper> cache_slots;

  CacheMngr(std::shared_ptr<ModuleMeta> metas,
            std::shared_ptr<ModelLoader> model_loader);
  ~CacheMngr();

  bool is_in_cache(ExpertHandler* expert) {
    return prefetched_experts.find(expert) != prefetched_experts.end();
  }
  bool is_in_cache_ptr(ExpertHandler* expert) const {
    return prefetched_experts.find(expert) != prefetched_experts.end();
  }

  // legacy methods
  bool is_in_cache(int layer_id, int expert_id) {
    return is_in_cache(model_loader->get_source(layer_id, expert_id));
  }
  void init_gpu_mem_buffer(size_t num_buffers);
  std::vector<InitialExpert> build_manual_initial_plan() const;
  void validate_initial_plan_size(const std::vector<InitialExpert>& plan) const;
  void reset_cache_contents();
  void load_initial_plan_sync(cudaStream_t stream);


  // formal methods
  size_t query_per_layer_cache_len(int layer_idx = 0) {
    return cache_slots->to_slot(layer_idx)->full_len;
  }
  ExpertMemHanlderBase* evict(ExpertHandler *evict_e, ExpertHandler *incoming_e=nullptr, bool reserve_mem = false);
  void access(ExpertHandler *expert, bool is_precise);
  void hit(ExpertHandler *expert, bool is_precise);
  CacheLineOccupancyWaiter miss(ExpertHandler *expert, bool is_precise);
  CacheLineOccupancyWaiter miss(ExpertHandler *expert, bool is_precise, CacheRequestType request_type);
  void mark_reclaimable(int layer_idx, int expert_idx);
  void mark_layer_reclaimable(int layer_idx);
  void mark_layer_reclaimable_except(int layer_idx, const std::unordered_set<int>& needed_eids);
  void mark_layer_reclaimable_except(int layer_idx, const std::vector<uint8_t>& needed_mask);
  bool has_reclaimable_encoder() const;
  void mark_demand_protected(ExpertHandler* expert);
  void clear_demand_protected(ExpertHandler* expert);
  void clear_demand_protected(int layer_idx, int expert_idx);
  void clear_demand_protected_for_layer(int layer_idx);
  void clear_all_demand_protected();
  bool is_demand_protected(ExpertHandler* expert) const;
  bool has_unused_slot_for(ExpertHandler* expert) const;
  int encoder_layer_cache_occupancy(int layer_idx) const;

  void update_all_priority(torch::Tensor p);
  void update_some_priority(torch::Tensor p, int starting_layer);
  void update_priority(torch::Tensor p, int starting_layer);

  void set_cur_seq(uint64_t seq_id) {
    for (auto &slot : cache_slots->slots) {
      slot.policy->set_cur_seq(seq_id);
    }
  }
};
