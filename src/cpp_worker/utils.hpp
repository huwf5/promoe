#pragma once
#include <atomic>
#include <cassert>
#include <pthread.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <torch/extension.h>

class SpinLock {
    pthread_spinlock_t _spinlock;
  public:
    SpinLock() {
        pthread_spin_init(&_spinlock, 0);
    }
    ~SpinLock() {
        pthread_spin_destroy(&_spinlock);
    }
    void lock() {
        pthread_spin_lock(&_spinlock);
    }
    void unlock() {
        pthread_spin_unlock(&_spinlock);
    }
};

// enum ExpertStatus {
//   // kHost = 0,
//   kFetching = 0,
//   kReady = 2,
//   kUsing = 3,
// };

// fixme: is it necessary to distinguish idle, queue, fetching?
enum ExpertStatus {
  kIdle = 0,
  // kQueue,
  kFetching, // partially fetched. it may not be the current task
  kReady,
  kLaunching,
  kUsing,

  // fixme: a new status design
  // prefetcher: idle, queue, fetching, ready
  // inference: using
};

class AtomicMultiStatusLock {
  std::atomic_int lock_;
 public:
  AtomicMultiStatusLock() : lock_(0) {}
  void lock(ExpertStatus from, ExpertStatus to);
  void unlock(ExpertStatus from, ExpertStatus to);
  ExpertStatus try_transfer(ExpertStatus from, ExpertStatus to) {
    int from_ = from;
    lock_.compare_exchange_strong(from_, to);
    return (ExpertStatus)from_;
  }
  ExpertStatus transfer(ExpertStatus from, ExpertStatus to, bool abort_on_fail = true) {
    auto ret = try_transfer(from, to);
    if (abort_on_fail) {
      CHECK(ret == from) << "transfer "<< from << "->" << to << ", but current is " << ret;
    }
    return ret;
  }
  void wait(ExpertStatus from, ExpertStatus to) {
    int from_ = from, to_ = to;
    while (lock_.compare_exchange_strong(from_, to_) == false) {
      from_ = from;
    };
  }
  ExpertStatus exchange(ExpertStatus to) {
    return ExpertStatus(lock_.exchange(to));
  }
  
  bool try_unlock(ExpertStatus from, ExpertStatus to);
  bool is_locked(ExpertStatus locked_status);
  ExpertStatus get() { return ExpertStatus(lock_.load()); }
};

class AtomicLock {
  std::atomic_bool lock_;
 public:
  AtomicLock() : lock_(false) {}
  void lock();
  void unlock();
  // bool is_locked();
};

class AtomicQueueLock {
  // std::atomic_bool lock_;
  std::atomic_int num_req, num_locked;
 public:
  AtomicQueueLock() : num_req(0), num_locked(0) {}
  void lock() {
    int handle = num_req.fetch_add(1);
    while (num_locked.load() != handle) {}
  }
  void unlock() {
    num_locked.fetch_add(1);
  }
  // bool is_locked() {}
};



class ModuleMeta {
  // std::unordered_map<std::string, std::pair<int,int>> module_name_to_expert_idx;
  // std::vector<std::string> expert_idx_to_module_name;
 public:
  int num_layer, num_expert;
  int num_per_expert_param;
  std::vector<std::string> param_name_list;
  std::unordered_map<std::string, int> param_name_to_id;
  int num_predict_expert_per_layer;
  int max_prefetch_layer_distance = 1;
  bool per_layer_cache = true;
  std::string cache_policy = "fifo";

  bool can_do_layer(int cur_preempted_layer, int target_layer) {
    // (cur_preempted_layer, cur_preempted_layer + max_prefetch_layer_distance]
    return ((target_layer + num_layer - 1 - cur_preempted_layer) % num_layer) < max_prefetch_layer_distance;
  }

  ModuleMeta(int num_layer, int num_expert) : num_layer(num_layer), num_expert(num_expert) {}

  void init_param_list(std::vector<std::string> params) {
    param_name_list = params;
    num_per_expert_param = params.size();
    for (int i = 0; i < num_per_expert_param; i++) {
      param_name_to_id[param_name_list[i]] = i;
    }
  }

  inline int squeeze_expert_idx(int layer_id, int expert_id) {
    return layer_id * num_expert + expert_id;
  }
  inline std::pair<int,int> unsqueeze_expert_idx(int flatten_expert_id) {
    return std::make_pair(flatten_expert_id / num_expert, flatten_expert_id % num_expert);
  }
};

std::string tensor_to_str(torch::Tensor t);
// inline std::string expert_meta_to_str(int layer, int e) {
//   return std::to_string(layer) + "." + std::to_string(e);
// }
// inline std::string expert_meta_to_str(int layer, int e, int p) {
//   return std::to_string(layer) + "." + std::to_string(e) + "." + std::to_string(p);
// }

template<typename T>
std::string array_to_str(T* array, size_t len) {
  std::stringstream ss;
  for (int i = 0; i < len; i++) {
    ss << array[i] << ",";
  }
  return ss.str();
}

// inline void CHECK(bool exp) { assert(exp); }

template<typename DATA_T>
struct DoubleLinkedList {
  struct Node {
    Node *prev = nullptr, *next = nullptr;
    DATA_T data;
  };
  Node guard_head, guard_tail;
  DoubleLinkedList() {
    guard_head.next = &guard_tail;
    guard_tail.prev = &guard_head;
  }
  ~DoubleLinkedList() {
    while (empty() == false) {
      delete pop_front();
    }
  }
  void insert_after(Node* new_node, Node* after_me) {
    new_node->next = after_me->next;
    new_node->prev = after_me;
    new_node->next->prev = new_node;
    new_node->prev->next = new_node;
  }
  Node* remove(Node* n) {
    n->next->prev = n->prev;
    n->prev->next = n->next;
    n->next = nullptr;
    n->prev = nullptr;
    return n;
  }

  bool empty() {
    return (guard_head.next == &guard_tail);
  }

  Node* pop_front() {
    return remove(front());
  }
  Node* pop_back() {
    return remove(back());
  }
  Node* front() {
    return guard_head.next;
  }
  Node* back() {
    return guard_head.prev;
  }

  inline void append(Node* n) { return insert_after(n, guard_tail.prev); }
  inline void push_back(Node* n) { return append(n); }
  inline void push_front(Node* n) { return insert_after(n, &guard_head); }

};