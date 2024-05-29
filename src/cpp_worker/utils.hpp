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

enum ExpertStatus {
  // kHost = 0,
  kFetching = 0,
  kReady = 2,
  kUsing = 3,
};

class AtomicMultiStatusLock {
  std::atomic_int lock_;
 public:
  AtomicMultiStatusLock() : lock_(0) {}
  void lock(ExpertStatus from, ExpertStatus to);
  void unlock(ExpertStatus from, ExpertStatus to);
  bool try_unlock(ExpertStatus from, ExpertStatus to);
  bool is_locked(ExpertStatus locked_status);
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

template<typename T>
std::string array_to_str(T* array, size_t len) {
  std::stringstream ss;
  for (int i = 0; i < len; i++) {
    ss << array[i] << ",";
  }
  return ss.str();
}

// inline void CHECK(bool exp) { assert(exp); }