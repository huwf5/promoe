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
  void wait(ExpertStatus target) {
    while (lock_.load() != target) {};
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
  int num_expert_per_token;
  int max_prefetch_layer_distance = 1;
  bool per_layer_cache = true;
  bool reorder_experts = true;
  bool promote_hit_in_prefetch = true;
  bool early_preempt = true;
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

template<typename DATA_T>
struct MinHeap {
  struct Node {
    float priority;
    int current_idx;
    DATA_T data;
  };
  std::vector<Node*> heap_buffer;
  MinHeap() {}
  void rebuild() {
    for (int i = heap_buffer.size() / 2; i >= 0; i--) {
      heapify_down(i);
    }
  }
  int to_left(int i) { return 2 * i + 1; }
  int to_right(int i) { return 2 * i + 2; }
  int to_parent(int i) { return (i - 1) / 2;}
  void swap(int i, int j) {
    std::swap(heap_buffer[i], heap_buffer[j]);
    heap_buffer[i]->current_idx = i;
    heap_buffer[j]->current_idx = j;
  }
  void heapify_down(int i) {
    int left = to_left(i);
    int right = to_right(i);
    int smallest = i;
    if (left < heap_buffer.size() && heap_buffer[left]->priority < heap_buffer[smallest]->priority) {
      smallest = left;
    }
    if (right < heap_buffer.size() && heap_buffer[right]->priority < heap_buffer[smallest]->priority) {
      smallest = right;
    }
    if (smallest != i) {
      swap(i, smallest);
      heapify_down(smallest);
    }
  }
  void push(Node* v) {
    v->current_idx = heap_buffer.size();
    heap_buffer.push_back(v);
    heapify_up(heap_buffer.size() - 1);
  }

  void update_priority(Node* v, float new_priority) {
    auto orig_p = v->priority;
    v->priority = new_priority;
    if (new_priority < orig_p) {
      heapify_up(v->current_idx);
    } else {
      heapify_down(v->current_idx);
    }
  }

  void heapify_up(int i) {
    while (i > 0) {
      int parent = to_parent(i);
      if (heap_buffer[parent]->priority > heap_buffer[i]->priority) {
        swap(parent, i);
        i = parent;
      } else {
        break;
      }
    }
  }
  Node* pop() {
    Node* ret = heap_buffer[0];
    heap_buffer[0] = heap_buffer.back();
    heap_buffer.pop_back();
    heapify_down(0);
    return ret;
  }
  Node* remove(Node* n) {
    int i = n->current_idx;
    swap(i, heap_buffer.size() - 1);
    heap_buffer.pop_back();
    if (i < heap_buffer.size()) {
      heapify_down(i);
    }
    return n;
  }
  Node* top() {
    return heap_buffer[0];
  }
  Node* front() { return top(); }
  bool empty() {
    return heap_buffer.empty();
  }
};

struct DummyStruct{};

template<typename T>
struct DummyContainer {
  std::vector<T> data;
  // Forward vector constructor
  template<typename... Args>
  DummyContainer(Args&&... args) : data(std::forward<Args>(args)...) {}
  // Forward vector methods
  typename std::vector<T>::iterator begin() { return data.begin(); }
  typename std::vector<T>::iterator end() { return data.end(); }
  typename std::vector<T>::size_type size() const { return data.size(); }
  void resize(typename std::vector<T>::size_type count) { data.resize(count); }
  void clear() { data.clear(); }
  bool empty() const { return data.empty(); }
  void swap(DummyContainer& other) { data.swap(other.data); }
  T& operator[](typename std::vector<T>::size_type pos) { return data[pos]; }
  T& at(typename std::vector<T>::size_type pos) { return data.at(pos); }
  T& front() { return data.front(); }
  T& back() { return data.back(); }
  void push_back(const T& value) { data.push_back(value); }
  void pop_back() { data.pop_back(); }
};


template<>
struct DummyContainer<DummyStruct> {
  using T=DummyStruct;
  static T _place_holder;
  size_t cur_len = 0;
  DummyContainer() {}
  DummyContainer(size_t count) : cur_len(count) {}
  T* begin()        { return nullptr; }
  T* end()          { return nullptr; }
  size_t size() const     { return cur_len; }
  void resize(size_t count) { cur_len = count; }
  void clear() { cur_len = 0; }
  bool empty() const { return cur_len == 0; }
  void swap(DummyContainer& other) { std::swap(cur_len, other.cur_len); }
  T& operator[](size_t pos) { return _place_holder; }
  T& at(size_t pos) { return _place_holder; }
  T& front() { return _place_holder; }
  T& back() { return _place_holder; }
  void push_back(const T& value) { cur_len++; }
  void pop_back() { cur_len--; }
};

template<typename ELEM_T>
class Queue {
  DummyContainer<ELEM_T> queue_buffer;
  int start, stop;
  void extend() {
    auto orig_size = queue_buffer.size();
    queue_buffer.resize(queue_buffer.size() * 2);
    if (start < stop) { return; }
    std::copy(queue_buffer.begin(), queue_buffer.begin() + stop, queue_buffer.begin() + orig_size);
    stop = orig_size + stop;
  }
  int next(int a) { return (a + 1) % queue_buffer.size(); }
 public:
  Queue() : queue_buffer(10), start(0), stop(0) {}
  void clear() {
    start = stop;
  }
  bool empty() {
    return start == stop;
  }
  ELEM_T front() {
    CHECK(empty() == false);
    return queue_buffer[start];
  }
  void pop() {
    CHECK(empty() == false);
    start = next(start);
  }
  void push(ELEM_T task) {
    if (next(stop) == start) {
      extend();
    }
    queue_buffer[stop] = task;
    stop = next(stop);
  }
};

inline bool string_is_on(const std::string s) {
  return s == "on" || s == "true" || s == "1" || s == "ON" || s == "TRUE";
}