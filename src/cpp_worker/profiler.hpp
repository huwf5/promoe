#pragma once
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <torch/torch.h>

#include "utils.hpp"

enum ThreadType {
  kPythonMain = 0,
  kHook,
  kFetchScheduler,
  kFetcher,
  kUnlocker,
  kCache,
  kPredictor,
  kGPU,
  kThreadTypeNum,
};
enum EventType {
  kCustomEvent = 0,
};


class Timer {
 public:
  uint64_t start;
  static uint64_t cur_ts_us() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }
  Timer() {
    start = cur_ts_us();
  }
  uint64_t dur_us() {
    return cur_ts_us() - start;
  }
};

class TraceEvent {
 public:
  int pid = 0, tid;
  uint64_t start_ts, stop_ts;
  EventType event_type;
  std::string event_name;
  char phase = 'X';
  std::unordered_map<std::string, std::string> *args = nullptr;
  void start() {
    start_ts = Timer::cur_ts_us();
  }
  void stop() {
    stop_ts = Timer::cur_ts_us();
  }
  ~TraceEvent() {
    if (args) { delete args; }
  }
};

class TraceEventCollector {
  std::vector<std::vector<TraceEvent>> event_list;
  bool meta_added = false;
 public:
  static bool globally_enabled;
  TraceEventCollector();
  static TraceEventCollector& singleton() {
    static TraceEventCollector s;
    return s;
  }
  static void reload_env() { 
    TraceEventCollector::globally_enabled = (getenv("SPARSE_CACHE_ENABLE_TRACE") != nullptr && string_is_on(getenv("SPARSE_CACHE_ENABLE_TRACE")));
  }
  void add_event(TraceEvent & e) {
    event_list[e.tid].push_back(e);
    e.args = nullptr;
  }
  static std::string event_to_str(TraceEvent & e, int id) {
    std::stringstream ss;
    event_to_str(ss, e, id);
    return ss.str();
  }
  static void event_to_str(std::ostream & os, TraceEvent & e, int id) {
    if (id > 0) {
      os << ",";
    }
    os << "{"
       << "\"ts\":"   << e.start_ts  << ","
       << "\"name\":" << "\"" << e.event_name << "\"" << ","
       << "\"ph\":"   << "\"" << e.phase << "\"" << ","
       << "\"pid\":"  << e.pid << ","
       << "\"tid\":"  << e.tid << ",";
    os << "\"dur\":"  << e.stop_ts - e.start_ts << ",";
    // os << "\"cat\":"  << "\"" << cat << "\"" << ",";
    os << "\"id\":"   << id;

    if (e.args) {
      os << ", \"args\" : {";
      bool first = true;
      for (auto & [k,v] : *(e.args)) {
        if (first) {
          first = false;
        } else {
          os << ",";
        }
        os << " \"" << k << "\" : \"" << v << "\" ";
      }
      os << "}";
    }

    os << "}\n";
  }
  void add_meta_event();
  void dump_json_to_stream(std::ostream & os) {
    if (meta_added == false) {
      add_meta_event();
      meta_added = true;
    }
    os << "{ \"traceEvents\" : [\n";
    int id = 0;
    for (auto & v : event_list) {
      for (auto & e : v) {
        event_to_str(os, e, id++);
      }
    }
    os << "]}\n";
  }
  std::string dump_json_to_string() {
    std::stringstream ss;
    dump_json_to_stream(ss);
    return ss.str();
  }
};


class TraceEventGuard {
 protected:
  bool initialized = false;
 public:
  TraceEvent event;
  TraceEventGuard(int tid, std::string name, char phase = 'X') {
    init(tid, name, phase);
  }
  TraceEventGuard() {}
  void init(int tid, std::string name, char phase = 'X') {
    event.event_name = name;
    event.tid = tid;
    event.phase = phase;
    event.start();
    initialized = true;
  }
  void release() {
    event.stop();
    TraceEventCollector::singleton().add_event(event);
    initialized = false;
  }
  ~TraceEventGuard() {
    if (initialized) release();
  }
};

// class TraceEventGuardWithArg : public TraceEventGuard<TraceEventWithArgs> {
//  public:
//   using TraceEventGuard::TraceEventGuard;
//   void init(int tid, std::string name) {
//     event.event_name = name;
//     event.tid = tid;
//     event.start();
//     initialized = true;
//   }
//   void release() {
//     event.stop();
//     TraceEventCollector::singleton().add_event(event);
//     initialized = false;
//   }
//   ~TraceEventGuardWithArg() {
//     if (initialized) release();
//   }
// };

#define TRACE_EVENT_GURAD(tid, name) TraceEventGuard guard; { \
    if (TraceEventCollector::globally_enabled) { \
      guard.init(tid, name); \
    } \
  }

#define TRACE_EVENT_GURAD_NAME(tid, name, guard_name) TraceEventGuard guard_name; { \
    if (TraceEventCollector::globally_enabled) { \
      guard_name.init(tid, name); \
    } \
  }

#define TRACE_EVENT_GURAD_WITH_ARGS(tid, name, phase, arg_var, CODE_BLOCK) TraceEventGuard guard; { \
    if (TraceEventCollector::globally_enabled) { \
      guard.init(tid, name, phase); \
      guard.event.args = new std::unordered_map<std::string, std::string>(); \
      auto & arg_var = *guard.event.args; \
      { CODE_BLOCK } \
    } \
  }

class CacheStatistics {
  struct HitMissCnt {
    int hit = 0, miss = 0;
    int cnt() { return hit + miss; }
    float hit_rate() { return (float)hit / (cnt() == 0 ? 1 : 0); }
  };
  std::vector<HitMissCnt> per_iter_per_layer_cnts;
  std::vector<std::function<void(CacheStatistics*)>> reporters;
 public:
  CacheStatistics() {
    per_iter_per_layer_cnts.reserve(10000);
  }
  ~CacheStatistics() {
    for (auto & r : reporters) {
      r(this);
    }
  }
  void add_reporter(std::function<void(CacheStatistics*)> reporter) {
    reporters.push_back(reporter);
  }
  void forward() {
    per_iter_per_layer_cnts.push_back(HitMissCnt());
  }
  void hit()         { per_iter_per_layer_cnts.back().hit++; }
  void hit(int cnt)  { per_iter_per_layer_cnts.back().hit+=cnt; }
  void miss()        { per_iter_per_layer_cnts.back().miss++; }
  void miss(int cnt) { per_iter_per_layer_cnts.back().miss+=cnt; }
  torch::Tensor to_tensor() {
    torch::Tensor ret = torch::zeros({static_cast<long>(per_iter_per_layer_cnts.size()), 2});
    for (int i = 0; i < per_iter_per_layer_cnts.size(); i++) {
      ret[i][0] = per_iter_per_layer_cnts[i].hit;
      ret[i][1] = per_iter_per_layer_cnts[i].miss;
    }
    return ret;
  }
  torch::Tensor dump_average(int prefill_cnt_threshold) {
    auto tensor = to_tensor();
    // remove iteration of prefill
    tensor = tensor.index({tensor.sum(1) <= prefill_cnt_threshold});
    tensor = tensor.mean(0);
    return tensor;
  }
  torch::Tensor dump_average_per_layer(int num_layer, int prefill_cnt_threshold) {
    CHECK(per_iter_per_layer_cnts.size() % num_layer == 0);
    auto tensor = to_tensor();
    tensor = tensor.index({tensor.sum(1) <= prefill_cnt_threshold});
    tensor = tensor.reshape({-1, num_layer, 2});
    tensor = tensor.mean(0);
    return tensor;
  }
};

class TimeProfiler : public std::enable_shared_from_this<TimeProfiler> {
  struct MetricMeta {
    bool is_cum = true;
    bool should_drop_last = false;
  };
  std::vector<std::vector<uint64_t>> buffer;
  std::vector<std::function<void(TimeProfiler*)>> reporters;
  std::vector<MetricMeta> metric_metas;
 public:
  enum TimeType {
    kModelForward = 0,   // per forward
    kCntActivatedExpert, // per forward
    kHitCnt,  // per forward
    kMissCnt, // per forward
    kReadyCnt,   // per forward
    kUnreadyCnt, // per forward
    kPrefetchHitCnt,  // per forward
    kPrefetchMissCnt, // per forward
    kSeqLen, // per forward
    kNumTimeType,
  };
  TimeProfiler() {
    metric_metas.resize(kNumTimeType);
    metric_metas[ kModelForward       ].is_cum = false; metric_metas[ kModelForward       ].should_drop_last = false;
    metric_metas[ kCntActivatedExpert ].is_cum = true; metric_metas[ kCntActivatedExpert ].should_drop_last = true;
    metric_metas[ kHitCnt             ].is_cum = true; metric_metas[ kHitCnt             ].should_drop_last = true;
    metric_metas[ kMissCnt            ].is_cum = true; metric_metas[ kMissCnt            ].should_drop_last = true;
    metric_metas[ kReadyCnt           ].is_cum = true; metric_metas[ kReadyCnt           ].should_drop_last = true;
    metric_metas[ kUnreadyCnt         ].is_cum = true; metric_metas[ kUnreadyCnt         ].should_drop_last = true;
    metric_metas[ kPrefetchHitCnt     ].is_cum = true; metric_metas[ kPrefetchHitCnt     ].should_drop_last = true;
    metric_metas[ kPrefetchMissCnt    ].is_cum = true; metric_metas[ kPrefetchMissCnt    ].should_drop_last = true;
    metric_metas[ kSeqLen             ].is_cum = false; metric_metas[ kSeqLen             ].should_drop_last = false;
    buffer.resize(kNumTimeType);
    for (int i = 0; i < kNumTimeType; i++) {
      auto &b = buffer[i];
      b.reserve(10000);
      if (metric_metas[i].is_cum) {
        b.push_back(0);
      }
    }
  }
  ~TimeProfiler() {
    for (auto & r : reporters) {
      r(this);
    }
  }
  void add_reporter(std::function<void(TimeProfiler*)> reporter) {
    reporters.push_back(reporter);
  }
  void push(TimeType type, uint64_t dur) {
    buffer[type].push_back(dur);
  }
  void add(TimeType type, uint64_t dur) {
    buffer[type].back() += dur;
  }
  torch::Tensor to_tensor(TimeType type) {
    auto ret = torch::from_blob(buffer[type].data(), {static_cast<long>(buffer[type].size())}, torch::kLong).clone();
    if (metric_metas[type].should_drop_last) {
      ret = ret.index({torch::indexing::Slice(0, -1)});
    }
    return ret;
  }
};

class TimerGuard {
 protected:
  bool initialized = false;
  uint64_t start, stop;
  TimeProfiler::TimeType type;
  TimeProfiler* profiler;
 public:
  TimerGuard(TimeProfiler* profiler) : profiler(profiler) {}
  TimerGuard(TimeProfiler* profiler, TimeProfiler::TimeType t) : profiler(profiler) {
    this->init(t);
  }
  inline void init(TimeProfiler::TimeType t) {
    this->initialized = true;
    this->type = t;
    this->start = Timer::cur_ts_us();
  }
  void release() {
    this->stop = Timer::cur_ts_us();
    this->profiler->push(type, stop-start);
    this->initialized = false;
  }
  ~TimerGuard() {
    if (initialized) release();
  }
};

class PrecisionProfiler {
 public:
  int decode_expert_per_token = 0;
  class LayerInfo {
   public:
    int layer_id;
    std::vector<int64_t> experts;
    LayerInfo() {}
    LayerInfo(int layer_id) : layer_id(layer_id) {}
    LayerInfo(int layer_id, const int64_t* experts, size_t num_experts) : layer_id(layer_id), experts(experts, experts + num_experts) {}
    void add(int64_t e) { experts.push_back(e); }
    int intersect(const LayerInfo & other) {
      std::unordered_set<int64_t> s(experts.begin(), experts.end());
      int cnt = 0;
      for (auto e : other.experts) {
        if (s.count(e)) {
          cnt++;
        }
      }
      return cnt;
    }
  };
  int previous_layer_id = -1;
  std::vector<LayerInfo> predicted_experts;
  std::vector<LayerInfo> activated_experts;
  void record_predicted_experts(int layer_id, const int64_t* experts, size_t num_experts) {
    predicted_experts.push_back(LayerInfo(layer_id, experts, num_experts));
  }
  void record_activated_experts(int layer_id, const int64_t *experts, size_t num_experts);
  void record_activated_experts_by_append(int layer_id, int64_t expert) {
    if (layer_id != previous_layer_id) {
      activated_experts.push_back(LayerInfo(layer_id));
      previous_layer_id = layer_id;
    }
    activated_experts.back().add(expert);
  }
  void report();
  ~PrecisionProfiler() {
    report();
  }
};