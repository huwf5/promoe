#pragma once
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <torch/torch.h>

enum ThreadType {
  kPythonMain = 0,
  kHook,
  kPrefetch,
  kFetcher,
  kCache,
  kPredict,
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