#pragma once
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>

enum ThreadType {
  kPythonMain = 0,
  kCacheLib,
  kPrefetch,
  kGPU,
};
enum EventType {
  kCustomEvent = 0,
};

class TraceEvent {
 public:
  int pid = 0, tid;
  uint64_t start_ts, stop_ts;
  EventType event_type;
  std::string event_name;
};

class TraceEventCollector {
  static int kNumMaxThread;
  std::vector<std::vector<TraceEvent>> event_list;
 public:
  TraceEventCollector() {
    event_list.resize(kNumMaxThread);
  }
  static TraceEventCollector& singleton() {
    static TraceEventCollector s;
    return s;
  }
  void add_event(TraceEvent e) {
    event_list[e.tid].push_back(e);
  }
  static std::string event_to_str(TraceEvent e, int id) {
    std::stringstream ss;
    event_to_str(ss, e, id);
    return ss.str();
  }
  static void event_to_str(std::ostream & os, TraceEvent e, int id) {
    if (id > 0) {
      os << ",";
    }
    os << "{"
       << "\"name\":" << "\"" << e.event_name << "\"" << ","
       << "\"ph\":"   << "\"" << "X" << "\"" << ","
       << "\"pid\":"  << e.pid << ","
       << "\"tid\":"  << e.tid << ","
       << "\"ts\":"   << e.start_ts  << ",";
    os << "\"dur\":"  << e.stop_ts - e.start_ts << ",";
    // os << "\"cat\":"  << "\"" << cat << "\"" << ",";
    os << "\"id\":"   << id;
    os << "}\n";
  }
  void dump_json_to_stream(std::ostream & os) {
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
  bool initialized = false;
  TraceEvent event;
 public:
  static bool globally_enabled;
  TraceEventGuard(int tid, std::string name) {
    init(tid, name);
  }
  TraceEventGuard() {}
  void init(int tid, std::string name) {
    event.event_name = name;
    event.tid = tid;
    event.start_ts = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    initialized = true;
  }
  void release() {
    event.stop_ts = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    TraceEventCollector::singleton().add_event(event);
    initialized = false;
  }
  ~TraceEventGuard() {
    if (initialized) release();
  }
};

#define TRACE_EVENT_GURAD(...) TraceEventGuard guard; { \
    if (TraceEventGuard::globally_enabled) { \
      guard.init(__VA_ARGS__); \
    } \
  }