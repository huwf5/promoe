#include "profiler.hpp"
#include "logging.hpp"

bool TraceEventCollector::globally_enabled = false;
TraceEventCollector::TraceEventCollector() {
  reload_env();
  event_list.resize(kThreadTypeNum);
}
void TraceEventCollector::add_meta_event() {
  { TRACE_EVENT_GURAD_WITH_ARGS(kPythonMain,     "thread_name", 'M', arg_var, { arg_var["name"] = "kPythonMain";     }); }
  { TRACE_EVENT_GURAD_WITH_ARGS(kHook,           "thread_name", 'M', arg_var, { arg_var["name"] = "kHook";           }); }
  { TRACE_EVENT_GURAD_WITH_ARGS(kFetchScheduler, "thread_name", 'M', arg_var, { arg_var["name"] = "kFetchScheduler"; }); }
  { TRACE_EVENT_GURAD_WITH_ARGS(kFetcher,        "thread_name", 'M', arg_var, { arg_var["name"] = "kFetcher";        }); }
  { TRACE_EVENT_GURAD_WITH_ARGS(kUnlocker,       "thread_name", 'M', arg_var, { arg_var["name"] = "kUnlocker";       }); }
  { TRACE_EVENT_GURAD_WITH_ARGS(kCache,          "thread_name", 'M', arg_var, { arg_var["name"] = "kCache";          }); }
  { TRACE_EVENT_GURAD_WITH_ARGS(kPredictor,      "thread_name", 'M', arg_var, { arg_var["name"] = "kPredictor";      }); }
  { TRACE_EVENT_GURAD_WITH_ARGS(kGPU,            "thread_name", 'M', arg_var, { arg_var["name"] = "kGPU";            }); }
}
void PrecisionProfiler::report() {
  if (activated_experts.empty() || predicted_experts.empty()) {
    return;
  }
  std::map<int, std::vector<double>> per_layer_rates;
  size_t actual_idx = 0;
  size_t predicted_idx = 0;
  while (actual_idx < activated_experts.size() && predicted_idx < predicted_experts.size()) {
    auto &a = activated_experts[actual_idx];
    auto &p = predicted_experts[predicted_idx];
    if (a.layer_id < p.layer_id) {
      actual_idx++;
      continue;
    }
    if (a.layer_id > p.layer_id) {
      predicted_idx++;
      continue;
    }
    // std::cerr << "layer " << a.layer_id << " "
    //           << "intersect " << a.intersect(p) << " / " << a.experts.size()
    //           << " / " << p.experts.size() << ", rate "
    //           << (float)a.intersect(p) / p.experts.size() << "\n";
    if (per_layer_rates.count(a.layer_id) == 0) {
      per_layer_rates[a.layer_id] = std::vector<double>();
    }
    if (a.experts.size() != decode_expert_per_token) {
      actual_idx++;
      continue;
    }
    if (p.experts.size() == 0) {
      actual_idx++;
      predicted_idx++;
      continue;
    }
    per_layer_rates[a.layer_id].push_back((float)a.intersect(p) / p.experts.size());
    actual_idx++;
    predicted_idx++;
  }
  for (auto &[layer_id, rates] : per_layer_rates) {
    if (rates.empty()) {
      continue;
    }
    double sum = 0;
    for (auto r : rates) {
      sum += r;
    }
    std::cerr << "layer " << layer_id << " average rate " << sum / rates.size() << "\n";
  }
}
void PrecisionProfiler::record_activated_experts(int layer_id, const int64_t *experts, size_t num_experts) {
  activated_experts.push_back(LayerInfo(layer_id, experts, num_experts));
}
