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
  std::map<int, std::vector<double>> per_layer_rates;
  for (int i = 0; i < activated_experts.size(); i++) {
    auto &a = activated_experts[i], &p = predicted_experts[i];
    if (a.layer_id != p.layer_id) {
      std::cerr << "layer id mismatch " << a.layer_id << " " << p.layer_id << "\n";
      break;
    }
    // std::cerr << "layer " << a.layer_id << " "
    //           << "intersect " << a.intersect(p) << " / " << a.experts.size()
    //           << " / " << p.experts.size() << ", rate "
    //           << (float)a.intersect(p) / p.experts.size() << "\n";
    if (per_layer_rates.count(a.layer_id) == 0) {
      per_layer_rates[a.layer_id] = std::vector<double>();
    }
    if (a.experts.size() != decode_expert_per_token) {
      continue;
    }
    if (p.experts.size() == 0) {
      continue;
    }
    per_layer_rates[a.layer_id].push_back((float)a.intersect(p) / p.experts.size());
  }
  for (auto &[layer_id, rates] : per_layer_rates) {
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
