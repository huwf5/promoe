#include "profiler.hpp"

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
