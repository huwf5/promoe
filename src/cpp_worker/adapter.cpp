#include <torch/extension.h>
#include "model_loader.hpp"
#include "prefetcher.hpp"
#include "profiler.hpp"

std::string dump_trace_event_collector_singleton() {
  return TraceEventCollector::singleton().dump_json_to_string();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ModuleMeta, std::shared_ptr<ModuleMeta>>(m, "ModuleMeta")
    .def(py::init<int,int>())
    .def_readwrite("num_layer",                    &ModuleMeta::num_layer)
    .def_readwrite("num_expert",                   &ModuleMeta::num_expert)
    .def_readwrite("num_per_expert_param",         &ModuleMeta::num_per_expert_param)
    .def_readwrite("num_predict_expert_per_layer", &ModuleMeta::num_predict_expert_per_layer)
    .def_readwrite("num_expert_per_token",         &ModuleMeta::num_expert_per_token)
    .def_readwrite("max_prefetch_layer_distance",  &ModuleMeta::max_prefetch_layer_distance)
    .def_readwrite("per_layer_cache",              &ModuleMeta::per_layer_cache)
    .def_readwrite("cache_policy",                 &ModuleMeta::cache_policy)
    .def_readwrite("reorder_experts",              &ModuleMeta::reorder_experts)
    .def_readwrite("promote_hit_in_prefetch",      &ModuleMeta::promote_hit_in_prefetch)
    .def_readwrite("early_preempt",                &ModuleMeta::early_preempt)
    .def_readwrite("predict_input_mode",           &ModuleMeta::predict_input_mode)
    .def_readwrite("layer_predict_interval",       &ModuleMeta::layer_predict_interval)
    .def_readwrite("layer_predict_max_window",     &ModuleMeta::layer_predict_max_window)
    .def_readwrite("layer_predict_replace_first_input_with_last_output",     &ModuleMeta::layer_predict_replace_first_input_with_last_output)
    .def("init_param_list", &ModuleMeta::init_param_list)
  ;

  py::class_<Predictor, std::shared_ptr<Predictor>>(m, "Predictor")
    .def(py::init<std::shared_ptr<ModuleMeta>>())
    .def("load_model",          &Predictor::load_model)
    .def("predict",             &Predictor::predict)
    // .def("clear_access_buffer", &Predictor::clear_access_buffer)
    .def("add_one_layer", static_cast<void (Predictor::*)(int, int64_t*, size_t)>(&Predictor::add_one_layer))
    .def("add_one_layer", static_cast<void (Predictor::*)(int, torch::Tensor)>(&Predictor::add_one_layer))
  ;

  py::class_<ModelLoader, std::shared_ptr<ModelLoader>>(m, "ModelLoader")
    .def(py::init<std::shared_ptr<ModuleMeta>>())
    .def("pin_memory", &ModelLoader::pin_memory)
    .def("add_one_expert_param", static_cast<void (ModelLoader::*)(torch::Tensor, int, int, std::string)>(&ModelLoader::add_one_expert_param))
    .def("add_one_expert_param", static_cast<void (ModelLoader::*)(torch::Tensor, int, int, int)>(&ModelLoader::add_one_expert_param))
    .def("build_logical_expert_param", static_cast<void (ModelLoader::*)()>(&ModelLoader::build_logical_expert_param))
    .def("ref_one_expert_param", static_cast<torch::Tensor (ModelLoader::*)(int, int, std::string)>(&ModelLoader::ref_one_expert_param))
    .def("ref_one_expert_param", static_cast<torch::Tensor (ModelLoader::*)(int, int, int)>(&ModelLoader::ref_one_expert_param))
    // .def("add_one_expert", &ModelLoader::add_one_expert)
  ;

  py::class_<CacheMngr, std::shared_ptr<CacheMngr>>(m, "CacheMngr")
    .def("set_cur_seq",       &CacheMngr::set_cur_seq)
    .def_readwrite("cache_oracle", &CacheMngr::cache_oracle)
  ;

  py::class_<CacheOracle, std::shared_ptr<CacheOracle>>(m, "CacheOracle")
    .def("load_from_file",       &CacheOracle::load_from_file)
  ;

  py::class_<PrefetchMngr, std::shared_ptr<PrefetchMngr>>(m, "PrefetchMngr")
    .def(py::init<std::shared_ptr<ModuleMeta>, std::shared_ptr<ModelLoader>, std::shared_ptr<Predictor>>())
    .def("launch_thread",       &PrefetchMngr::launch_thread)
    .def("init_gpu_mem_buffer", &PrefetchMngr::init_gpu_mem_buffer)
    .def("reload_env",          &PrefetchMngr::reload_env)
    .def("report_one_expert",   &PrefetchMngr::report_one_expert)
    .def("one_expert_done",     &PrefetchMngr::one_expert_done)
    .def("report_one_layer",    &PrefetchMngr::report_one_layer)
    .def("one_moe_layer_done",  &PrefetchMngr::one_moe_layer_done)
    .def("report_moe_attn_logits",    &PrefetchMngr::report_moe_attn_logits)
    .def("report_moe_layer_logits",   &PrefetchMngr::report_moe_layer_logits)
    .def("build_timer",         &PrefetchMngr::build_timer)
    .def_readwrite("cache_stats", &PrefetchMngr::cache_stats)
    .def_readwrite("profiler",    &PrefetchMngr::profiler)
    .def_readwrite("cache",       &PrefetchMngr::cache)
  ;

  py::class_<TraceEventGuard, std::shared_ptr<TraceEventGuard>>(m, "TraceEventGuard")
    .def(py::init<>())
    .def("init", &TraceEventGuard::init, "docstring", py::arg(), py::arg(), py::arg("phase")='X')
    .def("release", &TraceEventGuard::release)
  ;

  py::class_<CacheStatistics, std::shared_ptr<CacheStatistics>>(m, "CacheStatistics")
    .def(py::init<>())
    .def("to_tensor",              &CacheStatistics::to_tensor)
    .def("dump_average",           &CacheStatistics::dump_average)
    .def("dump_average_per_layer", &CacheStatistics::dump_average_per_layer)
  ;

  py::class_<TimerGuard>(m, "TimerGuard")
    .def("init",    &TimerGuard::init)
    .def("release", &TimerGuard::release)
  ;

  m.def("dump_trace_event_collector_singleton", &dump_trace_event_collector_singleton);

  py::enum_<ThreadType>(m, "ThreadType")
    .value("kPythonMain",     ThreadType::kPythonMain)
    .value("kHook",           ThreadType::kHook)
    .value("kFetchScheduler", ThreadType::kFetchScheduler)
    .value("kGPU",            ThreadType::kGPU)
    .export_values();

  py::enum_<PredictInputMode>(m, "PredictInputMode")
    .value("kOneToken",                PredictInputMode::kOneToken)
    .value("kDecodeCumsum",            PredictInputMode::kDecodeCumsum)
    .value("kLastUseDistance",         PredictInputMode::kLastUseDistance)
    .value("kWeighedDecodeCumsum",     PredictInputMode::kWeighedDecodeCumsum)
    .value("kFirstMoeAttnInputLogits", PredictInputMode::kFirstMoeAttnInputLogits)
    .value("kMoeAttnInputLogits",      PredictInputMode::kMoeAttnInputLogits)
    .value("kMoeLayerLogits",          PredictInputMode::kMoeLayerLogits)
    .export_values();

  py::enum_<TimeProfiler::TimeType>(m, "TimeType")
    .value("kModelForward", TimeProfiler::TimeType::kModelForward)
    .export_values();
};