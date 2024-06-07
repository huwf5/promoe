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
    .def_readwrite("num_layer", &ModuleMeta::num_layer)
    .def_readwrite("num_expert", &ModuleMeta::num_expert)
    .def_readwrite("num_per_expert_param", &ModuleMeta::num_per_expert_param)
    .def_readwrite("num_predict_expert_per_layer", &ModuleMeta::num_predict_expert_per_layer)
    .def_readwrite("max_prefetch_layer_distance", &ModuleMeta::max_prefetch_layer_distance)
    .def_readwrite("per_layer_cache", &ModuleMeta::per_layer_cache)
    .def("init_param_list", &ModuleMeta::init_param_list)
  ;

  py::class_<Predictor, std::shared_ptr<Predictor>>(m, "Predictor")
    .def(py::init<std::shared_ptr<ModuleMeta>>())
    .def("load_model", &Predictor::load_model)
    .def("predict", &Predictor::predict)
    // .def("init_expert_access_buffer", &Predictor::init_expert_access_buffer)
    .def("clear_access_buffer", &Predictor::clear_access_buffer)
    .def("add_one_layer", static_cast<void (Predictor::*)(int, int64_t*, size_t)>(&Predictor::add_one_layer))
    .def("add_one_layer", static_cast<void (Predictor::*)(int, torch::Tensor)>(&Predictor::add_one_layer))
  ;

  py::class_<ModelLoader, std::shared_ptr<ModelLoader>>(m, "ModelLoader")
    .def(py::init<std::shared_ptr<ModuleMeta>>())
    .def("pin_memory", &ModelLoader::pin_memory)
    .def("add_one_expert_param", static_cast<void (ModelLoader::*)(torch::Tensor, int, int, std::string)>(&ModelLoader::add_one_expert_param))
    .def("add_one_expert_param", static_cast<void (ModelLoader::*)(torch::Tensor, int, int, int)>(&ModelLoader::add_one_expert_param))
    .def("ref_one_expert_param", static_cast<torch::Tensor (ModelLoader::*)(int, int, std::string)>(&ModelLoader::ref_one_expert_param))
    .def("ref_one_expert_param", static_cast<torch::Tensor (ModelLoader::*)(int, int, int)>(&ModelLoader::ref_one_expert_param))
    // .def("add_one_expert", &ModelLoader::add_one_expert)
  ;

  py::class_<PrefetchMngr, std::shared_ptr<PrefetchMngr>>(m, "PrefetchMngr")
    .def(py::init<std::shared_ptr<ModuleMeta>, std::shared_ptr<ModelLoader>, std::shared_ptr<Predictor>>())
    .def("launch_thread", &PrefetchMngr::launch_thread)
    .def("init_gpu_mem_buffer", &PrefetchMngr::init_gpu_mem_buffer)
    .def("wait_expert", &PrefetchMngr::wait_expert)
    .def("mark_expert_using", &PrefetchMngr::mark_expert_using)
    // .def("try_release_expert", &PrefetchMngr::try_release_expert)
    // .def("try_release_expert_in_layer", &PrefetchMngr::try_release_expert_in_layer)
    // .def("add_one_layer_task", &PrefetchMngr::add_one_layer_task)
    .def("preempt_and_launch_one_layer", &PrefetchMngr::preempt_and_launch_one_layer)
    .def("record_then_predict_and_prefetch", &PrefetchMngr::record_then_predict_and_prefetch)
  ;

  py::class_<PrefetchEngine, std::shared_ptr<PrefetchEngine>>(m, "PrefetchEngine")
    .def(py::init<>())
    .def_readwrite("prefetch_worker", &PrefetchEngine::prefetch_worker)
    .def_readwrite("metas", &PrefetchEngine::metas)
    .def_readwrite("model_loader", &PrefetchEngine::model_loader)
    .def_readwrite("predictor", &PrefetchEngine::predictor)
    .def("init_prefetch_worker", &PrefetchEngine::init_prefetch_worker)
  ;

  py::class_<TraceEventGuard, std::shared_ptr<TraceEventGuard>>(m, "TraceEventGuard")
    .def(py::init<>())
    .def("init", &TraceEventGuard::init, "docstring", py::arg(), py::arg(), py::arg("phase")='X')
    .def("release", &TraceEventGuard::release)
  ;

  m.def("dump_trace_event_collector_singleton", &dump_trace_event_collector_singleton);

  py::enum_<ThreadType>(m, "ThreadType")
    .value("kPythonMain", ThreadType::kPythonMain)
    .value("kHook", ThreadType::kHook)
    .value("kPrefetch", ThreadType::kPrefetch)
    .value("kGPU", ThreadType::kGPU)
    .export_values();
};