#include <torch/extension.h>
#include "model_loader.hpp"
#include "prefetcher.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ModuleMeta, std::shared_ptr<ModuleMeta>>(m, "ModuleMeta")
    .def(py::init<int,int>())
    .def("init_param_list", &ModuleMeta::init_param_list)
  ;

  py::class_<Predictor, std::shared_ptr<Predictor>>(m, "Predictor")
    .def(py::init<>())
    .def("load_model", &Predictor::load_model)
    .def("predict", &Predictor::predict)
  ;

  py::class_<ModelLoader, std::shared_ptr<ModelLoader>>(m, "ModelLoader")
    .def(py::init<std::shared_ptr<ModuleMeta>>())
    .def("add_one_expert_param", static_cast<void (ModelLoader::*)(torch::Tensor, int, int, std::string)>(&ModelLoader::add_one_expert_param))
    .def("add_one_expert_param", static_cast<void (ModelLoader::*)(torch::Tensor, int, int, int)>(&ModelLoader::add_one_expert_param))
    .def("ref_one_expert_param", static_cast<torch::Tensor (ModelLoader::*)(int, int, std::string)>(&ModelLoader::ref_one_expert_param))
    .def("ref_one_expert_param", static_cast<torch::Tensor (ModelLoader::*)(int, int, int)>(&ModelLoader::ref_one_expert_param))
    // .def("add_one_expert", &ModelLoader::add_one_expert)
  ;

  py::class_<PrefetchMngr, std::shared_ptr<PrefetchMngr>>(m, "PrefetchMngr")
    .def(py::init<std::shared_ptr<ModuleMeta>, std::shared_ptr<ModelLoader>>())
    .def("launch_prefetch_thread", &PrefetchMngr::launch_prefetch_thread)
    .def("init_gpu_mem_buffer", &PrefetchMngr::init_gpu_mem_buffer)
    .def("wait_and_lock_expert", &PrefetchMngr::wait_and_lock_expert)
    .def("try_release_expert", &PrefetchMngr::try_release_expert)
    .def("try_release_expert_in_layer", &PrefetchMngr::try_release_expert_in_layer)
    .def("add_one_layer_task", &PrefetchMngr::add_one_layer_task)
    .def("preempt_one_layer", &PrefetchMngr::preempt_one_layer)
  ;

  py::class_<PrefetchEngine, std::shared_ptr<PrefetchEngine>>(m, "PrefetchEngine")
    .def(py::init<>())
    .def_readwrite("prefetch_worker", &PrefetchEngine::prefetch_worker)
    .def_readwrite("metas", &PrefetchEngine::metas)
    .def_readwrite("model_loader", &PrefetchEngine::model_loader)
    .def_readwrite("predictor", &PrefetchEngine::predictor)
    .def("init_prefetch_worker", &PrefetchEngine::init_prefetch_worker)
  ;
};