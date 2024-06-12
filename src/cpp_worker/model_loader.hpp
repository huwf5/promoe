#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include <torch/torch.h>
#include <cuda_runtime.h>

#include "utils.hpp"

class MemBuffer {
  // void* ptr;
  // size_t len;
  // void* ptr;
  // size_t len;
  torch::Tensor data;
 public:
  MemBuffer() {}
  MemBuffer(torch::Tensor t) : data(t) {}
  void set_tensor(torch::Tensor t) { data = t; }
  torch::Tensor get_tensor() { return data; }
  void* ptr() {return data.data_ptr();}
  size_t len() {return data.nbytes();}
};

class ExpertMemHanlder {
 public:
  std::vector<MemBuffer> mem_buffers;
};

class ExpertHandler {
 public:
  // std::shared_ptr<torch::nn::Module> expert_module;
  ExpertMemHanlder host_data;
  ExpertMemHanlder reference_to_model_param;
  ExpertMemHanlder* gpu_data = nullptr;
  int num_ready = 0;
  int layer_idx, expert_idx;
  // SpinLock lock;
  AtomicMultiStatusLock expert_status;
  cudaEvent_t event = nullptr;
  ExpertHandler() : expert_status() {}
  void wait() {
    expert_status.lock(kReady, kUsing);
  }
  std::string toString() const {
    return std::to_string(layer_idx) + "." + std::to_string(expert_idx);
  }
};

class ModelLoader {
  std::vector<ExpertHandler*> source_list;
  std::shared_ptr<ModuleMeta> metas;
 public:
  ModelLoader(std::shared_ptr<ModuleMeta> metas) : metas(metas) {
    source_list.resize(metas->num_layer * metas->num_expert, nullptr);
  }
  // void add_one_expert(std::shared_ptr<torch::nn::Module> expert, int layer_id, int expert_id) {
  //   auto params = expert->parameters();
  //   if (layer_id == 0 && expert_id == 0) {
  //     metas->num_per_expert_param = params.size();
  //     metas->param_name_list = expert->named_parameters().keys();
  //   }
  //   auto expert_handler = new ExpertHandler();
  //   // expert_handler->expert_module = expert;
  //   expert_handler->expert_idx = expert_id;
  //   expert_handler->layer_idx = layer_id;
  //   for (int i = 0; i < params.size(); i++) {
  //     expert_handler->host_data.mem_buffers[i].set_tensor(params[i].pin_memory());
  //   }
  //   expert->to(torch::Device("meta"));
  //   source_list[metas->squeeze_expert_idx(layer_id, expert_id)] = expert_handler;
  // }
  void add_one_expert_param(torch::Tensor param, int layer_id, int expert_id, std::string param_name) {
    return add_one_expert_param(param, layer_id, expert_id, metas->param_name_to_id[param_name]);
  }
  void add_one_expert_param(torch::Tensor param, int layer_id, int expert_id, int param_id);
  torch::Tensor ref_one_expert_param(int layer_id, int expert_id, std::string param_name) {
    return ref_one_expert_param(layer_id, expert_id, metas->param_name_to_id[param_name]);
  }
  torch::Tensor ref_one_expert_param(int layer_id, int expert_id, int param_id) {
    return source_list[metas->squeeze_expert_idx(layer_id, expert_id)]->reference_to_model_param.mem_buffers[param_id].get_tensor();
  }
  void pin_memory();
  ExpertHandler* get_source(int layer_id, int expert_id) {
    return source_list[metas->squeeze_expert_idx(layer_id, expert_id)];
  }
};
