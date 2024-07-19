#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include <torch/torch.h>
#include <cuda_runtime.h>

#include "utils.hpp"

class MemBuffer {
  torch::Tensor data;
 public:
  MemBuffer() {}
  MemBuffer(torch::Tensor t) : data(t) {}
  torch::Tensor get_tensor() { return data; }
  void* ptr() {return data.data_ptr();}
  size_t len() {return data.nbytes();}
  void map_to(MemBuffer& physical) {
    data.set_(physical.data, 0, physical.data.sizes(), physical.data.strides());
  }
  void make_logical(torch::IntArrayRef shape, torch::TensorOptions options) {
    data = torch::empty({0}, options);
  }
  void make_logical(torch::TensorOptions options) {
    make_logical({0}, options);
  }
  void pin_memory() {
    data = data.pin_memory();
  }
  void allocate_like(MemBuffer& other, torch::TensorOptions options) {
    data = torch::empty_like(other.data, options);
  }
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
    return std::to_string(layer_idx) + "." + std::to_string(expert_idx) + "(" + std::to_string(num_ready) + ")";
  }
};

class ModelLoader {
  std::vector<ExpertHandler*> source_list;
  std::shared_ptr<ModuleMeta> metas;
 public:
  ModelLoader(std::shared_ptr<ModuleMeta> metas) : metas(metas) {
    source_list.resize(metas->num_layer * metas->num_expert, nullptr);
  }
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
