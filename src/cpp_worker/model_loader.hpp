#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "utils.hpp"

#define MEM_WRAP_USE_CU_DRIVER
// #define MEM_WRAP_USE_TORCH_TENSOR

#ifdef MEM_WRAP_USE_TORCH_TENSOR
#define LogicalMemHandler  LogicalMemHandlerTensor
#define PhysicalMemHandler PhysicalMemHandlerTensor
#endif
#ifdef MEM_WRAP_USE_CU_DRIVER
#define LogicalMemHandler  LogicalMemHandlerCUDriver
#define PhysicalMemHandler PhysicalMemHandlerCUDriver
#endif

// class MemBuffer {
//   torch::Tensor data;
//  public:
//   MemBuffer() {}
//   MemBuffer(torch::Tensor t) : data(t) {}
//   torch::Tensor get_tensor() { return data; }
//   void* ptr() {return data.data_ptr();}
//   size_t len() {return data.nbytes();}
//   void map_to(MemBuffer& physical) {
//     data.set_(physical.data, 0, physical.data.sizes(), physical.data.strides());
//   }
//   void make_logical(torch::IntArrayRef shape, torch::TensorOptions options) {
//     data = torch::empty({0}, options);
//   }
//   void make_logical(torch::TensorOptions options) {
//     make_logical({0}, options);
//   }
//   void pin_memory() {
//     data = data.pin_memory();
//   }
//   void allocate_like(MemBuffer& other, torch::TensorOptions options) {
//     data = torch::empty_like(other.data, options);
//   }
// };

// class ExpertMemHanlder {
//  public:
//   std::vector<MemBuffer> mem_buffers;
// };

class PhysicalMemHandlerTensor;
class PhysicalMemHandlerCUDriver;
class HostMemWrapper {
  friend class PhysicalMemHandlerTensor;
  friend class LogicalMemHandlerTensor;
  friend class PhysicalMemHandlerCUDriver;
  friend class LogicalMemHandlerCUDriver;
  friend class ExpertParamWrapper;
  friend class ExpertMemHanlder;
 public:
  torch::Tensor data;
  HostMemWrapper() {}
  HostMemWrapper(torch::Tensor t) : data(t) {}
  void* ptr() {return data.data_ptr();}
  size_t len() {return data.nbytes();}
  void pin_memory() {
    data = data.pin_memory();
  }
};

class HostExpertMemHanlder {
 public:
  std::vector<HostMemWrapper> mem_buffers;
  HostExpertMemHanlder() {}
  HostExpertMemHanlder(int num) : mem_buffers(num) {}
  void pin_memory() {
    for (auto & m : mem_buffers) { m.pin_memory(); }
  }
  void*  ptr(int idx) { return mem_buffers[idx].ptr(); }
  size_t len(int idx) { return mem_buffers[idx].len(); }
  void set(int idx, torch::Tensor &param);
};
class MemMngrCtx;

class LogicalMemHandlerTensor {
  torch::Tensor data;
  friend class PhysicalMemHandlerTensor;
 public:
  LogicalMemHandlerTensor() {}
  torch::Tensor get_tensor() { return data; }
  void* ptr() {return data.data_ptr();}
  size_t len() {return data.nbytes();}
  void map_to(PhysicalMemHandlerTensor &physical, MemMngrCtx* ctx);
  void unmap();
  void make_logical(torch::IntArrayRef shape, torch::TensorOptions options, MemMngrCtx* ctx);
  void make_logical(torch::TensorOptions options, MemMngrCtx* ctx) { make_logical({0}, options, ctx); }
};

class PhysicalMemHandlerTensor {
  torch::Tensor data;
  friend class LogicalMemHandlerTensor;
  friend class MemMngrCtx;
  size_t nbytes;
 public:
  LogicalMemHandlerTensor logical_ptr;
  PhysicalMemHandlerTensor() {}
  void allocate_like(HostMemWrapper &other, torch::TensorOptions options, MemMngrCtx* ctx);
  void allocate_like(HostMemWrapper &other, MemMngrCtx* ctx);
  void allocate(size_t nbytes, MemMngrCtx *ctx);
};

class LogicalMemHandlerCUDriver {
  torch::Tensor data;

  CUdeviceptr device_ptr;
  size_t nbytes;
  friend class PhysicalMemHandlerCUDriver;
 public:
  LogicalMemHandlerCUDriver() {}
  torch::Tensor get_tensor() { return data; }
  void* ptr() {return data.data_ptr();}
  size_t len() {return data.nbytes();}
  void map_to(PhysicalMemHandlerCUDriver &physical, MemMngrCtx* ctx);
  void unmap();
  void make_logical(torch::IntArrayRef shape, torch::TensorOptions options, MemMngrCtx* ctx);
  void make_logical(torch::TensorOptions options, MemMngrCtx* ctx) { make_logical({0}, options, ctx); }
};

class PhysicalMemHandlerCUDriver {
  friend class LogicalMemHandlerCUDriver;
  friend class MemMngrCtx;
  CUmemGenericAllocationHandle handle;
  size_t nbytes;
 public:
  LogicalMemHandlerCUDriver logical_ptr;
  PhysicalMemHandlerCUDriver() {}
  // PhysicalMemHandlerCUDriver(torch::Tensor t) : data(t) {}
  // void* ptr() {return data.data_ptr();}
  // size_t len() {return data.nbytes();}
  void allocate_like(HostMemWrapper &other, torch::TensorOptions options, MemMngrCtx* ctx);
  void allocate_like(HostMemWrapper &other, MemMngrCtx* ctx);
  void allocate(size_t nbytes, MemMngrCtx *ctx);
};

class MemMngrCtx {
 public:
  PhysicalMemHandlerCUDriver dummy_mem;
  CUmemAllocationProp prop{};
  size_t granularity = 0;
  CUmemAccessDesc accessDesc = {};
  int device_id = 0;

  MemMngrCtx();
};

class ExpertMemHanlder {
  std::vector<PhysicalMemHandler> mem_buffers;
  friend class ExpertParamWrapper;
 public:
  void allocate_like(HostExpertMemHanlder& other, MemMngrCtx* ctx) {
    this->mem_buffers.resize(other.mem_buffers.size());
    for (int j = 0; j < other.mem_buffers.size(); j++) {
      this->mem_buffers[j].allocate_like(other.mem_buffers[j], ctx);
    }
  }
  void* ptr(int idx) { return mem_buffers[idx].logical_ptr.ptr(); }
  // size_t len(int idx) { return mem_buffers[idx].logical_ptr.len(); }
};

class ExpertParamWrapper {
  std::vector<LogicalMemHandler> mem_buffers;
 public:
  ExpertParamWrapper() {}
  ExpertParamWrapper(int num) : mem_buffers(num) {}
  void unmap();
  void map_to(ExpertMemHanlder* physical, MemMngrCtx* ctx) {
    for (int i = 0; i < mem_buffers.size(); i++) {
      mem_buffers[i].map_to(physical->mem_buffers[i], ctx);
    }
  }
  void make_logical(HostExpertMemHanlder& other, MemMngrCtx* ctx) {
    mem_buffers.resize(other.mem_buffers.size());
    for (int i = 0; i < other.mem_buffers.size(); i++) {
      auto options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other.mem_buffers[i].data.dtype());
      mem_buffers[i].make_logical(other.mem_buffers[i].data.sizes(), options, ctx);
    }
  }
  torch::Tensor get_tensor(int idx) { return mem_buffers[idx].get_tensor(); }
};

class ExpertHandler {
 public:
  // std::shared_ptr<torch::nn::Module> expert_module;
  HostExpertMemHanlder host_data;
  ExpertParamWrapper reference_to_model_param;
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
  std::shared_ptr<MemMngrCtx> mem_mngr_ctx;
  ModelLoader(std::shared_ptr<ModuleMeta> metas);
  void add_one_expert_param(torch::Tensor param, int layer_id, int expert_id, std::string param_name) {
    return add_one_expert_param(param, layer_id, expert_id, metas->param_name_to_id[param_name]);
  }
  void add_one_expert_param(torch::Tensor param, int layer_id, int expert_id, int param_id);
  void build_logical_expert_param() {
    for (int layer_id = 0; layer_id < metas->num_layer; layer_id++) {
      for (int expert_id = 0; expert_id < metas->num_expert; expert_id++) {
        build_logical_expert_param(layer_id, expert_id);
      }
    }
  }
  void build_logical_expert_param(int layer_id, int expert_id) {
    auto e = source_list[metas->squeeze_expert_idx(layer_id, expert_id)];
    e->reference_to_model_param.make_logical(e->host_data, mem_mngr_ctx.get());
  }
  torch::Tensor ref_one_expert_param(int layer_id, int expert_id, std::string param_name) {
    return ref_one_expert_param(layer_id, expert_id, metas->param_name_to_id[param_name]);
  }
  torch::Tensor ref_one_expert_param(int layer_id, int expert_id, int param_id) {
    return source_list[metas->squeeze_expert_idx(layer_id, expert_id)]->reference_to_model_param.get_tensor(param_id);
  }
  void pin_memory();
  ExpertHandler* get_source(int layer_id, int expert_id) {
    return source_list[metas->squeeze_expert_idx(layer_id, expert_id)];
  }
};
