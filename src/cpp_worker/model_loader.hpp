#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "utils.hpp"

class ExpertMemHanlderBase;
class MemMngrCtx {
 public:
  // CUmemGenericAllocationHandle dummy_mem_handle = 0;
  // size_t dummy_mem_nbyte = 0;

  ExpertMemHanlderBase* dummy_physical;

  CUmemAllocationProp prop{};
  size_t granularity = 0;
  CUmemAccessDesc accessDesc = {};
  int device_id = 0;
  MemMngrCtx();

  uint8_t * global_unified_mem = nullptr;
  size_t global_unified_mem_size = 0;
  size_t global_unified_mem_offset = 0;
  // void build_dummy(size_t dummy_size);
  // void destroy_dummy();

  void cu_mem_create(CUmemGenericAllocationHandle *handle, size_t size);
  static void cu_address_reserve(CUdeviceptr *ptr, size_t size);
  void cu_map_address(CUdeviceptr ptr, size_t size, CUmemGenericAllocationHandle handle);
  void cu_set_access(CUdeviceptr ptr, size_t size);
  static void cu_unmap_address(CUdeviceptr ptr, size_t size);
};

class HostMemWrapper {
 public:
  torch::Tensor data_;
  size_t alloc_nbytes_ = 0;
  HostMemWrapper() {}
  HostMemWrapper(torch::Tensor t) : data_(t) {
    alloc_nbytes_ = t.nbytes();
  }
  void*     ptr() { return data_.data_ptr();}
  size_t nbytes() { return data_.nbytes();}
  auto    dtype() { return data_.dtype(); }
  size_t alloc_nbytes() { return alloc_nbytes_; }
  void pin_memory() {
    data_ = data_.pin_memory();
  }
};

class HostExpertMemHanlderBase {
  std::vector<HostMemWrapper> mem_buffers;
 public:
  HostExpertMemHanlderBase() {}
  HostExpertMemHanlderBase(int num) : mem_buffers(num) {}
  void pin_memory() {
    for (auto & m : mem_buffers) { m.pin_memory(); }
  }
  void*     ptr(int idx) { return mem_buffers[idx].ptr(); }
  size_t nbytes(int idx) { return mem_buffers[idx].nbytes(); }
  size_t total_nbytes() {
    size_t total = 0;
    for (auto & m : mem_buffers) { total += m.nbytes(); }
    return total;
  }
  auto    dtype(int idx) { return mem_buffers[idx].dtype(); }
  void set(int idx, torch::Tensor &param, size_t alloc_nbytes = 0) {
    this->mem_buffers.at(idx) = HostMemWrapper(param);
    if (alloc_nbytes == 0) {
      alloc_nbytes = param.nbytes();
    }
    this->mem_buffers.at(idx).alloc_nbytes_ = alloc_nbytes;
  }
  size_t num_chunk() { return mem_buffers.size(); }
  torch::Tensor get_tensor(int idx) { return mem_buffers[idx].data_; }
  size_t alloc_nbytes(int idx) { return mem_buffers[idx].alloc_nbytes(); }
  size_t total_alloc_nbytes() {
    size_t total = 0;
    for (auto & m : mem_buffers) { total += m.alloc_nbytes(); }
    return total;
  }
};

class HostExpertMemHanlder : public HostExpertMemHanlderBase {
 public:
  using HostExpertMemHanlderBase::HostExpertMemHanlderBase;
};

class ExpertMemHanlderBase {
 protected:
  std::vector<torch::Tensor> prebuilt_tensors;
  size_t total_allocation_nbytes = 0;
 public:
  virtual void allocate_like(HostExpertMemHanlderBase* other, MemMngrCtx* ctx) = 0;
  torch::Tensor & get_prebuilt_tensor(int idx) { return prebuilt_tensors[idx]; }
  virtual void* ptr(int idx) { return prebuilt_tensors[idx].data_ptr(); }
  virtual ~ExpertMemHanlderBase() {}
  virtual size_t get_allocation_nbytes() { return total_allocation_nbytes; }
};

class ExpertMemHanlderTensor : public ExpertMemHanlderBase {
 public:
  std::vector<torch::Tensor> storages;
  void allocate_like(HostExpertMemHanlderBase *other, MemMngrCtx *ctx) override;
};
class ExpertMemHanlderTensorUnified : public ExpertMemHanlderBase {
 public:
  torch::Tensor storage;
  std::vector<size_t> offsets_of_each_param;
  void allocate_like(HostExpertMemHanlderBase *other, MemMngrCtx *ctx) override;
};
class ExpertMemHanlderTensorGlobalUnified : public ExpertMemHanlderBase {
 public:
  void allocate_like(HostExpertMemHanlderBase *other, MemMngrCtx *ctx) override;
};

class ExpertMemHanlderCUDriver : public ExpertMemHanlderBase {
  std::vector<CUmemGenericAllocationHandle> handles;
  std::vector<CUdeviceptr> prebuilt_ptrs;
  friend class ExpertParamWrapperCUDriver;
 public:
  void allocate_like(HostExpertMemHanlderBase* other, MemMngrCtx* ctx) override;
  void* ptr(int idx) override { return (void*)prebuilt_ptrs[idx]; }
};

class ExpertMemHanlderCUDriverUnified : public ExpertMemHanlderBase {
  CUmemGenericAllocationHandle handle;
  CUdeviceptr prebuilt_ptr;
  std::vector<size_t> offsets_of_each_param;
  friend class ExpertParamWrapperCUDriverUnified;
 public:
  void allocate_like(HostExpertMemHanlderBase* other, MemMngrCtx* ctx) override;
  void* ptr(int idx) override {
    return (uint8_t*)prebuilt_ptr + offsets_of_each_param[idx];
  }
};

class ExpertParamWrapperBase {
 protected:
  std::vector<torch::Tensor> model_parameter_reference;
 public:
  ExpertParamWrapperBase() {}
  virtual void unmap() = 0;
  virtual void map_to(ExpertMemHanlderBase* physical, MemMngrCtx* ctx) = 0;
  virtual void make_logical(HostExpertMemHanlderBase* other, MemMngrCtx* ctx) = 0;
  torch::Tensor get_tensor(int idx) { return model_parameter_reference[idx]; }
  virtual ~ExpertParamWrapperBase() {}
};

class ExpertParamWrapperTensor : public ExpertParamWrapperBase {
 public:
  ExpertParamWrapperTensor() {}
  void unmap() override {}
  void map_to(ExpertMemHanlderBase *physical, MemMngrCtx *ctx) override;
  void make_logical(HostExpertMemHanlderBase* other, MemMngrCtx* ctx) override {
    model_parameter_reference.resize(other->num_chunk());
    for (int i = 0; i < model_parameter_reference.size(); i++) {
      model_parameter_reference[i] = torch::empty({0}, torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other->dtype(i)));
    }
    this->map_to(ctx->dummy_physical, ctx);
  }
};

class ExpertParamWrapperCUDriver : public ExpertParamWrapperBase {
  std::vector<CUdeviceptr> ptrs;
  std::vector<size_t> mapped_nbytes;
  std::vector<size_t> address_range_nbytes;
 public:
  ExpertParamWrapperCUDriver() {}
  void make_logical(HostExpertMemHanlderBase* other, MemMngrCtx* ctx) override {
    ptrs.resize(other->num_chunk());
    mapped_nbytes.resize(other->num_chunk());
    address_range_nbytes.resize(other->num_chunk());
    model_parameter_reference.resize(other->num_chunk());

    for (int i = 0; i < model_parameter_reference.size(); i++) {
      address_range_nbytes[i] = round_up(other->nbytes(i), ctx->granularity);
      ctx->cu_address_reserve(&ptrs[i], address_range_nbytes[i]);
    }

    this->map_to(ctx->dummy_physical, ctx);
    for (int i = 0; i < model_parameter_reference.size(); i++) {
      torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other->dtype(i));
      model_parameter_reference[i] = torch::from_blob((void*)ptrs[i], other->get_tensor(i).sizes(), options);
    }
  }
  void unmap() override;
  void map_to(ExpertMemHanlderBase *physical_base, MemMngrCtx *ctx) override;
};

class ExpertParamWrapperCUDriverUnified : public ExpertParamWrapperBase {
  CUdeviceptr ptr;
  size_t mapped_nbyte, address_range_nbyte;
  std::vector<size_t> offsets_of_each_param;
 public:
  ExpertParamWrapperCUDriverUnified() {}
  void make_logical(HostExpertMemHanlderBase* other, MemMngrCtx* ctx) override {
    model_parameter_reference.resize(other->num_chunk());

    offsets_of_each_param = {0};
    address_range_nbyte = 0;
    for (int i = 0; i < other->num_chunk(); i++) {
      address_range_nbyte += other->nbytes(i);
      offsets_of_each_param.push_back(address_range_nbyte);
    }

    address_range_nbyte = round_up(address_range_nbyte, ctx->granularity);
    ctx->cu_address_reserve(&ptr, address_range_nbyte);

    this->map_to(ctx->dummy_physical, ctx);

    for (int i = 0; i < model_parameter_reference.size(); i++) {
      torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other->dtype(i));
      model_parameter_reference[i] = torch::from_blob((uint8_t*)ptr + offsets_of_each_param[i], other->get_tensor(i).sizes(), options);
    }
  }
  void unmap() override;
  void map_to(ExpertMemHanlderBase *physical_base, MemMngrCtx *ctx) override;
};

class ExpertMemParamFactory {
 public:
  std::map<std::string, std::function<ExpertMemHanlderBase*()>>   physical_registry;
  std::map<std::string, std::function<ExpertParamWrapperBase*()>> logical_registry;
  ExpertMemParamFactory() {
    physical_registry["tensor"]                = []() { return new ExpertMemHanlderTensor() ;};
    physical_registry["tensor_unified"]        = []() { return new ExpertMemHanlderTensorUnified() ;};
    physical_registry["tensor_global_unified"] = []() { return new ExpertMemHanlderTensorGlobalUnified() ;};
    physical_registry["cudriver"]              = []() { return new ExpertMemHanlderCUDriver() ;};
    physical_registry["cudriver_unified"]      = []() { return new ExpertMemHanlderCUDriverUnified() ;};

    logical_registry["tensor"]           = []() { return new ExpertParamWrapperTensor(); };
    logical_registry["cudriver"]         = []() { return new ExpertParamWrapperCUDriver(); };
    logical_registry["cudriver_unified"] = []() { return new ExpertParamWrapperCUDriverUnified(); };
  }

  ExpertMemHanlderBase*   create_physical(std::string name) { return physical_registry[name](); }
  ExpertParamWrapperBase* create_logical(std::string name)  { return logical_registry[name]();  }

  static ExpertMemParamFactory& get() {
    static ExpertMemParamFactory instance;
    return instance;
  }
};

class ExpertHandler {
 public:
  // std::shared_ptr<torch::nn::Module> expert_module;
  // HostExpertMemHanlder host_data;
  // ExpertParamWrapper reference_to_model_param;

  std::unique_ptr<HostExpertMemHanlderBase> host_data = nullptr;
  std::shared_ptr<ExpertParamWrapperBase> reference_to_model_param = nullptr;

  ExpertMemHanlderBase* gpu_data = nullptr;
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
    return add_one_expert_param(param, layer_id, expert_id, param_name, /*alloc_nbytes*/ 0);
  }
  void add_one_expert_param(torch::Tensor param, int layer_id, int expert_id, std::string param_name, size_t alloc_nbytes) {
    return add_one_expert_param(param, layer_id, expert_id, metas->param_name_to_id[param_name], alloc_nbytes);
  }
  void add_one_expert_param(torch::Tensor param, int layer_id, int expert_id, int param_id, size_t alloc_nbytes);
  void add_all_dummy_expert_params();
  void build_logical_expert_param() {
    // size_t required_dummy_nbytes = 1024;
    // if (metas->logical_mem_impl == "cudriver_unified") {
    //   required_dummy_nbytes = source_list[0]->host_data->total_nbytes();
    // }
    // mem_mngr_ctx->build_dummy(required_dummy_nbytes);
    for (int layer_id = 0; layer_id < metas->num_layer; layer_id++) {
      for (int expert_id = 0; expert_id < metas->num_expert; expert_id++) {
        build_logical_expert_param(layer_id, expert_id);
      }
    }
    // mem_mngr_ctx->destroy_dummy();
  }
  void build_logical_expert_param(int layer_id, int expert_id) {
    auto e = source_list[metas->squeeze_expert_idx(layer_id, expert_id)];
    e->reference_to_model_param = std::shared_ptr<ExpertParamWrapperBase>(ExpertMemParamFactory::get().logical_registry[metas->logical_mem_impl]());
    e->reference_to_model_param->make_logical(e->host_data.get(), mem_mngr_ctx.get());
  }
  torch::Tensor ref_one_expert_param(int layer_id, int expert_id, std::string param_name) {
    return ref_one_expert_param(layer_id, expert_id, metas->param_name_to_id[param_name]);
  }
  torch::Tensor ref_one_expert_param(int layer_id, int expert_id, int param_id) {
    return source_list[metas->squeeze_expert_idx(layer_id, expert_id)]->reference_to_model_param->get_tensor(param_id);
  }
  void pin_memory();
  ExpertHandler* get_source(int layer_id, int expert_id) {
    return source_list[metas->squeeze_expert_idx(layer_id, expert_id)];
  }
};

torch::Tensor to_um(torch::Tensor t);