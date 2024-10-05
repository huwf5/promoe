#include <torch/extension.h>
#include <cuda.h>
#include "model_loader.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "utils.hpp"

ModelLoader::ModelLoader(std::shared_ptr<ModuleMeta> metas) : metas(metas) {
  source_list.resize(metas->num_layer * metas->num_expert, nullptr);
  mem_mngr_ctx = std::make_shared<MemMngrCtx>();
  for (int layer_id = 0; layer_id < metas->num_layer; layer_id++) {
    for (int expert_id = 0; expert_id < metas->num_expert; expert_id++) {
      auto &expert_handler = source_list[metas->squeeze_expert_idx(layer_id, expert_id)];
      expert_handler = new ExpertHandler();
      expert_handler->expert_idx = expert_id;
      expert_handler->layer_idx = layer_id;
      expert_handler->host_data = std::make_unique<HostExpertMemHanlder>(metas->num_per_expert_param);
      CUDA_CALL(cudaEventCreateWithFlags(&expert_handler->event, cudaEventDisableTiming));
    }
  }
}

void ModelLoader::add_one_expert_param(torch::Tensor param, int layer_id,
                                       int expert_id, int param_id, size_t alloc_nbytes) {
  auto & expert_handler = source_list[metas->squeeze_expert_idx(layer_id, expert_id)];
  expert_handler->host_data->set(param_id, param, alloc_nbytes);

  // this does not trigger actual memory allocaiton
  // auto options = torch::TensorOptions().device("cuda").dtype(param.dtype());
  // expert_handler->reference_to_model_param.mem_buffers[param_id].make_logical(param.sizes(), options, this->mem_mngr_ctx.get());
}

void ModelLoader::add_all_dummy_expert_params() {
  // create only one dummy param to speed up memory allocation
  auto & expert_example = source_list[0];
  size_t current_nbytes = 0;
  for (int i = 0; i < metas->num_per_expert_param - metas->num_dummy_params; i++) {
    current_nbytes += expert_example->host_data->nbytes(i);
  }
  long dummy_nbytes = current_nbytes * (metas->expert_mem_scale - 1);
  long per_dummy_nbytes = dummy_nbytes / metas->num_dummy_params;
  per_dummy_nbytes = round_up<long>(per_dummy_nbytes, 16);
  auto dummy_tensor = torch::zeros({per_dummy_nbytes}, torch::TensorOptions().device("cpu").pinned_memory(true).dtype(torch::kUInt8));

  for (int layer_id = 0; layer_id < metas->num_layer; layer_id++) {
    for (int expert_id = 0; expert_id < metas->num_expert; expert_id++) {
      auto & expert_handler = source_list[metas->squeeze_expert_idx(layer_id, expert_id)];
      // size_t current_nbytes = 0;
      // for (int i = 0; i < metas->num_per_expert_param - metas->num_dummy_params; i++) {
      //   current_nbytes += expert_handler->host_data->nbytes(i);
      // }
      // long dummy_nbytes = current_nbytes * (metas->expert_mem_scale - 1);
      // long per_dummy_nbytes = dummy_nbytes / metas->num_dummy_params;
      // per_dummy_nbytes = round_up<long>(per_dummy_nbytes, 16);
      for (int i = metas->num_per_expert_param - metas->num_dummy_params; i < metas->num_per_expert_param; i++) {
        // auto dummy_tensor = torch::zeros({per_dummy_nbytes}, torch::TensorOptions().device("cpu").pinned_memory(true).dtype(torch::kUInt8));
        expert_handler->host_data->set(i, dummy_tensor, per_dummy_nbytes);
      }
    }
  }
}

void ModelLoader::pin_memory() {
  LOG(INFO) << "pin expert memorys on cpu...";
  #pragma omp parallel for num_threads(metas->num_layer)
  for (int l = 0; l < metas->num_layer; l++) {
    for (int e = 0; e < metas->num_expert; e++) {
      source_list[metas->squeeze_expert_idx(l, e)]->host_data->pin_memory();
    }
  }
  LOG(INFO) << "pin expert memorys on cpu...done";
}

MemMngrCtx::MemMngrCtx() {
  CUDA_CALL(cudaSetDevice(0));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

  CU_CALL(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  LOG(ERROR) << "Minimal granularity: " << granularity;

  CU_CALL(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  LOG(ERROR) << "Recommended granularity: " << granularity;

  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = 0;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

// void MemMngrCtx::build_dummy(size_t dummy_size) {
//   dummy_mem_nbyte = dummy_size;
//   dummy_mem_nbyte = round_up(dummy_mem_nbyte, granularity);
//   CU_CALL(cuMemCreate(&dummy_mem_handle, dummy_mem_nbyte, &prop, 0));
// }
// void MemMngrCtx::destroy_dummy() {
//   CU_CALL(cuMemRelease(dummy_mem_handle));
//   dummy_mem_nbyte = 0;
// }

void MemMngrCtx::cu_mem_create(CUmemGenericAllocationHandle *handle, size_t size) {
  CU_CALL(cuMemCreate(handle, size, &prop, 0));
}
void MemMngrCtx::cu_address_reserve(CUdeviceptr *ptr, size_t size) {
  CU_CALL(cuMemAddressReserve(ptr, size, 0, 0, 0));
}
void MemMngrCtx::cu_map_address(CUdeviceptr ptr, size_t size, CUmemGenericAllocationHandle handle) {
  CU_CALL(cuMemMap(ptr, size, 0, handle, 0));
}
void MemMngrCtx::cu_set_access(CUdeviceptr ptr, size_t size) {
  CU_CALL(cuMemSetAccess(ptr, size, &accessDesc, 1));
}

void MemMngrCtx::cu_unmap_address(CUdeviceptr ptr, size_t size) {
  CU_CALL(cuMemUnmap(ptr, size));
}
void ExpertParamWrapperCUDriver::map_to(ExpertMemHanlderBase *physical_base, MemMngrCtx *ctx) {
  auto physical = dynamic_cast<ExpertMemHanlderCUDriver *>(physical_base);
  CHECK(physical != nullptr) << "mismatch physical - logical mem type";
  {
    TRACE_EVENT_GURAD(kCache, "map");
    for (int i = 0; i < ptrs.size(); i++) {
      mapped_nbytes[i] = address_range_nbytes[i];
      ctx->cu_map_address(ptrs[i], address_range_nbytes[i], physical->handles[i]);
    }
  }
  {
    TRACE_EVENT_GURAD(kCache, "set_access");
    for (int i = 0; i < ptrs.size(); i++) {
      ctx->cu_set_access(ptrs[i], address_range_nbytes[i]);
    }
  }
}
void ExpertParamWrapperCUDriverUnified::map_to(ExpertMemHanlderBase *physical_base, MemMngrCtx *ctx) {
  auto physical = dynamic_cast<ExpertMemHanlderCUDriverUnified *>(physical_base);
  CHECK(physical != nullptr) << "mismatch physical - logical mem type";
  {
    TRACE_EVENT_GURAD(kCache, "map");
    ctx->cu_map_address(ptr, address_range_nbyte, physical->handle);
    mapped_nbyte = address_range_nbyte;
  }
  {
    TRACE_EVENT_GURAD(kCache, "set_access");
    ctx->cu_set_access(ptr, address_range_nbyte);
  }
}
void ExpertParamWrapperTensor::map_to(ExpertMemHanlderBase *physical, MemMngrCtx *ctx) {
  TRACE_EVENT_GURAD(kCache, "map");
  for (int i = 0; i < model_parameter_reference.size(); i++) {
    auto &prebuilt_t = physical->get_prebuilt_tensor(i);
    model_parameter_reference[i].set_(prebuilt_t, 0, prebuilt_t.sizes(), prebuilt_t.strides());
  }
}
void ExpertParamWrapperCUDriver::unmap() {
  TRACE_EVENT_GURAD(kCache, "unmap");
  for (int i = 0; i < ptrs.size(); i++) {
    MemMngrCtx::cu_unmap_address(ptrs[i], mapped_nbytes[i]);
  }
}
void ExpertParamWrapperCUDriverUnified::unmap() {
  TRACE_EVENT_GURAD(kCache, "unmap");
  MemMngrCtx::cu_unmap_address(ptr, mapped_nbyte);
}

torch::Tensor to_um(torch::Tensor t) {
  void* ptr = nullptr;
  CUDA_CALL(cudaMallocManaged(&ptr, t.nbytes()));
  CUDA_CALL(cudaMemcpyAsync(ptr, t.data_ptr(), t.nbytes(), cudaMemcpyDefault, 0));
  CUDA_CALL(cudaStreamSynchronize(0));
  return torch::from_blob(ptr, t.sizes(), t.options().device(torch::kCUDA, 0));
}
void ExpertMemHanlderTensor::allocate_like(HostExpertMemHanlderBase *other, MemMngrCtx *ctx) {
  prebuilt_tensors.resize(other->num_chunk());
  storages.resize(other->num_chunk());
  total_allocation_nbytes = 0;
  for (int i = 0; i < other->num_chunk(); i++) {
    {
      // storage
      torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(torch::kUInt8);
      void *cuda_ptr;
      CUDA_CALL(cudaMalloc(&cuda_ptr, other->alloc_nbytes(i)));
      storages[i] = torch::from_blob(cuda_ptr, {static_cast<long>(other->alloc_nbytes(i))}, options);
      total_allocation_nbytes += other->alloc_nbytes(i);
    }
    {
      // tensors
      torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other->dtype(i));
      prebuilt_tensors[i] = torch::from_blob(storages[i].data_ptr(), other->get_tensor(i).sizes(), options);
    }
  }
}
void ExpertMemHanlderTensorUnified::allocate_like(HostExpertMemHanlderBase *other, MemMngrCtx *ctx) {
  prebuilt_tensors.resize(other->num_chunk());
  offsets_of_each_param.resize(other->num_chunk());

  total_allocation_nbytes = 0;
  for (int i = 0; i < other->num_chunk(); i++) {
    offsets_of_each_param[i] = total_allocation_nbytes;
    total_allocation_nbytes += other->alloc_nbytes(i);
  }

  {
    // storage
    torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(torch::kUInt8);
    void *cuda_ptr;
    CUDA_CALL(cudaMalloc(&cuda_ptr, total_allocation_nbytes));
    CUDA_CALL(cudaMemset(cuda_ptr, 0, total_allocation_nbytes));
    storage = torch::from_blob(cuda_ptr, {static_cast<long>(total_allocation_nbytes)}, options);
  }

  for (int i = 0; i < other->num_chunk(); i++) {
    {
      // tensors
      torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other->dtype(i));
      prebuilt_tensors[i] = torch::from_blob(storage.data_ptr<uint8_t>() + offsets_of_each_param[i], other->get_tensor(i).sizes(), options);
    }
  }
}

void ExpertMemHanlderTensorGlobalUnified::allocate_like(HostExpertMemHanlderBase *other, MemMngrCtx *ctx) {
  total_allocation_nbytes = 0;
  prebuilt_tensors.resize(other->num_chunk());
  for (int i = 0; i < other->num_chunk(); i++) {
    // tensors
    torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other->dtype(i));
    prebuilt_tensors[i] = torch::from_blob(ctx->global_unified_mem + ctx->global_unified_mem_offset, other->get_tensor(i).sizes(), options);
    ctx->global_unified_mem_offset += other->alloc_nbytes(i);
    CHECK(ctx->global_unified_mem_offset <= ctx->global_unified_mem_size);
    total_allocation_nbytes += other->alloc_nbytes(i);
  }
}
void ExpertMemHanlderCUDriver::allocate_like(HostExpertMemHanlderBase *other, MemMngrCtx *ctx) {
  total_allocation_nbytes = 0;
  handles.resize(other->num_chunk());
  prebuilt_ptrs.resize(other->num_chunk());
  prebuilt_tensors.resize(other->num_chunk());

  for (int i = 0; i < handles.size(); i++) {
    size_t size = other->nbytes(i);
    size = round_up(size, ctx->granularity);
    total_allocation_nbytes += size;
    ctx->cu_mem_create(&handles[i], size);

    ctx->cu_address_reserve(&prebuilt_ptrs[i], size);

    ctx->cu_map_address(prebuilt_ptrs[i], size, handles[i]);
    ctx->cu_set_access(prebuilt_ptrs[i], size);

    torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other->dtype(i));
    prebuilt_tensors[i] = torch::from_blob(ptr(i), other->get_tensor(i).sizes(), options);
  }
}
void ExpertMemHanlderCUDriverUnified::allocate_like(HostExpertMemHanlderBase *other, MemMngrCtx *ctx) {
  total_allocation_nbytes = 0;
  prebuilt_tensors.resize(other->num_chunk());
  offsets_of_each_param = {0};
  size_t size = 0;
  for (int i = 0; i < other->num_chunk(); i++) {
    size += other->nbytes(i);
    offsets_of_each_param.push_back(size);
  }
  size = round_up(size, ctx->granularity);
  total_allocation_nbytes = size;
  ctx->cu_mem_create(&handle, size);
  ctx->cu_address_reserve(&prebuilt_ptr, size);
  ctx->cu_map_address(prebuilt_ptr, size, handle);
  ctx->cu_set_access(prebuilt_ptr, size);

  for (int i = 0; i < other->num_chunk(); i++) {
    torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other->dtype(i));
    prebuilt_tensors[i] = torch::from_blob(ptr(i), other->get_tensor(i).sizes(), options);
  }
}
