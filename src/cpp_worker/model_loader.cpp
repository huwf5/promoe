#include <torch/extension.h>
#include "model_loader.hpp"
#include "logging.hpp"
#include "profiler.hpp"
void ModelLoader::add_one_expert_param(torch::Tensor param, int layer_id,
                                       int expert_id, int param_id) {
  ExpertHandler *expert_handler = nullptr;
  if (param_id == 0) {
    expert_handler = new ExpertHandler();
    expert_handler->expert_idx = expert_id;
    expert_handler->layer_idx = layer_id;
    expert_handler->host_data.mem_buffers.resize(metas->num_per_expert_param);
    expert_handler->reference_to_model_param.mem_buffers.resize(metas->num_per_expert_param);
    CUDA_CALL(cudaEventCreateWithFlags(&expert_handler->event, cudaEventDisableTiming));
    source_list[metas->squeeze_expert_idx(layer_id, expert_id)] = expert_handler;
  } else {
    expert_handler = source_list[metas->squeeze_expert_idx(layer_id, expert_id)];
  }
  expert_handler->host_data.mem_buffers[param_id] = HostMemWrapper(param);

  // this does not trigger actual memory allocaiton
  auto options = torch::TensorOptions().device("cuda").dtype(param.dtype());
  expert_handler->reference_to_model_param.mem_buffers[param_id].make_logical(param.sizes(), options, this->mem_mngr_ctx.get());
}
void ModelLoader::pin_memory() {
  LOG(INFO) << "pin expert memorys on cpu...";
  #pragma omp parallel for num_threads(metas->num_layer)
  for (int l = 0; l < metas->num_layer; l++) {
    for (int e = 0; e < metas->num_expert; e++) {
      for (int p = 0; p < metas->num_per_expert_param; p++) {
        source_list[metas->squeeze_expert_idx(l, e)]->host_data.mem_buffers[p].pin_memory();
      }
    }
  }
  LOG(INFO) << "pin expert memorys on cpu...done";
}
void PhysicalMemHandler::allocate_like(HostMemWrapper &other, torch::TensorOptions options, MemMngrCtx* ctx) {
  // data = torch::empty_like(other.data, options);

  nbytes = other.data.nbytes();
  nbytes = round_up(nbytes, ctx->granularity);

  CU_CALL(cuMemCreate(&handle, nbytes, &ctx->prop, 0));
  // LOG(ERROR) << "Allocating physical addr " << std::hex << handle << std::dec << ", len " << nbytes << ", sizes " << other.data.sizes() << ", dtype " << other.data.dtype();

  this->logical_ptr.make_logical(other.data.sizes(), options, ctx);
  this->logical_ptr.unmap();
  this->logical_ptr.map_to(*this, ctx);
  CHECK(this->logical_ptr.nbytes == nbytes);
}
void PhysicalMemHandler::allocate_like(HostMemWrapper &other, MemMngrCtx* ctx) {
  torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, ctx->device_id).dtype(other.data.dtype());
  allocate_like(other, options, ctx);
}
void PhysicalMemHandler::allocate(size_t nb, MemMngrCtx *ctx) {
  CHECK(false) << "Unimplemented";
  nb = round_up(nb, ctx->granularity);
  this->nbytes = nb;
  CU_CALL(cuMemCreate(&handle, nbytes, &ctx->prop, 0));
  // LOG(ERROR) << "Allocating physical addr " << std::hex << handle << std::dec << ", len " << size;

}

void LogicalMemHandler::map_to(PhysicalMemHandler &physical, MemMngrCtx* ctx) {
  // LOG(ERROR) << "Mapping logical addr " << std::hex << device_ptr << " to " << std::hex << physical.handle << std::dec << ", len " << nbytes;
  // data.set_(physical.data, 0, physical.data.sizes(), physical.data.strides());
  CHECK(physical.nbytes == nbytes);
  {
    TRACE_EVENT_GURAD(kCache, "map");
    CU_CALL(cuMemMap(device_ptr, nbytes, 0, physical.handle, 0));
  }
  {
    TRACE_EVENT_GURAD(kCache, "set access");
    CU_CALL(cuMemSetAccess(device_ptr, nbytes, &ctx->accessDesc, 1));
  }
}
void LogicalMemHandler::make_logical(torch::IntArrayRef shape, torch::TensorOptions options, MemMngrCtx* ctx) {
  // data = torch::empty({0}, options);

  nbytes = c10::multiply_integers(shape) * torch::elementSize(options.dtype().toScalarType());
  nbytes = round_up(nbytes, ctx->granularity);
  CU_CALL(cuMemAddressReserve(&device_ptr, nbytes, 0, 0, 0));
  // LOG(ERROR) << "Making logical addr range " << std::hex <<  device_ptr << std::dec << ", len " << nbytes;

  CU_CALL(cuMemMap(device_ptr, ctx->dummy_mem.nbytes, 0, ctx->dummy_mem.handle, 0));
  data = torch::from_blob((void*)device_ptr, shape, options);
}
void LogicalMemHandler::unmap() {
  CU_CALL(cuMemUnmap(device_ptr, nbytes));
}
void ExpertParamWrapper::unmap() {
  TRACE_EVENT_GURAD(kCache, "unmap");
  for (auto &mem : mem_buffers) {
    mem.unmap();
  }
}
MemMngrCtx::MemMngrCtx() {
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  CU_CALL(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = 0;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  dummy_mem.nbytes = 1024;
  dummy_mem.nbytes = round_up(dummy_mem.nbytes, granularity);
  CU_CALL(cuMemCreate(&dummy_mem.handle, dummy_mem.nbytes, &prop, 0));
}
