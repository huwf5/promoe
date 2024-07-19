#include <torch/extension.h>
#include "model_loader.hpp"
#include "logging.hpp"
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
  expert_handler->host_data.mem_buffers[param_id] = MemBuffer(param);

  // this does not trigger actual memory allocaiton
  auto options = torch::TensorOptions().device("cuda").dtype(param.dtype());
  expert_handler->reference_to_model_param.mem_buffers[param_id].make_logical({0}, options);
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
