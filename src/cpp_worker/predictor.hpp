#pragma once
#include <torch/script.h>
#include "utils.hpp"

class Predictor {
  std::shared_ptr<ModuleMeta> metas;
  torch::jit::script::Module predict_model;
  // single sequence for now
  torch::Tensor expert_access_buffer;
  void init_expert_access_buffer() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    expert_access_buffer = torch::zeros({metas->num_layer, metas->num_expert}, options);
  }

 public:
  Predictor(std::shared_ptr<ModuleMeta> metas) : metas(metas) {
    init_expert_access_buffer();
  }
  void load_model(std::string model_path);

  torch::Tensor predict();

  void add_one_layer(int layer_id, torch::Tensor experts);
  void add_one_layer(int layer_id, int64_t *experts, size_t num_expert);
  void clear_access_buffer() {
    expert_access_buffer.fill_(0);
  }
};