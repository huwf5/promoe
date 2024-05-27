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
  void load_model(std::string model_path) {
    c10::Device cpu_device(c10::DeviceType::CPU);
    predict_model = torch::jit::load(model_path, cpu_device);
    predict_model.eval();
  }

  torch::Tensor predict() {
    std::vector<torch::jit::IValue> inputs{this->expert_access_buffer.flatten().unsqueeze(0)};
    return predict_model.forward(inputs).toTensor();
  }

  void add_one_layer(int layer_id, torch::Tensor experts) {
    add_one_layer(layer_id, experts.data_ptr<int64_t>(), experts.numel());
  }
  void add_one_layer(int layer_id, int64_t* experts, size_t num_expert) {
    for (int i = 0; i < num_expert; i++) {
      expert_access_buffer[layer_id][experts[i]] = 1;
    }
  }
  void clear_access_buffer() {
    expert_access_buffer.fill_(0);
  }
};