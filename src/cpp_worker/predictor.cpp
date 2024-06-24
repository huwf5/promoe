#include "predictor.hpp"

#include "logging.hpp"

#include "profiler.hpp"

void Predictor::add_one_layer(int layer_id, int64_t *experts, size_t num_expert) {
  for (int i = 0; i < num_expert; i++) {
    expert_access_buffer[layer_id][experts[i]] += 1;
  }
}
torch::Tensor Predictor::predict() {
  TRACE_EVENT_GURAD(kPredictor, "predict");
  if (metas->predict_input_mode == kDecodeCumsum) {
    this->expert_access_buffer /= this->expert_access_buffer.sum(1, true);
    this->expert_access_buffer = this->expert_access_buffer.nan_to_num(0);
  }
  std::vector<torch::jit::IValue> inputs{this->expert_access_buffer.flatten().unsqueeze(0)};
  return predict_model.forward(inputs).toTensor();
}
void Predictor::load_model(std::string model_path) {
  c10::Device cpu_device(c10::DeviceType::CPU);
  predict_model = torch::jit::load(model_path, cpu_device);
  predict_model.eval();
}
void Predictor::add_one_layer(int layer_id, torch::Tensor experts) {
  add_one_layer(layer_id, experts.data_ptr<int64_t>(), experts.numel());
}
