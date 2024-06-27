#include "predictor.hpp"

#include "logging.hpp"

#include "profiler.hpp"

void Predictor::add_one_layer(int layer_id, int64_t *experts, size_t num_expert) {
  last_use_distance_buffer[layer_id] += 1;
  weighted_access_freq_sum_buffer[layer_id] /= metas->predict_input_decay;
  for (int i = 0; i < num_expert; i++) {
    expert_access_buffer[layer_id][experts[i]] += 1;
    last_use_distance_buffer[layer_id][experts[i]] = 0;
    weighted_access_freq_sum_buffer[layer_id][experts[i]] += 1;
  }
}
torch::Tensor Predictor::predict() {
  TRACE_EVENT_GURAD(kPredictor, "predict");
  torch::Tensor input = this->expert_access_buffer.clone();
  if (metas->predict_input_mode == kDecodeCumsum) {
    auto s = input.sum(1, true);
    input /= s;
    input = input.nan_to_num(0);
  } else if (metas->predict_input_mode == kLastUseDistance) {
    input = this->last_use_distance_buffer.clone();
    // predict xxx
    input = torch::max(
      torch::ones_like(input) * metas->predict_input_reuse_distance_max - input,
      torch::zeros_like(input)
    );
    input /= metas->predict_input_reuse_distance_max;
  } else if (metas->predict_input_mode == kWeighedDecodeCumsum) {
    input = this->weighted_access_freq_sum_buffer.clone();
  }
  std::vector<torch::jit::IValue> inputs{input.flatten().unsqueeze(0)};
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
