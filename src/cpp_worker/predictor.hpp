#pragma once
#include <torch/script.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include "utils.hpp"

class Predictor {
  std::shared_ptr<ModuleMeta> metas;
  // torch::jit::script::Module predict_model;
  // single sequence for now
  torch::Tensor expert_access_buffer;
  torch::Tensor last_use_distance_buffer;
  torch::Tensor weighted_access_freq_sum_buffer;
  torch::Tensor first_moe_attn_input_logits_buffer;
  std::vector<torch::Tensor> moe_attn_input_logits_buffer_list;
  std::vector<torch::jit::script::Module> predict_model_list;
  void init_expert_access_buffer() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    switch (metas->predict_input_mode) {
      case kOneToken: {
        expert_access_buffer = torch::zeros({metas->num_layer, metas->num_expert}, options);
        break;
      }
      case kDecodeCumsum: {
        expert_access_buffer = torch::zeros({metas->num_layer, metas->num_expert}, options);
        break;
      }
      case kLastUseDistance: {
        last_use_distance_buffer = torch::zeros({metas->num_layer, metas->num_expert}, options);
        break;
      }
      case kWeighedDecodeCumsum: {
        weighted_access_freq_sum_buffer = torch::zeros({metas->num_layer, metas->num_expert}, options);
        break;
      }
      case kFirstMoeAttnInputLogits : { break; }
      case kMoeAttnInputLogits :      { 
        moe_attn_input_logits_buffer_list.resize(metas->num_layer);
        break;
      }
      default : { CHECK(false) << "Unknown predict input mode"; }
    }
  }

  void load_one_model(std::string model_path, int idx = 0);
 public:
  Predictor(std::shared_ptr<ModuleMeta> metas) : metas(metas) {
    init_expert_access_buffer();
  }
  void load_model(std::string model_path);

  torch::Tensor predict(int input_layer_id);

  void add_one_layer(int layer_id, torch::Tensor experts);
  void add_one_layer(int layer_id, int64_t *experts, size_t num_expert);
  void record_moe_attn_logits(int layer_id, torch::Tensor attn_logits);
  void end_of_one_token_prediction();
  void start_of_new_sequence();
  // void clear_access_buffer() {
  //   expert_access_buffer.fill_(0);
  //   last_use_distance_buffer.fill_(0);
  //   weighted_access_freq_sum_buffer.fill_(0);
  // }
};