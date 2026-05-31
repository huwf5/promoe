#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <torch/script.h>

#include "utils.hpp"

class ErppEncoderPredictor {
 public:
  explicit ErppEncoderPredictor(ModuleMeta* metas);
  ~ErppEncoderPredictor();

  void load_model_from(const std::string& path);
  void record_encoder_layer0(torch::Tensor hidden,
                             torch::Tensor attention_mask,
                             cudaStream_t compute_stream,
                             int64_t forward_epoch,
                             int64_t generate_epoch);
  std::vector<std::vector<int64_t>> predict_recorded();
  void reset_sequence_state();
  std::vector<int> parse_budgets(const std::string& spec) const;
  int encoder_budget_for_layer(int layer_idx) const;
  int encoder_jit_floor() const;
  int encoder_jit_ranking_limit(int layer_idx) const;
  bool should_prefetch_layer(int layer_idx) const;
  std::vector<std::vector<int64_t>> predict(torch::Tensor hidden, torch::Tensor attention_mask);

 private:
  ModuleMeta* metas;
  torch::jit::script::Module model;
  bool loaded = false;
  std::vector<int> budgets;
  std::vector<uint8_t> enabled_layers;

  torch::Tensor hidden_buffer;
  torch::Tensor attention_mask_buffer;
  cudaEvent_t record_event = nullptr;
  bool input_recorded = false;
  std::vector<std::vector<int64_t>> predict_from_cpu_tensors(torch::Tensor hidden_cpu,
                                                             torch::Tensor attention_mask_cpu);
  std::vector<uint8_t> parse_enabled_layers(const std::string& spec) const;
  torch::Tensor normalize_attention_mask(torch::Tensor attention_mask, int64_t batch_size, int64_t seq_len) const;
};
