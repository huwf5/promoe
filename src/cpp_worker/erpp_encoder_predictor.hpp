#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <torch/script.h>

#include "utils.hpp"

struct ErppEncoderPrediction {
  std::vector<std::vector<int64_t>> rankings;
  std::vector<int> budgets;
  using iterator = std::vector<std::vector<int64_t>>::iterator;
  using const_iterator = std::vector<std::vector<int64_t>>::const_iterator;

  size_t size() const { return rankings.size(); }
  bool empty() const { return rankings.empty(); }
  iterator begin() { return rankings.begin(); }
  iterator end() { return rankings.end(); }
  const_iterator begin() const { return rankings.begin(); }
  const_iterator end() const { return rankings.end(); }
  std::vector<int64_t>& operator[](size_t idx) { return rankings[idx]; }
  const std::vector<int64_t>& operator[](size_t idx) const { return rankings[idx]; }
  operator std::vector<std::vector<int64_t>>&() { return rankings; }
  operator const std::vector<std::vector<int64_t>>&() const { return rankings; }
};

int erpp_noisy_or_sum_budget(torch::Tensor layer_scores, int num_expert);

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
  ErppEncoderPrediction predict_recorded();
  void reset_sequence_state();
  std::vector<int> parse_budgets(const std::string& spec) const;
  int encoder_budget_for_layer(int layer_idx) const;
  int encoder_jit_floor() const;
  int encoder_jit_ranking_limit(int layer_idx) const;
  int encoder_jit_ranking_limit(int layer_idx, int budget) const;
  bool should_prefetch_layer(int layer_idx) const;
  ErppEncoderPrediction predict(torch::Tensor hidden, torch::Tensor attention_mask);

 private:
  ModuleMeta* metas;
  torch::jit::script::Module model;
  bool loaded = false;
  bool dynamic_noisy_or_budget = false;
  std::vector<int> budgets;
  std::vector<uint8_t> enabled_layers;

  torch::Tensor hidden_buffer;
  cudaEvent_t record_event = nullptr;
  bool input_recorded = false;
  int64_t recorded_forward_epoch = -1;
  int64_t recorded_generate_epoch = -1;
  int budget_from_scores(torch::Tensor layer_scores) const;
  ErppEncoderPrediction predict_from_cpu_tensors(torch::Tensor hidden_cpu);
  std::vector<uint8_t> parse_enabled_layers(const std::string& spec) const;
};
