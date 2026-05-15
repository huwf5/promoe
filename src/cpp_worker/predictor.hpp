#pragma once
#include <torch/script.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <cuda_runtime.h>
#include "profiler.hpp"
#include "utils.hpp"

struct PredictOutput {
  torch::Tensor prob;
  torch::Tensor experts;
  int input_layer_id = 0;
  int start_output_layer_id = 0;
  PredictOutput(torch::Tensor prob, int input_layer_id, int start_output_layer_id) : prob(prob), input_layer_id(input_layer_id), start_output_layer_id(start_output_layer_id) {}

  void slice_layer(int start, int stop) {
    prob = prob.slice(0, start, stop);
    start_output_layer_id = start_output_layer_id + start;
  }
  void slice_expert(int start, int stop) {
    prob = prob.slice(1, start, stop);
  }
  void rank_experts(int n_top_e) {
    n_top_e = std::min(n_top_e, num_output_expert());
    auto sorted = prob.sort(-1, true);
    experts = std::get<1>(sorted).slice(1, 0, n_top_e);
  }

  int num_output_layer() const { return prob.size(0); }
  int num_output_expert() const { return prob.size(1); }
  int num_top_experts() const { return experts.size(1); }
  int inner_l_to_outer_l(int inner_l) const { return start_output_layer_id + inner_l; }
  int64_t * top_experts(int inner_l) const { return experts[inner_l].data_ptr<int64_t>(); }

  static PredictOutput empty(long num_output_layer, int input_layer_id, int start_output_layer_id) {
    return PredictOutput(torch::empty({num_output_layer, 0}, torch::kFloat32), input_layer_id, start_output_layer_id);
  }
};

class PredictorBase {
 protected:
  std::shared_ptr<ModuleMeta> metas;
  PredictorBase(std::shared_ptr<ModuleMeta> metas) : metas(metas) {}
 public:
  std::shared_ptr<TimeProfiler> profiler;
  cudaStream_t compute_stream;
  virtual PredictOutput predict(int input_layer_id) = 0;
  virtual void load_model() { this->load_model_from(metas->predictor_model_path); };
  virtual void load_model_from(std::string model_path) = 0;
  virtual void add_one_layer(int layer_id, torch::Tensor experts) {};
  virtual void add_one_layer(int layer_id, int64_t *experts, size_t num_expert) {};
  virtual void record_moe_attn_logits(int layer_id, torch::Tensor attn_logits) {};
  virtual void record_moe_layer_logits(int layer_id, torch::Tensor layer_logits) {};
  virtual void end_of_one_token_prediction() {};
  virtual void start_of_new_sequence() {};
  virtual void reset_sequence_state() {}
  virtual void slice_predict_output_layer(PredictOutput &output) = 0;
  virtual bool layer_predict_enabled(int layer_id) = 0;
  virtual ~PredictorBase() = default;

  virtual int  query_predict_jobs(int layer_id) { return 1; }
  virtual PredictOutput predict_one_job(int layer_id, int job_idx) { return this->predict(layer_id); }

  static std::shared_ptr<PredictorBase> create(std::shared_ptr<ModuleMeta> metas);
};

class LegacyPredictor : public PredictorBase {
 private:
  struct PredictModel {
    torch::jit::script::Module model;
    // torch::ScalarType dtype = torch::ScalarType::Undefined;
    int orig_output_start_layer, orig_output_stop_layer;
    int slice_start, slice_stop;
    int orig_num_output_layer() const { return orig_output_stop_layer - orig_output_start_layer; }
    int output_layer_start() const { return orig_output_start_layer + slice_start; }
    int output_layer_stop() const { return orig_output_start_layer + slice_stop; }
    int output_layer(int l_in_slice) const { return orig_output_start_layer + slice_start + l_in_slice; }
    int num_output_layer() const { return slice_stop - slice_start; }
  };
  // torch::jit::script::Module predict_model;
  // single sequence for now
  torch::Tensor expert_access_buffer;
  torch::Tensor last_use_distance_buffer;
  torch::Tensor weighted_access_freq_sum_buffer;
  torch::Tensor first_moe_attn_input_logits_buffer;
  std::unordered_map<int, torch::Tensor> moe_attn_input_logits_buffer_list;
  std::unordered_map<int, torch::Tensor> moe_layer_logits_buffer_list;
  std::unordered_map<int, cudaEvent_t> logits_record_event;
  std::unordered_map<int, PredictModel> predict_models;

 private:
  void init_expert_access_buffer() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    switch (metas->predict_input_mode) {
      case kNoPredict:               { break; }
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
      case kMoeAttnInputLogits :      { break; }
      case kMoeLayerLogits :          { break; }
      default : { CHECK(false) << "Unknown predict input mode"; }
    }
  }

  void load_one_model(std::string model_path, int idx = 0);
  friend class PredictWorker;

 public:
  LegacyPredictor(std::shared_ptr<ModuleMeta> metas);
  void load_model_from(std::string model_path) override;

  PredictOutput predict(int input_layer_id) override;

  void add_one_layer(int layer_id, torch::Tensor experts) override;
  void add_one_layer(int layer_id, int64_t *experts, size_t num_expert) override;
  void record_moe_attn_logits(int layer_id, torch::Tensor attn_logits) override;
  void record_moe_layer_logits(int layer_id, torch::Tensor layer_logits) override;
  void end_of_one_token_prediction() override;
  void start_of_new_sequence() override;
  void reset_sequence_state() override;
  bool layer_predict_enabled(int layer_id) override;
  void slice_predict_output_layer(PredictOutput &output) override;

  // void clear_access_buffer() {
  //   expert_access_buffer.fill_(0);
  //   last_use_distance_buffer.fill_(0);
  //   weighted_access_freq_sum_buffer.fill_(0);
  // }
};

class SepPredictor : public PredictorBase {
 private:
  struct PredictSepModel {
    std::unordered_map<int, torch::jit::script::Module> models;
    // torch::ScalarType dtype = torch::ScalarType::Undefined;
    // int input_layer_id;
    std::vector<int> enabled_output_layers;
    int num_output_layer() const { return enabled_output_layers.size(); }
  };
  std::unordered_map<int, torch::Tensor> moe_layer_logits_buffer_list;
  std::unordered_map<int, cudaEvent_t> logits_record_event;
  std::unordered_map<int, PredictSepModel> predict_models;

 private:
  friend class PredictWorker;

 public:
  SepPredictor(std::shared_ptr<ModuleMeta> metas);
  void load_model_from(std::string model_path) override;

  PredictOutput predict(int input_layer_id) override;

  void record_moe_layer_logits(int layer_id, torch::Tensor layer_logits) override;
  void reset_sequence_state() override;
  bool layer_predict_enabled(int layer_id) override { return predict_models[layer_id].enabled_output_layers.size() > 0; }
  void slice_predict_output_layer(PredictOutput &output) override;

  int  query_predict_jobs(int layer_id) override { return predict_models[layer_id].enabled_output_layers.size(); }
  PredictOutput predict_one_job(int layer_id, int job_idx) override;
};
