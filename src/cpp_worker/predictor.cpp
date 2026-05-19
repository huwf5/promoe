#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include "predictor.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "utils.hpp"
#include "nvtx_utils.hpp"

nlohmann::json load_global_predictor_outputs(const std::string &model_path) {
  const std::string metas_path = model_path + "/metas.json";
  std::ifstream meta_file(metas_path);
  CHECK(meta_file.good()) << "Predictor directory must contain global v2 metas.json: " << metas_path;
  nlohmann::json predictor_meta = nlohmann::json::parse(meta_file);
  meta_file.close();
  CHECK(predictor_meta.value("schema_version", 1) == 2)
      << "Predictor metas.json must use schema_version 2 global ids";
  CHECK(predictor_meta.value("id_space", std::string("")) == "global")
      << "Predictor metas.json must use global id_space";
  CHECK(predictor_meta.contains("outputs") && predictor_meta["outputs"].is_object())
      << "Predictor metas.json missing global outputs";
  return predictor_meta["outputs"];
}


std::vector<std::pair<int, int>> build_predict_layer_mapping(ModuleMeta * metas) {
  std::vector<std::pair<int, int>> ret(metas->num_layer + 1, {0, 0});
  const int first_decoder = metas->first_decoder_layer();

  for (int src = 0; src <= metas->num_layer; src++) {
    if (src < first_decoder) {
      ret[src] = {0, 0};
      continue;
    }

    bool enabled = true;
    if (src < metas->num_layer) {
      enabled = ((src - first_decoder) % metas->layer_predict_interval) == 0;
    }
    if (!enabled) {
      ret[src] = {0, 0};
      continue;
    }

    // if src is the last layer, the start is the first decoder layer, otherwise it is src
    const int start = (src == metas->num_layer) ? first_decoder : src;
    int window = metas->layer_predict_max_window;
    if (src == first_decoder && metas->limit_layer_0_window != -1) {
      window = metas->limit_layer_0_window;
    }
    const int stop = std::min(start + window, metas->num_layer);
    ret[src] = {start, stop};
  }

  return ret;
}

void LegacyPredictor::add_one_layer(int layer_id, int64_t *experts, size_t num_expert) {
  switch (metas->predict_input_mode) {
    case kNoPredict:                { break; }
    case kOneToken:                 {
      for (int i = 0; i < num_expert; i++) {
        expert_access_buffer[layer_id][experts[i]] += 1;
      }
      break;
    }
    case kDecodeCumsum:             {
      for (int i = 0; i < num_expert; i++) {
        expert_access_buffer[layer_id][experts[i]] += 1;
      }
      break;
    }
    case kLastUseDistance:          { 
      last_use_distance_buffer[layer_id] += 1;
      for (int i = 0; i < num_expert; i++) {
        last_use_distance_buffer[layer_id][experts[i]] = 0;
      }
      break;
    }
    case kWeighedDecodeCumsum:      {
      weighted_access_freq_sum_buffer[layer_id] /= metas->predict_input_decay;
      for (int i = 0; i < num_expert; i++) {
        weighted_access_freq_sum_buffer[layer_id][experts[i]] += 1;
      }
      break;
    }
    case kFirstMoeAttnInputLogits : { break; }
    case kMoeAttnInputLogits :      { break; }
    case kMoeLayerLogits :          { break; }
    default : { CHECK(false) << "Unknown predict input mode"; }
  }
}

/** Legacy method to convert model dtype. Turns out it's slower on cpu */
torch::ScalarType get_jit_model_dtype(torch::jit::script::Module &model) {
  for (auto param : model.parameters()) {
    return param.dtype().toScalarType();
  }
  LOG(ERROR) << "Model has no parameters";
  return torch::ScalarType::Undefined;
}

void recursive_convert_jit_model_dtype(torch::jit::script::Module &model, torch::ScalarType dtype) {
  auto children = model.named_children();
  for (auto n : children) {
    recursive_convert_jit_model_dtype(n.value, dtype);
  }
  if (children.size() > 0) {
    return;
  }
  auto original_dtype = get_jit_model_dtype(model);
  std::unordered_map<std::string, torch::Tensor> params;
  std::unordered_map<std::string, torch::Tensor> buffers;
  for (auto param : model.named_parameters(false)) {
    params[param.name] = param.value.to(dtype);
  }
  for (auto buffer : model.named_buffers(false)) {
    buffers[buffer.name] = buffer.value.to(dtype);
  }
  for (auto &param : params) {
    model.register_parameter(param.first, param.second, false);
  }
  for (auto &buffer : buffers) {
    model.register_buffer(buffer.first, buffer.second);
  }
  auto after_dtype = get_jit_model_dtype(model);
  LOG(ERROR) << "recursive convert model dtype from " << original_dtype << " to " << dtype << ", after dtype is " << after_dtype;
  for (auto param : model.named_parameters()) {
    std::cout << param.name << " " << param.value.dtype().toScalarType() << std::endl;
  }

}

void convert_jit_model_dtype(torch::jit::script::Module &model, torch::ScalarType dtype) {
  auto original_dtype = get_jit_model_dtype(model);
  recursive_convert_jit_model_dtype(model, dtype);
  auto after_dtype = get_jit_model_dtype(model);
  LOG(ERROR) << "convert model dtype from " << original_dtype << " to " << dtype << ", after dtype is " << after_dtype;
}

PredictOutput LegacyPredictor::predict(int input_layer_id) {
  TRACE_EVENT_GURAD(kPredictor, "predict " + std::to_string(input_layer_id));
  NVTX_RANGE("predict/legacy L" + std::to_string(input_layer_id));
  LOG(DEBUG) << "predictor, predict " + std::to_string(input_layer_id);
  torch::Tensor input;
  auto model = predict_models[0].model;
  {
    TRACE_EVENT_GURAD_NAME(kPredictor, "logits copy", guardguard);
    NVTX_RANGE("predict/wait_logits L" + std::to_string(input_layer_id));
    CUDA_CALL(cudaEventSynchronize(logits_record_event[input_layer_id]));
  }
  switch (metas->predict_input_mode) {
    case kNoPredict:                {
      return PredictOutput::empty(metas->num_layer, input_layer_id, -1);
    }
    case kOneToken: { 
      CHECK(input_layer_id == 0);
      input = this->expert_access_buffer.clone();
      break;
    }
    case kDecodeCumsum: {
      CHECK(input_layer_id == 0);
      input = this->expert_access_buffer.clone();
      auto s = input.sum(1, true);
      input /= s;
      input = input.nan_to_num(0);
      break;
    }
    case kLastUseDistance: {
      CHECK(input_layer_id == 0);
      input = this->last_use_distance_buffer.clone();
      input = torch::max(
        torch::ones_like(input) * metas->predict_input_reuse_distance_max - input,
        torch::zeros_like(input)
      );
      input /= metas->predict_input_reuse_distance_max;
      break;
    }
    case kWeighedDecodeCumsum: {
      CHECK(input_layer_id == 0);
      input = this->weighted_access_freq_sum_buffer.clone();
      break;
    }
    case kFirstMoeAttnInputLogits: {
      CHECK(input_layer_id == 0);
      input = this->first_moe_attn_input_logits_buffer;
      if (input.numel() == 0) {
        LOG(DEBUG) << "skip prediction due to prefill";
        return PredictOutput::empty(metas->num_layer, input_layer_id, -1);
      } else {
        LOG_BLOCK(DEBUG, logger, {
          logger << "predictor, predict with input shape " << input.sizes() << " " << input.numel();
        });
      }
      input = input.clone().to(torch::kFloat32);
      break;
    }
    case kMoeAttnInputLogits: {
      input = this->moe_attn_input_logits_buffer_list[input_layer_id];
      if (input.numel() == 0) {
        LOG(DEBUG) << "skip prediction due to prefill";
        return PredictOutput::empty(predict_models[input_layer_id].orig_num_output_layer(), input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
      }
      if (predict_models[input_layer_id].num_output_layer() == 0) {
        LOG(DEBUG) << "skip prediction due to empty output layers";
        return PredictOutput::empty(predict_models[input_layer_id].orig_num_output_layer(), input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
      }
      LOG_BLOCK(DEBUG, logger, {
        logger << "predictor, predict with input shape " << input.sizes() << " " << input.numel();
      });
      model = predict_models[input_layer_id].model;
      input = input.clone().to(torch::kFloat32);
      break;
    }
    case kMoeLayerLogits: {
      input = this->moe_layer_logits_buffer_list[input_layer_id];
      if (input.numel() == 0) {
        LOG(DEBUG) << "skip prediction due to prefill";
        return PredictOutput::empty(predict_models[input_layer_id].orig_num_output_layer(), input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
      }
      if (predict_models[input_layer_id].num_output_layer() == 0) {
        LOG(DEBUG) << "skip prediction due to empty output layers";
        return PredictOutput::empty(predict_models[input_layer_id].orig_num_output_layer(), input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
      }
      LOG_BLOCK(DEBUG, logger, {
        logger << "predictor, predict with input shape " << input.sizes() << " " << input.numel();
      });
      model = predict_models[input_layer_id].model;
      input = input.to(torch::kFloat32);
      // input = input.toType(torch::kF16);
      break;
    }
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
  // if (metas->predict_input_mode == kDecodeCumsum) {
  //   auto s = input.sum(1, true);
  //   input /= s;
  //   input = input.nan_to_num(0);
  // } else if (metas->predict_input_mode == kLastUseDistance) {
  //   input = this->last_use_distance_buffer.clone();
  //   // predict xxx
  //   input = torch::max(
  //     torch::ones_like(input) * metas->predict_input_reuse_distance_max - input,
  //     torch::zeros_like(input)
  //   );
  //   input /= metas->predict_input_reuse_distance_max;
  // } else if (metas->predict_input_mode == kWeighedDecodeCumsum) {
  //   input = this->weighted_access_freq_sum_buffer.clone();
  // }
  Timer t;
  auto bs = input.size(0);
  std::vector<torch::jit::IValue> inputs{input.flatten(1, -1)};
  // std::vector<torch::jit::IValue> inputs{input.flatten().unsqueeze(0)};
  torch::NoGradGuard no_grad;
  torch::Tensor output;
  {
    NVTX_RANGE("predict/forward L" + std::to_string(input_layer_id));
    output = model.forward(inputs).toTensor();
  }
  output = output.reshape({bs, -1, metas->num_expert});
  output = output.sum({0});
  profiler->push(TimeProfiler::kPredictTime, t.dur_us());
  return PredictOutput(output, input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
  // return model.forward(inputs).toTensor().reshape({-1, metas->num_expert});
}
void LegacyPredictor::load_one_model(std::string model_path, int idx) {
  c10::Device cpu_device(c10::DeviceType::CPU);
  predict_models[idx].model = torch::jit::load(model_path, cpu_device);
  predict_models[idx].model.eval();
  // convert_jit_model_dtype(predict_models[idx].model, torch::kF16);
  // for (const auto& param : predict_models[idx].model.parameters()) {
  //   predict_models[idx].dtype = param.dtype().toScalarType();
  //   break;
  // }
}
void LegacyPredictor::load_model_from(std::string model_path) {
  if (metas->predict_input_mode == kNoPredict) {
    predict_models[0] = PredictModel();
    predict_models[0].orig_output_start_layer = 0;
    predict_models[0].orig_output_stop_layer = metas->num_layer;
    predict_models[0].slice_start = 0;
    predict_models[0].slice_stop = metas->num_layer;
    return;
  }
  struct stat path_stat;
  auto stat_ret = stat(model_path.c_str(), &path_stat);
  CHECK(stat_ret == 0) << "Model path not found: " << model_path;
  CHECK(S_ISDIR(path_stat.st_mode))
      << "Predictor model path must be a directory with global v2 metas.json: " << model_path;

  auto output_layers = load_global_predictor_outputs(model_path);

  DIR *dir = opendir(model_path.c_str());
  CHECK(dir != nullptr) << "Failed to open directory: " << model_path;
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_name == std::string(".") || entry->d_name == std::string("..") || entry->d_name == std::string("train_log")) {
      continue;
    }
    std::string name(entry->d_name);
    std::string file_name_without_ext = std::string(entry->d_name).substr(0, name.find_last_of("."));
    std::string file_ext = std::string(entry->d_name).substr(name.find_last_of(".") + 1);
    if (file_ext == "pt") {
      LOG(TRACE) << "Loading model: " << name << " " << file_name_without_ext;
      uint64_t model_id = std::stoull(file_name_without_ext);
      if (predict_models.find(model_id) == predict_models.end()) {
        predict_models[model_id] = PredictModel();
      }
      load_one_model(model_path + "/" + name, model_id);
    }
  }
  closedir(dir);

  for (auto &el : output_layers.items()) {
    uint64_t model_id = std::stoull(el.key());
    int start_l = el.value()[0].get<int>();
    int stop_l = el.value()[1].get<int>();
    if (start_l == stop_l && predict_models.find(model_id) == predict_models.end()) {
      predict_models[model_id] = PredictModel();
    } else {
      CHECK(predict_models.find(model_id) != predict_models.end())
          << "No legacy predictor model file for global source layer " << model_id;
    }
    predict_models[model_id].orig_output_start_layer = start_l;
    predict_models[model_id].orig_output_stop_layer  = stop_l;
  }

  auto runtime_predict_layers = build_predict_layer_mapping(metas.get());

  for (int l = 0; l <= metas->num_layer; l++) {
    CHECK(runtime_predict_layers[l].first <= runtime_predict_layers[l].second)
        << "Invalid predict layer range: " << runtime_predict_layers[l].first << " " << runtime_predict_layers[l].second;
    if (predict_models.find(l) == predict_models.end()) {
      predict_models[l] = PredictModel();
      predict_models[l].orig_output_start_layer = runtime_predict_layers[l].first;
      predict_models[l].orig_output_stop_layer = runtime_predict_layers[l].first;
    }
    auto &model = predict_models[l];
    const int slice_start_layer = std::max(model.orig_output_start_layer, runtime_predict_layers[l].first);
    const int slice_stop_layer = std::min(model.orig_output_stop_layer, runtime_predict_layers[l].second);
    if (slice_start_layer >= slice_stop_layer) {
      model.slice_start = 0;
      model.slice_stop = 0;
    } else {
      CHECK(model.orig_output_start_layer <= slice_start_layer)
          << "Invalid predict layer range: " << slice_start_layer << " " << slice_stop_layer;
      CHECK(model.orig_output_stop_layer >= slice_stop_layer)
          << "Invalid predict layer range: " << slice_start_layer << " " << slice_stop_layer;
      model.slice_start = slice_start_layer - model.orig_output_start_layer;
      model.slice_stop = slice_stop_layer - model.orig_output_start_layer;
    }
    LOG(ERROR) << "predict model " << l << ", "
               << "orig [" << model.orig_output_start_layer << ":" << model.orig_output_stop_layer << "], "
               << "slice [" << model.slice_start << ":" << model.slice_stop << "], "
               << "into [" << model.output_layer_start() << ":" << model.output_layer_stop() << "]";
  }
}
bool LegacyPredictor::layer_predict_enabled(int layer_id) {
  if (layer_id < 0 || layer_id > metas->num_layer) {
    return false;
  }
  if (predict_models.find(layer_id) == predict_models.end()) {
    return false;
  }
  return predict_models[layer_id].num_output_layer() > 0;
}

void LegacyPredictor::add_one_layer(int layer_id, torch::Tensor experts) {
  add_one_layer(layer_id, experts.data_ptr<int64_t>(), experts.numel());
}
void LegacyPredictor::reset_sequence_state() {
  for (int layer_id = 0; layer_id <= metas->num_layer; layer_id++) {
    if (logits_record_event[layer_id] != 0) {
      CUDA_CALL(cudaEventSynchronize(logits_record_event[layer_id]));
    }
  }
  if (expert_access_buffer.defined()) expert_access_buffer.zero_();
  if (last_use_distance_buffer.defined()) last_use_distance_buffer.zero_();
  if (weighted_access_freq_sum_buffer.defined()) weighted_access_freq_sum_buffer.zero_();
  first_moe_attn_input_logits_buffer = torch::empty({0});
  moe_attn_input_logits_buffer_list.clear();
  moe_layer_logits_buffer_list.clear();
}
void LegacyPredictor::start_of_new_sequence() {
  LOG(DEBUG) << "predictor, start_of_new_sequence";
  switch (metas->predict_input_mode) {
    case kNoPredict:               { break; }
    case kOneToken:                { break; }
    case kDecodeCumsum:            { expert_access_buffer.fill_(0); break; }
    case kLastUseDistance:         { last_use_distance_buffer.fill_(0); break; }
    case kWeighedDecodeCumsum:     { weighted_access_freq_sum_buffer.fill_(0); break; }
    case kFirstMoeAttnInputLogits: { break; }
    case kMoeAttnInputLogits:      { break; }
    case kMoeLayerLogits:          { break; }
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
}
void LegacyPredictor::end_of_one_token_prediction() {
  LOG(DEBUG) << "predictor, end_of_one_token_prediction";
  switch (metas->predict_input_mode) {
    case kNoPredict:               { break; }
    case kOneToken:                { expert_access_buffer.fill_(0); break;}
    case kDecodeCumsum:            { break;}
    case kLastUseDistance:         { break;}
    case kWeighedDecodeCumsum:     { break;}
    case kFirstMoeAttnInputLogits: { break;}
    case kMoeAttnInputLogits:      { break;}
    case kMoeLayerLogits:          { break;}
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
}
void LegacyPredictor::record_moe_attn_logits(int layer_id, torch::Tensor attn_logits) {
  TRACE_EVENT_GURAD(kHook, "record_moe_attn_logits " + std::to_string(layer_id));
  LOG(DEBUG) << "predictor, record_moe_attn_logits " << layer_id;
  switch (metas->predict_input_mode) {
    case kNoPredict:               { break; }
    case kOneToken:            { break; }
    case kDecodeCumsum:        { break; }
    case kLastUseDistance:     { break; }
    case kWeighedDecodeCumsum: { break; }
    case kFirstMoeAttnInputLogits: {
      if (layer_id == 0) {
        if (attn_logits.numel() == attn_logits.size(-1)) {
          LOG(DEBUG) << "predictor, record attn logits";
          first_moe_attn_input_logits_buffer = torch::empty_like(attn_logits, attn_logits.options().device(torch::kCPU).pinned_memory(true));
          {
            NVTX_RANGE("logits/attn_d2h L" + std::to_string(layer_id) + " bytes=" + std::to_string(attn_logits.nbytes()));
            CUDA_CALL(cudaMemcpyAsync(first_moe_attn_input_logits_buffer.data_ptr(), attn_logits.data_ptr(), attn_logits.nbytes(), cudaMemcpyDeviceToHost, this->compute_stream));
          }
          CUDA_CALL(cudaEventRecord(logits_record_event[layer_id], this->compute_stream));
          // first_moe_attn_input_logits_buffer = attn_logits.to("cpu");
        } else {
          LOG(DEBUG) << "predictor, skip record due to prefill";
          first_moe_attn_input_logits_buffer = torch::empty({0});
        }
      }
      break;
    }
    case kMoeAttnInputLogits: {
      CHECK(attn_logits.dim() == 3) << "input logits must be in shape [num_batch, seq_len, num_expert]";
      CHECK(attn_logits.size(0) == 1) << "batch > 1 not supported";
      const int first_decoder = metas->first_decoder_layer();
      if (layer_id < first_decoder || layer_id > metas->num_layer) {
        LOG(DEBUG) << "predictor, skip record due to non-decoder boundary";
        moe_attn_input_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (layer_id < metas->num_layer && ((layer_id - first_decoder) % metas->layer_predict_interval != 0)) {
        LOG(DEBUG) << "predictor, skip record due to interval";
        moe_attn_input_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (attn_logits.size(1) != 1) {
        LOG(DEBUG) << "predictor, skip record due to prefill";
        moe_attn_input_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      LOG(DEBUG) << "predictor, record attn logits";
      moe_attn_input_logits_buffer_list[layer_id] = torch::empty_like(attn_logits, attn_logits.options().device(torch::kCPU).pinned_memory(true));
      {
        NVTX_RANGE("logits/attn_d2h L" + std::to_string(layer_id) + " bytes=" + std::to_string(attn_logits.nbytes()));
        CUDA_CALL(cudaMemcpyAsync(moe_attn_input_logits_buffer_list[layer_id].data_ptr(), attn_logits.data_ptr(), attn_logits.nbytes(), cudaMemcpyDeviceToHost, this->compute_stream));
      }
      CUDA_CALL(cudaEventRecord(logits_record_event[layer_id], this->compute_stream));
      // moe_attn_input_logits_buffer_list[layer_id] = attn_logits.to("cpu");
      break;
    }
    case kMoeLayerLogits: { break; }
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
}

void LegacyPredictor::record_moe_layer_logits(int layer_id, torch::Tensor layer_logits) {
  TRACE_EVENT_GURAD(kHook, "record_moe_layer_logits " + std::to_string(layer_id));
  LOG(DEBUG) << "predictor, record_moe_layer_logits " << layer_id;
  switch (metas->predict_input_mode) {
    case kNoPredict:               { break; }
    case kOneToken:                { break; }
    case kDecodeCumsum:            { break; }
    case kLastUseDistance:         { break; }
    case kWeighedDecodeCumsum:     { break; }
    case kFirstMoeAttnInputLogits: { break; }
    case kMoeAttnInputLogits:      { break; }
    case kMoeLayerLogits: {
      CHECK(layer_logits.dim() == 3) << "moe layer predictor input must be in shape [num_batch, seq_len, feature_dim], but found " << layer_logits.sizes();
      // CHECK(layer_logits.size(0) == 1) << "batch > 1 not supported";
      if (layer_logits.size(1) == 0) {
        LOG(ERROR) << "predictor, skip record due to empty";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (layer_logits.size(1) != 1) {
        LOG(DEBUG) << "predictor, skip record due to prefill";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      const int first_decoder = metas->first_decoder_layer();
      if (layer_id < first_decoder || layer_id > metas->num_layer) {
        LOG(DEBUG) << "predictor, skip record due to non-decoder boundary";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (layer_id < metas->num_layer && ((layer_id - first_decoder) % metas->layer_predict_interval != 0)) {
        LOG(DEBUG) << "predictor, skip record due to interval";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (metas->layer_predict_replace_first_input_with_last_output && layer_id == metas->first_decoder_layer()) {
        LOG(DEBUG) << "predictor, skip record due to replace_first_input_with_last_output";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      LOG(DEBUG) << "predictor, record moe layer logits";
      moe_layer_logits_buffer_list[layer_id] = torch::empty_like(layer_logits, layer_logits.options().device(torch::kCPU).pinned_memory(true));
      {
        NVTX_RANGE("logits/layer_d2h L" + std::to_string(layer_id) + " bytes=" + std::to_string(layer_logits.nbytes()));
        CUDA_CALL(cudaMemcpyAsync(moe_layer_logits_buffer_list[layer_id].data_ptr(), layer_logits.data_ptr(), layer_logits.nbytes(), cudaMemcpyDeviceToHost, this->compute_stream));
      }
      CUDA_CALL(cudaEventRecord(logits_record_event[layer_id], this->compute_stream));
      // moe_layer_logits_buffer_list[layer_id] = layer_logits.to("cpu");
      break;
    }
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
}
LegacyPredictor::LegacyPredictor(std::shared_ptr<ModuleMeta> metas) : PredictorBase(metas) {
  init_expert_access_buffer();
  for (int l = 0; l <= metas->num_layer; l++) {
    logits_record_event[l] = 0;
    CUDA_CALL(cudaEventCreateWithFlags(&logits_record_event[l], cudaEventDisableTiming));
  }
}
void LegacyPredictor::slice_predict_output_layer(PredictOutput &output) {
  const auto & p_m_metas = predict_models[output.input_layer_id];
  LOG_BLOCK(DEBUG, logger, {
    logger << "predict worker: predict " << output.input_layer_id << " " << output.prob.sizes() << ", slice it with [" << p_m_metas.slice_start << ":" << p_m_metas.slice_stop << "]";
  });
  output.slice_layer(p_m_metas.slice_start, p_m_metas.slice_stop);
  CHECK(output.predictor_start_output_layer_id() == p_m_metas.output_layer_start());
}

PredictOutput SepPredictor::predict(int input_layer_id) {
  TRACE_EVENT_GURAD(kPredictor, "predict " + std::to_string(input_layer_id));
  NVTX_RANGE("predict/sep L" + std::to_string(input_layer_id));
  LOG(DEBUG) << "predictor, predict " + std::to_string(input_layer_id);
  CHECK(layer_predict_enabled(input_layer_id)) << "layer " << input_layer_id << " not enabled";
  torch::Tensor input;
  auto * model = &predict_models[0];
  {
    TRACE_EVENT_GURAD_NAME(kPredictor, "logits copy", guardguard);
    NVTX_RANGE("predict/wait_logits L" + std::to_string(input_layer_id));
    CUDA_CALL(cudaEventSynchronize(logits_record_event[input_layer_id]));
  }
  switch (metas->predict_input_mode) {
    case kNoPredict:                {
      return PredictOutput::empty(metas->num_layer, input_layer_id, -1);
    }
    case kMoeLayerLogits: {
      auto logits_it = this->moe_layer_logits_buffer_list.find(input_layer_id);
      if (logits_it == this->moe_layer_logits_buffer_list.end() || !logits_it->second.defined() || logits_it->second.numel() == 0) {
        LOG(DEBUG) << "skip prediction due to missing or empty logits";
        return PredictOutput::empty(
          predict_models[input_layer_id].num_output_layer(),
          input_layer_id,
          predict_models[input_layer_id].enabled_output_layers[0]
        );
      }
      input = logits_it->second;
      LOG_BLOCK(DEBUG, logger, {
        logger << "predictor, predict with input shape " << input.sizes() << " " << input.numel();
      });
      model = &predict_models[input_layer_id];
      // input = input.to(torch::kF16);
      input = input.to(torch::kFloat32);
      break;
    }
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
  Timer t;
  auto bs = input.size(0);
  std::vector<torch::jit::IValue> inputs{input.flatten(1, -1)};
  // std::vector<torch::jit::IValue> inputs{input.flatten().unsqueeze(0)};
  torch::NoGradGuard no_grad;
  std::vector<torch::Tensor> per_layer_outputs;
  for (auto output_layer_id : model->enabled_output_layers) {
    auto& m = model->models[output_layer_id];
    torch::Tensor output;
    {
      NVTX_RANGE("predict/forward inL" + std::to_string(input_layer_id) + " outL" + std::to_string(output_layer_id));
      output = m.forward(inputs).toTensor().reshape({bs, -1, metas->num_expert});
    }
    per_layer_outputs.push_back(output);
  }
  torch::Tensor output;
  if (per_layer_outputs.size() == 1) {
    output = per_layer_outputs[0];
  } else {
    output = torch::cat(per_layer_outputs, 1);
  }
  if (output.size(0) > 1  ) {
    output = output.sum({0});
  } else {
    output = output.reshape({-1, metas->num_expert});
  }
  profiler->push(TimeProfiler::kPredictTime, t.dur_us());
  return PredictOutput(output, input_layer_id, predict_models[input_layer_id].enabled_output_layers[0]);
}

PredictOutput SepPredictor::predict_one_job(int input_layer_id, int job_idx) {
  TRACE_EVENT_GURAD(kPredictor, "predict_one_job " + std::to_string(input_layer_id));
  NVTX_RANGE("predict/job inL" + std::to_string(input_layer_id) + " job" + std::to_string(job_idx));
  LOG(DEBUG) << "predictor, predict_one_job " + std::to_string(input_layer_id);
  CHECK(layer_predict_enabled(input_layer_id)) << "layer " << input_layer_id << " not enabled";
  torch::Tensor input;
  auto * model = &predict_models[0];
  if (job_idx == 0) {
    TRACE_EVENT_GURAD_NAME(kPredictor, "logits copy", guardguard);
    NVTX_RANGE("predict/wait_logits L" + std::to_string(input_layer_id));
    CUDA_CALL(cudaEventSynchronize(logits_record_event[input_layer_id]));
  }
  switch (metas->predict_input_mode) {
    case kNoPredict:                {
      return PredictOutput::empty(1, input_layer_id, -1);
    }
    case kMoeLayerLogits: {
      auto logits_it = this->moe_layer_logits_buffer_list.find(input_layer_id);
      if (logits_it == this->moe_layer_logits_buffer_list.end() || !logits_it->second.defined() || logits_it->second.numel() == 0) {
        LOG(DEBUG) << "skip prediction due to missing or empty logits";
        return PredictOutput::empty(
          1,
          input_layer_id,
          predict_models[input_layer_id].enabled_output_layers[job_idx]
        );
      }
      if (logits_it->second.dtype() != torch::kFloat32) {
        logits_it->second = logits_it->second.to(torch::kFloat32);
      }
      input = logits_it->second;
      CHECK(predict_models[input_layer_id].num_output_layer() > 0);
      LOG_BLOCK(DEBUG, logger, {
        logger << "predictor, predict with input shape " << input.sizes() << " " << input.numel();
      });
      model = &predict_models[input_layer_id];
      // input = input.to(torch::kFloat32);
      break;
    }
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
  Timer t;
  auto bs = input.size(0);
  std::vector<torch::jit::IValue> inputs{input.flatten(1, -1)};
  torch::NoGradGuard no_grad;
  auto output_layer_id = model->enabled_output_layers[job_idx];
  auto& m = model->models[output_layer_id];
  torch::Tensor output;
  {
    NVTX_RANGE("predict/forward inL" + std::to_string(input_layer_id) + " outL" + std::to_string(output_layer_id));
    output = m.forward(inputs).toTensor().reshape({bs, -1, metas->num_expert});
  }
  if (output.size(0) > 1  ) {
    output = output.sum({0});
  } else {
    output = output.reshape({-1, metas->num_expert});
  }
  profiler->push(TimeProfiler::kPredictTime, t.dur_us());
  return PredictOutput(
    output,
    input_layer_id,
    predict_models[input_layer_id].enabled_output_layers[job_idx]
  );
}


void SepPredictor::load_model_from(std::string model_path) {
  struct stat path_stat;
  auto stat_ret = stat(model_path.c_str(), &path_stat);
  CHECK(stat_ret == 0) << "Model path not found: " << model_path;
  CHECK(S_ISDIR(path_stat.st_mode))
      << "Predictor model path must be a directory with global v2 metas.json: " << model_path;

  for (int l = 0; l <= metas->num_layer; l++) {
    predict_models[l] = PredictSepModel();
  }
  c10::Device cpu_device(c10::DeviceType::CPU);
  auto output_layers = load_global_predictor_outputs(model_path);
  auto runtime_predict_layers = build_predict_layer_mapping(metas.get());
  for (auto &el : output_layers.items()) {
    int src_l = std::stoi(el.key());
    CHECK(src_l >= 0 && src_l <= metas->num_layer)
        << "Predictor source layer outside global boundary range: " << src_l;
    int start_l = el.value()[0].get<int>();
    int stop_l = el.value()[1].get<int>();
    CHECK(start_l >= 0 && start_l <= stop_l && stop_l <= metas->num_layer)
        << "Predictor output range outside global layer range: [" << start_l << ":" << stop_l << ")";
    const int enabled_start_l = std::max(start_l, runtime_predict_layers[src_l].first);
    const int enabled_stop_l = std::min(stop_l, runtime_predict_layers[src_l].second);
    for (int dst_l = enabled_start_l; dst_l < enabled_stop_l; dst_l++) {
      predict_models[src_l].models[dst_l] = torch::jit::load(model_path + "/" + std::to_string(src_l) + "-" + std::to_string(dst_l) + ".pt", cpu_device);
      predict_models[src_l].models[dst_l].eval();
      predict_models[src_l].enabled_output_layers.push_back(dst_l);
      // convert_jit_model_dtype(predict_models[src_l].models[dst_l], torch::kF16);
    }

    LOG(ERROR) << "predict model " << src_l << ", "
               << " predicts [" << enabled_start_l << ":" << enabled_stop_l << ")";
    // if (predict_models[src_l].enabled_output_layers.size() > 0) {
    //   predict_models[src_l].dtype = get_jit_model_dtype(predict_models[src_l].models[predict_models[src_l].enabled_output_layers[0]]);
    // }
  }
}

void SepPredictor::record_moe_layer_logits(int layer_id, torch::Tensor layer_logits) {
  TRACE_EVENT_GURAD(kHook, "record_moe_layer_logits " + std::to_string(layer_id));
  LOG(DEBUG) << "predictor, record_moe_layer_logits " << layer_id;
  switch (metas->predict_input_mode) {
    case kNoPredict:               { break; }
    case kOneToken:                { break; }
    case kDecodeCumsum:            { break; }
    case kLastUseDistance:         { break; }
    case kWeighedDecodeCumsum:     { break; }
    case kFirstMoeAttnInputLogits: { break; }
    case kMoeAttnInputLogits:      { break; }
    case kMoeLayerLogits: {
      CHECK(layer_logits.dim() == 3) << "moe layer predictor input must be in shape [num_batch, seq_len, feature_dim], but found " << layer_logits.sizes();
      // CHECK(layer_logits.size(0) == 1) << "batch > 1 not supported";
      if (layer_logits.size(1) == 0) {
        LOG(ERROR) << "predictor, skip record due to empty";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (layer_logits.size(1) != 1) {
        LOG(DEBUG) << "predictor, skip record due to prefill";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      const int first_decoder = metas->first_decoder_layer();
      if (layer_id < first_decoder || layer_id > metas->num_layer) {
        LOG(DEBUG) << "predictor, skip record due to non-decoder boundary";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (layer_id < metas->num_layer && ((layer_id - first_decoder) % metas->layer_predict_interval != 0)) {
        LOG(DEBUG) << "predictor, skip record due to interval";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (metas->layer_predict_replace_first_input_with_last_output && layer_id == metas->first_decoder_layer()) {
        LOG(DEBUG) << "predictor, skip record due to replace_first_input_with_last_output";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      LOG(DEBUG) << "predictor, record moe layer logits";
      moe_layer_logits_buffer_list[layer_id] = torch::empty_like(layer_logits, layer_logits.options().device(torch::kCPU).pinned_memory(true));
      {
        NVTX_RANGE("logits/layer_d2h L" + std::to_string(layer_id) + " bytes=" + std::to_string(layer_logits.nbytes()));
        CUDA_CALL(cudaMemcpyAsync(moe_layer_logits_buffer_list[layer_id].data_ptr(), layer_logits.data_ptr(), layer_logits.nbytes(), cudaMemcpyDeviceToHost, this->compute_stream));
      }
      CUDA_CALL(cudaEventRecord(logits_record_event[layer_id], this->compute_stream));
      // moe_layer_logits_buffer_list[layer_id] = layer_logits.to("cpu");
      break;
    }
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
}
SepPredictor::SepPredictor(std::shared_ptr<ModuleMeta> metas) : PredictorBase(metas) {
  for (int l = 0; l <= metas->num_layer; l++) {
    logits_record_event[l] = 0;
    CUDA_CALL(cudaEventCreateWithFlags(&logits_record_event[l], cudaEventDisableTiming));
  }
}
void SepPredictor::reset_sequence_state() {
  for (int layer_id = 0; layer_id <= metas->num_layer; layer_id++) {
    if (logits_record_event[layer_id] != 0) {
      CUDA_CALL(cudaEventSynchronize(logits_record_event[layer_id]));
    }
  }
  moe_layer_logits_buffer_list.clear();
}
void SepPredictor::slice_predict_output_layer(PredictOutput &output) {
  // const auto & p_m_metas = predict_models[output.input_layer_id];
  // LOG_BLOCK(DEBUG, logger, {
  //   logger << "predict worker: predict " << output.input_layer_id << " " << output.prob.sizes() << ", slice it with [" << p_m_metas.slice_start << ":" << p_m_metas.slice_stop << "]";
  // });
  // output.slice_layer(p_m_metas.slice_start, p_m_metas.slice_stop);
  // CHECK(output.start_output_layer_id == p_m_metas.output_layer_start());
}

std::shared_ptr<PredictorBase> PredictorBase::create(std::shared_ptr<ModuleMeta> metas) {
  switch (metas->predictor_type) {
    case kLegacyPredictor: {
      return std::make_shared<LegacyPredictor>(metas);
    }
    case kSepPredictor: {
      return std::make_shared<SepPredictor>(metas);
    }
    default: {
      CHECK(false) << "Unknown predictor type";
    }
  }
}
