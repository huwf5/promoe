#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include "predictor.hpp"
#include "logging.hpp"
#include "profiler.hpp"
#include "utils.hpp"

std::vector<std::pair<int, int>> build_predict_layer_mapping(ModuleMeta * metas) {
  std::vector<int> layers_to_predict;
  for (int l = 0; l < metas->num_layer; l+= metas->layer_predict_interval) {
    layers_to_predict.push_back(l);
  }

  std::vector<int> predict_layers(metas->num_layer + 1, 0);

  int stop_l = 0;
  for (auto l : layers_to_predict) {
    auto window = metas->layer_predict_max_window;
    if (l == 0 && metas->limit_layer_0_window != -1) {
      window = metas->limit_layer_0_window;
    }
    predict_layers[l] = stop_l;
    predict_layers[l + 1] = (l % metas->num_layer) + window;
    predict_layers[l + 1] = std::min(predict_layers[l + 1], metas->num_layer);
    stop_l = predict_layers[l + 1];

    LOG(ERROR) << "predict model " << l << ", "
               << "predicts [" << predict_layers[l] << ":" << predict_layers[l + 1] << ")";
  }

  for (int l = 1; l <= metas->num_layer; l++) {
    if (predict_layers[l] < predict_layers[l - 1]) {
      predict_layers[l] = predict_layers[l - 1];
    }
  }

  std::vector<std::pair<int, int>> ret(metas->num_layer + 1, {0, 0});
  for (int l = 0; l < metas->num_layer; l++) {
    ret[l] = {predict_layers[l], predict_layers[l + 1]};
  }

  if (metas->layer_predict_replace_first_input_with_last_output) {
    CHECK(metas->predict_input_mode == kMoeLayerLogits);
    ret[metas->num_layer] = ret[0];
    ret[0] = {0, 0};
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
  LOG(DEBUG) << "predictor, predict " + std::to_string(input_layer_id);
  torch::Tensor input;
  auto model = predict_models[0].model;
  {
    TRACE_EVENT_GURAD_NAME(kPredictor, "logits copy", guardguard);
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
  torch::Tensor output = model.forward(inputs).toTensor();
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
void LegacyPredictor::load_model(std::string model_path) {
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
  CHECK(stat_ret == 0) << "Model file not found: " << model_path;
  if (S_ISREG(path_stat.st_mode)) {
    load_one_model(model_path, 0);
    CHECK(predict_models.size() == 1);
    predict_models[0] = PredictModel();
    predict_models[0].orig_output_start_layer = 0;
    predict_models[0].orig_output_stop_layer = metas->num_layer;
  } else if (S_ISDIR(path_stat.st_mode)) {
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
      } else if (file_ext == "json") {
        LOG(ERROR) << "Loading json: " << name;
        std::ifstream trace_file(model_path + "/" + name);
        nlohmann::json output_layers_list = nlohmann::json::parse(trace_file);
        trace_file.close();
        for (auto &el : output_layers_list.items()) {
          uint64_t model_id = std::stoull(el.key());
          if (predict_models.find(model_id) == predict_models.end()) {
            predict_models[model_id] = PredictModel();
          }
          predict_models[model_id].orig_output_start_layer = el.value()[0].get<int>();
          predict_models[model_id].orig_output_stop_layer  = el.value()[1].get<int>();
        }
      }
    }
    closedir(dir);
  } else {
    CHECK(false) << "Model path is not a regular file or directory: "
                 << model_path;
  }

  auto predict_layers = build_predict_layer_mapping(metas.get());

  for (int l = 0; l < metas->num_layer + 1; l++) {
    CHECK(predict_models.find(l) != predict_models.end()) << "No model meta for layer " << l;
    auto &model = predict_models[l];
    CHECK(predict_layers[l].first <= predict_layers[l].second) << "Invalid predict layer range: " << predict_layers[l].first << " " << predict_layers[l].second;
    if (predict_layers[l].first == predict_layers[l].second) {
      model.slice_start = 0;
      model.slice_stop = 0;
    } else {
      CHECK(model.orig_output_start_layer <= predict_layers[l].first) << "Invalid predict layer range: " << predict_layers[l].first << " " << predict_layers[l].second;
      CHECK(model.orig_output_stop_layer >= predict_layers[l].second) << "Invalid predict layer range: " << predict_layers[l].first << " " << predict_layers[l].second;
      model.slice_start = predict_layers[l].first - model.orig_output_start_layer;
      model.slice_stop = predict_layers[l].second - model.orig_output_start_layer;
    }
    LOG(ERROR) << "predict model " << l << ", "
               << "orig [" << model.orig_output_start_layer << ":" << model.orig_output_stop_layer << "], "
               << "slice [" << model.slice_start << ":" << model.slice_stop << "], "
               << "into [" << model.output_layer_start() << ":" << model.output_layer_stop() << "]";
  }
}

bool LegacyPredictor::layer_predict_enabled(int layer_id) {
  if (predict_models.find(layer_id) == predict_models.end()) {
    return false;
  }
  return predict_models[layer_id].num_output_layer() > 0;
}

void LegacyPredictor::add_one_layer(int layer_id, torch::Tensor experts) {
  add_one_layer(layer_id, experts.data_ptr<int64_t>(), experts.numel());
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
          CUDA_CALL(cudaMemcpyAsync(first_moe_attn_input_logits_buffer.data_ptr(), attn_logits.data_ptr(), attn_logits.nbytes(), cudaMemcpyDeviceToHost, this->compute_stream));
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
      if (layer_id % metas->layer_predict_interval != 0) {
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
      CUDA_CALL(cudaMemcpyAsync(moe_attn_input_logits_buffer_list[layer_id].data_ptr(), attn_logits.data_ptr(), attn_logits.nbytes(), cudaMemcpyDeviceToHost, this->compute_stream));
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
      CHECK(layer_logits.dim() == 3) << "input logits must be in shape [num_batch, seq_len, num_expert], but found " << layer_logits.sizes();
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
      if ((layer_id % metas->num_layer) % metas->layer_predict_interval != 0) {
        LOG(DEBUG) << "predictor, skip record due to interval";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (metas->layer_predict_replace_first_input_with_last_output && layer_id == 0) {
        LOG(DEBUG) << "predictor, skip record due to replace_first_input_with_last_output";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      LOG(DEBUG) << "predictor, record attn logits";
      moe_layer_logits_buffer_list[layer_id] = torch::empty_like(layer_logits, layer_logits.options().device(torch::kCPU).pinned_memory(true));
      CUDA_CALL(cudaMemcpyAsync(moe_layer_logits_buffer_list[layer_id].data_ptr(), layer_logits.data_ptr(), layer_logits.nbytes(), cudaMemcpyDeviceToHost, this->compute_stream));
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
  CHECK(output.start_output_layer_id == p_m_metas.output_layer_start());
}

PredictOutput SepPredictor::predict(int input_layer_id) {
  TRACE_EVENT_GURAD(kPredictor, "predict " + std::to_string(input_layer_id));
  LOG(DEBUG) << "predictor, predict " + std::to_string(input_layer_id);
  CHECK(layer_predict_enabled(input_layer_id)) << "layer " << input_layer_id << " not enabled";
  torch::Tensor input;
  auto * model = &predict_models[0];
  {
    TRACE_EVENT_GURAD_NAME(kPredictor, "logits copy", guardguard);
    CUDA_CALL(cudaEventSynchronize(logits_record_event[input_layer_id]));
  }
  switch (metas->predict_input_mode) {
    case kNoPredict:                {
      return PredictOutput::empty(metas->num_layer, input_layer_id, -1);
    }
    case kMoeLayerLogits: {
      input = this->moe_layer_logits_buffer_list[input_layer_id];
      if (input.numel() == 0) {
        LOG(DEBUG) << "skip prediction due to prefill";
        return PredictOutput::empty(predict_models[input_layer_id].num_output_layer(), input_layer_id, predict_models[input_layer_id].enabled_output_layers[0]);
      }
      if (predict_models[input_layer_id].num_output_layer() == 0) {
        LOG(ERROR) << "skip prediction due to empty output layers";
        return PredictOutput::empty(predict_models[input_layer_id].num_output_layer(), input_layer_id, predict_models[input_layer_id].enabled_output_layers[0]);
      }
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
    torch::Tensor output = m.forward(inputs).toTensor().reshape({bs, -1, metas->num_expert});
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
  LOG(DEBUG) << "predictor, predict_one_job " + std::to_string(input_layer_id);
  CHECK(layer_predict_enabled(input_layer_id)) << "layer " << input_layer_id << " not enabled";
  torch::Tensor input;
  auto * model = &predict_models[0];
  if (job_idx == 0) {
    TRACE_EVENT_GURAD_NAME(kPredictor, "logits copy", guardguard);
    CUDA_CALL(cudaEventSynchronize(logits_record_event[input_layer_id]));
  }
  switch (metas->predict_input_mode) {
    case kNoPredict:                {
      return PredictOutput::empty(1, input_layer_id, -1);
    }
    case kMoeLayerLogits: {
      if (this->moe_layer_logits_buffer_list[input_layer_id].dtype() != torch::kFloat32) {
        this->moe_layer_logits_buffer_list[input_layer_id] = this->moe_layer_logits_buffer_list[input_layer_id].to(torch::kFloat32);
      }
      input = this->moe_layer_logits_buffer_list[input_layer_id];
      if (input.numel() == 0) {
        LOG(DEBUG) << "skip prediction due to prefill";
        return PredictOutput::empty(1, input_layer_id, predict_models[input_layer_id].enabled_output_layers[job_idx]);
      }
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
  torch::Tensor output = m.forward(inputs).toTensor().reshape({bs, -1, metas->num_expert});
  if (output.size(0) > 1  ) {
    output = output.sum({0});
  } else {
    output = output.reshape({-1, metas->num_expert});
  }
  profiler->push(TimeProfiler::kPredictTime, t.dur_us());
  return PredictOutput(output, input_layer_id, predict_models[input_layer_id].enabled_output_layers[job_idx]);
}


void SepPredictor::load_model(std::string model_path) {
  struct stat path_stat;
  auto stat_ret = stat(model_path.c_str(), &path_stat);
  CHECK(stat_ret == 0) << "Model file not found: " << model_path;
  CHECK(S_ISDIR(path_stat.st_mode)) << "Model file is not a directory: " << model_path;
  // fixme: meta json?

  for (int l = 0; l < metas->num_layer + 1; l++) {
    predict_models[l] = PredictSepModel();
  }
  c10::Device cpu_device(c10::DeviceType::CPU);
  auto predict_layers = build_predict_layer_mapping(metas.get());
  for (int src_l = 0; src_l < metas->num_layer + 1; src_l++) {
    for (int dst_l = predict_layers[src_l].first; dst_l < predict_layers[src_l].second; dst_l++) {
      predict_models[src_l].models[dst_l] = torch::jit::load(model_path + "/" + std::to_string(src_l) + "-" + std::to_string(dst_l) + ".pt", cpu_device);
      predict_models[src_l].models[dst_l].eval();
      predict_models[src_l].enabled_output_layers.push_back(dst_l);
      // convert_jit_model_dtype(predict_models[src_l].models[dst_l], torch::kF16);
    }

    LOG(ERROR) << "predict model " << src_l << ", "
               << " predicts [" << predict_layers[src_l].first << ":" << predict_layers[src_l].second << ")";
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
      CHECK(layer_logits.dim() == 3) << "input logits must be in shape [num_batch, seq_len, num_expert], but found " << layer_logits.sizes();
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
      if ((layer_id % metas->num_layer) % metas->layer_predict_interval != 0) {
        LOG(DEBUG) << "predictor, skip record due to interval";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      if (metas->layer_predict_replace_first_input_with_last_output && layer_id == 0) {
        LOG(DEBUG) << "predictor, skip record due to replace_first_input_with_last_output";
        moe_layer_logits_buffer_list[layer_id] = torch::empty({0});
        break;
      }
      LOG(DEBUG) << "predictor, record attn logits";
      moe_layer_logits_buffer_list[layer_id] = torch::empty_like(layer_logits, layer_logits.options().device(torch::kCPU).pinned_memory(true));
      CUDA_CALL(cudaMemcpyAsync(moe_layer_logits_buffer_list[layer_id].data_ptr(), layer_logits.data_ptr(), layer_logits.nbytes(), cudaMemcpyDeviceToHost, this->compute_stream));
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