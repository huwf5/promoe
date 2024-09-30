#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include "predictor.hpp"
#include "logging.hpp"
#include "profiler.hpp"

void Predictor::add_one_layer(int layer_id, int64_t *experts, size_t num_expert) {
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
PredictOutput Predictor::predict(int input_layer_id) {
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
      return PredictOutput(metas->num_layer, input_layer_id, -1);
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
        return PredictOutput(metas->num_layer, input_layer_id, -1);
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
        return PredictOutput(predict_models[input_layer_id].orig_num_output_layer(), input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
      }
      if (predict_models[input_layer_id].num_output_layer() == 0) {
        LOG(DEBUG) << "skip prediction due to empty output layers";
        return PredictOutput(predict_models[input_layer_id].orig_num_output_layer(), input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
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
        return PredictOutput(predict_models[input_layer_id].orig_num_output_layer(), input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
      }
      if (predict_models[input_layer_id].num_output_layer() == 0) {
        LOG(DEBUG) << "skip prediction due to empty output layers";
        return PredictOutput(predict_models[input_layer_id].orig_num_output_layer(), input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
      }
      LOG_BLOCK(DEBUG, logger, {
        logger << "predictor, predict with input shape " << input.sizes() << " " << input.numel();
      });
      model = predict_models[input_layer_id].model;
      input = input.clone().to(torch::kFloat32);
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
  auto bs = input.size(0);
  std::vector<torch::jit::IValue> inputs{input.flatten(1, -1)};
  // std::vector<torch::jit::IValue> inputs{input.flatten().unsqueeze(0)};
  torch::NoGradGuard no_grad;
  torch::Tensor output = model.forward(inputs).toTensor();
  output = output.reshape({bs, -1, metas->num_expert});
  output = output.sum({0});
  return PredictOutput(output, input_layer_id, predict_models[input_layer_id].orig_output_start_layer);
  // return model.forward(inputs).toTensor().reshape({-1, metas->num_expert});
}
void Predictor::load_one_model(std::string model_path, int idx) {
  c10::Device cpu_device(c10::DeviceType::CPU);
  predict_models[idx].model = torch::jit::load(model_path, cpu_device);
  predict_models[idx].model.eval();
}
void Predictor::load_model(std::string model_path) {
  if (metas->predict_input_mode == kNoPredict) {
    predict_models[0] = PredictModel();
    predict_models[0].orig_output_start_layer = 0;
    predict_models[0].orig_output_stop_layer = metas->num_layer;
    predict_models[0].slice_start = 0;
    predict_models[0].slice_stop = metas->num_layer;
    layer_predict_enabled.resize(metas->num_layer + 1, false);
    layer_predict_enabled[0] = true;
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

  layer_predict_enabled.resize(metas->num_layer + 1, false);
  std::vector<int> layers_to_predict;
  for (int l = 0; l < metas->num_layer; l+= metas->layer_predict_interval) {
    layers_to_predict.push_back(l);
    layer_predict_enabled[l] = true;
  }

  if (metas->layer_predict_replace_first_input_with_last_output) {
    CHECK(metas->predict_input_mode == kMoeLayerLogits);
    CHECK(predict_models.find(metas->num_layer) != predict_models.end());
    CHECK(layers_to_predict[0] == 0);
    layers_to_predict[0] = metas->num_layer;
    layer_predict_enabled[0] = false;
    layer_predict_enabled[metas->num_layer] = true;
  }

  int stop_l = 0;
  for (auto l : layers_to_predict) {
    CHECK(predict_models.find(l) != predict_models.end()) << "No model meta for layer " << l;
    auto & p_m_metas = predict_models[l];
    CHECK(p_m_metas.orig_num_output_layer() > 0) << "No output layers for model at layer 0";
    CHECK(p_m_metas.orig_output_stop_layer <= metas->num_layer) << "Orig model predict more than num_layer";
    CHECK(p_m_metas.orig_output_start_layer <= stop_l) << "Output layers not in order";
    p_m_metas.slice_start = stop_l - p_m_metas.orig_output_start_layer;

    // orig layers: [orig_output_start_layer, orig_output_stop_layer)
    // build a slice: orig_output[slice_start:slice_stop] -> [orig_output_start_layer + slice_start, orig_output_start_layer + slice_stop)
    // ------
    // orig_output_start_layer + slice_stop <= orig_output_stop_layer
    // orig_output_start_layer + slice_stop <= l + metas->layer_predict_max_window

    p_m_metas.slice_stop = std::min<int>(p_m_metas.orig_num_output_layer(), (l % metas->num_layer) + metas->layer_predict_max_window - p_m_metas.orig_output_start_layer); 
    CHECK(p_m_metas.slice_start <= p_m_metas.slice_stop) << "No output layers for model at layer 0";
    stop_l = p_m_metas.output_layer_stop();
    LOG(ERROR) << "predict model " << l << ", "
               << "orig [" << p_m_metas.orig_output_start_layer << ":" << p_m_metas.orig_output_stop_layer << "], "
               << "slice [" << p_m_metas.slice_start << ":" << p_m_metas.slice_stop << "], "
               << "into [" << p_m_metas.output_layer_start() << ":" << p_m_metas.output_layer_stop() << "]";
  }
}
void Predictor::add_one_layer(int layer_id, torch::Tensor experts) {
  add_one_layer(layer_id, experts.data_ptr<int64_t>(), experts.numel());
}
void Predictor::start_of_new_sequence() {
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
void Predictor::end_of_one_token_prediction() {
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
void Predictor::record_moe_attn_logits(int layer_id, torch::Tensor attn_logits) {
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

void Predictor::record_moe_layer_logits(int layer_id, torch::Tensor layer_logits) {
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
Predictor::Predictor(std::shared_ptr<ModuleMeta> metas) : metas(metas) {
  init_expert_access_buffer();
  for (int l = 0; l <= metas->num_layer; l++) {
    logits_record_event[l] = 0;
    CUDA_CALL(cudaEventCreateWithFlags(&logits_record_event[l], cudaEventDisableTiming));
  }
}
