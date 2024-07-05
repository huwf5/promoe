#include "predictor.hpp"

#include "logging.hpp"

#include "profiler.hpp"

void Predictor::add_one_layer(int layer_id, int64_t *experts, size_t num_expert) {
  switch (metas->predict_input_mode) {
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
    default : { CHECK(false) << "Unknown predict input mode"; }
  }
}
torch::Tensor Predictor::predict(int input_layer_id) {
  TRACE_EVENT_GURAD(kPredictor, "predict " + std::to_string(input_layer_id));
  LOG(DEBUG) << "predictor, predict " + std::to_string(input_layer_id);
  torch::Tensor input;
  auto model = predict_model_list[0];
  switch (metas->predict_input_mode) {
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
        return torch::empty({metas->num_layer, 0}, torch::kFloat32);
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
        return torch::empty({metas->layer_predict_window, 0}, torch::kFloat32);
      }
      LOG_BLOCK(DEBUG, logger, {
        logger << "predictor, predict with input shape " << input.sizes() << " " << input.numel();
      });
      model = predict_model_list[input_layer_id];
      input = input.clone().to(torch::kFloat32);
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
  std::vector<torch::jit::IValue> inputs{input.flatten().unsqueeze(0)};
  return model.forward(inputs).toTensor().reshape({-1, metas->num_expert});
}
void Predictor::load_one_model(std::string model_path, int idx) {
  c10::Device cpu_device(c10::DeviceType::CPU);
  if (predict_model_list.size() < idx + 1) {
    predict_model_list.resize(idx + 1);
  }
  predict_model_list[idx] = torch::jit::load(model_path, cpu_device);
  predict_model_list[idx].eval();
}
void Predictor::load_model(std::string model_path) {
  struct stat path_stat;
  auto stat_ret = stat(model_path.c_str(), &path_stat);
  CHECK(stat_ret == 0) << "Model file not found: " << model_path;
  if (S_ISREG(path_stat.st_mode)) {
    load_one_model(model_path, 0);
  } else if (S_ISDIR(path_stat.st_mode)) {
    DIR *dir = opendir(model_path.c_str());
    CHECK(dir != nullptr) << "Failed to open directory: " << model_path;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      if (entry->d_name == std::string(".") || entry->d_name == std::string("..")) {
        continue;
      }
      std::string name(entry->d_name);
      std::string file_name_without_ext = std::string(entry->d_name).substr(0, name.find_last_of("."));
      LOG(ERROR) << "Loading model: " << name << " " << file_name_without_ext;
      load_one_model(model_path + "/" + name, std::stoi(file_name_without_ext));
    }
    closedir(dir);
  } else {
    CHECK(false) << "Model path is not a regular file or directory: "
                 << model_path;
  }
}
void Predictor::add_one_layer(int layer_id, torch::Tensor experts) {
  add_one_layer(layer_id, experts.data_ptr<int64_t>(), experts.numel());
}
void Predictor::start_of_new_sequence() {
  LOG(DEBUG) << "predictor, start_of_new_sequence";
  switch (metas->predict_input_mode) {
    case kOneToken:                { break; }
    case kDecodeCumsum:            { expert_access_buffer.fill_(0); break; }
    case kLastUseDistance:         { last_use_distance_buffer.fill_(0); break; }
    case kWeighedDecodeCumsum:     { weighted_access_freq_sum_buffer.fill_(0); break; }
    case kFirstMoeAttnInputLogits: { break; }
    case kMoeAttnInputLogits:      { break; }
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
}
void Predictor::end_of_one_token_prediction() {
  LOG(DEBUG) << "predictor, end_of_one_token_prediction";
  switch (metas->predict_input_mode) {
    case kOneToken:                { expert_access_buffer.fill_(0); break;}
    case kDecodeCumsum:            { break;}
    case kLastUseDistance:         { break;}
    case kWeighedDecodeCumsum:     { break;}
    case kFirstMoeAttnInputLogits: { break;}
    case kMoeAttnInputLogits:      { break;}
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
}
void Predictor::record_moe_attn_logits(int layer_id, torch::Tensor attn_logits) {
  TRACE_EVENT_GURAD(kPredictor, "record_moe_attn_logits " + std::to_string(layer_id));
  LOG(DEBUG) << "predictor, record_moe_attn_logits " << layer_id;
  switch (metas->predict_input_mode) {
    case kOneToken:            { break; }
    case kDecodeCumsum:        { break; }
    case kLastUseDistance:     { break; }
    case kWeighedDecodeCumsum: { break; }
    case kFirstMoeAttnInputLogits: {
      if (layer_id == 0) {
        if (attn_logits.numel() == attn_logits.size(-1)) {
          LOG(DEBUG) << "predictor, record attn logits";
          first_moe_attn_input_logits_buffer = attn_logits.to("cpu");
        } else {
          LOG(DEBUG) << "predictor, skip record due to prefill";
          first_moe_attn_input_logits_buffer = torch::empty({0});
        }
      }
      break;
    }
    case kMoeAttnInputLogits: {
      if (attn_logits.numel() == attn_logits.size(-1)) {
        LOG(DEBUG) << "predictor, record attn logits";
        moe_attn_input_logits_buffer_list[layer_id] = attn_logits.to("cpu");
      } else {
        LOG(DEBUG) << "predictor, skip record due to prefill";
        moe_attn_input_logits_buffer_list[layer_id] = torch::empty({0});
      }
      break;
    }
    default: {
      CHECK(false) << "Unknown predict input mode";
    }
  }
}
