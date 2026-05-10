#include "utils.hpp" 
#include "logging.hpp"


void AtomicMultiStatusLock::unlock(ExpertStatus from, ExpertStatus to) {
  int actual_cur_status = lock_.exchange(to);
  CHECK(actual_cur_status == from) << actual_cur_status << "!=" << from;
}
bool AtomicMultiStatusLock::try_unlock(ExpertStatus from, ExpertStatus to) {
  int from_ = from, to_ = to;
  return lock_.compare_exchange_strong(from_, to_);
  // return (actual_cur_status == from);
}
void AtomicMultiStatusLock::lock(ExpertStatus from, ExpertStatus to) {
  int from_ = from, to_ = to;
  while (lock_.compare_exchange_strong(from_, to_) == false) {
    from_ = from;
  };
}
bool AtomicMultiStatusLock::is_locked(ExpertStatus locked_status) {
  return lock_.load() == locked_status;
}

void AtomicLock::unlock() {
  bool assume_cur_status = true;
  CHECK(lock_.compare_exchange_strong(assume_cur_status, false));
  // CHECK(actual_cur_status == true);
}
void AtomicLock::lock() {
  bool cur_status = false;
  while (lock_.compare_exchange_strong(cur_status, true) == false) {
    cur_status = false;
  };
}
// bool AtomicLock::is_locked() { return lock_.load(); }

std::string tensor_to_str(torch::Tensor t) {
  std::stringstream ss;
  t = t.flatten();
  for (int i = 0; i < t.numel(); i++) {
    ss << t[i].item() << ",";
  }
  return ss.str();
}

DummyStruct DummyContainer<DummyStruct>::_place_holder = DummyStruct();
void ModuleMeta::init_param_list(std::vector<std::string> params) {
  param_name_list = params;
  num_per_expert_param = params.size();
  for (int i = 0; i < num_per_expert_param; i++) {
    param_name_to_id[param_name_list[i]] = i;
  }
}
void ModuleMeta::init_from_map(std::unordered_map<std::string, std::string> config_map) {
  auto required_str = [&config_map](std::string key) -> std::string {
    if (config_map.find(key) == config_map.end()) {
      LOG(FATAL) << "required key " << key << " not found";
    }
    if (config_map[key] == "None" || config_map[key] == "") {
      LOG(FATAL) << "required key " << key << " is " << config_map[key] << ", should be a non-empty string";
    }
    auto ret = config_map[key];
    config_map.erase(key);
    return ret;
  };
  auto optional_str = [&config_map](std::string key, std::string default_val) -> std::string {
    if (config_map.find(key) == config_map.end()) {
      return default_val;
    }
    if (config_map[key] == "None" || config_map[key] == "") {
      LOG(ERROR) << "optional key " << key << " is " << config_map[key] << ", use default value " << default_val;
      config_map.erase(key);
      return default_val;
    }
    auto ret = config_map[key];
    config_map.erase(key);
    return ret;
  };
  auto optional_int   = [&optional_str](std::string key, int   default_val) -> int   { try { return std::stoi(optional_str(key, std::to_string(default_val))); } catch (const std::invalid_argument& e) { LOG(FATAL) << "invalid key " << key << " with value " << optional_str(key, std::to_string(default_val)); } };
  auto optional_float = [&optional_str](std::string key, float default_val) -> float { try { return std::stof(optional_str(key, std::to_string(default_val))); } catch (const std::invalid_argument& e) { LOG(FATAL) << "invalid key " << key << " with value " << optional_str(key, std::to_string(default_val)); } };
  auto optional_bool  = [&optional_str](std::string key, bool  default_val) -> bool  { return string_is_on(optional_str(key, default_val ? "true" : "false")); };
  auto required_int   = [&required_str](std::string key) -> int   { try { return std::stoi(required_str(key)); } catch (const std::invalid_argument& e) { LOG(FATAL) << "invalid key " << key << " with value " << required_str(key); } };
  auto required_float = [&required_str](std::string key) -> float { try { return std::stof(required_str(key)); } catch (const std::invalid_argument& e) { LOG(FATAL) << "invalid key " << key << " with value " << required_str(key); } };
  auto required_bool  = [&required_str](std::string key) -> bool  { return string_is_on(required_str(key)); };

  translate_dialect(config_map);

  // num_layer            = required_int("num_layer");
  // num_expert           = required_int("num_expert");
  model_arch_string    = optional_str("model_arch_string", model_arch_string);
  num_expert_per_token = required_int("num_expert_per_token");

  cache_rate  = optional_float("cache_rate", cache_rate);
  num_predict_expert_per_layer = optional_int("num_predict_expert_per_layer", num_predict_expert_per_layer);
  if (num_predict_expert_per_layer == -1) {
    num_predict_expert_per_layer = num_expert_per_token;
  }
  reorder_experts = optional_bool("reorder_experts", num_predict_expert_per_layer > 0);
  early_preempt   = optional_bool("early_preempt",   num_predict_expert_per_layer > 0);
  chunk_prefetch  = optional_bool("chunk_prefetch",  num_predict_expert_per_layer > 0);

  auto predict_input_mode_map = std::unordered_map<std::string, PredictInputMode>{
    {"no_predict",                  kNoPredict},
    {"one_token",                   kOneToken},
    {"decode_cumsum",               kDecodeCumsum},
    {"last_use_distance",           kLastUseDistance},
    {"weighted_decode_cumsum",      kWeighedDecodeCumsum},
    {"first_moe_attn_input_logits", kFirstMoeAttnInputLogits},
    {"moe_attn_input_logits",       kMoeAttnInputLogits},
    {"moe_layer_logits",            kMoeLayerLogits},
  };
  auto predictor_type_map = std::unordered_map<std::string, PredictorType>{
    {"legacy", kLegacyPredictor},
    {"sep",    kSepPredictor},
  };
  predict_input_mode_str = optional_str("predict_input_mode", predict_input_mode_str);
  predict_input_mode     = predict_input_mode_map[predict_input_mode_str];
  predictor_type_str     = optional_str("predictor_type", predictor_type_str);
  predictor_type         = predictor_type_map[predictor_type_str];

  predictor_model_path     = optional_str("predictor_model_path", predictor_model_path);
  predictor_num_layer      = optional_int("predictor_num_layer", predictor_num_layer);
  predictor_layer_offset   = optional_int("predictor_layer_offset", predictor_layer_offset);
  layer_predict_interval   = optional_int("layer_predict_interval",   layer_predict_interval);
  layer_predict_max_window = optional_int("layer_predict_max_window", layer_predict_max_window);
  layer_predict_replace_first_input_with_last_output = optional_bool("layer_predict_replace_first_input_with_last_output", layer_predict_replace_first_input_with_last_output);

  limit_layer_0_window      = optional_int("limit_layer_0_window",      limit_layer_0_window);
  limit_layer_0_num_predict = optional_int("limit_layer_0_num_predict", limit_layer_0_num_predict);

  max_prefetch_layer_distance      = optional_int  ("max_prefetch_layer_distance",      max_prefetch_layer_distance);
  cache_only                       = optional_bool ("cache_only",                       cache_only);
  per_layer_cache                  = optional_bool ("per_layer_cache",                  per_layer_cache);
  promote_hit_in_prefetch          = optional_bool ("promote_hit_in_prefetch",          promote_hit_in_prefetch);
  cache_policy                     = optional_str  ("cache_policy",                     cache_policy);
  predict_input_reuse_distance_max = optional_int  ("predict_input_reuse_distance_max", predict_input_reuse_distance_max);
  predict_input_decay              = optional_int  ("predict_input_decay",              predict_input_decay);

  if (config_map.size() > 0) {
    std::cerr << "====================== unrecognized configs ======================\n";
    for (auto & config_pair : config_map) {
      std::cerr << config_pair.first << " = " << config_pair.second << std::endl;
    }
    std::cerr << "====================== unrecognized configs ======================\n";
    CHECK(false);
  }
}

void ModuleMeta::translate_dialect(std::unordered_map<std::string, std::string> & config_map) {
  std::unordered_map<std::string, std::string> new_config_map;
  std::unordered_map<std::string, std::string> dialects = {
    {"cross_token_pred",                "layer_predict_replace_first_input_with_last_output"},
    {"predict_using_last_token_output", "layer_predict_replace_first_input_with_last_output"},
    {"num_predict",                     "num_predict_expert_per_layer"},
    {"predict_interval",                "layer_predict_interval"},
    {"predict_window",                  "layer_predict_max_window"},
    {"moe_cache_policy",                "cache_policy"},
    {"moe_cache_rate",                  "cache_rate"},
    {"pred_model_path",                 "predictor_model_path"},
  };
  for (auto & config_pair : config_map) {
    auto key = config_pair.first;
    std::replace(key.begin(), key.end(), '-', '_');
    if (dialects.find(key) != dialects.end()) {
      key = dialects[key];
    }
    if (new_config_map.find(key) != new_config_map.end()) {
      LOG(ERROR) << "duplicate config key " << key;
      CHECK(false);
    }
    new_config_map[key] = config_pair.second;
  }
  config_map = new_config_map;
}


void ModuleMeta::log_configs() {
  #define LOG_CONFIG(x)            { std::cerr << #x << ":" << x << std::endl; }
  #define LOG_CONFIG_BOOL(x)       { std::cerr << #x << ":" << (x ? "true" : "false") << std::endl; }
  #define LOG_CONFIG_NAME(name, x) { std::cerr << name << ":" << x << std::endl; }
  #define LOG_CONFIG_NAME_BOOL(name, x) { std::cerr << name << ":" << (x ? "true" : "false") << std::endl; }

  LOG_CONFIG(model_arch_string);
  LOG_CONFIG(num_layer);
  LOG_CONFIG(num_expert);
  LOG_CONFIG(num_expert_per_token);

  LOG_CONFIG(cache_rate);
  LOG_CONFIG(num_predict_expert_per_layer);
  LOG_CONFIG_BOOL(reorder_experts);
  LOG_CONFIG_BOOL(early_preempt);
  LOG_CONFIG_BOOL(chunk_prefetch);

  LOG_CONFIG_NAME("predict_input_mode", predict_input_mode_str);
  LOG_CONFIG_NAME("predictor_type", predictor_type_str);

  LOG_CONFIG(predictor_model_path);
  LOG_CONFIG(predictor_num_layer);
  LOG_CONFIG(predictor_layer_offset);
  LOG_CONFIG(layer_predict_interval);
  LOG_CONFIG(layer_predict_max_window);
  LOG_CONFIG_NAME_BOOL("cross_token_pred", layer_predict_replace_first_input_with_last_output);

  LOG_CONFIG(limit_layer_0_window);
  LOG_CONFIG(limit_layer_0_num_predict);

  LOG_CONFIG(max_prefetch_layer_distance);
  LOG_CONFIG_BOOL(cache_only);
  LOG_CONFIG_BOOL(per_layer_cache);
  LOG_CONFIG_BOOL(promote_hit_in_prefetch);
  LOG_CONFIG(cache_policy);
  LOG_CONFIG(predict_input_reuse_distance_max);
  LOG_CONFIG(predict_input_decay);

  LOG_CONFIG(sleep_on_report_logits_us);
  LOG_CONFIG(expert_mem_scale);
  LOG_CONFIG(num_dummy_params);
  LOG_CONFIG(physical_mem_impl);
  LOG_CONFIG(logical_mem_impl);
  #undef LOG_CONFIG
  #undef LOG_CONFIG_BOOL
  #undef LOG_CONFIG_NAME
  #undef LOG_CONFIG_NAME_BOOL
}

void ModuleMeta::handle_uninited_configs() {
  if (predictor_num_layer == -1) {
    predictor_num_layer = num_layer;
  }
  // if (layer_predict_interval == -1) {
  //   layer_predict_interval   = num_layer;
  //   layer_predict_max_window = num_layer;
  // }
  if (max_prefetch_layer_distance == -1) {
    max_prefetch_layer_distance = num_layer - 1;
  }

  if (getenv("SPARSE_CACHE_PHYSICAL_MEM_IMPL") != nullptr) {
    physical_mem_impl = getenv("SPARSE_CACHE_PHYSICAL_MEM_IMPL");
  }
  if (getenv("SPARSE_CACHE_LOGICAL_MEM_IMPL") != nullptr) {
    logical_mem_impl = getenv("SPARSE_CACHE_LOGICAL_MEM_IMPL");
  }
  if (getenv("SPARSE_CACHE_SLEEP_ON_REPORT_LOGITS_US") != nullptr) {
    sleep_on_report_logits_us = std::stoul(getenv("SPARSE_CACHE_SLEEP_ON_REPORT_LOGITS_US"));
  }
  if (getenv("SPARSE_CACHE_EXPERT_MEM_SCALE") != nullptr) {
    expert_mem_scale = std::stof(getenv("SPARSE_CACHE_EXPERT_MEM_SCALE"));
  }
  std::transform(model_arch_string.begin(), model_arch_string.end(), model_arch_string.begin(), ::tolower);
  // if (limit_layer_0_num_predict == -1) {
  //   if (model_arch_string.find("deepseek") != std::string::npos) {
  //     limit_layer_0_num_predict = num_predict_expert_per_layer;
  //   } else {
  //     limit_layer_0_num_predict = num_predict_expert_per_layer / 2;
  //   }
  // }
  if (limit_layer_0_window == -1) {
    if (model_arch_string.find("deepseek") != std::string::npos) {
      limit_layer_0_window = layer_predict_max_window;
    } else {
      limit_layer_0_window = 1;
    }
  }

  if (expert_mem_scale != 1.0) {
    LOG(ERROR) << "expert memory scale is set to " << expert_mem_scale;
    CHECK(expert_mem_scale > 1.0);
    num_dummy_params = std::ceil(num_per_expert_param * (expert_mem_scale - 1));
    for (int i = 0; i < num_dummy_params; i++) {
      auto dummy_name = "dummy-" + std::to_string(i);
      auto dummy_id = num_per_expert_param + i;
      LOG(ERROR) << "add dummy param: " << dummy_name << " with id " << dummy_id;
      param_name_list.push_back(dummy_name);
      param_name_to_id[dummy_name] = dummy_id;
    }
    num_per_expert_param += num_dummy_params;
  }
  log_configs();
}
