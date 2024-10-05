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
void ModuleMeta::handle_uninited_configs() {
  if (layer_predict_interval == -1) {
    layer_predict_interval   = num_layer;
    layer_predict_max_window = num_layer;
  }
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
    LOG(ERROR) << "expert memory scale is set to " << expert_mem_scale << ", this is experimental feature.";
    LOG_BLOCK(ERROR, logger, {
      logger << "before scale, num_per_expert_param: " << num_per_expert_param << "\n";
      for (int i = 0; i < num_per_expert_param; i++) {
        logger << "param_name_list[" << i << "]: " << param_name_list[i] << ", mapped " << param_name_to_id[param_name_list[i]] << "\n";
      }
    });
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
    LOG_BLOCK(ERROR, logger, {
      logger << "after scale, num_per_expert_param: " << num_per_expert_param << "\n";
      for (int i = 0; i < num_per_expert_param; i++) {
        logger << "param_name_list[" << i << "]: " << param_name_list[i] << ", mapped " << param_name_to_id[param_name_list[i]] << "\n";
      }
    });
  }
}
