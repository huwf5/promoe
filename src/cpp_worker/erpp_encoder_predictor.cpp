#include "erpp_encoder_predictor.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <utility>

#include "logging.hpp"
#include "nvtx_utils.hpp"
#include "profiler.hpp"

namespace {
bool log_erpp_encoder_prefetch_enabled() {
  const char* erpp_flag = std::getenv("SPARSE_CACHE_LOG_ERPP_ENCODER_PREFETCH");
  if (erpp_flag != nullptr && erpp_flag[0] != '\0' && erpp_flag[0] != '0') {
    return true;
  }
  const char* scheduler_flag = std::getenv("SPARSE_CACHE_LOG_PREFETCH_DECISION");
  return scheduler_flag != nullptr && scheduler_flag[0] != '\0' && scheduler_flag[0] != '0';
}

bool log_erpp_encoder_diagnostics_enabled() {
  const char* flag = std::getenv("SPARSE_CACHE_LOG_ERPP_ENCODER_DIAGNOSTICS");
  return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}

std::string trim_copy(const std::string& value) {
  const auto start = value.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  const auto stop = value.find_last_not_of(" \t\n\r");
  return value.substr(start, stop - start + 1);
}
}  // namespace

ErppEncoderPredictor::ErppEncoderPredictor(ModuleMeta* metas) : metas(metas) {
  CHECK(metas != nullptr) << "ERPP encoder predictor requires ModuleMeta";
  dynamic_noisy_or_budget = trim_copy(metas->erpp_encoder_budgets) == "dynamic_noisy_or_sum";
  budgets = parse_budgets(metas->erpp_encoder_budgets);
  enabled_layers = parse_enabled_layers(metas->erpp_encoder_layers);
}

ErppEncoderPredictor::~ErppEncoderPredictor() {
  reset_sequence_state();
  if (record_event != nullptr) {
    CUDA_CALL(cudaEventDestroy(record_event));
    record_event = nullptr;
  }
}

std::vector<int> ErppEncoderPredictor::parse_budgets(const std::string& spec) const {
  std::vector<int> parsed;
  const auto normalized = trim_copy(spec);
  if (normalized.empty() || normalized == "fixed_p90") {
    parsed = {89, 59, 72, 72, 70, 62};
  } else if (normalized == "fixed_mean") {
    parsed = {70, 49, 56, 55, 55, 51};
  } else if (normalized == "dynamic_noisy_or_sum") {
    parsed.assign(metas->num_encoder_moe_layer, 1);
  } else {
    std::stringstream ss(normalized);
    std::string item;
    while (std::getline(ss, item, ',')) {
      item = trim_copy(item);
      CHECK(!item.empty()) << "empty ERPP encoder budget entry: " << spec;
      parsed.push_back(std::stoi(item));
    }
  }

  CHECK(parsed.size() == static_cast<size_t>(metas->num_encoder_moe_layer))
      << "ERPP budget count must match num_encoder_moe_layer, got " << parsed.size()
      << " budgets for " << metas->num_encoder_moe_layer << " encoder layers";
  for (int budget : parsed) {
    CHECK(budget > 0 && budget <= metas->num_expert)
        << "ERPP budget out of range: " << budget
        << ", expected [1," << metas->num_expert << "]";
  }
  return parsed;
}

int ErppEncoderPredictor::encoder_budget_for_layer(int layer_idx) const {
  CHECK(layer_idx >= 0 && layer_idx < static_cast<int>(budgets.size()))
      << "ERPP encoder budget layer out of range: " << layer_idx;
  return budgets[layer_idx];
}

int erpp_noisy_or_sum_budget(torch::Tensor layer_scores, int num_expert) {
  CHECK(num_expert > 0) << "ERPP num_expert must be positive";
  CHECK(layer_scores.defined()) << "ERPP score tensor is undefined";
  CHECK(layer_scores.dim() == 1) << "ERPP layer scores must be [E]";
  CHECK(layer_scores.size(0) == num_expert)
      << "ERPP layer score expert mismatch, got " << layer_scores.size(0)
      << ", expected " << num_expert;
  auto sum_tensor = layer_scores.to(torch::kFloat32).sum();
  const float raw = sum_tensor.item<float>();
  CHECK(std::isfinite(raw)) << "ERPP dynamic noisy-or budget score sum is not finite";
  const int budget = static_cast<int>(std::ceil(raw));
  return std::clamp(budget, 1, num_expert);
}

int ErppEncoderPredictor::budget_from_scores(torch::Tensor layer_scores) const {
  return erpp_noisy_or_sum_budget(layer_scores, metas->num_expert);
}

int ErppEncoderPredictor::encoder_jit_floor() const {
  CHECK(metas->num_encoder_moe_layer > 0)
      << "ERPP encoder JIT floor requires encoder MoE layers";
  int floor_value = 0;
  if (metas->erpp_encoder_jit_refill_floor_mode == "fixed" &&
      metas->erpp_encoder_jit_refill_floor_value > 0) {
    floor_value = metas->erpp_encoder_jit_refill_floor_value;
  } else if (metas->initial_cache_policy == "fixed" && !metas->initial_layer_budgets.empty()) {
    std::stringstream ss(metas->initial_layer_budgets);
    std::string item;
    while (std::getline(ss, item, ',')) {
      item = trim_copy(item);
      if (item.empty()) {
        continue;
      }
      const auto sep = item.find(':');
      const std::string value = sep == std::string::npos ? item : item.substr(sep + 1);
      floor_value = std::max(floor_value, std::stoi(trim_copy(value)));
    }
  } else {
    floor_value = static_cast<int>(
        std::floor(metas->cache_rate * metas->num_layer * metas->num_expert)) /
        metas->num_encoder_moe_layer;
  }
  return std::clamp(floor_value, 1, metas->num_expert);
}

int ErppEncoderPredictor::encoder_jit_ranking_limit(int layer_idx) const {
  return encoder_jit_ranking_limit(layer_idx, encoder_budget_for_layer(layer_idx));
}

int ErppEncoderPredictor::encoder_jit_ranking_limit(int layer_idx, int budget) const {
  CHECK(layer_idx >= 0 && layer_idx < metas->num_encoder_moe_layer)
      << "ERPP encoder JIT ranking layer out of range: " << layer_idx;
  CHECK(budget > 0 && budget <= metas->num_expert)
      << "ERPP encoder JIT ranking budget out of range: " << budget
      << ", expected [1," << metas->num_expert << "]";
  if (!metas->enable_erpp_encoder_jit_refill) {
    return budget;
  }
  if (metas->erpp_encoder_jit_refill_floor_mode == "budget") {
    return budget;
  }
  const int floor_value = encoder_jit_floor();
  return std::min(metas->num_expert, std::max(floor_value, budget));
}

std::vector<uint8_t> ErppEncoderPredictor::parse_enabled_layers(const std::string& spec) const {
  const auto normalized = trim_copy(spec);
  CHECK(!normalized.empty()) << "ERPP encoder layers config is empty";

  std::vector<uint8_t> enabled(metas->num_encoder_moe_layer, 0);
  if (normalized == "all") {
    std::fill(enabled.begin(), enabled.end(), 1);
    return enabled;
  }

  CHECK(normalized.find("all") == std::string::npos)
      << "ERPP encoder layers 'all' cannot be mixed with explicit ids: " << spec;

  CHECK(normalized.back() != ',') << "empty ERPP encoder layer entry: " << spec;

  std::stringstream ss(normalized);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim_copy(item);
    CHECK(!item.empty()) << "empty ERPP encoder layer entry: " << spec;
    size_t pos = 0;
    int layer_idx = 0;
    try {
      layer_idx = std::stoi(item, &pos);
    } catch (const std::exception&) {
      CHECK(false) << "invalid ERPP encoder layer entry: " << item
                   << ", config=" << spec;
    }
    CHECK(pos == item.size())
        << "invalid ERPP encoder layer entry: " << item
        << ", config=" << spec;
    if (layer_idx < 0) {
      layer_idx += metas->num_encoder_moe_layer;
    }
    CHECK(layer_idx >= 0 && layer_idx < metas->num_encoder_moe_layer)
        << "ERPP encoder layer out of range: " << item
        << ", normalized=" << layer_idx
        << ", num_encoder_moe_layer=" << metas->num_encoder_moe_layer;
    CHECK(!enabled[layer_idx])
        << "duplicate ERPP encoder layer: " << item
        << ", normalized=" << layer_idx
        << ", config=" << spec;
    enabled[layer_idx] = 1;
  }

  return enabled;
}

bool ErppEncoderPredictor::should_prefetch_layer(int layer_idx) const {
  CHECK(layer_idx >= 0 && layer_idx < static_cast<int>(enabled_layers.size()))
      << "ERPP encoder layer query out of range: " << layer_idx;
  return enabled_layers[layer_idx] != 0;
}

void ErppEncoderPredictor::load_model_from(const std::string& path) {
  CHECK(!path.empty()) << "ERPP encoder predictor path is empty";
  c10::Device cpu_device(c10::DeviceType::CPU);
  model = torch::jit::load(path, cpu_device);
  model.eval();
  loaded = true;
  LOG(INFO) << "loaded ERPP encoder predictor from " << path;
}

void ErppEncoderPredictor::reset_sequence_state() {
  if (record_event != nullptr) {
    CUDA_CALL(cudaEventSynchronize(record_event));
  }
  hidden_buffer = torch::Tensor();
  attention_mask_buffer = torch::Tensor();
  input_recorded = false;
  recorded_forward_epoch = -1;
  recorded_generate_epoch = -1;
}

void ErppEncoderPredictor::record_encoder_layer0(
    torch::Tensor hidden,
    torch::Tensor attention_mask,
    cudaStream_t compute_stream,
    int64_t forward_epoch,
    int64_t generate_epoch) {
  NVTX_RANGE("erpp/record_encoder_layer0 forward_epoch=" + std::to_string(forward_epoch) +
             " generate_epoch=" + std::to_string(generate_epoch));
  CHECK(hidden.defined()) << "ERPP hidden is undefined";
  CHECK(attention_mask.defined()) << "ERPP attention_mask is undefined";

  CHECK(hidden.is_cuda()) << "ERPP hidden must be a CUDA tensor for async recording";
  CHECK(attention_mask.is_cuda()) << "ERPP attention_mask must be a CUDA tensor for async recording";
  CHECK(hidden.get_device() == attention_mask.get_device())
      << "ERPP hidden and attention_mask must be on the same CUDA device";
  CHECK(hidden.is_contiguous()) << "ERPP hidden must be contiguous for async recording";
  CHECK(attention_mask.is_contiguous())
      << "ERPP attention_mask must be contiguous for async recording";

  auto torch_stream = at::cuda::getStreamFromExternal(compute_stream, hidden.get_device());
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  hidden_buffer = torch::empty_like(
      hidden,
      hidden.options().device(torch::kCPU).pinned_memory(true));
  attention_mask_buffer = torch::empty_like(
      attention_mask,
      attention_mask.options().device(torch::kCPU).pinned_memory(true));

  {
    NVTX_RANGE("erpp/hidden_d2h bytes=" + std::to_string(hidden.nbytes()));
    CUDA_CALL(cudaMemcpyAsync(
        hidden_buffer.data_ptr(),
        hidden.data_ptr(),
        hidden.nbytes(),
        cudaMemcpyDeviceToHost,
        compute_stream));
  }
  {
    NVTX_RANGE("erpp/attention_mask_d2h bytes=" + std::to_string(attention_mask.nbytes()));
    CUDA_CALL(cudaMemcpyAsync(
        attention_mask_buffer.data_ptr(),
        attention_mask.data_ptr(),
        attention_mask.nbytes(),
        cudaMemcpyDeviceToHost,
        compute_stream));
  }
  if (record_event == nullptr) {
    CUDA_CALL(cudaEventCreateWithFlags(&record_event, cudaEventDisableTiming));
  }
  CUDA_CALL(cudaEventRecord(record_event, compute_stream));
  input_recorded = true;
  recorded_forward_epoch = forward_epoch;
  recorded_generate_epoch = generate_epoch;
}

ErppEncoderPrediction ErppEncoderPredictor::predict_recorded() {
  if (!input_recorded) {
    if (log_erpp_encoder_prefetch_enabled()) {
      LOG(INFO) << "erpp_encoder_prefetch: no recorded input";
    }
    return {};
  }
  uint64_t record_wait_us = 0;
  {
    NVTX_RANGE("erpp/wait_record_event");
    Timer timer;
    CUDA_CALL(cudaEventSynchronize(record_event));
    record_wait_us = timer.dur_us();
  }
  input_recorded = false;
  Timer predict_timer;
  auto predictions = predict_from_cpu_tensors(hidden_buffer, attention_mask_buffer);
  const uint64_t predict_cpu_us = predict_timer.dur_us();
  if (log_erpp_encoder_diagnostics_enabled()) {
    int64_t ranking_experts_total = 0;
    int64_t budget_experts_total = 0;
    int enabled_layers = 0;
    for (size_t layer = 0; layer < predictions.rankings.size(); layer++) {
      ranking_experts_total += static_cast<int64_t>(predictions.rankings[layer].size());
      const int budget = layer < predictions.budgets.size() ? predictions.budgets[layer] : 0;
      budget_experts_total += budget;
      if (budget > 0 || !predictions.rankings[layer].empty()) {
        enabled_layers += 1;
      }
    }
    LOG(INFO) << "erpp_encoder_diagnostics: predictor_timing"
              << " forward_epoch=" << recorded_forward_epoch
              << " generate_epoch=" << recorded_generate_epoch
              << " record_wait_us=" << record_wait_us
              << " predict_cpu_us=" << predict_cpu_us
              << " layers=" << predictions.rankings.size()
              << " enabled_layers=" << enabled_layers
              << " ranking_experts_total=" << ranking_experts_total
              << " budget_experts_total=" << budget_experts_total;
  }
  return predictions;
}

torch::Tensor ErppEncoderPredictor::normalize_attention_mask(
    torch::Tensor attention_mask,
    int64_t batch_size,
    int64_t seq_len) const {
  CHECK(attention_mask.defined()) << "ERPP attention_mask is undefined";
  bool extended_float_mask = attention_mask.dim() > 2 && attention_mask.is_floating_point();
  auto mask = attention_mask.detach().to(torch::kCPU);
  while (mask.dim() > 2) {
    bool squeezed = false;
    for (int dim = 1; dim < mask.dim() - 1; dim++) {
      if (mask.size(dim) == 1) {
        mask = mask.squeeze(dim);
        squeezed = true;
        break;
      }
    }
    CHECK(squeezed) << "attention_mask cannot normalize to [B,T]";
  }
  CHECK(mask.dim() == 2) << "attention_mask must normalize to [B,T]";
  CHECK(mask.size(0) == batch_size && mask.size(1) == seq_len)
      << "attention_mask shape mismatch, got [" << mask.size(0) << "," << mask.size(1)
      << "] expected [" << batch_size << "," << seq_len << "]";

  if (extended_float_mask) {
    mask = mask >= 0;
  } else if (mask.is_floating_point()) {
    mask = mask != 0;
  } else {
    mask = mask.to(torch::kBool);
  }
  return mask.to(torch::kLong).contiguous();
}

ErppEncoderPrediction ErppEncoderPredictor::predict(
    torch::Tensor hidden,
    torch::Tensor attention_mask) {
  CHECK(hidden.defined()) << "ERPP hidden is undefined";
  CHECK(attention_mask.defined()) << "ERPP attention_mask is undefined";
  auto hidden_cpu = hidden.detach().to(torch::kCPU).contiguous();
  auto attention_mask_cpu = attention_mask.detach().to(torch::kCPU).contiguous();
  return predict_from_cpu_tensors(hidden_cpu, attention_mask_cpu);
}

ErppEncoderPrediction ErppEncoderPredictor::predict_from_cpu_tensors(
    torch::Tensor hidden_cpu,
    torch::Tensor attention_mask_cpu) {
  CHECK(loaded) << "ERPP encoder predictor model is not loaded";
  CHECK(hidden_cpu.defined()) << "ERPP hidden is undefined";
  CHECK(hidden_cpu.dim() == 3) << "ERPP hidden must be [B,T,H]";
  CHECK(hidden_cpu.size(0) == 1) << "ERPP encoder prefetch supports batch size 1";

  auto mask = normalize_attention_mask(attention_mask_cpu, hidden_cpu.size(0), hidden_cpu.size(1));
  auto hidden_float_cpu = hidden_cpu.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();

  torch::NoGradGuard guard;
  std::vector<torch::jit::IValue> inputs{hidden_float_cpu, mask};
  torch::Tensor logits;
  {
    NVTX_RANGE("erpp/predict_forward");
    logits = model.forward(inputs).toTensor().to(torch::kCPU).contiguous();
  }
  CHECK(logits.dim() == 3) << "ERPP logits must be [1,L,E]";
  CHECK(logits.size(0) == 1) << "ERPP logits batch must be 1";
  CHECK(logits.size(1) == metas->num_encoder_moe_layer)
      << "ERPP logits layer mismatch, got " << logits.size(1)
      << ", expected " << metas->num_encoder_moe_layer;
  CHECK(logits.size(2) == metas->num_expert)
      << "ERPP logits expert mismatch, got " << logits.size(2)
      << ", expected " << metas->num_expert;

  ErppEncoderPrediction result;
  result.rankings.reserve(metas->num_encoder_moe_layer);
  result.budgets.reserve(metas->num_encoder_moe_layer);
  for (int layer = 0; layer < metas->num_encoder_moe_layer; layer++) {
    if (!should_prefetch_layer(layer)) {
      result.rankings.emplace_back();
      result.budgets.push_back(0);
      continue;
    }
    auto layer_scores = logits[0][layer].contiguous();
    const int budget = dynamic_noisy_or_budget
        ? budget_from_scores(layer_scores)
        : encoder_budget_for_layer(layer);
    const int limit = encoder_jit_ranking_limit(layer, budget);
    auto top = std::get<1>(layer_scores.topk(limit, -1, true, true));
    auto top_cpu = top.to(torch::kLong).contiguous();
    int64_t* data = top_cpu.data_ptr<int64_t>();
    if (log_erpp_encoder_prefetch_enabled()) {
      LOG(INFO) << "erpp_encoder_prefetch: predicted layer L" << layer
                << " budget=" << budget
                << " ranking_limit=" << limit
                << " experts=[" << array_to_str(data, limit) << "]";
    }
    result.rankings.emplace_back(data, data + limit);
    result.budgets.push_back(budget);
  }
  return result;
}
