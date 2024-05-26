#include <torch/script.h>
#include "utils.hpp"

class Predictor {
  std::shared_ptr<ModuleMeta> metas;
  torch::jit::script::Module predict_model;
 public:
  void load_model(std::string model_path) {
    predict_model = torch::jit::load(model_path);
    predict_model.eval();
  }

  torch::Tensor predict(torch::Tensor freq) {
    std::vector<torch::jit::IValue> inputs{freq};
    return predict_model.forward(inputs).toTensor();
  }
};