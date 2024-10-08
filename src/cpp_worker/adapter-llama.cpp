#include "adapter-llama.hpp"
#include "logging.hpp"
#include <cstdint>

namespace {

void *sparse_llm_cache_eated_cuda_memory = nullptr;
uint64_t sparse_llm_cache_eated_cuda_memory_size = 0;

}

extern "C" {
void get_gpu_mem_info(uint64_t *free_byte, uint64_t *total_byte) {
  CUDA_CALL(cudaMemGetInfo(free_byte, total_byte));
}

void log_gpu_mem_info() {
  uint64_t free_byte, total_byte;
  get_gpu_mem_info(&free_byte, &total_byte);
  std::cerr << "gpu_memory_usage_MiB:" << (total_byte - free_byte) / (1024 * 1024) << std::endl;
  std::cerr << "eval_eaten_cuda_memory_MiB:" << sparse_llm_cache_eated_cuda_memory_size / (1024 * 1024) << std::endl;
}

void eat_cuda_memory(uint64_t nbytes) {
  if (sparse_llm_cache_eated_cuda_memory != nullptr) {
    CHECK(false) << "eat_cuda_memory is not allowed to be called twice";
  }
  LOG(ERROR) << "eating cuda memory:" << nbytes / (1024 * 1024) << " MiB";
  CUDA_CALL(cudaMalloc(&sparse_llm_cache_eated_cuda_memory, nbytes));
  sparse_llm_cache_eated_cuda_memory_size = nbytes;
}

void auto_eat_cuda_memory() {
  uint64_t nbytes = 0;
  nbytes = std::stoull(GetEnv("SPARSE_EVAL_EAT_CUDA_MEMORY", "0"));
  if (nbytes > 0) {
    eat_cuda_memory(nbytes);
    return;
  }
  nbytes = std::stoull(GetEnv("SPARSE_EVAL_EAT_CUDA_MEMORY_MiB", "0"));
  if (nbytes > 0) {
    eat_cuda_memory(nbytes * 1024 * 1024);
    return;
  }
}

}
