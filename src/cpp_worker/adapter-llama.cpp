#include "adapter-llama.hpp"
#include "logging.hpp"
extern "C" {
void get_gpu_mem_info(uint64_t *free_byte, uint64_t *total_byte) {
  CUDA_CALL(cudaMemGetInfo(free_byte, total_byte));
}

void log_gpu_mem_info() {
  uint64_t free_byte, total_byte;
  get_gpu_mem_info(&free_byte, &total_byte);
  std::cerr << "gpu_memory_usage_MiB:" << (total_byte - free_byte) / (1024 * 1024) << std::endl;
}
}
