#include "adapter-llama.hpp"
#include "logging.hpp"
extern "C" {
void get_gpu_mem_info(uint64_t *free_byte, uint64_t *total_byte) {
  CUDA_CALL(cudaMemGetInfo(free_byte, total_byte));
}

void log_gpu_mem_info() {
  uint64_t free_byte, total_byte;
  get_gpu_mem_info(&free_byte, &total_byte);
  LOG(ERROR) << "GPU memory: free " << free_byte/1024.0/1024.0/1024.0 << "GB, total " << total_byte/1024.0/1024.0/1024.0 << "GB";
}
}
