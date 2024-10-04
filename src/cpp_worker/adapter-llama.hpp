#include "model_loader.hpp"
#include "prefetcher.hpp"
#include "profiler.hpp"

extern "C" {
  void get_gpu_mem_info(uint64_t* free, uint64_t* total);
  void log_gpu_mem_info();
}