#include "model_loader.hpp"
#include "prefetcher.hpp"
#include "profiler.hpp"

extern "C" {
  void get_gpu_mem_info(size_t* free, size_t* total);
  void log_gpu_mem_info();
}