#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

#if defined(SPARSE_LLM_CACHE_ENABLE_NVTX)
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define SPARSE_LLM_CACHE_HAS_NVTX 1
#endif
#endif
#endif

#ifndef SPARSE_LLM_CACHE_HAS_NVTX
#define SPARSE_LLM_CACHE_HAS_NVTX 0
#endif

class NvtxRangeGuard {
 public:
  explicit NvtxRangeGuard(const char* name) {
#if SPARSE_LLM_CACHE_HAS_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  explicit NvtxRangeGuard(const std::string& name) : NvtxRangeGuard(name.c_str()) {}
  ~NvtxRangeGuard() {
#if SPARSE_LLM_CACHE_HAS_NVTX
    nvtxRangePop();
#endif
  }

  NvtxRangeGuard(const NvtxRangeGuard&) = delete;
  NvtxRangeGuard& operator=(const NvtxRangeGuard&) = delete;
};

inline bool NvtxDetailEnabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("SPARSE_LLM_CACHE_NVTX_DETAIL");
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
  }();
  return enabled;
}

#if SPARSE_LLM_CACHE_HAS_NVTX
inline void NvtxMark(const char* name) {
  nvtxMarkA(name);
}
inline void NvtxMark(const std::string& name) {
  NvtxMark(name.c_str());
}
#endif

#define NVTX_CONCAT_INNER(a, b) a##b
#define NVTX_CONCAT(a, b) NVTX_CONCAT_INNER(a, b)
#if SPARSE_LLM_CACHE_HAS_NVTX
#define NVTX_RANGE(name) NvtxRangeGuard NVTX_CONCAT(_nvtx_range_, __LINE__)(name)
#define NVTX_MARK(name) NvtxMark(name)
#define NVTX_DETAIL_MARK(name) do { if (NvtxDetailEnabled()) { NvtxMark(name); } } while (0)
#else
#define NVTX_RANGE(name) do {} while (0)
#define NVTX_MARK(name) do {} while (0)
#define NVTX_DETAIL_MARK(name) do {} while (0)
#endif
