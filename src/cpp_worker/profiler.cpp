#include "profiler.hpp"

int TraceEventCollector::kNumMaxThread = 10;
bool TraceEventGuard::globally_enabled = (getenv("SPARSE_CACHE_ENABLE_TRACE") != nullptr);