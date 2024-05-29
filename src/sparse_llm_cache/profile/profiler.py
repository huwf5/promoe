from torch.profiler import profile, record_function, ProfilerActivity
# profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True)
# profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True)
# profiler, profile_fname = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_modules=True), "nllb-transformer-trace-module.json"
# profiler, profile_fname = profile(activities=[ProfilerActivity.CUDA], with_stack=True), "nllb-transformer-trace-cuda-only-stack.json"

_global_profiler : profile = None
# profile_fname = "nllb-transformer-trace-stack-short.json"

def create_profile(record_shapes=True,profile_memory=True,with_stack=True, with_cpu=True, with_cuda=True):
  global _global_profiler
  activities = []
  if with_cpu:
    activities.append(ProfilerActivity.CPU)
  if with_cuda:
    activities.append(ProfilerActivity.CUDA)
  _global_profiler = profile(
    activities=activities,
    with_stack=with_stack,
    record_shapes=record_shapes,
    profile_memory=profile_memory,
  )

def start_profile() :
  global _global_profiler
  _global_profiler.__enter__()

def stop_profile():
  global _global_profiler
  _global_profiler.__exit__(None, None, None)

def export_trace(fname):
  _global_profiler.export_chrome_trace(fname)