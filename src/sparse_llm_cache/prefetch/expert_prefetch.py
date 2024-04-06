from dataclasses import dataclass
import torch
from accelerate.hooks import AlignDevicesHook, SequentialHook, ModelHook
from .oracle_prefetch_policy import OraclePolicy

class CachePrefetchHook(ModelHook):
  def __init__(self, weights_map, prefetch_mngr):
    self.weights_map = weights_map
    self.prefetch_mngr = prefetch_mngr
  def pre_forward(self, module : torch.nn.Module, *args, **kwargs):
    self.prefetch_mngr.on_pre_forward(self, module)
    return args, kwargs
  def post_forward(self, module : torch.nn.Module, output):
    self.prefetch_mngr.on_post_forward(self, module)
    return output

class UntypedBuffer:
  max_len = 0
  cls_to_name_to_offset = {}
  def __init__(self, device):
    self.buffer = torch.empty(UntypedBuffer.max_len, dtype=torch.uint8, device=device)

  @staticmethod
  def required_len(module: torch.nn.Module):
    name_to_offset = None
    if type(module) not in UntypedBuffer.cls_to_name_to_offset:
      name_to_offset = {}
    nbytes = 0
    for n, p in module.named_parameters():
      if not name_to_offset is None:
        name_to_offset[n] = nbytes
      nbytes += p.element_size() * p.nelement()
      nbytes = ((nbytes + 128 - 1) // 128) * 128
    if not name_to_offset is None:
      UntypedBuffer.cls_to_name_to_offset[type(module)] = name_to_offset
    return nbytes
  
  @staticmethod
  def record_possible_len(module: torch.nn.Module):
    UntypedBuffer.max_len = max(UntypedBuffer.max_len, UntypedBuffer.required_len(module))

  def separate_by(self, module: torch.nn.Module) -> dict[str, torch.Tensor]:
    storage = self.buffer.storage().untyped()
    tensors = {}
    for n, p in module.named_parameters():
      t = torch.empty((0), dtype=p.dtype, device=storage.device)
      offset_in_bytes = UntypedBuffer.cls_to_name_to_offset[type(module)][n]
      offset_in_nelem = offset_in_bytes // p.element_size()
      t.set_(source=storage,
             storage_offset = offset_in_nelem,
             size=p.size(),
             stride=p.stride())
      tensors[n] = t
    return tensors

def _handle_acc_hook_offload(hook):
  if hasattr(hook, "_old_offload"):
    return
  hook._old_offload = hook.offload
  hook.offload = False

def _replace_one_hook_offload(hook: AlignDevicesHook, module, prefetch_mngr):
  _handle_acc_hook_offload(hook)
  if hook._old_offload:
    prefetch_mngr.module_map[hook.weights_map.prefix] = module
    prefetch_mngr.hook_map[hook.weights_map.prefix] = hook
    module._hf_hook = SequentialHook(module._hf_hook, CachePrefetchHook(hook.weights_map, prefetch_mngr))
    UntypedBuffer.record_possible_len(module)

def replace_pre_forward(model, prefetch_mngr):
  # fixed: named_module is already recursive
  for n,module in model.named_modules():
    if n == "": continue
    if hasattr(module, "_hf_hook"):
      if isinstance(module._hf_hook, AlignDevicesHook):
        _replace_one_hook_offload(module._hf_hook, module, prefetch_mngr)
      elif isinstance(module._hf_hook, SequentialHook):
        for hook in module._hf_hook.hooks:
          if isinstance(hook, AlignDevicesHook):
            _replace_one_hook_offload(hook, module, prefetch_mngr)

@dataclass
class PrefetchJob:
  prefix : str = None
  module : torch.nn.Module = None
  event : torch.cuda.Event = None
  buffer : UntypedBuffer = None
  tensor_map : dict[str, torch.Tensor] = None

  def is_done(self):
    if self.event:
      return self.event.query()
    else:
      return True
  
  def sync(self):
    if self.is_done():
      return
    self.event.synchronize()

# '''
# assumes all experts have same parameter and shape
# '''
# class ExpertPrefetchMngr:
#   def __init__(self, cache_len = 24):
#     self.module_map : dict[str, torch.nn.Module] = {}
#     self.hook_map : dict[str, ModelHook] = {}
#     self.cache_len = cache_len
#     self.prefetch_buffer = []
#     # self.modules_on_prefetch = {}
#     self.module_on_prefetch : PrefetchJob = None
#     self.done_or_doing_jobs : dict[str, PrefetchJob] = {}
#     self.stream : torch.cuda.Stream = None
#     self.cur_time : int = None
#   def _locate_next_module_to_prefetch(self):
#     raise RuntimeError("Unimplemented")

#   def launch_copy(self, buffer : UntypedBuffer, key):
#     hook = self.hook_map[key]
#     module = self.module_map[key]
#     target_tensors = buffer.separate_by(module)
#     with torch.cuda.stream(self.stream):
#       for n, p in module.named_parameters():
#         source_tensor = hook.weights_map[n]
#         source_tensor : torch.Tensor
#         target_tensors[n].copy_(source_tensor, non_blocking=True)
#       event = self.stream.record_event()
#     return event, target_tensors

#   def on_post_forward(self, hook: CachePrefetchHook, module: torch.nn.Module):
#     m_to_prefetch = self._locate_next_module_to_prefetch()
#     if m_to_prefetch == None:
#       return
#     prefix = hook.weights_map.prefix
#     buffer = hook.buffer
#     # build dependency: finish current forward, then copy
#     self.stream.wait_stream(torch.cuda.default_stream())

#     event, target_tensors = self.launch_copy(buffer, prefix)
#     self.modules_on_prefetch[prefix] = (buffer, target_tensors, event)
  
#   def has_ongoing_job(self):
#     return self.modules_on_prefetch[next(reversed(self.modules_on_prefetch.keys()))][2]

#   def on_pre_forward(self, hook: CachePrefetchHook, module : torch.nn.Module):
#     if hook.weights_map.prefix in self.done_or_doing_jobs:
#     # cur_prefetch_job = self._current_module_on_prefetch()

#     if hook.weights_map.prefix == cur_prefetch_job.prefix:
#       event = cur_prefetch_job.event
#       if event.query():
#         pass
#       else:
#         event.synchronize()

'''
Oracle prefetch
One prefetch job
No cache
'''
class NaiveExpertPrefetchMngr:
  def __init__(self):
    self.module_map : dict[str, torch.nn.Module] = {}
    self.hook_map : dict[str, ModelHook] = {}
    self.cur_prefetch_job : PrefetchJob = None
    self.cur_forward_job : PrefetchJob = None
    self.stream : torch.cuda.Stream = None
    self.cur_time : int = None
    self.policy : OraclePolicy = None
  def init(self, exec_device = 0):
    self.stream = torch.cuda.Stream()
    self.policy = OraclePolicy()
    self.cur_prefetch_job = PrefetchJob(buffer=UntypedBuffer(exec_device))
    self.cur_forward_job = PrefetchJob(buffer=UntypedBuffer(exec_device))
  def _locate_next_module_to_prefetch(self):
    raise RuntimeError("Unimplemented")

  def launch_copy(self, buffer : UntypedBuffer, key):
    hook = self.hook_map[key]
    module = self.module_map[key]
    target_tensors = buffer.separate_by(module)
    with torch.cuda.stream(self.stream):
      for n, p in module.named_parameters():
        source_tensor = hook.weights_map[n]
        source_tensor : torch.Tensor
        target_tensors[n].copy_(source_tensor, non_blocking=True)
      event = self.stream.record_event()
    return event, target_tensors

  def on_post_forward(self, hook: CachePrefetchHook, module: torch.nn.Module):
    e = torch.cuda.default_stream().record_event()
    self.stream.wait_event(e)

  def create_launch_job(self, job: PrefetchJob, prefix : str):
    job.prefix = prefix
    job.module = self.module_map[prefix]
    job.event, job.tensor_map = self.launch_copy(job.buffer, prefix)

  def set_module_tensor(self, tensor_map, module):
      for n,_ in module.named_parameters():
        param_cls = type(module._parameters[n])
        p = param_cls(tensor_map[n])
        module._parameters[n] = p

  def on_pre_forward(self, hook: CachePrefetchHook, module : torch.nn.Module):
    # print(hook.weights_map.prefix, self.cur_forward_job.prefix, self.cur_prefetch_job.prefix)
    if hook.weights_map.prefix == self.cur_prefetch_job.prefix:
      self.cur_forward_job, self.cur_prefetch_job = self.cur_prefetch_job, self.cur_forward_job
      self.cur_forward_job.event.synchronize()
      self.set_module_tensor(self.cur_forward_job.tensor_map, module)
      self.policy._access(hook.weights_map.prefix)

      next_prefix = self.policy._predict_next()
      if next_prefix:
        self.create_launch_job(self.cur_prefetch_job, next_prefix)
    elif self.policy.cur_time != -1:
      # old predict fail
      assert(self.cur_forward_job.is_done())
      print(f"Warning: oracle predict failed: {hook.weights_map.prefix} != {self.cur_prefetch_job.prefix}")
      self.cur_forward_job, self.cur_prefetch_job = self.cur_prefetch_job, self.cur_forward_job
      self.cur_forward_job.event.synchronize()
      cur_prefix : str = hook.weights_map.prefix
      self.create_launch_job(self.cur_forward_job, cur_prefix)
      self.cur_forward_job.event.synchronize()
      self.set_module_tensor(self.cur_forward_job.tensor_map, module)

    else:
      # is a new sequence. it's normal to fail
      # assert(self.policy.cur_time == -1)
      assert(self.cur_forward_job.is_done())
      assert(self.cur_prefetch_job.is_done())
      assert(self.policy._predict_next() == hook.weights_map.prefix)
      cur_prefix : str = hook.weights_map.prefix
      self.create_launch_job(self.cur_forward_job, cur_prefix)
      self.cur_forward_job.event.synchronize()
      self.set_module_tensor(self.cur_forward_job.tensor_map, module)
      self.policy._access(hook.weights_map.prefix)

      next_prefix = self.policy._predict_next()
      if next_prefix:
        self.create_launch_job(self.cur_prefetch_job, next_prefix)
