import sys
from functools import wraps

from accelerate.hooks import AlignDevicesHook, SequentialHook, ModelHook
from accelerate.utils import set_module_tensor_to_device
from torch.nn import Module
from expert_selection_tracer.call_tracer import report_called
from collections import OrderedDict

def _handle_acc_hook_offload(hook):
  if hasattr(hook, "_old_offload"):
    return
  hook._old_offload = hook.offload
  hook.offload = False

class CacheOffloadHook(ModelHook):
  def __init__(self, weights_map):
    self.weights_map = weights_map
  def pre_forward(self, module, *args, **kwargs):
    ExpertCacheMngr.inst().get(self.weights_map.prefix)
    return args, kwargs

def _replace_one_hook_offload(hook: AlignDevicesHook, module):
  _handle_acc_hook_offload(hook)
  if hook._old_offload:
    ExpertCacheMngr.inst().module_map[hook.weights_map.prefix] = module
    ExpertCacheMngr.inst().hook_map[hook.weights_map.prefix] = hook
    module._hf_hook = SequentialHook(module._hf_hook, CacheOffloadHook(hook.weights_map))

# def recursively_replace_pre_forward(module):
#   did_replace = False
#   if hasattr(module, "_hf_hook"):
#     if isinstance(module._hf_hook, AlignDevicesHook):
#       did_replace = _replace_one_hook_offload(module._hf_hook, module) or did_replace
#     elif isinstance(module._hf_hook, SequentialHook):
#       for hook in module._hf_hook.hooks:
#         if isinstance(hook, AlignDevicesHook):
#           did_replace = _replace_one_hook_offload(hook, module) or did_replace
#   for n,m in module.named_modules():
#     if n == "": continue
#     did_replace = recursively_replace_pre_forward(m) or did_replace
#     if did_replace:
#       print(n)
#   return did_replace

def replace_pre_forward(model):
  # fixed: named_module is already recursive
  for n,module in model.named_modules():
    if n == "": continue
    if hasattr(module, "_hf_hook"):
      if isinstance(module._hf_hook, AlignDevicesHook):
        _replace_one_hook_offload(module._hf_hook, module)
      elif isinstance(module._hf_hook, SequentialHook):
        for hook in module._hf_hook.hooks:
          if isinstance(hook, AlignDevicesHook):
            _replace_one_hook_offload(hook, module)


class LRUPolicy:
  def __init__(self):
    # from old to new
    self.last_use_order = OrderedDict()
    pass
  def _choose_to_evict(self):
    return next(iter(self.last_use_order))
  def _evict(self, key):
    self.last_use_order.pop(key)
  def _access(self, key):
    if key in self.last_use_order:
      self.last_use_order.move_to_end(key)
    else:
      self.last_use_order[key] = None
  def clear(self):
    self.last_use_order.clear()

'''
Currently supports single device
'''

class ExpertCacheMngr:
  _inst = None

  @staticmethod
  def set(o) :
    ExpertCacheMngr._inst = o
  
  @staticmethod
  def inst():
    return ExpertCacheMngr._inst

  def __init__(self, cache_len = 24, exec_device = 0, policy_cls = LRUPolicy):
    self.module_map = {}
    self.hook_map = {}
    self.cached_map : dict[str, any] = {}
    self.cache_len = cache_len
    self.exec_device = exec_device
    self.policy = policy_cls()
  def get(self, key : str) -> None:
    if key in self.cached_map:
      self._handle_hit(key)
    else:
      self._handle_miss(key)
    self._access(key)


  def _locate_key_to_evict(self) -> str:
    return self.policy._choose_to_evict()

  def _evict(self, key, _) -> None:
    module = self.module_map[key]
    self.cached_map.pop(key)
    self.policy._evict(key)
    for name, _ in module.named_parameters():
      set_module_tensor_to_device(module, name, "meta")

  def clear_cache(self, simulate_evict = False) -> None:
    if simulate_evict:
      while len(self.cached_map) > 0:
        self._evict(self._locate_key_to_evict())
    else:
      self.cached_map.clear()
      self.policy.clear()

  def _access(self, key: str) -> None:
    self.policy._access(key)

  def _handle_hit(self, key: str)-> None:
    pass

  def _handle_miss(self, key : str) -> None:
    module = self.module_map[key]
    weights_map = self.hook_map[key].weights_map
    if len(self.cached_map) >= self.cache_len:
      key_to_evict = self._locate_key_to_evict()
      self._evict(key_to_evict, module)
    self.cached_map[key] = weights_map
    for name, _ in module.named_parameters():
      set_module_tensor_to_device(module, name, self.exec_device, value = weights_map[name])