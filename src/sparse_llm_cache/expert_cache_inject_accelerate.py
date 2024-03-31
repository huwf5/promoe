import sys
from functools import wraps

from accelerate.hooks import AlignDevicesHook, SequentialHook, ModelHook
from accelerate.utils import set_module_tensor_to_device
from torch.nn import Module
from expert_selection_tracer.call_tracer import report_called

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

def recursively_replace_pre_forward(module):
  if hasattr(module, "_hf_hook"):
    if isinstance(module._hf_hook, AlignDevicesHook):
      _replace_one_hook_offload(module._hf_hook, module)
    elif isinstance(module._hf_hook, SequentialHook):
      for hook in module._hf_hook.hooks:
        if isinstance(hook, AlignDevicesHook):
          _replace_one_hook_offload(hook, module)
  for n,m in module.named_modules():
    if n == "": continue
    recursively_replace_pre_forward(m)



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

  def __init__(self, cache_len = 24, exec_device = 0):
    self.module_map = {}
    self.hook_map = {}
    self.cached_map = {}
    self.cache_len = cache_len
    self.exec_device = exec_device
  def get(self, key : str) -> None:
    if key in self.cached_map:
      self._on_hit(key)
      return
    self._on_miss(key)


  def _locate_key_to_evict(self) -> str:
    # todo: remove correct entry
    return list(self.cached_map.keys())[0]

  def _evict(self, key, _) -> None:
    module = self.module_map[key]
    self.cached_map.pop(key)
    for name, _ in module.named_parameters():
      set_module_tensor_to_device(module, name, "meta")

  def clear_cache(self) -> None:
    for k in self.cached_map.keys():
      self._evict(k, self.cached_map[k])

  def _on_hit(self, key: str)-> None:
    pass

  def _on_miss(self, key : str) -> None:
    module = self.module_map[key]
    weights_map = self.hook_map[key].weights_map
    if len(self.cached_map) >= self.cache_len:
      key_to_evict = self._locate_key_to_evict()
      self._evict(key_to_evict, module)
    self.cached_map[key] = weights_map
    for name, _ in module.named_parameters():
      set_module_tensor_to_device(module, name, self.exec_device, value = weights_map[name])