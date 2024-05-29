import re

from .hooks import *

class Filter:
  def __init__(self):
    pass
  def __call__(self, *args, **kwds):
    return True
  def reverse(self):
    return ReverseFilter(self)

class ReverseFilter(Filter):
  def __init__(self, filter : Filter):
    self.filter = filter
  def __call__(self, *args, **kwds):
    return not self.filter(*args, **kwds)

class RegexFilter(Filter):
  def __init__(self, pattern):
    self.pattern = pattern
  def __call__(self, name, *args, **kwds):
    return re.match(self.pattern, name) != None

def recursive_traverse_childrens(module, func, filter=Filter(), prefix='', reverse_filter=False):
  if filter(prefix):
    func(module, prefix)
  for child_name, child in module.named_children():
    recursive_traverse_childrens(child, func, filter, f'{prefix}.{child_name}' if prefix != '' else child_name, reverse_filter)

def recursive_traverse_childrens_leaf_only(module, func, filter=Filter(), prefix='', reverse_filter=False):
  num_children = 0
  for child_name, child in module.named_children():
    recursive_traverse_childrens_leaf_only(child, func, filter, f'{prefix}.{child_name}' if prefix != '' else child_name, reverse_filter)
    num_children+=1
  if num_children == 0:
    if filter(prefix):
      func(module, prefix)

def add_metadata_to_submodules(model, parse_expert_meta_from_name):
  def f(module, name):
    module._prefix = name
    expert_meta = parse_expert_meta_from_name(name)
    try:
      module._layer_id = int(expert_meta[0])
      module._expert_id = int(expert_meta[1])
    except TypeError as e:
      pass
  recursive_traverse_childrens(model, f)

def register_expert_params(model, model_loader, filter=RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)$')):
  def f(module, name):
    # print(f'registering expert {name}')
    for k,v in module.named_parameters():
      model_loader.add_one_expert_param(v, module._layer_id, module._expert_id, k)

    def update_child_param(child_module, child_name):
      for n,_ in child_module.named_parameters():
        child_module._parameters[n] = model_loader.ref_one_expert_param(module._layer_id, module._expert_id, child_name + '.' + n)
    recursive_traverse_childrens_leaf_only(module, update_child_param)

  recursive_traverse_childrens(model, f, filter)

def move_non_moe_to_gpu(model, device='cuda', filter=RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)').reverse()):
  def f(module : torch.nn.Module, name:str):
      # print(f'moving {name} to gpu')
      module.to(device)
  recursive_traverse_childrens_leaf_only(model, f, filter)

def replace_mlp_report_experts(model, prefetch_mngr, predictor, num_moe_layer, num_expert, num_predict_expert, filter=RegexFilter(r'.*layers\.([1-9]\d*)\.mlp$')):
  def f(module, name):
    def new_report_experts(experts):
      # print(module._layer_id, experts)
      prefetch_mngr.preempt_one_layer(module._layer_id, experts)

      prefetch_mngr.record_then_predict_and_launch(module._layer_id, experts)
      # predictor.add_one_layer(module._layer_id, experts)
      # if module._layer_id == num_moe_layer-1:
      #   predicted_prob = predictor.predict()
      #   # print(predicted_prob.shape)
      #   predicted_prob = predicted_prob.reshape(num_moe_layer, num_expert)
      #   _, _predict_idx = predicted_prob.sort(dim=-1, descending=True)
      #   for i in range(num_moe_layer):
      #     prefetch_mngr.add_one_layer_task(i, _predict_idx[i][:num_predict_expert])
      #   predictor.clear_access_buffer()

      return experts
    module.report_experts = new_report_experts
  recursive_traverse_childrens(model, f, filter)

def add_hook_to_experts(model, prefetch_mngr, filter=RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)$')):
  hook = ExpertHook(prefetch_mngr)
  def f(module, name):
    # print(f'adding hook to {name}')
    hooks.add_hook_to_module(module, hook)
  recursive_traverse_childrens(model, f, filter)

def add_hook_to_some_modules(model, hook, filter=RegexFilter(r'.*'), append=False):
  def f(module, name):
    # print(f'adding hook to {name}')
    hooks.add_hook_to_module(module, hook, append)
  recursive_traverse_childrens(model, f, filter)

'''
Legacy
'''
def pin_weights_map(weights_map) :
  for t in weights_map.dataset.values():
    pinned_t = t.pin_memory()
    t.set_(source = pinned_t)
def pin_hook_map(hook_map : dict[str, any]):
  for hook in hook_map.values():
    pin_weights_map(hook.weights_map)