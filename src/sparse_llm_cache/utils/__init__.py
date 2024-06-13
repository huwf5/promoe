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
      prefetch_mngr.report_one_layer(module._layer_id, experts)

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


def inject_model(
    model : torch.nn.Module,
    num_moe_layer : int,
    num_expert_per_layer : int,
    cache_len : int,
    num_predict_expert_per_layer : int,
    expert_meta_parser,
    expert_name_filter,
    moe_layer_name_filter,
    max_prefetch_layer_distance = None,
    enable_per_layer_cache : bool = True,
    cache_policy : str = 'lru',
    cache_device : str|int = 'cuda',
    pin_memory : bool  = True,
    enable_timing : bool = False,
  ):
  """
  Injects a model with cache-related functionality.

  Args:
    model (torch.nn.Module):
      The target model.
    num_moe_layer (int):
      The number of mixture-of-experts (MoE) layers in the model.
    num_expert_per_layer (int):
      The number of experts per MoE layer.
    cache_len (int): The length of the cache.
    num_predict_expert_per_layer (int): The number of experts to predict per MoE layer.
      Set to 0 can avoid prefetch. Fetch is only triggered on demand
      Set to >0 enables prefetch.
    expert_meta_parser:
      The expert meta parser.
    expert_name_filter:
      The expert name filter.
    moe_layer_name_filter:
      The MoE layer name filter.
    max_prefetch_layer_distance (int, optional):
      The maximum prefetch layer distance, avoiding prefetcher goes to fast.
      Defaults to None, meaning infinite.
    enable_per_layer_cache (bool, optional):
      Whether to enable per-layer cache. Defaults to True.
    cache_policy (str, optional):
      The cache policy. Defaults to 'lru'.
    cache_device (str, optional):
      The cache device. Defaults to 'cuda'.
    pin_memory (bool, optional):
      Whether to pin model parameters on CPU. Defaults to True.
    enable_timing (bool, optional):
      Whether to enable timing. Defaults to False.
  """
  print("initializing cache lib...")
  meta = cpp_worker.ModuleMeta(num_moe_layer, num_expert_per_layer)
  meta.init_param_list([k for k,_ in model.model.layers[1].mlp.experts[0].named_parameters()])
  meta.num_predict_expert_per_layer = num_predict_expert_per_layer
  if max_prefetch_layer_distance is None:
    max_prefetch_layer_distance = num_moe_layer - 1
  meta.max_prefetch_layer_distance = max_prefetch_layer_distance
  meta.per_layer_cache = enable_per_layer_cache
  meta.cache_policy = cache_policy
  model_loader = cpp_worker.ModelLoader(meta)
  predictor = cpp_worker.Predictor(meta)
  prefetch_mngr = cpp_worker.PrefetchMngr(meta, model_loader, predictor)
  print("initializing cache lib...done")

  add_metadata_to_submodules(model, expert_meta_parser)

  print("injecting model...")
  register_expert_params(model, model_loader, expert_name_filter)
  # model.to(cache_device)

  replace_mlp_report_experts(model, prefetch_mngr, predictor, num_moe_layer, num_expert_per_layer, num_predict_expert_per_layer, moe_layer_name_filter)

  predictor.load_model("/nvme/songxiaoniu/moe/moe-predict-models/models--deepseek-ai--deepseek-moe-16b-chat.pt")
  prefetch_mngr.init_gpu_mem_buffer(cache_len)

  add_hook_to_experts(model, prefetch_mngr, expert_name_filter)

  if enable_timing:
    timing_hook = hooks.TimingHook()
    add_hook_to_some_modules(model, timing_hook, append=True)
  print("injecting model...done")
  if pin_memory:
    print("pin model parameters on cpu...")
    model_loader.pin_memory()
    print("pin model parameters on cpu...done")
  prefetch_mngr.launch_thread()
  torch.set_num_threads(16)

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