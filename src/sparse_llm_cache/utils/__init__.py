import re

from .hooks import *
from .filter import *
from .common_metas import *

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


def repo_folder_name(repo_id: str, repo_type: str = 'model') -> str:
  """
  Copied from huggingface_hub
  Return a serialized version of a hf.co repo name and type, safe for disk storage
  as a single non-nested folder.

  Example: models--julien-c--EsperBERTo-small
  """
  # remove all `/` occurrences to correctly convert repo to directory name
  parts = [f"{repo_type}s", *repo_id.split("/")]
  return '--'.join(parts)

def inject_model(
    model : torch.nn.Module,
    # cache configs
    cache_rate : float,
    num_predict_expert_per_layer : int,
    cache_len : int = None,
    max_prefetch_layer_distance = -1,
    per_layer_cache : bool = True,
    cache_policy : str = 'lru',
    cache_device : str|int = 'cuda',
    # metadatas of model
    num_moe_layer : int = None,
    num_expert_per_layer : int = None,
    expert_meta_parser = None,
    expert_name_filter = None,
    moe_layer_name_filter = None,
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
    cache_rate (float): The cache rate.
    cache_len (int): The length of the cache.
      Default is None. This overrides cache_rate.
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
    per_layer_cache (bool, optional):
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
  model_id = model.config._name_or_path

  if num_moe_layer is None:
    auto_infered_model_metas = auto_infer_model_metas(model_id, return_dict=False)
    num_moe_layer         = auto_infered_model_metas.num_moe_layer
    num_expert_per_layer  = auto_infered_model_metas.num_expert_per_layer
    expert_meta_parser    = auto_infered_model_metas.expert_meta_parser
    expert_name_filter    = auto_infered_model_metas.expert_name_filter
    moe_layer_name_filter = auto_infered_model_metas.moe_layer_name_filter

  if max_prefetch_layer_distance is None or max_prefetch_layer_distance == -1:
    max_prefetch_layer_distance = num_moe_layer - 1
  if cache_len is None:
    if per_layer_cache:
      cache_len = round(cache_rate * num_expert_per_layer) * num_moe_layer
    else:
      cache_len = round(cache_rate * num_moe_layer * num_expert_per_layer)

  meta = cpp_worker.ModuleMeta(num_moe_layer, num_expert_per_layer)
  meta.init_param_list([k for k,_ in model.model.layers[1].mlp.experts[0].named_parameters()])
  meta.num_predict_expert_per_layer = num_predict_expert_per_layer
  meta.max_prefetch_layer_distance = max_prefetch_layer_distance
  meta.per_layer_cache = per_layer_cache
  meta.cache_policy = cache_policy

  model_loader  = cpp_worker.ModelLoader(meta)
  predictor     = cpp_worker.Predictor(meta)
  prefetch_mngr = cpp_worker.PrefetchMngr(meta, model_loader, predictor)
  print("initializing cache lib...done")

  add_metadata_to_submodules(model, expert_meta_parser)

  print("injecting model...")
  register_expert_params(model, model_loader, expert_name_filter)
  replace_mlp_report_experts(model, prefetch_mngr, predictor, num_moe_layer, num_expert_per_layer, num_predict_expert_per_layer, moe_layer_name_filter)

  # fixme: a general model path
  predictor.load_model(f"/nvme/songxiaoniu/moe/moe-predict-models/{repo_folder_name(repo_id = model_id)}.pt")
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