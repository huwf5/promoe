import os
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
  recursive_traverse_childrens(model, f, filter)

  model_loader.build_logical_expert_param()
  def f(module, name):
    def update_child_param(child_module, child_name):
      for n,_ in child_module.named_parameters():
        child_module._parameters[n] = model_loader.ref_one_expert_param(module._layer_id, module._expert_id, child_name + '.' + n)
    recursive_traverse_childrens_leaf_only(module, update_child_param)

  recursive_traverse_childrens(model, f, filter)

def attach_prefetch_mngr_to_all_module(model, prefetch_mngr):
  def f(module, name):
    module._prefetch_mngr = prefetch_mngr
  recursive_traverse_childrens(model, f, filter=RegexFilter(r'.*'))

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

def add_hook_to_moe_attns(model, prefetch_mngr, filter):
  hook = MoEAttnHook(prefetch_mngr)
  def f(module, name):
    # print(f'adding hook to {name}')
    hooks.add_hook_to_module(module, hook)
  recursive_traverse_childrens(model, f, filter)

def add_hook_to_moe_layers(model, prefetch_mngr, filter):
  hook = MoeLayerHook(prefetch_mngr)
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
    reorder_experts : bool = True,
    promote_hit_in_prefetch : bool = True,
    early_preempt : bool = True,
    # metadatas of model
    num_moe_layer : int = None,
    num_expert_per_layer : int = None,
    num_expert_per_token : int = None,
    expert_meta_parser = None,
    expert_name_filter = None,
    moe_mlp_name_filter = None,
    moe_layer_name_filter = None,
    pin_memory : bool  = True,
    module_trace_event : bool = False,
    enable_model_timer : bool = False,
    trace_event : bool = False,
    cache_trace_path : str = None,
    predictor_model_path : str = None,
    predict_input_mode = None,
    layer_predict_interval   = -1,
    layer_predict_max_window = -1,
    model_id = None,
    layer_predict_replace_first_input_with_last_output = False,
    launch_now = True,
    **kwargs
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
    num_expert_per_token (int):
      The number of experts activated per token.
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
    reorder_experts (bool, optional):
      Whether to reorder experts. Defaults to True.
    cache_device (str, optional):
      The cache device. Defaults to 'cuda'.
    pin_memory (bool, optional):
      Whether to pin model parameters on CPU. Defaults to True.
    enable_module_trace_event (bool, optional):
      Whether to record event of each pytorch module. Defaults to False.
    enable_model_timer (bool, optional):
      Whether to enable timing of entire model. Defaults to False.
  """
  print("initializing cache lib...")
  if len(kwargs) > 0:
    print("warning, unused kwargs", kwargs)
  if model_id is None:
    model_id = model.config._name_or_path

  if num_moe_layer is None:
    auto_infered_model_metas = auto_infer_model_metas(model_id, return_dict=False)
    num_moe_layer         = auto_infered_model_metas.num_moe_layer
    num_expert_per_layer  = auto_infered_model_metas.num_expert_per_layer
    num_expert_per_token  = auto_infered_model_metas.num_expert_per_token
    expert_meta_parser    = auto_infered_model_metas.expert_meta_parser
    expert_name_filter    = auto_infered_model_metas.expert_name_filter
    moe_mlp_name_filter   = auto_infered_model_metas.moe_mlp_name_filter
    moe_layer_name_filter = auto_infered_model_metas.moe_layer_name_filter
    moe_attn_name_filter  = auto_infered_model_metas.moe_attn_name_filter

  if cache_len is None:
    if per_layer_cache:
      cache_len = round(cache_rate * num_expert_per_layer) * num_moe_layer
    else:
      cache_len = round(cache_rate * num_moe_layer * num_expert_per_layer)

  if predict_input_mode is None:
    predict_input_mode = 'one_token'

  meta = cpp_worker.ModuleMeta(num_moe_layer, num_expert_per_layer)

  def find_first_expert(model, filter):
    first_expert_module = [None]
    def f(module, name):
      if first_expert_module[0] is None:
        first_expert_module[0] = module
    recursive_traverse_childrens(model, f, filter)
    return first_expert_module[0]
  first_expert = find_first_expert(model, expert_name_filter)
  param_key_list = [k for k,_ in first_expert.named_parameters()]
  print(param_key_list)
  meta.init_param_list(param_key_list)

  meta.num_predict_expert_per_layer = num_predict_expert_per_layer
  meta.max_prefetch_layer_distance = max_prefetch_layer_distance
  meta.num_expert_per_token = num_expert_per_token
  meta.per_layer_cache = per_layer_cache
  meta.cache_policy = cache_policy
  meta.reorder_experts = reorder_experts
  meta.promote_hit_in_prefetch = promote_hit_in_prefetch
  meta.early_preempt = early_preempt
  meta.layer_predict_replace_first_input_with_last_output = layer_predict_replace_first_input_with_last_output
  meta.predict_input_mode = {
    'one_token': cpp_worker.kOneToken,
    'decode_cumsum': cpp_worker.kDecodeCumsum,
    'last_use_distance': cpp_worker.kLastUseDistance,
    'weighted_decode_cumsum': cpp_worker.kWeighedDecodeCumsum,
    'first_moe_attn_input_logits': cpp_worker.kFirstMoeAttnInputLogits,
    'moe_attn_input_logits': cpp_worker.kMoeAttnInputLogits,
    'moe_layer_logits': cpp_worker.kMoeLayerLogits,
  }[predict_input_mode]

  meta.layer_predict_interval = layer_predict_interval
  meta.layer_predict_max_window = layer_predict_max_window

  meta.handle_uninited_configs()

  model_loader  = cpp_worker.ModelLoader(meta)
  predictor     = cpp_worker.Predictor(meta)
  prefetch_mngr = cpp_worker.PrefetchMngr(meta, model_loader, predictor)
  print("initializing cache lib...done")

  add_metadata_to_submodules(model, expert_meta_parser)

  print("injecting model...")
  register_expert_params(model, model_loader, expert_name_filter)
  prefetch_mngr.init_gpu_mem_buffer(cache_len)
  replace_expert_param_reference(model, model_loader, expert_name_filter)
  replace_mlp_report_experts(model, prefetch_mngr, predictor, num_moe_layer, num_expert_per_layer, num_predict_expert_per_layer, moe_mlp_name_filter)

  # fixme: a general model path
  if predictor_model_path == None:
    predictor_model_path = f"/nvme/songxiaoniu/moe/moe-predict-models/{repo_folder_name(repo_id = model_id)}.pt"
  predictor.load_model(predictor_model_path)

  if meta.cache_policy == 'min':
    prefetch_mngr.cache.cache_oracle.load_from_file(cache_trace_path)

  add_hook_to_experts(model, prefetch_mngr, expert_name_filter)
  add_hook_to_moe_attns(model, prefetch_mngr, moe_attn_name_filter)
  add_hook_to_moe_layers(model, prefetch_mngr, moe_layer_name_filter)
  attach_prefetch_mngr_to_all_module(model, prefetch_mngr)

  if trace_event:
    os.environ['SPARSE_CACHE_ENABLE_TRACE'] = '1'
    prefetch_mngr.reload_env()
    if module_trace_event:
      trace_event_hook = hooks.TraceEventHook()
      add_hook_to_some_modules(model, trace_event_hook, append=True)
  if enable_model_timer:
    timing_hook = hooks.TimingHook(prefetch_mngr)
    add_hook_to_some_modules(model, timing_hook, filter=RegexFilter('^$'), append=True)
  print("injecting model...done")
  if pin_memory:
    print("pin model parameters on cpu...")
    model_loader.pin_memory()
    print("pin model parameters on cpu...done")
  model._prefetch_mngr = prefetch_mngr
  if launch_now:
    launch(model)
  return prefetch_mngr

def launch(model : torch.nn.Module):
  prefetch_mngr = model._prefetch_mngr
  prefetch_mngr.launch_thread()
  torch.set_num_threads(16)
  return prefetch_mngr

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