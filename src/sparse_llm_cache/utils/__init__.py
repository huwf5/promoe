import os
import re
import math

from . import hooks
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

def add_metadata_to_submodules(model, parse_or_attach):
  def f(module, name):
    if hasattr(parse_or_attach, "__self__"):
      parse_or_attach(module, name)
      return
    module._prefix = name
    expert_meta = parse_or_attach(name)
    try:
      module._layer_id = int(expert_meta[0])
      module._expert_id = int(expert_meta[1])
    except TypeError as e:
      pass
  recursive_traverse_childrens(model, f)

def add_prefix_to_submodules(model):
  def f(module, name):
    module._prefix = name
  recursive_traverse_childrens(model, f)

def param_buffer_name_to_prefetch(name):
  if name.find('qweight') != -1:
    return True
  if name.find('zero') != -1:
    return True
  if name.find('scale') != -1:
    return True
  return False

def register_expert_params(model, model_loader, filter=RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)$')):
  def f(module, name):
    # print(f'registering expert {name}')
    for k,v in module.named_parameters():
      model_loader.add_one_expert_param(v, module._layer_id, module._expert_id, k)
    for k,v in module.named_buffers():
      if not param_buffer_name_to_prefetch(k): continue
      model_loader.add_one_expert_param(v, module._layer_id, module._expert_id, k)
  recursive_traverse_childrens(model, f, filter)

def replace_expert_param_reference(model, model_loader, filter=RegexFilter(r'.*layers\.(\d+)\.mlp\.experts\.(\d+)$')):
  model_loader.build_logical_expert_param()
  def f(module, name):
    def update_child_param(child_module, child_name):
      for n,_ in child_module.named_parameters():
        child_module._parameters[n] = model_loader.ref_one_expert_param(module._layer_id, module._expert_id, child_name + '.' + n)
      for n,_ in child_module.named_buffers():
        if not param_buffer_name_to_prefetch(n): continue
        child_module._buffers[n] = model_loader.ref_one_expert_param(module._layer_id, module._expert_id, child_name + '.' + n)
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

def replace_mlp_report_experts(model, prefetch_mngr, predictor, num_moe_layer, num_expert, num_predict_expert, filter=RegexFilter(r'.*layers\.([1-9]\d*)\.mlp$'), adapter=None):
  def f(module, name):
    if adapter is not None and not adapter.should_patch_report_experts(module):
      return
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

def add_hook_to_moe_layers(model, prefetch_mngr, filter, adapter=None):
  hook = MoeLayerHook(prefetch_mngr, adapter=adapter)
  def f(module, name):
    # print(f'adding hook to {name}')
    hooks.add_hook_to_module(module, hook)
  recursive_traverse_childrens(model, f, filter)

def add_hook_to_quant_expert_post_init(model, prefetch_mngr, filter):
  first_hook = AutoGPTQFirstPostInitHook(prefetch_mngr)
  last_hook  = AutoGPTQLastPostInitHook(prefetch_mngr)
  def f(module, name):
    first_module = None
    last_module  = None
    for name, submodule in module.named_modules():
      if hasattr(submodule, "QUANT_TYPE") == False: continue
      if first_module is None:
        first_module = submodule
      last_module = submodule
    if first_module != None:
      hooks.add_hook_to_module_custom_method(first_module, first_hook, method_name='post_init', hook_attr_name='_post_init_hook')
      hooks.add_hook_to_module_custom_method(last_module,  last_hook,  method_name='post_init', hook_attr_name='_post_init_hook')
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

def wrap_generate_with_initial_cache(model, prefetch_mngr):
  if hasattr(model, "_sparse_cache_old_generate"):
    return
  model._sparse_cache_old_generate = model.generate

  def generate_with_initial_cache(*args, **kwargs):
    prefetch_mngr.reset_for_generate()
    return model._sparse_cache_old_generate(*args, **kwargs)

  model.generate = generate_with_initial_cache

def round_like_cpp(value):
  return math.floor(float(value) + 0.5)

def _resolve_initial_cache_inputs(
    initial_cache_policy,
    initial_layer_budgets,
    initial_hot_expert_file,
    per_layer_cache,
):
  if initial_cache_policy is None:
    if initial_layer_budgets:
      raise ValueError("initial_layer_budgets requires initial_cache_policy=manual")
    if initial_hot_expert_file:
      raise ValueError("initial_hot_expert_file requires initial_cache_policy=hot_expert")
    return {
      "initial_cache_policy": None,
      "initial_layer_budgets": None,
      "initial_hot_expert_file": None,
      "reset_cache_on_generate_start": False,
      "per_layer_cache": per_layer_cache,
    }

  if initial_cache_policy == "manual":
    if not initial_layer_budgets:
      raise ValueError("manual initial cache requires initial_layer_budgets")
    if initial_hot_expert_file:
      raise ValueError("manual initial cache does not use initial_hot_expert_file")
  elif initial_cache_policy == "hot_expert":
    if not initial_hot_expert_file:
      raise ValueError("hot_expert initial cache requires initial_hot_expert_file")
    if initial_layer_budgets:
      raise ValueError("hot_expert initial cache does not use initial_layer_budgets")
  else:
    raise ValueError(f"unsupported initial_cache_policy={initial_cache_policy!r}")

  return {
    # "hot_expert" is also use "manual" into cpp
    "initial_cache_policy": "manual",
    "initial_layer_budgets": initial_layer_budgets if initial_cache_policy == "manual" else None,
    "initial_hot_expert_file": initial_hot_expert_file if initial_cache_policy == "hot_expert" else None,
    "reset_cache_on_generate_start": True,
    "per_layer_cache": False,
  }

def inject_model(
    model : torch.nn.Module,
    model_id = None,
    # metadatas of model
    num_moe_layer : int = None,
    num_expert_per_layer : int = None,
    num_expert_per_token : int = None,
    # cache configs
    cache_rate : float = None,
    num_predict_expert_per_layer : int = None,
    reorder_experts : bool = None,
    early_preempt   : bool = None,
    chunk_prefetch  : bool = None,

    predict_input_mode = None,
    predictor_type : str = None,

    predictor_model_path : str = None,
    layer_predict_interval     = None,
    layer_predict_max_window   = None,
    layer_predict_replace_first_input_with_last_output = False,

    limit_layer_0_window      = None,
    limit_layer_0_num_predict = None,

    # deprecated
    max_prefetch_layer_distance = None,
    cache_only = False,
    per_layer_cache : bool = None,
    promote_hit_in_prefetch : bool = None,
    cache_policy : str = None,
    initial_cache_policy: str | None = None,
    initial_layer_budgets: str | None = None,
    initial_hot_expert_file: str | None = None,
    enable_decoder_warmup_overlap: bool = False,

    cache_device : str|int = 'cuda',
    pin_memory : bool  = True,
    module_trace_event : bool = False,
    enable_model_timer : bool = True,
    trace_event : bool = False,
    cache_trace_path : str = None,
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
  model_id = model.config._name_or_path if model_id is None else model_id
  from sparse_llm_cache.model_adapters import get_model_adapter
  adapter = get_model_adapter(model, model_id)

  if num_moe_layer is None:
    num_moe_layer         = adapter.num_moe_layer
    num_expert_per_layer  = adapter.num_expert_per_layer
    num_expert_per_token  = adapter.num_expert_per_token
    expert_meta_parser    = adapter.expert_meta_parser
    expert_name_filter    = adapter.expert_name_filter
    moe_mlp_name_filter   = adapter.moe_mlp_name_filter
    moe_layer_name_filter = adapter.moe_layer_name_filter
    moe_attn_name_filter  = adapter.moe_attn_name_filter
  else:
    auto_infered_model_metas = auto_infer_model_metas(model_id, return_dict=False)
    num_moe_layer         = auto_infered_model_metas.num_moe_layer
    num_expert_per_layer  = auto_infered_model_metas.num_expert_per_layer
    num_expert_per_token  = auto_infered_model_metas.num_expert_per_token
    expert_meta_parser    = auto_infered_model_metas.expert_meta_parser
    expert_name_filter    = auto_infered_model_metas.expert_name_filter
    moe_mlp_name_filter   = auto_infered_model_metas.moe_mlp_name_filter
    moe_layer_name_filter = auto_infered_model_metas.moe_layer_name_filter
    moe_attn_name_filter  = auto_infered_model_metas.moe_attn_name_filter

  meta = cpp_worker.ModuleMeta(num_moe_layer, num_expert_per_layer)
  meta.model_arch_string = model_id
  meta.num_expert_per_token = num_expert_per_token

  def find_first_expert(model, filter):
    first_expert_module = [None]
    def f(module, name):
      if first_expert_module[0] is None:
        first_expert_module[0] = module
    recursive_traverse_childrens(model, f, filter)
    return first_expert_module[0]
  first_expert = find_first_expert(model, expert_name_filter)
  param_key_list = [k for k,_ in first_expert.named_parameters()]
  param_key_list += [k for k,_ in first_expert.named_buffers() if param_buffer_name_to_prefetch(k)]
  print(param_key_list)
  meta.init_param_list(param_key_list)

  initial_inputs = _resolve_initial_cache_inputs(
    initial_cache_policy,
    initial_layer_budgets,
    initial_hot_expert_file,
    per_layer_cache,
  )
  initial_cache_policy = initial_inputs["initial_cache_policy"]
  initial_layer_budgets = initial_inputs["initial_layer_budgets"]
  initial_hot_expert_file = initial_inputs["initial_hot_expert_file"]
  reset_cache_on_generate_start = initial_inputs["reset_cache_on_generate_start"]
  per_layer_cache = initial_inputs["per_layer_cache"]

  adapter.configure_module_meta(meta)
  initial_plan = None
  initial_expert_plan = None
  decoder_warmup_expert_plan = None
  if initial_hot_expert_file:
    from sparse_llm_cache.utils.hot_experts import (
      build_decoder_warmup_overlap_plan,
      build_hot_initial_plan,
      format_initial_expert_plan,
    )
    effective_cache_rate = 0.5 if cache_rate is None else float(cache_rate)
    initial_total_slots = round_like_cpp(
      effective_cache_rate * int(num_moe_layer) * int(num_expert_per_layer)
    )
    initial_plan = build_hot_initial_plan(
      initial_hot_expert_file,
      adapter,
      total_slots=initial_total_slots,
    )
    initial_expert_plan = format_initial_expert_plan(initial_plan)
  if enable_decoder_warmup_overlap:
    if not initial_hot_expert_file:
      raise ValueError("enable_decoder_warmup_overlap requires initial_hot_expert_file")
    decoder_warmup_plan = build_decoder_warmup_overlap_plan(
      initial_hot_expert_file,
      adapter,
      initial_plan=set(initial_plan or []),
    )
    decoder_warmup_expert_plan = format_initial_expert_plan(decoder_warmup_plan)

  param_dict = {
    'model_arch_string'            : str(model_id),
    'num_expert_per_token'         : str(num_expert_per_token),
    'cache_rate'                   : str(cache_rate),
    'num_predict_expert_per_layer' : str(num_predict_expert_per_layer),
    'reorder_experts'              : str(reorder_experts),
    'early_preempt'                : str(early_preempt),
    'chunk_prefetch'               : str(chunk_prefetch),
    'predict_input_mode'           : str(predict_input_mode),
    'predictor_type'               : str(predictor_type),
    'predictor_model_path'         : str(predictor_model_path),
    'layer_predict_interval'       : str(layer_predict_interval),
    'layer_predict_max_window'     : str(layer_predict_max_window),
    'cross_token_pred'             : str(layer_predict_replace_first_input_with_last_output),
    'limit_layer_0_window'         : str(limit_layer_0_window),
    'limit_layer_0_num_predict'    : str(limit_layer_0_num_predict),
    'max_prefetch_layer_distance'  : str(max_prefetch_layer_distance),
    'cache_only'                   : str(cache_only),
    'per_layer_cache'              : str(per_layer_cache),
    'promote_hit_in_prefetch'      : str(promote_hit_in_prefetch),
    'cache_policy'                 : str(cache_policy),
    'num_encoder_moe_layer'        : str(meta.num_encoder_moe_layer),
    'num_decoder_moe_layer'        : str(meta.num_decoder_moe_layer),
    'reset_cache_on_generate_start': str(reset_cache_on_generate_start),
    'initial_cache_policy'         : str(initial_cache_policy),
    'initial_layer_budgets'        : str(initial_layer_budgets),
    'initial_expert_plan'          : str(initial_expert_plan),
    'enable_decoder_warmup_overlap': str(enable_decoder_warmup_overlap),
    'decoder_warmup_expert_plan'   : str(decoder_warmup_expert_plan),
  }

  meta.init_from_map(param_dict)

  adapter.configure_module_meta(meta)
  meta.handle_uninited_configs()
  adapter.validate_predictor_path(
    predictor_model_path, num_predict_expert_per_layer, predictor_type
  )

  model_loader  = cpp_worker.ModelLoader(meta)
  predictor     = cpp_worker.PredictorBase.create(meta)
  prefetch_mngr = cpp_worker.PrefetchMngr(meta, model_loader, predictor)
  torch.cuda.set_stream(torch.cuda.ExternalStream(prefetch_mngr.compute_stream, 0))
  print("initializing cache lib...done")

  add_metadata_to_submodules(model, adapter.add_metadata_to_module)

  print("injecting model...")
  register_expert_params(model, model_loader, expert_name_filter)
  prefetch_mngr.init_gpu_mem_buffer()
  replace_expert_param_reference(model, model_loader, expert_name_filter)
  replace_mlp_report_experts(
    model,
    prefetch_mngr,
    predictor,
    num_moe_layer,
    num_expert_per_layer,
    num_predict_expert_per_layer,
    moe_mlp_name_filter,
    adapter=adapter,
  )

  if num_predict_expert_per_layer:
    predictor.load_model()

  if meta.cache_policy == 'min':
    # prefetch_mngr.cache.cache_oracle.load_from_file(cache_trace_path)
    prefill_expert_len       = torch.load(f'{cache_trace_path}/prefill_expert_len.pt')
    prefill_expert_selection = torch.load(f'{cache_trace_path}/prefill_expert_selection.pt')
    decode_expert_selection  = torch.load(f'{cache_trace_path}/decode_expert_selection.pt')
    entry_to_metas           = torch.load(f'{cache_trace_path}/entry_to_metas.pt')
    prefetch_mngr.cache.cache_oracle.load_from_tensor(entry_to_metas, prefill_expert_len, prefill_expert_selection, decode_expert_selection)

  add_hook_to_experts(model, prefetch_mngr, expert_name_filter)
  # add_hook_to_moe_attns(model, prefetch_mngr, moe_attn_name_filter)
  add_hook_to_moe_layers(model, prefetch_mngr, moe_layer_name_filter, adapter=adapter)
  add_hook_to_quant_expert_post_init(model, prefetch_mngr, expert_name_filter)
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
  if reset_cache_on_generate_start:
    wrap_generate_with_initial_cache(model, prefetch_mngr)
  if launch_now:
    launch(model)
  return prefetch_mngr

def launch(model : torch.nn.Module):
  prefetch_mngr = model._prefetch_mngr
  prefetch_mngr.launch_thread()
  torch.set_num_threads(8)
  return prefetch_mngr


'''
Hiject Transformers
'''
import functools
import importlib
from transformers.modeling_utils import PreTrainedModel

if importlib.util.find_spec('auto_gptq') is not None:
  from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear

def hack_auto_gptq():
  if hasattr(QuantLinear, '_old_post_init'):
    return
  QuantLinear._old_post_init = QuantLinear.post_init
  @functools.wraps(QuantLinear.post_init)
  def new_post_init(module):
    if hasattr(module, '_expert_id'):
      module._prefetch_mngr.temp_move_expert_to_gpu(module._layer_id, module._expert_id)
    ret = QuantLinear._old_post_init(module)
    return ret
  QuantLinear.post_init = new_post_init

def hack_transformers(**sparse_cache_kwargs):
  PreTrainedModel._old_load_pretrained_model = PreTrainedModel._load_pretrained_model
  @classmethod
  @functools.wraps(PreTrainedModel._load_pretrained_model)
  def new_load_pretrained_model(cls, *args, **kwargs):
    assert len(kwargs['device_map']) == 1 and '' in kwargs['device_map'], "only support single device"
    orig_device_map = kwargs['device_map']
    kwargs['device_map'] = {'' : 'cpu'}
    ret = PreTrainedModel._old_load_pretrained_model.__func__(cls, *args, **kwargs)
    kwargs['device_map'] = orig_device_map
    model = ret[0]
    inject_model(model, **sparse_cache_kwargs, launch_now=False)
    return ret
  PreTrainedModel._load_pretrained_model = new_load_pretrained_model

  PreTrainedModel._old_from_pretrained = PreTrainedModel.from_pretrained
  @classmethod
  @functools.wraps(PreTrainedModel.from_pretrained)
  def new_from_pretrained(cls, *args, **kwargs):
    ret = PreTrainedModel._old_from_pretrained.__func__(cls, *args, **kwargs)
    model = ret
    launch(model)
    return ret
  PreTrainedModel.from_pretrained = new_from_pretrained

  # if importlib.util.find_spec('auto_gptq') is not None:
  #   hack_auto_gptq()

def inject_model_um(
    model,
    model_id = None,
  ):
  from sparse_llm_cache.cpp_worker import to_um
  if model_id is None:
    model_id = model.config._name_or_path
  auto_infered_model_metas = auto_infer_model_metas(model_id, return_dict=False)
  add_metadata_to_submodules(model, auto_infered_model_metas.expert_meta_parser)
  def replace_expert_param_reference(model, filter):
    def f(module, name):
      def update_child_param(child_module, child_name):
        for n,v in child_module.named_parameters():
          child_module._parameters[n] = to_um(v)
        for n,v in child_module.named_buffers():
          if not param_buffer_name_to_prefetch(n): continue
          child_module._buffers[n] = to_um(v)
      recursive_traverse_childrens_leaf_only(module, update_child_param)
    recursive_traverse_childrens(model, f, filter)

  replace_expert_param_reference(model, auto_infered_model_metas.expert_name_filter)

def hack_transformers_um():
  from sparse_llm_cache.cpp_worker import to_um
  PreTrainedModel._old_load_pretrained_model = PreTrainedModel._load_pretrained_model
  @classmethod
  @functools.wraps(PreTrainedModel._load_pretrained_model)
  def new_load_pretrained_model(cls, *args, **kwargs):
    assert len(kwargs['device_map']) == 1 and '' in kwargs['device_map'], "only support single device"
    orig_device_map = kwargs['device_map']
    kwargs['device_map'] = {'' : 'cpu'}
    ret = PreTrainedModel._old_load_pretrained_model.__func__(cls, *args, **kwargs)
    kwargs['device_map'] = orig_device_map
    model = ret[0]

    auto_infered_model_metas = auto_infer_model_metas(model.config._name_or_path, return_dict=False)
    add_metadata_to_submodules(model, auto_infered_model_metas.expert_meta_parser)

    def replace_expert_param_reference(model, filter):
      def f(module, name):
        def update_child_param(child_module, child_name):
          for n,v in child_module.named_parameters():
            child_module._parameters[n] = to_um(v)
          for n,v in child_module.named_buffers():
            if not param_buffer_name_to_prefetch(n): continue
            child_module._buffers[n] = to_um(v)
        recursive_traverse_childrens_leaf_only(module, update_child_param)
      recursive_traverse_childrens(model, f, filter)

    replace_expert_param_reference(model, auto_infered_model_metas.expert_name_filter)
    return ret
  PreTrainedModel._load_pretrained_model = new_load_pretrained_model


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
