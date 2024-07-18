'''
Hook from huggingface accerlerate
'''
import functools
import torch
from .. import cpp_worker

class ModelHook:
    """
    A hook that contains callbacks to be executed just before and after the forward method of a model. The difference
    with PyTorch existing hooks is that they get passed along the kwargs.

    Class attribute:
    - **no_grad** (`bool`, *optional*, defaults to `False`) -- Whether or not to execute the actual forward pass under
      the `torch.no_grad()` context manager.
    """

    no_grad = False

    def init_hook(self, module):
        """
        To be executed when the hook is attached to the module.

        Args:
            module (`torch.nn.Module`): The module attached to this hook.
        """
        return module

    def pre_forward(self, module, *args, **kwargs):
        """
        To be executed just before the forward method of the model.

        Args:
            module (`torch.nn.Module`): The module whose forward pass will be executed just after this event.
            args (`Tuple[Any]`): The positional arguments passed to the module.
            kwargs (`Dict[Str, Any]`): The keyword arguments passed to the module.

        Returns:
            `Tuple[Tuple[Any], Dict[Str, Any]]`: A tuple with the treated `args` and `kwargs`.
        """
        return args, kwargs

    def post_forward(self, module, output):
        """
        To be executed just after the forward method of the model.

        Args:
            module (`torch.nn.Module`): The module whose forward pass been executed just before this event.
            output (`Any`): The output of the module.

        Returns:
            `Any`: The processed `output`.
        """
        return output

    def detach_hook(self, module):
        """
        To be executed when the hook is detached from a module.

        Args:
            module (`torch.nn.Module`): The module detached from this hook.
        """
        return module

class SequentialHook(ModelHook):
    """
    A hook that can contain several hooks and iterates through them at each event.
    """

    def __init__(self, *hooks):
        self.hooks = hooks

    def init_hook(self, module):
        for hook in self.hooks:
            module = hook.init_hook(module)
        return module

    def pre_forward(self, module, *args, **kwargs):
        for hook in self.hooks:
            args, kwargs = hook.pre_forward(module, *args, **kwargs)
        return args, kwargs

    def post_forward(self, module, output):
        for hook in reversed(self.hooks):
            output = hook.post_forward(module, output)
        return output

    def detach_hook(self, module):
        for hook in self.hooks:
            module = hook.detach_hook(module)
        return module

def add_hook_to_module(module: torch.nn.Module, hook: ModelHook, append: bool = False):
    """
    Adds a hook to a given module. This will rewrite the `forward` method of the module to include the hook, to remove
    this behavior and restore the original `forward` method, use `remove_hook_from_module`.

    <Tip warning={true}>

    If the module already contains a hook, this will replace it with the new hook passed by default. To chain two hooks
    together, pass `append=True`, so it chains the current and new hook into an instance of the `SequentialHook` class.

    </Tip>

    Args:
        module (`torch.nn.Module`):
            The module to attach a hook to.
        hook (`ModelHook`):
            The hook to attach.
        append (`bool`, *optional*, defaults to `False`):
            Whether the hook should be chained with an existing one (if module already contains a hook) or not.

    Returns:
        `torch.nn.Module`: The same module, with the hook attached (the module is modified in place, so the result can
        be discarded).
    """

    if append and (getattr(module, "_hf_hook", None) is not None):
        old_hook = module._hf_hook
        remove_hook_from_module(module)
        hook = SequentialHook(old_hook, hook)

    if hasattr(module, "_hf_hook") and hasattr(module, "_old_forward"):
        # If we already put some hook on this module, we replace it with the new one.
        old_forward = module._old_forward
    else:
        old_forward = module.forward
        module._old_forward = old_forward

    module = hook.init_hook(module)
    module._hf_hook = hook

    def new_forward(module, *args, **kwargs):
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
        if module._hf_hook.no_grad:
            with torch.no_grad():
                output = module._old_forward(*args, **kwargs)
        else:
            output = module._old_forward(*args, **kwargs)
        return module._hf_hook.post_forward(module, output)

    module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

    return module

def remove_hook_from_module(module: torch.nn.Module, recurse=False):
    """
    Removes any hook attached to a module via `add_hook_to_module`.

    Args:
        module (`torch.nn.Module`): The module to attach a hook to.
        recurse (`bool`, **optional**): Whether to remove the hooks recursively

    Returns:
        `torch.nn.Module`: The same module, with the hook detached (the module is modified in place, so the result can
        be discarded).
    """

    if hasattr(module, "_hf_hook"):
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")

    if hasattr(module, "_old_forward"):
        module.forward = module._old_forward
        delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module

class MoEAttnHook(ModelHook):
  def __init__(self, prefetch_mngr):
    self.prefetch_mngr = prefetch_mngr
    pass
  def pre_forward(self, module, *args, **kwargs):
    # self.prefetch_mngr.report_one_expert(module._layer_id, module._expert_id)
    self.prefetch_mngr.report_moe_attn_logits(module._layer_id, kwargs['hidden_states'])
    return args, kwargs
#   def post_forward(self, module, output):
#     # self.prefetch_mngr.one_expert_done(module._layer_id, module._expert_id)
#     return output
  # def detach_hook(self, module):
  #   return super().detach_hook(module)

class ExpertHook(ModelHook):
  def __init__(self, prefetch_mngr):
    self.prefetch_mngr = prefetch_mngr
    pass
  # def init_hook(self, module):
  #   return super().init_hook(module)
  def pre_forward(self, module, *args, **kwargs):
    self.prefetch_mngr.report_one_expert(module._layer_id, module._expert_id)
    return args, kwargs
  def post_forward(self, module, output):
    self.prefetch_mngr.one_expert_done(module._layer_id, module._expert_id)
    return output
  # def detach_hook(self, module):
  #   return super().detach_hook(module)

class MoeLayerHook(ModelHook):
  def __init__(self, prefetch_mngr, extract_logits_from_input = None, extract_logits_from_output = None):
    self.prefetch_mngr = prefetch_mngr
    if extract_logits_from_input is None:
      def extract_logits_from_input(*args, **kwargs):
        return args[0]
    if extract_logits_from_output is None:
      def extract_logits_from_output(output):
        return output[0]
    self.extract_logits_from_input = extract_logits_from_input
    self.extract_logits_from_output = extract_logits_from_output
    pass
  # def init_hook(self, module):
  #   return super().init_hook(module)
  def pre_forward(self, module, *args, **kwargs):
    if module._layer_id == 0:
      self.prefetch_mngr.report_moe_layer_logits(module._layer_id, self.extract_logits_from_input(*args, **kwargs))
    return args, kwargs
  def post_forward(self, module, output):
    self.prefetch_mngr.report_moe_layer_logits(module._layer_id + 1, self.extract_logits_from_output(output))
    self.prefetch_mngr.one_moe_layer_done(module._layer_id)
    return output
  # def detach_hook(self, module):
  #   return super().detach_hook(module)

class TraceEventHook(ModelHook):
  def __init__(self):
    self.trace_guard_stack = []
  # def init_hook(self, module):
  #   return super().init_hook(module)
  def pre_forward(self, module, *args, **kwargs):
    self.trace_guard_stack.append(cpp_worker.TraceEventGuard())
    self.trace_guard_stack[-1].init(cpp_worker.kPythonMain, module._prefix)
    return args, kwargs
  def post_forward(self, module, output):
    self.trace_guard_stack.pop().release()
    return output
  # def detach_hook(self, module):
  #   return super().detach_hook(module)

class TimingHook(ModelHook):
  def __init__(self, prefetch_mngr):
    self.timing_guard = prefetch_mngr.build_timer()
#   def init_hook(self, module):
#     print(f"add timing hook to {module._prefix}")
#     return super().init_hook(module)
  def pre_forward(self, module, *args, **kwargs):
    self.timing_guard.init(cpp_worker.kModelForward)
    return args, kwargs
  def post_forward(self, module, output):
    self.timing_guard.release()
    return output
  # def detach_hook(self, module):
  #   return super().detach_hook(module)

# class GateHook(hooks.ModelHook):
#   def __init__(self):
#     pass
#   # def init_hook(self, module):
#   #   return super().init_hook(module)
#   # def pre_forward(self, module, *args, **kwargs):
#   #   return args, kwargs
#   # def post_forward(self, module, output):
#   #   return output
#   # def detach_hook(self, module):
#   #   return super().detach_hook(module)

# class SharedExpertHook(hooks.ModelHook):
#   def __init__(self, num_moe_layer, num_expert):
#     self.num_moe_layer = num_moe_layer
#     self.num_expert = num_expert
#   # def init_hook(self, module):
#   #   return super().init_hook(module)
#   def pre_forward(self, module, *args, **kwargs):
#     pre_layer_id = (module._layer_id + 1) % self.num_moe_layer
#     for e in range(self.num_expert):
#       prefetch_mngr.try_release_expert(pre_layer_id, e)
#     return args, kwargs
#   # def post_forward(self, module, output):
#   #   return output
#   # def detach_hook(self, module):
#   #   return super().detach_hook(module)

# from functools import wraps

# class Hook:
#   def on_call(self, func, *args, **kwds):
#     self.call_report_impl(func, *args, **kwds)
#   def call_report_impl(self, *args, **kwds):
#     pass
#   def on_ret(self, func, ret):
#     self.ret_report_impl(func, ret)
#   def ret_report_impl(self, func, ret):
#     pass
# class DefaultHook(Hook):
#   def call_report_impl(self, func, *args, **kwds):
#     print(f"calling {func}")
#   def ret_report_impl(self, func, ret):
#     print(f"return from {func}")
# class MultiHook(Hook):
#   def __init__(self, reporters = []):
#     self.reporters = reporters
#   def add(self, reporter):
#     self.reporters.append(reporter)
#     return self
#   def clear(self):
#     self.reporters = []
#     return self
#   def call_report_impl(self, func, *args, **kwds):
#     for reporter in self.reporters:
#       reporter.call_report_impl(func, *args, **kwds)
#   def ret_report_impl(self, func, ret):
#     for reporter in self.reporters:
#       reporter.ret_report_impl(func, ret)

# # reporter = MultiHook([])

# def add_hook(func, hook_reporter):
#   @wraps(func)
#   def wrapper(*args, **kwargs):
#     hook_reporter.on_call(func, *args, **kwargs)
#     ret = func(*args, **kwargs)
#     hook_reporter.on_ret(func, ret)
#     return ret
#   return wrapper


# def handle_deepseek_model(model):
#   model


# class ModelHook:
#   def __init__(self) -> None:
#     pass

# class GateHook:
#   def __init__(self) -> None:
#     pass