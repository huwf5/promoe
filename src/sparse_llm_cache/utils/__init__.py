import re

from .hooks import *

def recursive_traverse_childrens(module, func, regex_filter=None, prefix='', reverse_filter=False):
  # match=None
  # if regex_filter:
  #   match = re.match(regex_filter, prefix)
  # if not regex_filter or (bool(match) != reverse_filter):
  #   func(module, prefix, match)
  if not regex_filter:
    func(module, prefix, None)
  else:
    match = re.match(regex_filter, prefix)
    if bool(match) != reverse_filter:
      func(module, prefix, match)

  for child_name, child in module.named_children():
    recursive_traverse_childrens(child, func, regex_filter, f'{prefix}.{child_name}' if prefix != '' else child_name, reverse_filter)

def recursive_traverse_childrens_leaf_only(module, func, regex_filter=None, prefix='', reverse_filter=False):
  num_children = 0
  for child_name, child in module.named_children():
    recursive_traverse_childrens_leaf_only(child, func, regex_filter, f'{prefix}.{child_name}' if prefix != '' else child_name, reverse_filter)
    num_children+=1
  if num_children == 0:
    if not regex_filter:
      func(module, prefix, None)
    else:
      match = re.match(regex_filter, prefix)
      if bool(match) != reverse_filter:
        func(module, prefix, match)


# def map_childrens(module, func, prefix='', regex_filter=None):

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