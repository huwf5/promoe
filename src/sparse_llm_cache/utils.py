def pin_weights_map(weights_map) :
  for t in weights_map.dataset.values():
    pinned_t = t.pin_memory()
    t.set_(source = pinned_t)
def pin_hook_map(hook_map : dict[str, any]):
  for hook in hook_map.values():
    pin_weights_map(hook.weights_map)