from eval_helper.config import RunConfigBase, OptionCmdLine, OptionEnv, OptionApp, ConfigList, ResultFloat

my_app = RunConfigBase()
my_app.app = OptionApp('python transformers-deepseek-moe.py', 'deepseek-moe', 'deepseek-moe')
my_app.logdir = 'run-logs'
my_app.config_dict = {
  'num_predict_expert_per_layer' : OptionCmdLine('num_predict_expert_per_layer', readable_name='predict', logname='predict'),
  'cache_rate'                   : OptionCmdLine('cache_rate'),
  'cache_policy'                 : OptionCmdLine('cache_policy', readable_name='policy', logname='policy'),
  'per_layer_cache'              : OptionCmdLine('per_layer_cache'),
  'reorder_experts'              : OptionCmdLine('reorder_experts'),
  'max_prefetch_layer_distance'  : OptionCmdLine('max_prefetch_layer_distance'),
}
my_app.result_dict = {
  'decode_stage_forward_time'  : ResultFloat('decode_stage_forward_time'),
  'prefill_stage_forward_time' : ResultFloat('prefill_stage_forward_time'),
  'decode_stage_hit_rate'      : ResultFloat('decode_stage_hit_rate'),
  'prefill_stage_hit_rate'     : ResultFloat('prefill_stage_hit_rate'),
}
my_app['per_layer_cache'] = True
base_cfg_list = ConfigList.MakeList(my_app)

full_list = ConfigList.Empty()

### no cache, prefetch only
full_list.concat(base_cfg_list.copy()
  .override('num_predict_expert_per_layer', [0, 1, 2, 3, 4, 5, 6])
  .override('cache_rate', [3/64])
  .override('cache_policy', ['lru'])
  .override('per_layer_cache', [False])
  .override('max_prefetch_layer_distance', [1])
)

### cache only, no prefetch
full_list.concat(base_cfg_list.copy()
  .override('reorder_experts', [None, False])
  .override('num_predict_expert_per_layer', [0])
  .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]])
  .override('cache_policy', ['lru'])
  .override('per_layer_cache', [True])
)

### cache + prefetch
full_list.concat(base_cfg_list.copy()
  .override('reorder_experts', [None, False])
  .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]])
  .override('num_predict_expert_per_layer', [prefetch_len for prefetch_len in range(1, 7)])
  .override('cache_policy', ['lru'])
  .override('per_layer_cache', [True])
)

full_list.run(mock=True)
full_list.parse()
full_list.to_pdframe([
  'cache_policy',
  'per_layer_cache',
  'reorder_experts',
  'num_predict_expert_per_layer',
  'cache_rate',
  'decode_stage_hit_rate',
  'prefill_stage_hit_rate',
  'decode_stage_forward_time',
  'prefill_stage_forward_time',
]).to_csv('output.csv', index=False)