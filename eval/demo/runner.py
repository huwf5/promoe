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
  'early_preempt'                : OptionCmdLine('early_preempt'),
  'max_prefetch_layer_distance'  : OptionCmdLine('max_prefetch_layer_distance'),
  'predict_input_mode'           : OptionCmdLine('predict_input_mode'),
  'predictor_model_path'         : OptionCmdLine('predictor_model_path', readable_name=False, logname=False),
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

### options to control: prefetch, reorder, early_preempt

full_list.concat(base_cfg_list.copy()
  .override('cache_policy', ['lru'])
  .override('per_layer_cache', [True])
  .override('reorder_experts', [False, True])
  .override('num_predict_expert_per_layer', [prefetch_len for prefetch_len in [0, 5, 6]])
  .override('early_preempt', [False, True])
  .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]])
)

if __name__ == '__main__':
  from eval_helper.runner_args import parse_args
  args = parse_args()
  if 'run' in args.commands:
    full_list.run(mock=args.mock, durable_log=args.durable_log, fail_only=args.fail_only)
  if 'parse' in args.commands:
    full_list.override('logdir', [args.logdir])
    full_list.parse()
    full_list.to_pdframe([
      'cache_policy',
      'per_layer_cache',
      'reorder_experts',
      'early_preempt',
      'num_predict_expert_per_layer',
      'cache_rate',
      'decode_stage_hit_rate',
      'prefill_stage_hit_rate',
      'decode_stage_forward_time',
      'prefill_stage_forward_time',
    ]).to_csv(args.parse_output, index=False)
