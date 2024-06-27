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
  'cache_trace_path'             : OptionCmdLine('cache_trace_path', readable_name=False, logname=False),
  'predictor_model_path'         : OptionCmdLine('predictor_model_path', readable_name=False, logname=False),
  'log_level'                    : OptionEnv('SPARSE_CACHE_LOG_LEVEL', readable_name=False, logname=False),
}
my_app.result_dict = {
  'decode_stage_forward_time'       : ResultFloat('decode_stage_forward_time'),
  'prefill_stage_forward_time'      : ResultFloat('prefill_stage_forward_time'),
  'decode_stage_hit_rate'           : ResultFloat('decode_stage_hit_rate'),
  'prefill_stage_hit_rate'          : ResultFloat('prefill_stage_hit_rate'),
  'decode_stage_prefetch_hit_rate'  : ResultFloat('decode_stage_prefetch_hit_rate'),
  'prefill_stage_prefetch_hit_rate' : ResultFloat('prefill_stage_prefetch_hit_rate'),
  'decode_stage_ready_rate'         : ResultFloat('decode_stage_ready_rate'),
  'prefill_stage_ready_rate'        : ResultFloat('prefill_stage_ready_rate'),
  'legacy_decode_stage_hit_rate'    : ResultFloat('legacy_decode_stage_hit_rate'),
  'legacy_prefill_stage_hit_rate'   : ResultFloat('legacy_prefill_stage_hit_rate'),
}
my_app['per_layer_cache'] = True
base_cfg_list = ConfigList.MakeList(my_app)

full_list = ConfigList.Empty()

### options to control: prefetch, reorder, early_preempt

full_list.concat(base_cfg_list.copy()
  .override('cache_policy', ['min', 'lru'])
  .override('per_layer_cache', [True, False])
  .override('cache_trace_path', ['/nvme/songxiaoniu/moe/moe-traces/deepseek-moe-sharegpt-0412.json'])
  .override('reorder_experts', [False, True])
  .override('num_predict_expert_per_layer', [0])
  .override('early_preempt', [False])
  .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]])
)

# full_list.concat(base_cfg_list.copy()
#   .override('cache_policy', ['nn'])
#   .override('per_layer_cache', [True, False])
#   .override('cache_trace_path', ['/nvme/songxiaoniu/moe/moe-traces/deepseek-moe-sharegpt-0412.json'])
#   .override('reorder_experts', [False, True])
#   .override('num_predict_expert_per_layer', [0])
#   .override('early_preempt', [False])
#   .override('predict_input_mode', ['one_token'])
#   .override('predictor_model_path', ['/nvme/songxiaoniu/moe/moe-predict-models/models--deepseek-ai--deepseek-moe-16b-chat.pt'])
#   .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]])
# )
# full_list.concat(base_cfg_list.copy()
#   .override('cache_policy', ['nn'])
#   .override('per_layer_cache', [True, False])
#   .override('cache_trace_path', ['/nvme/songxiaoniu/moe/moe-traces/deepseek-moe-sharegpt-0412.json'])
#   .override('reorder_experts', [False, True])
#   .override('num_predict_expert_per_layer', [0])
#   .override('early_preempt', [False])
#   .override('predict_input_mode', ['decode_cumsum'])
#   .override('predictor_model_path', ['/nvme/songxiaoniu/moe/moe-predict-models/models--deepseek-ai--deepseek-moe-16b-chat-next-reuse.pt'])
#   .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]])
# )
full_list.concat(base_cfg_list.copy()
  .override('cache_policy', ['nn'])
  .override('per_layer_cache', [True, False])
  .override('cache_trace_path', ['/nvme/songxiaoniu/moe/moe-traces/deepseek-moe-sharegpt-0412.json'])
  .override('reorder_experts', [False, True])
  .override('num_predict_expert_per_layer', [0])
  .override('early_preempt', [False])
  .override('predict_input_mode', ['weighted_decode_cumsum'])
  .override('predictor_model_path', ['/nvme/songxiaoniu/moe/moe-predict-models/models--deepseek-ai--deepseek-moe-16b-chat-next-10-freq-weighted-sum.pt'])
  .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]])
)
# full_list.concat(base_cfg_list.copy()
#   .override('cache_policy', ['lru', 'nn'])
#   .override('per_layer_cache', [True])
#   .override('reorder_experts', [True])
#   .override('num_predict_expert_per_layer', [prefetch_len for prefetch_len in [6]])
#   .override('early_preempt', [True])
#   .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]])
# )

if __name__ == '__main__':
  from eval_helper.runner_args import parse_args
  args = parse_args()
  if 'run' in args.commands:
    full_list.run(mock=args.mock, durable_log=args.durable_log, fail_only=args.fail_only, parallel_workers=args.parallel_workers)
  if 'parse' in args.commands:
    full_list.override('logdir', [args.logdir])
    full_list.parse()
    full_list.to_pdframe([
      'cache_policy',
      'per_layer_cache',
      'reorder_experts',
      'predict_input_mode',
      # 'early_preempt',
      # 'num_predict_expert_per_layer',
      'cache_rate',
      'decode_stage_hit_rate',
      'prefill_stage_hit_rate',
      'decode_stage_ready_rate',
      'prefill_stage_ready_rate',
      'legacy_decode_stage_hit_rate',
      'legacy_prefill_stage_hit_rate',
    ]).to_csv(args.parse_output, index=False)
