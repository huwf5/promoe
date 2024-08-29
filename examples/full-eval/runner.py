from eval_helper.config import RunConfigBase, OptionCmdLine, OptionEnv, OptionApp, ConfigList, ResultFloat
import os

my_app = RunConfigBase()
my_app.app = OptionApp('python3 transformers-app.py', False, False)
my_app.logdir = 'run-logs'
my_app.config_dict = {
  'model_id'                      : OptionCmdLine('model_id'),
  'model_revision'                : OptionCmdLine('model_revision'),
  'dataset'                       : OptionCmdLine('dataset'),
  'batch_size'                    : OptionCmdLine('batch_size'),
  'num_predict_expert_per_layer'  : OptionCmdLine('num_predict_expert_per_layer', readable_name='predict', logname='predict'),
  'cache_rate'                    : OptionCmdLine('cache_rate'),
  'cache_policy'                  : OptionCmdLine('cache_policy', readable_name='policy', logname='policy'),
  'per_layer_cache'               : OptionCmdLine('per_layer_cache', logname=False),
  'reorder_experts'               : OptionCmdLine('reorder_experts', logname='reorder'),
  'early_preempt'                 : OptionCmdLine('early_preempt', logname='early'),
  'predict_input_mode'            : OptionCmdLine('predict_input_mode', logname=False),
  'layer_predict_interval'        : OptionCmdLine('layer_predict_interval', logname='p_int'),
  'layer_predict_max_window'      : OptionCmdLine('layer_predict_max_window', logname='p_win'),
  'layer_predict_use_last_output' : OptionCmdLine('layer_predict_use_last_output', logname='p_last'),
  'predictor_model_path'          : OptionCmdLine('predictor_model_path', readable_name=False, logname=False),
  'log_level'                     : OptionEnv('SPARSE_CACHE_LOG_LEVEL', readable_name=False, logname=False),
}

my_app.result_dict = {
  'decode_stage_forward_time'  : ResultFloat('decode_stage_forward_time'),
  'prefill_stage_forward_time' : ResultFloat('prefill_stage_forward_time'),
  'decode_stage_hit_rate'      : ResultFloat('decode_stage_hit_rate'),
  'prefill_stage_hit_rate'     : ResultFloat('prefill_stage_hit_rate'),
  'decode_stage_ready_rate'    : ResultFloat('decode_stage_ready_rate'),
  'prefill_stage_ready_rate'   : ResultFloat('prefill_stage_ready_rate'),
}
my_app['per_layer_cache'] = True
base_cfg_list = ConfigList.MakeList(my_app)

full_list = ConfigList.Empty()

### options to control: prefetch, reorder, early_preempt

template_cfg_list = (base_cfg_list.copy()
  .override('cache_policy', ['lru',])
  .override('batch_size', [1])
  .override('per_layer_cache', [True])
  .override('predict_input_mode', ['moe_layer_logits'])
  .override('layer_predict_interval', [1])
  .override('layer_predict_max_window', [3])
  .override('layer_predict_use_last_output', [
    True,
    # False,
  ])
  .override('dataset', ['chatgpt-prompts-small'])
)

full_list.concat(template_cfg_list.copy()
  .override('model_id', ['deepseek-ai/deepseek-moe-16b-chat',])
  .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32]])
  .override('predictor_model_path', ['/code/moe/moe-predict-models/models--deepseek-ai--deepseek-moe-16b-chat/moe-layer-logits'])
  .hyper_override(['num_predict_expert_per_layer', 'reorder_experts', 'early_preempt'], [
    [0, False, False], ## weak baseline
    [6,  True,  True], ## +p+opt
  ])
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
      'model_id',
      'batch_size',
      'reorder_experts',
      'early_preempt',
      'num_predict_expert_per_layer',
      'cache_rate',
      'decode_stage_hit_rate',
      'prefill_stage_hit_rate',
      'decode_stage_ready_rate',
      'prefill_stage_ready_rate',
      'decode_stage_forward_time',
      'prefill_stage_forward_time',
    ]).to_csv(args.parse_output, index=False)
