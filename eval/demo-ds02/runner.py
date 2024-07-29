from eval_helper.config import RunConfigBase, OptionCmdLine, OptionEnv, OptionApp, ConfigList, ResultFloat
import os

my_app = RunConfigBase()
my_app.app = OptionApp('python transformers-app.py', False, False)
my_app.logdir = 'run-logs'
my_app.config_dict = {
  'model_id'                      : OptionCmdLine('model_id'),
  'num_predict_expert_per_layer'  : OptionCmdLine('num_predict_expert_per_layer', readable_name='predict', logname='predict'),
  'cache_rate'                    : OptionCmdLine('cache_rate'),
  'cache_policy'                  : OptionCmdLine('cache_policy', readable_name='policy', logname='policy'),
  'per_layer_cache'               : OptionCmdLine('per_layer_cache', logname=False),
  'reorder_experts'               : OptionCmdLine('reorder_experts', logname='reorder'),
  'early_preempt'                 : OptionCmdLine('early_preempt'),
  'max_prefetch_layer_distance'   : OptionCmdLine('max_prefetch_layer_distance'),
  'predict_input_mode'            : OptionCmdLine('predict_input_mode', logname=False),
  'layer_predict_interval'        : OptionCmdLine('layer_predict_interval'),
  'layer_predict_max_window'      : OptionCmdLine('layer_predict_max_window'),
  'layer_predict_use_last_output' : OptionCmdLine('layer_predict_use_last_output'),
  'predictor_model_path'          : OptionCmdLine('predictor_model_path', readable_name=False, logname=False),
  'log_level'                     : OptionEnv('SPARSE_CACHE_LOG_LEVEL', readable_name=False, logname=False),
  'physical_impl'                 : OptionEnv('SPARSE_CACHE_PHYSICAL_MEM_IMPL', readable_name='physical_impl', logname=False),
  'logical_impl'                  : OptionEnv('SPARSE_CACHE_LOGICAL_MEM_IMPL',  readable_name='logical_impl',  logname=False),
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

template_cfg_list = (base_cfg_list.copy()
  .override('cache_policy', [
    # 'nn',
    'lru',
  ])
  .override('per_layer_cache', [True])
  .override('reorder_experts', [True, False])
  .override('early_preempt',   [True, False])
  .override('predict_input_mode', ['moe_layer_logits'])
  .override('layer_predict_interval', [
    1,
    # 2,
  ])
  .override('layer_predict_max_window', [3])
  .override('layer_predict_use_last_output', [
    True,
    False,
  ])
)

full_list.concat(template_cfg_list.copy()
  .override('model_id', ['deepseek-ai/deepseek-moe-16b-chat',])
  # .override('num_predict_expert_per_layer', [0,6,8,])
  .override('num_predict_expert_per_layer', [0,6,])
  .override('cache_rate', [cache_item/64 for cache_item in [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]])
  .override('predictor_model_path', ['/nvme/sxn/moe/moe-predict-models/models--deepseek-ai--deepseek-moe-16b-chat/moe-layer-logits'])
)
full_list.concat(template_cfg_list.copy()
  .override('model_id', ['Qwen/Qwen1.5-MoE-A2.7B-Chat',])
  # .override('num_predict_expert_per_layer', [0,4,6,])
  .override('num_predict_expert_per_layer', [0,4,])
  .override('cache_rate', [cache_item/60 for cache_item in [1, 2, 4, 8, 12, 16, 24, 30, 36, 42]])
  .override('predictor_model_path', ['/nvme/sxn/moe/moe-predict-models/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/moe-layer-logits'])
)
full_list.concat(template_cfg_list.copy()
  .override('model_id', ['TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ',])
  # .override('num_predict_expert_per_layer', [0,2,3,])
  .override('num_predict_expert_per_layer', [0,2,])
  .override('cache_rate', [cache_item/8 for cache_item in [1, 2, 3, 4, 5, 6]])
  .override('predictor_model_path', ['/nvme/sxn/moe/moe-predict-models/models--mistralai--Mixtral-8x7B-Instruct-v0.1/moe-layer-logits'])
  .override('physical_impl', ['cudriver_unified'])
  .override('logical_impl', ['cudriver_unified'])
)

if __name__ == '__main__':
  from eval_helper.runner_args import parse_args
  args = parse_args()
  # def call_back_fn(cfg : RunConfigBase):
  #   if cfg['trace_event']:
  #     os.system(f'mv trace.json {cfg.get_log_fname()}.json')
  if 'run' in args.commands:
    full_list.run(mock=args.mock, durable_log=args.durable_log, fail_only=args.fail_only)
    # full_list.run(mock=args.mock, durable_log=args.durable_log, fail_only=args.fail_only, callback=call_back_fn)
  if 'parse' in args.commands:
    full_list.override('logdir', [args.logdir])
    full_list.parse()
    full_list.to_pdframe([
      'model_id',
      # 'per_layer_cache',
      'reorder_experts',
      'early_preempt',
      'num_predict_expert_per_layer',
      'layer_predict_interval',
      'cache_rate',
      # 'layer_predict_use_last_output',
      # 'cache_policy',
      'decode_stage_hit_rate',
      'prefill_stage_hit_rate',
      'decode_stage_forward_time',
      'prefill_stage_forward_time',
    ]).to_csv(args.parse_output, index=False)
