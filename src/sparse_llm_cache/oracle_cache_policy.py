from dataclasses import dataclass
import heapq
import json

@dataclass
class Config:
  n_encoder_layer : int
  n_decoder_layer : int
  first_expert : int
  n_fc : int = 2
  expert_step : int = 4
  num_expert : int = 128
  def num_encoder_moe_layer(self):
    return self.n_encoder_layer // self.expert_step
  def num_decoder_moe_layer(self):
    return self.n_decoder_layer // self.expert_step
  def is_encoder_moe_layer(self, moe_layer_id):
    return moe_layer_id < self.num_encoder_moe_layer()
  def moe_layer_id_to_full_layer_id(self, moe_layer_id):
    if not self.is_encoder_moe_layer(moe_layer_id):
      moe_layer_id -= self.num_encoder_moe_layer()
    return moe_layer_id * self.expert_step + self.first_expert
  def iterate_decoder_moe_layer(self):
    return range(
      self.num_encoder_moe_layer(),
      self.num_encoder_moe_layer() + self.num_decoder_moe_layer()
    )

nllb_config = Config(n_decoder_layer=24, n_encoder_layer=24, n_fc=2, expert_step=4,first_expert=3,num_expert=128)

def meta_to_module_key(config: Config, moe_layer_id, e_id, fc):
  if config.is_encoder_moe_layer(moe_layer_id) :
    e_or_d_str = "encoder"
  else:
    e_or_d_str = "decoder"
  layer_id = config.moe_layer_id_to_full_layer_id(moe_layer_id)
  return f'model.{e_or_d_str}.layers.{layer_id}.ffn.experts.expert_{e_id}.fc{fc}.',

def record_one_seq_time_key(seq_id, key, time):
  print(seq_id, time, key)

def traverse_one_seq_time(j, seq_id, config: Config):
  seq_time = 0

  for encoder_moe_layer_id in range(config.num_encoder_moe_layer()):
    dedup_expert_in_cur_layer = [
      e for _, expert_list in j[seq_id][str(encoder_moe_layer_id)].items()
        for e in expert_list
      ]
    dedup_expert_in_cur_layer = list(set(dedup_expert_in_cur_layer))
    dedup_expert_in_cur_layer.sort()
    for eid in dedup_expert_in_cur_layer:
      for fc in range(config.n_fc):
        # time = (encoder_moe_layer_id, prompt_token_idx, eid, fc)
        key = meta_to_module_key(config, encoder_moe_layer_id, eid, fc)
        record_one_seq_time_key(seq_id, key, seq_time)
        seq_time += 1

  for rply_token_idx in range(get_rply_len(j, seq_id, config)):
    for decoder_moe_layer_id in config.iterate_decoder_moe_layer():
      for eid in j[seq_id][str(decoder_moe_layer_id)][str(rply_token_idx)]:
        for fc in range(config.n_fc):
          # time = (rply_token_idx, decoder_moe_layer_id, eid, fc)
          key = meta_to_module_key(config, decoder_moe_layer_id, eid, fc)
          record_one_seq_time_key(seq_id, key, seq_time)
          seq_time += 1



def get_rply_len(j, seq_id, config : Config):
  return len(j[seq_id][str(config.num_encoder_moe_layer())])

