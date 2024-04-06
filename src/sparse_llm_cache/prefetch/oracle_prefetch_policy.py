from bisect import bisect
from dataclasses import dataclass
import json
from math import inf
import os
import tqdm

from sortedcontainers import SortedList

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
  return f'model.{e_or_d_str}.layers.{layer_id}.ffn.experts.expert_{e_id}.fc{fc+1}.'

def record_one_seq_time_key(seq_id, key, time):
  print(seq_id, time, key)

def traverse_one_seq_time(j, seq_id, config: Config, recorder):
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
        recorder(seq_id, key, seq_time)
        seq_time += 1

  for rply_token_idx in range(get_rply_len(j, seq_id, config)):
    for decoder_moe_layer_id in config.iterate_decoder_moe_layer():
      for eid in j[seq_id][str(decoder_moe_layer_id)][str(rply_token_idx)]:
        for fc in range(config.n_fc):
          # time = (rply_token_idx, decoder_moe_layer_id, eid, fc)
          key = meta_to_module_key(config, decoder_moe_layer_id, eid, fc)
          recorder(seq_id, key, seq_time)
          seq_time += 1


def get_prompt_len(j, seq_id):
  prompt_len = len(j[seq_id]['0'])
  while len(j[seq_id]['0'][prompt_len - 1]) == 0:
    prompt_len -= 1
  return prompt_len

def get_rply_len(j, seq_id, config : Config):
  return len(j[seq_id][str(config.num_encoder_moe_layer())])

class OraclePolicy:
  @staticmethod
  def tqdm_wrapper(o):
    if 'DISABLE_MOE_CACHE_TQDM' in os.environ:
      return o
    else:
      return tqdm.tqdm(o)
  def __init__(self) -> None:

    self.next_use_time_map = {}
    self.cur_time = 0
    self.cur_seq_id = None

    # list of tuple(next_use_time, key)
    self.next_use_time_queue = SortedList()

  def set_cur_seq_id(self, seq_id):
    self.cur_seq_id = seq_id
    self.cur_time = -1
    self.next_use_time_queue.clear()
    for key in self.next_use_time_map:
      self.next_use_time_map[key] = self._find_next_use_time(key, -1)
      self.next_use_time_queue.add((self.next_use_time_map[key], key))

  def load_expert_trace(self, fname):
    with open(fname) as f:
      self.original_expert_history = json.load(f)
    if "nllb" in fname:
      self.config = nllb_config
    else:
      raise RuntimeError("Unimplemented")
    self.module_use_time = {}
    self.time_to_module = {}
    def recorder(seq_id, key, time):
      if seq_id not in self.module_use_time:
        self.module_use_time[seq_id] = {}
        self.time_to_module[seq_id] = []
      if key not in self.module_use_time[seq_id]:
        self.module_use_time[seq_id][key] = []
      self.module_use_time[seq_id][key].append(time)
      assert(len(self.time_to_module[seq_id]) == time)
      self.time_to_module[seq_id].append(key)

    for seq_id in self.tqdm_wrapper(self.original_expert_history):
      traverse_one_seq_time(self.original_expert_history, seq_id, self.config, recorder)

  def _choose_to_evict(self):
    item = self.next_use_time_queue[-1]
    return item[1]

  def _evict(self, key):
    assert(self._choose_to_evict() == key)
    self.next_use_time_queue.pop(-1)
    self.next_use_time_map.pop(key)

  def _find_next_use_time(self, key, cur_time):
    if key not in self.module_use_time[self.cur_seq_id]:
      return inf
    history_time = self.module_use_time[self.cur_seq_id][key]
    idx = bisect(history_time, cur_time)
    assert(idx == len(history_time) or history_time[idx] > cur_time)
    assert(idx == 0 or history_time[idx - 1] <= cur_time)
    if idx == len(history_time):
      return inf
    else:
      return history_time[idx]

  def _access(self, key):
    if key in self.next_use_time_map:
      # is in cache, hit
      # must be the nearest one
      assert(self.next_use_time_queue[0][0] == self.cur_time + 1)
      assert(self.next_use_time_queue[0][1] == key)

      self.cur_time = self.next_use_time_map[key]

      next_use_time = self._find_next_use_time(key, self.cur_time)

      self.next_use_time_map[key] = next_use_time
      self.next_use_time_queue.pop(0)
      self.next_use_time_queue.add((next_use_time, key))
    else:
      assert(self.cur_time + 1 == self._find_next_use_time(key, self.cur_time))
      self.cur_time = self._find_next_use_time(key, self.cur_time)
      next_use_time = self._find_next_use_time(key, self.cur_time)
      self.next_use_time_map[key] = next_use_time
      self.next_use_time_queue.add((next_use_time, key))

  def _predict_next(self):
    if len(self.time_to_module[self.cur_seq_id]) > self.cur_time + 1:
      return self.time_to_module[self.cur_seq_id][self.cur_time + 1]
    else:
      return None

  def clear(self):
    self.next_use_time_map.clear()
    self.next_use_time_queue.clear()