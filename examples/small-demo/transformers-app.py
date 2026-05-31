import contextlib
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['HUGGINGFACE_OFFLINE'] = "1"

from transformers.utils import logging
from transformers.generation.utils import TimeProfiler, recursive_attach
import torch
torch.cuda.set_device(0)
from transformers import AutoModelForCausalLM, AutoTokenizer, SwitchTransformersForConditionalGeneration

import sparse_llm_cache
import time
from dataclasses import dataclass, field

from sparse_llm_cache.utils.runner_util import parse_args
cache_configs = parse_args()
gpu_mem_limit_gb = cache_configs.pop('gpu_mem_limit_gb', None)
for k, v in cache_configs.items(): print(k,v)


@contextlib.contextmanager
def _nvtx_range(name: str):
  """Nsight Systems: mark a span on the NVTX row (no-op if no CUDA)."""
  if not torch.cuda.is_available():
    yield
    return
  torch.cuda.nvtx.range_push(name)
  try:
    yield
  finally:
    torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def _nvtx_benchmark_batch(batch_idx: int, warmup_samples: int):
  """Align with PROMOE_BENCHMARK_WARMUP / _ExplicitGenTimingAgg: warmup vs steady batches."""
  phase = "warmup" if batch_idx < max(warmup_samples, 0) else "steady"
  with _nvtx_range(f"promoe/benchmark/{phase}/batch_{batch_idx}"):
    yield


sparse_llm_cache.utils.hack_transformers(**cache_configs, pin_memory=True, enable_model_timer=True)

if torch.cuda.is_available():
  if gpu_mem_limit_gb is not None:
    if gpu_mem_limit_gb <= 0:
      raise ValueError(f"gpu_mem_limit_gb should be > 0, got {gpu_mem_limit_gb}")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_mem_fraction = min(gpu_mem_limit_gb / total_mem_gb, 1.0)
    torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=0)
    print(f"set torch per-process memory fraction to {gpu_mem_fraction:.6f} on cuda:0 (limit {gpu_mem_limit_gb}GB / total {total_mem_gb:.2f}GB)")

print("loading model...")
load_time_start = time.time()
logging.disable_progress_bar()
model_id = cache_configs['model_id']

def is_switch_model_id(value: str) -> bool:
  return "switch-" in value.lower() or value.lower().endswith("switch-base-128")

def resolve_switch_model_path(value: str) -> str:
  if not value.startswith("google/"):
    return value
  repo_root = Path(__file__).resolve().parents[2]
  local_path = repo_root / "deps" / "sparse-llm-cache-scripts" / "huggingface-modules" / "modules" / "transformers_modules" / "google" / value.split("/", 1)[1]
  if local_path.exists():
    return str(local_path)
  return value

model_cls = SwitchTransformersForConditionalGeneration if is_switch_model_id(model_id) else AutoModelForCausalLM
model_load_id = resolve_switch_model_path(model_id) if is_switch_model_id(model_id) else model_id
torch_dtype = 'auto'
if 'GPTQ' in model_id:
  torch_dtype = None
print("dtype is", torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_load_id, trust_remote_code=True)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
with _nvtx_range("promoe/model_load"):
  model = model_cls.from_pretrained(
    model_load_id,
    # torch_dtype=torch.float16,
    torch_dtype=torch_dtype,
    local_files_only=True,
    device_map=0,
    trust_remote_code=True,
    revision=cache_configs['model_revision'],
  )
if getattr(model.config, "is_encoder_decoder", False) and model.config.decoder_start_token_id is None:
  model.config.decoder_start_token_id = tokenizer.pad_token_id
  model.generation_config.decoder_start_token_id = tokenizer.pad_token_id
load_model_time = time.time() - load_time_start
print("loading model...done", load_model_time)

time_profiler = TimeProfiler()
recursive_attach(model, time_profiler, '_time_profiler')

def maybe_log_time_profiler(time_profiler):
  if (
    getattr(time_profiler, "num_prefill_iters", 0) > 0
    and getattr(time_profiler, "num_decode_iters", 0) > 0
    and getattr(time_profiler, "prefill_time", 0) > 0
    and getattr(time_profiler, "decode_time", 0) > 0
  ):
    time_profiler.log()
    print(
      "(HF TimeProfiler summary above: trans_ttft / trans_tpot follow Transformers definitions; "
      "see explicit_cache_init_ms plus explicit_ttft_ms / explicit_decode_tpot_excl_first_ms below for ours.)",
      flush=True,
    )
  else:
    print("time_profiler.log skipped: insufficient profiling samples")


@dataclass
class _ExplicitGenTimingAgg:
  """Aggregate over gen_batch calls (batch_size=1 recommended)."""
  warmup_samples: int = int(os.environ.get("PROMOE_BENCHMARK_WARMUP", "4"))
  samples: list[tuple[float, float, int, int, float | None]] = field(default_factory=list)
  cache_init_samples: list[float] = field(default_factory=list)

  def add(
      self,
      ttft_s: float | None,
      tpot_excl_first_s: float | None,
      output_token_count: int,
      cache_init_s: float | None = None,
  ):
    if ttft_s is None:
      return
    decode_token_count = max(output_token_count - 1, 0)
    decode_s = 0.0
    if tpot_excl_first_s is not None:
      decode_s = tpot_excl_first_s * decode_token_count
    self.samples.append((ttft_s, decode_s, output_token_count, decode_token_count, tpot_excl_first_s))
    if cache_init_s is not None:
      self.cache_init_samples.append(cache_init_s)

  def _benchmark_samples(self):
    skip = min(max(self.warmup_samples, 0), len(self.samples))
    return skip, self.samples[skip:]

  def report(self):
    skip, samples = self._benchmark_samples()
    cache_init_samples = self.cache_init_samples[skip:]
    print(f"benchmark_warmup_skipped:{skip}", flush=True)
    print(f"benchmark_samples:{len(samples)}", flush=True)
    if len(samples) == 0:
      print("benchmark_cache_init_ms_avg: n/a (no samples)", flush=True)
      print("benchmark_ttft_ms_avg: n/a (no samples)", flush=True)
      print("benchmark_decode_tokens_per_second_excl_first: n/a (no samples)", flush=True)
      print("benchmark_e2e_tokens_per_second: n/a (no samples)", flush=True)
      return

    sum_ttft_s = sum(sample[0] for sample in samples)
    sum_decode_s = sum(sample[1] for sample in samples)
    sum_generate_tokens = sum(sample[2] for sample in samples)
    sum_decode_tokens = sum(sample[3] for sample in samples)
    valid_tpots = [sample[4] for sample in samples if sample[4] is not None]

    print(f"benchmark_generate_tokens:{sum_generate_tokens}", flush=True)
    print(f"benchmark_decode_tokens_excl_first:{sum_decode_tokens}", flush=True)
    if len(cache_init_samples) == 0:
      print("benchmark_cache_init_ms_avg: n/a (disabled)", flush=True)
    else:
      print(f"benchmark_cache_init_ms_avg:{1000.0 * sum(cache_init_samples) / len(cache_init_samples):.6f}", flush=True)
    print(f"benchmark_ttft_ms_avg:{1000.0 * sum_ttft_s / len(samples):.6f}", flush=True)
    if len(valid_tpots) == 0 or sum_decode_s <= 0.0:
      print("benchmark_decode_tokens_per_second_excl_first: n/a (no decode samples)", flush=True)
    else:
      print(f"benchmark_decode_tpot_ms_avg:{1000.0 * sum(valid_tpots) / len(valid_tpots):.6f}", flush=True)
      print(f"benchmark_decode_tokens_per_second_excl_first:{sum_decode_tokens / sum_decode_s:.6f}", flush=True)
    total_generate_s = sum_ttft_s + sum_decode_s
    if total_generate_s <= 0.0:
      print("benchmark_e2e_tokens_per_second: n/a (zero elapsed time)", flush=True)
    else:
      print(f"benchmark_e2e_tokens_per_second:{sum_generate_tokens / total_generate_s:.6f}", flush=True)


class _ForwardEndCollector:
  """
  Per generate(): record wall time after each root model forward (CUDA-synced).
  - TTFT: time from generate() start until end of the first forward (first generated token path).
  - Decode TPOT excluding first token: mean of (t[i]-t[i-1]) for i>=1, i.e. (t_last-t_first)/(N-1)
    when N>=2 forwards were observed (steady decode steps for tokens 2..N).
  """

  def __init__(self, root_module: torch.nn.Module):
    self._root = root_module
    self._handles = []
    self._active = False
    self._t_start: float | None = None
    self._ends: list[float] = []

  def _sync_time(self) -> float:
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    return time.perf_counter()

  def _post_hook(self, module, args, kwargs, output):
    if not self._active:
      return
    self._ends.append(self._sync_time())

  def attach(self):
    if self._handles:
      return
    try:
      self._handles.append(self._root.register_forward_hook(self._post_hook, with_kwargs=True))
    except TypeError:
      self._handles.append(self._root.register_forward_hook(lambda mod, inp, out: self._post_hook(mod, inp, None, out)))

  def remove(self):
    for handle in self._handles:
      handle.remove()
    self._handles.clear()

  def begin_generate(self):
    self._ends.clear()
    self._active = True
    self._t_start = self._sync_time()

  def end_generate(self):
    self._active = False

  def compute(self, output_token_count: int) -> tuple[float | None, float | None]:
    """
    output_token_count: number of generated (new) tokens for this batch.
    Returns (ttft_s, tpot_decode_excl_first_s).

    TTFT must use the *first* root forward after generate() starts — never slice ends from
    the tail; taking ends[-N:] made ttft look like mid-decode latency (~hundreds of ms).
    """
    if self._t_start is None or len(self._ends) == 0:
      return None, None
    raw = self._ends
    # First forward completion ≈ first-token path (encoder+decoder step 1 for Switch).
    ttft_s = raw[0] - self._t_start
    # Assume one root forward per new token; ignore excess trailing forwards if any.
    if output_token_count < 2:
      return ttft_s, None
    n = min(len(raw), output_token_count)
    aligned = raw[:n]
    if len(aligned) < 2:
      return ttft_s, None
    # Mean wall time between end of forward i and end of forward i+1 for i=0..n-2:
    # equals (t_last - t_first) / (n-1), excludes the gap before raw[0] (TTFT).
    tpot_excl_first_s = (aligned[-1] - aligned[0]) / (len(aligned) - 1)
    return ttft_s, tpot_excl_first_s


_explicit_timing = _ForwardEndCollector(model)
_explicit_timing.attach()
_explicit_agg = _ExplicitGenTimingAgg()


def gen_batch(text_list, do_print=False, max_new_tokens=100):
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  input_start = time.perf_counter()
  with _nvtx_range("promoe/gen/tokenize_to_cuda"):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True).to(f"cuda") # input_ids, attention_mask
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  input_s = time.perf_counter() - input_start
  input_len = inputs['input_ids'].shape[1]
  cache_init_s = None
  generate_fn = model.generate
  if hasattr(model, "_sparse_cache_old_generate") and hasattr(model, "_prefetch_mngr"):
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    cache_init_start = time.perf_counter()
    with _nvtx_range("promoe/gen/cache_init/reset_and_load_initial_cache"):
      model._prefetch_mngr.reset_and_load_initial_cache()
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    cache_init_s = time.perf_counter() - cache_init_start
    generate_fn = model._sparse_cache_old_generate
  _explicit_timing.begin_generate()
  try:
    with _nvtx_range("promoe/gen/generate_forward_loop"):
      outputs = generate_fn(**inputs, max_new_tokens=max_new_tokens)
  finally:
    _explicit_timing.end_generate()
  if getattr(model.config, "is_encoder_decoder", False):
    generated_tokens = outputs
  else:
    generated_tokens = outputs[:, input_len:]
  output_len = generated_tokens.shape[1]
  ttft_s, tpot_excl_s = _explicit_timing.compute(output_len)
  _explicit_agg.add(ttft_s, tpot_excl_s, output_len, cache_init_s)
  if do_print:
    n_steps = len(_explicit_timing._ends)
    cache_init_ms = 1000.0 * cache_init_s if cache_init_s is not None else float("nan")
    ttft_ms = 1000.0 * ttft_s if ttft_s is not None else float("nan")
    tpot_ms = 1000.0 * tpot_excl_s if tpot_excl_s is not None else float("nan")
    print(
      f"explicit_input_ms:{1000.0 * input_s:.6f} explicit_cache_init_ms:{cache_init_ms:.6f} "
      f"explicit_ttft_ms:{ttft_ms:.6f} explicit_decode_tpot_excl_first_ms:{tpot_ms:.6f} "
      f"(gen_forward_steps:{n_steps} new_tokens:{output_len})",
      flush=True,
    )
  output_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  if do_print:
    print(text_list, output_str, flush=True)
  return input_len, output_len

# dataset_path = f'/code/sparse-llm-cache-scripts/dataset/{cache_configs["dataset"]}/prompt_list.pt'
dataset_path = f'/mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/dataset/mmlu/professional_law/{cache_configs["dataset"]}/prompt_list.pt'
print(dataset_path)
prompts = torch.load(dataset_path)
original_prompt_indices = list(range(len(prompts)))
sample_indices_env = os.environ.get("PROMOE_SAMPLE_INDICES", "").strip()

def _parse_sample_indices(spec: str) -> list[int]:
  result = []
  for raw_item in spec.split(","):
    item = raw_item.strip()
    if not item:
      continue
    if "-" in item:
      start_s, stop_s = item.split("-", 1)
      start = int(start_s.strip())
      stop = int(stop_s.strip())
      if stop < start:
        raise ValueError(f"invalid descending PROMOE_SAMPLE_INDICES range: {item}")
      result.extend(range(start, stop + 1))
    else:
      result.append(int(item))
  return result

if sample_indices_env:
  sample_indices = _parse_sample_indices(sample_indices_env)
  invalid_indices = [idx for idx in sample_indices if idx < 0 or idx >= len(prompts)]
  if invalid_indices:
    raise ValueError(
      f"PROMOE_SAMPLE_INDICES contains out-of-range indices {invalid_indices}; "
      f"dataset has {len(prompts)} prompts"
    )
  prompts = [prompts[idx] for idx in sample_indices]
  original_prompt_indices = sample_indices
  cache_configs["max_num_batch"] = min(cache_configs["max_num_batch"], len(prompts))
  print(f"PROMOE_SAMPLE_INDICES={sample_indices}", flush=True)
  print(f"selected_prompt_count={len(prompts)}", flush=True)

from torch.utils.data import Dataset
class StringListDataset(Dataset):
  def __init__(self, string_list):
    self.string_list = string_list
  def __len__(self):
    return len(self.string_list)
  def __getitem__(self, idx):
    return self.string_list[idx]
ds = StringListDataset(prompts)
dl = torch.utils.data.DataLoader(ds, batch_size=cache_configs['batch_size'], shuffle=False)

print(
  f"nvtx: batch i in [0..warmup-1] -> promoe/benchmark/warmup/batch_i, else steady "
  f"(PROMOE_BENCHMARK_WARMUP={_explicit_agg.warmup_samples})",
  flush=True,
)
eval_time_start = time.time()
with _nvtx_range("promoe/eval_all_batches"):
  for seq_id,text_list in enumerate(dl):
    if seq_id >= cache_configs['max_num_batch']:
      print("max_num_batch reached")
      break
    batch_start = seq_id * cache_configs['batch_size']
    batch_stop = min(batch_start + cache_configs['batch_size'], len(original_prompt_indices))
    original_seq_ids = original_prompt_indices[batch_start:batch_stop]
    print(f'Seq {seq_id}/{cache_configs["max_num_batch"]}, original_seq_ids={original_seq_ids}, decoding...', flush=True)
    with _nvtx_benchmark_batch(seq_id, _explicit_agg.warmup_samples):
      input_len, output_len = gen_batch(text_list, max_new_tokens=cache_configs['max_new_tokens'], do_print=True)
eval_time = time.time() - eval_time_start

maybe_log_time_profiler(time_profiler)
_explicit_agg.report()
sparse_llm_cache.cpp_worker.log_gpu_mem_info()
print("load_model_time:", load_model_time)
print("eval_time:", eval_time)
