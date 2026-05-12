# Deterministic Initial Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each `generate` start from a reproducible global cache state by resetting runtime/cache state and synchronously loading a manual initial expert plan.

**Architecture:** Keep the change narrow: Python wraps `model.generate`, C++ `PrefetchMngr` exposes one reset/load entrypoint, `ModuleMeta` carries stage metadata and manual initial plan config, and `CacheMngr` owns cache clearing plus synchronous plan loading. Existing demand/prefetch correctness remains governed by `ExpertStatus` and `CacheMngr::miss`.

**Tech Stack:** Python, C++17, PyBind11, CUDA/PyTorch C++ extension, existing `src/sparse_llm_cache` injection and `src/cpp_worker` cache/prefetch runtime.

---

## File Structure

- Modify `src/cpp_worker/utils.hpp` and `src/cpp_worker/utils.cpp`
  - Add stage metadata fields and helpers.
  - Add deterministic initial cache config fields and parsing.
- Modify `src/sparse_llm_cache/model_adapters/base.py` and `src/sparse_llm_cache/model_adapters/switch.py`
  - Populate stage metadata defaults and Switch-specific encoder/decoder counts.
- Modify `src/sparse_llm_cache/utils/__init__.py`
  - Add user-facing config parameters.
  - Pass config into `ModuleMeta`.
  - Wrap `model.generate` when reset-on-generate is enabled.
- Modify `src/cpp_worker/cache.hpp` and `src/cpp_worker/cache.cpp`
  - Add cache reset helpers.
  - Add manual initial plan parser/loader support.
- Modify `src/cpp_worker/prefetcher.hpp` and `src/cpp_worker/prefetcher.cpp`
  - Add `reset_and_load_initial_cache()` entrypoint.
  - Coordinate scheduler queue reset and cache reset/load.
- Modify `src/cpp_worker/adapter.cpp`
  - Expose the new `PrefetchMngr` entrypoint to Python.
- Add tests in `tests/test_deterministic_initial_cache.py`
  - Cover Python config plumbing and manual plan validation where possible without a CUDA model.

---

### Task 1: Add ModuleMeta Config And Stage Metadata

**Files:**
- Modify: `src/cpp_worker/utils.hpp`
- Modify: `src/cpp_worker/utils.cpp`
- Modify: `src/sparse_llm_cache/model_adapters/base.py`
- Modify: `src/sparse_llm_cache/model_adapters/switch.py`

- [ ] **Step 1: Add fields and helpers to `ModuleMeta`**

In `src/cpp_worker/utils.hpp`, add public fields near existing config fields:

```cpp
int num_encoder_moe_layer = 0;
int num_decoder_moe_layer = -1;

bool reset_cache_on_generate_start = false;
std::string initial_cache_policy = "";
std::string initial_layer_budgets = "";
std::string initial_expert_order = "sequential";
bool initial_cache_ready_barrier = true;
```

Add helper declarations:

```cpp
int first_decoder_layer() const {
  return num_encoder_moe_layer;
}
bool is_encoder_layer(int layer_idx) const {
  return layer_idx >= 0 && layer_idx < num_encoder_moe_layer;
}
bool is_decoder_layer(int layer_idx) const {
  return layer_idx >= first_decoder_layer() && layer_idx < num_layer;
}
```

- [ ] **Step 2: Parse new config keys**

In `ModuleMeta::init_from_map()` in `src/cpp_worker/utils.cpp`, parse:

```cpp
num_encoder_moe_layer = optional_int("num_encoder_moe_layer", num_encoder_moe_layer);
num_decoder_moe_layer = optional_int("num_decoder_moe_layer", num_decoder_moe_layer);
reset_cache_on_generate_start = optional_bool("reset_cache_on_generate_start", reset_cache_on_generate_start);
initial_cache_policy = optional_str("initial_cache_policy", initial_cache_policy);
initial_layer_budgets = optional_str("initial_layer_budgets", initial_layer_budgets);
initial_expert_order = optional_str("initial_expert_order", initial_expert_order);
initial_cache_ready_barrier = optional_bool("initial_cache_ready_barrier", initial_cache_ready_barrier);
```

Place these before the `config_map.size() > 0` unrecognized-config check.

- [ ] **Step 3: Fill default decoder count and validate stage counts**

In `ModuleMeta::handle_uninited_configs()`, add:

```cpp
if (num_decoder_moe_layer == -1) {
  num_decoder_moe_layer = num_layer - num_encoder_moe_layer;
}
CHECK(num_encoder_moe_layer >= 0);
CHECK(num_decoder_moe_layer >= 0);
CHECK(num_encoder_moe_layer + num_decoder_moe_layer == num_layer)
    << "invalid stage split: encoder=" << num_encoder_moe_layer
    << ", decoder=" << num_decoder_moe_layer
    << ", num_layer=" << num_layer;
if (reset_cache_on_generate_start || initial_cache_policy != "" || initial_layer_budgets != "") {
  CHECK(per_layer_cache == false)
      << "deterministic initial cache requires per_layer_cache=false";
  CHECK(initial_cache_policy == "manual")
      << "only initial_cache_policy=manual is supported";
  CHECK(initial_expert_order == "sequential")
      << "only initial_expert_order=sequential is supported";
}
```

- [ ] **Step 4: Log the new config**

In `ModuleMeta::log_configs()`, add:

```cpp
LOG_CONFIG(num_encoder_moe_layer);
LOG_CONFIG(num_decoder_moe_layer);
LOG_CONFIG_BOOL(reset_cache_on_generate_start);
LOG_CONFIG(initial_cache_policy);
LOG_CONFIG(initial_layer_budgets);
LOG_CONFIG(initial_expert_order);
LOG_CONFIG_BOOL(initial_cache_ready_barrier);
```

- [ ] **Step 5: Populate defaults in adapter base**

In `src/sparse_llm_cache/model_adapters/base.py`, update `ModelAdapter.configure_module_meta()`:

```python
def configure_module_meta(self, meta) -> None:
  meta.num_encoder_moe_layer = 0
  meta.num_decoder_moe_layer = self.num_moe_layer
```

- [ ] **Step 6: Populate Switch stage counts**

In `SwitchAdapter.configure_module_meta()` in `src/sparse_llm_cache/model_adapters/switch.py`, preserve existing predictor fields and add:

```python
meta.num_encoder_moe_layer = self.num_encoder_sparse_layers
meta.num_decoder_moe_layer = self.num_decoder_sparse_layers
```

- [ ] **Step 7: Build check**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build succeeds, or fails only on later tasks referencing not-yet-added methods if tasks are applied out of order.

---

### Task 2: Add Python Config Plumbing And Generate Wrapper

**Files:**
- Modify: `src/sparse_llm_cache/utils/__init__.py`
- Modify: `src/cpp_worker/adapter.cpp`

- [ ] **Step 1: Add `inject_model()` parameters**

In `inject_model()` add parameters with defaults:

```python
    reset_cache_on_generate_start: bool = False,
    initial_cache_policy: str | None = None,
    initial_layer_budgets: str | None = None,
    initial_expert_order: str | None = "sequential",
    initial_cache_ready_barrier: bool = True,
```

- [ ] **Step 2: Pass config into `param_dict`**

Add keys:

```python
    'num_encoder_moe_layer'           : str(getattr(meta, 'num_encoder_moe_layer', 0)),
    'num_decoder_moe_layer'           : str(getattr(meta, 'num_decoder_moe_layer', -1)),
    'reset_cache_on_generate_start'   : str(reset_cache_on_generate_start),
    'initial_cache_policy'            : str(initial_cache_policy),
    'initial_layer_budgets'           : str(initial_layer_budgets),
    'initial_expert_order'            : str(initial_expert_order),
    'initial_cache_ready_barrier'     : str(initial_cache_ready_barrier),
```

Then call `adapter.configure_module_meta(meta)` before `meta.init_from_map(param_dict)`, so adapter-populated stage counts are available to the dict.

- [ ] **Step 3: Preserve adapter predictor configuration**

Because `adapter.configure_module_meta(meta)` currently runs after `meta.init_from_map(param_dict)`, moving it earlier changes ordering. Keep behavior by calling it once before dict construction for stage defaults, then again after `meta.init_from_map(param_dict)` only for adapter-enforced predictor fields if needed. The simpler acceptable implementation is:

```python
adapter.configure_module_meta(meta)
param_dict = {
  ...
  'num_encoder_moe_layer': str(meta.num_encoder_moe_layer),
  'num_decoder_moe_layer': str(meta.num_decoder_moe_layer),
  ...
}
meta.init_from_map(param_dict)
adapter.configure_module_meta(meta)
meta.handle_uninited_configs()
```

This keeps Switch predictor fields enforced after user config parsing, matching current behavior.

- [ ] **Step 4: Add generate wrapper**

In `src/sparse_llm_cache/utils/__init__.py`, add:

```python
def wrap_generate_with_initial_cache(model, prefetch_mngr):
  if hasattr(model, "_sparse_cache_old_generate"):
    return
  model._sparse_cache_old_generate = model.generate

  def generate_with_initial_cache(*args, **kwargs):
    prefetch_mngr.reset_and_load_initial_cache()
    return model._sparse_cache_old_generate(*args, **kwargs)

  model.generate = generate_with_initial_cache
```

After `model._prefetch_mngr = prefetch_mngr`, add:

```python
if meta.reset_cache_on_generate_start:
  wrap_generate_with_initial_cache(model, prefetch_mngr)
```

- [ ] **Step 5: Expose C++ method in pybind**

In `src/cpp_worker/adapter.cpp`, add a binding on `PrefetchMngr`:

```cpp
.def("reset_and_load_initial_cache", &PrefetchMngr::reset_and_load_initial_cache)
```

- [ ] **Step 6: Python import smoke**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python - <<'PY'
from sparse_llm_cache.utils import wrap_generate_with_initial_cache
print("ok")
PY
```

Expected:

```text
ok
```

---

### Task 3: Implement Manual Initial Plan Parsing

**Files:**
- Modify: `src/cpp_worker/cache.hpp`
- Modify: `src/cpp_worker/cache.cpp`

- [ ] **Step 1: Add plan item type and method declarations**

In `src/cpp_worker/cache.hpp`, add inside `CacheMngr` public section:

```cpp
using InitialExpert = std::pair<int, int>;

std::vector<InitialExpert> build_manual_initial_plan() const;
void validate_initial_plan_size(const std::vector<InitialExpert>& plan) const;
```

- [ ] **Step 2: Implement parser**

In `src/cpp_worker/cache.cpp`, add:

```cpp
std::vector<CacheMngr::InitialExpert> CacheMngr::build_manual_initial_plan() const {
  CHECK(metas->initial_cache_policy == "manual");
  CHECK(metas->initial_expert_order == "sequential");
  CHECK(!metas->initial_layer_budgets.empty());

  std::vector<InitialExpert> plan;
  std::unordered_set<int> seen_layers;
  std::stringstream ss(metas->initial_layer_budgets);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto colon = item.find(':');
    CHECK(colon != std::string::npos) << "invalid initial_layer_budgets item: " << item;
    int layer = std::stoi(item.substr(0, colon));
    int budget = std::stoi(item.substr(colon + 1));
    CHECK(layer >= 0 && layer < metas->num_layer)
        << "initial layer out of range: " << layer << ", num_layer=" << metas->num_layer;
    CHECK(seen_layers.insert(layer).second)
        << "duplicate layer in initial_layer_budgets: " << layer;
    CHECK(budget >= 0 && budget <= metas->num_expert)
        << "invalid budget for layer " << layer << ": " << budget
        << ", num_expert=" << metas->num_expert;
    for (int expert = 0; expert < budget; expert++) {
      plan.push_back({layer, expert});
    }
  }
  validate_initial_plan_size(plan);
  return plan;
}
```

- [ ] **Step 3: Implement plan size validation**

In `src/cpp_worker/cache.cpp`, add:

```cpp
void CacheMngr::validate_initial_plan_size(const std::vector<InitialExpert>& plan) const {
  size_t expected = cache_len;
  size_t actual = plan.size();
  if (actual < expected) {
    CHECK(false) << "initial cache plan smaller than cache size: cache_size=" << expected
                 << ", plan_size=" << actual
                 << ", missing=" << (expected - actual)
                 << ", initial_layer_budgets=" << metas->initial_layer_budgets;
  }
  if (actual > expected) {
    CHECK(false) << "initial cache plan larger than cache size: cache_size=" << expected
                 << ", plan_size=" << actual
                 << ", overflow=" << (actual - expected)
                 << ", initial_layer_budgets=" << metas->initial_layer_budgets;
  }
}
```

- [ ] **Step 4: Unit-build check**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build succeeds after Task 1 and Task 2 declarations are present.

---

### Task 4: Add Cache Reset And Synchronous Initial Loading

**Files:**
- Modify: `src/cpp_worker/cache.hpp`
- Modify: `src/cpp_worker/cache.cpp`
- Modify: `src/cpp_worker/prefetcher.hpp`
- Modify: `src/cpp_worker/prefetcher.cpp`

- [ ] **Step 1: Retain all physical cache lines**

In `CacheSlot` in `src/cpp_worker/cache.hpp`, add:

```cpp
std::vector<ExpertMemHanlderBase*> all_mems;
```

In `CacheMngr::init_gpu_mem_buffer()` in `src/cpp_worker/cache.cpp`, whenever a `cache_line` is allocated and assigned to `unused_mems`, also push it into `all_mems`:

```cpp
cache_slot.all_mems.push_back(cache_line);
```

- [ ] **Step 2: Add cache reset/load declarations**

In `CacheMngr` public section:

```cpp
void reset_cache_contents();
void load_initial_plan_sync(cudaStream_t stream);
```

In `PrefetchMngr` public section:

```cpp
void reset_and_load_initial_cache();
```

- [ ] **Step 3: Implement cache reset**

In `src/cpp_worker/cache.cpp`, implement:

```cpp
void CacheMngr::reset_cache_contents() {
  for (auto &pair : prefetched_experts) {
    auto expert = pair.first;
    expert->gpu_data = nullptr;
    expert->num_ready = 0;
    expert->expert_status.exchange(kIdle);
  }
  prefetched_experts.clear();

  for (auto &slot : cache_slots->slots) {
    slot.unused_mems.clear();
    slot.unused_mems.reserve(slot.all_mems.size());
    for (auto *mem : slot.all_mems) {
      slot.unused_mems.push_back(mem);
    }
    slot.policy = policy_factory.create_policy(metas->cache_policy);
  }

  size_t num_cache_slot = cache_slots->slots.size();
  CHECK(num_cache_slot == 1) << "deterministic initial cache requires global cache";
  CHECK(cache_slots->slots[0].unused_mems.size() == cache_len)
      << "cache reset did not restore all cache lines: restored="
      << cache_slots->slots[0].unused_mems.size()
      << ", cache_len=" << cache_len;
}
```

- [ ] **Step 4: Implement synchronous initial loading**

In `src/cpp_worker/cache.cpp`, implement:

```cpp
void CacheMngr::load_initial_plan_sync(cudaStream_t stream) {
  auto plan = build_manual_initial_plan();
  for (auto [layer, expert_idx] : plan) {
    auto expert = model_loader->get_source(layer, expert_idx);
    CHECK(!is_in_cache(expert));
    auto waiter = miss(expert, false);
    waiter();
    expert->expert_status.transfer(kIdle, kFetching);
    for (int mem_buf_idx = 0; mem_buf_idx < metas->num_per_expert_param; mem_buf_idx++) {
      CUDA_CALL(cudaMemcpyAsync(
          expert->gpu_data->ptr(mem_buf_idx),
          expert->host_data->ptr(mem_buf_idx),
          expert->host_data->nbytes(mem_buf_idx),
          cudaMemcpyHostToDevice,
          stream));
    }
    expert->reference_to_model_param->unmap();
    expert->reference_to_model_param->map_to(expert->gpu_data, model_loader->mem_mngr_ctx.get());
    CUDA_CALL(cudaStreamSynchronize(stream));
    expert->num_ready = metas->num_per_expert_param;
    expert->expert_status.transfer(kFetching, kReady);
  }
}
```

- [ ] **Step 5: Implement PrefetchMngr entrypoint**

In `src/cpp_worker/prefetcher.cpp`, implement:

```cpp
void PrefetchMngr::reset_and_load_initial_cache() {
  if (!metas->reset_cache_on_generate_start) {
    return;
  }
  prefetch_generation += 1;
  fetch_schedule_thread->generation_start_task.generation = prefetch_generation;
  auto handler = fetch_schedule_thread->add_one_task(&fetch_schedule_thread->generation_start_task);
  fetch_schedule_thread->wait_progress(handler);
  cache->reset_cache_contents();
  cache->load_initial_plan_sync((cudaStream_t)copy_stream);
}
```

- [ ] **Step 6: Build check**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build succeeds.

---

### Task 5: Add Focused Validation Tests

**Files:**
- Add: `tests/test_deterministic_initial_cache.py`

- [ ] **Step 1: Add Python tests for generate wrapping**

Create `tests/test_deterministic_initial_cache.py`:

```python
from sparse_llm_cache.utils import wrap_generate_with_initial_cache


class DummyPrefetchMngr:
  def __init__(self):
    self.calls = 0

  def reset_and_load_initial_cache(self):
    self.calls += 1


class DummyModel:
  def __init__(self):
    self.generate_calls = 0

  def generate(self, *args, **kwargs):
    self.generate_calls += 1
    return {"args": args, "kwargs": kwargs}


def test_generate_wrapper_resets_before_generate():
  model = DummyModel()
  mngr = DummyPrefetchMngr()

  wrap_generate_with_initial_cache(model, mngr)
  result = model.generate(1, x=2)

  assert mngr.calls == 1
  assert model.generate_calls == 1
  assert result == {"args": (1,), "kwargs": {"x": 2}}


def test_generate_wrapper_is_idempotent():
  model = DummyModel()
  mngr = DummyPrefetchMngr()

  wrap_generate_with_initial_cache(model, mngr)
  wrap_generate_with_initial_cache(model, mngr)
  model.generate()

  assert mngr.calls == 1
  assert model.generate_calls == 1
```

- [ ] **Step 2: Run Python tests**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m pytest tests/test_deterministic_initial_cache.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 3: Run existing Switch adapter tests**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m pytest tests/test_switch_adapter.py tests/test_switch_adapter_integration.py -q
```

Expected: all selected tests pass.

---

### Task 6: Manual Runtime Smoke

**Files:**
- No source edits.

- [ ] **Step 1: Build extension**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build succeeds.

- [ ] **Step 2: Run a small generate with deterministic init disabled**

Run:

```bash
cd /mnt/huwf5/promoe/examples/small-demo
CUDA_VISIBLE_DEVICES=0 /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python transformers-app.py \
  --model_id google/switch-base-128 \
  --dataset validation \
  --batch_size 1 \
  --num_predict_expert_per_layer 0 \
  --cache_rate 0.375 \
  --cache_policy lru \
  --per_layer_cache True \
  --reorder_experts False \
  --early_preempt False \
  --predict_input_mode moe_layer_logits \
  --layer_predict_interval 1 \
  --layer_predict_max_window 3 \
  --layer_predict_use_last_output True \
  --predictor_model_path /mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/moe-predict-models/switch-base-128-mmlu-professional_law-test-train-validation-val/decoder \
  --gpu_mem_limit_gb 24
```

Expected: behavior matches the pre-change path; no deterministic-init validation runs.

- [ ] **Step 3: Run a small generate with deterministic init enabled**

For Switch base with `cache_rate=0.375`, `num_layer=12`, and `num_expert=128`, expected cache size is `round(0.375 * 12 * 128) = 576`. Run:

```bash
cd /mnt/huwf5/promoe/examples/small-demo
CUDA_VISIBLE_DEVICES=0 /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python transformers-app.py \
  --model_id google/switch-base-128 \
  --dataset validation \
  --batch_size 1 \
  --num_predict_expert_per_layer 0 \
  --cache_rate 0.375 \
  --cache_policy lru \
  --per_layer_cache False \
  --reorder_experts False \
  --early_preempt False \
  --predict_input_mode moe_layer_logits \
  --layer_predict_interval 1 \
  --layer_predict_max_window 3 \
  --layer_predict_use_last_output True \
  --predictor_model_path /mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/moe-predict-models/switch-base-128-mmlu-professional_law-test-train-validation-val/decoder \
  --gpu_mem_limit_gb 24 \
  --reset_cache_on_generate_start True \
  --initial_cache_policy manual \
  --initial_layer_budgets 0:128,1:128,2:128,3:128,4:64 \
  --initial_expert_order sequential \
  --initial_cache_ready_barrier True
```

Expected: generate starts, initial plan loads synchronously, and no plan-size error is raised.

- [ ] **Step 4: Run a negative config**

Run the same command as Step 3 with:

```bash
--initial_layer_budgets 0:128,1:128,2:128,3:128,4:63
```

Expected: process fails with a message containing `cache_size`, `plan_size`, `missing`, and `initial_layer_budgets`.

---

## Self-Review

- Spec coverage: tasks cover generate reset/init, stage metadata, global cache/manual plan config, plan validation, Python wrapping, and compatibility defaults.
- Scope control: no SchedulerAware policy, no three-stage I/O, no hot expert file, no active urgent set.
- Type consistency: public names are `reset_and_load_initial_cache`, `reset_cache_contents`, `load_initial_plan_sync`, `initial_layer_budgets`, and `num_encoder_moe_layer`.
