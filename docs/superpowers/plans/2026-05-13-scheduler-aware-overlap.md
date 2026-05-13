# Scheduler-Aware Eviction And Decoder Warmup Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scheduler-aware eviction and phase-aware prefetching so encoder cache space is reclaimed for decoder hot expert warmup before decoder predictor takes over.

**Architecture:** Extend the existing C++ runtime instead of replacing it. `CachePolicySchedulerAware` owns reclaimable/encoder/decoder victim choice, `CacheMngr` carries request type into miss handling, `FetchScheduleWorker` adds phase-aware task selection and a decoder warmup queue, and Python derives the warmup plan from the existing `initial_hot_expert_file`.

**Tech Stack:** Python 3.11, C++17, PyBind11, CUDA/PyTorch C++ extension, existing `src/sparse_llm_cache` injection and `src/cpp_worker` fetch scheduler.

---

## File Structure

- Modify `src/cpp_worker/cache.hpp` and `src/cpp_worker/cache.cpp`
  - Add cache request type.
  - Add `CachePolicySchedulerAware`.
  - Add policy signal helpers on `CacheMngr`.
  - Make `miss()` support skip for decoder warmup overlap.
- Modify `src/cpp_worker/worker.hpp`
  - Add request type to `CopyTask`.
- Modify `src/cpp_worker/prefetcher.hpp` and `src/cpp_worker/prefetcher.cpp`
  - Add scheduler phase.
  - Add decoder warmup queue.
  - Mark encoder experts reclaimable from existing hook events.
  - Add phase-aware normal prefetch selection and overlap submission.
- Modify `src/cpp_worker/utils.hpp` and `src/cpp_worker/utils.cpp`
  - Add config fields for decoder warmup overlap.
- Modify `src/sparse_llm_cache/utils/hot_experts.py`
  - Add decoder warmup plan builder using `initial_hot_expert_file`.
- Modify `src/sparse_llm_cache/utils/__init__.py`
  - Generate `decoder_warmup_expert_plan` from the same hot file.
  - Pass new config to `ModuleMeta`.
- Modify `src/sparse_llm_cache/utils/runner_util.py`
  - Add CLI flag for `enable_decoder_warmup_overlap`.
- Add or extend tests:
  - `tests/test_src_hot_expert_initial_cache.py`
  - `tests/test_deterministic_initial_cache.py`

---

### Task 1: Add Request Type And Scheduler-Aware Policy Skeleton

**Files:**
- Modify: `src/cpp_worker/cache.hpp`
- Modify: `src/cpp_worker/cache.cpp`
- Modify: `src/cpp_worker/worker.hpp`

- [ ] **Step 1: Add cache request type**

In `src/cpp_worker/cache.hpp`, near `CachePolicy`, add:

```cpp
enum CacheRequestType {
  kCacheRequestDemand = 0,
  kCacheRequestPrefetch,
  kCacheRequestDecoderWarmupOverlap,
};
```

- [ ] **Step 2: Extend `CopyTask`**

In `src/cpp_worker/worker.hpp`, add a request type field to `CopyTask`:

```cpp
CacheRequestType request_type = kCacheRequestPrefetch;
```

Update `CopyTask::toString()` to include it:

```cpp
ss << expert->toString() << ".[" << start_mem_buf_idx << "," << stop_mem_buf_idx
   << "), precise " << (is_precise ? "true" : "false")
   << ", gen " << generation
   << ", request_type " << request_type;
```

- [ ] **Step 3: Extend policy interface minimally**

In `CachePolicy`, add virtual methods:

```cpp
virtual ExpertHandler* select_for_evict(ExpertHandler* incoming, CacheRequestType request_type) {
  return select_for_evict(incoming);
}
virtual void mark_reclaimable(ExpertHandler* expert) {}
virtual bool has_reclaimable_encoder() const { return false; }
```

Keep the old `select_for_evict(ExpertHandler*)` so existing policies remain source-compatible.

- [ ] **Step 4: Declare `CachePolicySchedulerAware`**

In `cache.hpp`, add:

```cpp
class CachePolicySchedulerAware : public CachePolicy {
  using LL = DoubleLinkedList<ExpertHandler*>;
  std::vector<LL::Node*> node_free_buffer;
  LL global_lru;
  LL encoder_lru;
  LL decoder_lru;
  LL reclaimable_encoder_lru;
  std::unordered_map<ExpertHandler*, LL::Node*> global_map;
  std::unordered_map<ExpertHandler*, LL::Node*> encoder_map;
  std::unordered_map<ExpertHandler*, LL::Node*> decoder_map;
  std::unordered_map<ExpertHandler*, LL::Node*> reclaimable_map;

  LL::Node* new_node(ExpertHandler* expert);
  void recycle_node(LL::Node* node);
  void touch(std::unordered_map<ExpertHandler*, LL::Node*>& map, LL& list, ExpertHandler* expert);
  ExpertHandler* first_loaded_candidate(std::unordered_map<ExpertHandler*, LL::Node*>& map, LL& list);
 public:
  using CachePolicy::CachePolicy;
  ~CachePolicySchedulerAware();
  ExpertHandler* select_for_evict(ExpertHandler* incoming) override;
  ExpertHandler* select_for_evict(ExpertHandler* incoming, CacheRequestType request_type) override;
  void evict(ExpertHandler* expert) override;
  void access_on_hit(ExpertHandler* expert) override;
  void access_on_miss(ExpertHandler* expert) override;
  void mark_reclaimable(ExpertHandler* expert) override;
  bool has_reclaimable_encoder() const override;
  std::string toString() override;
};
```

- [ ] **Step 5: Register policy**

In `CacheMngr::CacheMngr()` in `cache.cpp`, register:

```cpp
policy_factory.register_policy("scheduler_aware", [this]() -> std::shared_ptr<CachePolicy>{
  return std::make_shared<CachePolicySchedulerAware>(this);
});
```

- [ ] **Step 6: Build check**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build fails only because methods are declared but not implemented. Include order remains `worker.hpp` including `cache.hpp` before `CopyTask` uses `CacheRequestType`.

---

### Task 2: Implement Scheduler-Aware Victim Selection

**Files:**
- Modify: `src/cpp_worker/cache.cpp`
- Modify: `src/cpp_worker/cache.hpp`

- [ ] **Step 1: Make `CachePolicy` able to inspect `CacheMngr`**

In `CachePolicy` in `cache.hpp`, change the `cache` member from private to protected:

```cpp
protected:
  CacheMngr* cache;
public:
  CachePolicy(CacheMngr* cache) : cache(cache) {}
```

- [ ] **Step 2: Add loaded check helper**

In `CacheMngr` public section, add:

```cpp
bool is_in_cache_ptr(ExpertHandler* expert) const {
  return prefetched_experts.find(expert) != prefetched_experts.end();
}
```

Keep existing `is_in_cache()` unchanged.

- [ ] **Step 3: Implement node helpers**

In `cache.cpp`, implement:

```cpp
CachePolicySchedulerAware::~CachePolicySchedulerAware() {
  while (!node_free_buffer.empty()) {
    delete node_free_buffer.back();
    node_free_buffer.pop_back();
  }
}

CachePolicySchedulerAware::LL::Node* CachePolicySchedulerAware::new_node(ExpertHandler* expert) {
  LL::Node* node = nullptr;
  if (node_free_buffer.empty()) {
    node = new LL::Node;
  } else {
    node = node_free_buffer.back();
    node_free_buffer.pop_back();
  }
  node->data = expert;
  return node;
}

void CachePolicySchedulerAware::recycle_node(LL::Node* node) {
  node_free_buffer.push_back(node);
}
```

- [ ] **Step 4: Implement touch**

```cpp
void CachePolicySchedulerAware::touch(
    std::unordered_map<ExpertHandler*, LL::Node*>& map,
    LL& list,
    ExpertHandler* expert) {
  auto it = map.find(expert);
  if (it != map.end()) {
    auto node = list.remove(it->second);
    list.push_back(node);
    return;
  }
  auto node = new_node(expert);
  map[expert] = node;
  list.push_back(node);
}
```

- [ ] **Step 5: Implement access and reclaimable updates**

```cpp
void CachePolicySchedulerAware::access_on_hit(ExpertHandler* expert) {
  touch(global_map, global_lru, expert);
  if (cache->metas->is_encoder_layer(expert->layer_idx)) {
    touch(encoder_map, encoder_lru, expert);
  } else if (cache->metas->is_decoder_layer(expert->layer_idx)) {
    touch(decoder_map, decoder_lru, expert);
  }
}

void CachePolicySchedulerAware::access_on_miss(ExpertHandler* expert) {
  access_on_hit(expert);
}

void CachePolicySchedulerAware::mark_reclaimable(ExpertHandler* expert) {
  if (!cache->metas->is_encoder_layer(expert->layer_idx)) {
    return;
  }
  if (!cache->is_in_cache_ptr(expert)) {
    return;
  }
  touch(reclaimable_map, reclaimable_encoder_lru, expert);
}
```

- [ ] **Step 6: Implement removal**

```cpp
void CachePolicySchedulerAware::evict(ExpertHandler* expert) {
  auto erase_from = [this, expert](std::unordered_map<ExpertHandler*, LL::Node*>& map, LL& list) {
    auto it = map.find(expert);
    if (it == map.end()) {
      return;
    }
    auto node = list.remove(it->second);
    recycle_node(node);
    map.erase(it);
  };
  erase_from(global_map, global_lru);
  erase_from(encoder_map, encoder_lru);
  erase_from(decoder_map, decoder_lru);
  erase_from(reclaimable_map, reclaimable_encoder_lru);
}
```

- [ ] **Step 7: Implement victim selection**

```cpp
ExpertHandler* CachePolicySchedulerAware::first_loaded_candidate(
    std::unordered_map<ExpertHandler*, LL::Node*>& map,
    LL& list) {
  for (auto node = list.front(); node != &list.guard_tail; node = node->next) {
    auto expert = node->data;
    if (map.find(expert) != map.end() && cache->is_in_cache_ptr(expert)) {
      return expert;
    }
  }
  return nullptr;
}

bool CachePolicySchedulerAware::has_reclaimable_encoder() const {
  for (auto& pair : reclaimable_map) {
    if (cache->is_in_cache_ptr(pair.first)) {
      return true;
    }
  }
  return false;
}

ExpertHandler* CachePolicySchedulerAware::select_for_evict(ExpertHandler* incoming) {
  return select_for_evict(incoming, kCacheRequestPrefetch);
}

ExpertHandler* CachePolicySchedulerAware::select_for_evict(
    ExpertHandler* incoming,
    CacheRequestType request_type) {
  if (auto victim = first_loaded_candidate(reclaimable_map, reclaimable_encoder_lru)) {
    return victim;
  }
  if (request_type == kCacheRequestDecoderWarmupOverlap) {
    return nullptr;
  }
  if (auto victim = first_loaded_candidate(encoder_map, encoder_lru)) {
    return victim;
  }
  if (auto victim = first_loaded_candidate(decoder_map, decoder_lru)) {
    return victim;
  }
  if (auto victim = first_loaded_candidate(global_map, global_lru)) {
    return victim;
  }
  CHECK(false) << "scheduler_aware policy could not choose victim for incoming "
               << incoming->toString();
}
```

- [ ] **Step 8: Implement `toString()`**

Use a compact count string:

```cpp
std::string CachePolicySchedulerAware::toString() {
  std::stringstream ss;
  ss << "scheduler_aware(global=" << global_map.size()
     << ",encoder=" << encoder_map.size()
     << ",decoder=" << decoder_map.size()
     << ",reclaimable=" << reclaimable_map.size()
     << ")";
  return ss.str();
}
```

- [ ] **Step 9: Build check**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build succeeds.

---

### Task 3: Carry Request Type Through Cache Miss

**Files:**
- Modify: `src/cpp_worker/cache.hpp`
- Modify: `src/cpp_worker/cache.cpp`
- Modify: `src/cpp_worker/prefetcher.cpp`

- [ ] **Step 1: Add overloads to `CacheMngr`**

In `cache.hpp`, add:

```cpp
CacheLineOccupancyWaiter miss(ExpertHandler* expert, bool is_precise, CacheRequestType request_type);
```

Keep the old method as a wrapper:

```cpp
CacheLineOccupancyWaiter miss(ExpertHandler* expert, bool is_precise) {
  return miss(expert, is_precise, is_precise ? kCacheRequestDemand : kCacheRequestPrefetch);
}
```

- [ ] **Step 2: Update implementation signature**

In `cache.cpp`, change:

```cpp
CacheMngr::CacheLineOccupancyWaiter CacheMngr::miss(ExpertHandler *incoming_e, bool is_precise)
```

to:

```cpp
CacheMngr::CacheLineOccupancyWaiter CacheMngr::miss(
    ExpertHandler *incoming_e,
    bool is_precise,
    CacheRequestType request_type)
```

- [ ] **Step 3: Pass request type to policy**

In the no-free-slot branch, replace:

```cpp
auto e_to_evict = cache_slot->policy->select_for_evict(incoming_e);
```

with:

```cpp
auto e_to_evict = cache_slot->policy->select_for_evict(incoming_e, request_type);
if (e_to_evict == nullptr) {
  CHECK(request_type == kCacheRequestDecoderWarmupOverlap)
      << "only decoder warmup overlap may skip eviction";
  return [](){};
}
```

- [ ] **Step 4: Avoid marking skipped warmup as cached**

Immediately after the call to `cache_miss` in `FetchScheduleWorker::send_one_job()`, check whether warmup skip left `gpu_data` null:

```cpp
lambda_wait = cache_miss(task->expert, task->is_precise, task->request_type);
if (task->request_type == kCacheRequestDecoderWarmupOverlap &&
    task->expert->gpu_data == nullptr) {
  return false;
}
```

To make this compile, change `FetchScheduleWorker::cache_miss()` signature in `prefetcher.hpp`:

```cpp
CacheMngr::CacheLineOccupancyWaiter cache_miss(
    ExpertHandler* e,
    bool is_precise,
    CacheRequestType request_type) {
  profiler->add(is_precise ? TimeProfiler::kMissCnt : TimeProfiler::kPrefetchMissCnt, 1);
  return cache->miss(e, is_precise, request_type);
}
```

Update existing calls to pass:

```cpp
task->request_type
```

- [ ] **Step 5: Set request type when adding tasks**

Change `add_single_tasks_for_one_expert` declaration and definition to accept:

```cpp
CacheRequestType request_type
```

Set:

```cpp
task.request_type = request_type;
```

For precise tasks pass `kCacheRequestDemand`; for normal prefetch pass `kCacheRequestPrefetch`; decoder warmup task creation in a later task will pass `kCacheRequestDecoderWarmupOverlap`.

- [ ] **Step 6: Build check**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build succeeds.

---

### Task 4: Mark Encoder Experts Reclaimable

**Files:**
- Modify: `src/cpp_worker/cache.hpp`
- Modify: `src/cpp_worker/cache.cpp`
- Modify: `src/cpp_worker/prefetcher.cpp`

- [ ] **Step 1: Add `CacheMngr` policy signal helpers**

In `cache.hpp`, add:

```cpp
void mark_reclaimable(int layer_idx, int expert_idx);
void mark_layer_reclaimable(int layer_idx);
void mark_layer_reclaimable_except(int layer_idx, const std::unordered_set<int>& needed_eids);
bool has_reclaimable_encoder() const;
```

- [ ] **Step 2: Implement helpers**

In `cache.cpp`, implement:

```cpp
void CacheMngr::mark_reclaimable(int layer_idx, int expert_idx) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  auto expert = model_loader->get_source(layer_idx, expert_idx);
  if (!is_in_cache(expert)) {
    return;
  }
  cache_slots->to_slot(expert)->policy->mark_reclaimable(expert);
}

void CacheMngr::mark_layer_reclaimable(int layer_idx) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  for (int expert_idx = 0; expert_idx < metas->num_expert; expert_idx++) {
    mark_reclaimable(layer_idx, expert_idx);
  }
}

void CacheMngr::mark_layer_reclaimable_except(
    int layer_idx,
    const std::unordered_set<int>& needed_eids) {
  if (!metas->is_encoder_layer(layer_idx)) {
    return;
  }
  for (int expert_idx = 0; expert_idx < metas->num_expert; expert_idx++) {
    if (needed_eids.find(expert_idx) == needed_eids.end()) {
      mark_reclaimable(layer_idx, expert_idx);
    }
  }
}

bool CacheMngr::has_reclaimable_encoder() const {
  for (auto &slot : cache_slots->slots) {
    if (slot.policy->has_reclaimable_encoder()) {
      return true;
    }
  }
  return false;
}
```

- [ ] **Step 3: Mark unused encoder experts after routing**

In `PrefetchMngr::report_one_layer()`, before `preempt_and_launch_one_layer(...)`, add:

```cpp
if (metas->is_encoder_layer(layer_id)) {
  std::unordered_set<int> needed;
  for (int64_t i = 0; i < num_expert; i++) {
    needed.insert(int(experts[i]));
  }
  cache->mark_layer_reclaimable_except(layer_id, needed);
}
```

- [ ] **Step 4: Mark executed encoder expert done**

In `PrefetchMngr::one_expert_done()`, after `mark_expert_using(layer_id, expert_id);`, add:

```cpp
if (metas->is_encoder_layer(layer_id)) {
  cache->mark_reclaimable(layer_id, expert_id);
}
```

- [ ] **Step 5: Mark whole encoder layer done**

In `PrefetchMngr::one_moe_layer_done()`, near the start, add:

```cpp
if (metas->is_encoder_layer(layer_id)) {
  cache->mark_layer_reclaimable(layer_id);
}
```

- [ ] **Step 6: Build check**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build succeeds.

---

### Task 5: Build Decoder Warmup Plan From Existing Hot Expert File

**Files:**
- Modify: `src/sparse_llm_cache/utils/hot_experts.py`
- Modify: `src/sparse_llm_cache/utils/__init__.py`
- Modify: `src/sparse_llm_cache/utils/runner_util.py`
- Modify: `src/cpp_worker/utils.hpp`
- Modify: `src/cpp_worker/utils.cpp`
- Modify: `src/cpp_worker/adapter.cpp`
- Test: `tests/test_src_hot_expert_initial_cache.py`

- [ ] **Step 1: Add Python tests for depth-interleave warmup plan**

In `tests/test_src_hot_expert_initial_cache.py`, add:

```python
def test_build_decoder_warmup_overlap_plan_excludes_initial_and_interleaves_depth(tmp_path):
  payload = {
    "expert_usage_summary": {
      "decoder.block.1.layer.2.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 1, "count": 100},
          {"eid": 2, "count": 90},
          {"eid": 3, "count": 80},
        ],
      },
      "decoder.block.3.layer.2.mlp.router.classifier": {
        "top_token_counts": [
          {"eid": 4, "count": 100},
          {"eid": 5, "count": 90},
        ],
      },
    }
  }
  path = tmp_path / "hot.json"
  path.write_text(json.dumps(payload))
  adapter = SwitchAdapter(SimpleNamespace(config=_switch_config()), "google/switch-base-128")

  plan = build_decoder_warmup_overlap_plan(
    path,
    adapter,
    initial_plan={(2, 1), (3, 4)},
  )

  assert plan == [(2, 2), (3, 5), (2, 3)]
```

Import `build_decoder_warmup_overlap_plan` at the top of the test file.

- [ ] **Step 2: Implement builder**

In `src/sparse_llm_cache/utils/hot_experts.py`, add:

```python
def build_decoder_warmup_overlap_plan(
    hot_expert_file: str | Path,
    adapter,
    *,
    initial_plan: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
  payload = _read_hot_expert_payload(hot_expert_file)
  decoder_by_layer, _token_totals = _hot_pairs_by_layer(payload, adapter, "decoder")
  initial = set(initial_plan or set())
  ordered_by_layer: dict[int, list[int]] = {}
  max_depth = 0
  for stage_layer in range(int(adapter.num_decoder_sparse_layers)):
    global_layer = adapter.global_layer_id("decoder", stage_layer)
    eids: list[int] = []
    seen = set()
    for eid, _count in decoder_by_layer.get(global_layer, []):
      item = (global_layer, int(eid))
      if item in initial or int(eid) in seen:
        continue
      seen.add(int(eid))
      eids.append(int(eid))
    ordered_by_layer[global_layer] = eids
    max_depth = max(max_depth, len(eids))

  plan: list[tuple[int, int]] = []
  for depth in range(max_depth):
    for stage_layer in range(int(adapter.num_decoder_sparse_layers)):
      global_layer = adapter.global_layer_id("decoder", stage_layer)
      eids = ordered_by_layer.get(global_layer, [])
      if depth < len(eids):
        plan.append((global_layer, eids[depth]))
  return plan
```

- [ ] **Step 3: Add C++ config fields**

In `ModuleMeta` in `utils.hpp`, add:

```cpp
bool enable_decoder_warmup_overlap = false;
std::string decoder_warmup_expert_plan = "";
```

In `ModuleMeta::init_from_map()` parse:

```cpp
enable_decoder_warmup_overlap = optional_bool("enable_decoder_warmup_overlap", enable_decoder_warmup_overlap);
decoder_warmup_expert_plan = optional_str("decoder_warmup_expert_plan", decoder_warmup_expert_plan);
```

In `log_configs()` log both fields.

In `handle_uninited_configs()` add:

```cpp
if (enable_decoder_warmup_overlap) {
  CHECK(per_layer_cache == false)
      << "decoder warmup overlap requires per_layer_cache=false";
  CHECK(cache_policy == "scheduler_aware")
      << "decoder warmup overlap requires cache_policy=scheduler_aware";
}
```

- [ ] **Step 4: Expose fields in pybind**

In `adapter.cpp`, add `.def_readwrite` for:

```cpp
"enable_decoder_warmup_overlap"
"decoder_warmup_expert_plan"
```

- [ ] **Step 5: Add CLI flag**

In `runner_util.py`, add:

```python
parser.add_argument("--enable_decoder_warmup_overlap", action=CustomBooleanAction)
```

Update the `--cache_policy` choices to include `scheduler_aware`:

```python
parser.add_argument(
  "--cache_policy",
  type=str,
  choices=["lru", "fifo", "nn", "min", "static-1", "static-2", "scheduler_aware"],
)
```

- [ ] **Step 6: Wire Python builder in `inject_model()`**

In `inject_model()` parameters, add:

```python
    enable_decoder_warmup_overlap: bool = False,
```

After `initial_plan` is built from `initial_hot_expert_file`, derive:

```python
decoder_warmup_expert_plan = None
if enable_decoder_warmup_overlap:
  if not initial_hot_expert_file:
    raise ValueError("enable_decoder_warmup_overlap requires initial_hot_expert_file")
  from sparse_llm_cache.utils.hot_experts import (
    build_decoder_warmup_overlap_plan,
    format_initial_expert_plan,
  )
  initial_set = set(initial_plan or [])
  decoder_warmup_plan = build_decoder_warmup_overlap_plan(
    initial_hot_expert_file,
    adapter,
    initial_plan=initial_set,
  )
  decoder_warmup_expert_plan = format_initial_expert_plan(decoder_warmup_plan)
```

Add to `param_dict`:

```python
'enable_decoder_warmup_overlap': str(enable_decoder_warmup_overlap),
'decoder_warmup_expert_plan': str(decoder_warmup_expert_plan),
```

- [ ] **Step 7: Run Python tests**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m pytest -q tests/test_src_hot_expert_initial_cache.py tests/test_deterministic_initial_cache.py
```

Expected: all selected tests pass.

---

### Task 6: Add Phase-Aware Scheduler And Warmup Queue

**Files:**
- Modify: `src/cpp_worker/prefetcher.hpp`
- Modify: `src/cpp_worker/prefetcher.cpp`

- [ ] **Step 1: Add scheduler phase and warmup task storage**

In `FetchScheduleWorker`, add:

```cpp
enum SchedulerPhase {
  kEncoderPhase = 0,
  kDecoderPredictorPhase,
};

SchedulerPhase phase = kEncoderPhase;
std::queue<std::pair<int, int>> decoder_warmup_queue;
std::unordered_set<int64_t> decoder_warmup_seen;
```

Add helper declarations:

```cpp
void set_phase(SchedulerPhase next_phase);
void clear_decoder_warmup_queue();
void rebuild_decoder_warmup_queue();
bool parse_layer_expert_plan(const std::string& plan, std::vector<std::pair<int, int>>& out);
bool pop_next_normal_prefetch(CopyTask& task);
bool pop_next_decoder_warmup(CopyTask& task);
int64_t flatten_expert(int layer_idx, int expert_idx) const;
```

- [ ] **Step 2: Implement phase helpers**

In `prefetcher.cpp`, implement:

```cpp
int64_t FetchScheduleWorker::flatten_expert(int layer_idx, int expert_idx) const {
  return int64_t(layer_idx) * int64_t(metas->num_expert) + int64_t(expert_idx);
}

void FetchScheduleWorker::set_phase(SchedulerPhase next_phase) {
  if (phase == next_phase) {
    return;
  }
  phase = next_phase;
  if (phase == kDecoderPredictorPhase) {
    clear_decoder_warmup_queue();
  }
}

void FetchScheduleWorker::clear_decoder_warmup_queue() {
  while (!decoder_warmup_queue.empty()) {
    decoder_warmup_queue.pop();
  }
  decoder_warmup_seen.clear();
}
```

- [ ] **Step 3: Parse and rebuild warmup queue**

Use the same `layer:expert,layer:expert` string format:

```cpp
bool FetchScheduleWorker::parse_layer_expert_plan(
    const std::string& plan,
    std::vector<std::pair<int, int>>& out) {
  out.clear();
  if (plan.empty()) {
    return true;
  }
  std::stringstream ss(plan);
  std::string entry;
  while (std::getline(ss, entry, ',')) {
    auto sep = entry.find(':');
    CHECK(!entry.empty() && sep != std::string::npos)
        << "invalid decoder_warmup_expert_plan entry: " << entry
        << ", plan=" << plan;
    int layer_idx = std::stoi(entry.substr(0, sep));
    int expert_idx = std::stoi(entry.substr(sep + 1));
    CHECK(metas->is_decoder_layer(layer_idx))
        << "decoder_warmup_expert_plan contains non-decoder layer: " << layer_idx;
    CHECK(expert_idx >= 0 && expert_idx < metas->num_expert)
        << "decoder_warmup_expert_plan expert out of range: " << expert_idx;
    out.push_back({layer_idx, expert_idx});
  }
  return true;
}

void FetchScheduleWorker::rebuild_decoder_warmup_queue() {
  clear_decoder_warmup_queue();
  if (!metas->enable_decoder_warmup_overlap) {
    return;
  }
  std::vector<std::pair<int, int>> parsed;
  parse_layer_expert_plan(metas->decoder_warmup_expert_plan, parsed);
  for (auto [layer_idx, expert_idx] : parsed) {
    auto gid = flatten_expert(layer_idx, expert_idx);
    if (decoder_warmup_seen.insert(gid).second) {
      decoder_warmup_queue.push({layer_idx, expert_idx});
    }
  }
}
```

- [ ] **Step 4: Rebuild queue at generation start**

In `start_generation()`, after `clear_all_job_queues()`:

```cpp
set_phase(kEncoderPhase);
rebuild_decoder_warmup_queue();
```

- [ ] **Step 5: Enter decoder phase on actual decoder layer**

In `advance_actual_layer()`, after current layer update:

```cpp
if (metas->is_decoder_layer(layer_idx)) {
  set_phase(kDecoderPredictorPhase);
}
```

- [ ] **Step 6: Implement normal prefetch priority**

Replace the non-precise part of `pop_next_task()` with a helper:

```cpp
bool FetchScheduleWorker::pop_next_normal_prefetch(CopyTask& task) {
  int best_layer = -1;
  std::tuple<int, int, int> best_key{INT_MAX, INT_MAX, INT_MAX};
  for (int layer_idx = 0; layer_idx < int(per_layer_job_queues.size()); layer_idx++) {
    if (per_layer_job_queues[layer_idx].empty()) {
      continue;
    }
    int bucket = 0;
    int distance = 0;
    if (current_layer < 0) {
      bucket = 1;
      distance = layer_idx;
    } else if (layer_idx == current_layer) {
      bucket = 0;
      distance = 0;
    } else if (layer_idx > current_layer) {
      bucket = 1;
      distance = layer_idx - current_layer;
    } else {
      bucket = 2;
      distance = current_layer - layer_idx;
    }
    auto key = std::make_tuple(bucket, distance, layer_idx);
    if (key < best_key) {
      best_key = key;
      best_layer = layer_idx;
    }
  }
  if (best_layer < 0) {
    return false;
  }
  task = per_layer_job_queues[best_layer].front();
  per_layer_job_queues[best_layer].pop();
  return true;
}
```

- [ ] **Step 7: Implement decoder warmup pop**

```cpp
bool FetchScheduleWorker::pop_next_decoder_warmup(CopyTask& task) {
  if (phase != kEncoderPhase || !metas->enable_decoder_warmup_overlap) {
    return false;
  }
  if (!cache->has_reclaimable_encoder()) {
    return false;
  }
  while (!decoder_warmup_queue.empty()) {
    auto [layer_idx, expert_idx] = decoder_warmup_queue.front();
    decoder_warmup_queue.pop();
    auto expert = model_loader->get_source(layer_idx, expert_idx);
    if (cache->is_in_cache(expert)) {
      continue;
    }
    task.start_mem_buf_idx = 0;
    task.stop_mem_buf_idx = metas->num_per_expert_param;
    task.expert = expert;
    task.is_precise = false;
    task.generation = current_generation;
    task.request_type = kCacheRequestDecoderWarmupOverlap;
    return true;
  }
  return false;
}
```

- [ ] **Step 8: Update `pop_next_task()`**

```cpp
void FetchScheduleWorker::pop_next_task(CopyTask &task, bool &found) {
  found = false;
  if (!precise_job_queue.empty()) {
    task = precise_job_queue.front();
    precise_job_queue.pop();
    found = true;
    return;
  }
  if (pop_next_normal_prefetch(task)) {
    found = true;
    return;
  }
  if (pop_next_decoder_warmup(task)) {
    found = true;
    return;
  }
}
```

- [ ] **Step 9: Build check**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build succeeds.

---

### Task 7: Verification And Smoke

**Files:**
- Review: `src/cpp_worker/cache.hpp`
- Review: `src/cpp_worker/cache.cpp`
- Review: `src/cpp_worker/prefetcher.hpp`
- Review: `src/cpp_worker/prefetcher.cpp`
- Review: `src/sparse_llm_cache/utils/hot_experts.py`
- Review: `src/sparse_llm_cache/utils/__init__.py`

- [ ] **Step 1: Run Python tests**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python -m pytest -q tests/test_src_hot_expert_initial_cache.py tests/test_deterministic_initial_cache.py tests/test_switch_adapter.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Build extension**

Run:

```bash
/mnt/huwf5/conda-envs/promoe-moe-cache/bin/python setup.py build_ext --inplace
```

Expected: build succeeds.

- [ ] **Step 3: Run default behavior smoke**

Run an existing Switch baseline without the new policy:

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

Expected: no deterministic-init or scheduler-aware validation error.

- [ ] **Step 4: Run scheduler-aware overlap smoke**

Run with deterministic init, scheduler-aware policy, and decoder warmup overlap:

```bash
cd /mnt/huwf5/promoe/examples/small-demo
CUDA_VISIBLE_DEVICES=0 /mnt/huwf5/conda-envs/promoe-moe-cache/bin/python transformers-app.py \
  --model_id google/switch-base-128 \
  --dataset validation \
  --batch_size 1 \
  --num_predict_expert_per_layer 0 \
  --cache_rate 0.375 \
  --cache_policy scheduler_aware \
  --per_layer_cache False \
  --reorder_experts False \
  --early_preempt False \
  --predict_input_mode moe_layer_logits \
  --layer_predict_interval 1 \
  --layer_predict_max_window 3 \
  --layer_predict_use_last_output True \
  --predictor_model_path /mnt/huwf5/promoe/deps/sparse-llm-cache-scripts/moe-predict-models/switch-base-128-mmlu-professional_law-test-train-validation-val/decoder \
  --gpu_mem_limit_gb 24 \
  --initial_cache_policy hot_expert \
  --initial_hot_expert_file /mnt/huwf5/promoe/deps/moe-traces/switch-base-128-mmlu-professional_law-test/hot_experts/switch-base-128.test.json \
  --enable_decoder_warmup_overlap True
```

Expected: generate starts and decoder warmup overlap does not fail when reclaimable encoder is unavailable.

- [ ] **Step 5: Negative config smoke**

Run with:

```bash
--enable_decoder_warmup_overlap True --cache_policy lru
```

Expected: startup fails with:

```text
decoder warmup overlap requires cache_policy=scheduler_aware
```

- [ ] **Step 6: Diff hygiene**

Run:

```bash
git diff --check -- src/cpp_worker/cache.hpp src/cpp_worker/cache.cpp src/cpp_worker/prefetcher.hpp src/cpp_worker/prefetcher.cpp src/cpp_worker/utils.hpp src/cpp_worker/utils.cpp src/sparse_llm_cache/utils/hot_experts.py src/sparse_llm_cache/utils/__init__.py src/sparse_llm_cache/utils/runner_util.py
```

Expected: no output, exit code 0.

---

## Self-Review

- Spec coverage: tasks cover scheduler-aware victim selection, reclaimable marking, phase-aware scheduler selection, decoder warmup queue, and hot-file-derived interleaved decoder plan.
- Scope control: no encoder predictor implementation, no new hot file parameter, no active urgent set, no multi-stream scheduler rewrite.
- Type consistency: request type is `CacheRequestType`; policy name is `scheduler_aware`; user flag is `enable_decoder_warmup_overlap`; generated C++ string config is `decoder_warmup_expert_plan`.
