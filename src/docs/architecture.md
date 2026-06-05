# Promoe `src/` 架构梳理

## 1. 总体架构（两层协作）

`src/` 采用 **Python 控制面 + C++/CUDA 执行面** 的双层架构，目标是为 MoE（Mixture of Experts）模型提供专家参数缓存、预测与预取能力。

- Python 层（`sparse_llm_cache/`）负责：模型注入、模块 Hook、元信息推断、策略编排。
- C++ 层（`cpp_worker/`）负责：缓存管理、线程调度、预测执行、异步拷贝、PyBind 暴露。

核心思想：  
在 Python 中“无侵入”接管模型执行事件，在 C++ 中完成高性能缓存与预取调度，使专家参数在需要时已经就位，减少运行时等待。

---

## 2. 目录与模块边界

```text
src/
├── sparse_llm_cache/                 # Python控制面
│   ├── utils/
│   │   ├── __init__.py               # 注入主入口（inject_model/hack_transformers）
│   │   ├── hooks.py                  # Hook框架与事件上报
│   │   ├── common_metas.py           # 模型元信息自动推断
│   │   ├── filter.py                 # 模块过滤规则
│   │   └── runner_util.py            # 运行辅助
│   ├── prefetch/
│   │   ├── expert_prefetch.py        # Python侧预取实验实现
│   │   └── oracle_prefetch_policy.py # oracle预取策略
│   ├── oracle_cache_policy.py        # oracle缓存策略
│   ├── expert_cache_inject_accelerate.py
│   └── profile/profiler.py
└── cpp_worker/                       # C++/CUDA执行面
    ├── adapter.cpp                   # PyBind11 API入口
    ├── prefetcher.hpp/.cpp           # 预取中枢 PrefetchMngr
    ├── worker.hpp/.cpp               # Fetch/Predict/Unlock 多线程执行
    ├── cache.hpp/.cpp                # CacheMngr 与替换策略
    ├── predictor.hpp/.cpp            # 预测器
    ├── model_loader.hpp/.cpp         # 专家参数注册与内存映射
    ├── utils.hpp/.cpp                # ModuleMeta/通用并发与状态设施
    ├── profiler.hpp/.cpp             # 性能统计
    ├── adapter-llama.hpp/.cpp
    ├── logging.cc/.hpp
    └── cuda_helper_func.cu
```

---

## 3. 关键组件职责

### 3.1 Python 控制面

- `sparse_llm_cache/utils/__init__.py`
  - `inject_model()`：注入总入口。自动推断模型 MoE 元信息、构建 C++ 核心对象、注册专家参数、替换参数引用、安装 Hook、启动后台线程。
  - `hack_transformers()`：劫持 HuggingFace `PreTrainedModel` 的加载流程，实现自动注入与启动。
- `sparse_llm_cache/utils/common_metas.py`
  - 按模型家族提供专家命名解析与过滤规则，用于自动定位 MoE 层与专家模块。
- `sparse_llm_cache/utils/hooks.py`
  - 定义 `ExpertHook` / `MoeLayerHook` / `MoEAttnHook` / `TimingHook` 等。
  - 将 Python 侧 forward 事件转换为对 C++ `PrefetchMngr` 的上报调用。
- `sparse_llm_cache/oracle_cache_policy.py`、`prefetch/oracle_prefetch_policy.py`
  - 使用离线轨迹做 oracle 决策（如 MIN next-use-time），用于策略上界评估。
- `expert_cache_inject_accelerate.py`、`prefetch/expert_prefetch.py`
  - 偏实验/验证的 Python 侧缓存与预取逻辑。

### 3.2 C++ 执行面

- `cpp_worker/adapter.cpp`
  - 通过 `PYBIND11_MODULE` 暴露 `ModuleMeta`、`ModelLoader`、`PredictorBase`、`PrefetchMngr`、`CacheMngr`、`Profiler` 等能力给 Python。
- `cpp_worker/prefetcher.hpp/.cpp`
  - `PrefetchMngr` 是运行期中枢，统一协调缓存、预测、异步拷贝与线程生命周期。
  - 对外 API 包括：`report_one_expert`、`report_one_layer`、`report_moe_layer_logits`、`one_expert_done`、`launch_thread` 等。
- `cpp_worker/worker.hpp/.cpp`
  - `FetchScheduleWorker`：将层级专家集合转成可执行 `CopyTask`。
  - `FetchWorker`：执行具体拷贝任务（Host -> Device）。
  - `PredictWorker`：根据 logits/历史触发预测并下发预取请求。
  - `ExpertUnlockWorker`：跟踪专家生命周期并解锁。
- `cpp_worker/cache.hpp/.cpp`
  - `CacheMngr` 维护缓存槽位、命中/失效路径、替换策略（LRU/FIFO/NN/MIN 等）。
- `cpp_worker/model_loader.hpp/.cpp`
  - 管理专家参数注册、逻辑参数映射、内存布局与 pinned memory。
- `cpp_worker/predictor.hpp/.cpp`
  - 统一预测器接口，加载预测模型并产出未来层专家候选。
- `cpp_worker/utils.hpp/.cpp`
  - `ModuleMeta` 配置承载体（支持从参数 map / 环境变量初始化），以及并发辅助设施。

---

## 4. 组件交互方式

## 4.1 初始化交互（装配阶段）

1. Python 调用 `inject_model(model, ...)`。
2. 通过 `common_metas` 自动推断模型结构（层数、专家数、命名规则）。
3. 构建并初始化 C++ 对象：`ModuleMeta -> ModelLoader -> Predictor -> PrefetchMngr`。
4. 注册专家参数并建立逻辑引用映射（将专家参数引用重定向到可缓存/可换入的缓冲）。
5. 在专家模块、MoE 层等位置安装 Hook。
6. `launch_thread()` 启动 C++ 后台线程（预测、调度、拷贝、解锁）。

## 4.2 运行期交互（事件驱动）

在模型 forward 过程中，Hook 持续将事件上报给 `PrefetchMngr`：

- `ExpertHook.pre_forward` -> `report_one_expert(layer, expert)`  
  表示某专家即将执行，需要确保其参数可用。
- `ExpertHook.post_forward` -> `one_expert_done(layer, expert)`  
  表示专家执行完成，可进入后续解锁/状态推进。
- `MoeLayerHook.pre/post_forward` -> `report_moe_layer_logits(...)`  
  提供预测输入并在层结束时推进调度。
- `report_one_layer(layer, experts)`  
  直接上报某层专家集合，用于预取规划。

`PrefetchMngr` 接收事件后，触发内部管线：

1. `PredictWorker` 根据 logits/历史计算未来专家候选。
2. `FetchScheduleWorker` 将候选转换为 `CopyTask` 队列。
3. `CacheMngr` 决定命中、分配、驱逐（策略可配）。
4. `FetchWorker` 执行异步拷贝并更新专家可用状态。
5. 执行线程等待/唤醒对应专家，保证正确性与并发效率。

---

## 5. 数据流与 I/O 边界

- **Host -> Device 数据流**
  - 专家参数在需要时从 CPU（可 pinned）异步拷贝到 GPU 缓存区。
- **Device -> Host 数据流**
  - MoE logits 从 GPU 回传（或映射）供预测器使用。
- **参数引用流**
  - Python 注入阶段将专家参数引用改写为逻辑映射引用，运行时由缓存系统动态指向真实物理位置。
- **外部 I/O**
  - 读取预测模型（如 `.pt`）、oracle 轨迹张量/JSON、运行配置。
  - 输出运行日志、trace 与统计信息。

---

## 6. 典型时序（简化）

1. 模型进入某 MoE 层，Hook 上报 logits/专家选择。
2. `PredictWorker` 给出下一步可能需要的专家集合。
3. `CacheMngr` 评估命中与驱逐，`FetchWorker` 执行预取拷贝。
4. 当前 expert 执行前，通过 `wait_expert` 保证参数就绪。
5. expert 执行完成后，上报 `one_expert_done`，释放或降级状态。
6. 循环进入下一层，持续形成“执行-预测-预取”流水线。

---

## 7. 快速理解建议（阅读顺序）

建议按以下顺序阅读源码，可最快建立全局认知：

1. `sparse_llm_cache/utils/__init__.py`（Python 注入总入口）
2. `cpp_worker/adapter.cpp`（Python/C++ API 边界）
3. `cpp_worker/prefetcher.hpp`（运行期中枢接口）
4. `cpp_worker/worker.hpp`（线程与任务模型）
5. `cpp_worker/cache.hpp`（缓存策略与状态）
6. `sparse_llm_cache/utils/hooks.py`（事件来源与上报语义）

---

## 8. 一句话总结

这个架构本质上是：**Python 负责“感知与编排”，C++ 负责“高性能执行与调度”**，通过 Hook 事件桥接把模型运行时行为转化为可预测、可预取、可缓存的专家参数流水线。

## Global Predictor Runtime Ids

Runtime predictor inputs and outputs use global MoE layer ids. For encoder-decoder models such as Switch, encoder MoE layers occupy `0..E-1`, decoder MoE layers occupy `E..L-1`, and decoder predictor input boundaries occupy `E..L`. Boundary `L` is only a predictor input for cross-token prediction; it is not a cache layer and must not be submitted as a prefetch layer task.

Switch runtime reports only decoder layers to the predictor. The first decoder pre-forward report uses boundary `E`, and each decoder post-forward report uses `global_layer_id + 1`. Predictor directories must use v2 global metadata: `metas.json` with `schema_version: 2`, `id_space: "global"`, and global `outputs` ranges. Old decoder-local predictor artifacts must be retrained instead of loaded with an offset.

