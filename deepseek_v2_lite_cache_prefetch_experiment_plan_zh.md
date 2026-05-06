# DeepSeek-V2-Lite Cache/Prefetch 实验计划

本文档固定当前阶段的实验计划。目标不是复现某一篇现有系统论文，而是在真实 `DeepSeek-V2-Lite` 路由轨迹上，把 cache residency 与 prefetch utility 分开评估，为后续方法和论文主线提供稳定指标。

## 1. 实验目标

本实验回答三个问题：

1. 最简单的 cache 策略本身能带来多少收益：`LRU`、`LFU` 在没有 prefetch 的情况下，对 hit rate、transfer、TPOT 的影响。
2. 最简单的 prefetch 策略本身能带来多少收益：在没有复杂 cache 策略兜底时，简单 next-layer prefetch 是否真的及时、非冗余、并减少 stall。
3. 高 cache hit 是否会掩盖低质量 prefetch：当 cache residency 较高时，prefetch 的 accuracy、coverage、timely utility 与 stall saved 是否同步提升。

论文主线应围绕以下判断展开：

```text
cache hit 高，不等于 prefetch 有用。
真正需要单独评价的是 prefetch 的冗余性、及时性和增量 stall reduction。
```

## 2. 硬件与环境约束

实验服务器使用 AutoDL：

| 项目 | 配置 |
|---|---|
| GPU | 1 x NVIDIA T4 |
| GPU 显存 | 15 GB |
| CPU 内存 | 47 GB |
| 系统限制 | 不依赖 apt 命令 |
| 推荐环境 | conda + pip |
| 工作目录 | `/root/autodl-tmp` |

环境原则：

| 原则 | 说明 |
|---|---|
| 模型、缓存、临时文件放入数据盘 | 避免系统盘空间不足 |
| 不依赖 notebook | 使用终端脚本运行 |
| 不使用 apt | Python 依赖尽量通过 conda/pip 解决 |
| 优先 trace/replay，再做小规模真实端到端验证 | 降低 T4 显存和实现复杂度风险 |

建议设置：

```bash
export HF_HOME=/root/autodl-tmp/hf_home
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_home/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_home/transformers
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
```

## 3. 模型与运行边界

模型采用 `DeepSeek-V2-Lite`。由于 T4 15GB 无法舒适承载完整高精度模型，实验默认采用量化加载或 CPU offload。

运行边界：

| 项目 | 设置 |
|---|---|
| 阶段 | 主要关注 decode |
| batch size | `1` |
| prompt length | 正常集约 `256` tokens；2x2 集按模板控制 |
| max new tokens | `64` |
| smoke test | 每组 `8` requests |
| 正式主实验 | 每组 `64` requests |
| 扩展实验 | 每组 `128` requests，视 AutoDL 稳定性决定 |

DeepSeek-V2-Lite 的 MoE 参数不手写，运行时从模型 config 读取：

```text
num_layers
num_experts
top_k
moe layer index
expert hidden size
```

## 4. 总体实验方法

实验采用两阶段设计：

1. 采集真实路由轨迹：运行 DeepSeek-V2-Lite，记录每个 request、token、layer 的真实 expert assignment。
2. 策略回放与小规模验证：在同一条 assignment trace 上回放不同 cache/prefetch 策略，并用小规模真实运行验证趋势。

这样做的原因：

| 设计 | 作用 |
|---|---|
| 固定 assignment trace | 保证各策略比较公平，避免生成内容或路由变化混入策略效果 |
| 先 replay 后真实验证 | 在 T4 上可控，同时保留真实模型路由依据 |
| 指标沿用同一套定义 | 后续方法和论文结果不需要换指标口径 |

所有 fixed-trajectory 策略要求：

```text
trajectory_consistency = 1
```

如果未来加入会改变路由的 baseline，需要单独报告，不能与 fixed-trajectory cache/prefetch 策略混写结论。

## 5. 策略矩阵

主实验固定 5 个策略。

| Policy | Cache 策略 | Prefetch 策略 | 目的 |
|---|---|---|---|
| `OnDemand` | 无持久 cache | 无 prefetch | 最低基线 |
| `LRU_only` | LRU | 无 prefetch | 共识 cache baseline |
| `LFU_only` | LFU | 无 prefetch | 共识 cache baseline |
| `Prefetch_only` | 无复杂 cache，仅 transient prefetch buffer | 简单 next-layer prefetch | 隔离 prefetch 本身 |
| `LRU_plus_prefetch` | LRU | 简单 next-layer prefetch | 检查 cache 是否掩盖 prefetch |

策略定义：

| 策略 | 精确定义 |
|---|---|
| `OnDemand` | 每次 expert demand 若不在 GPU，则按需加载；不保留跨步专家或只保留最小运行必要状态 |
| `LRU_only` | GPU expert cache 满时淘汰最近最少使用 expert；不发出 prefetch |
| `LFU_only` | GPU expert cache 满时淘汰历史使用频率最低 expert；不发出 prefetch |
| `Prefetch_only` | prefetch 写入一个有限 transient buffer；demand 访问后可消费，但不执行 LRU/LFU 这类长期替换策略 |
| `LRU_plus_prefetch` | demand 与 prefetch 共享 LRU cache；如果 prefetch 目标已在 cache 中，记为 redundant prefetch |

简单 prefetch 策略先固定为：

```text
next-layer same-expert prefetch
```

含义：在当前 layer 看到 token 访问 expert `e` 后，尝试为下一 MoE layer 预取同编号或映射后的 expert `e`。如果 DeepSeek-V2-Lite 的专家编号跨层不可直接对应，则退化为：

```text
previous-step layer-local top-k prefetch
```

即用同一 layer 最近一步的热门 expert 作为下一步预取候选。

## 6. Cache 容量设置

主实验使用 3 个 cache ratio。

| cache ratio | 作用 |
|---:|---|
| `0.05` | 低容量压力，容易暴露 miss 和 eviction |
| `0.10` | 主实验默认值 |
| `0.20` | 较高 residency，观察 cache 掩盖 prefetch 的程度 |

如果 T4 显存不足，优先保留：

```text
cache_ratio = 0.10
```

然后再补 `0.05` 和 `0.20` 的敏感性实验。

## 7. 主指标体系

指标分为共识指标和本实验独特指标。论文主表建议固定使用 8 个指标，不再无限扩展。

### 7.1 共识指标

| 指标 | 定义 | 说明 |
|---|---|---|
| `TPOT_ms` | decode 阶段平均每 token 时间 | 系统论文共识性能指标 |
| `tokens_per_sec` | 每秒生成 token 数 | 便于和 benchmark 对比 |
| `cache_hit_assignment_ratio` | demand assignment 发生时 expert 已在 GPU/cache/buffer 的比例 | 衡量 cache residency |
| `cpu_gpu_transfer_bytes` | 因 expert loading 产生的 CPU->GPU 字节数 | 衡量数据移动开销 |
| `prefetch_accuracy` | 被发出的 prefetch 中，未来被真实 demand 使用的比例 | 传统 prefetch 正确性指标 |
| `prefetch_coverage` | 真实 demand assignment 中，被提前 prefetch 覆盖的比例 | 传统 prefetch 覆盖能力指标 |

公式：

```text
cache_hit_assignment_ratio =
  cache_hit_assignments / total_assignments

prefetch_accuracy =
  used_prefetches / issued_prefetches

prefetch_coverage =
  assignments_covered_by_prefetch / total_assignments
```

### 7.2 独特指标

| 指标 | 定义 | 用途 |
|---|---|---|
| `prefetch_redundant_ratio` | 发出时已在 cache、重复发出、或最终被 cache residency 掩盖的 prefetch 比例 | 判断 prefetch 是否有增量价值 |
| `timely_incremental_utility_ratio` | 非冗余 prefetch 中，按时到达并真实避免 on-demand stall 的比例 | 判断 prefetch 是否真正有用 |
| `stall_saved_ratio` | 相比 `OnDemand`，实际减少的 expert-loading stall 占比 | 将机制指标接回系统性能 |

公式：

```text
prefetch_redundant_ratio =
  redundant_prefetches / issued_prefetches

timely_incremental_utility_ratio =
  timely_incremental_useful_prefetches / non_redundant_prefetches

stall_saved_ratio =
  (on_demand_expert_loading_stall_ms - policy_expert_loading_stall_ms)
  / on_demand_expert_loading_stall_ms
```

这三个指标用于说明：

```text
高 cache hit 不能替 prefetch 背书。

如果 cache_hit_assignment_ratio 高，
但 prefetch_redundant_ratio 也高，
timely_incremental_utility_ratio 低，
stall_saved_ratio 没有同步提升，
则说明 prefetch 的表面成功主要被 cache residency 掩盖。
```

### 7.3 Sanity 指标

这些指标用于检查实验有效性，不作为论文主指标。

| 指标 | 用途 |
|---|---|
| `trajectory_consistency` | fixed-trajectory 策略必须为 1 |
| `issued_prefetch_count` | 检查 prefetch 是否真的被触发 |
| `eviction_count` | 检查 cache 压力 |
| `peak_gpu_memory_GB` | 检查 T4 显存安全 |
| `request_domain` | 标记 normal / translation / summarization |
| `condition` | 标记 2x2 workload cell |
| `seed` | 支持多 seed 聚合 |

## 8. 数据集设计

实验包含两组数据。

### 8.1 正常数据集

目的：验证指标在常规自然 workload 下成立，避免论文只依赖构造数据。

建议数据：

| 项目 | 设置 |
|---|---|
| 数据源 | `WikiText-103` 小样本，或 C4 小样本 |
| 主推荐 | `WikiText-103` |
| request 数 | `64` 主实验，`128` 扩展 |
| prompt 长度 | 约 `256` tokens |
| max new tokens | `64` |
| seed | `0` 主实验，后续补 `1,2` |

正常数据集输出：

```text
results/deepseek_v2_lite_cache_prefetch/normal/request_trace.jsonl
results/deepseek_v2_lite_cache_prefetch/normal/policy_summary.csv
results/deepseek_v2_lite_cache_prefetch/normal/policy_metrics_by_request.csv
```

### 8.2 2x2 Workload 数据集

目的：诊断 cross-request shift 和 intra-request mixing 对 cache/prefetch utility 的影响。

四个条件：

| Condition | Cross-request shift | Intra-request mixing | 作用 |
|---|---:|---:|---|
| `stable_homogeneous` | No | No | clean control |
| `shifted_homogeneous` | Yes | No | 只引入跨请求分布漂移 |
| `stable_mixed` | No | Yes | 只引入请求内跨域混合 |
| `shifted_mixed` | Yes | Yes | 同时引入漂移与混合 |

推荐 domain：

| Domain | Prompt 类型 |
|---|---|
| `translation` | 翻译任务 |
| `summarization` | 摘要任务 |

推荐参数：

| 参数 | 值 |
|---|---|
| `num_requests_per_cell` | `64` |
| `shift_block_size` | `16` |
| `shift_major_fraction` | `0.8` |
| `stable_mix_fraction` | `0.5` |
| `mix_mode` | `interleave` |
| `interleave_chunk_words` | `16` |
| `max_new_tokens` | `64` |
| `cache_ratio` | `0.10` 主实验，`0.05/0.20` 敏感性 |
| `seed` | `0` 主实验，后续补 `1,2` |

2x2 数据集输出：

```text
results/deepseek_v2_lite_cache_prefetch/2x2/request_trace.jsonl
results/deepseek_v2_lite_cache_prefetch/2x2/workload_sanity.csv
results/deepseek_v2_lite_cache_prefetch/2x2/policy_summary.csv
results/deepseek_v2_lite_cache_prefetch/2x2/factorial_effects.csv
```

## 9. 运行流程

主流程分 5 步。

| 步骤 | 动作 | 输出 |
|---|---|---|
| 1 | 环境检查与 smoke test | `env_check.json` |
| 2 | 正常数据集采集真实 routing trace | `normal/request_trace.jsonl` |
| 3 | 2x2 数据集采集真实 routing trace | `2x2/request_trace.jsonl` |
| 4 | 在固定 trace 上回放 5 个策略 | `policy_summary.csv` |
| 5 | 小规模真实端到端验证 | `e2e_validation.csv` |

正式运行前必须通过 smoke test：

```text
normal: 8 requests
2x2: 每个 cell 2 requests
max_new_tokens: 16
```

正式主实验：

```text
normal: 64 requests
2x2: 4 cells x 64 requests
policies: 5
cache_ratios: 0.10
seed: 0
```

扩展实验：

```text
cache_ratios: 0.05, 0.10, 0.20
seeds: 0, 1, 2
normal: 128 requests
2x2: 4 cells x 128 requests
```

## 10. 结果表设计

论文主表建议拆成两张。

### 10.1 Benchmark 表

用于回答：我们是否在共识指标上打得过 baseline。

| Dataset | Policy | Cache Ratio | TPOT ms | Tokens/s | Cache Hit | Transfer GB | Prefetch Acc. | Prefetch Cov. |
|---|---|---:|---:|---:|---:|---:|---:|---:|

### 10.2 Utility Diagnosis 表

用于回答：高 cache hit 是否掩盖低 prefetch utility。

| Dataset | Condition | Policy | Cache Hit | Redundant Prefetch | Timely Incremental Utility | Stall Saved |
|---|---|---|---:|---:|---:|---:|

最重要的对比：

```text
LRU_only vs LRU_plus_prefetch
Prefetch_only vs LRU_plus_prefetch
stable_homogeneous vs shifted_mixed
normal vs 2x2
```

## 11. 预期现象与判读

本实验不是预设某个策略一定最好，而是检查指标链是否能解释性能。

可能结果与解释：

| 现象 | 解释 |
|---|---|
| `LRU_only` 提高 cache hit 和 TPOT | 简单 cache baseline 有效 |
| `Prefetch_only` accuracy 不低但 TPOT 提升有限 | 预测对不等于及时省 stall |
| `LRU_plus_prefetch` cache hit 高但 redundant ratio 高 | cache residency 掩盖 prefetch |
| `shifted_mixed` utility 明显下降 | 2x2 workload 成功暴露非平稳压力 |
| `normal` 与 `2x2` 差异明显 | 独特数据集提供额外诊断价值 |

论文中最稳的结论：

```text
现有 cache hit / prefetch accuracy 不能完整解释 prefetch 的系统价值。
在非平稳 mixed workload 下，高 cache residency 可能与低 timely incremental utility 共存。
因此，MoE expert offloading 需要把 cache residency 与 prefetch utility 分开评估。
```

## 12. 风险与降级方案

| 风险 | 降级方案 |
|---|---|
| T4 显存不足 | 降低 cache_ratio，减少 max_new_tokens，优先 trace/replay |
| DeepSeek-V2-Lite 加载失败 | 先跑路由抽取 smoke test，再逐步打开 offload/quantization |
| 正常数据集下载慢 | 使用本地 prompt 文件或手写 64 条 prompt |
| 2x2 prompt 太长 | 降低 interleave chunk 数量，保持 domain 结构不变 |
| 真实异步 prefetch 难实现 | 先用 trace-level ready time 模型，后续再替换真实 runtime |
| 指标过碎 | 主表固定 8 个指标，其余只作为 sanity |

## 13. 不应过度声称的内容

当前实验不能直接声称：

```text
我们已经完整复现 DALI、HybriMoE、MoE-Infinity 或 SP-MoE。
```

也不能直接声称：

```text
我们的策略已经在大规模生产 serving 中优于所有系统。
```

当前实验可以支撑：

```text
在真实 DeepSeek-V2-Lite 路由轨迹和可控 2x2 workload 下，
cache residency 与 prefetch utility 可以显著不一致。
传统 cache hit / prefetch accuracy 指标不足以解释 prefetch 的增量系统价值。
```

## 14. 后续实现顺序

实现时按以下顺序推进：

1. 实现 DeepSeek-V2-Lite routing trace 采集。
2. 实现固定 trace 的 `OnDemand`、`LRU_only`、`LFU_only` 回放。
3. 实现 `Prefetch_only` 与 `LRU_plus_prefetch` 回放。
4. 输出主指标与 sanity 指标。
5. 跑 normal smoke test。
6. 跑 2x2 smoke test。
7. 跑主实验。
8. 再决定是否做真实异步 prefetch 的端到端版本。

