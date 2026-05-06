# 2x2 Workload 最小实验基座说明

本目录包含一个纯 Python、无外部依赖的 MoE expert cache/prefetch 对比基座：

```bash
python scripts/prefetch_2x2_compare.py --results_root results/minimal_compare
```

它用于在同一套受控 synthetic token-expert trajectory 上，比较不同论文机制在 `2x2 workload` 压力下的相对表现。它不是各论文系统的完整复现，也不声称复现真实模型精度或真实硬件吞吐。

## 这次修正后的核心前提

- 路由不再使用随机抽样，而是更接近真实 MoE router 的 deterministic `top-k` 选择。
- `stable_homogeneous` 默认开启 `domain marginal` 控制，使 clean control 不再因为全局领域边际不同而和 shifted cell 混淆。
- prefetch 不再“发出即命中”。现在会受 `prefetch_bandwidth_slots` 和 ready step 约束，超过带宽的预取会延后生效。
- 混合 CPU/GPU 策略不再把每个 miss 独立按 `min(cpu, transfer)` 记账，而是用单步内的双资源 lane 近似调度，减少对 HybriMoE/DALI 的不公平高估。
- 新增 capacity-aware routing baseline，但它会改写 token-expert assignment，因此 `trajectory_consistency` 不再恒为 1。它只能作为 routing-side 对照，不能与 fixed-trajectory prefetch 结论混写。

## 实验设计

脚本实现了实验指南中的四个 workload cell：

| Condition | Cross-request shift | Intra-request mixing |
|---|---:|---:|
| `stable_homogeneous` | No | No |
| `shifted_homogeneous` | Yes | No |
| `stable_mixed` | No | Yes |
| `shifted_mixed` | Yes | Yes |

默认配置：

- `num_requests=128`
- `shift_block_size=32`
- `tokens_per_request=64`
- `num_layers=8`
- `num_experts=16`
- `experts_per_device=2`
- `top_k=2`
- `cache_ratios=0.03,0.10,0.40`
- `seeds=0,1,2`
- `ttl_steps=1`

## 论文到最小策略的映射

| 论文 | 策略名 | 最小机制 |
|---|---|---|
| ZeRO-Offload | `zero_offload_min` | 最朴素 CPU offload 对照：无持久 MoE expert cache，按需迁移 |
| Pre-gated MoE | `pregated_next_layer` | next-layer activated-expert prefetch |
| SpecMoEOff | `spec_moe_off` | speculative multi-step lookahead + draft overhead |
| MoE-Infinity | `moe_infinity_eam` | request-level EAM 相似度预测 + 稀疏感知替换 |
| Diff-MoE | `diff_moe_priority` | global hot experts + dynamic local priority cache |
| HybriMoE | `hybrimoe_mrs` | hybrid CPU/GPU miss handling + score-aware cache |
| DALI | `dali_workload_aware` | greedy hybrid assignment + workload-window cache replacement |
| Capacity-Aware Inference | `capacity_token_drop` | score-ordered capacity cap token retention |
| Capacity-Aware Inference | `capacity_expanded_drop` | local reroute under the same cache runtime |

## 输出文件

运行后生成：

| 文件 | 内容 |
|---|---|
| `results/minimal_compare/summary.csv` | 每个 seed/condition/cache/policy 的原始指标 |
| `results/minimal_compare/aggregate.csv` | 跨 seed 的 mean/std/95% CI |
| `results/minimal_compare/workload_sanity.csv` | 2x2 workload sanity 指标 |
| `results/minimal_compare/factorial_effects.csv` | shift、mixing、interaction effect |
| `results/minimal_compare/comparison_report.md` | 自动生成的对比报告 |

## 主指标

assignment-level 主指标仍优先报告：

- `issue_f1_global`
- `cache_hit_assignment_ratio`
- `prefetch_assignment_redundant_ratio`
- `prefetch_assignment_utility_ratio`
- `prefetch_covered_assignment_ratio`
- `prefetch_over_on_demand`

同时补充以下 sanity / realism 指标：

- `mean_peak_expert_load_ratio`
- `mean_peak_device_load_ratio`
- `mean_expert_load_cv`
- `trajectory_consistency`
- `routing_assignment_retention_ratio`
- `routing_reroute_ratio`
- `routing_drop_ratio`

## 解释边界

这些结果适合回答：

- 在 2x2 workload 压力下，不同 cache/prefetch 机制的方向性差异
- 更高 cache residency 是否伴随更高冗余 prefetch
- `TTL=1` 的 timely utility 是否真的覆盖 assignment
- shift/mixing 对 latency ratio 与机制指标的影响
- routing-side straggler mitigation 与 fixed-trajectory prefetch 的差异

这些结果不适合直接声称：

- 某篇论文真实系统在某 GPU/CPU 上的绝对吞吐
- 某个真实 MoE 模型的精度或真实路由分布
- 完整异步 transfer ready/consume 的硬件级归因

如果要推进到论文级系统实验，下一步应将 `simulate_policy()` 中的 synthetic trace 替换为真实 runtime 的 `cache_prefetch_trace.json` / `prefetch_trace.tsv`，并保留同一套 assignment-level 分析指标。
