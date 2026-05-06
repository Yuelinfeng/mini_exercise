# 受控 2x2 Workload 实验设计与实现指南

本文档用于指导 Pregated MoE 项目中的受控 `2x2 workload` 实验设计、实现、运行、检查与论文表述。

## 1. 实验目标

### 1.1 核心问题

Decoder MoE expert prefetch 的隐含前提是：短时间内的 expert 访问具有足够稳定的短程可预测性。也就是说，prefetch 不只是依赖“哪些 expert 总体常被访问”，还依赖“从当前层、当前 token、当前请求状态出发，下一步会访问哪些 expert”这一短程 transition locality。

本实验用一个受控的 `2x2 workload` 框架，将 workload 非平稳性拆成两个正交压力源：

- `cross-request workload shift`：请求流在相邻窗口之间发生任务/领域分布变化。
- `intra-request cross-domain mixing`：单个请求内部同时包含多个任务/领域片段。

实验的目标不是构造“最坏情况 prompt”，而是构造一个可复现、可解释、可扩展的 workload class，用来回答：

1. 只发生跨请求 shift 时，prefetch 是否退化？
2. 只发生请求内 mixing 时，prefetch 是否退化？
3. 两者同时存在时，是否形成更稳定的 failure condition？
4. 退化是否来自短程 expert predictability、prefetch utility、timeliness 或 cache/prefetch action 解耦？

### 1.2 最终确认的指标故事线

本文档最终采用以下证据链来讲述实验结果：

```text
trajectory consistency
  -> issue_f1_global
  -> cache_hit_assignment_ratio
  -> prefetch_assignment_redundant_ratio
  -> prefetch_assignment_utility_ratio
  -> prefetch_over_on_demand
```

它对应的问题是：

1. `trajectory_consistency`：prefetch 与 on-demand 的端到端比较是否公平？
2. `issue_f1_global`：预取策略本身是否具备非平凡的 expert set 预测能力？
3. `cache_hit_assignment_ratio`：真实 token-expert assignment 中有多少已经由 GPU cache 覆盖？
4. `prefetch_assignment_redundant_ratio`：预取候选中有多少在发出时已经被 cache 覆盖、缺少增量搬运价值？
5. `prefetch_assignment_utility_ratio`：非冗余预取候选中，有多少在短 TTL 窗口内真正被使用？
6. `prefetch_over_on_demand`：真实系统结果中，prefetch 相比 on-demand 是否带来端到端加速？

因此本文的主结论不应写成简单的“prefetch 不准”，也不应只写成“cache hit 高”。更准确的目标表述是：

```text
在固定生成轨迹且 predictor 具有非平凡预测能力的前提下，
更高的 GPU expert cache residency 会提高预取发出时目标 expert 已在 cache 中的比例。
虽然这会表现为更高的 prefetch-event cache hit，
但其中更大一部分是冗余或非增量的 cache 命中，无法转化为有效的 latency hiding。
因此，cache-incremental timely utility 提升有限，
最终使 prefetch 的端到端收益受限。
```

## 2. 因子定义

设请求流为：

```text
R = {r_0, r_1, ..., r_{T-1}}
```

每个请求 `r_i` 有两个分布：

```text
q_i(d): request-level domain distribution inside request r_i
P_b(d): stream-level domain distribution inside block/window b
```

其中 `d` 是 domain，例如：

```text
translation, summarization
```

### 2.1 Cross-Request Shift

`cross-request shift` 指相邻请求窗口的 stream-level domain distribution 发生变化。

在当前实现中，请求流被切成固定大小的 block：

```text
block_index = request_index // shift_block_size
relative_position = request_index % shift_block_size
```

对于两域 `A/B`，shifted 条件采用 alternating phase：

```text
block 0: A-major
block 1: B-major
block 2: A-major
block 3: B-major
...
```

如果 `shift_major_fraction = 0.8`，则：

```text
A-major block: 80% A + 20% B
B-major block: 20% A + 80% B
```

### 2.2 Intra-Request Mixing

`intra-request mixing` 指单个请求内部包含两个 domain 的文本片段。

形式上，当请求内部 domain distribution 的支持集大小大于 1 时：

```text
|{d : q_i(d) > 0}| >= 2
```

当前实现中，mixed request 由一个 primary domain prompt 和一个 secondary domain prompt 拼接或交错构成：

```text
mixed prompt = mix(primary_prompt, secondary_prompt)
```

支持两种 mixing 方式：

```text
--mix_mode concat
--mix_mode interleave
```

推荐主实验使用：

```text
--mix_mode interleave
--interleave_chunk_words 16
```

这样可以避免“前半段一个任务、后半段一个任务”过于块状，同时更强地破坏短程语义一致性。

## 3. 2x2 主实验矩阵

主实验包含四个条件：

| Condition | Cross-request shift | Intra-request mixing | 作用 |
|---|---:|---:|---|
| `stable_homogeneous` | No | No | 正常稳定单域请求流，作为 clean control |
| `shifted_homogeneous` | Yes | No | 只引入跨请求 domain shift |
| `stable_mixed` | No | Yes | 只引入请求内 cross-domain mixing |
| `shifted_mixed` | Yes | Yes | 同时引入 shift 与 mixing |

### 3.1 Stable Homogeneous

每个请求只包含一个 domain。

当前默认实现：

```text
condition = stable_homogeneous
stable_domain = translation

request 0: translation
request 1: translation
request 2: translation
...
```

如果开启：

```text
--match_domain_marginal
```

则 `stable_homogeneous` 会在 request 级交替 A/B，但每个请求仍然是 homogeneous：

```text
request 0: A
request 1: B
request 2: A
request 3: B
...
```

注意：如果论文要严格控制四个 cell 的全局 domain marginal，应开启 `--match_domain_marginal` 或者额外报告 domain marginal control。

### 3.2 Shifted Homogeneous

每个请求仍然只包含一个 domain，但 block 间 domain ratio 改变。

以 `shift_block_size = 32`、`shift_major_fraction = 0.8`、`domain_order = translation,summarization` 为例：

```text
block 0: first 26 requests translation, last 6 requests summarization
block 1: first 6 requests translation, last 26 requests summarization
block 2: first 26 requests translation, last 6 requests summarization
block 3: first 6 requests translation, last 26 requests summarization
```

这里 `26` 来自：

```text
round(32 * 0.8) = 26
```

该条件隔离的是 stream-level distribution shift，而不是请求内部混合。

### 3.3 Stable Mixed

每个请求都是 mixed request，但每个请求的 domain ratio 和 order 保持稳定。

以 `stable_mix_fraction = 0.5`、`domain_order = translation,summarization` 为例：

```text
request 0: mixed(translation 50%, summarization 50%)
request 1: mixed(translation 50%, summarization 50%)
request 2: mixed(translation 50%, summarization 50%)
...
```

如果使用 `interleave`：

```text
translation chunk 0
summarization chunk 0
translation chunk 1
summarization chunk 1
...
```

该条件隔离的是 intra-request mixing。

### 3.4 Shifted Mixed

每个请求都是 mixed request，并且 block 间 primary domain 和 mixing ratio 发生变化。

以 `shift_major_fraction = 0.8` 为例：

```text
block 0: mixed(translation 80%, summarization 20%)
block 1: mixed(summarization 80%, translation 20%)
block 2: mixed(translation 80%, summarization 20%)
block 3: mixed(summarization 80%, translation 20%)
```

该条件同时引入：

- 请求内部包含两个 domain。
- 请求流的主导 domain 随 block 改变。
- 在默认 `shifted_mixed` 中，domain ratio 和 domain order 都随 block 改变。

## 4. 当前代码中的构造逻辑

核心函数位于：

```text
scripts/eval_prefetch_shift.py
```

### 4.1 Domain A/B

由命令行参数确定：

```text
--domain_order translation,summarization
```

代码中解析为：

```text
domain_a = translation
domain_b = summarization
```

### 4.2 Homogeneous Request

对应函数：

```text
homogeneous_domain_for_request(...)
```

逻辑：

```text
if condition == shifted_homogeneous:
    if block_index is even:
        domain_a_fraction = shift_major_fraction
    else:
        domain_a_fraction = 1 - shift_major_fraction

    domain = choose_domain_by_fraction(
        domain_a,
        domain_b,
        relative_position,
        shift_block_size,
        domain_a_fraction
    )

elif match_domain_marginal:
    domain = domain_a if relative_position is even else domain_b

else:
    domain = stable_domain
```

### 4.3 Mixed Request

对应函数：

```text
mixed_plan_for_request(...)
```

主条件逻辑：

```text
stable_mixed:
    primary = domain_a
    secondary = domain_b
    primary_fraction = stable_mix_fraction
    shift_axis = none

shifted_mixed:
    if block_index is even:
        primary = domain_a
        secondary = domain_b
    else:
        primary = domain_b
        secondary = domain_a
    primary_fraction = shift_major_fraction
    shift_axis = domain_ratio_and_order
```

然后调用：

```text
mix_prompt(primary_prompt, secondary_prompt, args, primary_fraction)
```

文本预算由：

```text
primary_budget = int(mix_word_budget * primary_fraction)
secondary_budget = mix_word_budget - primary_budget
```

控制。

### 4.4 Workload Metadata

每个 request 会写入：

```text
request_trace.jsonl
```

其中包含：

```text
request_index
block_index
relative_position_in_block
is_shift_boundary
distance_to_nearest_boundary
domain_sequence
request_domain_distribution
stream_phase_distribution
request_entropy_bits
stream_entropy_bits
intra_request_mixing
cross_request_shift
shift_axis
```

这些字段用于 sanity check、分组分析和论文附录说明。

## 5. 推荐实验配置

### 5.1 主实验参数

推荐主实验使用：

```text
--conditions stable_homogeneous shifted_homogeneous stable_mixed shifted_mixed
--methods on_demand prefetch
--cache_ratios 0.03 0.1 0.4
--num_requests 128
--shift_block_size 32
--domain_order translation,summarization
--homogeneous_word_budget 256
--mix_word_budget 256
--stable_mix_fraction 0.5
--shift_major_fraction 0.8
--mix_mode interleave
--interleave_chunk_words 16
--beam_width 1
--max_seq_len 128
--sampling_topk 1
--sampling_topp 0.0
--moe_topk 1
--data_type fp32
--tensor_para_size 1
--pipeline_para_size 1
--cache_policy LFU
```

### 5.2 是否开启 `--match_domain_marginal`

分两种实验口径。

第一种：当前主实验口径。

```text
不加 --match_domain_marginal
```

优点：

- `stable_homogeneous` 是非常干净的单域稳定 baseline。
- 能展示从 ideal stable condition 到 mixed/shifted condition 的退化。

风险：

- 审稿人可能质疑四个 cell 的全局 domain marginal 不完全一致。

第二种：论文级 domain marginal control。

```text
加 --match_domain_marginal
```

优点：

- 更严格地控制全局 domain distribution。
- 更容易反驳“只是 summarization 多所以更慢”的质疑。

风险：

- `stable_homogeneous` 不再是单一 domain，而是 request 级交替 A/B；它仍然是 homogeneous request，但不再是 single-domain stream。

推荐论文策略：

- 主文中报告不带 `--match_domain_marginal` 的 clean baseline。
- 补充实验中报告带 `--match_domain_marginal` 的 control，证明结论不是 domain marginal artifact。

## 6. 标准运行命令

### 6.1 单 seed 正式 2x2

```bash
python scripts/run_prefetch_shift_formal.py \
  --results_root "$RUNS/main_2x2_seed0" \
  --seed 0 \
  --model_path "$MODEL_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --offload_path "$OFFLOAD_PATH" \
  --lib_path "$LIB_PATH" \
  --conditions stable_homogeneous shifted_homogeneous stable_mixed shifted_mixed \
  --methods on_demand prefetch \
  --cache_ratios 0.03 0.1 0.4 \
  --num_requests 128 \
  --shift_block_size 32 \
  --domain_order translation,summarization \
  --homogeneous_word_budget 256 \
  --mix_word_budget 256 \
  --stable_mix_fraction 0.5 \
  --shift_major_fraction 0.8 \
  --mix_mode interleave \
  --interleave_chunk_words 16 \
  --beam_width 1 \
  --max_seq_len 128 \
  --sampling_topk 1 \
  --sampling_topp 0.0 \
  --moe_topk 1 \
  --data_type fp32 \
  --tensor_para_size 1 \
  --pipeline_para_size 1 \
  --cache_policy LFU \
  --max_retries 3 \
  --skip_plots
```

### 6.2 多 seed 重复实验

```bash
python scripts/run_prefetch_shift_repeated.py \
  --results_root "$RUNS/main_repeat" \
  --seeds 0 1 2 3 4 \
  --model_path "$MODEL_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --offload_path "$OFFLOAD_PATH" \
  --lib_path "$LIB_PATH" \
  --conditions stable_homogeneous shifted_homogeneous stable_mixed shifted_mixed \
  --methods on_demand prefetch \
  --cache_ratios 0.03 0.1 0.4 \
  --num_requests 128 \
  --shift_block_size 32 \
  --domain_order translation,summarization \
  --homogeneous_word_budget 256 \
  --mix_word_budget 256 \
  --stable_mix_fraction 0.5 \
  --shift_major_fraction 0.8 \
  --mix_mode interleave \
  --interleave_chunk_words 16 \
  --beam_width 1 \
  --max_seq_len 128 \
  --sampling_topk 1 \
  --sampling_topp 0.0 \
  --moe_topk 1 \
  --data_type fp32 \
  --tensor_para_size 1 \
  --pipeline_para_size 1 \
  --cache_policy LFU \
  --max_retries 3 \
  --skip_plots
```

### 6.3 只生成 workload，不跑模型

用于快速检查 request 构造是否符合预期：

```bash
python scripts/eval_prefetch_shift.py \
  --output_dir "$RUNS/trace_only/stable_mixed" \
  --trace_id stable_mixed_trace_only \
  --condition stable_mixed \
  --method prefetch \
  --model_path "$MODEL_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --offload_path "$OFFLOAD_PATH" \
  --num_requests 128 \
  --shift_block_size 32 \
  --domain_order translation,summarization \
  --homogeneous_word_budget 256 \
  --mix_word_budget 256 \
  --stable_mix_fraction 0.5 \
  --shift_major_fraction 0.8 \
  --mix_mode interleave \
  --interleave_chunk_words 16 \
  --write_trace_only
```

检查输出：

```text
request_trace.jsonl
workload_summary.json
run_status.json
```

### 6.4 本仓库 Python runtime 的细粒度统计配置

如果使用当前 `mixtral-offloading` Python runtime，而不是 C++ runtime，需要在构建模型时启用 trace-only prefetch，并保存 `ExpertCache` 的统计：

```python
offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
    cache_policy="lru",
    prefetch_policy="next_layer_same_expert",
    prefetch_ttl_steps=1,
)
```

每个 cell 或每个 request batch 开始前清零：

```python
model.expert_cache.reset_stats()
```

运行结束后保存：

```python
summary = model.expert_cache.get_stats()
by_layer = model.expert_cache.get_group_stats()
model.expert_cache.save_stats("cache_prefetch_trace.json")
```

主实验输出至少打印：

```python
print(summary["demand_assignments"])
print(summary["cache_hit_assignment_ratio"])
print(summary["prefetch_assignment_utility_ratio"])
print(summary["prefetch_assignment_redundant_ratio"])
print(summary["prefetch_covered_assignment_ratio"])
```

其中 `prefetch_ttl_steps=1` 是 timely utility 口径；若设为 `None`，则结果是 eventual utility，不应与及时预取收益混用。

当前仓库提供的运行入口是：

```bash
python scripts/run_prefetch_2x2.py \
  --results_root "$RUNS/python_2x2" \
  --state_path "$STATE_PATH" \
  --quantized_model_name "$STATE_PATH" \
  --device cuda:0 \
  --conditions stable_homogeneous shifted_homogeneous stable_mixed shifted_mixed \
  --methods on_demand prefetch \
  --offload_per_layers 6 4 3 \
  --cache_policy lru \
  --prefetch_policy next_layer_same_expert \
  --prefetch_ttl_steps 1 \
  --num_requests 128 \
  --shift_block_size 32 \
  --domain_order translation,summarization \
  --homogeneous_word_budget 256 \
  --mix_word_budget 256 \
  --stable_mix_fraction 0.5 \
  --shift_major_fraction 0.8 \
  --mix_mode interleave \
  --interleave_chunk_words 16 \
  --max_new_tokens 128
```

汇总主表：

```bash
python scripts/analyze_prefetch_2x2.py \
  --results_root "$RUNS/python_2x2"
```

汇总结果会写入：

```text
formal_analysis/main_metrics.csv
formal_analysis/trajectory_consistency.csv
formal_analysis/analysis_summary.json
```

## 7. 输出文件结构

单个 cell 的输出目录：

```text
<results_root>/<method>/<condition>/cr<cache_ratio>/
```

例如：

```text
main_repeat/seed0/prefetch/shifted_mixed/cr0.1/
```

关键文件：

| 文件 | 含义 |
|---|---|
| `request_trace.jsonl` | workload 请求流定义 |
| `workload_summary.json` | workload 分布摘要 |
| `request_metrics.jsonl` | 每个 request 的 latency、token count、metadata |
| `cache_prefetch_trace.json` | Python runtime 侧 cache/prefetch 细粒度统计，包含 `summary`、`by_eviction_group`、`access_log`、`prefetch_log` |
| `prefetch_trace.tsv` | C++ runtime 侧 prefetch/actual/confusion trace |
| `confusion_boundary8.csv` | expert-set prediction confusion 分析 |
| `run_status.json` | 当前 cell 是否完成 |
| `cpp_config.ini` | 当前 cell 的 runtime/cache/fetcher 配置 |

单 seed 运行后还会生成：

```text
latency_compare.csv
formal_analysis/
validate.log
```

多 seed 运行后会生成：

```text
repeat_summary/
```

## 8. Workload Sanity Check

每次正式跑大实验前，应先检查 workload 本身。

### 8.1 检查 `workload_summary.json`

重点字段：

```text
condition
num_requests
domain_marginal
mean_request_entropy_bits
mixed_request_fraction
first_phase_distribution
second_phase_distribution
phase_js_divergence_bits
shift_axes
```

预期：

| Condition | mixed_request_fraction | phase_js_divergence_bits | mean_request_entropy_bits |
|---|---:|---:|---:|
| `stable_homogeneous` | 0 | 0 | 0 |
| `shifted_homogeneous` | 0 | > 0 | 0 |
| `stable_mixed` | 1 | 0 | > 0 |
| `shifted_mixed` | 1 | > 0 | > 0 |

### 8.2 检查长度

从 `request_metrics.jsonl` 检查：

```text
input_token_count
output_token_count
```

至少报告：

```text
mean / std / p50 / p95
```

如果 mixed 条件显著更长，必须补长度控制实验，否则 slowdown 容易被质疑为 length artifact。

### 8.3 检查 domain marginal

如果主实验不使用 `--match_domain_marginal`，应在文中诚实说明：

```text
stable_homogeneous is a clean single-domain positive control.
```

如果要反驳 domain marginal confound，应补充带 `--match_domain_marginal` 的 control。

## 9. 分析指标

### 9.1 系统结果

主指标：

```text
latency_ratio = mean_latency(prefetch) / mean_latency(on_demand)
prefetch_over_on_demand = latency(prefetch) / latency(on_demand)
```

解释：

- `< 1`：prefetch 带来加速。
- `= 1`：无收益。
- `> 1`：prefetch 变慢。

建议报告：

```text
mean latency
p50 latency
p95 latency
prefetch / on-demand ratio
95% CI across seeds
```

`prefetch_over_on_demand` 是最终系统结果，但它不是机制解释。只有在 `trajectory_consistency` 通过后，才能把它解释为 prefetch/on-demand 机制差异，而不是不同生成轨迹、输出长度或 expert 路由差异造成的假象。

### 9.2 Fairness Gate：Trajectory Consistency

端到端 latency 对比前必须先检查公平性：

```text
trajectory_consistency
```

它回答：

```text
prefetch 与 on_demand 是否处理了同一条或足够接近的 token/expert 访问轨迹？
```

最低要求：

```text
same prompts
same decoding config
do_sample=False 或固定 seed
same max_new_tokens
input_token_count / output_token_count 接近或一致
```

更严格要求：

```text
generated token trajectory 一致
expert access trace 一致
```

建议在 `request_metrics.jsonl` 或 cell summary 中报告：

```text
input_token_count_mean / p95
output_token_count_mean / p95
output_token_count_delta_ratio
expert_trace_match_ratio
```

如果无法 replay 完全相同的 expert trace，至少要报告长度和 decoding 配置一致性，并在论文中说明该结果是 workload-level 对比而非严格 replay 对比。

### 9.3 记录粒度定义

本实验不再以粗粒度的 layer-level 或单纯 prefetch event 作为主分析口径。为了和 MoE 路由实际计算负载对齐，主指标采用 **token-expert assignment 加权粒度**。

三种粒度的关系如下：

```text
layer-level:
  一层 forward 算一次。

request-level:
  当前层实际访问到的每个 unique expert 算一次。
  单位是 (layer_id, expert_id)。

assignment-level:
  每个 token 的每个 top-k expert assignment 算一次。
  单位是 (layer_id, token_position, topk_slot, expert_id)。
```

对于 Mixtral top-2 routing，若一层有 `N` 个 token，则：

```text
demand_assignments = N * 2
```

如果 100 个 token assignment 都路由到同一个 expert，那么：

```text
request-level 只记 1 次 expert request
assignment-level 记 100 次 token-expert assignment
```

当前实现为了控制 trace 体积，不逐条保存每个 token assignment，而是在每个 expert request 上聚合记录：

```text
assignment_count:
  当前 (layer_id, expert_id) 被多少个 token-expert assignment 使用。
```

因此准确表述应为：

```text
记录形式是 expert-request 聚合；
统计权重是 token-expert assignment。
```

### 9.4 Expert Prediction Accuracy

来自：

```text
confusion_boundary8.csv
```

主指标：

```text
issue_f1_global
```

定义：

```text
issue_precision_global =
  sum_i |P_i ∩ A_i| / sum_i |P_i|

issue_recall_global =
  sum_i |P_i ∩ A_i| / sum_i |A_i|

issue_f1_global =
  2 * issue_precision_global * issue_recall_global
  / (issue_precision_global + issue_recall_global)
```

其中：

```text
P_i = 第 i 次 prefetch issue 预测的 expert set
A_i = TTL 窗口内真实访问的 expert set
```

它回答：

```text
prefetch predictor 本身是否具备非平凡预测能力？
```

辅助指标：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2TP / (2TP + FP + FN)
```

它回答：

```text
prefetch predictor 是否预测到了当前 issue 对应的真实 expert set？
```

如果使用本仓库 Python runtime 的 trace-only prefetch 统计，则预测准确性还应按 assignment 加权补充报告：

```text
issue_f1_assignment
prefetch_assignment_use_ratio =
  prefetch_assignment_used / prefetch_assignment_requests
```

注意它是 eventual-use 指标；如果设置 `prefetch_ttl_steps=1`，则更接近 one-step timely prediction。

如果 `issue_f1_global` 很低，则低 prefetch utility 很可能只是 predictor 本身失败，不能强行解释为“高 cache hit 压低 prefetch utility”。只有当 `issue_f1_global` 显示 predictor 有一定预测能力时，后续的 cache redundancy 与 timely utility 分析才构成更有力的机制解释。

### 9.5 Cache Hit 与 Prefetch Utility

#### 9.5.1 主口径：assignment-level

论文主文中比较 cache hit 与 prefetch utility 时，优先使用 assignment-level 指标：

```text
cache_hit_assignment_ratio =
  cache_hit_assignments / demand_assignments

prefetch_assignment_utility_ratio =
  prefetch_assignment_issued_used / prefetch_assignment_issued

prefetch_assignment_redundant_ratio =
  prefetch_assignment_redundant / prefetch_assignment_requests

prefetch_covered_assignment_ratio =
  prefetch_covered_assignments / demand_assignments
```

字段含义：

```text
demand_assignments:
  真实 token-expert assignment 总数。

cache_hit_assignments:
  由 GPU cache hit 覆盖的 token-expert assignment 数。

prefetch_assignment_requests:
  所有预取候选的 assignment 加权总数。

prefetch_assignment_redundant:
  发出预取候选时，对应 expert 已经在 GPU cache 中的 assignment 加权数。

prefetch_assignment_issued:
  非冗余预取候选的 assignment 加权数。

prefetch_assignment_issued_used:
  非冗余预取候选中，后来在有效 TTL 窗口内被真实访问消费的 assignment 加权数。
```

对于 assignment-weighted prefetch utility，当前实现采用：

```text
used_assignment_count =
  min(predicted_assignment_count, actual_assignment_count)
```

这样可以避免某个 expert 预测数量偏低却因为实际使用很多而过度放大 utility。

#### 9.5.2 辅助口径：request-level

request-level 指标仍然保留，用于说明专家搬运事件本身：

```text
cache_hit_ratio =
  cache_hits / demand_accesses

prefetch_utility_ratio =
  prefetch_issued_used / prefetch_issued

prefetch_redundant_ratio =
  prefetch_redundant / prefetch_requests
```

其中：

```text
demand_accesses:
  unique expert request 数，即 (layer_id, expert_id) 数。

prefetch_issued:
  非冗余 unique expert prefetch 候选数。
```

request-level 更接近权重搬运粒度，但会隐藏一个 expert 服务了多少 token。因此它只作为辅助对照，不作为本文判断 workload 对 token 计算影响的主口径。

#### 9.5.3 Timely Utility 与 TTL

如果：

```python
prefetch_ttl_steps=None
```

则 `prefetch_*_utility_ratio` 表示 eventual utility，即未来任意时间被用到都算 useful。这容易高估真实预取价值。

如果：

```python
prefetch_ttl_steps=1
```

则 utility 更接近 one-step timely utility，即预取候选必须在下一次 cache demand step 中被消费才算 useful。

本文主实验建议使用：

```text
prefetch_ttl_steps = 1
```

并优先报告：

```text
cache_hit_assignment_ratio
prefetch_assignment_utility_ratio
prefetch_assignment_redundant_ratio
prefetch_covered_assignment_ratio
```

#### 9.5.4 与旧 event-level 指标的关系

如果底层 runtime 仍输出 `PREFETCH_EXPERT` event，可以保留以下 event-level 指标作为补充：

```text
broad_utility_ratio =
  useful prefetch expert events / all prefetch expert events

timely_utility_ratio =
  useful and ready-before-consume prefetch expert events / all prefetch expert events

prefetch_event_cache_hit_ratio =
  prefetch expert events whose expert is already in cache at issue time
  / all prefetch expert events
```

但它们不再作为主文主指标。主文应说明：

```text
Primary cache/prefetch comparison is performed at token-expert assignment granularity.
Request-level and event-level metrics are used only as auxiliary diagnostics.
```

### 9.6 Incremental Utility

当前 Python trace 可以将 `cache_resident_at_issue` 作为 issue-time cache residency 使用，因此可以分析冗余预取和部分 incremental utility：

```text
prefetch_assignment_redundant_ratio
prefetch_assignment_utility_ratio
```

但它仍不等价于完整的 latency-hiding 边际贡献，因为 trace-only prefetch 不真的搬运权重，也没有记录：

```text
inflight_at_issue
prefetch_start_time
prefetch_ready_time
consume_time
actual transfer latency
```

因此论文中应避免声称“prefetch 已经带来/没有带来完整系统加速”，除非后续补充真实异步预取和完整 runtime attribution trace。

### 9.7 最终主表字段

主文主表建议统一使用以下字段：

```text
condition
offload_per_layer
GPU_experts_per_layer
trajectory_consistency
issue_f1_global
cache_hit_assignment_ratio
prefetch_assignment_redundant_ratio
prefetch_assignment_utility_ratio
prefetch_covered_assignment_ratio
prefetch_over_on_demand
```

其中：

```text
GPU_experts_per_layer = num_local_experts - offload_per_layer
```

如果版面允许，可以在补充列中加入：

```text
demand_assignments
prefetch_assignment_requests
prefetch_assignment_issued
cache_hit_ratio
prefetch_utility_ratio
prefetch_redundant_ratio
```

最终故事线应按如下顺序解释：

1. `trajectory_consistency` 说明 latency 对比公平。
2. `issue_f1_global` 排除“策略完全不准”的解释。
3. `cache_hit_assignment_ratio` 证明 cache residency 是否足够高。
4. `prefetch_assignment_redundant_ratio` 解释有多少预取已经被 cache 覆盖。
5. `prefetch_assignment_utility_ratio` 说明短窗口及时效用是否低。
6. `prefetch_over_on_demand` 给出最终系统收益。

## 10. 2x2 Factorial Effect

对每个 cache ratio，可计算 shift、mixing 以及交互项。

设：

```text
Y00 = latency_ratio(stable_homogeneous)
Y10 = latency_ratio(shifted_homogeneous)
Y01 = latency_ratio(stable_mixed)
Y11 = latency_ratio(shifted_mixed)
```

则：

```text
shift_effect  = Y10 - Y00
mixing_effect = Y01 - Y00
interaction   = Y11 - Y10 - Y01 + Y00
```

解释规则：

- 如果 `interaction > 0` 且 CI 不跨 0，可以说 shift 和 mixing 有超加性交互。
- 如果 `interaction` 不显著，不能说两者发生强交互爆炸。
- 即使没有正交互，也仍可说 `shifted_mixed` 是一个 combined stress condition，但应避免夸大。

## 11. Ratio/Order Ablation

为了避免 reviewer 质疑 `shifted_mixed` 只是某种人为 order artifact，代码还支持以下 ablation：

| Condition | 目的 |
|---|---|
| `stable_mixed_ab` | 稳定 AB 顺序 |
| `stable_mixed_ba` | 稳定 BA 顺序 |
| `stable_mixed_balanced_order` | AB/BA 交替，控制 order marginal |
| `shifted_mixed_ratio_only` | 只改变 A/B ratio，不改变 order |
| `shifted_mixed_order_only` | 只改变 AB/BA order，不改变 ratio |
| `shifted_mixed_ratio_and_order` | 同时改变 ratio 和 order |

推荐命令：

```bash
python scripts/run_prefetch_shift_formal.py \
  --results_root "$RUNS/ratio_order_ablation" \
  --seed 0 \
  --model_path "$MODEL_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --offload_path "$OFFLOAD_PATH" \
  --lib_path "$LIB_PATH" \
  --conditions stable_mixed_ab stable_mixed_ba stable_mixed_balanced_order shifted_mixed_ratio_only shifted_mixed_order_only shifted_mixed_ratio_and_order \
  --methods on_demand prefetch \
  --cache_ratios 0.03 0.1 0.4 \
  --num_requests 128 \
  --shift_block_size 32 \
  --domain_order translation,summarization \
  --homogeneous_word_budget 256 \
  --mix_word_budget 256 \
  --stable_mix_fraction 0.5 \
  --shift_major_fraction 0.8 \
  --mix_mode interleave \
  --interleave_chunk_words 16 \
  --beam_width 1 \
  --max_seq_len 128 \
  --sampling_topk 1 \
  --sampling_topp 0.0 \
  --moe_topk 1 \
  --data_type fp32 \
  --tensor_para_size 1 \
  --pipeline_para_size 1 \
  --cache_policy LFU \
  --max_retries 3 \
  --skip_plots
```

## 12. 必须补充的控制实验

### 12.1 长度控制

目的：

```text
排除 mixed/shifted 条件更慢只是因为 prompt 更长或 output 更长。
```

需要记录并比较：

```text
input_token_count
output_token_count
total_token_count
```

通过标准：

```text
长度对齐后，主要趋势仍然存在。
```

### 12.2 Domain Marginal Control

目的：

```text
排除某个 condition 只是因为包含更多 summarization 或更难 domain。
```

建议：

```text
加 --match_domain_marginal 重跑 2x2。
```

### 12.3 Order Control

目的：

```text
排除 shifted_mixed 只是 AB/BA 顺序变化 artifact。
```

使用 ratio/order ablation。

### 12.4 Mixing Continuum

目的：

```text
证明 failure 不是二元开关，而是随 mixing 强度连续变化。
```

建议 sweep：

```text
stable_mix_fraction or primary/secondary ratio:
0.0, 0.25, 0.5, 0.75, 1.0
```

### 12.5 Drift Continuum

目的：

```text
证明 failure 随 cross-request drift 强度变化。
```

建议 sweep：

```text
shift_major_fraction:
0.5, 0.6, 0.7, 0.8, 0.9
```

## 13. 论文图表设计

### 13.1 主图

推荐主图结构：

```text
Panel A: 2x2 workload schematic
Panel B: trajectory consistency and prefetch_over_on_demand
Panel C: issue_f1_global vs timely assignment-level utility
Panel D: cache_hit_assignment_ratio vs prefetch_assignment_redundant_ratio
Panel E: per-layer assignment-level utility or predictor F1 breakdown
```

### 13.2 必须避免的图表误导

不要把 request-level `avg_cache_hit_rate` 和 event-level `prefetch_utility_ratio` 直接画在一起，然后宣称已严格对齐。

主图中应优先使用 assignment-level 指标：

```text
issue_f1_global
cache_hit_assignment_ratio
prefetch_assignment_utility_ratio
prefetch_assignment_redundant_ratio
prefetch_covered_assignment_ratio
prefetch_over_on_demand
```

request-level 指标只放在补充材料或 ablation 表中：

```text
cache_hit_ratio
prefetch_utility_ratio
prefetch_redundant_ratio
```

更安全的图名：

```text
High assignment-level cache residency, low timely prefetch contribution
```

而不是：

```text
High hit, low utility
```

除非正文明确说明二者的分母和粒度。

推荐图注写法：

```text
Cache hit and prefetch utility are both reported at token-expert assignment granularity. Each expert request is weighted by the number of top-k token assignments routed to that expert.
```

若图中包含 latency ratio，还应在图注中说明：

```text
End-to-end latency ratios are interpreted only for runs that pass the trajectory consistency check.
```

## 14. 论文写法模板

### 14.1 英文实验设计描述

```text
To isolate the workload properties that stress decoder expert prefetching, we construct a controlled 2x2 workload framework with two factors: cross-request workload shift and intra-request cross-domain mixing. Cross-request shift changes the dominant domain distribution across fixed-size request blocks, while intra-request mixing composes each request from two domain segments under a controlled word budget. This design yields four workload cells: stable homogeneous, shifted homogeneous, stable mixed, and shifted mixed. By keeping model, cache budget, decoding configuration, cache policy, and request count fixed across cells, the framework lets us attribute changes in latency and prefetch utility to stream-level non-stationarity, request-internal semantic mixing, or their combined effect.
```

### 14.2 中文实验设计描述

```text
为了隔离影响 decoder expert prefetch 的 workload 因素，我们设计了一个受控的 2x2 workload 框架。该框架包含两个因子：跨请求 workload shift 和请求内 cross-domain mixing。前者通过固定大小的请求 block 改变主导 domain 分布，后者通过在单个请求内部按受控 word budget 混合两个 domain 的文本片段实现。由此得到 stable homogeneous、shifted homogeneous、stable mixed 和 shifted mixed 四个条件。在模型、cache budget、解码参数、cache policy 和请求数量保持一致的前提下，该框架可以将 latency 与 prefetch utility 的变化归因到 stream-level 非平稳性、request-internal 语义混合，或二者的组合压力。
```

### 14.3 保守结论模板

```text
The 2x2 workload does not by itself prove that prefetch is accurate but unused. Instead, it reveals a more specific mismatch: high cache residency can coexist with low matched and low incremental timely prefetch contribution. This suggests that cache-level success metrics are insufficient to characterize the effectiveness of decoder expert prefetching under non-stationary mixed workloads.
```

### 14.4 最终结论模板

```text
After verifying trajectory consistency, we first measure issue-level predictor quality to rule out the trivial explanation that the prefetcher is simply random. The predictor shows non-trivial expert-set accuracy, yet under larger GPU expert caches, assignment-level cache hit ratio increases while redundant prefetch ratio also rises. With a one-step validity window, assignment-level timely prefetch utility remains low, indicating that many predicted experts are either already cache-resident or not consumed within the useful latency-hiding window. The end-to-end prefetch/on-demand latency ratio then quantifies how this limited timely utility translates into system-level speedup.
```

中文对应表述：

```text
在确认 prefetch 与 on-demand 的生成轨迹具有可比性后，我们先报告 issue-level predictor accuracy，排除“预取器完全乱猜”的简单解释。结果显示 predictor 具有一定 expert-set 预测能力；但随着 GPU expert cache 增大，assignment-level cache hit ratio 上升的同时，冗余预取比例也上升。在 TTL=1 的及时窗口内，assignment-level prefetch utility 仍然较低，说明许多预测到的 expert 要么已经在 cache 中，要么没有在能隐藏延迟的窗口内被消费。最终通过 prefetch_over_on_demand 量化这种低及时效用是否转化为有限的系统加速。
```

## 15. 实现检查清单

正式跑实验前：

- 确认 `request_trace.jsonl` 中四个 condition 的 `intra_request_mixing` 与 `cross_request_shift` 符合预期。
- 确认 `workload_summary.json` 中 `mixed_request_fraction` 与 `phase_js_divergence_bits` 符合 2x2 定义。
- 确认 `input_token_count` 和 `output_token_count` 没有严重偏差。
- 确认 `on_demand` 与 `prefetch` 除 fetcher mode 外配置一致。
- 确认 `cache_ratio`、`cache_policy`、`moe_topk`、`max_seq_len` 固定。
- 确认 `trajectory_consistency` 可报告：至少包含 decoding 配置一致性、输入/输出 token 数统计；最好包含 generated token 或 expert access trace match。
- 确认可计算 `issue_f1_global`，并明确 `P_i` 与 `A_i` 的 TTL 窗口定义。
- 确认 `cache_prefetch_trace.json` 中存在 assignment-level 字段：
  `demand_assignments`、`cache_hit_assignment_ratio`、`prefetch_assignment_utility_ratio`、`prefetch_assignment_redundant_ratio`。
- 确认 `access_log` 中每条记录包含 `assignment_count`，`prefetch_log` 中每条记录包含 `predicted_assignment_count` 和 `used_assignment_count`。
- 主实验若要分析 timely utility，应确认 `prefetch_ttl_steps=1`；若使用 `None`，必须在文中称为 eventual utility。
- 确认可报告 `prefetch_over_on_demand`，且只在 trajectory consistency 通过时解释为系统收益。
- 确认多 seed 不是完全复制同一 workload。如果 seed 不影响当前构造，应在文中说明 seed 主要反映 runtime repetition，而不是 workload resampling。
- 使用复合键关联 `CONFUSION` 与 `PREFETCH_EXPERT`：`trace_id + request_id + step_id + source_layer + target_layer + prefetch_issue_id`。

正式写论文前：

- 主结果必须有 mean/std/95% CI。
- 不要声称有强交互，除非 factorial interaction 的 CI 支持。
- 不要声称 prefetch 准确，除非 `issue_f1_global` 或 precision/recall/F1 支持。
- 不要把 low prefetch utility 直接归因于 cache redundancy，除非同时报告 predictor accuracy 与 redundant ratio。
- 不要用 request-level hit ratio 直接对比 assignment-level prefetch utility。主文统一使用 assignment-level，request-level 作为辅助诊断。
- 不要声称 prefetch 的完整边际贡献，除非 trace 记录了完整 issue/ready/consume/cache/inflight 状态。
- 将 2x2 framework 定位为 workload diagnosis framework，而不是最终方法本身。

## 16. 推荐后续实现扩展

为了让该实验更接近 CCF A 论文标准，建议补充：

1. 增强 runtime trace，记录 `cached_at_issue`、`inflight_at_issue`、`prefetch_start_time`、`prefetch_ready_time`、`consume_time`、`actual_expert_access`。
2. 增加 `--ordering_mode`，支持 block、shuffle、round-robin、gradual drift、bursty drift。
3. 增加 length-matched prompt sampler，按 token bucket 对齐 homogeneous 与 mixed 条件。
4. 增加 mixing intensity sweep 与 drift intensity sweep。
5. 增加 semi-real traffic bridge，用公开多任务数据构造更自然的请求流。
6. 将 2x2 diagnosis 与新的 utility-aware/prefetch policy 方法连接，形成“发现问题 + 机制解释 + 系统改进”的完整论文线。
