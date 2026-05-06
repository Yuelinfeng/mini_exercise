# DeepSeek-V2-Lite Cache/Prefetch 实验代码说明

本目录实现 `deepseek_v2_lite_cache_prefetch_experiment_plan_zh.md` 中的实验计划。代码默认面向 AutoDL 远程环境：

```text
CUDA 12.4 + Ubuntu 22.04 + Python 3.12 + Torch 2.5.1
GPU: Tesla T4 15GB
CPU memory: 47GB
data disk: /root/autodl-tmp
apt: unavailable
```

## 1. 文件结构

| 文件 | 作用 |
|---|---|
| `workloads.py` | 生成 normal 与 2x2 workload prompts |
| `collect_trace.py` | 加载 DeepSeek-V2-Lite 并通过 MoE gate hook 采集真实 expert assignment trace |
| `replay.py` | 在固定 trace 上回放 `OnDemand`、`LRU_only`、`LFU_only`、`Prefetch_only`、`LRU_plus_prefetch` |
| `run.py` | CLI 子命令入口 |
| `scripts/deepseek_cache_prefetch_experiment.py` | 仓库根目录下的可执行入口 |
| `scripts/run_deepseek_cache_prefetch_smoke.sh` | AutoDL smoke test 脚本 |

## 2. AutoDL 环境准备

在 AutoDL 终端中进入代码目录后执行：

```bash
cd /root/autodl-tmp/Reccurence_paper

export HF_HOME=/root/autodl-tmp/hf_home
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_home/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_home/transformers
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" "$PIP_CACHE_DIR"

conda create -n moe-exp python=3.12 -y
conda activate moe-exp
pip install -r requirements-deepseek-cache-prefetch.txt
```

如果镜像已经有可用的 torch 2.5.1 CUDA 环境，不要强行重装 torch。先跑：

```bash
python scripts/deepseek_cache_prefetch_experiment.py env-check \
  --output /root/autodl-tmp/deepseek_cache_prefetch/env_check.json
```

## 3. 本地逻辑 smoke test

不加载模型，只验证 prompt generation 与 replay：

```bash
bash scripts/run_deepseek_cache_prefetch_smoke.sh
```

输出目录默认：

```text
/root/autodl-tmp/deepseek_cache_prefetch_smoke
```

## 4. 真实模型 smoke test

先只跑极小规模：

```bash
RUN_MODEL=1 bash scripts/run_deepseek_cache_prefetch_smoke.sh
```

默认模型：

```text
deepseek-ai/DeepSeek-V2-Lite
```

如果模型已经下载到本地，可以使用：

```bash
MODEL=/root/autodl-tmp/models/DeepSeek-V2-Lite \
LOCAL_FILES_ONLY=1 \
RUN_MODEL=1 \
bash scripts/run_deepseek_cache_prefetch_smoke.sh
```

## 5. 主实验命令

生成 normal prompts：

```bash
python scripts/deepseek_cache_prefetch_experiment.py make-prompts \
  --suite normal \
  --num-requests 64 \
  --output /root/autodl-tmp/deepseek_cache_prefetch/normal/prompts.jsonl
```

生成 2x2 prompts：

```bash
python scripts/deepseek_cache_prefetch_experiment.py make-prompts \
  --suite 2x2 \
  --num-requests-per-cell 64 \
  --shift-block-size 16 \
  --shift-major-fraction 0.8 \
  --stable-mix-fraction 0.5 \
  --mix-mode interleave \
  --interleave-chunk-words 16 \
  --output /root/autodl-tmp/deepseek_cache_prefetch/2x2/prompts.jsonl
```

检查 2x2 workload 构造：

```bash
python scripts/deepseek_cache_prefetch_experiment.py workload-sanity \
  --prompts /root/autodl-tmp/deepseek_cache_prefetch/2x2/prompts.jsonl \
  --output /root/autodl-tmp/deepseek_cache_prefetch/2x2/workload_sanity.csv
```

采集 normal trace：

```bash
python scripts/deepseek_cache_prefetch_experiment.py collect-trace \
  --prompts /root/autodl-tmp/deepseek_cache_prefetch/normal/prompts.jsonl \
  --output /root/autodl-tmp/deepseek_cache_prefetch/normal/request_trace.jsonl \
  --meta-output /root/autodl-tmp/deepseek_cache_prefetch/normal/trace_meta.json \
  --request-summary-output /root/autodl-tmp/deepseek_cache_prefetch/normal/request_summary.jsonl \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --quantization 4bit \
  --max-gpu-memory 14GiB \
  --max-cpu-memory 42GiB \
  --max-prompt-tokens 512 \
  --max-new-tokens 64
```

采集 2x2 trace：

```bash
python scripts/deepseek_cache_prefetch_experiment.py collect-trace \
  --prompts /root/autodl-tmp/deepseek_cache_prefetch/2x2/prompts.jsonl \
  --output /root/autodl-tmp/deepseek_cache_prefetch/2x2/request_trace.jsonl \
  --meta-output /root/autodl-tmp/deepseek_cache_prefetch/2x2/trace_meta.json \
  --request-summary-output /root/autodl-tmp/deepseek_cache_prefetch/2x2/request_summary.jsonl \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --quantization 4bit \
  --max-gpu-memory 14GiB \
  --max-cpu-memory 42GiB \
  --max-prompt-tokens 512 \
  --max-new-tokens 64
```

回放 normal 策略：

```bash
python scripts/deepseek_cache_prefetch_experiment.py replay \
  --trace /root/autodl-tmp/deepseek_cache_prefetch/normal/request_trace.jsonl \
  --output /root/autodl-tmp/deepseek_cache_prefetch/normal/policy_summary.csv \
  --cache-ratios 0.05,0.10,0.20
```

回放 2x2 策略：

```bash
python scripts/deepseek_cache_prefetch_experiment.py replay \
  --trace /root/autodl-tmp/deepseek_cache_prefetch/2x2/request_trace.jsonl \
  --output /root/autodl-tmp/deepseek_cache_prefetch/2x2/policy_summary.csv \
  --factorial-output /root/autodl-tmp/deepseek_cache_prefetch/2x2/factorial_effects.csv \
  --cache-ratios 0.05,0.10,0.20
```

## 6. 主输出指标

`policy_summary.csv` 固定包含：

```text
TPOT proxy:
  estimated_tpot_ms
  tokens_per_sec_est

Common metrics:
  cache_hit_assignment_ratio
  cache_miss_assignment_ratio
  cpu_gpu_transfer_bytes
  prefetch_accuracy
  prefetch_coverage

Utility metrics:
  prefetch_redundant_ratio
  timely_incremental_utility_ratio
  stall_saved_ratio
```

关键判读链：

```text
cache_hit_assignment_ratio 高
prefetch_redundant_ratio 高
timely_incremental_utility_ratio 低
stall_saved_ratio 没有同步提升
```

这说明高 cache residency 正在掩盖低 prefetch utility。

## 7. 当前实现边界

当前实现已经完成：

| 模块 | 状态 |
|---|---|
| prompt generation | 完成 |
| 2x2 workload generation | 完成 |
| DeepSeek MoE gate hook trace collection | 完成，需 AutoDL 实机验证 |
| fixed-trace policy replay | 完成 |
| core metrics | 完成 |
| synthetic trace self-test | 完成 |

当前实现没有做：

```text
真实异步 expert transfer 调度修改
真实 cache eviction 对模型权重加载路径的侵入式改写
完整端到端系统吞吐优化
```

因此第一阶段结果应表述为：

```text
真实 DeepSeek-V2-Lite routing trace 上的 cache/prefetch policy replay 诊断实验。
```
