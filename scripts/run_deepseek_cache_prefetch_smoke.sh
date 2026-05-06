#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT="${OUT:-/root/autodl-tmp/deepseek_cache_prefetch_smoke}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
RUN_MODEL="${RUN_MODEL:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

mkdir -p "$OUT" "$OUT/normal" "$OUT/2x2"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf_home/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_home/transformers}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip_cache}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" "$PIP_CACHE_DIR"

cd "$ROOT"

python scripts/deepseek_cache_prefetch_experiment.py env-check \
  --output "$OUT/env_check.json"

python scripts/deepseek_cache_prefetch_experiment.py make-prompts \
  --suite normal \
  --num-requests 4 \
  --output "$OUT/normal/prompts.jsonl"

python scripts/deepseek_cache_prefetch_experiment.py make-prompts \
  --suite 2x2 \
  --num-requests-per-cell 2 \
  --output "$OUT/2x2/prompts.jsonl"

python scripts/deepseek_cache_prefetch_experiment.py workload-sanity \
  --prompts "$OUT/2x2/prompts.jsonl" \
  --output "$OUT/2x2/workload_sanity.csv"

python scripts/deepseek_cache_prefetch_experiment.py make-synthetic-trace \
  --output "$OUT/synthetic_trace.jsonl" \
  --requests-per-condition 2 \
  --steps 4 \
  --layers 3 \
  --experts 6

python scripts/deepseek_cache_prefetch_experiment.py replay \
  --trace "$OUT/synthetic_trace.jsonl" \
  --output "$OUT/synthetic_policy_summary.csv" \
  --factorial-output "$OUT/synthetic_factorial_effects.csv" \
  --cache-ratios 0.10,0.20

if [[ "$RUN_MODEL" == "1" ]]; then
  extra_args=()
  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    extra_args+=(--local-files-only)
  fi

  python scripts/deepseek_cache_prefetch_experiment.py collect-trace \
    --prompts "$OUT/normal/prompts.jsonl" \
    --output "$OUT/normal/request_trace.jsonl" \
    --meta-output "$OUT/normal/trace_meta.json" \
    --request-summary-output "$OUT/normal/request_summary.jsonl" \
    --model "$MODEL" \
    --limit 2 \
    --max-new-tokens 16 \
    --max-prompt-tokens 256 \
    --quantization 4bit \
    --max-gpu-memory 14GiB \
    --max-cpu-memory 42GiB \
    "${extra_args[@]}"

  python scripts/deepseek_cache_prefetch_experiment.py replay \
    --trace "$OUT/normal/request_trace.jsonl" \
    --output "$OUT/normal/policy_summary.csv" \
    --cache-ratios 0.10
fi

echo "smoke outputs: $OUT"
