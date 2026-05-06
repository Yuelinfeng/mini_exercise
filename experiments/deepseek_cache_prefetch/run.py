from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from .collect_trace import collect_trace
from .io_utils import write_json
from .replay import make_synthetic_trace, replay_trace
from .workloads import write_prompts, write_workload_sanity


def cmd_env_check(args: Any) -> None:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
    except Exception as exc:
        info["torch_error"] = repr(exc)
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception as exc:
        info["transformers_error"] = repr(exc)
    try:
        import bitsandbytes as bnb

        info["bitsandbytes"] = getattr(bnb, "__version__", "unknown")
    except Exception as exc:
        info["bitsandbytes_error"] = repr(exc)

    if args.output:
        write_json(args.output, info)
    print(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_make_prompts(args: Any) -> None:
    count = write_prompts(
        output=args.output,
        suite=args.suite,
        num_requests=args.num_requests,
        num_requests_per_cell=args.num_requests_per_cell,
        seed=args.seed,
        shift_block_size=args.shift_block_size,
        shift_major_fraction=args.shift_major_fraction,
        stable_mix_fraction=args.stable_mix_fraction,
        mix_mode=args.mix_mode,
        interleave_chunk_words=args.interleave_chunk_words,
    )
    print(f"wrote prompts: {args.output} records={count}")


def cmd_workload_sanity(args: Any) -> None:
    count = write_workload_sanity(args.prompts, args.output)
    print(f"wrote workload sanity: {args.output} rows={count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepseek_cache_prefetch_experiment",
        description="DeepSeek-V2-Lite cache/prefetch trace and replay experiment.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env-check", help="Print AutoDL/Python/Torch environment information.")
    env.add_argument("--output", default=None)
    env.set_defaults(func=cmd_env_check)

    prompts = sub.add_parser("make-prompts", help="Generate normal or 2x2 prompt JSONL.")
    prompts.add_argument("--suite", choices=["normal", "2x2"], required=True)
    prompts.add_argument("--output", required=True)
    prompts.add_argument("--num-requests", type=int, default=64)
    prompts.add_argument("--num-requests-per-cell", type=int, default=64)
    prompts.add_argument("--seed", type=int, default=0)
    prompts.add_argument("--shift-block-size", type=int, default=16)
    prompts.add_argument("--shift-major-fraction", type=float, default=0.8)
    prompts.add_argument("--stable-mix-fraction", type=float, default=0.5)
    prompts.add_argument("--mix-mode", choices=["concat", "interleave"], default="interleave")
    prompts.add_argument("--interleave-chunk-words", type=int, default=16)
    prompts.set_defaults(func=cmd_make_prompts)

    sanity = sub.add_parser("workload-sanity", help="Summarize prompt/workload composition.")
    sanity.add_argument("--prompts", required=True)
    sanity.add_argument("--output", required=True)
    sanity.set_defaults(func=cmd_workload_sanity)

    trace = sub.add_parser("collect-trace", help="Collect real MoE routing trace from DeepSeek-V2-Lite.")
    trace.add_argument("--prompts", required=True)
    trace.add_argument("--output", required=True)
    trace.add_argument("--meta-output", default=None)
    trace.add_argument("--request-summary-output", default=None)
    trace.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    trace.add_argument("--max-new-tokens", type=int, default=64)
    trace.add_argument("--max-prompt-tokens", type=int, default=512)
    trace.add_argument("--limit", type=int, default=None)
    trace.add_argument("--include-prefill", action="store_true")
    trace.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    trace.add_argument("--torch-dtype", choices=["auto", "float16"], default="float16")
    trace.add_argument("--device-map", default="auto")
    trace.add_argument("--max-gpu-memory", default="14GiB")
    trace.add_argument("--max-cpu-memory", default="42GiB")
    trace.add_argument("--local-files-only", action="store_true")
    trace.add_argument("--trust-remote-code", action="store_true", default=True)
    trace.add_argument("--use-slow-tokenizer", action="store_true")
    trace.set_defaults(func=collect_trace)

    synthetic = sub.add_parser("make-synthetic-trace", help="Create a small synthetic trace for replay testing.")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--requests-per-condition", type=int, default=2)
    synthetic.add_argument("--steps", type=int, default=8)
    synthetic.add_argument("--layers", type=int, default=4)
    synthetic.add_argument("--experts", type=int, default=8)
    synthetic.set_defaults(func=make_synthetic_trace)

    replay = sub.add_parser("replay", help="Replay cache/prefetch policies on a trace JSONL.")
    replay.add_argument("--trace", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--factorial-output", default=None)
    replay.add_argument("--phase", choices=["decode", "prefill", "all"], default="decode")
    replay.add_argument(
        "--policies",
        default="OnDemand,LRU_only,LFU_only,Prefetch_only,LRU_plus_prefetch",
    )
    replay.add_argument("--cache-ratios", default="0.10")
    replay.add_argument("--expert-size-mb", type=float, default=16.5)
    replay.add_argument("--transfer-latency-ms", type=float, default=1.0)
    replay.add_argument("--compute-ms-per-assignment", type=float, default=0.02)
    replay.add_argument("--prefetch-ready-slots", type=int, default=1)
    replay.add_argument("--prefetch-ttl-slots", type=int, default=128)
    replay.set_defaults(func=replay_trace)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
