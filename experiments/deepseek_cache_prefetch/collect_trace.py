from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_json, write_jsonl


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def module_layer_idx(name: str) -> int | None:
    match = LAYER_RE.search(name)
    if not match:
        return None
    return int(match.group(1))


def is_candidate_gate(name: str) -> bool:
    lower = name.lower()
    if "gate_proj" in lower:
        return False
    return lower.endswith(".mlp.gate") or lower.endswith(".gate")


def flatten_tensors(obj: Any) -> list[Any]:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (list, tuple)):
        tensors = []
        for item in obj:
            tensors.extend(flatten_tensors(item))
        return tensors
    if isinstance(obj, dict):
        tensors = []
        for item in obj.values():
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def extract_gate_tensors(output: Any) -> tuple[Any | None, Any | None]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for trace collection") from exc

    tensors = flatten_tensors(output)
    index_tensor = None
    weight_tensor = None
    for tensor in tensors:
        if tensor.dtype in (torch.int16, torch.int32, torch.int64, torch.long):
            if tensor.ndim >= 1:
                index_tensor = tensor
                break
    if index_tensor is None:
        return None, None

    index_shape = tuple(index_tensor.shape)
    for tensor in tensors:
        if tensor is index_tensor:
            continue
        if torch.is_floating_point(tensor) and tuple(tensor.shape) == index_shape:
            weight_tensor = tensor
            break
    return index_tensor, weight_tensor


def normalize_gate_matrix(tensor: Any) -> Any:
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


class RouterTraceHooks:
    def __init__(self, model: Any):
        self.model = model
        self.raw_events: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.gate_names: list[str] = []
        self.gate_layers: list[int] = []
        self.call_index = 0

    def __enter__(self) -> "RouterTraceHooks":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()

    def register(self) -> None:
        for name, module in self.model.named_modules():
            if not is_candidate_gate(name):
                continue
            layer_idx = module_layer_idx(name)
            if layer_idx is None:
                continue

            def hook(_module, _inputs, output, *, gate_name=name, gate_layer=layer_idx):
                self._capture(gate_name, gate_layer, output)

            self.handles.append(module.register_forward_hook(hook))
            self.gate_names.append(name)
            self.gate_layers.append(layer_idx)

    def clear(self) -> None:
        self.raw_events.clear()
        self.call_index = 0

    def _capture(self, gate_name: str, layer_idx: int, output: Any) -> None:
        idx_tensor, weight_tensor = extract_gate_tensors(output)
        if idx_tensor is None:
            return
        idx_tensor = normalize_gate_matrix(idx_tensor.detach())
        if weight_tensor is not None:
            weight_tensor = normalize_gate_matrix(weight_tensor.detach())
        expert_ids = idx_tensor.to("cpu").tolist()
        expert_weights = weight_tensor.to("cpu").tolist() if weight_tensor is not None else None
        self.raw_events.append(
            {
                "call_index": self.call_index,
                "gate_name": gate_name,
                "layer_idx": layer_idx,
                "num_tokens": len(expert_ids),
                "expert_ids": expert_ids,
                "expert_weights": expert_weights,
            }
        )
        self.call_index += 1

    @property
    def num_gates(self) -> int:
        return len(self.gate_layers)


def build_trace_events(
    prompt_record: dict[str, Any],
    raw_events: list[dict[str, Any]],
    num_gates: int,
    include_prefill: bool,
    generated_tokens: int,
    elapsed_ms: float,
) -> list[dict[str, Any]]:
    if num_gates <= 0:
        raise RuntimeError("no MoE gate hooks were registered")

    request_id = int(prompt_record["request_id"])
    trace_id = f"{prompt_record.get('dataset', 'unknown')}:{request_id}"
    events: list[dict[str, Any]] = []
    forward_id = 0
    decode_step = 0

    raw_events = sorted(raw_events, key=lambda item: item["call_index"])
    for start in range(0, len(raw_events), num_gates):
        group = raw_events[start : start + num_gates]
        if len(group) != num_gates:
            continue
        phase = "prefill" if max(item["num_tokens"] for item in group) > 1 else "decode"
        if phase == "prefill" and not include_prefill:
            forward_id += 1
            continue

        for item in group:
            rows = item["expert_ids"]
            weights = item.get("expert_weights")
            for row_idx, expert_ids in enumerate(rows):
                if phase == "decode":
                    token_index = decode_step
                    step_id = decode_step
                else:
                    token_index = row_idx
                    step_id = forward_id
                events.append(
                    {
                        "schema_version": 1,
                        "trace_id": trace_id,
                        "request_id": request_id,
                        "dataset": prompt_record.get("dataset", "unknown"),
                        "condition": prompt_record.get("condition", "unknown"),
                        "domain": prompt_record.get("domain", "unknown"),
                        "phase": phase,
                        "step_id": step_id,
                        "token_index": token_index,
                        "layer_idx": item["layer_idx"],
                        "expert_ids": [int(x) for x in expert_ids],
                        "expert_weights": weights[row_idx] if weights is not None else None,
                        "generated_tokens": generated_tokens,
                        "elapsed_ms": elapsed_ms,
                        "prompt_hash": prompt_hash(prompt_record["prompt"]),
                    }
                )
        if phase == "decode":
            decode_step += 1
        forward_id += 1
    return events


def load_model_and_tokenizer(args: Any) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quantization_config = None
    if args.quantization in {"4bit", "8bit"}:
        from transformers import BitsAndBytesConfig

        if args.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        use_fast=not args.use_slow_tokenizer,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = None
    if args.max_gpu_memory or args.max_cpu_memory:
        max_memory = {}
        if args.max_gpu_memory:
            max_memory[0] = args.max_gpu_memory
        if args.max_cpu_memory:
            max_memory["cpu"] = args.max_cpu_memory

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float16 if args.torch_dtype == "float16" else "auto",
        device_map=args.device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.eval()
    return model, tokenizer


def collect_trace(args: Any) -> None:
    import torch

    prompts = list(read_jsonl(args.prompts))
    if args.limit is not None:
        prompts = prompts[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args)
    all_events: list[dict[str, Any]] = []
    request_summaries: list[dict[str, Any]] = []

    with RouterTraceHooks(model) as hooks:
        if hooks.num_gates == 0:
            raise RuntimeError("No MoE gate modules found. Check the model architecture or hook rules.")

        for prompt_record in prompts:
            hooks.clear()
            encoded = tokenizer(
                prompt_record["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_tokens,
            )
            input_len = int(encoded["input_ids"].shape[-1])
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            generated_tokens = max(0, int(output.shape[-1]) - input_len)
            request_events = build_trace_events(
                prompt_record,
                hooks.raw_events,
                hooks.num_gates,
                include_prefill=args.include_prefill,
                generated_tokens=generated_tokens,
                elapsed_ms=elapsed_ms,
            )
            all_events.extend(request_events)
            request_summaries.append(
                {
                    "request_id": int(prompt_record["request_id"]),
                    "dataset": prompt_record.get("dataset", "unknown"),
                    "condition": prompt_record.get("condition", "unknown"),
                    "domain": prompt_record.get("domain", "unknown"),
                    "input_tokens": input_len,
                    "generated_tokens": generated_tokens,
                    "elapsed_ms": elapsed_ms,
                    "captured_gate_calls": len(hooks.raw_events),
                    "trace_events": len(request_events),
                }
            )
            print(
                f"request={prompt_record['request_id']} generated={generated_tokens} "
                f"events={len(request_events)} elapsed_ms={elapsed_ms:.1f}",
                flush=True,
            )

    write_jsonl(args.output, all_events)
    if args.request_summary_output:
        write_jsonl(args.request_summary_output, request_summaries)
    if args.meta_output:
        config = getattr(model, "config", None)
        meta = {
            "model": args.model,
            "num_prompts": len(prompts),
            "num_trace_events": len(all_events),
            "num_gate_modules": hooks.num_gates,
            "gate_layers": sorted(set(hooks.gate_layers)),
            "gate_names": hooks.gate_names,
            "include_prefill": args.include_prefill,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "config": config.to_dict() if config is not None and hasattr(config, "to_dict") else {},
        }
        write_json(args.meta_output, meta)
    print(f"wrote trace events: {args.output}")

