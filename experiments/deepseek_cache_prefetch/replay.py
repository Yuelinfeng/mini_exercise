from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, read_jsonl


POLICIES = [
    "OnDemand",
    "LRU_only",
    "LFU_only",
    "Prefetch_only",
    "LRU_plus_prefetch",
]


@dataclass(frozen=True)
class Assignment:
    dataset: str
    condition: str
    request_id: int
    step_id: int
    layer_idx: int
    expert_id: int
    rank: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer_idx, self.expert_id)


@dataclass
class CacheEntry:
    last_used: int
    freq: int = 0
    prefetch_issue_id: int | None = None
    prefetch_expire_time: int | None = None
    prefetch_consumed: bool = False


class CacheState:
    def __init__(self, capacity: int, policy: str):
        self.capacity = max(0, capacity)
        self.policy = policy
        self.entries: dict[tuple[int, int], CacheEntry] = {}

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self.entries

    def get(self, key: tuple[int, int], now: int) -> CacheEntry | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        entry.last_used = now
        entry.freq += 1
        return entry

    def insert(
        self,
        key: tuple[int, int],
        now: int,
        prefetch_issue_id: int | None = None,
        prefetch_expire_time: int | None = None,
    ) -> CacheEntry | None:
        if self.capacity <= 0:
            return None
        if key in self.entries:
            entry = self.entries[key]
            entry.last_used = now
            entry.freq += 1
            if prefetch_issue_id is not None:
                entry.prefetch_issue_id = prefetch_issue_id
                entry.prefetch_expire_time = prefetch_expire_time
                entry.prefetch_consumed = False
            return None
        evicted = None
        if len(self.entries) >= self.capacity:
            evict_key = self._select_evict_key()
            evicted = self.entries.pop(evict_key)
        self.entries[key] = CacheEntry(
            last_used=now,
            freq=1,
            prefetch_issue_id=prefetch_issue_id,
            prefetch_expire_time=prefetch_expire_time,
        )
        return evicted

    def _select_evict_key(self) -> tuple[int, int]:
        if self.policy == "lfu":
            return min(self.entries, key=lambda key: (self.entries[key].freq, self.entries[key].last_used))
        return min(self.entries, key=lambda key: self.entries[key].last_used)


@dataclass
class PrefetchRecord:
    issue_id: int
    key: tuple[int, int]
    issue_time: int
    ready_time: int
    expire_time: int
    redundant_reason: str | None = None
    completed: bool = False
    consumed: bool = False


def expand_assignments(events: list[dict[str, Any]], phase: str) -> list[Assignment]:
    assignments: list[Assignment] = []
    for event in events:
        if phase != "all" and event.get("phase") != phase:
            continue
        expert_ids = event.get("expert_ids") or []
        for rank, expert_id in enumerate(expert_ids):
            assignments.append(
                Assignment(
                    dataset=str(event.get("dataset", "unknown")),
                    condition=str(event.get("condition", "unknown")),
                    request_id=int(event.get("request_id", 0)),
                    step_id=int(event.get("step_id", 0)),
                    layer_idx=int(event.get("layer_idx", 0)),
                    expert_id=int(expert_id),
                    rank=rank,
                )
            )
    assignments.sort(key=lambda item: (item.request_id, item.step_id, item.layer_idx, item.rank))
    return assignments


def infer_cache_capacity(assignments: list[Assignment], cache_ratio: float) -> tuple[int, int, int]:
    layers = {item.layer_idx for item in assignments}
    experts = {item.expert_id for item in assignments}
    num_layers = max(1, len(layers))
    num_experts = max(1, max(experts) + 1 if experts else 1)
    total_expert_slots = num_layers * num_experts
    capacity = max(1, round(total_expert_slots * cache_ratio))
    return capacity, num_layers, num_experts


def make_layer_successor(assignments: list[Assignment]) -> dict[int, int]:
    layers = sorted({item.layer_idx for item in assignments})
    return {layer: layers[idx + 1] for idx, layer in enumerate(layers[:-1])}


def policy_config(policy: str) -> dict[str, Any]:
    if policy == "OnDemand":
        return {"cache": None, "prefetch": False}
    if policy == "LRU_only":
        return {"cache": "lru", "prefetch": False}
    if policy == "LFU_only":
        return {"cache": "lfu", "prefetch": False}
    if policy == "Prefetch_only":
        return {"cache": None, "prefetch": True}
    if policy == "LRU_plus_prefetch":
        return {"cache": "lru", "prefetch": True}
    raise ValueError(f"unknown policy: {policy}")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def replay_policy(
    assignments: list[Assignment],
    policy: str,
    cache_ratio: float,
    expert_size_bytes: int,
    transfer_latency_ms: float,
    compute_ms_per_assignment: float,
    prefetch_ready_slots: int,
    prefetch_ttl_slots: int,
) -> dict[str, Any]:
    cfg = policy_config(policy)
    capacity, num_layers, num_experts = infer_cache_capacity(assignments, cache_ratio)
    successor = make_layer_successor(assignments)
    cache = CacheState(capacity=capacity if cfg["cache"] else 0, policy=cfg["cache"] or "lru")
    pending: dict[int, PrefetchRecord] = {}
    transient_buffer: dict[tuple[int, int], PrefetchRecord] = {}
    next_issue_id = 1
    now = 0

    counters = defaultdict(float)
    counters["total_assignments"] = len(assignments)
    counters["cache_capacity"] = capacity if cfg["cache"] else 0
    counters["num_layers"] = num_layers
    counters["num_experts"] = num_experts

    def account_evicted(entry: CacheEntry | None) -> None:
        if entry is None:
            return
        if entry.prefetch_issue_id is not None and not entry.prefetch_consumed:
            counters["prefetch_evicted_unused"] += 1

    def complete_ready_prefetches(current_time: int) -> None:
        completed_ids = []
        for issue_id, record in pending.items():
            if record.completed or record.ready_time > current_time:
                continue
            record.completed = True
            completed_ids.append(issue_id)
            if record.redundant_reason is not None:
                continue
            if cfg["cache"]:
                if record.key in cache:
                    counters["prefetch_masked_by_cache"] += 1
                    record.redundant_reason = "masked_by_cache"
                    continue
                evicted = cache.insert(
                    record.key,
                    current_time,
                    prefetch_issue_id=record.issue_id,
                    prefetch_expire_time=record.expire_time,
                )
                account_evicted(evicted)
            else:
                transient_buffer[record.key] = record
        for issue_id in completed_ids:
            pending.pop(issue_id, None)
        for key, record in list(transient_buffer.items()):
            if record.expire_time < current_time and not record.consumed:
                counters["prefetch_expired_unused"] += 1
                transient_buffer.pop(key, None)

    def issue_prefetch(source: Assignment, current_time: int) -> None:
        nonlocal next_issue_id
        if not cfg["prefetch"]:
            return
        target_layer = successor.get(source.layer_idx)
        if target_layer is None:
            return
        target = (target_layer, source.expert_id)
        counters["issued_prefetch_count"] += 1
        issue_id = next_issue_id
        next_issue_id += 1
        record = PrefetchRecord(
            issue_id=issue_id,
            key=target,
            issue_time=current_time,
            ready_time=current_time + prefetch_ready_slots,
            expire_time=current_time + prefetch_ttl_slots,
        )
        if cfg["cache"] and target in cache:
            record.redundant_reason = "already_in_cache"
            counters["prefetch_redundant_at_issue"] += 1
        elif not cfg["cache"] and target in transient_buffer and transient_buffer[target].expire_time >= current_time:
            record.redundant_reason = "already_in_prefetch_buffer"
            counters["prefetch_duplicate"] += 1
        elif any(
            pending_record.key == target
            and not pending_record.consumed
            and pending_record.expire_time >= current_time
            for pending_record in pending.values()
        ):
            record.redundant_reason = "duplicate_pending"
            counters["prefetch_duplicate"] += 1
        else:
            counters["prefetch_transfer_bytes"] += expert_size_bytes
        pending[issue_id] = record

    for assignment in assignments:
        now += 1
        complete_ready_prefetches(now)
        key = assignment.key
        hit = False
        hit_via_prefetch = False
        timely_incremental = False

        if cfg["cache"]:
            entry = cache.get(key, now)
            if entry is not None:
                hit = True
                if entry.prefetch_issue_id is not None and not entry.prefetch_consumed:
                    hit_via_prefetch = True
                    entry.prefetch_consumed = True
                    if entry.prefetch_expire_time is None or now <= entry.prefetch_expire_time:
                        timely_incremental = True
        else:
            record = transient_buffer.get(key)
            if record is not None and record.ready_time <= now:
                hit = True
                hit_via_prefetch = True
                record.consumed = True
                transient_buffer.pop(key, None)
                if now <= record.expire_time:
                    timely_incremental = True

        if hit:
            counters["cache_hit_assignments"] += 1
            if hit_via_prefetch:
                counters["used_prefetches"] += 1
                counters["prefetch_covered_assignments"] += 1
                if timely_incremental:
                    counters["timely_incremental_useful_prefetches"] += 1
        else:
            counters["cache_miss_assignments"] += 1
            counters["demand_transfer_bytes"] += expert_size_bytes
            counters["expert_loading_stall_ms"] += transfer_latency_ms
            if cfg["cache"]:
                evicted = cache.insert(key, now)
                account_evicted(evicted)

        issue_prefetch(assignment, now)

    complete_ready_prefetches(now + prefetch_ttl_slots + prefetch_ready_slots)

    issued = counters["issued_prefetch_count"]
    redundant_total = (
        counters["prefetch_redundant_at_issue"]
        + counters["prefetch_duplicate"]
        + counters["prefetch_masked_by_cache"]
    )
    non_redundant = max(0.0, issued - redundant_total)
    total_transfer_bytes = counters["demand_transfer_bytes"] + counters["prefetch_transfer_bytes"]
    on_demand_stall_ms = counters["total_assignments"] * transfer_latency_ms
    estimated_total_ms = (
        counters["total_assignments"] * compute_ms_per_assignment
        + counters["expert_loading_stall_ms"]
    )
    generated_steps = max(1, len({(item.request_id, item.step_id) for item in assignments}))

    result = {
        "policy": policy,
        "cache_ratio": cache_ratio,
        "cache_capacity": int(counters["cache_capacity"]),
        "total_assignments": int(counters["total_assignments"]),
        "cache_hit_assignment_ratio": safe_ratio(counters["cache_hit_assignments"], counters["total_assignments"]),
        "cache_miss_assignment_ratio": safe_ratio(counters["cache_miss_assignments"], counters["total_assignments"]),
        "cpu_gpu_transfer_bytes": int(total_transfer_bytes),
        "demand_transfer_bytes": int(counters["demand_transfer_bytes"]),
        "prefetch_transfer_bytes": int(counters["prefetch_transfer_bytes"]),
        "issued_prefetch_count": int(issued),
        "prefetch_accuracy": safe_ratio(counters["used_prefetches"], issued),
        "prefetch_coverage": safe_ratio(counters["prefetch_covered_assignments"], counters["total_assignments"]),
        "prefetch_redundant_ratio": safe_ratio(redundant_total, issued),
        "timely_incremental_utility_ratio": safe_ratio(
            counters["timely_incremental_useful_prefetches"],
            non_redundant,
        ),
        "stall_saved_ratio": safe_ratio(on_demand_stall_ms - counters["expert_loading_stall_ms"], on_demand_stall_ms),
        "expert_loading_stall_ms": counters["expert_loading_stall_ms"],
        "estimated_tpot_ms": safe_ratio(estimated_total_ms, generated_steps),
        "tokens_per_sec_est": safe_ratio(generated_steps * 1000.0, estimated_total_ms),
        "prefetch_redundant_at_issue": int(counters["prefetch_redundant_at_issue"]),
        "prefetch_duplicate": int(counters["prefetch_duplicate"]),
        "prefetch_masked_by_cache": int(counters["prefetch_masked_by_cache"]),
        "prefetch_evicted_unused": int(counters["prefetch_evicted_unused"]),
        "prefetch_expired_unused": int(counters["prefetch_expired_unused"]),
        "num_layers": int(counters["num_layers"]),
        "num_experts": int(counters["num_experts"]),
        "trajectory_consistency": 1.0,
    }
    return result


def group_assignments(assignments: list[Assignment]) -> dict[tuple[str, str], list[Assignment]]:
    groups: dict[tuple[str, str], list[Assignment]] = defaultdict(list)
    for assignment in assignments:
        groups[(assignment.dataset, assignment.condition)].append(assignment)
    return groups


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = ensure_parent(path)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_factorial_effects(path: str | Path, rows: list[dict[str, Any]]) -> None:
    metrics = [
        "cache_hit_assignment_ratio",
        "prefetch_redundant_ratio",
        "timely_incremental_utility_ratio",
        "stall_saved_ratio",
        "estimated_tpot_ms",
        "cpu_gpu_transfer_bytes",
    ]
    by_key: dict[tuple[str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["condition"] in {
            "stable_homogeneous",
            "shifted_homogeneous",
            "stable_mixed",
            "shifted_mixed",
        }:
            by_key[(row["policy"], float(row["cache_ratio"]))][row["condition"]] = row

    effect_rows: list[dict[str, Any]] = []
    for (policy, cache_ratio), cells in sorted(by_key.items()):
        required = {
            "stable_homogeneous",
            "shifted_homogeneous",
            "stable_mixed",
            "shifted_mixed",
        }
        if not required.issubset(cells):
            continue
        base = cells["stable_homogeneous"]
        shift = cells["shifted_homogeneous"]
        mixed = cells["stable_mixed"]
        both = cells["shifted_mixed"]
        for metric in metrics:
            base_value = float(base[metric])
            shift_value = float(shift[metric])
            mixed_value = float(mixed[metric])
            both_value = float(both[metric])
            effect_rows.append(
                {
                    "policy": policy,
                    "cache_ratio": cache_ratio,
                    "metric": metric,
                    "stable_homogeneous": base_value,
                    "shifted_homogeneous": shift_value,
                    "stable_mixed": mixed_value,
                    "shifted_mixed": both_value,
                    "shift_effect": shift_value - base_value,
                    "mixing_effect": mixed_value - base_value,
                    "interaction_effect": both_value - shift_value - mixed_value + base_value,
                    "combined_effect": both_value - base_value,
                }
            )
    write_csv(path, effect_rows)


def replay_trace(args: Any) -> None:
    events = list(read_jsonl(args.trace))
    assignments = expand_assignments(events, phase=args.phase)
    if not assignments:
        raise RuntimeError(f"no assignments found in trace for phase={args.phase}")

    cache_ratios = [float(item) for item in args.cache_ratios.split(",") if item.strip()]
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")

    expert_size_bytes = int(args.expert_size_mb * 1024 * 1024)
    rows: list[dict[str, Any]] = []
    for (dataset, condition), group in sorted(group_assignments(assignments).items()):
        for cache_ratio in cache_ratios:
            for policy in policies:
                result = replay_policy(
                    group,
                    policy=policy,
                    cache_ratio=cache_ratio,
                    expert_size_bytes=expert_size_bytes,
                    transfer_latency_ms=args.transfer_latency_ms,
                    compute_ms_per_assignment=args.compute_ms_per_assignment,
                    prefetch_ready_slots=args.prefetch_ready_slots,
                    prefetch_ttl_slots=args.prefetch_ttl_slots,
                )
                result = {
                    "dataset": dataset,
                    "condition": condition,
                    **result,
                }
                rows.append(result)
                print(
                    f"{dataset}/{condition} policy={policy} cache={cache_ratio} "
                    f"hit={result['cache_hit_assignment_ratio']:.3f} "
                    f"red={result['prefetch_redundant_ratio']:.3f} "
                    f"util={result['timely_incremental_utility_ratio']:.3f}",
                    flush=True,
                )
    write_csv(args.output, rows)
    if args.factorial_output:
        write_factorial_effects(args.factorial_output, rows)
        print(f"wrote factorial effects: {args.factorial_output}")
    print(f"wrote replay summary: {args.output}")


def make_synthetic_trace(args: Any) -> None:
    from .io_utils import write_jsonl

    records = []
    request_id = 0
    for condition in ["stable_homogeneous", "shifted_homogeneous", "stable_mixed", "shifted_mixed"]:
        for local_request in range(args.requests_per_condition):
            for step_id in range(args.steps):
                for layer_idx in range(args.layers):
                    expert = (local_request + step_id + layer_idx) % args.experts
                    if condition.startswith("shifted") and step_id >= args.steps // 2:
                        expert = (expert + 1) % args.experts
                    if condition.endswith("mixed"):
                        expert = (expert + (step_id % 2)) % args.experts
                    records.append(
                        {
                            "schema_version": 1,
                            "trace_id": f"synthetic:{request_id}",
                            "request_id": request_id,
                            "dataset": "2x2_synthetic",
                            "condition": condition,
                            "domain": "synthetic",
                            "phase": "decode",
                            "step_id": step_id,
                            "token_index": step_id,
                            "layer_idx": layer_idx,
                            "expert_ids": [expert, (expert + 1) % args.experts],
                            "expert_weights": [0.6, 0.4],
                            "generated_tokens": args.steps,
                            "elapsed_ms": 0.0,
                            "prompt_hash": "synthetic",
                        }
                    )
            request_id += 1
    write_jsonl(args.output, records)
    print(f"wrote synthetic trace: {args.output} events={len(records)}")
