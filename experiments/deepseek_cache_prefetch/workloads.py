from __future__ import annotations

import random
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, read_jsonl, write_jsonl


NORMAL_PASSAGES = [
    "Large language models are increasingly deployed in interactive services where response latency matters as much as raw model quality.",
    "Modern operating systems use caching, scheduling, and prefetching to reduce the visible cost of slow storage and memory movement.",
    "Scientific workflows often combine stable repeated tasks with sudden shifts in input distribution and user intent.",
    "A small expert cache can improve inference latency when future expert activations are predictable over short time windows.",
    "Distributed systems must separate throughput gains from incidental reuse when evaluating an optimization.",
    "Autonomous agents often perform retrieval, planning, and tool execution in a loop, which creates changing computational demand.",
    "Hardware accelerators benefit from overlapping data transfer with computation, but incorrect overlap can increase contention.",
    "Evaluation datasets should include both natural workload samples and controlled stress cases that isolate specific mechanisms.",
]

TRANSLATION_PASSAGES = [
    "The research team measured inference latency under several cache replacement policies and compared the results.",
    "A sudden workload shift can make recent activation history less reliable for predicting the next expert.",
    "The server uses limited GPU memory, so expert weights must be moved between host memory and device memory.",
    "A useful prefetch must arrive before the expert is consumed and must avoid a real on-demand transfer.",
    "The experiment keeps the routing trajectory fixed so that policy effects can be compared fairly.",
    "Mixed requests combine multiple semantic domains and may weaken short-term expert locality.",
]

SUMMARIZATION_PASSAGES = [
    "This paragraph describes an inference system that keeps a subset of experts on the GPU and loads the rest from CPU memory when needed. The main challenge is to decide which experts should stay resident and which experts should be prefetched.",
    "The paper argues that cache hit rate alone is insufficient for judging prefetch quality. A prefetch may be redundant if the same expert was already present in the cache before the request consumed it.",
    "The benchmark contains stable homogeneous requests, shifted homogeneous requests, stable mixed requests, and shifted mixed requests. These cases isolate different forms of workload non-stationarity.",
    "A practical evaluation should report common latency and hit-rate metrics, while also measuring whether prefetches are timely, non-redundant, and able to reduce stalls.",
    "The AutoDL environment has one T4 GPU and limited device memory. The experiment therefore uses trace collection and replay before attempting full asynchronous runtime changes.",
    "The proposed method should eventually beat simple baselines on both conventional performance metrics and utility-specific diagnostic metrics.",
]


def build_normal_prompts(num_requests: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    prompts: list[dict[str, Any]] = []
    for request_id in range(num_requests):
        passages = rng.sample(NORMAL_PASSAGES, k=min(3, len(NORMAL_PASSAGES)))
        prompt = (
            "Continue the following technical discussion in a concise paragraph.\n\n"
            + "\n".join(f"- {passage}" for passage in passages)
            + "\n\nContinuation:"
        )
        prompts.append(
            {
                "request_id": request_id,
                "dataset": "normal",
                "condition": "normal",
                "domain": "natural",
                "seed": seed,
                "prompt": prompt,
            }
        )
    return prompts


def choose_domain_by_fraction(
    domain_a: str,
    domain_b: str,
    relative_position: int,
    block_size: int,
    domain_a_fraction: float,
) -> str:
    cutoff = round(block_size * domain_a_fraction)
    return domain_a if relative_position < cutoff else domain_b


def chunk_words(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def homogeneous_prompt(domain: str, passage: str) -> str:
    if domain == "translation":
        return (
            "Translate the following English passage into French. Preserve technical terms.\n\n"
            f"Passage:\n{passage}\n\nTranslation:"
        )
    if domain == "summarization":
        return (
            "Summarize the following passage in two concise sentences.\n\n"
            f"Passage:\n{passage}\n\nSummary:"
        )
    raise ValueError(f"unknown domain: {domain}")


def mixed_prompt(
    primary_domain: str,
    primary_text: str,
    secondary_domain: str,
    secondary_text: str,
    mix_fraction: float,
    mix_mode: str,
    interleave_chunk_words: int,
) -> str:
    primary_words = primary_text.split()
    secondary_words = secondary_text.split()
    total_primary = max(1, round(len(primary_words) * mix_fraction))
    total_secondary = max(1, round(len(secondary_words) * (1.0 - mix_fraction)))
    primary_slice = " ".join(primary_words[:total_primary])
    secondary_slice = " ".join(secondary_words[:total_secondary])

    if mix_mode == "concat":
        body = (
            f"[{primary_domain.upper()}]\n{primary_slice}\n\n"
            f"[{secondary_domain.upper()}]\n{secondary_slice}"
        )
    elif mix_mode == "interleave":
        chunks_a = chunk_words(primary_slice, interleave_chunk_words)
        chunks_b = chunk_words(secondary_slice, interleave_chunk_words)
        pieces = []
        for idx in range(max(len(chunks_a), len(chunks_b))):
            if idx < len(chunks_a):
                pieces.append(f"[{primary_domain.upper()}] {chunks_a[idx]}")
            if idx < len(chunks_b):
                pieces.append(f"[{secondary_domain.upper()}] {chunks_b[idx]}")
        body = "\n".join(pieces)
    else:
        raise ValueError(f"unknown mix_mode: {mix_mode}")

    return (
        "Complete the labeled tasks. For TRANSLATION sections, translate into French. "
        "For SUMMARIZATION sections, summarize the content in English.\n\n"
        f"{body}\n\nAnswer:"
    )


def build_2x2_prompts(
    num_requests_per_cell: int,
    seed: int,
    shift_block_size: int,
    shift_major_fraction: float,
    stable_mix_fraction: float,
    mix_mode: str,
    interleave_chunk_words: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    conditions = [
        "stable_homogeneous",
        "shifted_homogeneous",
        "stable_mixed",
        "shifted_mixed",
    ]
    records: list[dict[str, Any]] = []
    global_request_id = 0
    domain_a = "translation"
    domain_b = "summarization"

    for condition in conditions:
        for local_id in range(num_requests_per_cell):
            block_index = local_id // shift_block_size
            relative_position = local_id % shift_block_size
            phase_a_major = block_index % 2 == 0
            major_a_fraction = shift_major_fraction if phase_a_major else 1.0 - shift_major_fraction
            primary_domain = choose_domain_by_fraction(
                domain_a,
                domain_b,
                relative_position,
                shift_block_size,
                major_a_fraction,
            )
            secondary_domain = domain_b if primary_domain == domain_a else domain_a

            translation_text = rng.choice(TRANSLATION_PASSAGES)
            summarization_text = rng.choice(SUMMARIZATION_PASSAGES)
            text_by_domain = {
                "translation": translation_text,
                "summarization": summarization_text,
            }

            if condition == "stable_homogeneous":
                domain = domain_a
                prompt = homogeneous_prompt(domain, text_by_domain[domain])
                mix_fraction = 1.0
            elif condition == "shifted_homogeneous":
                domain = primary_domain
                prompt = homogeneous_prompt(domain, text_by_domain[domain])
                mix_fraction = 1.0
            elif condition == "stable_mixed":
                domain = "mixed"
                primary_domain = domain_a
                secondary_domain = domain_b
                mix_fraction = stable_mix_fraction
                prompt = mixed_prompt(
                    primary_domain,
                    text_by_domain[primary_domain],
                    secondary_domain,
                    text_by_domain[secondary_domain],
                    mix_fraction,
                    mix_mode,
                    interleave_chunk_words,
                )
            elif condition == "shifted_mixed":
                domain = "mixed"
                mix_fraction = shift_major_fraction if phase_a_major else 1.0 - shift_major_fraction
                prompt = mixed_prompt(
                    primary_domain,
                    text_by_domain[primary_domain],
                    secondary_domain,
                    text_by_domain[secondary_domain],
                    mix_fraction,
                    mix_mode,
                    interleave_chunk_words,
                )
            else:
                raise AssertionError(condition)

            records.append(
                {
                    "request_id": global_request_id,
                    "local_request_id": local_id,
                    "dataset": "2x2",
                    "condition": condition,
                    "domain": domain,
                    "primary_domain": primary_domain,
                    "secondary_domain": secondary_domain,
                    "mix_fraction": mix_fraction,
                    "seed": seed,
                    "prompt": prompt,
                }
            )
            global_request_id += 1
    return records


def write_prompts(
    output: str | Path,
    suite: str,
    num_requests: int,
    num_requests_per_cell: int,
    seed: int,
    shift_block_size: int,
    shift_major_fraction: float,
    stable_mix_fraction: float,
    mix_mode: str,
    interleave_chunk_words: int,
) -> int:
    if suite == "normal":
        records = build_normal_prompts(num_requests=num_requests, seed=seed)
    elif suite == "2x2":
        records = build_2x2_prompts(
            num_requests_per_cell=num_requests_per_cell,
            seed=seed,
            shift_block_size=shift_block_size,
            shift_major_fraction=shift_major_fraction,
            stable_mix_fraction=stable_mix_fraction,
            mix_mode=mix_mode,
            interleave_chunk_words=interleave_chunk_words,
        )
    else:
        raise ValueError(f"unknown suite: {suite}")
    write_jsonl(output, records)
    return len(records)


def write_workload_sanity(prompts_path: str | Path, output: str | Path) -> int:
    records = list(read_jsonl(prompts_path))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("dataset", "unknown")), str(record.get("condition", "unknown")))].append(record)

    rows: list[dict[str, Any]] = []
    for (dataset, condition), items in sorted(grouped.items()):
        domains = Counter(str(item.get("domain", "unknown")) for item in items)
        primary = Counter(str(item.get("primary_domain", "unknown")) for item in items)
        mix_values = [float(item.get("mix_fraction", 0.0)) for item in items if "mix_fraction" in item]
        prompt_lengths = [len(str(item.get("prompt", "")).split()) for item in items]
        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "num_requests": len(items),
                "domain_counts": dict(domains),
                "primary_domain_counts": dict(primary),
                "mean_mix_fraction": sum(mix_values) / len(mix_values) if mix_values else 0.0,
                "min_prompt_words": min(prompt_lengths) if prompt_lengths else 0,
                "max_prompt_words": max(prompt_lengths) if prompt_lengths else 0,
                "mean_prompt_words": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0,
            }
        )

    target = ensure_parent(output)
    with target.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "condition",
            "num_requests",
            "domain_counts",
            "primary_domain_counts",
            "mean_mix_fraction",
            "min_prompt_words",
            "max_prompt_words",
            "mean_prompt_words",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
