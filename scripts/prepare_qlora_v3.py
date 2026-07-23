#!/usr/bin/env python3
"""Prepare the isolated V3 constrained-recovery selection experiment.

V3 changes exactly one scientific variable relative to V2: trajectory
selection.  Prompt construction, SFT labels, model, source data, split,
training schedule, and evaluation remain frozen to V2 so the new arm can be
compared directly with the completed V2 controls.

The selector uses the V2 random-success selection as a training-only anchor.
It searches a deterministic family of recovery-enriched selections and accepts
only candidates that retain task diversity, non-recovery supervision, Random
anchor content, and a similar target-tool distribution.  No validation or test
outcomes are consulted.

This is a diagnostic follow-up on an already inspected test set.  It is not a
paper-final, end-to-end, or executable Agent evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from prepare_qlora_v2 import (
    GRADIENT_ACCUMULATION,
    MAX_COMPLETION_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_SEQ_LEN,
    MODEL,
    MODEL_REVISION,
    SEED,
    arm_stats,
    common_source_selection,
    examples_for_trace,
    fixed_schedule,
    load_success_records,
    recovery_features,
    sha256_file,
    subset_fill,
    task_split,
)

PROTOCOL = "qlora_v3_constrained_recovery"
ARM = "constrained_recovery"
V2_RANDOM_ARM = "random_success"
TRAIN_MICROBATCHES = 1088
OPTIMIZER_STEPS = 68
EXPECTED_SOURCE_BUDGETS = {
    "gpt-4o-retail": 523_182,
    "sonnet-35-new-retail": 1_167_747,
}
EXPECTED_RAW_SHA256 = {
    "gpt-4o-retail.json": "df01707894836168ff0ec9616b0bf08f66c7e5afcf313e5fe4f7a2f5c2ec938b",
    "sonnet-35-new-retail.json": "0df526398e9d2720c32d340815cffb04fe8c4f8a61b1f4f84bf3bb558f760131",
}
EXPECTED_VALIDATION_SHA256 = "71f66c41394a50e3d992b7b860bb444774e7215ada3ad95e7d749b911916057a"
EXPECTED_TEST_SHA256 = "0da63463a65d3b377b3ef3a7e0032a8ffabdc8ab3e439c33850a1eea1ee8fd96"
EXPECTED_V2_RANDOM_TRACE_SHA256 = "90580570b3c4262df2ca5a80933544dd97c7c00477b7b7f8227d48b07910ece2"
EXPECTED_V3_TRACE_SHA256 = "a65bba64baf7c9a6e816e721b382511211aa9df6f5204e7c4cce74f78b992cc5"
EXPECTED_V3_TRAIN_SHA256 = "6b991fe03c7b79132438f8681dccee9e4fab2003a5859bd1abce26ba32ed046d"
EXPECTED_V3_SCHEDULE_SHA256 = "46acdd204d3dc213389af9b44ed6884031899a82615f8a9be47e024c30e2ea38"
EXPECTED_V2_RANDOM_STATS = {
    "trajectories": 141,
    "examples": 1066,
    "unique_tasks": 63,
    "selected_sft_tokens": 1_690_929,
    "recovery_targets": 78,
    "agent_initiated_targets": 13,
    "scheduled_loss_tokens": 36_706,
}
EXPECTED_V3_STATS = {
    "trajectories": 167,
    "examples": 1069,
    "unique_tasks": 76,
    "selected_sft_tokens": 1_690_929,
    "recovery_targets": 102,
    "agent_initiated_targets": 20,
    "non_recovery_targets": 967,
    "failed_action_targets": 104,
    "unique_recovery_signatures": 26,
    "unique_agent_recovery_signatures": 15,
    "anchor_task_overlap": 63,
    "scheduled_loss_tokens": 36_599,
    "scheduled_target_tool_tvd_from_v2_random": 0.07444852941176469,
}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write the frozen Windows-V2 CRLF byte representation on every OS.

    The completed V2 run was prepared on Windows, where text-mode JSONL used
    CRLF.  Writing CRLF explicitly makes the validation/test byte hashes
    portable while preserving exactly the V2 examples and parser semantics.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\r\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic LF JSON metadata on every operating system."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

# The search family is fixed before V3 evaluation.  Different profiles trade
# off recovery enrichment, task novelty, random-anchor retention, target-tool
# distribution, and clean supervision.  Profile selection uses training-data
# contracts only.
SEARCH_PROFILES = (
    {"name": "anchor_heavy", "recovery": 1.0, "task": 12.0, "anchor": 14.0, "tool": 18.0, "ordinary": 0.10},
    {"name": "anchor_medium", "recovery": 2.0, "task": 12.0, "anchor": 8.0, "tool": 18.0, "ordinary": 0.08},
    {"name": "balanced_a", "recovery": 3.0, "task": 16.0, "anchor": 5.0, "tool": 24.0, "ordinary": 0.08},
    {"name": "balanced_b", "recovery": 4.0, "task": 20.0, "anchor": 3.0, "tool": 28.0, "ordinary": 0.08},
    {"name": "task_heavy", "recovery": 3.0, "task": 28.0, "anchor": 3.0, "tool": 24.0, "ordinary": 0.08},
    {"name": "tool_heavy", "recovery": 3.0, "task": 16.0, "anchor": 4.0, "tool": 42.0, "ordinary": 0.08},
    {"name": "recovery_medium", "recovery": 6.0, "task": 20.0, "anchor": 3.0, "tool": 32.0, "ordinary": 0.10},
    {"name": "recovery_high", "recovery": 9.0, "task": 24.0, "anchor": 2.0, "tool": 38.0, "ordinary": 0.12},
)
CORE_FRACTIONS = (0.75, 0.82, 0.88, 0.90)

# Tiers are frozen before evaluation.  The strict tier is known to be feasible
# for the pinned tau-bench files.  The fallback is deliberately still
# scientifically defensible, and its use is surfaced as a protocol warning;
# neither tier may be edited after evaluation starts.
CONSTRAINT_TIERS = (
    {
        "name": "strict",
        "min_examples": 1045,
        "max_examples": 1088,
        "min_unique_tasks": 60,
        "min_anchor_task_overlap": 57,
        "min_non_recovery_targets": 939,
        "min_recovery_targets": 100,
        "max_recovery_targets": 139,
        "min_agent_initiated_targets": 20,
        "max_target_tool_tvd": 0.075,
        "max_scheduled_target_tool_tvd": 0.075,
        "min_anchor_token_overlap": 0.65,
        "max_loss_token_relative_delta": 0.05,
        "max_unpaired_failed_action_excess": 3,
    },
    {
        "name": "predeclared_fallback",
        "min_examples": 1045,
        "max_examples": 1088,
        "min_unique_tasks": 60,
        "min_anchor_task_overlap": 57,
        "min_non_recovery_targets": 939,
        "min_recovery_targets": 100,
        "max_recovery_targets": 139,
        "min_agent_initiated_targets": 17,
        "max_target_tool_tvd": 0.10,
        "max_scheduled_target_tool_tvd": 0.10,
        "min_anchor_token_overlap": 0.65,
        "max_loss_token_relative_delta": 0.05,
        "max_unpaired_failed_action_excess": 3,
    },
)


def trace_set_fingerprint(trace_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(trace_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tool_counts(selected: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        example["target_tool"]
        for record in selected
        for example in record["examples"]
    )


def scheduled_tool_counts(schedule: list[dict[str, Any]]) -> Counter[str]:
    """Count target tools after the fixed 1,088-exposure schedule is built."""
    return Counter(example["target_tool"] for example in schedule)


def tool_tvd(left: Counter[str], right: Counter[str]) -> float:
    """Total-variation distance between two target-tool distributions."""
    left_total, right_total = sum(left.values()), sum(right.values())
    if not left_total or not right_total:
        return 1.0
    # Sort explicitly: Python hash randomization otherwise changes the
    # last floating-point bit across fresh processes and can trip a frozen
    # reproducibility contract despite identical integer counts.
    tools = sorted(set(left) | set(right))
    return 0.5 * sum(
        abs(left.get(tool, 0) / left_total - right.get(tool, 0) / right_total)
        for tool in tools
    )


def target_outcome_error(item: dict[str, Any], target_index: int) -> str | None:
    """Classify whether the labelled call itself receives an error result.

    V2's label rule is intentionally retained, so calls that later fail remain
    SFT labels.  V3 reports them separately and constrains excess failed labels
    instead of silently changing the learning objective.
    """
    from prepare_qlora_v2 import error_type

    trajectory = item["record"]["traj"]
    for message in trajectory[target_index + 1:]:
        if message.get("role") == "tool":
            return error_type(str(message.get("content") or ""))
        if message.get("role") == "assistant":
            break
    return None


def recovery_signature(example: dict[str, Any]) -> str | None:
    if example["recovery_mode"] == "none":
        return None
    return "|".join(
        str(value or "unknown")
        for value in (
            example["recovery_mode"],
            example.get("prior_error_type"),
            example.get("failed_tool"),
            example.get("repair_tool"),
        )
    )


def selection_stats(
    selected: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
    schedule: list[dict[str, Any]] | None = None,
    anchor_schedule: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    examples = [example for record in selected for example in record["examples"]]
    recovery = [example for example in examples if example["recovery_mode"] != "none"]
    agent = [example for example in examples if example["recovery_mode"] == "agent_initiated"]
    failed = [example for example in examples if example.get("target_leads_to_error")]
    ordinary = [
        example
        for example in examples
        if example["recovery_mode"] == "none" and not example.get("target_leads_to_error")
    ]
    selected_tools = tool_counts(selected)
    anchor_tools = tool_counts(anchor)
    selected_scheduled_tools = (
        scheduled_tool_counts(schedule) if schedule is not None else None
    )
    anchor_scheduled_tools = (
        scheduled_tool_counts(anchor_schedule)
        if anchor_schedule is not None
        else None
    )
    anchor_ids = {record["trace_id"] for record in anchor}
    selected_ids = {record["trace_id"] for record in selected}
    anchor_tasks = {record["task_key"] for record in anchor}
    selected_tasks = {record["task_key"] for record in selected}
    anchor_tokens = sum(record["sft_token_cost"] for record in anchor)
    overlap_tokens = sum(
        record["sft_token_cost"]
        for record in anchor
        if record["trace_id"] in selected_ids
    )
    source_overlap = {}
    for source in sorted({record["source"] for record in anchor}):
        source_anchor = [record for record in anchor if record["source"] == source]
        source_total = sum(record["sft_token_cost"] for record in source_anchor)
        source_common = [
            record for record in source_anchor if record["trace_id"] in selected_ids
        ]
        source_overlap[source] = {
            "traces": len(source_common),
            "anchor_traces": len(source_anchor),
            "tokens": sum(record["sft_token_cost"] for record in source_common),
            "anchor_tokens": source_total,
            "token_ratio": (
                sum(record["sft_token_cost"] for record in source_common) / source_total
                if source_total
                else 0.0
            ),
        }
    signatures = {signature for example in recovery if (signature := recovery_signature(example))}
    agent_signatures = {signature for example in agent if (signature := recovery_signature(example))}
    return {
        "trajectories": len(selected),
        "examples": len(examples),
        "unique_tasks": len(selected_tasks),
        "selected_sft_tokens": sum(record["sft_token_cost"] for record in selected),
        "recovery_targets": len(recovery),
        "agent_initiated_targets": len(agent),
        "user_assisted_targets": len(recovery) - len(agent),
        "non_recovery_targets": len(examples) - len(recovery),
        "failed_action_targets": len(failed),
        "unpaired_failed_action_excess": len(failed) - len(recovery),
        "ordinary_success_targets": len(ordinary),
        "unique_recovery_signatures": len(signatures),
        "unique_agent_recovery_signatures": len(agent_signatures),
        "target_tool_tvd_from_v2_random": tool_tvd(selected_tools, anchor_tools),
        "target_tool_counts": dict(sorted(selected_tools.items())),
        "scheduled_target_tool_tvd_from_v2_random": (
            tool_tvd(selected_scheduled_tools, anchor_scheduled_tools)
            if selected_scheduled_tools is not None
            and anchor_scheduled_tools is not None
            else None
        ),
        "scheduled_target_tool_counts": (
            dict(sorted(selected_scheduled_tools.items()))
            if selected_scheduled_tools is not None
            else None
        ),
        "anchor_trace_overlap": len(anchor_ids & selected_ids),
        "anchor_trace_overlap_ratio": len(anchor_ids & selected_ids) / len(anchor_ids),
        "anchor_token_overlap": overlap_tokens,
        "anchor_token_overlap_ratio": overlap_tokens / anchor_tokens,
        "anchor_task_overlap": len(anchor_tasks & selected_tasks),
        "anchor_task_overlap_ratio": len(anchor_tasks & selected_tasks) / len(anchor_tasks),
        "anchor_overlap_by_source": source_overlap,
        "selected_completion_loss_tokens": sum(example["completion_tokens"] for example in examples),
        "scheduled_loss_tokens": (
            sum(example["completion_tokens"] for example in schedule)
            if schedule is not None
            else None
        ),
        "scheduled_unique_examples": (
            len({example["example_id"] for example in schedule})
            if schedule is not None
            else None
        ),
        "trace_set_sha256": trace_set_fingerprint(record["trace_id"] for record in selected),
    }


def record_recovery_value(record: dict[str, Any]) -> float:
    agent = sum(example["recovery_mode"] == "agent_initiated" for example in record["examples"])
    user = sum(example["recovery_mode"] == "user_assisted" for example in record["examples"])
    feature_bonus = 0.25 * len(record["recovery_features"])
    return 4.0 * agent + user + feature_bonus


def constrained_preference(
    records: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
    profile: dict[str, float | str],
) -> list[dict[str, Any]]:
    """Produce a deterministic dynamic ranking under a fixed search profile."""
    remaining = list(records)
    anchor_ids = {record["trace_id"] for record in anchor}
    reference_tools = tool_counts(anchor)
    selected: list[dict[str, Any]] = []
    selected_tasks: set[str] = set()
    selected_tools: Counter[str] = Counter()
    while remaining:
        def score(record: dict[str, Any]) -> tuple[float, int, str]:
            record_tools = Counter(example["target_tool"] for example in record["examples"])
            candidate_tools = selected_tools + record_tools
            new_task = float(record["task_key"] not in selected_tasks)
            ordinary_success = sum(
                example["recovery_mode"] == "none"
                and not example.get("target_leads_to_error")
                for example in record["examples"]
            )
            utility = (
                float(profile["recovery"]) * record_recovery_value(record)
                + float(profile["task"]) * new_task
                + float(profile["anchor"]) * float(record["trace_id"] in anchor_ids)
                + float(profile["ordinary"]) * ordinary_success
                - float(profile["tool"]) * tool_tvd(candidate_tools, reference_tools)
            )
            return utility / max(1, record["sft_token_cost"]), -record["sft_token_cost"], record["trace_id"]

        best = max(remaining, key=score)
        remaining.remove(best)
        selected.append(best)
        selected_tasks.add(best["task_key"])
        selected_tools.update(example["target_tool"] for example in best["examples"])
    return selected


def ranked_to_budget(preferred: list[dict[str, Any]], budget: int, core_fraction: float) -> list[dict[str, Any]]:
    core_target = int(core_fraction * budget)
    core, used = [], 0
    for record in preferred:
        if used >= core_target:
            break
        if used + record["sft_token_cost"] <= budget:
            core.append(record)
            used += record["sft_token_cost"]
    core_ids = {record["trace_id"] for record in core}
    filler = subset_fill(
        [record for record in preferred if record["trace_id"] not in core_ids],
        budget - used,
    )
    return core + filler


def source_candidates(
    records: list[dict[str, Any]],
    budget: int,
    anchor: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    def add(selected: list[dict[str, Any]], profile: str, core_fraction: float | None) -> None:
        if sum(record["sft_token_cost"] for record in selected) != budget:
            return
        fingerprint = trace_set_fingerprint(record["trace_id"] for record in selected)
        candidates.setdefault(fingerprint, {
            "selected": selected,
            "profile": profile,
            "core_fraction": core_fraction,
        })

    add(list(anchor), "v2_random_anchor", None)
    for profile in SEARCH_PROFILES:
        preferred = constrained_preference(records, anchor, profile)
        for fraction in CORE_FRACTIONS:
            add(ranked_to_budget(preferred, budget, fraction), str(profile["name"]), fraction)
    return list(candidates.values())


def choose_candidate(
    source_options: dict[str, list[dict[str, Any]]],
    source_budgets: dict[str, int],
    anchor: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchor = sorted(anchor, key=lambda item: item["trace_id"])
    anchor_rows = [example for record in anchor for example in record["examples"]]
    anchor_schedule = fixed_schedule(anchor_rows, V2_RANDOM_ARM, TRAIN_MICROBATCHES)
    anchor_stats = selection_stats(
        anchor,
        anchor,
        anchor_schedule,
        anchor_schedule,
    )
    evaluated = []
    sources = sorted(source_options)
    for choices in itertools.product(*(source_options[source] for source in sources)):
        selected = [
            record
            for source, choice in zip(sources, choices)
            for record in choice["selected"]
        ]
        selected.sort(key=lambda item: item["trace_id"])
        if len({record["trace_id"] for record in selected}) != len(selected):
            continue
        rows = [example for record in selected for example in record["examples"]]
        if not (CONSTRAINT_TIERS[-1]["min_examples"] <= len(rows) <= TRAIN_MICROBATCHES):
            continue
        schedule = fixed_schedule(rows, ARM, TRAIN_MICROBATCHES)
        stats = selection_stats(
            selected,
            anchor,
            schedule,
            anchor_schedule,
        )
        source_totals = {
            source: sum(record["sft_token_cost"] for record in selected if record["source"] == source)
            for source in sources
        }
        if (
            source_totals != source_budgets
            or stats["scheduled_unique_examples"] != stats["examples"]
        ):
            continue
        evaluated.append({
            "selected": selected,
            "stats": stats,
            "source_profiles": {
                source: {
                    "profile": choice["profile"],
                    "core_fraction": choice["core_fraction"],
                }
                for source, choice in zip(sources, choices)
            },
        })

    for tier in CONSTRAINT_TIERS:
        feasible = []
        for candidate in evaluated:
            stats = candidate["stats"]
            if not tier["min_examples"] <= stats["examples"] <= tier["max_examples"]:
                continue
            if stats["unique_tasks"] < tier["min_unique_tasks"]:
                continue
            if stats["anchor_task_overlap"] < tier["min_anchor_task_overlap"]:
                continue
            if stats["non_recovery_targets"] < tier["min_non_recovery_targets"]:
                continue
            if not tier["min_recovery_targets"] <= stats["recovery_targets"] <= tier["max_recovery_targets"]:
                continue
            if stats["agent_initiated_targets"] < tier["min_agent_initiated_targets"]:
                continue
            if stats["target_tool_tvd_from_v2_random"] > tier["max_target_tool_tvd"]:
                continue
            if (
                stats["scheduled_target_tool_tvd_from_v2_random"]
                > tier["max_scheduled_target_tool_tvd"]
            ):
                continue
            if stats["anchor_token_overlap_ratio"] < tier["min_anchor_token_overlap"]:
                continue
            loss_delta = abs(
                stats["scheduled_loss_tokens"] - EXPECTED_V2_RANDOM_STATS["scheduled_loss_tokens"]
            ) / EXPECTED_V2_RANDOM_STATS["scheduled_loss_tokens"]
            if loss_delta > tier["max_loss_token_relative_delta"]:
                continue
            if stats["unpaired_failed_action_excess"] > tier["max_unpaired_failed_action_excess"]:
                continue
            feasible.append(candidate)
        if feasible:
            # Saturate distinct recovery signatures first rather than
            # maximizing raw recovery volume.  Then minimize target-tool drift
            # and intervention size to protect the V2 Random distribution.
            def ranking(item: dict[str, Any]) -> tuple[Any, ...]:
                return (
                    item["stats"]["unique_agent_recovery_signatures"],
                    item["stats"]["unique_recovery_signatures"],
                    item["stats"]["agent_initiated_targets"],
                    -item["stats"]["scheduled_target_tool_tvd_from_v2_random"],
                    -item["stats"]["target_tool_tvd_from_v2_random"],
                    item["stats"]["anchor_task_overlap"],
                    item["stats"]["anchor_token_overlap_ratio"],
                    item["stats"]["unique_tasks"],
                    -item["stats"]["recovery_targets"],
                    item["stats"]["trace_set_sha256"],
                )

            chosen = max(feasible, key=ranking)
            ranked_feasible = sorted(feasible, key=ranking, reverse=True)
            audit = {
                "selected_tier": tier,
                "fallback_used": tier["name"] != "strict",
                "anchor_stats": anchor_stats,
                "selected_stats": chosen["stats"],
                "source_profiles": chosen["source_profiles"],
                "source_candidate_counts": {
                    source: len(options) for source, options in source_options.items()
                },
                "combined_candidates_evaluated": len(evaluated),
                "feasible_candidates_in_selected_tier": len(feasible),
                "feasible_candidate_summaries": [
                    {
                        "rank": rank,
                        "source_profiles": candidate["source_profiles"],
                        **{
                            key: candidate["stats"][key]
                            for key in (
                                "examples",
                                "unique_tasks",
                                "recovery_targets",
                                "agent_initiated_targets",
                                "non_recovery_targets",
                                "failed_action_targets",
                                "unique_recovery_signatures",
                                "unique_agent_recovery_signatures",
                                "target_tool_tvd_from_v2_random",
                                "scheduled_target_tool_tvd_from_v2_random",
                                "anchor_token_overlap_ratio",
                                "anchor_task_overlap",
                                "scheduled_loss_tokens",
                                "trace_set_sha256",
                            )
                        },
                    }
                    for rank, candidate in enumerate(ranked_feasible, start=1)
                ],
            }
            return chosen["selected"], audit
    raise RuntimeError(
        "no constrained-recovery candidate satisfies either predeclared tier; "
        "abort before GPU training instead of silently "
        "relaxing the protocol"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, default=Path("results/selection_v3"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/qlora_v3"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("tokenizer must define eos_token")

    observed_raw_hashes = {
        name: sha256_file(args.data_dir / name)
        for name in sorted(EXPECTED_RAW_SHA256)
    }
    if observed_raw_hashes != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            f"pinned tau-bench historical files drifted: "
            f"observed={observed_raw_hashes}, expected={EXPECTED_RAW_SHA256}"
        )

    records = load_success_records(args.data_dir)
    train_tasks, validation_tasks, test_tasks = task_split(records)
    if (
        train_tasks & validation_tasks
        or train_tasks & test_tasks
        or validation_tasks & test_tasks
    ):
        raise AssertionError("task-level train/validation/test split overlap")
    for item in records:
        item["examples"] = examples_for_trace(
            item,
            tokenizer,
            MAX_PROMPT_TOKENS,
            MAX_COMPLETION_TOKENS,
            MAX_SEQ_LEN,
        )
        if item["task_key"] in train_tasks:
            for example in item["examples"]:
                outcome = target_outcome_error(item, example["target_message_index"])
                example["target_leads_to_error"] = outcome is not None
                example["target_result_error_type"] = outcome
        item["sft_token_cost"] = sum(example["sequence_tokens"] for example in item["examples"])
        item["recovery_features"] = recovery_features(item)
    records = [item for item in records if item["examples"]]
    train_records = [item for item in records if item["task_key"] in train_tasks]

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in train_records:
        by_source[item["source"]].append(item)

    source_budgets, v2_source_selections = {}, {}
    for source, source_records in sorted(by_source.items()):
        initial = round(0.30 * sum(record["sft_token_cost"] for record in source_records))
        source_budgets[source], v2_source_selections[source] = common_source_selection(
            source_records,
            initial,
            source,
        )
    if source_budgets != EXPECTED_SOURCE_BUDGETS:
        raise RuntimeError(
            f"V2 source-budget anchor drift: observed={source_budgets}, "
            f"expected={EXPECTED_SOURCE_BUDGETS}"
        )

    anchor = [
        record
        for source in sorted(v2_source_selections)
        for record in v2_source_selections[source][V2_RANDOM_ARM]
    ]
    anchor.sort(key=lambda item: item["trace_id"])
    anchor_fingerprint = trace_set_fingerprint(item["trace_id"] for item in anchor)
    if anchor_fingerprint != EXPECTED_V2_RANDOM_TRACE_SHA256:
        raise RuntimeError(
            f"V2 Random anchor drift: observed={anchor_fingerprint}, "
            f"expected={EXPECTED_V2_RANDOM_TRACE_SHA256}"
        )
    anchor_schedule = fixed_schedule(
        [example for item in anchor for example in item["examples"]],
        V2_RANDOM_ARM,
        TRAIN_MICROBATCHES,
    )
    observed_anchor = selection_stats(
        anchor,
        anchor,
        anchor_schedule,
        anchor_schedule,
    )
    for key, expected in EXPECTED_V2_RANDOM_STATS.items():
        if observed_anchor.get(key) != expected:
            raise RuntimeError(
                f"V2 Random anchor statistic drift for {key}: "
                f"observed={observed_anchor.get(key)!r}, expected={expected!r}"
            )

    options = {
        source: source_candidates(
            by_source[source],
            source_budgets[source],
            v2_source_selections[source][V2_RANDOM_ARM],
        )
        for source in sorted(by_source)
    }
    selected, selection_audit = choose_candidate(options, source_budgets, anchor)
    selected.sort(key=lambda item: item["trace_id"])
    selected_stats = selection_audit["selected_stats"]
    if selection_audit["fallback_used"]:
        raise RuntimeError(
            "pinned data unexpectedly required the V3 fallback tier; stop for "
            "audit instead of training a different dataset"
        )
    if selected_stats["trace_set_sha256"] != EXPECTED_V3_TRACE_SHA256:
        raise RuntimeError(
            f"V3 selected trace set drift: observed={selected_stats['trace_set_sha256']}, "
            f"expected={EXPECTED_V3_TRACE_SHA256}"
        )
    for key, expected in EXPECTED_V3_STATS.items():
        if selected_stats.get(key) != expected:
            raise RuntimeError(
                f"V3 selected statistic drift for {key}: "
                f"observed={selected_stats.get(key)!r}, expected={expected!r}"
            )

    args.selection_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    for split, tasks in (("validation", validation_tasks), ("test", test_tasks)):
        rows = [
            example
            for item in records
            if item["task_key"] in tasks
            for example in item["examples"]
        ]
        write_jsonl(args.processed_dir / "shared" / f"{split}.jsonl", rows)

    validation_path = args.processed_dir / "shared" / "validation.jsonl"
    test_path = args.processed_dir / "shared" / "test.jsonl"
    if sha256_file(validation_path) != EXPECTED_VALIDATION_SHA256:
        raise RuntimeError("V3 validation split/prompt hash differs from frozen V2")
    if sha256_file(test_path) != EXPECTED_TEST_SHA256:
        raise RuntimeError("V3 test split/prompt hash differs from frozen V2")

    rows = [example for item in selected for example in item["examples"]]
    if len(rows) > TRAIN_MICROBATCHES:
        raise RuntimeError(
            f"V3 selected {len(rows)} examples but has only "
            f"{TRAIN_MICROBATCHES} fixed microbatches"
        )
    if len({row["example_id"] for row in rows}) != len(rows):
        raise RuntimeError("V3 selected duplicate training example IDs")
    schedule = fixed_schedule(rows, ARM, TRAIN_MICROBATCHES)
    if len({row["example_id"] for row in schedule}) != len(rows):
        raise RuntimeError("fixed V3 schedule did not expose every selected example")
    write_jsonl(args.processed_dir / ARM / "train.jsonl", rows)
    write_jsonl(args.processed_dir / ARM / "train_schedule.jsonl", schedule)

    manifest = {
        "protocol": PROTOCOL,
        "training_and_evaluation_protocol": "qlora_v2_frozen",
        "seed": SEED,
        "model": args.model,
        "resolved_model_revision": args.model_revision,
        "raw_file_sha256": observed_raw_hashes,
        "source_token_budgets": source_budgets,
        "selected_sft_tokens": sum(item["sft_token_cost"] for item in selected),
        "trace_ids": [item["trace_id"] for item in selected],
        "traces": [
            {
                "trace_id": item["trace_id"],
                "task_key": item["task_key"],
                "source": item["source"],
                "sft_token_cost": item["sft_token_cost"],
            }
            for item in selected
        ],
        "selection_audit": selection_audit,
    }
    manifest_path = args.selection_dir / f"{ARM}_manifest.json"
    write_json(manifest_path, manifest)

    summary = arm_stats(ARM, selected, schedule)
    summary.update(selection_audit["selected_stats"])
    summary["train_file_sha256"] = sha256_file(args.processed_dir / ARM / "train.jsonl")
    summary["train_schedule_sha256"] = sha256_file(args.processed_dir / ARM / "train_schedule.jsonl")
    if summary["train_file_sha256"] != EXPECTED_V3_TRAIN_SHA256:
        raise RuntimeError("V3 deterministic train-file hash drift")
    if summary["train_schedule_sha256"] != EXPECTED_V3_SCHEDULE_SHA256:
        raise RuntimeError("V3 deterministic schedule hash drift")

    contract = {
        "protocol": PROTOCOL,
        "training_and_evaluation_protocol": "qlora_v2_frozen",
        "seed": SEED,
        "claim_boundary": (
            "diagnostic offline held-out next-tool-call imitation on an already "
            "inspected V2 test set; not paper-final or executable Agent success"
        ),
        "data": {
            "split_unit": "task_key",
            "task_counts": {
                "train": len(train_tasks),
                "validation": len(validation_tasks),
                "test": len(test_tasks),
            },
            "successful_trajectories_only": True,
            "deduplication": "within_source_task_and_tool_sequence",
            "v2_label_rule_retained": True,
            "failed_action_labels_removed": False,
            "raw_file_sha256": observed_raw_hashes,
        },
        "model": {
            "name": args.model,
            "requested_revision": args.model_revision,
            "resolved_revision": args.model_revision,
        },
        "context": {
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "max_sequence_tokens": MAX_SEQ_LEN,
            "system_policy": "full_required",
            "history": "bounded_task_anchor_plus_contiguous_recent_messages",
        },
        "selection": {
            "arm": ARM,
            "anchor": "v2_random_success_training_selection",
            "uses_validation_outcomes": False,
            "uses_test_outcomes": False,
            "source_token_budgets": source_budgets,
            "total_token_budget": sum(source_budgets.values()),
            "search_profiles": list(SEARCH_PROFILES),
            "core_fractions": list(CORE_FRACTIONS),
            "constraint_tiers": list(CONSTRAINT_TIERS),
            "search_is_global_optimum": False,
            "selection_objective": (
                "lexicographic: distinct agent-recovery signatures, distinct "
                "recovery signatures, agent targets, lower scheduled and "
                "selected target-tool TVD, anchor task/token retention, task "
                "diversity, lower recovery volume"
            ),
            "audit": selection_audit,
        },
        "training": {
            "microbatches": TRAIN_MICROBATCHES,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "optimizer_steps": OPTIMIZER_STEPS,
            "pad_to_max_sequence_tokens": MAX_SEQ_LEN,
            "padded_tokens": TRAIN_MICROBATCHES * MAX_SEQ_LEN,
        },
        "shared_splits": {
            "validation_examples": sum(1 for line in validation_path.read_text(encoding="utf-8").splitlines() if line),
            "test_examples": sum(1 for line in test_path.read_text(encoding="utf-8").splitlines() if line),
        },
        "arm": summary,
        "hashes": {
            "validation_jsonl": sha256_file(validation_path),
            "test_jsonl": sha256_file(test_path),
            "manifest_json": sha256_file(manifest_path),
        },
    }
    write_json(args.processed_dir / "build_summary.json", contract)
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
