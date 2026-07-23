#!/usr/bin/env python3
"""Prepare the isolated, tokenizer-exact QLoRA v2 experiment.

V2 fixes the main validity problems found in v1.1:

* selection cost is measured with the exact model tokenizer after each
  trajectory is expanded into the SFT examples that training will consume;
* task splits happen before selection and source-model token quotas are shared
  by every arm;
* prompts always retain the complete system policy, a bounded initial user
  request, and bounded recent history using valid message records;
* every arm gets the same number of padded training microbatches and optimizer
  steps, while the selected-data token budget is reported separately.

This remains an offline next-tool-call imitation experiment.  It does not
measure executable or end-to-end agent success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SEED = 20260722
ARMS = ("random_success", "shortest_success", "recovery_coverage")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_PROMPT_TOKENS = 1664
MAX_COMPLETION_TOKENS = 384
MAX_SEQ_LEN = 2048
GRADIENT_ACCUMULATION = 16
SELECTION_FRACTION = 0.30

ERROR_RULES = {
    "not_found": re.compile(r"\b(not found|does not exist|unknown)\b", re.I),
    "invalid_argument": re.compile(r"\b(invalid|malformed|incorrect|must provide|missing)\b", re.I),
    "unauthorized": re.compile(r"\b(unauthorized|forbidden|not permitted)\b", re.I),
    "conflict": re.compile(r"\b(conflict|already|not available)\b", re.I),
    "generic_error": re.compile(r"^\s*error\s*[:\-]", re.I),
}


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return SEED ^ int.from_bytes(digest[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids(tokenizer, text: str, *, special: bool = False) -> list[int]:
    return tokenizer(text, add_special_tokens=special)["input_ids"]


def canonical_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return re.sub(r"\s+", " ", value.strip())
    if isinstance(value, dict):
        return {key: canonical_arguments(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical_arguments(item) for item in value]
    return value


def tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name"):
            result.append({
                "name": function["name"],
                "arguments": canonical_arguments(function.get("arguments", {})),
            })
    return result


def call_signature(call: dict[str, Any]) -> tuple[str, str]:
    return call["name"], json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def error_type(text: str) -> str | None:
    return next((name for name, rule in ERROR_RULES.items() if rule.search(text or "")), None)


def trace_id(record: dict[str, Any], source: str) -> str:
    return f"{source}:task{record['task_id']}:trial{record.get('trial', 0)}"


def load_success_records(data_dir: Path) -> list[dict[str, Any]]:
    """Deduplicate within source; cross-source histories remain distinct."""
    raw = []
    for path in sorted(data_dir.glob("*retail.json")):
        source = path.stem
        for record in json.loads(path.read_text(encoding="utf-8")):
            if record.get("reward") == 1:
                calls = [call_signature(call) for message in record["traj"] for call in tool_calls(message)]
                raw.append({
                    "record": record,
                    "source": source,
                    "trace_id": trace_id(record, source),
                    "task_key": f"retail:{record['task_id']}",
                    "dedup_signature": (source, f"retail:{record['task_id']}", tuple(calls)),
                })
    seen, result = set(), []
    for item in sorted(raw, key=lambda x: (x["task_key"], x["source"], x["trace_id"])):
        if item["dedup_signature"] not in seen:
            seen.add(item["dedup_signature"])
            result.append(item)
    return result


def task_split(records: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    tasks = sorted({item["task_key"] for item in records})
    random.Random(SEED).shuffle(tasks)
    n_train, n_validation = round(0.70 * len(tasks)), round(0.10 * len(tasks))
    return set(tasks[:n_train]), set(tasks[n_train:n_train + n_validation]), set(tasks[n_train + n_validation:])


def message_record(message: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message.get("role") or "unknown"}
    if message.get("name"):
        result["name"] = message["name"]
    if message.get("content") not in (None, ""):
        result["content"] = message["content"]
    calls = tool_calls(message)
    if calls:
        result["tool_calls"] = calls
    return result


def render_history_message(message: dict[str, Any]) -> str:
    return json.dumps(message_record(message), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def clip_content(tokenizer, content: str, max_tokens: int, mode: str) -> str:
    ids = token_ids(tokenizer, content)
    if len(ids) <= max_tokens:
        return content
    if max_tokens <= 0:
        return "[CONTENT_OMITTED]"
    marker = "[CONTENT_TRUNCATED]"
    marker_ids = token_ids(tokenizer, marker)
    keep = max(0, max_tokens - len(marker_ids))
    if mode == "head_tail":
        left = (keep + 1) // 2
        right = keep // 2
        kept = ids[:left] + ids[-right:] if right else ids[:left]
    else:
        kept = ids[-keep:] if keep else []
    decoded = tokenizer.decode(kept, skip_special_tokens=False)
    return marker + decoded


def clipped_message(tokenizer, message: dict[str, Any], max_content_tokens: int, mode: str) -> dict[str, Any]:
    result = dict(message)
    if result.get("content") not in (None, ""):
        result["content"] = clip_content(tokenizer, str(result["content"]), max_content_tokens, mode)
    return result


def assemble_prompt(system_text: str, task_message: dict[str, Any] | None, history: list[dict[str, Any]], omitted: bool) -> str:
    parts = ["<|system_policy|>", system_text, "<|end_system_policy|>"]
    if task_message is not None:
        parts.extend(["<|task_request|>", render_history_message(task_message), "<|end_task_request|>"])
    parts.append("<|recent_history|>")
    if omitted:
        parts.append('{"role":"meta","content":"[OLDER_HISTORY_OMITTED]"}')
    parts.extend(render_history_message(message) for message in history)
    parts.extend(["<|end_recent_history|>", "<|assistant_tool_call|>"])
    return "\n".join(parts) + "\n"


def structured_prompt(tokenizer, messages: list[dict[str, Any]], max_prompt_tokens: int) -> tuple[str, dict[str, Any]]:
    """Keep full system policy, bounded task anchor, and a contiguous recent suffix."""
    system_messages = [message for message in messages if message.get("role") == "system"]
    system_text = "\n".join(str(message.get("content") or "") for message in system_messages)
    if not system_text:
        raise ValueError("trajectory has no system policy")
    non_system = [message for message in messages if message.get("role") != "system"]
    task_index = next((i for i, message in enumerate(non_system) if message.get("role") == "user"), None)
    task_original = non_system[task_index] if task_index is not None else None
    newest_index = len(non_system) - 1 if non_system else None

    # These are content-token caps, not character slicing.  The outer JSON
    # message record always remains syntactically valid.
    task_caps = (256, 192, 128, 64, 32, 0)
    newest_caps = (384, 256, 192, 128, 64, 32, 0)
    base = None
    chosen_task = None
    chosen_newest = None
    for task_cap in task_caps:
        task = clipped_message(tokenizer, task_original, task_cap, "head_tail") if task_original is not None else None
        for newest_cap in newest_caps:
            newest = None
            if newest_index is not None and newest_index != task_index:
                newest = clipped_message(tokenizer, non_system[newest_index], newest_cap, "tail")
            history = [newest] if newest is not None else []
            candidate = assemble_prompt(system_text, task, history, omitted=True)
            if len(token_ids(tokenizer, candidate, special=True)) <= max_prompt_tokens:
                base, chosen_task, chosen_newest = candidate, task, newest
                break
        if base is not None:
            break
    if base is None:
        policy_tokens = len(token_ids(tokenizer, assemble_prompt(system_text, None, [], omitted=False), special=True))
        raise ValueError(f"full system policy cannot fit in max_prompt_tokens={max_prompt_tokens}; base tokens={policy_tokens}")

    selected_indices = set()
    if task_index is not None:
        selected_indices.add(task_index)
    if newest_index is not None and newest_index != task_index:
        selected_indices.add(newest_index)
    selected_messages: dict[int, dict[str, Any]] = {}
    if newest_index is not None and chosen_newest is not None:
        selected_messages[newest_index] = chosen_newest

    # Extend the recent suffix with whole message records. Stop at the first
    # record that does not fit; never skip around it or cut serialized JSON.
    if newest_index is not None:
        for index in range(newest_index - 1, -1, -1):
            if index == task_index:
                continue
            proposed = dict(selected_messages)
            proposed[index] = non_system[index]
            ordered = [proposed[i] for i in sorted(proposed)]
            omitted = len(selected_indices | set(proposed)) < len(non_system)
            candidate = assemble_prompt(system_text, chosen_task, ordered, omitted)
            if len(token_ids(tokenizer, candidate, special=True)) <= max_prompt_tokens:
                selected_messages = proposed
            else:
                break

    retained = selected_indices | set(selected_messages)
    omitted = len(retained) < len(non_system)
    prompt = assemble_prompt(system_text, chosen_task, [selected_messages[i] for i in sorted(selected_messages)], omitted)
    count = len(token_ids(tokenizer, prompt, special=True))
    if count > max_prompt_tokens:
        raise AssertionError(f"structured prompt has {count} tokens, cap is {max_prompt_tokens}")
    if system_text not in prompt:
        raise AssertionError("full system policy was not retained")
    return prompt, {
        "prompt_tokens": count,
        "history_messages_total": len(non_system),
        "history_messages_retained": len(retained),
        "history_truncated": omitted,
        "system_policy_retained_full": True,
        "task_anchor_retained": task_index is not None,
    }


def prior_recovery(traj: list[dict[str, Any]], target_index: int, target: dict[str, Any]) -> dict[str, Any]:
    tool_index = next((i for i in range(target_index - 1, -1, -1) if traj[i].get("role") == "tool"), None)
    if tool_index is None:
        return {"mode": "none"}
    failure_type = error_type(str(traj[tool_index].get("content") or ""))
    if failure_type is None:
        return {"mode": "none"}
    failed_index = next((i for i in range(tool_index - 1, -1, -1) if tool_calls(traj[i])), None)
    if failed_index is None:
        return {"mode": "none"}
    failed = tool_calls(traj[failed_index])[0]
    if call_signature(failed) == call_signature(target):
        return {"mode": "none"}
    user_assisted = any(message.get("role") == "user" for message in traj[tool_index + 1:target_index])
    return {
        "mode": "user_assisted" if user_assisted else "agent_initiated",
        "error_type": failure_type,
        "failed_tool": failed["name"],
        "failed_arguments": failed["arguments"],
        "repair_tool": target["name"],
    }


def examples_for_trace(item: dict[str, Any], tokenizer, max_prompt_tokens: int, max_completion_tokens: int, max_seq_len: int) -> list[dict[str, Any]]:
    traj = item["record"]["traj"]
    examples = []
    for index, message in enumerate(traj):
        calls = tool_calls(message)
        if message.get("role") != "assistant" or not calls:
            continue
        if len(calls) != 1:
            raise ValueError(f"v2 expects one tool call per assistant turn: {item['trace_id']} action {index}")
        target = calls[0]
        prompt, prompt_meta = structured_prompt(tokenizer, traj[:index], max_prompt_tokens)
        completion = json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        completion_tokens = len(token_ids(tokenizer, completion + tokenizer.eos_token))
        if completion_tokens > max_completion_tokens:
            raise ValueError(f"completion has {completion_tokens} tokens, cap is {max_completion_tokens}: {item['trace_id']} action {index}")
        sequence_tokens = prompt_meta["prompt_tokens"] + completion_tokens
        if sequence_tokens > max_seq_len:
            raise AssertionError(f"sequence has {sequence_tokens} tokens, cap is {max_seq_len}")
        recovery = prior_recovery(traj, index, target)
        examples.append({
            "example_id": f"{item['trace_id']}:action{index}",
            "trace_id": item["trace_id"],
            "task_key": item["task_key"],
            "source": item["source"],
            "target_message_index": index,
            "prompt": prompt,
            "completion": completion,
            "target_tool": target["name"],
            "recovery_mode": recovery["mode"],
            "is_error_resolution_target": recovery["mode"] != "none",
            "prior_error_type": recovery.get("error_type"),
            "failed_tool": recovery.get("failed_tool"),
            "repair_tool": recovery.get("repair_tool"),
            "prompt_tokens": prompt_meta["prompt_tokens"],
            "completion_tokens": completion_tokens,
            "sequence_tokens": sequence_tokens,
            "context_audit": {key: value for key, value in prompt_meta.items() if key != "prompt_tokens"},
        })
    return examples


def recovery_features(trace: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for example in trace["examples"]:
        mode = example["recovery_mode"]
        if mode == "none":
            continue
        error = example["prior_error_type"] or "unknown"
        failed = example["failed_tool"] or "unknown"
        repair = example["repair_tool"] or "unknown"
        weighted = {
            f"mode:{mode}": 4.0 if mode == "agent_initiated" else 1.0,
            f"error:{error}": 2.0,
            f"transition:{failed}->{repair}": 3.0,
            f"signature:{mode}|{error}|{failed}|{repair}": 5.0,
        }
        for feature, weight in weighted.items():
            features[feature] = max(features.get(feature, 0.0), weight)
    return features


def recovery_preference(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = list(records)
    counts: Counter[str] = Counter()
    ordered = []
    while remaining:
        def score(record: dict[str, Any]) -> tuple[float, int, str]:
            gain = sum(weight * (math.sqrt(counts[feature] + 1) - math.sqrt(counts[feature])) for feature, weight in record["recovery_features"].items())
            return gain / record["sft_token_cost"], -record["sft_token_cost"], record["trace_id"]
        best = max(remaining, key=score)
        remaining.remove(best)
        ordered.append(best)
        counts.update(best["recovery_features"])
    return ordered


def subset_fill(records: list[dict[str, Any]], capacity: int) -> list[dict[str, Any]]:
    """Find the highest reachable token sum <= capacity with integer bitsets."""
    candidates = [record for record in records if record["sft_token_cost"] <= capacity]
    if not candidates or capacity <= 0:
        return []
    reachable = 1
    mask = (1 << (capacity + 1)) - 1
    before = []
    for record in candidates:
        before.append(reachable)
        reachable |= (reachable << record["sft_token_cost"]) & mask
    target = reachable.bit_length() - 1
    selected = []
    for index in range(len(candidates) - 1, -1, -1):
        cost = candidates[index]["sft_token_cost"]
        if target >= cost and ((before[index] >> (target - cost)) & 1):
            selected.append(candidates[index])
            target -= cost
    return list(reversed(selected))


def select_to_budget(records: list[dict[str, Any]], budget: int, arm: str, source: str) -> list[dict[str, Any]]:
    if arm == "random_success":
        preferred = list(records)
        random.Random(stable_seed(arm, source)).shuffle(preferred)
    elif arm == "shortest_success":
        preferred = sorted(records, key=lambda item: (item["sft_token_cost"], item["trace_id"]))
    elif arm == "recovery_coverage":
        preferred = recovery_preference(records)
    else:
        raise ValueError(arm)

    core_target = int(0.90 * budget)
    core, used = [], 0
    for record in preferred:
        if used >= core_target:
            break
        if used + record["sft_token_cost"] <= budget:
            core.append(record)
            used += record["sft_token_cost"]
    core_ids = {record["trace_id"] for record in core}
    filler_candidates = [record for record in preferred if record["trace_id"] not in core_ids]
    filler = subset_fill(filler_candidates, budget - used)
    selected = core + filler
    if len({record["trace_id"] for record in selected}) != len(selected):
        raise AssertionError("duplicate selected trajectory")
    return selected


def common_source_selection(source_records: list[dict[str, Any]], initial_budget: int, source: str) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    """Lower a quota until all arm-specific subset sums hit the same budget."""
    budget = initial_budget
    selections = {}
    for _ in range(32):
        selections = {arm: select_to_budget(source_records, budget, arm, source) for arm in ARMS}
        realized = {arm: sum(record["sft_token_cost"] for record in selected) for arm, selected in selections.items()}
        common = min(realized.values())
        if len(set(realized.values())) == 1:
            return common, selections
        if common >= budget:
            break
        budget = common
    spread = max(sum(x["sft_token_cost"] for x in selected) for selected in selections.values()) - min(sum(x["sft_token_cost"] for x in selected) for selected in selections.values())
    if spread > max(32, int(0.0001 * initial_budget)):
        raise RuntimeError(f"could not equalize {source} selections: spread={spread} tokens")
    common = min(sum(x["sft_token_cost"] for x in selected) for selected in selections.values())
    final = {arm: select_to_budget(source_records, common, arm, source) for arm in ARMS}
    return common, final


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fixed_schedule(rows: list[dict[str, Any]], arm: str, microbatches: int) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"{arm} has no training examples")
    rng = random.Random(stable_seed("schedule", arm))
    pool = list(rows)
    schedule = []
    pass_index = 0
    while len(schedule) < microbatches:
        rng.shuffle(pool)
        take = min(len(pool), microbatches - len(schedule))
        for row in pool[:take]:
            scheduled = dict(row)
            scheduled["schedule_index"] = len(schedule)
            scheduled["schedule_pass"] = pass_index
            schedule.append(scheduled)
        pass_index += 1
    return schedule


def arm_stats(arm: str, selected: list[dict[str, Any]], schedule: list[dict[str, Any]]) -> dict[str, Any]:
    examples = [example for record in selected for example in record["examples"]]
    return {
        "arm": arm,
        "selected_sft_tokens": sum(record["sft_token_cost"] for record in selected),
        "trajectories": len(selected),
        "unique_tasks": len({record["task_key"] for record in selected}),
        "examples": len(examples),
        "scheduled_microbatches": len(schedule),
        "scheduled_unique_examples": len({row["example_id"] for row in schedule}),
        "scheduled_nonpad_tokens": sum(row["sequence_tokens"] for row in schedule),
        "scheduled_loss_tokens": sum(row["completion_tokens"] for row in schedule),
        "scheduled_padded_tokens": len(schedule) * MAX_SEQ_LEN,
        "recovery_targets": sum(row["recovery_mode"] != "none" for row in examples),
        "agent_initiated_targets": sum(row["recovery_mode"] == "agent_initiated" for row in examples),
        "user_assisted_targets": sum(row["recovery_mode"] == "user_assisted" for row in examples),
        "source_trajectory_counts": dict(sorted(Counter(record["source"] for record in selected).items())),
        "source_token_counts": dict(sorted((source, sum(record["sft_token_cost"] for record in selected if record["source"] == source)) for source in {record["source"] for record in selected})),
        "mean_sequence_tokens": round(sum(row["sequence_tokens"] for row in examples) / len(examples), 2),
        "truncated_context_examples": sum(row["context_audit"]["history_truncated"] for row in examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, default=Path("results/selection_v2"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/qlora_v2"))
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
    # Preparation defaults to an immutable Hugging Face commit.  Do not turn a
    # moving branch such as "main" into a falsely precise resolved revision.
    resolved_revision = args.model_revision

    records = load_success_records(args.data_dir)
    train_tasks, validation_tasks, test_tasks = task_split(records)
    for item in records:
        item["examples"] = examples_for_trace(item, tokenizer, MAX_PROMPT_TOKENS, MAX_COMPLETION_TOKENS, MAX_SEQ_LEN)
        item["sft_token_cost"] = sum(example["sequence_tokens"] for example in item["examples"])
        item["recovery_features"] = recovery_features(item)
    records = [item for item in records if item["examples"]]
    train_records = [item for item in records if item["task_key"] in train_tasks]

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in train_records:
        by_source[item["source"]].append(item)
    source_budgets, source_selections = {}, {}
    for source, source_records in sorted(by_source.items()):
        initial = round(SELECTION_FRACTION * sum(record["sft_token_cost"] for record in source_records))
        source_budgets[source], source_selections[source] = common_source_selection(source_records, initial, source)

    args.selection_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    shared_rows = {}
    for split, tasks in (("validation", validation_tasks), ("test", test_tasks)):
        rows = [example for item in records if item["task_key"] in tasks for example in item["examples"]]
        shared_rows[split] = rows
        write_jsonl(args.processed_dir / "shared" / f"{split}.jsonl", rows)

    selected_by_arm, rows_by_arm = {}, {}
    for arm in ARMS:
        selected = [record for source in sorted(source_selections) for record in source_selections[source][arm]]
        selected.sort(key=lambda item: item["trace_id"])
        rows = [example for item in selected for example in item["examples"]]
        selected_by_arm[arm] = selected
        rows_by_arm[arm] = rows

    # The shared schedule is the smallest full optimizer-step multiple that is
    # large enough to expose every selected example in every arm at least once.
    train_microbatches = math.ceil(max(len(rows) for rows in rows_by_arm.values()) / GRADIENT_ACCUMULATION) * GRADIENT_ACCUMULATION
    optimizer_steps = train_microbatches // GRADIENT_ACCUMULATION
    summaries = []
    for arm in ARMS:
        selected = selected_by_arm[arm]
        rows = rows_by_arm[arm]
        schedule = fixed_schedule(rows, arm, train_microbatches)
        write_jsonl(args.processed_dir / arm / "train.jsonl", rows)
        write_jsonl(args.processed_dir / arm / "train_schedule.jsonl", schedule)
        manifest = {
            "protocol": "qlora_v2",
            "seed": SEED,
            "model": args.model,
            "resolved_model_revision": resolved_revision,
            "source_token_budgets": source_budgets,
            "selected_sft_tokens": sum(item["sft_token_cost"] for item in selected),
            "trace_ids": [item["trace_id"] for item in selected],
            "traces": [{"trace_id": item["trace_id"], "task_key": item["task_key"], "source": item["source"], "sft_token_cost": item["sft_token_cost"]} for item in selected],
        }
        (args.selection_dir / f"{arm}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        stats = arm_stats(arm, selected, schedule)
        stats["train_file_sha256"] = sha256_file(args.processed_dir / arm / "train.jsonl")
        stats["train_schedule_sha256"] = sha256_file(args.processed_dir / arm / "train_schedule.jsonl")
        summaries.append(stats)

    selected_token_totals = {row["selected_sft_tokens"] for row in summaries}
    if len(selected_token_totals) != 1:
        raise AssertionError(f"v2 selected token totals differ across arms: {selected_token_totals}")
    for source in source_budgets:
        observed = {row["source_token_counts"].get(source, 0) for row in summaries}
        if len(observed) != 1:
            raise AssertionError(f"v2 source-token totals differ for {source}: {observed}")

    contract = {
        "protocol": "qlora_v2",
        "seed": SEED,
        "claim_boundary": "offline held-out next-tool-call imitation only",
        "data": {
            "split_unit": "task_key",
            "task_counts": {"train": len(train_tasks), "validation": len(validation_tasks), "test": len(test_tasks)},
            "successful_trajectories_only": True,
            "deduplication": "within_source_task_and_tool_sequence",
        },
        "model": {"name": args.model, "requested_revision": args.model_revision, "resolved_revision": resolved_revision},
        "context": {
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "max_sequence_tokens": MAX_SEQ_LEN,
            "system_policy": "full_required",
            "history": "bounded_task_anchor_plus_contiguous_recent_messages",
        },
        "selection": {
            "cost": "sum_of_final_prompt_plus_completion_tokens_after_sft_expansion",
            "fraction": SELECTION_FRACTION,
            "source_token_budgets": source_budgets,
            "total_token_budget": sum(source_budgets.values()),
            "arms": list(ARMS),
        },
        "training": {
            "microbatches": train_microbatches,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "optimizer_steps": optimizer_steps,
            "pad_to_max_sequence_tokens": MAX_SEQ_LEN,
            "padded_tokens_per_arm": train_microbatches * MAX_SEQ_LEN,
        },
        "shared_splits": {"validation_examples": len(shared_rows["validation"]), "test_examples": len(shared_rows["test"])},
        "arms": summaries,
    }
    (args.processed_dir / "build_summary.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    contract["hashes"] = {
        "validation_jsonl": sha256_file(args.processed_dir / "shared" / "validation.jsonl"),
        "test_jsonl": sha256_file(args.processed_dir / "shared" / "test.jsonl"),
    }
    (args.processed_dir / "build_summary.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
