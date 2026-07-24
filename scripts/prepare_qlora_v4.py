#!/usr/bin/env python3
"""Build and audit the V4 clean-SFT and recovery-preference datasets.

V4 deliberately keeps the frozen V3 constrained-recovery trajectory set and
changes the supervision objective.  It produces:

* a matched clean-SFT schedule in which calls that immediately returned an
  explicit tool error are replaced by successful calls with the same source
  and target tool;
* same-context preference pairs whose chosen call is an observed successful
  repair and whose rejected call replays the observed failed call after its
  error has been seen;
* validation and test-only metadata for objective-aligned diagnostics.

The repair result is used only to accept or reject a training label.  It is
never included in the prompt.  All pair construction is deterministic and
uses train tasks only.  The already inspected V2/V3 test remains exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
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
    call_signature,
    error_type,
    load_success_records,
    render_history_message,
    structured_prompt,
    task_split,
    token_ids,
    tool_calls,
)
from prepare_qlora_v3 import (
    EXPECTED_RAW_SHA256,
    EXPECTED_TEST_SHA256,
    EXPECTED_V3_SCHEDULE_SHA256,
    EXPECTED_V3_STATS,
    EXPECTED_V3_TRACE_SHA256,
    EXPECTED_V3_TRAIN_SHA256,
    EXPECTED_VALIDATION_SHA256,
    sha256_file,
    trace_set_fingerprint,
    write_json,
    write_jsonl,
)

PROTOCOL = "qlora_v4_clean_sft_recovery_dpo"
V3_ARM = "constrained_recovery"
CLEAN_SFT_MICROBATCHES = 1088
CLEAN_SFT_OPTIMIZER_STEPS = 68
PREFERENCE_MICROBATCHES = 144
PREFERENCE_GRADIENT_ACCUMULATION = 8
PREFERENCE_OPTIMIZER_STEPS = 18
PREFERENCE_EXPOSURES_PER_MODE = 72

EXPECTED_COUNTS = {
    "v3_unique_train": 1069,
    "clean_unique_train": 965,
    "failed_unique_train": 104,
    "clean_validation": 360,
    "failed_validation": 32,
    "strict_train_pairs": 79,
    "strict_validation_pairs": 24,
    "outcome_success_test": 902,
    "non_recovery_success_test": 852,
    "failed_test": 57,
    "recovery_success_test": 50,
    "strict_test_pairs": 48,
    "preference_smoke_pairs": 16,
}
EXPECTED_OUTPUT_SHA256 = {
    "clean_train_unique_sha256": "1bfa3d9df8e38e6a97237aa6efb47cc4bfacd9a15a1a396acdbf213b3f7ca1e8",
    "clean_train_schedule_sha256": "4b28da48082ef5bd3396e7df4b5b723c4efffe4b2e5438f47c8c2ca9d709f386",
    "clean_validation_sha256": "e1942db9c2766c44f3f59c7baa47a7412d4b8a8992a185d306f346a8c5c9fc53",
    "replacement_map_sha256": "f5b0ec35e7ffb46ac47a83c525906257d7012af268dc1164394250c3bfc7fceb",
    "preference_train_pairs_sha256": "f6b967d5decb4741e3b1fbee2c0a0b3ac4760dfdfb76d910fd0b6cf7d0adefe5",
    "preference_train_schedule_sha256": "f3cd0565cab0fd12252512b018a749dbe2c42a89d15ec92efd6b03a18f521341",
    "preference_validation_pairs_sha256": "095a235caa2f38d9d9b3b9e856792506bf0360464b843c848fe615eb841c59e0",
    "preference_smoke_pairs_sha256": "86fd923875ba3d11c50d635c409246e6f09437c771a4b4881c02cbba47190eb4",
    "test_outcomes_sha256": "9a4ec2b1e25ee512e5946e8ac770b0fbf6b0ed0d5b61f994c643fc04cd227b57",
    "test_preference_pairs_sha256": "b85548e9f1c041032358172e10b7f7f53f91710d1d15f26dfaa606a07799cf74",
}

TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "temporarily",
    "rate limit",
    "network",
    "connection",
    "try again",
    "unavailable",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_completion(call: dict[str, Any]) -> str:
    return json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tool_call_id(message: dict[str, Any]) -> str | None:
    raw = message.get("tool_calls") or []
    return raw[0].get("id") if len(raw) == 1 else None


def deterministic_error(content: str) -> bool:
    normalized = content.strip().lower()
    return (
        normalized.startswith("error:")
        and not any(marker in normalized for marker in TRANSIENT_ERROR_MARKERS)
    )


def linked_tool_outcome(
    item: dict[str, Any],
    target_index: int,
) -> dict[str, Any]:
    """Link one assistant call to the immediately following tool result.

    The pinned tau-bench files satisfy this structure for every tool call.  V4
    fails closed if a future source revision does not.
    """
    trajectory = item["record"]["traj"]
    if target_index < 0 or target_index >= len(trajectory):
        raise ValueError(f"target index out of range: {item['trace_id']}:{target_index}")
    target_message = trajectory[target_index]
    calls = tool_calls(target_message)
    if target_message.get("role") != "assistant" or len(calls) != 1:
        raise ValueError(
            f"target is not exactly one assistant tool call: "
            f"{item['trace_id']}:{target_index}"
        )
    result_index = target_index + 1
    if result_index >= len(trajectory):
        raise ValueError(f"tool result missing: {item['trace_id']}:{target_index}")
    result = trajectory[result_index]
    if result.get("role") != "tool":
        raise ValueError(
            f"tool result is not adjacent: {item['trace_id']}:{target_index}"
        )
    expected_id = tool_call_id(target_message)
    observed_id = result.get("tool_call_id")
    if not expected_id or observed_id != expected_id:
        raise ValueError(
            f"tool call/result id mismatch: {item['trace_id']}:{target_index}"
        )
    if result.get("name") != calls[0]["name"]:
        raise ValueError(
            f"tool call/result name mismatch: {item['trace_id']}:{target_index}"
        )
    content = str(result.get("content") or "")
    is_error = content.lstrip().startswith("Error:")
    classified = error_type(content)
    if is_error and classified is None:
        raise ValueError(
            f"explicit Error result is outside frozen classifier: "
            f"{item['trace_id']}:{target_index}"
        )
    if not is_error and classified is not None:
        raise ValueError(
            f"broad error rule disagrees with exact Error prefix: "
            f"{item['trace_id']}:{target_index}"
        )
    return {
        "status": "error" if is_error else "success",
        "result_index": result_index,
        "tool_call_id": expected_id,
        "tool_name": calls[0]["name"],
        "content": content,
        "error_type": classified,
        "deterministic_error": deterministic_error(content) if is_error else None,
    }


def validate_and_annotate_rows(
    rows: list[dict[str, Any]],
    by_trace: dict[str, dict[str, Any]],
    tokenizer: Any,
    split_name: str,
    *,
    allow_duplicate_ids: bool = False,
) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for row in rows:
        example_id = row.get("example_id")
        if not example_id or (example_id in seen and not allow_duplicate_ids):
            raise ValueError(f"{split_name} has missing/duplicate example_id: {example_id}")
        seen.add(example_id)
        item = by_trace.get(row.get("trace_id"))
        if item is None:
            raise ValueError(f"{split_name} trace missing from pinned raw data: {row.get('trace_id')}")
        index = row.get("target_message_index")
        if not isinstance(index, int):
            raise ValueError(f"{example_id} has no integer target_message_index")
        trajectory = item["record"]["traj"]
        calls = tool_calls(trajectory[index])
        if len(calls) != 1 or canonical_completion(calls[0]) != row.get("completion"):
            raise ValueError(f"{example_id} completion does not match raw target call")
        rebuilt_prompt, prompt_meta = structured_prompt(
            tokenizer,
            trajectory[:index],
            MAX_PROMPT_TOKENS,
        )
        if rebuilt_prompt != row.get("prompt"):
            raise ValueError(f"{example_id} prompt differs from prefix-only reconstruction")
        completion_tokens = len(
            token_ids(tokenizer, row["completion"] + tokenizer.eos_token)
        )
        if (
            prompt_meta["prompt_tokens"] != row.get("prompt_tokens")
            or completion_tokens != row.get("completion_tokens")
            or prompt_meta["prompt_tokens"] + completion_tokens
            != row.get("sequence_tokens")
        ):
            raise ValueError(f"{example_id} token contract drift")
        outcome = linked_tool_outcome(item, index)
        if "target_leads_to_error" in row:
            if bool(row["target_leads_to_error"]) != (outcome["status"] == "error"):
                raise ValueError(f"{example_id} V3 failed-label annotation drift")
        annotated = dict(row)
        annotated.update(
            {
                "linked_tool_outcome": outcome["status"],
                "linked_tool_result_index": outcome["result_index"],
                "linked_tool_call_id": outcome["tool_call_id"],
                "linked_tool_result_error_type": outcome["error_type"],
            }
        )
        result.append(annotated)
    return result


def build_preference_pair(
    row: dict[str, Any],
    item: dict[str, Any],
    tokenizer: Any,
    split_name: str,
) -> tuple[dict[str, Any] | None, str]:
    if row.get("recovery_mode") not in ("agent_initiated", "user_assisted"):
        return None, "not_recovery"
    if row["linked_tool_outcome"] != "success":
        return None, "chosen_failed"

    trajectory = item["record"]["traj"]
    target_index = row["target_message_index"]
    error_index = next(
        (
            index
            for index in range(target_index - 1, -1, -1)
            if trajectory[index].get("role") == "tool"
        ),
        None,
    )
    if error_index is None:
        return None, "no_prior_tool_result"
    failed_index = error_index - 1
    if failed_index < 0:
        return None, "no_failed_call"
    failed_outcome = linked_tool_outcome(item, failed_index)
    if failed_outcome["result_index"] != error_index:
        return None, "nonadjacent_failed_result"
    if failed_outcome["status"] != "error":
        return None, "prior_result_not_error"
    if not failed_outcome["deterministic_error"]:
        return None, "transient_or_unclassified_error"
    if any(
        tool_calls(message)
        for message in trajectory[error_index + 1 : target_index]
    ):
        return None, "intervening_tool_call"

    failed_calls = tool_calls(trajectory[failed_index])
    chosen_calls = tool_calls(trajectory[target_index])
    if len(failed_calls) != 1 or len(chosen_calls) != 1:
        return None, "call_cardinality"
    failed, chosen = failed_calls[0], chosen_calls[0]
    if call_signature(failed) == call_signature(chosen):
        return None, "chosen_equals_rejected"

    failed_line = render_history_message(trajectory[failed_index])
    error_line = render_history_message(trajectory[error_index])
    if failed_line not in row["prompt"] or error_line not in row["prompt"]:
        return None, "required_error_context_truncated"

    chosen_text = canonical_completion(chosen)
    rejected_text = canonical_completion(failed)
    if chosen_text != row["completion"]:
        raise ValueError(f"{row['example_id']} chosen completion drift")
    rejected_tokens = len(
        token_ids(tokenizer, rejected_text + tokenizer.eos_token)
    )
    if rejected_tokens > MAX_COMPLETION_TOKENS:
        return None, "rejected_completion_too_long"
    if row["prompt_tokens"] + rejected_tokens > MAX_SEQ_LEN:
        return None, "rejected_sequence_too_long"

    user_between = any(
        message.get("role") == "user"
        for message in trajectory[error_index + 1 : target_index]
    )
    derived_mode = "user_assisted" if user_between else "agent_initiated"
    if derived_mode != row["recovery_mode"]:
        raise ValueError(f"{row['example_id']} recovery-mode drift")

    chosen_outcome = linked_tool_outcome(item, target_index)
    pair_id = f"{row['example_id']}:prefer_repair_over_action{failed_index}"
    pair = {
        "pair_id": pair_id,
        "example_id": row["example_id"],
        "trace_id": row["trace_id"],
        "task_key": row["task_key"],
        "source": row["source"],
        "split": split_name,
        "prompt": row["prompt"],
        "chosen": chosen_text,
        "rejected": rejected_text,
        "prompt_tokens": row["prompt_tokens"],
        "chosen_tokens": row["completion_tokens"],
        "rejected_tokens": rejected_tokens,
        "chosen_sequence_tokens": row["prompt_tokens"] + row["completion_tokens"],
        "rejected_sequence_tokens": row["prompt_tokens"] + rejected_tokens,
        "recovery_mode": row["recovery_mode"],
        "error_type": failed_outcome["error_type"],
        "failed_tool": failed["name"],
        "repair_tool": chosen["name"],
        "failed_call_index": failed_index,
        "error_result_index": error_index,
        "repair_message_index": target_index,
        "chosen_result_index": chosen_outcome["result_index"],
        "rejected_observed_result_status": "error",
        "chosen_observed_result_status": "success",
        "rejected_origin": "counterfactual_replay_of_observed_failed_call",
        "prompt_sha256": text_sha256(row["prompt"]),
        "future_leakage_audit": {
            "prompt_source_max_index": target_index - 1,
            "chosen_index": target_index,
            "chosen_result_index": chosen_outcome["result_index"],
            "chosen_result_in_source_prefix": False,
            "failed_call_retained_verbatim": True,
            "error_result_retained_verbatim": True,
        },
    }
    return pair, "accepted"


def build_pairs(
    rows: list[dict[str, Any]],
    by_trace: dict[str, dict[str, Any]],
    tokenizer: Any,
    split_name: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pairs = []
    reasons: Counter[str] = Counter()
    for row in rows:
        item = by_trace[row["trace_id"]]
        pair, reason = build_preference_pair(row, item, tokenizer, split_name)
        reasons[reason] += 1
        if pair is not None:
            pairs.append(pair)
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError(f"{split_name} has duplicate preference pair ids")
    return pairs, reasons


def matched_clean_schedule(
    annotated_schedule: list[dict[str, Any]],
    clean_unique: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean_unique:
        pools[(row["source"], row["target_tool"])].append(row)
    for pool in pools.values():
        pool.sort(key=lambda row: row["example_id"])

    total_exposure: Counter[str] = Counter(
        row["example_id"]
        for row in annotated_schedule
        if row["linked_tool_outcome"] == "success"
    )
    replacement_exposure: Counter[str] = Counter()
    output, replacement_map = [], []
    for original in annotated_schedule:
        if original["linked_tool_outcome"] == "success":
            output.append(dict(original))
            continue
        key = (original["source"], original["target_tool"])
        candidates = pools.get(key) or []
        if not candidates:
            raise ValueError(
                f"no clean matched replacement for source/tool {key}"
            )
        chosen = min(
            candidates,
            key=lambda row: (
                row["recovery_mode"] != original["recovery_mode"],
                total_exposure[row["example_id"]],
                abs(row["completion_tokens"] - original["completion_tokens"]),
                abs(row["sequence_tokens"] - original["sequence_tokens"]),
                row["example_id"],
            ),
        )
        replacement = dict(chosen)
        replacement["schedule_index"] = original["schedule_index"]
        replacement["schedule_pass"] = original["schedule_pass"]
        replacement["v4_clean_replacement"] = {
            "removed_example_id": original["example_id"],
            "removed_error_type": original["linked_tool_result_error_type"],
            "hard_match_keys": ["source", "target_tool"],
            "preferred_match_key": "recovery_mode",
        }
        output.append(replacement)
        total_exposure[chosen["example_id"]] += 1
        replacement_exposure[chosen["example_id"]] += 1
        replacement_map.append(
            {
                "schedule_index": original["schedule_index"],
                "removed_example_id": original["example_id"],
                "replacement_example_id": chosen["example_id"],
                "source": original["source"],
                "target_tool": original["target_tool"],
                "removed_sequence_tokens": original["sequence_tokens"],
                "replacement_sequence_tokens": chosen["sequence_tokens"],
                "removed_completion_tokens": original["completion_tokens"],
                "replacement_completion_tokens": chosen["completion_tokens"],
                "removed_recovery_mode": original["recovery_mode"],
                "replacement_recovery_mode": chosen["recovery_mode"],
            }
        )
    if len(output) != CLEAN_SFT_MICROBATCHES:
        raise ValueError(
            f"clean schedule has {len(output)} rows, expected {CLEAN_SFT_MICROBATCHES}"
        )
    if any(row["linked_tool_outcome"] != "success" for row in output):
        raise ValueError("clean schedule still contains a failed-action label")
    return output, replacement_map


def balanced_preference_schedule(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_mode = {
        mode: sorted(
            (pair for pair in pairs if pair["recovery_mode"] == mode),
            key=lambda pair: pair["pair_id"],
        )
        for mode in ("agent_initiated", "user_assisted")
    }
    scheduled_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode, pool in by_mode.items():
        if not pool:
            raise ValueError(f"no {mode} preference pairs")
        rng = random.Random(
            SEED
            ^ int.from_bytes(
                hashlib.sha256(f"v4|preference|{mode}".encode()).digest()[:8],
                "big",
            )
        )
        scheduled = []
        pass_index = 0
        while len(scheduled) < PREFERENCE_EXPOSURES_PER_MODE:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            take = min(
                len(shuffled),
                PREFERENCE_EXPOSURES_PER_MODE - len(scheduled),
            )
            for pair in shuffled[:take]:
                row = dict(pair)
                row["mode_pass"] = pass_index
                scheduled.append(row)
            pass_index += 1
        scheduled_by_mode[mode] = scheduled

    output = []
    for index in range(PREFERENCE_EXPOSURES_PER_MODE):
        for mode in ("agent_initiated", "user_assisted"):
            row = dict(scheduled_by_mode[mode][index])
            row["schedule_index"] = len(output)
            output.append(row)
    if len(output) != PREFERENCE_MICROBATCHES:
        raise AssertionError("preference schedule length drift")
    return output


def preference_smoke_pairs(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Choose 16 deterministic train pairs covering length and group extremes."""
    if len(pairs) < 16:
        raise ValueError("at least 16 strict train pairs are required for smoke")

    def longest_key(pair: dict[str, Any]) -> tuple[int, str]:
        return (
            max(
                pair["chosen_sequence_tokens"],
                pair["rejected_sequence_tokens"],
            ),
            pair["pair_id"],
        )

    longest = max(pairs, key=longest_key)
    required = [
        longest,
        max(pairs, key=lambda pair: (pair["prompt_tokens"], pair["pair_id"])),
        max(pairs, key=lambda pair: (pair["chosen_tokens"], pair["pair_id"])),
        max(pairs, key=lambda pair: (pair["rejected_tokens"], pair["pair_id"])),
        min(pairs, key=lambda pair: (pair["prompt_tokens"], pair["pair_id"])),
    ]
    for mode in ("agent_initiated", "user_assisted"):
        pool = [pair for pair in pairs if pair["recovery_mode"] == mode]
        required.append(max(pool, key=longest_key))
    for kind in sorted({pair["error_type"] for pair in pairs}):
        pool = [pair for pair in pairs if pair["error_type"] == kind]
        required.append(max(pool, key=longest_key))

    selected: dict[str, dict[str, Any]] = {
        pair["pair_id"]: pair for pair in required
    }
    for pair in sorted(pairs, key=longest_key, reverse=True):
        if len(selected) >= 16:
            break
        selected.setdefault(pair["pair_id"], pair)
    rows = []
    for index, pair in enumerate(
        sorted(selected.values(), key=longest_key, reverse=True)
    ):
        row = dict(pair)
        row["smoke_index"] = index
        row["is_longest_pair"] = pair["pair_id"] == longest["pair_id"]
        rows.append(row)
    if len(rows) != 16:
        raise AssertionError(f"smoke selection has {len(rows)} rows, expected 16")
    return rows, longest["pair_id"]


def tool_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["target_tool"] for row in rows).items()))


def source_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["source"] for row in rows).items()))


def count_rows(rows: Iterable[dict[str, Any]], key: str, value: Any) -> int:
    return sum(row.get(key) == value for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--v3-selection-manifest",
        type=Path,
        default=Path("results/selection_v3/constrained_recovery_manifest.json"),
    )
    parser.add_argument(
        "--v3-processed-dir",
        type=Path,
        default=Path("data/processed/qlora_v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/qlora_v4"),
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    for filename, expected in EXPECTED_RAW_SHA256.items():
        observed = sha256_file(args.data_dir / filename)
        if observed != expected:
            raise ValueError(
                f"raw file hash drift for {filename}: {observed} != {expected}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("tokenizer must define eos_token")

    manifest = json.loads(
        args.v3_selection_manifest.read_text(encoding="utf-8")
    )
    trace_ids = manifest.get("trace_ids") or []
    if trace_set_fingerprint(trace_ids) != EXPECTED_V3_TRACE_SHA256:
        raise ValueError("V3 selection trace-set hash drift")

    records = load_success_records(args.data_dir)
    by_trace = {item["trace_id"]: item for item in records}
    train_tasks, validation_tasks, test_tasks = task_split(records)
    selected = [by_trace[trace_id] for trace_id in trace_ids]
    if any(item["task_key"] not in train_tasks for item in selected):
        raise ValueError("V3 selection contains a non-train task")

    v3_train_path = (
        args.v3_processed_dir / V3_ARM / "train.jsonl"
    )
    v3_schedule_path = (
        args.v3_processed_dir / V3_ARM / "train_schedule.jsonl"
    )
    validation_path = args.v3_processed_dir / "shared" / "validation.jsonl"
    test_path = args.v3_processed_dir / "shared" / "test.jsonl"
    expected_hashes = {
        v3_train_path: EXPECTED_V3_TRAIN_SHA256,
        v3_schedule_path: EXPECTED_V3_SCHEDULE_SHA256,
        validation_path: EXPECTED_VALIDATION_SHA256,
        test_path: EXPECTED_TEST_SHA256,
    }
    for path, expected in expected_hashes.items():
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"frozen V3 input hash drift for {path}: {observed}")

    v3_train = validate_and_annotate_rows(
        read_jsonl(v3_train_path),
        by_trace,
        tokenizer,
        "train",
    )
    v3_schedule = validate_and_annotate_rows(
        read_jsonl(v3_schedule_path),
        by_trace,
        tokenizer,
        "train_schedule",
        allow_duplicate_ids=True,
    )
    validation = validate_and_annotate_rows(
        read_jsonl(validation_path),
        by_trace,
        tokenizer,
        "validation",
    )
    test = validate_and_annotate_rows(
        read_jsonl(test_path),
        by_trace,
        tokenizer,
        "test",
    )

    train_task_keys = {row["task_key"] for row in v3_train}
    validation_task_keys = {row["task_key"] for row in validation}
    test_task_keys = {row["task_key"] for row in test}
    if train_task_keys & validation_task_keys or train_task_keys & test_task_keys:
        raise ValueError("train task leakage into validation/test")
    if validation_task_keys & test_task_keys:
        raise ValueError("validation/test task overlap")
    if validation_task_keys != validation_tasks or test_task_keys != test_tasks:
        raise ValueError("shared split task membership drift")

    clean_unique = [
        row for row in v3_train if row["linked_tool_outcome"] == "success"
    ]
    failed_unique = [
        row for row in v3_train if row["linked_tool_outcome"] == "error"
    ]
    clean_validation = [
        row for row in validation if row["linked_tool_outcome"] == "success"
    ]
    clean_schedule, replacement_map = matched_clean_schedule(
        v3_schedule,
        clean_unique,
    )
    recovery_mode_matched_replacements = sum(
        row["removed_recovery_mode"] == row["replacement_recovery_mode"]
        for row in replacement_map
    )
    if recovery_mode_matched_replacements != 103:
        raise ValueError(
            "frozen recovery-mode replacement match count drift: "
            f"{recovery_mode_matched_replacements} != 103"
        )

    train_pairs, train_pair_reasons = build_pairs(
        v3_train,
        by_trace,
        tokenizer,
        "train",
    )
    validation_pairs, validation_pair_reasons = build_pairs(
        validation,
        by_trace,
        tokenizer,
        "validation",
    )
    test_pairs, test_pair_reasons = build_pairs(
        test,
        by_trace,
        tokenizer,
        "test",
    )
    preference_schedule = balanced_preference_schedule(train_pairs)
    smoke_pairs, longest_pair_id = preference_smoke_pairs(train_pairs)

    test_outcomes = [
        {
            "example_id": row["example_id"],
            "trace_id": row["trace_id"],
            "task_key": row["task_key"],
            "target_tool": row["target_tool"],
            "recovery_mode": row["recovery_mode"],
            "linked_tool_outcome": row["linked_tool_outcome"],
            "linked_tool_result_error_type": row[
                "linked_tool_result_error_type"
            ],
            "is_outcome_success_target": row["linked_tool_outcome"] == "success",
            "is_non_recovery_success_target": (
                row["recovery_mode"] == "none"
                and row["linked_tool_outcome"] == "success"
            ),
            "is_recovery_success_target": (
                row["recovery_mode"] != "none"
                and row["linked_tool_outcome"] == "success"
            ),
        }
        for row in test
    ]

    observed_counts = {
        "v3_unique_train": len(v3_train),
        "clean_unique_train": len(clean_unique),
        "failed_unique_train": len(failed_unique),
        "clean_validation": len(clean_validation),
        "failed_validation": len(validation) - len(clean_validation),
        "strict_train_pairs": len(train_pairs),
        "strict_validation_pairs": len(validation_pairs),
        "outcome_success_test": count_rows(
            test,
            "linked_tool_outcome",
            "success",
        ),
        "non_recovery_success_test": sum(
            row["linked_tool_outcome"] == "success"
            and row["recovery_mode"] == "none"
            for row in test
        ),
        "failed_test": count_rows(test, "linked_tool_outcome", "error"),
        "recovery_success_test": sum(
            row["linked_tool_outcome"] == "success"
            and row["recovery_mode"] != "none"
            for row in test
        ),
        "strict_test_pairs": len(test_pairs),
        "preference_smoke_pairs": len(smoke_pairs),
    }
    if observed_counts != EXPECTED_COUNTS:
        raise ValueError(
            f"frozen V4 count contract drift: {observed_counts} "
            f"!= {EXPECTED_COUNTS}"
        )

    v3_schedule_loss_tokens = sum(
        row["completion_tokens"] for row in v3_schedule
    )
    clean_schedule_loss_tokens = sum(
        row["completion_tokens"] for row in clean_schedule
    )
    loss_token_relative_delta = abs(
        clean_schedule_loss_tokens - v3_schedule_loss_tokens
    ) / v3_schedule_loss_tokens
    if loss_token_relative_delta > 0.01:
        raise ValueError(
            f"clean schedule completion-token drift is {loss_token_relative_delta:.4%}"
        )
    if tool_counts(clean_schedule) != tool_counts(v3_schedule):
        raise ValueError("clean schedule target-tool counts changed")
    if source_counts(clean_schedule) != source_counts(v3_schedule):
        raise ValueError("clean schedule source counts changed")
    if {pair["task_key"] for pair in train_pairs} - train_tasks:
        raise ValueError("DPO train pair uses a non-train task")
    if {pair["task_key"] for pair in validation_pairs} - validation_tasks:
        raise ValueError("DPO validation pair split drift")
    if {pair["task_key"] for pair in test_pairs} - test_tasks:
        raise ValueError("DPO test diagnostic pair split drift")
    if any(
        pair["chosen"] == pair["rejected"]
        or pair["future_leakage_audit"]["chosen_result_in_source_prefix"]
        for pair in train_pairs + validation_pairs + test_pairs
    ):
        raise ValueError("invalid preference pair")

    output = args.output_dir
    write_jsonl(output / "clean_sft" / "train_unique.jsonl", clean_unique)
    write_jsonl(output / "clean_sft" / "train_schedule.jsonl", clean_schedule)
    write_jsonl(output / "clean_sft" / "validation.jsonl", clean_validation)
    write_jsonl(
        output / "clean_sft" / "replacement_map.jsonl",
        replacement_map,
    )
    write_jsonl(
        output / "preference" / "train_pairs.jsonl",
        train_pairs,
    )
    write_jsonl(
        output / "preference" / "train_schedule.jsonl",
        preference_schedule,
    )
    write_jsonl(
        output / "preference" / "validation_pairs.jsonl",
        validation_pairs,
    )
    write_jsonl(
        output / "preference" / "smoke_pairs.jsonl",
        smoke_pairs,
    )
    write_jsonl(
        output / "evaluation" / "test_outcomes.jsonl",
        test_outcomes,
    )
    write_jsonl(
        output / "evaluation" / "test_preference_pairs.jsonl",
        test_pairs,
    )

    exposure_counts = Counter(
        row["example_id"] for row in clean_schedule
    )
    preference_mode_counts = Counter(
        row["recovery_mode"] for row in preference_schedule
    )
    replacement_source_tool_counts: dict[str, dict[str, int]] = {}
    for source in sorted({row["source"] for row in replacement_map}):
        replacement_source_tool_counts[source] = dict(
            sorted(
                Counter(
                    row["target_tool"]
                    for row in replacement_map
                    if row["source"] == source
                ).items()
            )
        )
    error_templates = Counter(
        linked_tool_outcome(
            by_trace[row["trace_id"]],
            row["target_message_index"],
        )["content"]
        for row in failed_unique
    )
    summary = {
        "protocol": PROTOCOL,
        "claim_boundary": (
            "exploratory offline next-tool-call V4 on the already inspected "
            "V2/V3 test; not paper-final or end-to-end Agent success"
        ),
        "seed": SEED,
        "model": {
            "name": args.model,
            "revision": args.model_revision,
        },
        "source": {
            "v3_arm": V3_ARM,
            "v3_trace_set_sha256": EXPECTED_V3_TRACE_SHA256,
            "v3_train_sha256": EXPECTED_V3_TRAIN_SHA256,
            "v3_schedule_sha256": EXPECTED_V3_SCHEDULE_SHA256,
            "raw_sha256": EXPECTED_RAW_SHA256,
        },
        "counts": observed_counts,
        "clean_sft": {
            "definition": (
                "target assistant tool call has an adjacent, id/name-linked "
                "tool result whose content does not start with 'Error:'"
            ),
            "unique_examples": len(clean_unique),
            "removed_failed_labels": len(failed_unique),
            "scheduled_microbatches": len(clean_schedule),
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "optimizer_steps": CLEAN_SFT_OPTIMIZER_STEPS,
            "replacement_slots": len(replacement_map),
            "recovery_mode_matched_replacements": (
                recovery_mode_matched_replacements
            ),
            "recovery_mode_fallback_replacements": (
                len(replacement_map) - recovery_mode_matched_replacements
            ),
            "all_unique_clean_examples_exposed": (
                set(exposure_counts)
                == {row["example_id"] for row in clean_unique}
            ),
            "min_exposure": min(exposure_counts.values()),
            "max_exposure": max(exposure_counts.values()),
            "exposure_histogram": dict(
                sorted(Counter(exposure_counts.values()).items())
            ),
            "scheduled_source_counts": source_counts(clean_schedule),
            "scheduled_target_tool_counts": tool_counts(clean_schedule),
            "scheduled_nonpad_tokens": sum(
                row["sequence_tokens"] for row in clean_schedule
            ),
            "selected_completion_loss_tokens": sum(
                row["completion_tokens"] for row in clean_unique
            ),
            "scheduled_loss_tokens": clean_schedule_loss_tokens,
            "v3_scheduled_loss_tokens": v3_schedule_loss_tokens,
            "loss_token_relative_delta": loss_token_relative_delta,
            "replacement_source_target_tool_counts": (
                replacement_source_tool_counts
            ),
        },
        "preference": {
            "pair_definition": (
                "same post-error prompt; chosen is the observed non-error "
                "repair; rejected replays the linked failed call"
            ),
            "train_unique_pairs": len(train_pairs),
            "validation_unique_pairs": len(validation_pairs),
            "test_diagnostic_pairs": len(test_pairs),
            "train_mode_counts": dict(
                sorted(Counter(pair["recovery_mode"] for pair in train_pairs).items())
            ),
            "scheduled_microbatches": len(preference_schedule),
            "scheduled_mode_counts": dict(sorted(preference_mode_counts.items())),
            "gradient_accumulation": PREFERENCE_GRADIENT_ACCUMULATION,
            "optimizer_steps": PREFERENCE_OPTIMIZER_STEPS,
            "smoke_pairs": len(smoke_pairs),
            "longest_pair_id": longest_pair_id,
            "train_error_type_counts": dict(
                sorted(Counter(pair["error_type"] for pair in train_pairs).items())
            ),
            "train_source_counts": dict(
                sorted(Counter(pair["source"] for pair in train_pairs).items())
            ),
            "train_repair_tool_counts": dict(
                sorted(Counter(pair["repair_tool"] for pair in train_pairs).items())
            ),
            "train_filter_reasons": dict(sorted(train_pair_reasons.items())),
            "validation_filter_reasons": dict(
                sorted(validation_pair_reasons.items())
            ),
            "test_filter_reasons": dict(sorted(test_pair_reasons.items())),
        },
        "evaluation_subsets": {
            "all_test": len(test),
            "outcome_success": observed_counts["outcome_success_test"],
            "non_recovery_success": observed_counts[
                "non_recovery_success_test"
            ],
            "failed_gold": observed_counts["failed_test"],
            "recovery_success": observed_counts["recovery_success_test"],
            "strict_preference": observed_counts["strict_test_pairs"],
        },
        "failed_label_error_templates": [
            {"content": content, "count": count}
            for content, count in sorted(error_templates.items())
        ],
        "invalid_claims": [
            "paper_final_confirmation",
            "independent_held_out_confirmation",
            "end_to_end_agent_success",
            "executable_tool_success",
            "cross_seed_robustness",
            "guaranteed_positive_result",
        ],
    }
    write_json(output / "build_summary.json", summary)

    hashes = {
        "clean_train_unique_sha256": sha256_file(
            output / "clean_sft" / "train_unique.jsonl"
        ),
        "clean_train_schedule_sha256": sha256_file(
            output / "clean_sft" / "train_schedule.jsonl"
        ),
        "clean_validation_sha256": sha256_file(
            output / "clean_sft" / "validation.jsonl"
        ),
        "replacement_map_sha256": sha256_file(
            output / "clean_sft" / "replacement_map.jsonl"
        ),
        "preference_train_pairs_sha256": sha256_file(
            output / "preference" / "train_pairs.jsonl"
        ),
        "preference_train_schedule_sha256": sha256_file(
            output / "preference" / "train_schedule.jsonl"
        ),
        "preference_validation_pairs_sha256": sha256_file(
            output / "preference" / "validation_pairs.jsonl"
        ),
        "preference_smoke_pairs_sha256": sha256_file(
            output / "preference" / "smoke_pairs.jsonl"
        ),
        "test_outcomes_sha256": sha256_file(
            output / "evaluation" / "test_outcomes.jsonl"
        ),
        "test_preference_pairs_sha256": sha256_file(
            output / "evaluation" / "test_preference_pairs.jsonl"
        ),
    }
    if hashes != EXPECTED_OUTPUT_SHA256:
        raise ValueError(
            f"frozen V4 output hash drift: {hashes} != {EXPECTED_OUTPUT_SHA256}"
        )
    audit = {
        "protocol": PROTOCOL,
        "valid": True,
        "errors": [],
        "counts": observed_counts,
        "hashes": hashes,
        "checks": {
            "raw_hashes_exact": True,
            "v3_input_hashes_exact": True,
            "v3_trace_set_exact": True,
            "task_split_disjoint": True,
            "prefix_only_prompt_reconstruction_exact": True,
            "call_result_id_and_name_linkage_exact": True,
            "clean_schedule_failed_labels": 0,
            "clean_schedule_source_counts_equal_v3": True,
            "clean_schedule_target_tool_counts_equal_v3": True,
            "clean_schedule_recovery_mode_match_contract": (
                recovery_mode_matched_replacements == 103
            ),
            "clean_schedule_loss_token_delta_le_1pct": True,
            "all_unique_clean_examples_exposed": summary["clean_sft"][
                "all_unique_clean_examples_exposed"
            ],
            "preference_train_tasks_only": True,
            "pair_prompt_contains_failed_call_and_error": True,
            "chosen_result_excluded_from_prompt_source": True,
            "chosen_and_rejected_differ": True,
            "preference_modes_balanced_in_schedule": (
                preference_mode_counts
                == Counter(
                    {
                        "agent_initiated": PREFERENCE_EXPOSURES_PER_MODE,
                        "user_assisted": PREFERENCE_EXPOSURES_PER_MODE,
                    }
                )
            ),
        },
    }
    if not all(
        value is True or value == 0
        for value in audit["checks"].values()
    ):
        raise ValueError(f"V4 audit did not pass: {audit['checks']}")
    write_json(output / "contract_audit.json", audit)
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
