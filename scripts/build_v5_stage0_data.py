#!/usr/bin/env python3
"""Build leakage-safe Stage-0 SFT, DPO, and recovery-mixture datasets.

Only successful full τ² trajectories become SFT supervision. Injected failing
actions remain context, never labels. DPO pairs share the exact post-error
context: the teacher's first recovery response is preferred to repeating the
failed tool call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


RATIOS = (0.0, 0.10, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-mode", choices=("standard", "ground_truth"), required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_message(message: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message["role"]}
    if message.get("content") is not None:
        result["content"] = message["content"]
    if message.get("tool_calls"):
        result["tool_calls"] = [
            {
                "id": call.get("id"),
                "name": call["name"],
                "arguments": call.get("arguments") or {},
            }
            for call in message["tool_calls"]
        ]
    if message["role"] == "tool":
        result["id"] = message.get("id")
        result["error"] = bool(message.get("error"))
    return result


def is_injected(message: dict[str, Any]) -> bool:
    return bool((message.get("raw_data") or {}).get("v5_stage0_injected_fault"))


def successful(simulation: dict[str, Any]) -> bool:
    return float((simulation.get("reward_info") or {}).get("reward") or 0.0) == 1.0


def load_simulations(raw_dir: Path, condition: str) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for domain in ("retail", "airline"):
        path = raw_dir / f"{domain}_{condition}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend((domain, simulation) for simulation in payload["simulations"])
    return rows


def build_examples(
    simulations: list[tuple[str, dict[str, Any]]],
    *,
    condition: str,
    teacher_model: str,
    teacher_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    sft: list[dict[str, Any]] = []
    dpo: list[dict[str, Any]] = []
    stats = {"runs": len(simulations), "successful_runs": 0, "injected_errors": 0}
    for domain, simulation in simulations:
        if not successful(simulation):
            continue
        stats["successful_runs"] += 1
        messages = simulation.get("messages") or []
        injected_index = next(
            (index for index, message in enumerate(messages) if is_injected(message)),
            None,
        )
        error_index = None
        if injected_index is not None:
            for index in range(injected_index + 1, len(messages)):
                if messages[index].get("role") == "tool":
                    error_index = index
                    break
            if error_index is not None and messages[error_index].get("error"):
                stats["injected_errors"] += 1

        for index, message in enumerate(messages):
            if message.get("role") != "assistant" or is_injected(message):
                continue
            # Skip τ²'s fixed greeting and any assistant action before an injected error.
            if index == 0 and message.get("usage") is None:
                continue
            if condition == "recovery" and (error_index is None or index <= error_index):
                continue
            sft.append(
                {
                    "id": f"{condition}:{domain}:{simulation['task_id']}:{simulation.get('trial', 0)}:{index}",
                    "prompt_messages": [compact_message(item) for item in messages[:index]],
                    "label_message": compact_message(message),
                    "metadata": {
                        "domain": domain,
                        "task_id": str(simulation["task_id"]),
                        "trial": simulation.get("trial"),
                        "seed": simulation.get("seed"),
                        "source": condition,
                        "full_trajectory_reward": 1.0,
                        "teacher_model": teacher_model,
                        "teacher_mode": teacher_mode,
                    },
                }
            )

        if condition == "recovery" and error_index is not None:
            chosen_index = next(
                (
                    index
                    for index in range(error_index + 1, len(messages))
                    if messages[index].get("role") == "assistant"
                    and not is_injected(messages[index])
                ),
                None,
            )
            if chosen_index is not None:
                failed_call = compact_message(messages[injected_index])
                rejected = json.loads(json.dumps(failed_call))
                if rejected.get("tool_calls"):
                    rejected["tool_calls"][0]["id"] = (
                        f"v5-repeat-{domain}-{simulation['task_id']}-{simulation.get('trial', 0)}"
                    )
                chosen = compact_message(messages[chosen_index])
                if canonical(chosen) != canonical(rejected):
                    dpo.append(
                        {
                            "id": f"dpo:{domain}:{simulation['task_id']}:{simulation.get('trial', 0)}",
                            "prompt_messages": [
                                compact_message(item) for item in messages[: chosen_index]
                            ],
                            "chosen_message": chosen,
                            "rejected_message": rejected,
                            "metadata": {
                                "domain": domain,
                                "task_id": str(simulation["task_id"]),
                                "trial": simulation.get("trial"),
                                "seed": simulation.get("seed"),
                                "full_trajectory_reward": 1.0,
                                "preference_semantics": "successful_repair_over_repeat_failed_call",
                                "rejected_origin": "counterfactual_repeat_of_observed_failure",
                                "teacher_model": teacher_model,
                                "teacher_mode": teacher_mode,
                            },
                        }
                    )
    return sft, dpo, stats


def make_counter(tokenizer_name: str | None):
    if tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def count(row: dict[str, Any]) -> int:
            messages = list(row["prompt_messages"]) + [row["label_message"]]
            normalized = []
            for message in messages:
                item = {"role": message["role"], "content": message.get("content") or ""}
                if message.get("tool_calls"):
                    item["tool_calls"] = [
                        {
                            "id": call.get("id"),
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": canonical(call.get("arguments") or {}),
                            },
                        }
                        for call in message["tool_calls"]
                    ]
                normalized.append(item)
            try:
                return len(
                    tokenizer.apply_chat_template(
                        normalized, tokenize=True, add_generation_prompt=False
                    )
                )
            except Exception:
                return len(tokenizer(canonical(row), add_special_tokens=True)["input_ids"])

        return count, "hf_chat_template"

    return lambda row: max(1, len(canonical(row)) // 4), "utf8_char_div4_proxy"


def deterministic_rows(rows: list[dict[str, Any]], seed: int, salt: str) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(f"{seed}:{salt}").shuffle(shuffled)
    return shuffled


def take_to_budget(
    rows: list[dict[str, Any]], target: int, counter
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        size = counter(row)
        if selected and total + size > target:
            continue
        selected.append(row)
        total += size
        if total >= target:
            break
    return selected, total


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    allowed = {
        (row["domain"], str(row["task_id"])) for row in manifest["rows"]
    }
    clean_runs = load_simulations(args.raw_dir, "clean")
    recovery_runs = load_simulations(args.raw_dir, "error")
    observed = {
        (domain, str(simulation["task_id"]))
        for domain, simulation in clean_runs + recovery_runs
    }
    if not observed <= allowed:
        raise RuntimeError("Raw results contain tasks outside the inner-train manifest")

    clean, _, clean_stats = build_examples(
        clean_runs,
        condition="clean",
        teacher_model=args.teacher_model,
        teacher_mode=args.teacher_mode,
    )
    recovery, dpo, recovery_stats = build_examples(
        recovery_runs,
        condition="recovery",
        teacher_model=args.teacher_model,
        teacher_mode=args.teacher_mode,
    )
    if not clean or not recovery or not dpo:
        raise RuntimeError("Formal data requires non-empty clean SFT, recovery SFT, and DPO")
    if recovery_stats["injected_errors"] != recovery_stats["successful_runs"]:
        raise RuntimeError("Every successful recovery trajectory must contain the injected error")

    counter, token_counter = make_counter(args.tokenizer)
    clean_total = sum(counter(row) for row in clean)
    recovery_total = sum(counter(row) for row in recovery)
    capacities = []
    for ratio in RATIOS:
        clean_cap = clean_total / (1.0 - ratio) if ratio < 1.0 else float("inf")
        recovery_cap = recovery_total / ratio if ratio > 0.0 else float("inf")
        capacities.append(min(clean_cap, recovery_cap))
    target_budget = int(min(capacities))
    if target_budget <= 0:
        raise RuntimeError("Unable to derive a positive shared token budget")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sft_clean.jsonl", clean)
    write_jsonl(args.output_dir / "sft_recovery.jsonl", recovery)
    write_jsonl(args.output_dir / "dpo_pairs.jsonl", dpo)
    mixture_audit = {}
    for ratio in RATIOS:
        clean_target = round(target_budget * (1.0 - ratio))
        recovery_target = target_budget - clean_target
        selected_clean, clean_tokens = take_to_budget(
            deterministic_rows(clean, args.seed, f"clean:{ratio}"),
            clean_target,
            counter,
        )
        selected_recovery, recovery_tokens = take_to_budget(
            deterministic_rows(recovery, args.seed, f"recovery:{ratio}"),
            recovery_target,
            counter,
        )
        combined = deterministic_rows(
            selected_clean + selected_recovery, args.seed, f"combined:{ratio}"
        )
        ratio_label = str(int(ratio * 100))
        path = args.output_dir / "mixtures" / f"recovery_{ratio_label}.jsonl"
        write_jsonl(path, combined)
        mixture_audit[ratio_label] = {
            "examples": len(combined),
            "clean_examples": len(selected_clean),
            "recovery_examples": len(selected_recovery),
            "clean_tokens": clean_tokens,
            "recovery_tokens": recovery_tokens,
            "total_tokens": clean_tokens + recovery_tokens,
            "realized_recovery_token_ratio": (
                recovery_tokens / (clean_tokens + recovery_tokens)
                if clean_tokens + recovery_tokens
                else 0.0
            ),
            "sha256": sha256_file(path),
        }

    audit = {
        "status": "PASS",
        "protocol": "v5_stage0_formal_data_construction",
        "teacher_model": args.teacher_model,
        "teacher_mode": args.teacher_mode,
        "official_test_used": False,
        "raw": {"clean": clean_stats, "recovery": recovery_stats},
        "datasets": {
            "clean_sft_examples": len(clean),
            "recovery_sft_examples": len(recovery),
            "dpo_pairs": len(dpo),
        },
        "label_guarantees": {
            "only_successful_full_trajectories": True,
            "injected_failed_call_is_never_sft_label": True,
            "no_future_messages_in_sft_prompt": True,
            "dpo_chosen_and_rejected_share_post_error_context": True,
            "dpo_rejected_is_counterfactual_not_claimed_observed": True,
        },
        "token_counter": token_counter,
        "shared_target_token_budget": target_budget,
        "mixtures": mixture_audit,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
