#!/usr/bin/env python3
"""Run one registered V5.5 checkpoint on the sealed derived-validation split.

This is a real tau2 end-to-end evaluation: the trained model acts as the
customer-service agent, a separately served frozen model acts as user/judge,
and tau2's official composite reward defines task success.  The evaluator runs
both clean and controlled read-only-error conditions, rejects incomplete task
coverage, and writes one auditable row per task/condition/evaluation seed.

The official 60-task test is intentionally not accepted by this executable.
Opening that test requires a separate post-selection confirmation release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

try:
    import run_v5_sft_causal_eval as legacy
    import v5_5_full_protocol as full
    from build_v5_5_checkpoint_registry import (
        PROTOCOL as REGISTRY_PROTOCOL,
        TRAINER_ARMS,
    )
except ModuleNotFoundError:
    from scripts import run_v5_sft_causal_eval as legacy
    from scripts import v5_5_full_protocol as full
    from scripts.build_v5_5_checkpoint_registry import (
        PROTOCOL as REGISTRY_PROTOCOL,
        TRAINER_ARMS,
    )


PROTOCOL = "v5_5_tau2_end_to_end_validation_v1"
CONDITIONS = ("clean", "error")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_registry(
    path: Path,
    *,
    arm: str,
    training_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = read_json(path)
    if (
        registry.get("protocol") != REGISTRY_PROTOCOL
        or registry.get("design_protocol") != full.PROTOCOL
        or registry.get("design_version") != full.DESIGN_VERSION
        or registry.get("official_test_used") is not False
        or registry.get("official_test_sealed") is not True
    ):
        raise RuntimeError("V5.5 checkpoint registry protocol/test seal drift")
    if arm not in registry.get("registered_arms", []):
        raise RuntimeError(f"arm {arm!r} is not registered")
    if training_seed not in registry.get("registered_training_seeds", []):
        raise RuntimeError(f"training seed {training_seed} is not registered")
    entry = (
        (registry.get("entries") or {}).get(arm, {}).get(str(training_seed))
    )
    if not isinstance(entry, dict):
        raise RuntimeError(f"registry lacks {arm}/{training_seed}")
    if (
        entry.get("trainer_arm") != arm
        or entry.get("paper_arm") != TRAINER_ARMS[arm]
        or entry.get("training_seed") != training_seed
        or not str(entry.get("model_id", "")).startswith("openai/")
    ):
        raise RuntimeError("checkpoint registry entry identity drift")
    data = registry.get("data")
    if (
        not isinstance(data, dict)
        or data.get("pair_mode") not in {"reference", "natural"}
        or data.get("claim_level")
        not in {
            "diagnostic_only",
            "natural_counterfactual_training_candidate",
        }
    ):
        raise RuntimeError("checkpoint registry claim boundary drift")
    data_audit = Path(str(data["audit_path"]))
    if (
        not data_audit.is_file()
        or sha256_file(data_audit) != data.get("audit_sha256")
    ):
        raise RuntimeError("checkpoint registry data audit bytes are unavailable")
    return registry, entry


def _call_semantics(call: Any) -> tuple[str, str] | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = call.get("name")
        arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return name, canonical(arguments)


def _reward(simulation: dict[str, Any]) -> float:
    value = (simulation.get("reward_info") or {}).get("reward")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"task {simulation.get('task_id')} lacks numeric official reward"
        ) from error
    if not math.isfinite(result):
        raise RuntimeError("official reward is non-finite")
    return result


def _injection_audit(
    simulation: dict[str, Any],
    *,
    condition: str,
    expected_fault: dict[str, Any] | None,
) -> dict[str, Any]:
    messages = simulation.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError("simulation messages are missing")
    injected_indices: list[int] = []
    calls: list[tuple[int, tuple[str, str]]] = []
    tool_results = 0
    tool_errors = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RuntimeError("simulation contains a malformed message")
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                semantic = _call_semantics(call)
                if semantic is not None:
                    calls.append((index, semantic))
            raw = message.get("raw_data") or {}
            if isinstance(raw, dict) and raw.get("v5_stage1_injected_fault") is True:
                injected_indices.append(index)
        elif message.get("role") == "tool":
            tool_results += 1
            if message.get("error") is True:
                tool_errors += 1
    if condition == "clean":
        if injected_indices:
            raise RuntimeError("clean condition contains an injected fault")
        return {
            "injected_fault_count": 0,
            "injected_fault_observed_as_error": None,
            "repeated_same_injected_call": False,
            "tool_calls": len(calls),
            "tool_results": tool_results,
            "tool_errors": tool_errors,
        }
    if len(injected_indices) != 1 or not isinstance(expected_fault, dict):
        raise RuntimeError("error condition lacks exactly one registered injection")
    injected_index = injected_indices[0]
    injected_semantics = (
        str(expected_fault["tool_name"]),
        canonical(expected_fault["arguments"]),
    )
    if not any(
        index == injected_index and semantic == injected_semantics
        for index, semantic in calls
    ):
        raise RuntimeError("injected call bytes differ from the validation manifest")
    if (
        injected_index + 1 >= len(messages)
        or messages[injected_index + 1].get("role") != "tool"
        or messages[injected_index + 1].get("error") is not True
    ):
        raise RuntimeError("registered injection was not observed as a tool error")
    repeats = sum(
        index > injected_index and semantic == injected_semantics
        for index, semantic in calls
    )
    return {
        "injected_fault_count": 1,
        "injected_fault_observed_as_error": True,
        "repeated_same_injected_call": repeats > 0,
        "tool_calls": len(calls),
        "tool_results": tool_results,
        "tool_errors": tool_errors,
    }


def extract_result_rows(
    paths: list[Path],
    *,
    manifest_rows: list[dict[str, Any]],
    arm: str,
    training_seed: int,
    evaluation_seed: int,
    claim_level: str,
    training_fault_tools: set[str],
) -> list[dict[str, Any]]:
    manifest = {
        (str(row["domain"]), str(row["task_id"])): row
        for row in manifest_rows
    }
    output: list[dict[str, Any]] = []
    observed: set[tuple[str, str, str]] = set()
    for path in paths:
        stem = path.name.split(".")[0]
        domain, condition = stem.split("_", 1)[:2]
        if domain not in {"airline", "retail"} or condition not in CONDITIONS:
            raise RuntimeError(f"cannot infer domain/condition from {path.name}")
        payload = read_json(path)
        simulations = payload.get("simulations")
        if not isinstance(simulations, list):
            raise RuntimeError(f"{path}: simulations are missing")
        for simulation in simulations:
            task_id = str(simulation.get("task_id"))
            key = (domain, task_id, condition)
            if key in observed or (domain, task_id) not in manifest:
                raise RuntimeError(f"duplicate or unregistered result row {key}")
            observed.add(key)
            registered = manifest[(domain, task_id)]
            fault = (
                registered["error_condition"] if condition == "error" else None
            )
            interface = _injection_audit(
                simulation,
                condition=condition,
                expected_fault=fault,
            )
            official_reward = _reward(simulation)
            tool_key = (
                f"{domain}:{fault['tool_name']}"
                if fault is not None
                else None
            )
            output.append(
                {
                    "protocol": PROTOCOL,
                    "arm": arm,
                    "paper_arm": TRAINER_ARMS[arm],
                    "training_seed": training_seed,
                    "evaluation_seed": evaluation_seed,
                    "domain": domain,
                    "task_id": task_id,
                    "condition": condition,
                    "fault_family": (
                        fault.get("fault_family") if fault is not None else None
                    ),
                    "fault_scope": (
                        None
                        if tool_key is None
                        else "in_family"
                        if tool_key in training_fault_tools
                        else "out_of_family"
                    ),
                    "official_reward": official_reward,
                    "task_success": math.isclose(
                        official_reward,
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    "duration_seconds": float(simulation.get("duration") or 0.0),
                    "termination_reason": simulation.get("termination_reason"),
                    **interface,
                    "claim_level": claim_level,
                    "official_test_used": False,
                }
            )
    return sorted(
        output,
        key=lambda row: (
            row["condition"],
            row["domain"],
            int(row["task_id"]),
        ),
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
        if row["fault_scope"]:
            groups.setdefault(row["fault_scope"], []).append(row)
    metrics: dict[str, Any] = {}
    for name, values in groups.items():
        metrics[name] = {
            "tasks": len(values),
            "task_successes": sum(row["task_success"] for row in values),
            "task_success_rate": sum(row["task_success"] for row in values)
            / len(values),
            "mean_official_reward": sum(
                row["official_reward"] for row in values
            )
            / len(values),
            "repeated_same_error_rate": (
                sum(row["repeated_same_injected_call"] for row in values)
                / len(values)
            ),
            "tool_calls_per_task": sum(row["tool_calls"] for row in values)
            / len(values),
            "tool_error_rate": (
                sum(row["tool_errors"] for row in values)
                / max(1, sum(row["tool_results"] for row in values))
            ),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--protocol-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(TRAINER_ARMS), required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        choices=full.EVALUATION_SEEDS,
        required=True,
    )
    parser.add_argument("--agent-api-base", required=True)
    parser.add_argument("--agent-api-key", default="v55-agent-local")
    parser.add_argument("--user-api-base", required=True)
    parser.add_argument("--user-api-key", default="v55-user-local")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("clean", "error", "both"),
        default="both",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"output directory must be absent: {output_dir}")
    registry_path = args.checkpoint_registry.resolve()
    registry, entry = load_registry(
        registry_path,
        arm=args.arm,
        training_seed=args.training_seed,
    )
    if git_commit() != registry["source_commit"]:
        raise RuntimeError("local source commit differs from checkpoint registry")
    legacy.require_clean_tracked_source()
    split_path = args.split_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    split = legacy.load_split_manifest(split_path)
    manifest = legacy.load_manifest(
        validation_path,
        split_manifest=split,
        split_manifest_sha256=sha256_file(split_path),
    )
    protocol_audit = read_json(args.protocol_audit.resolve())
    if (
        protocol_audit.get("status") != "PASS"
        or protocol_audit.get("official_test_used") is not False
        or protocol_audit.get("validation_manifest_sha256")
        != sha256_file(validation_path)
    ):
        raise RuntimeError("validation fault-protocol audit is not frozen PASS")
    legacy.validate_local_manifest_sources(
        manifest,
        args.tau2_root.resolve(),
    )
    rows = legacy.shard_rows(
        list(manifest["rows"]),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if not rows:
        raise RuntimeError("selected validation shard is empty")
    model_id = entry["model_id"]
    user_judge = registry["user_judge"]
    legacy.verify_served_model_id(
        args.agent_api_base,
        args.agent_api_key,
        model_id,
    )
    legacy.verify_served_model_id(
        args.user_api_base,
        args.user_api_key,
        user_judge["model_id"],
    )
    output_dir.mkdir(parents=True)
    contract_path = output_dir / "run_contract.json"
    contract = {
        "protocol": PROTOCOL,
        "status": "INCOMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": registry["source_commit"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "checkpoint_entry": entry,
        "arm": args.arm,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "agent_model": model_id,
        "user_judge": user_judge,
        "split_manifest_sha256": sha256_file(split_path),
        "validation_manifest_sha256": sha256_file(validation_path),
        "protocol_audit_sha256": sha256_file(args.protocol_audit.resolve()),
        "conditions": (
            list(CONDITIONS)
            if args.condition == "both"
            else [args.condition]
        ),
        "task_ids": [
            f"{row['domain']}:{row['task_id']}" for row in rows
        ],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "decoding": {
            "temperature": 0,
            "max_tokens": 512,
            "max_steps": 60,
            "timeout_seconds": 900.0,
            "parallel_tool_calls": False,
        },
        "official_test_used": False,
        "official_test_sealed": True,
    }
    write_json(contract_path, contract)
    legacy.configure_tau2_path(args.tau2_root.resolve())
    legacy.register_fault_agent()
    agent_args = legacy.endpoint_args(
        args.agent_api_base,
        args.agent_api_key,
        max_tokens=512,
        seed=args.evaluation_seed,
    )
    user_args = legacy.endpoint_args(
        args.user_api_base,
        args.user_api_key,
        max_tokens=512,
        seed=args.evaluation_seed,
    )
    legacy.patch_local_nl_judge(user_judge["model_id"], user_args)
    run_args = SimpleNamespace(
        output_dir=output_dir,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        agent_model=model_id,
        user_model=user_judge["model_id"],
        num_trials=1,
        max_steps=60,
        timeout=900.0,
        seed=args.evaluation_seed,
    )
    output_files: list[Path] = []
    conditions = CONDITIONS if args.condition == "both" else (args.condition,)
    for domain in ("retail", "airline"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        if not domain_rows:
            continue
        for condition in conditions:
            path = legacy.run_condition(
                domain=domain,
                rows=domain_rows,
                condition=condition,
                args=run_args,
                agent_args=agent_args,
                user_args=user_args,
            )
            legacy.audit_result_interface(
                path,
                expected_task_ids=[
                    str(row["task_id"]) for row in domain_rows
                ],
                num_trials=1,
                max_tokens=512,
            )
            output_files.append(path)
    result_rows = extract_result_rows(
        output_files,
        manifest_rows=rows,
        arm=args.arm,
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        claim_level=registry["data"]["claim_level"],
        training_fault_tools=set(
            registry["data"]["training_fault_tools"]
        ),
    )
    expected_count = len(rows) * len(conditions)
    if len(result_rows) != expected_count:
        raise RuntimeError(
            f"incomplete evaluation rows: {len(result_rows)} != {expected_count}"
        )
    rows_path = output_dir / "rows.jsonl"
    write_jsonl(rows_path, result_rows)
    metrics = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "arm": args.arm,
        "paper_arm": TRAINER_ARMS[args.arm],
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "rows": len(result_rows),
        "metrics": aggregate(result_rows),
        "claim_level": registry["data"]["claim_level"],
        "official_test_used": False,
    }
    write_json(output_dir / "metrics.json", metrics)
    contract.update(
        {
            "status": "COMPLETE",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result_sha256": {
                path.name: sha256_file(path)
                for path in [*output_files, rows_path, output_dir / "metrics.json"]
            },
        }
    )
    write_json(contract_path, contract)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
