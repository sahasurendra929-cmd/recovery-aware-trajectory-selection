#!/usr/bin/env python3
"""Run one frozen V6.10 official-evaluation arm/seed/task shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.request import Request, urlopen

try:
    import prepare_v5_stage1_manifests as fault_protocol
    import run_v5_5_end_to_end_eval as result_protocol
    import run_v5_sft_causal_eval as legacy
    from build_v6_checkpoint_registry import (
        PROTOCOL as CHECKPOINT_PROTOCOL,
    )
    from build_v6_official_unseal_receipt import (
        PROTOCOL as UNSEAL_PROTOCOL,
        REQUIRED_ARTIFACT_LABELS,
    )
    from summarize_v6_task_clusters import (
        ARMS,
        EVALUATION_PROTOCOL as PROTOCOL,
        EVALUATION_SEEDS,
        TRAINING_SEED,
    )
except ModuleNotFoundError:
    from scripts import prepare_v5_stage1_manifests as fault_protocol
    from scripts import run_v5_5_end_to_end_eval as result_protocol
    from scripts import run_v5_sft_causal_eval as legacy
    from scripts.build_v6_checkpoint_registry import (
        PROTOCOL as CHECKPOINT_PROTOCOL,
    )
    from scripts.build_v6_official_unseal_receipt import (
        PROTOCOL as UNSEAL_PROTOCOL,
        REQUIRED_ARTIFACT_LABELS,
    )
    from scripts.summarize_v6_task_clusters import (
        ARMS,
        EVALUATION_PROTOCOL as PROTOCOL,
        EVALUATION_SEEDS,
        TRAINING_SEED,
    )


MANIFEST_PROTOCOL = "v6_10_official_readonly_fault_manifest_v1"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
CONDITIONS = ("clean", "controlled_error")
NUM_SHARDS = 2
MAX_STEPS = 60
TIMEOUT_SECONDS = 900.0
MAX_TOKENS = 512
NUM_TRIALS = 1
FAULT_INJECTIONS: dict[str, dict[str, Any]] = {}


class EvaluationError(RuntimeError):
    """The frozen V6.10 official-evaluation contract was violated."""


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise EvaluationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise EvaluationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_identity(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status.strip())


def _receipt_payload_sha256(receipt: Mapping[str, Any]) -> str:
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(canonical(unsigned).encode("utf-8")).hexdigest()


def validate_unseal(
    path: Path,
    *,
    source_root: Path,
    tau2_root: Path,
    split_manifest: Path,
) -> dict[str, Any]:
    receipt = read_json(path)
    source_commit, source_clean = git_identity(source_root)
    tau2_commit, tau2_clean = git_identity(tau2_root)
    evaluator_sha = sha256_file(Path(__file__).resolve())
    split_binding = receipt.get("split_manifest")
    bound_artifacts = receipt.get("bound_artifacts")
    summarizer_path = (
        source_root / "scripts" / "summarize_v6_task_clusters.py"
    ).resolve()
    if (
        receipt.get("protocol") != UNSEAL_PROTOCOL
        or receipt.get("status") != "UNSEALED_ONCE"
        or receipt.get("official_test_access_count_before_unseal") != 0
        or receipt.get("official_test_access_count") != 1
        or receipt.get("changes_after_unseal") != "FORBIDDEN"
        or receipt.get("source_commit") != source_commit
        or not source_clean
        or receipt.get("tau2_commit") != tau2_commit
        or tau2_commit != TAU2_COMMIT
        or not tau2_clean
        or receipt.get("evaluator_sha256") != evaluator_sha
        or receipt.get("evaluator_path")
        != str(Path(__file__).resolve())
        or receipt.get("summarizer_path") != str(summarizer_path)
        or not summarizer_path.is_file()
        or receipt.get("summarizer_sha256")
        != sha256_file(summarizer_path)
        or not isinstance(bound_artifacts, Mapping)
        or set(bound_artifacts) != REQUIRED_ARTIFACT_LABELS
        or not isinstance(split_binding, Mapping)
        or split_binding.get("path") != str(split_manifest.resolve())
        or split_binding.get("sha256") != sha256_file(split_manifest)
        or receipt.get("receipt_sha256") != _receipt_payload_sha256(receipt)
    ):
        raise EvaluationError("official-test unseal/source binding drift")
    for label, binding in bound_artifacts.items():
        bound_path = (
            Path(binding.get("path", "")).resolve()
            if isinstance(binding, Mapping)
            else Path()
        )
        if (
            not isinstance(binding, Mapping)
            or not bound_path.is_file()
            or binding.get("path") != str(bound_path)
            or binding.get("sha256") != sha256_file(bound_path)
            or binding.get("size_bytes") != bound_path.stat().st_size
        ):
            raise EvaluationError(
                f"official-test bound artifact drift: {label}"
            )
    return receipt


def _official_rows(
    *,
    tau2_root: Path,
    split_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    split = read_json(split_manifest)
    if split.get("protocol") != fault_protocol.SPLIT_PROTOCOL:
        raise EvaluationError("split manifest protocol drift")
    domains_root = tau2_root / "data" / "tau2" / "domains"
    used_parameters: set[str] = set()
    rows: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, str]] = {}
    for domain, expected_count in {"retail": 40, "airline": 20}.items():
        ids = [str(value) for value in split["domains"][domain]["sealed_test_ids"]]
        if len(ids) != expected_count or len(set(ids)) != expected_count:
            raise EvaluationError(f"{domain}: official task population drift")
        tasks_path = domains_root / domain / "tasks.json"
        database_path = domains_root / domain / "db.json"
        tasks = {
            str(task["id"]): task
            for task in fault_protocol.load_json(tasks_path)
        }
        database = fault_protocol.load_json(database_path)
        if not set(ids) <= set(tasks):
            raise EvaluationError(f"{domain}: official task IDs are missing")
        source_files[domain] = {
            "tasks_json_sha256": sha256_file(tasks_path),
            "db_json_sha256": sha256_file(database_path),
            "tools_py_sha256": sha256_file(
                tau2_root / "src" / "tau2" / "domains" / domain / "tools.py"
            ),
        }
        rows.extend(
            fault_protocol.build_fault_rows(
                domain=domain,
                task_ids=ids,
                phase="official_test",
                tasks=tasks,
                database=database,
                seed=TRAINING_SEED,
                used_parameters=used_parameters,
            )
        )
    if len(rows) != 60 or len(used_parameters) != 60:
        raise EvaluationError("official fault manifest must contain 60 unique tasks")
    return rows, source_files


def build_official_manifest(
    *,
    tau2_root: Path,
    split_manifest: Path,
    unseal_receipt: Path,
    source_root: Path,
) -> dict[str, Any]:
    receipt = validate_unseal(
        unseal_receipt,
        source_root=source_root,
        tau2_root=tau2_root,
        split_manifest=split_manifest,
    )
    fault_protocol.require_clean_pinned_tau2_sources(tau2_root)
    fault_protocol.validate_read_only_source(tau2_root)
    rows, source_files = _official_rows(
        tau2_root=tau2_root,
        split_manifest=split_manifest,
    )
    payload = {
        "protocol": MANIFEST_PROTOCOL,
        "fault_protocol": fault_protocol.fault_protocol_manifest(),
        "seed": TRAINING_SEED,
        "source_split": "official_test",
        "paired_task_count": 60,
        "domain_counts": {"retail": 40, "airline": 20},
        "rows": rows,
        "source_files": source_files,
        "tau2_commit": TAU2_COMMIT,
        "split_manifest_sha256": sha256_file(split_manifest),
        "unseal_receipt_sha256": sha256_file(unseal_receipt),
        "evaluator_sha256": receipt["evaluator_sha256"],
        "official_test_used": True,
        "official_test_access_count": 1,
        "assignment_uses_model_or_arm_outcomes": False,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        canonical(payload).encode("utf-8")
    ).hexdigest()
    return payload


def validate_manifest(
    path: Path,
    *,
    tau2_root: Path,
    split_manifest: Path,
    unseal_receipt: Path,
    source_root: Path,
) -> dict[str, Any]:
    observed = read_json(path)
    expected = build_official_manifest(
        tau2_root=tau2_root,
        split_manifest=split_manifest,
        unseal_receipt=unseal_receipt,
        source_root=source_root,
    )
    if observed != expected:
        raise EvaluationError("official fault manifest differs from recomputation")
    return observed


def shard_rows(
    rows: list[dict[str, Any]],
    *,
    shard_index: int,
    num_shards: int = NUM_SHARDS,
) -> list[dict[str, Any]]:
    if num_shards != NUM_SHARDS or shard_index not in range(NUM_SHARDS):
        raise EvaluationError("official evaluation requires exactly two shards")
    ordered = sorted(
        rows,
        key=lambda row: (str(row["domain"]), int(row["task_id"])),
    )
    selected = ordered[shard_index::num_shards]
    if len(selected) != 30:
        raise EvaluationError("each official evaluation shard must contain 30 tasks")
    return selected


def validate_checkpoint_registry(
    path: Path,
    *,
    receipt: Mapping[str, Any],
    arm: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = (receipt.get("bound_artifacts") or {}).get("checkpoint_registry")
    registry = read_json(path)
    if (
        not isinstance(binding, Mapping)
        or binding.get("path") != str(path.resolve())
        or binding.get("sha256") != sha256_file(path)
        or registry.get("protocol") != CHECKPOINT_PROTOCOL
        or registry.get("source_commit") != receipt.get("source_commit")
        or registry.get("registered_arms") != list(ARMS)
        or registry.get("registered_training_seeds") != [TRAINING_SEED]
        or registry.get("official_test_used") is not False
        or registry.get("official_test_sealed") is not True
    ):
        raise EvaluationError("checkpoint registry/unseal binding drift")
    entry = (registry.get("entries") or {}).get(arm)
    if (
        not isinstance(entry, Mapping)
        or entry.get("paper_arm") != arm
        or entry.get("training_seed") != TRAINING_SEED
        or not str(entry.get("model_id", "")).startswith("openai/")
    ):
        raise EvaluationError(f"{arm}: checkpoint registry entry drift")
    return registry, dict(entry)


def _served_ids(api_base: str, *, api_key: str) -> set[str]:
    request = Request(
        api_base.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise EvaluationError(f"model endpoint probe failed: {api_base}") from error
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise EvaluationError("model endpoint returned no model list")
    return {
        str(row["id"])
        for row in data
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }


def register_official_agents() -> None:
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.registry import registry

    class OfficialAgent(LLMAgent):
        def _generate_next_message(self, message, state):
            return legacy.normalize_tool_only_message(
                super()._generate_next_message(message, state)
            )

    class OfficialFaultAgent(OfficialAgent):
        def __init__(self, tools, domain_policy, task, llm, llm_args):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                llm=llm,
                llm_args=llm_args,
            )
            injection = FAULT_INJECTIONS.get(str(task.id))
            if injection is None:
                raise EvaluationError(f"task {task.id} lacks frozen fault data")
            self._injection = injection
            self._injected = False

        def _generate_next_message(self, message, state):
            if not self._injected and isinstance(message, UserMessage):
                state.messages.append(message)
                self._injected = True
                return AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=self._injection["tool_call_id"],
                            name=self._injection["tool_name"],
                            arguments=self._injection["arguments"],
                            requestor="assistant",
                        )
                    ],
                    cost=0.0,
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                    raw_data={
                        "v5_stage1_injected_fault": True,
                        "v6_10_official_controlled_error": True,
                        "expected_tool_error": True,
                    },
                    generation_time_seconds=0.0,
                )
            return super()._generate_next_message(message, state)

    def clean_factory(tools, domain_policy, **kwargs):
        return OfficialAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    def fault_factory(tools, domain_policy, **kwargs):
        return OfficialFaultAgent(
            tools=tools,
            domain_policy=domain_policy,
            task=kwargs.get("task"),
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    if registry.get_agent_factory("v6_10_official_agent") is None:
        registry.register_agent_factory(clean_factory, "v6_10_official_agent")
    if registry.get_agent_factory("v6_10_official_fault_agent") is None:
        registry.register_agent_factory(
            fault_factory, "v6_10_official_fault_agent"
        )


def run_condition(
    *,
    domain: str,
    rows: list[dict[str, Any]],
    condition: str,
    args: argparse.Namespace,
    agent_model: str,
) -> Path:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_tasks

    if condition == "error":
        FAULT_INJECTIONS.clear()
        FAULT_INJECTIONS.update(
            {
                str(row["task_id"]): dict(row["error_condition"])
                for row in rows
            }
        )
    output_path = (
        args.output_dir
        / "raw"
        / (
            f"{domain}_{condition}.shard-{args.shard_index:03d}"
            f"-of-{NUM_SHARDS:03d}.json"
        )
    )
    if output_path.exists():
        raise EvaluationError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dir = args.output_dir / "logs" / output_path.stem
    config = TextRunConfig(
        domain=domain,
        agent=(
            "v6_10_official_agent"
            if condition == "clean"
            else "v6_10_official_fault_agent"
        ),
        user="user_simulator",
        llm_agent=agent_model,
        llm_args_agent=legacy.endpoint_args(
            args.agent_api_base,
            args.agent_api_key,
            max_tokens=MAX_TOKENS,
            seed=args.evaluation_seed,
        ),
        llm_user=args.user_model,
        llm_args_user=legacy.endpoint_args(
            args.user_api_base,
            args.user_api_key,
            max_tokens=MAX_TOKENS,
            seed=args.evaluation_seed,
        ),
        num_trials=NUM_TRIALS,
        max_steps=MAX_STEPS,
        max_errors=10,
        timeout=TIMEOUT_SECONDS,
        max_concurrency=1,
        seed=args.evaluation_seed,
        log_level="INFO",
        max_retries=1,
        retry_delay=1.0,
        auto_resume=False,
        hallucination_retries=0,
        enforce_communication_protocol=False,
        verbose_logs=True,
    )
    run_tasks(
        config,
        legacy.select_tasks(domain, rows),
        save_path=output_path,
        save_dir=save_dir,
        evaluation_type=EvaluationType.ALL,
        console_display=True,
        results_format="json",
    )
    if not output_path.is_file():
        raise EvaluationError(f"Tau2 did not write {output_path}")
    return output_path


def extract_rows(
    paths: list[Path],
    *,
    manifest_rows: list[dict[str, Any]],
    arm: str,
    evaluation_seed: int,
    shard_index: int,
    unseal_receipt_sha256: str,
    evaluator_sha256: str,
    checkpoint_fingerprint: str,
    agent_service_receipt_sha256: str,
    user_judge_service_receipt_sha256: str,
) -> list[dict[str, Any]]:
    manifest = {
        (str(row["domain"]), str(row["task_id"])): row
        for row in manifest_rows
    }
    extracted: list[dict[str, Any]] = []
    observed: set[tuple[str, str, str]] = set()
    for path in paths:
        stem = path.name.split(".")[0]
        domain, condition = stem.split("_", 1)[:2]
        if domain not in {"retail", "airline"} or condition not in {
            "clean",
            "error",
        }:
            raise EvaluationError(
                f"cannot infer domain/condition from {path.name}"
            )
        simulations = read_json(path).get("simulations")
        if not isinstance(simulations, list):
            raise EvaluationError(f"{path}: simulations are missing")
        for simulation in simulations:
            if not isinstance(simulation, dict):
                raise EvaluationError(f"{path}: malformed simulation")
            task_id = str(simulation.get("task_id"))
            key = (domain, task_id, condition)
            if key in observed or (domain, task_id) not in manifest:
                raise EvaluationError(
                    f"duplicate or unregistered result row {key}"
                )
            observed.add(key)
            fault = (
                manifest[(domain, task_id)]["error_condition"]
                if condition == "error"
                else None
            )
            interface = result_protocol._injection_audit(
                simulation,
                condition=condition,
                expected_fault=fault,
            )
            reward = result_protocol._reward(simulation)
            extracted.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "condition": condition,
                    "fault_family": (
                        fault.get("fault_family") if fault is not None else None
                    ),
                    "official_reward": reward,
                    "task_success": math.isclose(
                        reward, 1.0, rel_tol=0.0, abs_tol=1e-12
                    ),
                    "duration_seconds": float(
                        simulation.get("duration") or 0.0
                    ),
                    "termination_reason": simulation.get(
                        "termination_reason"
                    ),
                    **interface,
                }
            )
    rows: list[dict[str, Any]] = []
    for row in extracted:
        condition = (
            "controlled_error" if row["condition"] == "error" else "clean"
        )
        rows.append(
            {
                "protocol": PROTOCOL,
                "terminal_status": "PASS",
                "arm": arm,
                "training_seed": TRAINING_SEED,
                "evaluation_seed": evaluation_seed,
                "domain": row["domain"],
                "task_id": row["task_id"],
                "condition": condition,
                "fault_family": row["fault_family"],
                "official_reward": row["official_reward"],
                "task_success": row["task_success"],
                "duration_seconds": row["duration_seconds"],
                "termination_reason": row["termination_reason"],
                "injection_count": row["injected_fault_count"],
                "injection_observed_as_error": row[
                    "injected_fault_observed_as_error"
                ],
                "injection_repeated_by_harness": False,
                "policy_repeated_same_injected_call": row[
                    "repeated_same_injected_call"
                ],
                "tool_calls": row["tool_calls"],
                "tool_results": row["tool_results"],
                "tool_errors": row["tool_errors"],
                "shard_index": shard_index,
                "num_shards": NUM_SHARDS,
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "agent_service_receipt_sha256": agent_service_receipt_sha256,
                "user_judge_service_receipt_sha256": (
                    user_judge_service_receipt_sha256
                ),
                "unseal_receipt_sha256": unseal_receipt_sha256,
                "evaluator_sha256": evaluator_sha256,
                "official_test_used": True,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        re.fullmatch(r"[0-9a-f]{64}", args.agent_service_receipt_sha256)
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            args.user_judge_service_receipt_sha256,
        )
        is None
    ):
        raise EvaluationError("service receipt SHA-256 is invalid")
    source_root = Path(__file__).resolve().parents[1]
    receipt = validate_unseal(
        args.unseal_receipt,
        source_root=source_root,
        tau2_root=args.tau2_root,
        split_manifest=args.split_manifest,
    )
    manifest = validate_manifest(
        args.manifest,
        tau2_root=args.tau2_root,
        split_manifest=args.split_manifest,
        unseal_receipt=args.unseal_receipt,
        source_root=source_root,
    )
    registry, entry = validate_checkpoint_registry(
        args.checkpoint_registry,
        receipt=receipt,
        arm=args.arm,
    )
    if args.evaluation_seed not in EVALUATION_SEEDS:
        raise EvaluationError("evaluation seed is not frozen")
    agent_model = str(entry["model_id"])
    user_judge = registry.get("user_judge")
    if (
        not isinstance(user_judge, Mapping)
        or args.user_model != user_judge.get("model_id")
    ):
        raise EvaluationError("user/judge model alias drift")
    if agent_model not in _served_ids(
        args.agent_api_base, api_key=args.agent_api_key
    ):
        raise EvaluationError("agent endpoint does not serve the registered arm")
    if args.user_model not in _served_ids(
        args.user_api_base, api_key=args.user_api_key
    ):
        raise EvaluationError("user endpoint does not serve the registered model")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise EvaluationError("evaluation output directory must be absent or empty")
    selected = shard_rows(
        list(manifest["rows"]),
        shard_index=args.shard_index,
    )
    legacy.configure_tau2_path(args.tau2_root)
    register_official_agents()
    legacy.patch_local_nl_judge(
        args.user_model,
        legacy.endpoint_args(
            args.user_api_base,
            args.user_api_key,
            max_tokens=MAX_TOKENS,
            seed=args.evaluation_seed,
        ),
    )
    raw_paths: list[Path] = []
    for domain in ("retail", "airline"):
        domain_rows = [row for row in selected if row["domain"] == domain]
        for condition in ("clean", "error"):
            raw_paths.append(
                run_condition(
                    domain=domain,
                    rows=domain_rows,
                    condition=condition,
                    args=args,
                    agent_model=agent_model,
                )
            )
    checkpoint = entry.get("checkpoint")
    fingerprint = (
        checkpoint.get("fingerprint")
        if isinstance(checkpoint, Mapping)
        else None
    )
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise EvaluationError("checkpoint fingerprint is missing")
    rows = extract_rows(
        raw_paths,
        manifest_rows=selected,
        arm=args.arm,
        evaluation_seed=args.evaluation_seed,
        shard_index=args.shard_index,
        unseal_receipt_sha256=sha256_file(args.unseal_receipt),
        evaluator_sha256=sha256_file(Path(__file__).resolve()),
        checkpoint_fingerprint=fingerprint,
        agent_service_receipt_sha256=args.agent_service_receipt_sha256,
        user_judge_service_receipt_sha256=(
            args.user_judge_service_receipt_sha256
        ),
    )
    if len(rows) != 60:
        raise EvaluationError("each shard must emit 60 condition rows")
    rows_path = args.output_dir / "rows.jsonl"
    write_jsonl(rows_path, rows)
    receipt_payload = {
        "protocol": "v6_10_official_evaluation_shard_receipt_v1",
        "status": "PASS",
        "arm": args.arm,
        "evaluation_seed": args.evaluation_seed,
        "shard_index": args.shard_index,
        "num_shards": NUM_SHARDS,
        "row_count": len(rows),
        "rows_sha256": sha256_file(rows_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "unseal_receipt_sha256": sha256_file(args.unseal_receipt),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "raw_result_sha256": {
            str(path.relative_to(args.output_dir)): sha256_file(path)
            for path in raw_paths
        },
        "official_test_used": True,
        "changes_after_unseal": False,
    }
    receipt_payload["receipt_sha256"] = hashlib.sha256(
        canonical(receipt_payload).encode("utf-8")
    ).hexdigest()
    write_json(args.output_dir / "evaluation_receipt.json", receipt_payload)
    return receipt_payload


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--unseal-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-manifest")
    prepare.add_argument("--tau2-root", type=Path, required=True)
    prepare.add_argument("--split-manifest", type=Path, required=True)
    prepare.add_argument("--unseal-receipt", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("run")
    _common_parser(execute)
    execute.add_argument("--checkpoint-registry", type=Path, required=True)
    execute.add_argument("--arm", choices=ARMS, required=True)
    execute.add_argument(
        "--evaluation-seed",
        type=int,
        choices=EVALUATION_SEEDS,
        required=True,
    )
    execute.add_argument(
        "--shard-index", type=int, choices=range(NUM_SHARDS), required=True
    )
    execute.add_argument("--agent-api-base", required=True)
    execute.add_argument("--agent-api-key", default="v610-agent-local")
    execute.add_argument("--user-model", required=True)
    execute.add_argument("--user-api-base", required=True)
    execute.add_argument("--user-api-key", default="v610-user-local")
    execute.add_argument("--agent-service-receipt-sha256", required=True)
    execute.add_argument("--user-judge-service-receipt-sha256", required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "tau2_root",
        "split_manifest",
        "unseal_receipt",
        "manifest",
        "checkpoint_registry",
        "output",
        "output_dir",
    ):
        value = getattr(args, name, None)
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def main() -> None:
    args = parse_args()
    if args.command == "prepare-manifest":
        source_root = Path(__file__).resolve().parents[1]
        payload = build_official_manifest(
            tau2_root=args.tau2_root,
            split_manifest=args.split_manifest,
            unseal_receipt=args.unseal_receipt,
            source_root=source_root,
        )
        write_json(args.output, payload)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "manifest": str(args.output),
                    "manifest_sha256": payload["manifest_sha256"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
