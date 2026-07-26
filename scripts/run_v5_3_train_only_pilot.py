#!/usr/bin/env python3
"""Build, run, and audit the frozen V5.3 train-only feasibility pilot.

The pilot uses 24 task IDs sampled before any V5.3 outcomes are observed.  It
generates twelve clean and twelve controlled-error attempts per task.  Pilot
trajectories are permanently excluded from formal SFT data; they only decide
whether the disjoint-seed formal generation is worth running.

Subcommands:

``manifest``
    Build the frozen 70-task arm-train universe and deterministic 24-task
    pilot.
``generate``
    Run one of three disjoint pilot task shards against already-running local
    vLLM endpoints.
``audit``
    Apply the task-level one-pair/second-pair go/no-go rule and emit a single
    fail-closed report.  This command never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import audit_v5_3_feasibility as feasibility
    import prepare_v5_3_sft_causal as partition_module
    import run_v5_sft_causal_generate as generation
    import v5_3_protocol as protocol
    import v5_judge_audit_contract as judge_contract
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts import audit_v5_3_feasibility as feasibility
    from scripts import prepare_v5_3_sft_causal as partition_module
    from scripts import run_v5_sft_causal_generate as generation
    from scripts import v5_3_protocol as protocol
    from scripts import v5_judge_audit_contract as judge_contract
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit


ROOT = Path(__file__).resolve().parents[1]
PILOT_PROTOCOL = protocol.PROTOCOL
PILOT_SELECTION_SALT = "v5.3-pilot"
EXPECTED_SPLIT_SHA256 = (
    "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
)
EXPECTED_PARTITION_SHA256 = (
    "c90255a2cf4100d8cf5b5fcf3e684144ed1a393fad1c214058146721c4ed6818"
)
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
NUM_SHARDS = 3
MAX_STEPS = 60
MAX_TOKENS = 512
TIMEOUT_SECONDS = 900.0
TRAJECTORY_TEMPERATURE = 0.2
TRAJECTORY_TOP_P = 0.95
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0


class PilotError(RuntimeError):
    """A V5.3 pilot invariant failed."""


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PilotError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise PilotError(f"expected JSON object: {path}")
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise PilotError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_sort_key(task_id: str) -> tuple[int, str]:
    try:
        return int(task_id), task_id
    except ValueError as error:
        raise PilotError(f"non-numeric tau2 task ID: {task_id!r}") from error


def build_effective_universe_and_pilot(
    *,
    tau2_root: Path,
    split_manifest_path: Path,
    generation_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create the frozen 70-task GT universe and 24-task stratified pilot."""

    protocol.validate_frozen_seed_schedules()
    if file_sha256(split_manifest_path) != EXPECTED_SPLIT_SHA256:
        raise PilotError("Stage-0 split SHA-256 drift")
    split = generation.load_split_manifest(split_manifest_path)
    full_generation = generation.load_inner_train_manifest(
        generation_manifest_path,
        split_manifest=split,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
    )
    if full_generation.get("tau2_commit") != TAU2_COMMIT:
        raise PilotError("generation manifest tau2 commit drift")
    generation.validate_local_manifest_sources(full_generation, tau2_root)
    gt_filter = full_generation.get("_validated_gt_compatibility_filter")
    if not isinstance(gt_filter, dict):
        raise PilotError("V5.3 generation manifest lacks the validated GT filter")
    if set(gt_filter["excluded_task_ids"]) != {
        f"{domain}:{task_id}"
        for domain, task_id in protocol.GT_INCOMPATIBLE_INNER_TRAIN_TASKS
    }:
        raise PilotError("V5.3 GT-incompatible exclusion set drift")
    partition = partition_module.build_task_partition(
        split,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
        seed=protocol.SEED,
        strict_formal=True,
    )
    if partition["canonical_sha256"] != EXPECTED_PARTITION_SHA256:
        raise PilotError("70/8 partition digest drift")

    manifest_by_key = {
        (str(row["domain"]), str(row["task_id"])): row
        for row in full_generation["rows"]
    }
    universe_rows: list[dict[str, Any]] = []
    for domain in ("retail", "airline"):
        for task_id in partition["domains"][domain]["train_ids"]:
            key = (domain, str(task_id))
            source = manifest_by_key.get(key)
            if source is None:
                raise PilotError(f"missing V5.3 source task {domain}:{task_id}")
            family = source["error_condition"]["fault_family"]
            universe_rows.append(
                {
                    "domain": domain,
                    "task_id": str(task_id),
                    "task_key": f"{domain}:{task_id}",
                    "fault_family": family,
                    "teacher_route": "ground_truth",
                    "source_split": "derived_inner_train",
                }
            )
    if len(universe_rows) != protocol.FORMAL_TRAIN_TASKS:
        raise PilotError("effective arm-train universe must contain 70 tasks")

    selected_keys: set[tuple[str, str]] = set()
    selected_rows: list[dict[str, Any]] = []
    for family, quota in protocol.PILOT_FAMILY_QUOTAS.items():
        candidates = [
            row
            for row in universe_rows
            if row["fault_family"] == family
        ]
        ordered = sorted(
            candidates,
            key=lambda row: (
                hashlib.sha256(
                    (
                        f"{PILOT_SELECTION_SALT}|{protocol.SEED}|"
                        f"{row['domain']}|{family}|{row['task_id']}"
                    ).encode("utf-8")
                ).hexdigest(),
                _task_sort_key(row["task_id"]),
            ),
        )
        if quota < 0 or len(ordered) < quota:
            raise PilotError(f"pilot quota cannot be filled for {family}")
        selected_rows.extend(ordered[:quota])
        selected_keys.update(
            (row["domain"], row["task_id"]) for row in ordered[:quota]
        )

    selected_rows = sorted(
        selected_rows,
        key=lambda row: (row["domain"], _task_sort_key(row["task_id"])),
    )
    if len(selected_rows) != protocol.PILOT_TASKS:
        raise PilotError("pilot must contain exactly 24 tasks")
    if len(selected_keys) != protocol.PILOT_TASKS:
        raise PilotError("pilot task selection contains duplicates")
    domain_counts = Counter(row["domain"] for row in selected_rows)
    family_counts = Counter(row["fault_family"] for row in selected_rows)
    route_counts = Counter(row["teacher_route"] for row in selected_rows)
    if domain_counts != {"retail": 18, "airline": 6}:
        raise PilotError(f"pilot domain strata drift: {domain_counts}")
    if dict(family_counts) != protocol.PILOT_FAMILY_QUOTAS:
        raise PilotError(f"pilot fault-family strata drift: {family_counts}")
    if route_counts != {"ground_truth": 24}:
        raise PilotError(f"pilot teacher-route strata drift: {route_counts}")

    universe = {
        "protocol": "v5_3_effective_arm_train_task_universe",
        "created_without_v5_3_outcomes": True,
        "source_split": "derived_inner_train",
        "official_test_used": False,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "inner_train_partition_sha256": EXPECTED_PARTITION_SHA256,
        "generation_manifest_sha256": file_sha256(generation_manifest_path),
        "tau2_commit": TAU2_COMMIT,
        "planned_task_ids": [row["task_key"] for row in universe_rows],
        "task_universe_complete": True,
        "counts": {
            "total": len(universe_rows),
            "ground_truth_route": len(universe_rows),
            "structurally_excluded_before_partition": len(
                protocol.GT_INCOMPATIBLE_INNER_TRAIN_TASKS
            ),
        },
        "gt_compatibility_rule": {
            "included": "evaluation_criteria.actions is nonempty",
            "excluded_task_ids": sorted(gt_filter["excluded_task_ids"]),
            "excluded_before_v5_3_rollouts": True,
            "outcome_dependent_exclusion_allowed": False,
            "standard_teacher_mixed_into_universe": False,
        },
        "rows": universe_rows,
    }
    universe_sha256 = protocol.sha256_text(protocol.canonical_json(universe))
    source_rows = {
        (row["domain"], str(row["task_id"])): row
        for row in full_generation["rows"]
    }
    pilot_rows = []
    for selected in selected_rows:
        source = dict(source_rows[(selected["domain"], selected["task_id"])])
        pilot_rows.append(source)
    pilot = {
        "protocol": PILOT_PROTOCOL,
        "stage": "train_only_pilot",
        "training": False,
        "formal_data": False,
        "pilot_trajectories_may_enter_formal_data": False,
        "created_without_v5_3_outcomes": True,
        "source_split": "derived_inner_train",
        "official_test_used": False,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "inner_train_partition_sha256": EXPECTED_PARTITION_SHA256,
        "generation_manifest_sha256": file_sha256(generation_manifest_path),
        "effective_task_universe_canonical_sha256": universe_sha256,
        "selection": {
            "seed": protocol.SEED,
            "salt": PILOT_SELECTION_SALT,
            "uses_rollout_or_validation_outcomes": False,
            "domain_counts": dict(domain_counts),
            "fault_family_counts": dict(family_counts),
            "teacher_route": "ground_truth",
        },
        "planned_task_ids": [
            f"{row['domain']}:{row['task_id']}" for row in pilot_rows
        ],
        "task_universe_complete": True,
        "attempts_per_task_per_condition": (
            protocol.ATTEMPTS_PER_TASK_PER_CONDITION
        ),
        "base_seed": protocol.PILOT_BASE_SEED,
        "trial_seeds": list(protocol.PILOT_TRIAL_SEEDS),
        "formal_base_seed": protocol.FORMAL_BASE_SEED,
        "formal_trial_seeds": list(protocol.FORMAL_TRIAL_SEEDS),
        "pilot_formal_seed_sets_disjoint": True,
        "rows": pilot_rows,
    }
    return universe, pilot


def command_manifest(args: argparse.Namespace) -> int:
    universe, pilot = build_effective_universe_and_pilot(
        tau2_root=args.tau2_root.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        generation_manifest_path=args.generation_manifest.resolve(),
    )
    output_dir = args.output_dir.resolve()
    universe_path = output_dir / "effective_task_universe.json"
    pilot_path = output_dir / "pilot_manifest.json"
    write_json_exclusive(universe_path, universe)
    write_json_exclusive(pilot_path, pilot)
    print(
        json.dumps(
            {
                "status": "PASS",
                "effective_task_universe": str(universe_path),
                "pilot_manifest": str(pilot_path),
                "pilot_tasks": len(pilot["rows"]),
                "official_test_used": False,
            },
            indent=2,
        )
    )
    return 0


def _validate_pilot_manifest(
    pilot_path: Path,
    full_generation_path: Path,
) -> dict[str, Any]:
    pilot = load_json_object(pilot_path)
    if pilot.get("protocol") != PILOT_PROTOCOL:
        raise PilotError("unexpected pilot manifest protocol")
    if (
        pilot.get("training") is not False
        or pilot.get("formal_data") is not False
        or pilot.get("pilot_trajectories_may_enter_formal_data") is not False
        or pilot.get("official_test_used") is not False
    ):
        raise PilotError("pilot/formal-data or official-test claim drift")
    rows = pilot.get("rows")
    if not isinstance(rows, list) or len(rows) != protocol.PILOT_TASKS:
        raise PilotError("pilot manifest must contain exactly 24 rows")
    if pilot.get("generation_manifest_sha256") != file_sha256(
        full_generation_path
    ):
        raise PilotError("pilot source generation manifest hash drift")
    if pilot.get("base_seed") != protocol.PILOT_BASE_SEED:
        raise PilotError("pilot base seed drift")
    if tuple(pilot.get("trial_seeds") or ()) != protocol.PILOT_TRIAL_SEEDS:
        raise PilotError("pilot trial seed schedule drift")
    if set(protocol.PILOT_TRIAL_SEEDS) & set(protocol.FORMAL_TRIAL_SEEDS):
        raise PilotError("pilot/formal seed overlap")
    identities = {
        (str(row.get("domain")), str(row.get("task_id"))) for row in rows
    }
    if len(identities) != protocol.PILOT_TASKS:
        raise PilotError("pilot manifest contains duplicate identities")
    for row in rows:
        if row.get("source_split") != "derived_inner_train":
            raise PilotError("pilot row is not derived-inner-train")
    return pilot


def _sampling_args(
    *,
    api_base: str,
    api_key: str,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "api_base": api_base,
        "api_key": api_key,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "parallel_tool_calls": False,
    }


def _run_pilot_condition(
    *,
    domain: str,
    condition: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    teacher_args: dict[str, Any],
    user_args: dict[str, Any],
) -> Path:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_tasks

    agent_name = (
        "v5_stage1_generation_gt_agent"
        if condition == "clean"
        else "v5_stage1_generation_fault_gt_agent"
    )
    if condition == "error":
        generation.FAULT_INJECTIONS.clear()
        generation.FAULT_INJECTIONS.update(
            {
                str(row["task_id"]): dict(row["error_condition"])
                for row in rows
            }
        )
    output_path = args.output_dir / generation.output_filename(
        domain, condition, args.shard_index, NUM_SHARDS
    )
    if output_path.exists():
        raise PilotError(f"refusing to overwrite pilot result: {output_path}")
    save_dir = args.output_dir / "logs" / output_path.stem
    config = TextRunConfig(
        domain=domain,
        agent=agent_name,
        user="user_simulator",
        llm_agent=generation.litellm_openai_model(protocol.TEACHER_MODEL),
        llm_args_agent=teacher_args,
        llm_user=generation.litellm_openai_model(protocol.USER_JUDGE_MODEL),
        llm_args_user=user_args,
        num_trials=protocol.ATTEMPTS_PER_TASK_PER_CONDITION,
        max_steps=MAX_STEPS,
        max_errors=10,
        timeout=TIMEOUT_SECONDS,
        max_concurrency=1,
        seed=protocol.PILOT_BASE_SEED,
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
        generation.select_tasks(domain, rows),
        save_path=output_path,
        save_dir=save_dir,
        evaluation_type=EvaluationType.ALL,
        console_display=True,
        results_format="json",
    )
    if not output_path.is_file():
        raise PilotError(f"tau2 did not write pilot result: {output_path}")
    _verify_result_seed_schedule(output_path, rows)
    return output_path


def _verify_result_seed_schedule(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    payload = load_json_object(path)
    simulations = payload.get("simulations")
    if not isinstance(simulations, list):
        raise PilotError(f"result lacks simulations: {path}")
    expected_tasks = {str(row["task_id"]) for row in rows}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for simulation in simulations:
        if not isinstance(simulation, dict):
            raise PilotError(f"malformed simulation in {path}")
        by_task[str(simulation.get("task_id"))].append(simulation)
    if set(by_task) != expected_tasks:
        raise PilotError(f"result task identities drift in {path}")
    for task_id, task_rows in by_task.items():
        trials = {row.get("trial") for row in task_rows}
        seeds = {row.get("seed") for row in task_rows}
        if trials != set(range(12)) or seeds != set(protocol.PILOT_TRIAL_SEEDS):
            raise PilotError(
                f"{path}: task {task_id} does not match the 12 fixed pilot seeds"
            )


def _write_incomplete_contract(
    *,
    args: argparse.Namespace,
    pilot: dict[str, Any],
    source_commit: str,
    dynamic_audit_path: Path,
    dynamic_audit_identity: dict[str, Any],
    rows: list[dict[str, Any]],
) -> Path:
    path = (
        args.output_dir
        / f"run_contract.shard-{args.shard_index:03d}-of-{NUM_SHARDS:03d}.json"
    )
    payload = {
        "protocol": "v5_stage1_inner_train_generation_run",
        "subprotocol": PILOT_PROTOCOL,
        "status": "INCOMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_split": "derived_inner_train",
        "official_test_used": False,
        "formal_data": False,
        "pilot_trajectories_may_enter_formal_data": False,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": file_sha256(args.split_manifest.resolve()),
        "generation_manifest": str(args.generation_manifest.resolve()),
        "generation_manifest_sha256": file_sha256(
            args.generation_manifest.resolve()
        ),
        "pilot_manifest": str(args.pilot_manifest.resolve()),
        "pilot_manifest_sha256": file_sha256(args.pilot_manifest.resolve()),
        "dynamic_audit": str(dynamic_audit_path),
        "dynamic_audit_identity": dynamic_audit_identity,
        "task_ids": [
            f"{row['domain']}:{row['task_id']}" for row in rows
        ],
        "task_universe_complete": False,
        "shard_index": args.shard_index,
        "num_shards": NUM_SHARDS,
        "num_trials": protocol.ATTEMPTS_PER_TASK_PER_CONDITION,
        "trial_seeds": list(protocol.PILOT_TRIAL_SEEDS),
        "teacher": {
            "model": protocol.TEACHER_MODEL,
            "revision": protocol.TEACHER_REVISION,
            "api_base": args.teacher_api_base,
            "tensor_parallel_size": 2,
        },
        "user_and_judge": {
            "model": protocol.USER_JUDGE_MODEL,
            "revision": protocol.USER_JUDGE_REVISION,
            "api_base": args.user_api_base,
            "strict_judge_schema": True,
            "judge_format_retries": 1,
        },
        "decoding": {
            "trajectory_temperature": TRAJECTORY_TEMPERATURE,
            "trajectory_top_p": TRAJECTORY_TOP_P,
            "judge_temperature": JUDGE_TEMPERATURE,
            "judge_top_p": JUDGE_TOP_P,
            "max_tokens": MAX_TOKENS,
            "max_steps": MAX_STEPS,
            "parallel_tool_calls": False,
        },
        "user_loop_db0_strategy": {
            "stronger_user_simulator": True,
            "bounded_nonzero_fixed_seed_attempts": True,
            "diagnostic_only": True,
            "hidden_rescue": False,
            "failed_attempts_repaired_or_relabelled": False,
            "extra_attempts_after_budget": False,
        },
        "result_sha256": {},
        "strict_judge_audit_evidence": {},
    }
    write_json_exclusive(path, payload)
    return path


def _finalize_contract(contract_path: Path, paths: Sequence[Path]) -> None:
    payload = load_json_object(contract_path)
    if payload.get("status") != "INCOMPLETE":
        raise PilotError("pilot run contract is not incomplete")
    result_paths = [path.resolve() for path in paths]
    payload["result_sha256"] = {
        path.name: file_sha256(path) for path in sorted(paths)
    }
    payload["strict_judge_audit_evidence"] = (
        judge_contract.validate_strict_judge_evidence(result_paths)
    )
    payload["status"] = "COMPLETE"
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    temporary = contract_path.with_name(f".{contract_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, contract_path)


def command_generate(args: argparse.Namespace) -> int:
    protocol.validate_frozen_seed_schedules()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_index not in range(NUM_SHARDS):
        raise PilotError("--shard-index must be 0, 1, or 2")
    if args.teacher_revision != protocol.TEACHER_REVISION:
        raise PilotError("32B teacher revision drift")
    if args.user_judge_revision != protocol.USER_JUDGE_REVISION:
        raise PilotError("14B user/judge revision drift")
    source_commit = generation.validate_provenance(
        expected_source_commit=args.expected_source_commit.lower(),
        teacher_revision=args.teacher_revision,
        user_revision=args.user_judge_revision,
        judge_revision=args.user_judge_revision,
    )
    split_path = args.split_manifest.resolve()
    full_manifest_path = args.generation_manifest.resolve()
    if file_sha256(split_path) != EXPECTED_SPLIT_SHA256:
        raise PilotError("Stage-0 split SHA-256 drift")
    split = generation.load_split_manifest(split_path)
    full_manifest = generation.load_inner_train_manifest(
        full_manifest_path,
        split_manifest=split,
        split_manifest_sha256=EXPECTED_SPLIT_SHA256,
    )
    pilot = _validate_pilot_manifest(
        args.pilot_manifest.resolve(),
        full_manifest_path,
    )
    dynamic_audit_path = args.dynamic_audit.resolve()
    dynamic_audit_identity = load_complete_dynamic_audit(
        dynamic_audit_path,
        manifest_path=full_manifest_path,
        split_manifest_path=split_path,
        expected_source_split="derived_inner_train",
        expected_task_ids={
            f"{row['domain']}:{row['task_id']}"
            for row in full_manifest["rows"]
        },
    )
    generation.validate_local_manifest_sources(
        full_manifest, args.tau2_root.resolve()
    )
    rows = generation.shard_rows(
        list(pilot["rows"]),
        shard_index=args.shard_index,
        num_shards=NUM_SHARDS,
    )
    if not rows:
        raise PilotError("selected pilot shard is empty")
    contract_path = _write_incomplete_contract(
        args=args,
        pilot=pilot,
        source_commit=source_commit,
        dynamic_audit_path=dynamic_audit_path,
        dynamic_audit_identity=dynamic_audit_identity,
        rows=rows,
    )
    generation.configure_tau2_path(args.tau2_root.resolve())
    gt_compatibility = generation.preflight_gt_compatibility(
        list(full_manifest["rows"]),
        teacher_mode="ground_truth",
        gt_compatibility_filter=full_manifest[
            "_validated_gt_compatibility_filter"
        ],
    )
    if gt_compatibility.get("status") != "PASS":
        raise PilotError("ground-truth compatibility preflight did not pass")
    os.environ.setdefault("OPENAI_API_KEY", args.teacher_api_key)
    generation.register_fault_agents()
    teacher_args = _sampling_args(
        api_base=args.teacher_api_base,
        api_key=args.teacher_api_key,
        temperature=TRAJECTORY_TEMPERATURE,
        top_p=TRAJECTORY_TOP_P,
        seed=protocol.PILOT_BASE_SEED,
    )
    user_args = _sampling_args(
        api_base=args.user_api_base,
        api_key=args.user_api_key,
        temperature=TRAJECTORY_TEMPERATURE,
        top_p=TRAJECTORY_TOP_P,
        seed=protocol.PILOT_BASE_SEED,
    )
    judge_args = _sampling_args(
        api_base=args.user_api_base,
        api_key=args.user_api_key,
        temperature=JUDGE_TEMPERATURE,
        top_p=JUDGE_TOP_P,
        seed=protocol.PILOT_BASE_SEED,
    )
    generation.patch_local_nl_judge(
        generation.litellm_openai_model(protocol.USER_JUDGE_MODEL),
        judge_args,
    )
    outputs: list[Path] = []
    for domain in ("retail", "airline"):
        group = [row for row in rows if row["domain"] == domain]
        if not group:
            continue
        for condition in ("clean", "error"):
            outputs.append(
                _run_pilot_condition(
                    domain=domain,
                    condition=condition,
                    rows=group,
                    args=args,
                    teacher_args=teacher_args,
                    user_args=user_args,
                )
            )
    _finalize_contract(contract_path, outputs)
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "train_only_pilot_generation",
                "shard_index": args.shard_index,
                "result_files": [str(path) for path in outputs],
                "training_started": False,
                "official_test_used": False,
            },
            indent=2,
        )
    )
    return 0


def _load_simulations(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        lowered = str(path).lower()
        if "official_test" in lowered or "sealed_test" in lowered:
            raise PilotError(f"prohibited test path: {path}")
        payload = load_json_object(path)
        simulations = payload.get("simulations")
        if not isinstance(simulations, list):
            raise PilotError(f"pilot result lacks simulations: {path}")
        for simulation in simulations:
            if not isinstance(simulation, dict):
                raise PilotError(f"malformed simulation in {path}")
            rows.append(simulation)
    return rows


def _judge_audit_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    calls = evidence.get("calls")
    if (
        evidence.get("protocol") != judge_contract.EVIDENCE_PROTOCOL
        or evidence.get("status") != "PASS"
        or not isinstance(calls, list)
        or not calls
        or evidence.get("expected_calls") != len(calls)
        or evidence.get("observed_unique_pass_audits") != len(calls)
    ):
        raise PilotError("strict-judge evidence mapping is malformed")
    return {
        "protocol": "v5_strict_nl_judge_v1",
        "evidence_protocol": judge_contract.EVIDENCE_PROTOCOL,
        "canonical_mapping_sha256": evidence["canonical_mapping_sha256"],
        "calls": len(calls),
        "format_rejections": sum(
            int(row["format_rejections"]) for row in calls
        ),
        "accepted_after_one_retry": sum(
            int(row["accepted_attempt"] == 2) for row in calls
        ),
        "exhausted_calls": 0,
        # The frozen runtime judge records raw bytes/hashes rather than
        # exposing a parse-method field.  Do not infer this value.
        "fenced_json_accepted": None,
        "all_nonempty_assertion_sets_have_exact_cardinality": True,
        "vacuous_all_empty_prevented": True,
    }


def _validate_pilot_manifest_for_audit(path: Path) -> dict[str, Any]:
    pilot = load_json_object(path)
    if (
        pilot.get("protocol") != PILOT_PROTOCOL
        or pilot.get("official_test_used") is not False
        or pilot.get("formal_data") is not False
        or pilot.get("pilot_trajectories_may_enter_formal_data") is not False
    ):
        raise PilotError("invalid pilot manifest claim boundary")
    rows = pilot.get("rows")
    planned = pilot.get("planned_task_ids")
    if (
        not isinstance(rows, list)
        or len(rows) != protocol.PILOT_TASKS
        or not isinstance(planned, list)
        or len(set(planned)) != protocol.PILOT_TASKS
    ):
        raise PilotError("pilot audit manifest must describe exactly 24 tasks")
    if tuple(pilot.get("trial_seeds") or ()) != protocol.PILOT_TRIAL_SEEDS:
        raise PilotError("pilot audit seed schedule drift")
    return pilot


def build_pilot_audit(
    *,
    input_paths: Sequence[Path],
    pilot_manifest_path: Path,
    judge_audit_paths: Sequence[Path],
) -> dict[str, Any]:
    pilot = _validate_pilot_manifest_for_audit(pilot_manifest_path)
    base = feasibility.audit_paths(
        list(input_paths),
        task_universe_path=pilot_manifest_path,
        min_tasks=1,
        min_pairs=1,
        max_pairs_per_task=protocol.MAX_PAIRS_PER_TASK,
        max_attempts=protocol.ATTEMPTS_PER_TASK_PER_CONDITION,
    )
    task_rows = base["task_audit"]
    if len(task_rows) != protocol.PILOT_TASKS:
        raise PilotError("feasibility audit did not cover all 24 pilot tasks")
    for row in task_rows:
        if row["observed_attempts"] != {"clean": 12, "error": 12}:
            raise PilotError(
                f"incomplete fixed attempt budget for {row['task_id']}"
            )
    one_pair = sum(
        row["constructible_cross_seed_pairs"] >= 1 for row in task_rows
    )
    two_pair = sum(
        row["constructible_cross_seed_pairs"] >= 2 for row in task_rows
    )
    one_rate = one_pair / protocol.PILOT_TASKS
    second_rate = two_pair / protocol.PILOT_TASKS
    projected_tasks = protocol.FORMAL_TRAIN_TASKS * one_rate
    projected_pairs = protocol.FORMAL_TRAIN_TASKS * (
        one_rate + second_rate
    )
    # The two integer count checks are the complete preregistered decision
    # rule.  Rates and 70-task projections below are interpretations of those
    # checks, not additional gates that can move after outcomes are observed.
    checks = {
        "one_pair_tasks_at_least_15": (
            one_pair >= protocol.PILOT_MIN_ONE_PAIR_TASKS
        ),
        "two_pair_tasks_at_least_4": (
            two_pair >= protocol.PILOT_MIN_TWO_PAIR_TASKS
        ),
    }
    go = all(checks.values())
    simulations = _load_simulations(input_paths)
    try:
        judge_evidence = judge_contract.validate_strict_judge_evidence(
            list(input_paths),
            audit_paths=judge_audit_paths,
        )
    except judge_contract.StrictJudgeEvidenceError as error:
        raise PilotError(str(error)) from error
    judge = _judge_audit_summary(judge_evidence)
    return {
        "protocol": PILOT_PROTOCOL,
        "status": "GO_FORMAL_GENERATION" if go else "NO_GO_STOP",
        "exit_code": 0 if go else 20,
        "provenance": {
            "source_commit": generation.git_commit(),
            "pilot_manifest_sha256": file_sha256(pilot_manifest_path),
            "input_sha256": {
                str(path.resolve()): file_sha256(path)
                for path in sorted(input_paths, key=lambda item: str(item.resolve()))
            },
            "strict_judge_audit_sha256": {
                row["audit_path"]: row["audit_sha256"]
                for row in judge_evidence["calls"]
            },
            "strict_judge_evidence_mapping_sha256": judge_evidence[
                "canonical_mapping_sha256"
            ],
            "official_test_used": False,
        },
        "decision": {
            "all_preregistered_checks_pass": go,
            "checks": checks,
            "observed": {
                "pilot_tasks": protocol.PILOT_TASKS,
                "tasks_with_at_least_one_pair": one_pair,
                "tasks_with_second_pair": two_pair,
                "one_pair_task_rate": one_rate,
                "second_pair_task_rate": second_rate,
                "projected_formal_tasks_with_pair": projected_tasks,
                "projected_formal_pairs": projected_pairs,
            },
            "thresholds": {
                "minimum_one_pair_tasks": (
                    protocol.PILOT_MIN_ONE_PAIR_TASKS
                ),
                "minimum_one_pair_rate": (
                    protocol.PILOT_MIN_ONE_PAIR_RATE
                ),
                "minimum_two_pair_tasks": (
                    protocol.PILOT_MIN_TWO_PAIR_TASKS
                ),
                "minimum_second_pair_rate": (
                    protocol.PILOT_MIN_SECOND_PAIR_RATE
                ),
                "implied_projected_formal_tasks": (
                    protocol.PILOT_MIN_PROJECTED_TASKS
                ),
                "implied_projected_formal_pairs": (
                    protocol.PILOT_MIN_PROJECTED_PAIRS
                ),
                "projection_is_diagnostic_not_extra_gate": True,
            },
        },
        "formal_gate_unchanged": {
            "minimum_distinct_task_ids": protocol.FORMAL_MIN_TASKS,
            "minimum_pairs": protocol.FORMAL_MIN_PAIRS,
            "maximum_pairs_per_task": protocol.MAX_PAIRS_PER_TASK,
            "may_be_relaxed_after_pilot": False,
        },
        "strict_judge": judge,
        "user_loop_db0_diagnostics": protocol.classify_user_loop_db0(
            simulations
        ),
        "base_task_level_audit": base,
        "claim_boundary": {
            "training_started": False,
            "pilot_trajectories_enter_formal_data": False,
            "pilot_outcomes_choose_arm_or_hyperparameters": False,
            "formal_seed_schedule_is_disjoint": True,
            "derived_validation_outcomes_used": False,
            "official_test_used": False,
            "positive_scientific_result_claimed": False,
        },
        "next_action": (
            "run the frozen formal 70-task generation with the disjoint "
            "formal seed schedule, then enforce the unchanged 40/48 gate"
            if go
            else "stop before formal generation and before every SFT arm"
        ),
    }


def command_audit(args: argparse.Namespace) -> int:
    result = build_pilot_audit(
        input_paths=[path.resolve() for path in args.inputs],
        pilot_manifest_path=args.pilot_manifest.resolve(),
        judge_audit_paths=[path.resolve() for path in args.judge_audits],
    )
    write_json_exclusive(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "observed": result["decision"]["observed"],
                "training_started": False,
                "official_test_used": False,
            },
            indent=2,
        )
    )
    return int(result["exit_code"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--tau2-root", type=Path, required=True)
    manifest.add_argument("--split-manifest", type=Path, required=True)
    manifest.add_argument("--generation-manifest", type=Path, required=True)
    manifest.add_argument("--output-dir", type=Path, required=True)
    manifest.set_defaults(handler=command_manifest)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--tau2-root", type=Path, required=True)
    generate.add_argument("--split-manifest", type=Path, required=True)
    generate.add_argument("--generation-manifest", type=Path, required=True)
    generate.add_argument("--dynamic-audit", type=Path, required=True)
    generate.add_argument("--pilot-manifest", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--shard-index", type=int, required=True)
    generate.add_argument("--teacher-api-base", default="http://127.0.0.1:8011/v1")
    generate.add_argument("--teacher-api-key", default="v5-3-teacher-local")
    generate.add_argument("--user-api-base", default="http://127.0.0.1:8001/v1")
    generate.add_argument("--user-api-key", default="v5-3-user-local")
    generate.add_argument(
        "--teacher-revision", default=protocol.TEACHER_REVISION
    )
    generate.add_argument(
        "--user-judge-revision", default=protocol.USER_JUDGE_REVISION
    )
    generate.add_argument("--expected-source-commit", required=True)
    generate.set_defaults(handler=command_generate)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--pilot-manifest", type=Path, required=True)
    audit.add_argument("--judge-audits", type=Path, nargs="*", default=[])
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("inputs", type=Path, nargs="+")
    audit.set_defaults(handler=command_audit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        PilotError,
        feasibility.DataError,
        RuntimeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(f"V5.3 pilot FAIL-CLOSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
