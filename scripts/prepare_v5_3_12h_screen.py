#!/usr/bin/env python3
"""Construct the isolated four-arm V5.3 12-hour exploratory screen.

The implementation reuses V5.3's trajectory eligibility, causal label
masking, tokenization, pair materialization, and deterministic joint matcher.
Only the screen orchestration and its smaller, separately registered gate are
new.  Screen artifacts are permanently ineligible for formal V5.3.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

try:
    import prepare_v5_3_sft_causal as v53
    import prepare_v5_sft_causal as v5
    import run_v5_sft_causal_generate as generation
    import v5_3_12h_protocol as protocol
    import v5_judge_audit_contract as judge_contract
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts import prepare_v5_3_sft_causal as v53
    from scripts import prepare_v5_sft_causal as v5
    from scripts import run_v5_sft_causal_generate as generation
    from scripts import v5_3_12h_protocol as protocol
    from scripts import v5_judge_audit_contract as judge_contract
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit


SPLIT_SHA256 = (
    "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
)
GENERATION_MANIFEST_PROTOCOL = "v5_3_multifault_data_construction"
SCREEN_MANIFEST_PROTOCOL = f"{protocol.PROTOCOL}:manifest_v1"
SCREEN_CONTRACT_PROTOCOL = "v5_stage1_inner_train_generation_run"
SCHEDULE_ROWS = v5.SCHEDULE_ROWS
MAX_SEQUENCE_TOKENS = 8192


class ScreenDataError(RuntimeError):
    """The screen data cannot cross its fail-closed barrier."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScreenDataError(f"invalid JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise ScreenDataError(f"expected JSON object: {path}")
    return payload


def build_screen_manifest(
    *,
    effective_universe: dict[str, Any],
    pilot_manifest: dict[str, Any],
    generation_manifest: dict[str, Any],
    generation_manifest_sha256: str,
) -> dict[str, Any]:
    """Transform only the outcome-free frozen pilot selection into the screen."""

    protocol.validate_constants()
    rows = pilot_manifest.get("rows")
    planned = pilot_manifest.get("planned_task_ids")
    universe_ids = effective_universe.get("planned_task_ids")
    generation_rows = generation_manifest.get("rows")
    generation_by_task = {
        f"{row.get('domain')}:{row.get('task_id')}": row
        for row in generation_rows or []
        if isinstance(row, dict)
    }
    pilot_selection = pilot_manifest.get("selection")
    if (
        effective_universe.get("protocol")
        != "v5_3_effective_arm_train_task_universe"
        or effective_universe.get("created_without_v5_3_outcomes") is not True
        or effective_universe.get("source_split") != "derived_inner_train"
        or effective_universe.get("official_test_used") is not False
        or effective_universe.get("split_manifest_sha256") != SPLIT_SHA256
        or effective_universe.get("generation_manifest_sha256")
        != generation_manifest_sha256
        or pilot_manifest.get("protocol")
        != "v5_3_train_only_feasibility_pilot"
        or pilot_manifest.get("training") is not False
        or pilot_manifest.get("formal_data") is not False
        or pilot_manifest.get("created_without_v5_3_outcomes") is not True
        or pilot_manifest.get("official_test_used") is not False
        or pilot_manifest.get("split_manifest_sha256") != SPLIT_SHA256
        or pilot_manifest.get("generation_manifest_sha256")
        != generation_manifest_sha256
        or pilot_manifest.get("base_seed") != protocol.PILOT_BASE_SEED
        or tuple(pilot_manifest.get("trial_seeds") or ())
        != protocol.PILOT_TRIAL_SEEDS
        or not isinstance(pilot_selection, dict)
        or pilot_selection.get("uses_rollout_or_validation_outcomes")
        is not False
        or not isinstance(rows, list)
        or len(rows) != protocol.TASKS
        or not isinstance(planned, list)
        or tuple(sorted(planned)) != tuple(sorted(protocol.PILOT_TASK_IDS))
        or [f"{row.get('domain')}:{row.get('task_id')}" for row in rows]
        != planned
        or not isinstance(universe_ids, list)
        or not set(planned) <= set(universe_ids)
        or generation_manifest.get("protocol")
        != GENERATION_MANIFEST_PROTOCOL
        or generation_manifest.get("paired_task_count") != 78
        or generation_manifest.get("official_test_used") is not False
        or not isinstance(generation_rows, list)
        or any(
            protocol.canonical(row)
            != protocol.canonical(generation_by_task.get(identity))
            for identity, row in zip(planned, rows, strict=True)
        )
    ):
        raise ScreenDataError("frozen V5.3 pilot selection identity drift")
    payload = {
        "protocol": SCREEN_MANIFEST_PROTOCOL,
        "screen_protocol": protocol.PROTOCOL,
        "design_version": protocol.DESIGN_VERSION,
        "stage": "exploratory_12h_screen",
        "formal_data": False,
        "created_without_screen_outcomes": True,
        "formal_v5_3_data": False,
        "shared_outcome_free_protocol_inputs_read_only": True,
        "prior_v5_3_rollout_checkpoint_metric_decision_bytes_reused": False,
        "screen_outputs_may_enter_formal_v5_3": False,
        "screen_outputs_may_enter_formal": False,
        "official_test_used": False,
        "official_test_sealed": True,
        "source_split": "derived_inner_train",
        "source_generation_manifest_sha256": generation_manifest_sha256,
        "generation_manifest_sha256": generation_manifest_sha256,
        "split_manifest_sha256": generation_manifest.get(
            "split_manifest_sha256"
        ),
        "tau2_commit": generation_manifest.get("tau2_commit"),
        "source_files": deepcopy(generation_manifest.get("source_files")),
        "fault_protocol": deepcopy(generation_manifest.get("fault_protocol")),
        "gt_compatibility_filter": deepcopy(
            generation_manifest.get("gt_compatibility_filter")
        ),
        "source_effective_universe_canonical_sha256": (
            protocol.canonical_sha256(effective_universe)
        ),
        "source_pilot_manifest_canonical_sha256": (
            protocol.canonical_sha256(pilot_manifest)
        ),
        "planned_task_ids": list(planned),
        "paired_task_count": protocol.TASKS,
        "task_universe_complete": True,
        "base_seed": protocol.BASE_SEED,
        "trial_seeds": list(protocol.TRIAL_SEEDS),
        "attempts_per_task_per_condition": (
            protocol.ATTEMPTS_PER_TASK_PER_CONDITION
        ),
        "conditions": list(protocol.CONDITIONS),
        "expected_rollouts": protocol.EXPECTED_ROLLOUTS,
        "formal_v5_3_rollout_seed_or_generated_trajectory_bytes_reused": False,
        "formal_v5_3_seed_or_bytes_reused": False,
        "replacement_or_rescue_attempts": False,
        "generation": {
            "teacher": {
                "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                "revision": "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c",
            },
            "user": {
                "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                "revision": "539535859b135b0244c91f3e59816150c8056698",
            },
            "judge": {
                "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                "revision": "539535859b135b0244c91f3e59816150c8056698",
            },
            "teacher_mode": "ground_truth",
            "temperature": 0.2,
            "top_p": 0.95,
            "max_model_len": 32768,
            "max_tokens": 512,
            "max_steps": 60,
            "task_timeout_seconds": 900.0,
            "num_shards": protocol.NUM_GENERATION_SHARDS,
            "num_trials": protocol.ATTEMPTS_PER_TASK_PER_CONDITION,
            "base_seed": protocol.BASE_SEED,
            "trial_seeds": list(protocol.TRIAL_SEEDS),
            "parallel_tool_calls": False,
        },
        "rows": deepcopy(rows),
        "claim_boundary": dict(protocol.CLAIM_BOUNDARY),
    }
    payload["canonical_sha256"] = protocol.canonical_sha256(payload)
    return payload


def validate_screen_manifest(
    screen_manifest: dict[str, Any],
    *,
    generation_manifest_sha256: str,
    generation_manifest: dict[str, Any] | None = None,
    effective_universe: dict[str, Any] | None = None,
    pilot_manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    core = {
        key: value
        for key, value in screen_manifest.items()
        if key != "canonical_sha256"
    }
    rows = screen_manifest.get("rows")
    planned = screen_manifest.get("planned_task_ids")
    if (
        screen_manifest.get("protocol") != SCREEN_MANIFEST_PROTOCOL
        or screen_manifest.get("screen_protocol") != protocol.PROTOCOL
        or screen_manifest.get("design_version") != protocol.DESIGN_VERSION
        or screen_manifest.get("official_test_used") is not False
        or screen_manifest.get("official_test_sealed") is not True
        or screen_manifest.get("formal_v5_3_data") is not False
        or screen_manifest.get("formal_data") is not False
        or screen_manifest.get("shared_outcome_free_protocol_inputs_read_only")
        is not True
        or screen_manifest.get(
            "prior_v5_3_rollout_checkpoint_metric_decision_bytes_reused"
        )
        is not False
        or screen_manifest.get("screen_outputs_may_enter_formal_v5_3")
        is not False
        or screen_manifest.get("screen_outputs_may_enter_formal")
        is not False
        or screen_manifest.get("replacement_or_rescue_attempts")
        is not False
        or screen_manifest.get("generation_manifest_sha256")
        != generation_manifest_sha256
        or screen_manifest.get("source_generation_manifest_sha256")
        != generation_manifest_sha256
        or not isinstance(
            screen_manifest.get(
                "source_effective_universe_canonical_sha256"
            ),
            str,
        )
        or not isinstance(
            screen_manifest.get("source_pilot_manifest_canonical_sha256"),
            str,
        )
        or screen_manifest.get("base_seed") != protocol.BASE_SEED
        or tuple(screen_manifest.get("trial_seeds") or ())
        != protocol.TRIAL_SEEDS
        or screen_manifest.get("attempts_per_task_per_condition")
        != protocol.ATTEMPTS_PER_TASK_PER_CONDITION
        or screen_manifest.get("paired_task_count") != protocol.TASKS
        or screen_manifest.get("expected_rollouts")
        != protocol.EXPECTED_ROLLOUTS
        or not isinstance(rows, list)
        or len(rows) != protocol.TASKS
        or not isinstance(planned, list)
        or tuple(sorted(planned)) != tuple(sorted(protocol.PILOT_TASK_IDS))
        or [f"{row.get('domain')}:{row.get('task_id')}" for row in rows]
        != planned
        or screen_manifest.get("canonical_sha256")
        != protocol.canonical_sha256(core)
    ):
        raise ScreenDataError("screen manifest contract drift")
    if generation_manifest is not None:
        for field in (
            "split_manifest_sha256",
            "tau2_commit",
            "source_files",
            "fault_protocol",
            "gt_compatibility_filter",
        ):
            if screen_manifest.get(field) != generation_manifest.get(field):
                raise ScreenDataError(
                    f"screen source generation provenance drift: {field}"
                )
        generation_by_task = {
            f"{row.get('domain')}:{row.get('task_id')}": row
            for row in generation_manifest.get("rows", [])
            if isinstance(row, dict)
        }
        if any(
            protocol.canonical(row)
            != protocol.canonical(generation_by_task.get(identity))
            for identity, row in zip(planned, rows, strict=True)
        ):
            raise ScreenDataError("screen/source generation row drift")
    if effective_universe is not None and (
        effective_universe.get("protocol")
        != "v5_3_effective_arm_train_task_universe"
        or screen_manifest.get(
            "source_effective_universe_canonical_sha256"
        )
        != protocol.canonical_sha256(effective_universe)
    ):
        raise ScreenDataError("screen effective-universe source hash drift")
    if pilot_manifest is not None and (
        pilot_manifest.get("protocol")
        != "v5_3_train_only_feasibility_pilot"
        or screen_manifest.get("source_pilot_manifest_canonical_sha256")
        != protocol.canonical_sha256(pilot_manifest)
        or pilot_manifest.get("rows") != rows
        or pilot_manifest.get("planned_task_ids") != planned
    ):
        raise ScreenDataError("screen pilot source hash/row drift")
    return list(rows)


def _load_attempts(
    raw_dir: Path,
    *,
    task_ids: set[str],
) -> tuple[
    dict[str, dict[str, dict[int, dict[str, Any]]]],
    list[Path],
]:
    by_condition: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        "clean": defaultdict(dict),
        "error": defaultdict(dict),
    }
    paths: list[Path] = []
    expected_indices = set(
        range(protocol.ATTEMPTS_PER_TASK_PER_CONDITION)
    )
    expected_seed_by_index = {
        index: str(seed) for index, seed in enumerate(protocol.TRIAL_SEEDS)
    }
    for domain in ("retail", "airline"):
        allowed = {
            identity.split(":", 1)[1]
            for identity in task_ids
            if identity.startswith(f"{domain}:")
        }
        for condition in protocol.CONDITIONS:
            for path in v5._raw_paths(raw_dir, domain, condition):
                paths.append(path)
                simulations = load_json(path).get("simulations")
                if not isinstance(simulations, list):
                    raise ScreenDataError(f"{path}: simulations must be a list")
                for simulation in simulations:
                    if not isinstance(simulation, dict):
                        raise ScreenDataError(
                            f"{path}: simulation must be an object"
                        )
                    task_id = str(simulation.get("task_id"))
                    if task_id not in allowed:
                        raise ScreenDataError(
                            f"{path}: unexpected screen task {domain}:{task_id}"
                        )
                    index = v53._attempt_index(simulation)
                    seed = v53._attempt_seed(simulation)
                    if expected_seed_by_index.get(index) != seed:
                        raise ScreenDataError(
                            f"{path}: attempt {index} has unregistered screen "
                            f"seed {seed}"
                        )
                    task_key = f"{domain}:{task_id}"
                    if index in by_condition[condition][task_key]:
                        raise ScreenDataError(
                            f"duplicate {condition} attempt {task_key}:{index}"
                        )
                    by_condition[condition][task_key][index] = simulation
        for task_id in allowed:
            key = f"{domain}:{task_id}"
            for condition in protocol.CONDITIONS:
                observed = set(by_condition[condition].get(key, {}))
                if observed != expected_indices:
                    raise ScreenDataError(
                        f"{key}/{condition}: expected attempts 0..5; "
                        f"observed={sorted(observed)}"
                    )
                observed_seeds = {
                    v53._attempt_seed(row)
                    for row in by_condition[condition][key].values()
                }
                if observed_seeds != set(expected_seed_by_index.values()):
                    raise ScreenDataError(f"{key}/{condition}: seed-set drift")
    if set(by_condition["clean"]) != task_ids:
        raise ScreenDataError("screen clean task coverage drift")
    if set(by_condition["error"]) != task_ids:
        raise ScreenDataError("screen error task coverage drift")
    if not paths:
        raise ScreenDataError("screen raw directory contains no result files")
    return by_condition, sorted(set(paths))


def _common_eligible_attempt_slots(
    clean: list[dict[str, Any]],
    error: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Intersect eligible clean/error candidates on one registered slot."""

    by_condition: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    for condition, rows in (("clean", clean), ("error", error)):
        indexed: dict[tuple[int, str], dict[str, Any]] = {}
        for row in rows:
            identity = row.get("identity")
            if not isinstance(identity, dict):
                raise ScreenDataError(
                    f"{condition} eligible candidate lacks identity"
                )
            try:
                key = (
                    int(identity["attempt_index"]),
                    str(identity["attempt_seed"]),
                )
            except (KeyError, TypeError, ValueError) as error_value:
                raise ScreenDataError(
                    f"{condition} eligible candidate has invalid identity"
                ) from error_value
            if key in indexed:
                raise ScreenDataError(
                    f"duplicate {condition} eligible slot: {key}"
                )
            indexed[key] = row
        by_condition[condition] = indexed

    expected_seed_by_index = {
        index: str(seed) for index, seed in enumerate(protocol.TRIAL_SEEDS)
    }
    common = set(by_condition["clean"]) & set(by_condition["error"])
    for index, seed in common:
        if expected_seed_by_index.get(index) != seed:
            raise ScreenDataError(
                f"common eligible slot has unregistered seed: {(index, seed)}"
            )
    return [
        (by_condition["clean"][key], by_condition["error"][key])
        for key in sorted(common)
    ]


def validate_generation_contracts(
    *,
    raw_dir: Path,
    screen_manifest_path: Path,
    generation_manifest_path: Path,
    dynamic_identity: dict[str, Any],
    expected_source_commit: str,
) -> dict[str, Any]:
    expected_tasks = set(protocol.PILOT_TASK_IDS)
    screen_manifest_payload = load_json(screen_manifest_path)
    source_generation_payload = load_json(generation_manifest_path)
    screen_rows = screen_manifest_payload.get("rows")
    if not isinstance(screen_rows, list):
        raise ScreenDataError("screen generation manifest rows are missing")
    observed_tasks: set[str] = set()
    contract_files: dict[str, str] = {}
    result_files: list[Path] = []
    for shard in range(protocol.NUM_GENERATION_SHARDS):
        path = (
            raw_dir
            / f"run_contract.shard-{shard:03d}-of-"
            f"{protocol.NUM_GENERATION_SHARDS:03d}.json"
        )
        contract = load_json(path)
        task_ids = contract.get("task_ids")
        hashes = contract.get("result_sha256")
        runtime_evidence = contract.get("runtime_evidence")
        expected_runtime_evidence = None
        if isinstance(runtime_evidence, dict) and isinstance(
            runtime_evidence.get("path"), str
        ):
            expected_runtime_evidence = (
                generation.validate_screen_runtime_evidence(
                    Path(runtime_evidence["path"]),
                    expected_source_commit=expected_source_commit,
                    max_model_len=32768,
                )
            )
        expected_teacher = {
            "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "revision": "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c",
            "api_base": "http://127.0.0.1:8011/v1",
            "mode": "ground_truth",
            "tensor_parallel_size": 2,
        }
        expected_user = {
            "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "revision": "539535859b135b0244c91f3e59816150c8056698",
            "api_base": "http://127.0.0.1:8001/v1",
            "tensor_parallel_size": 1,
        }
        judge = contract.get("judge")
        decoding = contract.get("decoding")
        expected_shard_rows = generation.shard_rows(
            screen_rows,
            shard_index=shard,
            num_shards=protocol.NUM_GENERATION_SHARDS,
        )
        expected_shard_tasks = [
            f"{row['domain']}:{row['task_id']}"
            for row in expected_shard_rows
        ]
        if (
            contract.get("protocol") != SCREEN_CONTRACT_PROTOCOL
            or contract.get("subprotocol")
            != protocol.GENERATION_SUBPROTOCOL
            or contract.get("status") != "COMPLETE"
            or contract.get("source_commit") != expected_source_commit
            or contract.get("official_test_used") is not False
            or contract.get("formal_data") is not False
            or contract.get("screen_outputs_may_enter_formal_v5_3")
            is not False
            or contract.get("shard_index") != shard
            or contract.get("num_shards")
            != protocol.NUM_GENERATION_SHARDS
            or contract.get("num_trials")
            != protocol.ATTEMPTS_PER_TASK_PER_CONDITION
            or contract.get("trial_seeds") != list(protocol.TRIAL_SEEDS)
            or contract.get("base_seed") != protocol.BASE_SEED
            or contract.get("screen_manifest_sha256")
            != v5.sha256_file(screen_manifest_path)
            or contract.get("generation_manifest_sha256")
            != v5.sha256_file(generation_manifest_path)
            or contract.get("source_generation_manifest_sha256")
            != v5.sha256_file(generation_manifest_path)
            or contract.get("generation_manifest_protocol")
            != SCREEN_MANIFEST_PROTOCOL
            or contract.get("manifest_sha256")
            != v5.sha256_file(screen_manifest_path)
            or contract.get("tau2_commit")
            != source_generation_payload.get("tau2_commit")
            or contract.get("source_files")
            != source_generation_payload.get("source_files")
            or contract.get("dynamic_audit_identity") != dynamic_identity
            or contract.get("teacher") != expected_teacher
            or contract.get("user") != expected_user
            or not isinstance(judge, dict)
            or {
                key: judge.get(key)
                for key in ("model", "revision", "api_base", "tensor_parallel_size")
            }
            != expected_user
            or judge.get("protocol") != "v5_strict_nl_judge_v1"
            or judge.get("schema_failure") != "fail_closed"
            or not isinstance(decoding, dict)
            or decoding.get("temperature") != 0.2
            or decoding.get("top_p") != 0.95
            or decoding.get("max_tokens") != 512
            or decoding.get("max_steps") != 60
            or decoding.get("task_timeout_seconds") not in {900, 900.0}
            or decoding.get("seed") != protocol.BASE_SEED
            or decoding.get("derived_trial_seeds")
            != list(protocol.TRIAL_SEEDS)
            or decoding.get("max_model_len") != 32768
            or runtime_evidence != expected_runtime_evidence
            or screen_manifest_payload.get("canonical_sha256")
            != protocol.canonical_sha256(
                {
                    key: value
                    for key, value in screen_manifest_payload.items()
                    if key != "canonical_sha256"
                }
            )
            or not isinstance(task_ids, list)
            or not task_ids
            or task_ids != expected_shard_tasks
            or len(task_ids) != len(set(task_ids))
            or observed_tasks & set(task_ids)
            or not isinstance(hashes, dict)
            or not hashes
        ):
            raise ScreenDataError(f"generation contract drift: {path}")
        observed_tasks.update(task_ids)
        expected_result_tasks: dict[str, list[str]] = {}
        for domain in ("retail", "airline"):
            domain_tasks = [
                str(row["task_id"])
                for row in expected_shard_rows
                if row["domain"] == domain
            ]
            if not domain_tasks:
                continue
            for condition in protocol.CONDITIONS:
                expected_result_tasks[
                    generation.output_filename(
                        domain,
                        condition,
                        shard,
                        protocol.NUM_GENERATION_SHARDS,
                    )
                ] = domain_tasks
        if set(hashes) != set(expected_result_tasks):
            raise ScreenDataError(
                f"generation shard result declaration drift: {path}"
            )
        shard_result_files: list[Path] = []
        for name, digest in hashes.items():
            result = raw_dir / str(name)
            if (
                Path(str(name)).name != name
                or not result.is_file()
                or v5.sha256_file(result) != digest
            ):
                raise ScreenDataError(
                    f"generation result hash/path drift: {result}"
                )
            simulations = load_json(result).get("simulations")
            expected_domain = name.split("_", 1)[0]
            expected_condition = next(
                (
                    condition
                    for condition in protocol.CONDITIONS
                    if f"_{condition}." in name
                ),
                None,
            )
            if (
                expected_domain not in {"retail", "airline"}
                or expected_condition is None
                or not isinstance(simulations, list)
            ):
                raise ScreenDataError(
                    f"generation result semantic schema drift: {result}"
                )
            expected_cases = {
                (task_id, index, str(seed))
                for task_id in expected_result_tasks[name]
                for index, seed in enumerate(protocol.TRIAL_SEEDS)
            }
            observed_cases: set[tuple[str, int, str]] = set()
            for simulation in simulations:
                if not isinstance(simulation, dict):
                    raise ScreenDataError(
                        "generation result contains invalid simulation: "
                        f"{result}"
                    )
                try:
                    case = (
                        str(simulation["task_id"]),
                        v53._attempt_index(simulation),
                        v53._attempt_seed(simulation),
                    )
                except (KeyError, RuntimeError) as error:
                    raise ScreenDataError(
                        f"generation result attempt identity drift: {result}"
                    ) from error
                if case in observed_cases:
                    raise ScreenDataError(
                        f"duplicate generation shard case {case}: {result}"
                    )
                observed_cases.add(case)
            if observed_cases != expected_cases:
                raise ScreenDataError(
                    f"generation shard semantic case ledger drift: {result}"
                )
            result_files.append(result.resolve())
            shard_result_files.append(result.resolve())
        try:
            shard_evidence = judge_contract.validate_strict_judge_evidence(
                shard_result_files,
                maximum_content_attempts=2,
            )
        except judge_contract.StrictJudgeEvidenceError as error:
            raise ScreenDataError(str(error)) from error
        if (
            shard_evidence.get("expected_calls", 0) <= 0
            or contract.get("strict_judge_audit_evidence")
            != shard_evidence
        ):
            raise ScreenDataError(
                f"generation contract strict-judge evidence drift: {path}"
            )
        contract_files[str(path.resolve())] = v5.sha256_file(path)
    if observed_tasks != expected_tasks:
        raise ScreenDataError("generation contracts do not partition 24 tasks")
    try:
        evidence = judge_contract.validate_strict_judge_evidence(
            result_files,
            maximum_content_attempts=2,
        )
    except judge_contract.StrictJudgeEvidenceError as error:
        raise ScreenDataError(str(error)) from error
    if (
        evidence.get("status") != "PASS"
        or not isinstance(evidence.get("expected_calls"), int)
        or evidence["expected_calls"] <= 0
        or evidence.get("observed_unique_pass_audits")
        != evidence["expected_calls"]
    ):
        raise ScreenDataError("strict-judge evidence is empty or incomplete")
    return {
        "protocol": f"{protocol.PROTOCOL}:generation_contract_audit_v1",
        "task_union": len(observed_tasks),
        "shards": protocol.NUM_GENERATION_SHARDS,
        "contract_files": contract_files,
        "result_file_sha256": {
            str(path): v5.sha256_file(path) for path in result_files
        },
        "strict_judge_evidence_mapping_sha256": evidence[
            "canonical_mapping_sha256"
        ],
        "official_test_used": False,
    }


def _write_fail_closed(
    *,
    output_dir: Path,
    audit: dict[str, Any],
    attempt_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    v5.write_jsonl(output_dir / "attempt_audit.jsonl", attempt_rows)
    v5.write_json(output_dir / "audit.json", audit)
    v5.write_json(
        output_dir / "hashes.json",
        {
            "attempt_audit.jsonl": v5.sha256_file(
                output_dir / "attempt_audit.jsonl"
            ),
            "audit.json": v5.sha256_file(output_dir / "audit.json"),
        },
    )


def prepare(
    *,
    split_manifest: Path,
    generation_manifest: Path,
    validation_manifest: Path,
    screen_manifest: Path,
    generation_dynamic_audit: Path,
    validation_dynamic_audit: Path,
    raw_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    tokenizer_name: str,
    tokenizer_revision: str,
    domain_contexts: dict[str, dict[str, Any]],
    expected_source_commit: str | None,
    strict_generation_contracts: bool = True,
    strict_dynamic_audits: bool = True,
    schedule_rows: int = SCHEDULE_ROWS,
) -> dict[str, Any]:
    protocol.validate_constants()
    if output_dir.exists():
        raise ScreenDataError(f"output directory must be absent: {output_dir}")
    if v5.sha256_file(split_manifest) != SPLIT_SHA256:
        raise ScreenDataError("Stage-0 split SHA drift")
    split = v5.load_split(split_manifest, strict_counts=True)
    inner_train_ids = {
        f"{domain}:{task_id}"
        for domain in ("retail", "airline")
        for task_id in split["domains"][domain]["inner_train_ids"]
    }
    derived_validation_ids = {
        f"{domain}:{task_id}"
        for domain in ("retail", "airline")
        for task_id in split["domains"][domain]["validation_ids"]
    }
    sealed_test_ids = {
        f"{domain}:{task_id}"
        for domain in ("retail", "airline")
        for task_id in split["domains"][domain]["sealed_test_ids"]
    }
    if (
        not set(protocol.PILOT_TASK_IDS) <= inner_train_ids
        or set(protocol.PILOT_TASK_IDS) & derived_validation_ids
        or set(protocol.PILOT_TASK_IDS) & sealed_test_ids
        or derived_validation_ids & sealed_test_ids
    ):
        raise ScreenDataError(
            "screen task IDs are not a sealed inner-train-only selection"
        )
    generation_payload = load_json(generation_manifest)
    generation_rows = generation_payload.get("rows")
    if (
        generation_payload.get("protocol") != GENERATION_MANIFEST_PROTOCOL
        or not isinstance(generation_rows, list)
        or generation_payload.get("official_test_used") is not False
    ):
        raise ScreenDataError("V5.3 generation manifest drift")
    source_universe_path = (
        screen_manifest.parent / "effective_task_universe.json"
    )
    source_pilot_path = screen_manifest.parent / "source_pilot_manifest.json"
    source_universe = load_json(source_universe_path)
    source_pilot = load_json(source_pilot_path)
    manifest_rows = validate_screen_manifest(
        load_json(screen_manifest),
        generation_manifest_sha256=v5.sha256_file(generation_manifest),
        generation_manifest=generation_payload,
        effective_universe=source_universe,
        pilot_manifest=source_pilot,
    )
    task_ids = set(protocol.PILOT_TASK_IDS)
    manifest_by_task = {
        f"{row.get('domain')}:{row.get('task_id')}": row
        for row in generation_rows
    }
    if not task_ids <= set(manifest_by_task):
        raise ScreenDataError("screen tasks absent from generation manifest")
    screen_rows_by_task = {
        f"{row.get('domain')}:{row.get('task_id')}": row
        for row in manifest_rows
    }
    if set(screen_rows_by_task) != task_ids:
        raise ScreenDataError("screen manifest row universe drift")
    if any(
        protocol.canonical(screen_rows_by_task[task_id])
        != protocol.canonical(manifest_by_task[task_id])
        for task_id in task_ids
    ):
        raise ScreenDataError(
            "screen selected row/fault descriptor differs from generation "
            "manifest"
        )

    validation_payload = v5.load_validation_manifest(
        validation_manifest,
        split=split,
        split_manifest_path=split_manifest,
    )
    if (
        validation_payload.get("paired_task_count") != 21
        or validation_payload.get("official_test_used") is not False
        or validation_payload.get("official_test_sealed") is not True
    ):
        raise ScreenDataError("derived-validation manifest drift")
    dynamic_identities: dict[str, Any] = {}
    if strict_dynamic_audits:
        all_generation_ids = {
            f"{row['domain']}:{row['task_id']}" for row in generation_rows
        }
        dynamic_identities["generation"] = load_complete_dynamic_audit(
            generation_dynamic_audit,
            manifest_path=generation_manifest,
            split_manifest_path=split_manifest,
            expected_source_split="derived_inner_train",
            expected_task_ids=all_generation_ids,
        )
        dynamic_identities["validation"] = load_complete_dynamic_audit(
            validation_dynamic_audit,
            manifest_path=validation_manifest,
            split_manifest_path=split_manifest,
            expected_source_split="derived_validation",
            expected_task_ids={
                f"{row['domain']}:{row['task_id']}"
                for row in validation_payload["rows"]
            },
        )
    else:
        # Unit-test fixtures may omit signed dynamic audits, but the shape is
        # still explicit and can never pass the formal GPU controller.
        dynamic_identities = {
            "generation": {"status": "TEST_ONLY"},
            "validation": {"status": "TEST_ONLY"},
        }

    generation_contract_audit: dict[str, Any] | None = None
    if strict_generation_contracts:
        if expected_source_commit is None:
            raise ScreenDataError("screen contracts require source commit")
        generation_contract_audit = validate_generation_contracts(
            raw_dir=raw_dir,
            screen_manifest_path=screen_manifest,
            generation_manifest_path=generation_manifest,
            dynamic_identity=dynamic_identities["generation"],
            expected_source_commit=expected_source_commit,
        )

    pools, raw_paths = _load_attempts(raw_dir, task_ids=task_ids)
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    attempt_audit_rows: list[dict[str, Any]] = []
    exclusion_reasons = {"clean": Counter(), "error": Counter()}
    task_yield: dict[str, Any] = {}
    for task_key in sorted(task_ids):
        domain, task_id = task_key.split(":", 1)
        fault = screen_rows_by_task[task_key]["error_condition"]
        ranked: dict[str, list[dict[str, Any]]] = {}
        for condition in protocol.CONDITIONS:
            eligible, rows, reasons = v53._rank_and_filter(
                domain=domain,
                task_id=task_id,
                condition=condition,
                attempts=pools[condition][task_key],
                seed=protocol.BASE_SEED,
            )
            eligible = v53._filter_training_compatible_attempts(
                domain=domain,
                task_id=task_id,
                condition=condition,
                eligible=eligible,
                audit_rows=rows,
                reasons=reasons,
                fault=fault,
                context=domain_contexts[domain],
                tokenizer=tokenizer,
                seed=protocol.BASE_SEED,
            )
            ranked[condition] = eligible
            attempt_audit_rows.extend(rows)
            exclusion_reasons[condition].update(reasons)
        common_slots = _common_eligible_attempt_slots(
            ranked["clean"], ranked["error"]
        )
        pairs = []
        for rank, (clean, error) in enumerate(
            common_slots[: protocol.MAX_PAIRS_PER_TASK]
        ):
            if clean["identity"] != error["identity"]:
                raise ScreenDataError(
                    f"{task_key}: clean/error paired-slot identity drift"
                )
            v53._check_injected_fault(
                error["analysis"],
                fault,
                where=f"{task_key}:screen-pair-{rank}",
            )
            pair = v53._materialize_pair(
                domain=domain,
                task_id=task_id,
                pair_rank=rank,
                clean=clean,
                error=error,
                fault=fault,
                context=domain_contexts[domain],
                tokenizer=tokenizer,
                seed=protocol.BASE_SEED,
                partition="screen_train_or_diagnostic_loss",
            )
            for source_name in (
                "perfect_success",
                "failure_raw",
                "repair_masked",
            ):
                pair_contract = pair[source_name]["metadata"][
                    "pair_contract"
                ]
                pair_contract.update(
                    {
                        "protocol": (
                            f"{protocol.PROTOCOL}:same_attempt_slot_pair_v1"
                        ),
                        "same_attempt_index": True,
                        "same_seed": True,
                        "paired_attempt_index": clean["identity"][
                            "attempt_index"
                        ],
                        "paired_attempt_seed": clean["identity"][
                            "attempt_seed"
                        ],
                    }
                )
                pair[source_name]["metadata"].update(
                    {
                        "design_protocol": protocol.DATA_PROTOCOL,
                        "design_version": protocol.DESIGN_VERSION,
                        "formal_v5_3_eligible": False,
                        "official_test_used": False,
                    }
                )
            pair["paired_attempt_index"] = clean["identity"][
                "attempt_index"
            ]
            pair["paired_attempt_seed"] = clean["identity"]["attempt_seed"]
            pairs.append(pair)
        selected_by_task[task_key] = pairs
        task_yield[task_key] = {
            "clean_eligible_attempts": len(ranked["clean"]),
            "error_eligible_attempts": len(ranked["error"]),
            "common_eligible_attempt_slots": len(common_slots),
            "common_eligible_attempt_identities": [
                {
                    "attempt_index": clean["identity"]["attempt_index"],
                    "attempt_seed": clean["identity"]["attempt_seed"],
                }
                for clean, _ in common_slots
            ],
            "selected_pairs": len(pairs),
            "selected_pair_ids": [pair["pair_id"] for pair in pairs],
            "selected_paired_attempt_indices": [
                pair["paired_attempt_index"] for pair in pairs
            ],
            "selected_paired_attempt_seeds": [
                pair["paired_attempt_seed"] for pair in pairs
            ],
        }

    pair_counts = {
        task_id: len(selected_by_task[task_id])
        for task_id in protocol.PILOT_TASK_IDS
    }
    gate = protocol.data_gate(pair_counts)
    base_audit = {
        "protocol": v5.DATA_AUDIT_PROTOCOL,
        "design_protocol": protocol.DATA_PROTOCOL,
        "design_version": protocol.DESIGN_VERSION,
        "screen_protocol": protocol.PROTOCOL,
        "seed": protocol.BASE_SEED,
        "trial_seeds": list(protocol.TRIAL_SEEDS),
        "model_tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "split_manifest_sha256": SPLIT_SHA256,
        "generation_manifest_sha256": v5.sha256_file(generation_manifest),
        "screen_manifest_sha256": v5.sha256_file(screen_manifest),
        "validation_manifest_sha256": v5.sha256_file(validation_manifest),
        "screen_task_ids": list(protocol.PILOT_TASK_IDS),
        "official_test_used": False,
        "official_test_sealed": True,
        "derived_validation_used_for_supervision": False,
        "shared_outcome_free_protocol_inputs_read_only": True,
        "prior_v5_3_rollout_checkpoint_metric_decision_bytes_reused": False,
        "screen_outputs_may_enter_formal_v5_3": False,
        "attempts_per_task_per_condition": (
            protocol.ATTEMPTS_PER_TASK_PER_CONDITION
        ),
        "expected_rollouts": protocol.EXPECTED_ROLLOUTS,
        "maximum_pairs_per_task": protocol.MAX_PAIRS_PER_TASK,
        "raw_files": {
            str(path.resolve()): v5.sha256_file(path) for path in raw_paths
        },
        "generation_contracts": generation_contract_audit,
        "dynamic_audits": dynamic_identities,
        "task_yield": task_yield,
        "data_gate": gate,
        "exclusions": {
            condition: dict(sorted(values.items()))
            for condition, values in exclusion_reasons.items()
        },
        "claim_boundary": dict(protocol.CLAIM_BOUNDARY),
    }
    if not gate["training_authorized"]:
        audit = {
            **base_audit,
            "status": "FAIL_CLOSED",
            "decision": "DO_NOT_TRAIN",
            "arm_files_written": False,
        }
        _write_fail_closed(
            output_dir=output_dir,
            audit=audit,
            attempt_rows=attempt_audit_rows,
        )
        raise ScreenDataError("12-hour screen data gate failed")

    # Both capped ranks remain available to the deterministic matcher.  This
    # is why the 17-pair gate is stronger than a task-coverage-only gate.
    # Fixed 64-step training performs no checkpoint selection, so the screen
    # deliberately has a hash-bound, zero-row/no-eval validation contract
    # instead of consuming rank-1 data or fabricating a validation example.
    train_pairs = {
        task: list(pairs)
        for task, pairs in selected_by_task.items()
        if pairs
    }
    if len(train_pairs) < protocol.MIN_TASKS_WITH_PAIR:
        raise ScreenDataError("screen train support drift after gate")

    scheduled_tasks = v5.task_schedule(
        sorted(train_pairs),
        schedule_rows,
        protocol.BASE_SEED,
    )
    perfect_by_task = {
        task: [pair["perfect_success"] for pair in pairs]
        for task, pairs in train_pairs.items()
    }
    raw_by_task = {
        task: [pair["failure_raw"] for pair in pairs]
        for task, pairs in train_pairs.items()
    }
    repair_by_task = {
        task: [pair["repair_masked"] for pair in pairs]
        for task, pairs in train_pairs.items()
    }
    sources, matching = v53._deterministic_tolerance_aware_joint_match(
        scheduled_tasks,
        {
            "perfect_success": perfect_by_task,
            "failure_raw": raw_by_task,
            "repair_50": {
                task: perfect_by_task[task] + repair_by_task[task]
                for task in train_pairs
            },
            "repair_100": repair_by_task,
        },
        # Batch size is one and each microbatch loss is mean-reduced, so the
        # causal optimization dose is the number of rows.  Token mass remains
        # a cross-arm budget control, not the Repair-50 mixture definition.
        recovery_ratios=None,
        recovery_row_ratios=protocol.ARM_RECOVERY_ROW_RATIOS,
        seed=protocol.BASE_SEED,
        supervised_tolerance=0.01,
        sequence_tolerance=0.02,
        recovery_row_ratio_tolerance=0.0,
        enforce_cross_arm_budget_tolerances=False,
    )
    if sources is None:
        audit = {
            **base_audit,
            "status": "FAIL_CLOSED",
            "decision": "DO_NOT_TRAIN",
            "arm_files_written": False,
            "matching": matching,
        }
        _write_fail_closed(
            output_dir=output_dir,
            audit=audit,
            attempt_rows=attempt_audit_rows,
        )
        raise ScreenDataError("screen matching search was inconclusive")

    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arms_audit: dict[str, Any] = {}
    task_multisets: dict[str, Counter[Any]] = {}
    for arm in protocol.TRAINED_ARMS:
        rows = [
            v5.source_clone(
                source,
                arm=arm,
                slot=index,
                fit_split="screen_train_schedule",
            )
            for index, source in enumerate(sources[arm])
        ]
        arm_rows[arm] = rows
        supervised = sum(
            row["token_contract"]["supervised_tokens"] for row in rows
        )
        nonpadding = sum(
            row["token_contract"]["sequence_tokens"] for row in rows
        )
        recovery = sum(
            row["token_contract"]["supervised_tokens"]
            for row in rows
            if row["metadata"]["source"] == "failure_rich"
        )
        recovery_rows = sum(
            row["metadata"]["source"] == "failure_rich"
            for row in rows
        )
        expected_row_ratio = protocol.ARM_RECOVERY_ROW_RATIOS[arm]
        expected_recovery_rows = int(schedule_rows * expected_row_ratio)
        if recovery_rows != expected_recovery_rows:
            raise ScreenDataError(
                f"{arm}: recovery row count {recovery_rows} != "
                f"{expected_recovery_rows}"
            )
        failed_labels = sum(
            sum(
                row["label_mask"][index]
                for index in row["metadata"][
                    "failed_assistant_message_indices"
                ]
            )
            for row in rows
        )
        if arm == "failure_raw" and failed_labels != len(rows):
            raise ScreenDataError("failure_raw label-mask contract drift")
        if arm != "failure_raw" and failed_labels:
            raise ScreenDataError(f"{arm}: failed action became a label")
        task_multisets[arm] = Counter(
            (row["metadata"]["domain"], row["metadata"]["task_id"])
            for row in rows
        )
        arms_audit[arm] = {
            "rows": len(rows),
            "supervised_tokens": supervised,
            "nonpadding_tokens": nonpadding,
            "recovery_supervised_tokens": recovery,
            "recovery_supervised_token_ratio": recovery / max(1, supervised),
            "recovery_rows": recovery_rows,
            "recovery_row_ratio": recovery_rows / len(rows),
            "expected_recovery_row_ratio": expected_row_ratio,
            "recovery_mixture_basis": (
                "row_mean_microbatch_equal_weight"
            ),
            "controlled_failed_action_labels": failed_labels,
        }
    reference = task_multisets[protocol.TRAINED_ARMS[0]]
    if any(
        task_multisets[arm] != reference
        for arm in protocol.TRAINED_ARMS[1:]
    ):
        raise ScreenDataError("screen arm task-support mismatch")
    supervised_values = [
        arms_audit[arm]["supervised_tokens"]
        for arm in protocol.TRAINED_ARMS
    ]
    sequence_values = [
        arms_audit[arm]["nonpadding_tokens"]
        for arm in protocol.TRAINED_ARMS
    ]
    supervised_range = (
        max(supervised_values) - min(supervised_values)
    ) / min(supervised_values)
    sequence_range = (
        max(sequence_values) - min(sequence_values)
    ) / min(sequence_values)
    budget_targets_met = (
        supervised_range <= 0.01 and sequence_range <= 0.02
    )

    train_sources = {
        row["metadata"]["source_example_id"]
        for rows in arm_rows.values()
        for row in rows
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    v5.write_jsonl(output_dir / "attempt_audit.jsonl", attempt_audit_rows)
    v5.write_jsonl(
        output_dir / "paired_master_pool.jsonl",
        [
            pair
            for task in sorted(selected_by_task)
            for pair in selected_by_task[task]
        ],
    )
    for arm, rows in arm_rows.items():
        path = output_dir / "arms" / arm / "train.jsonl"
        v5.write_jsonl(path, rows)
        arms_audit[arm]["sha256"] = v5.sha256_file(path)
    validation_path = output_dir / "validation_loss.jsonl"
    validation_path.write_text("", encoding="utf-8")
    v5.write_json(output_dir / "validation_manifest.json", validation_payload)
    audit = {
        **base_audit,
        "status": "PASS",
        "decision": "SCREEN_TRAINING_AUTHORIZED",
        "arm_files_written": True,
        "train_schedule_rows_per_arm": schedule_rows,
        "eligible_distinct_tasks": len(train_pairs),
        "paired_master_slots": sum(pair_counts.values()),
        "core_task_id_multiset_equal": True,
        "common_eligible_pair_support": True,
        "train_validation_source_overlap": 0,
        "cross_arm_supervised_token_relative_range": supervised_range,
        "cross_arm_nonpadding_token_relative_range": sequence_range,
        "cross_arm_token_budget_targets_met": budget_targets_met,
        "cross_arm_token_budgets_gate_training": False,
        "arm_mixture_contract": {
            "weight_unit": "training_row",
            "reason": (
                "batch_size_one_mean_reduced_loss_and_equal_gradient_accumulation"
            ),
            "target_recovery_row_ratios": (
                protocol.ARM_RECOVERY_ROW_RATIOS
            ),
            "supervised_token_mass_role": (
                "best_effort_cross_arm_information_budget_control_only"
            ),
            "supervised_token_relative_range_target": 0.01,
            "nonpadding_token_relative_range_target": 0.02,
            "target_miss_blocks_training": False,
        },
        "matching": matching,
        "arms": arms_audit,
        "validation_loss": {
            "rows": 0,
            "distinct_source_examples": 0,
            "source_split": "not_applicable_fixed_step_screen",
            "used_for_checkpoint_selection": False,
            "validation_disabled_reason": "fixed_steps_exploratory",
            "sha256": v5.sha256_file(validation_path),
        },
        "label_guarantees": {
            "final_failure_never_positive_sft": True,
            "raw_labels_only_controlled_failed_action": True,
            "repair_failure_is_context_not_label": True,
            "repair_has_verified_post_error_success": True,
            "causal_prefix_tokenization_verified": True,
            "official_test_and_derived_validation_label_leakage": 0,
        },
    }
    v5.write_json(output_dir / "audit.json", audit)
    hash_paths = [
        output_dir / "attempt_audit.jsonl",
        output_dir / "paired_master_pool.jsonl",
        output_dir / "validation_loss.jsonl",
        output_dir / "validation_manifest.json",
        output_dir / "audit.json",
        *[
            output_dir / "arms" / arm / "train.jsonl"
            for arm in protocol.TRAINED_ARMS
        ],
    ]
    v5.write_json(
        output_dir / "hashes.json",
        {
            str(path.relative_to(output_dir)): v5.sha256_file(path)
            for path in hash_paths
        },
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--generation-dynamic-audit", type=Path, required=True)
    parser.add_argument("--validation-dynamic-audit", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=v5.MODEL)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokenizer_revision != v5.MODEL_REVISION:
        raise ScreenDataError("screen tokenizer revision drift")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    audit = prepare(
        split_manifest=args.split_manifest.resolve(),
        generation_manifest=args.generation_manifest.resolve(),
        validation_manifest=args.validation_manifest.resolve(),
        screen_manifest=args.screen_manifest.resolve(),
        generation_dynamic_audit=args.generation_dynamic_audit.resolve(),
        validation_dynamic_audit=args.validation_dynamic_audit.resolve(),
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        domain_contexts=v5.load_tau2_contexts(args.tau2_root.resolve()),
        expected_source_commit=args.expected_source_commit.lower(),
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
