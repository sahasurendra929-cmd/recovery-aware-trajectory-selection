#!/usr/bin/env python3
"""Hash-bound raw recomputation for the V5.3 12-hour exploratory screen.

No count is accepted from a report or command line.  The script requires the
three core arms, each with exactly 21 derived-validation task IDs in clean and
controlled-error conditions, and recomputes success from tau2's official
composite reward in the contract-bound raw JSON files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

try:
    import run_v5_sft_causal_eval as evaluation
    import summarize_v5_sft_causal as causal_summary
    import v5_3_12h_protocol as protocol
    import v5_judge_audit_contract as judge_contract
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts import run_v5_sft_causal_eval as evaluation
    from scripts import summarize_v5_sft_causal as causal_summary
    from scripts import v5_3_12h_protocol as protocol
    from scripts import v5_judge_audit_contract as judge_contract
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit


SUMMARY_PROTOCOL = f"{protocol.PROTOCOL}:raw_summary_v1"
INFRASTRUCTURE_TERMINATIONS = {
    "infrastructure_error",
    "unexpected_error",
    "context_window_exceeded",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCREEN_GPU_MODEL = "NVIDIA GeForce RTX 5090"
SCREEN_PHASES_REQUIRED_FOR_SUMMARY = (
    "train_entry",
    "train_complete",
    "registry_entry",
    "registry_complete",
    "evaluate_entry",
    "core_evaluation_complete",
    "summarize_entry",
)


class SummaryError(RuntimeError):
    """Raw evidence is absent, duplicated, inconsistent, or unbound."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise SummaryError(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_phase_hardware_receipts(
    core_complete_receipt_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    phase_root = core_complete_receipt_path.parent / "phase_hardware"
    receipt_hashes: dict[str, str] = {}
    static_hashes: dict[str, str] | None = None
    for phase in SCREEN_PHASES_REQUIRED_FOR_SUMMARY:
        path = phase_root / f"{phase}.json"
        payload = load_json(path)
        body = {
            key: value
            for key, value in payload.items()
            if key != "canonical_sha256"
        }
        inventory = payload.get("gpu_inventory")
        bindings = payload.get("bound_artifact_sha256")
        current_static = payload.get("static_protocol_artifact_sha256")
        if (
            payload.get("protocol")
            != f"{protocol.PROTOCOL}:phase_hardware_v1"
            or payload.get("phase") != phase
            or payload.get("expected_gpu_model") != SCREEN_GPU_MODEL
            or payload.get("minimum_gpu_memory_mib") != 30_000
            or payload.get("cuda_version") != "12.8"
            or payload.get("mid_run_gpu_substitution_allowed") is not False
            or payload.get("official_test_used") is not False
            or not isinstance(inventory, list)
            or len(inventory) != 4
            or any(
                not isinstance(row, dict)
                or row.get("index") != index
                or row.get("name") != SCREEN_GPU_MODEL
                or not isinstance(row.get("memory_mib"), int)
                or row["memory_mib"] < 30_000
                for index, row in enumerate(inventory)
            )
            or not isinstance(bindings, dict)
            or not isinstance(current_static, dict)
            or payload.get("canonical_sha256")
            != protocol.canonical_sha256(body)
        ):
            raise SummaryError(f"{phase}: hardware receipt drift")
        for raw_path, digest in bindings.items():
            artifact = Path(raw_path)
            if (
                not artifact.is_file()
                or not isinstance(digest, str)
                or sha256_file(artifact) != digest
            ):
                raise SummaryError(
                    f"{phase}: bound artifact hash drift: {artifact}"
                )
        if static_hashes is None:
            static_hashes = dict(current_static)
        elif current_static != static_hashes:
            raise SummaryError("static protocol hashes changed across phases")
        receipt_hashes[phase] = sha256_file(path)
    if static_hashes is None:
        raise SummaryError("phase hardware receipts are absent")
    return receipt_hashes, static_hashes


def official_reward(simulation: dict[str, Any]) -> float:
    reward_info = simulation.get("reward_info")
    value = reward_info.get("reward") if isinstance(reward_info, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(
            f"simulation {simulation.get('id')} lacks numeric official reward"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SummaryError("official reward is non-finite")
    return numeric


def official_success(simulation: dict[str, Any]) -> int:
    return int(
        math.isclose(
            official_reward(simulation), 1.0, rel_tol=0.0, abs_tol=1e-12
        )
    )


def _call_identity(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    function = value.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = value.get("name")
        arguments = value.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return canonical({"name": name, "arguments": arguments})


def repeated_failed_call(
    simulation: dict[str, Any], expected_fault: dict[str, Any]
) -> int:
    target = canonical(
        {
            "name": expected_fault["tool_name"],
            "arguments": expected_fault["arguments"],
        }
    )
    observed = []
    for message in simulation.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            identity = _call_identity(call)
            if identity is not None:
                observed.append(identity)
    # The injected call itself is expected exactly once.  A second identical
    # call is the diagnostic replay event.
    return int(sum(value == target for value in observed) >= 2)


def _load_simulations(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path).get("simulations")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise SummaryError(f"{path}: malformed simulations")
    return rows


def _condition_from_name(name: str) -> str:
    for condition in protocol.EVALUATION_CONDITIONS:
        if f"_{condition}." in name:
            return condition
    raise SummaryError(f"cannot infer condition from result filename {name}")


def _domain_from_name(name: str) -> str:
    domain = name.split("_", 1)[0]
    if domain not in {"retail", "airline"}:
        raise SummaryError(f"cannot infer domain from result filename {name}")
    return domain


def _expected_contract_paths(arm_dir: Path) -> list[Path]:
    paths = sorted(arm_dir.glob("run_contract.shard-*-of-003.json"))
    if len(paths) != 3:
        raise SummaryError(
            f"{arm_dir}: core arm requires exactly three run contracts"
        )
    return paths


def load_arm_raw(
    *,
    arm: str,
    arm_dir: Path,
    expected_rows: dict[str, dict[str, Any]],
    registry: dict[str, Any],
    registry_sha256: str,
    evaluation_manifest_sha256: str,
    split_manifest_sha256: str,
    dynamic_identity: dict[str, Any],
    expected_registry_profile: str = protocol.REGISTRY_PROFILE,
    expected_user_judge_model_id: str = protocol.USER_JUDGE_MODEL_ID,
    expected_user_judge_revision: str = protocol.USER_JUDGE_REVISION,
    expected_decoding: dict[str, Any] | None = None,
    expected_trial_seed: int = protocol.TRIAL_SEEDS[0],
    required_contract_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if expected_decoding is None:
        expected_decoding = evaluation.V5_3_12H_FROZEN_DECODING
    if required_contract_metadata is None:
        required_contract_metadata = {}
    cases: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    tasks_by_shard: dict[str, list[str]] = {}
    contract_hashes: dict[str, str] = {}
    result_hashes: dict[str, str] = {}
    evidence_hashes: dict[str, str] = {}
    task_union: set[str] = set()
    for shard, path in enumerate(_expected_contract_paths(arm_dir)):
        contract = load_json(path)
        evaluation.validate_contract_core(contract)
        expected_name = f"run_contract.shard-{shard:03d}-of-003.json"
        if path.name != expected_name:
            raise SummaryError(f"{arm}: shard contract naming/order drift")
        expected_user_judge = {
            "model": expected_user_judge_model_id,
            "revision": expected_user_judge_revision,
            "api_base": contract.get("user", {}).get("api_base"),
        }
        low_support_endpoint_drift = False
        if (
            expected_registry_profile
            == evaluation.V5_3_LOW_SUPPORT_PROFILE
        ):
            low_support_endpoint_drift = (
                contract.get("agent")
                != {
                    "model": registry["entries"][arm]["model_id"],
                    "revision": registry["base_model_revision"],
                    "api_base": (
                        f"http://127.0.0.1:{8101 + shard}/v1"
                    ),
                }
                or expected_user_judge["api_base"]
                != "http://127.0.0.1:8001/v1"
            )
        judge = contract.get("judge")
        if (
            contract.get("status") != "COMPLETE"
            or contract.get("arm") != arm
            or contract.get("shard_index") != shard
            or contract.get("num_shards") != 3
            or contract.get("conditions") != ["clean", "error"]
            or contract.get("checkpoint_registry_provenance_profile")
            != expected_registry_profile
            or contract.get("checkpoint_registry_sha256")
            != registry_sha256
            or contract.get("checkpoint_entry")
            != registry["entries"][arm]
            or contract.get("evaluation_manifest_sha256")
            != evaluation_manifest_sha256
            or contract.get("split_manifest_sha256")
            != split_manifest_sha256
            or contract.get("dynamic_audit_identity") != dynamic_identity
            or contract.get("decoding") != expected_decoding
            or contract.get("official_test_used") is not False
            or contract.get("user") != expected_user_judge
            or low_support_endpoint_drift
            or not isinstance(judge, dict)
            or judge.get("model") != expected_user_judge_model_id
            or judge.get("revision") != expected_user_judge_revision
            or judge.get("api_base") != expected_user_judge["api_base"]
            or judge.get("strict_backend")
            != {
                "module": evaluation.STRICT_NL_JUDGE_MODULE,
                "entrypoint": evaluation.STRICT_NL_JUDGE_ENTRYPOINT,
                "mode": "strict_json_schema_fail_closed",
            }
        ):
            raise SummaryError(f"{arm}: run contract metadata drift: {path}")
        if any(
            contract.get(field) != value
            for field, value in required_contract_metadata.items()
        ):
            raise SummaryError(
                f"{arm}: run contract diagnostic metadata drift: {path}"
            )
        task_ids = contract.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or len(task_ids) != len(set(task_ids))
            or task_union & set(task_ids)
        ):
            raise SummaryError(f"{arm}: duplicate/malformed shard task IDs")
        task_union.update(task_ids)
        tasks_by_shard[str(shard)] = list(task_ids)
        expected_files = evaluation._expected_result_task_ids(contract)
        declared = contract.get("result_sha256")
        if not isinstance(declared, dict) or set(declared) != set(
            expected_files
        ):
            raise SummaryError(f"{arm}: result declaration set drift")
        result_paths: list[Path] = []
        recomputed_completion: dict[str, Any] = {}
        for name, expected_task_ids in expected_files.items():
            result_path = arm_dir / name
            if (
                Path(name).name != name
                or not result_path.is_file()
                or sha256_file(result_path) != declared[name]
            ):
                raise SummaryError(f"{arm}: result hash/path drift: {name}")
            result_paths.append(result_path.resolve())
            recomputed_completion[name] = evaluation.audit_result_interface(
                result_path,
                expected_task_ids=expected_task_ids,
                num_trials=1,
                max_tokens=512,
            )
            domain = _domain_from_name(name)
            condition = _condition_from_name(name)
            for simulation in _load_simulations(result_path):
                task_id = str(simulation.get("task_id"))
                pair_id = f"{domain}:{task_id}"
                trial = simulation.get("trial")
                derived_seed = simulation.get("seed")
                if trial != 0 or derived_seed != expected_trial_seed:
                    raise SummaryError(
                        f"{arm}/{condition}/{pair_id}: "
                        "trial or derived seed drift"
                    )
                if pair_id not in expected_rows:
                    raise SummaryError(f"{arm}: unexpected validation task {pair_id}")
                if simulation.get("termination_reason") in (
                    INFRASTRUCTURE_TERMINATIONS
                ):
                    raise SummaryError(
                        f"{arm}/{condition}/{pair_id}: infrastructure termination"
                    )
                expected_fault = expected_rows[pair_id]["error_condition"]
                if condition == "clean":
                    causal_summary.verify_clean_has_no_injection(
                        simulation, expected_fault
                    )
                    verified_behavior: dict[str, Any] = {}
                else:
                    verified_behavior = causal_summary.analyze_error_run(
                        simulation, expected_fault
                    )
                key = (domain, task_id, condition, trial)
                if key in cases:
                    raise SummaryError(f"{arm}: duplicate raw case {key}")
                cases[key] = {
                    "success": official_success(simulation),
                    "reward": official_reward(simulation),
                    "repeated_failed_call": (
                        repeated_failed_call(
                            simulation,
                            expected_rows[pair_id]["error_condition"],
                        )
                        if condition == "error"
                        else 0
                    ),
                    "termination_reason": simulation.get("termination_reason"),
                    "controlled_fault_verified": condition == "error",
                    **verified_behavior,
                }
            result_hashes[name] = declared[name]
        observed_evidence = judge_contract.validate_strict_judge_evidence(
            result_paths, maximum_content_attempts=2
        )
        if (
            observed_evidence.get("expected_calls", 0) <= 0
            or contract.get("strict_judge_audit_evidence")
            != observed_evidence
        ):
            raise SummaryError(f"{arm}: strict-judge evidence drift/empty")
        evidence_hashes[path.name] = observed_evidence[
            "canonical_mapping_sha256"
        ]
        completion = contract.get("completion_audit")
        if (
            not isinstance(completion, dict)
            or completion.get("result_files")
            != dict(sorted(recomputed_completion.items()))
            or completion.get("status") != "PASS"
        ):
            raise SummaryError(f"{arm}: completion audit is not raw-recomputed")
        contract_hashes[path.name] = sha256_file(path)

    expected_task_ids = set(expected_rows)
    if task_union != expected_task_ids:
        raise SummaryError(
            f"{arm}: shard union differs from exact 21 validation tasks"
        )
    expected_cases = {
        (row["domain"], str(row["task_id"]), condition, 0)
        for row in expected_rows.values()
        for condition in protocol.EVALUATION_CONDITIONS
    }
    if set(cases) != expected_cases or len(cases) != 42:
        raise SummaryError(
            f"{arm}: expected exact 21x2 raw cases, found {len(cases)}"
        )
    return {
        "arm": arm,
        "cases": cases,
        "tasks_by_shard": tasks_by_shard,
        "contract_sha256": dict(sorted(contract_hashes.items())),
        "result_sha256": dict(sorted(result_hashes.items())),
        "strict_judge_mapping_sha256": dict(sorted(evidence_hashes.items())),
    }


def arm_metrics(
    loaded: dict[str, Any], expected_rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    cases = loaded["cases"]
    by_condition: dict[str, Any] = {}
    for condition in protocol.EVALUATION_CONDITIONS:
        rows = [
            (key, value)
            for key, value in cases.items()
            if key[2] == condition
        ]
        successes = sum(value["success"] for _, value in rows)
        domain = {
            name: {
                "tasks": sum(key[0] == name for key, _ in rows),
                "successes": sum(
                    value["success"]
                    for key, value in rows
                    if key[0] == name
                ),
            }
            for name in ("retail", "airline")
        }
        for value in domain.values():
            value["success_rate"] = value["successes"] / value["tasks"]
        by_condition[condition] = {
            "tasks": 21,
            "successes": successes,
            "success_rate": successes / 21,
            "domain": domain,
            "repeated_failed_call_count": (
                sum(value["repeated_failed_call"] for _, value in rows)
                if condition == "error"
                else 0
            ),
            "repeated_failed_call_rate": (
                sum(value["repeated_failed_call"] for _, value in rows) / 21
                if condition == "error"
                else 0.0
            ),
        }
    return {
        "conditions": by_condition,
        "contract_sha256": loaded["contract_sha256"],
        "result_sha256": loaded["result_sha256"],
        "strict_judge_mapping_sha256": loaded[
            "strict_judge_mapping_sha256"
        ],
    }


def _binomial_tail(n: int, start: int) -> float:
    return sum(math.comb(n, value) for value in range(start, n + 1)) / (
        2**n
    )


def mcnemar_exact(b: int, c: int) -> dict[str, Any]:
    discordant = b + c
    if discordant == 0:
        one_sided = two_sided = 1.0
    else:
        one_sided = _binomial_tail(discordant, b)
        two_sided = min(
            1.0,
            2.0
            * sum(
                math.comb(discordant, value)
                for value in range(0, min(b, c) + 1)
            )
            / (2**discordant),
        )
    return {
        "discordant_pairs": discordant,
        "exact_one_sided_treatment_greater_p": one_sided,
        "exact_two_sided_p": two_sided,
    }


def _beta_fraction(a: float, b: float, x: float) -> float:
    maximum = 300
    epsilon = 3e-14
    floor = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    h = d
    for m in range(1, maximum + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    raise SummaryError("incomplete beta fraction did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _beta_fraction(a, b, x) / a
    return 1 - front * _beta_fraction(b, a, 1 - x) / b


def beta_quantile(probability: float, a: float, b: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if regularized_beta(middle, a, b) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def clopper_pearson(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    alpha = 1 - protocol.CONFIDENCE_LEVEL
    lower = (
        0.0
        if successes == 0
        else beta_quantile(alpha / 2, successes, trials - successes + 1)
    )
    upper = (
        1.0
        if successes == trials
        else beta_quantile(
            1 - alpha / 2, successes + 1, trials - successes
        )
    )
    return [lower, upper]


def _comparison_seed(treatment: str, control: str) -> int:
    digest = hashlib.sha256(
        f"{protocol.TASK_BOOTSTRAP_SEED}:{treatment}:{control}".encode()
    ).hexdigest()
    return int(digest[:16], 16)


def stratified_bootstrap(
    rows: list[dict[str, Any]], *, treatment: str, control: str
) -> dict[str, list[float]]:
    groups = {
        domain: [row for row in rows if row["domain"] == domain]
        for domain in ("retail", "airline")
    }
    if {name: len(value) for name, value in groups.items()} != {
        "retail": 15,
        "airline": 6,
    }:
        raise SummaryError("bootstrap strata must be exactly retail=15/airline=6")
    rng = random.Random(_comparison_seed(treatment, control))
    draws = {"clean": [], "error": [], "difference_in_differences": []}
    for _ in range(protocol.TASK_BOOTSTRAP_REPLICATES):
        sample = [
            group[rng.randrange(len(group))]
            for group in groups.values()
            for _ in range(len(group))
        ]
        clean = sum(row["clean_delta"] for row in sample) / 21
        error = sum(row["error_delta"] for row in sample) / 21
        draws["clean"].append(clean)
        draws["error"].append(error)
        draws["difference_in_differences"].append(error - clean)
    result = {}
    for key, values in draws.items():
        values.sort()
        low = values[math.floor(0.025 * (len(values) - 1))]
        high = values[math.ceil(0.975 * (len(values) - 1))]
        result[key] = [low, high]
    return result


def compare(
    treatment: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    treatment_cases = treatment["cases"]
    control_cases = control["cases"]
    if set(treatment_cases) != set(control_cases):
        raise SummaryError("paired comparison raw case set drift")
    task_ids = sorted(
        {
            f"{domain}:{task_id}"
            for domain, task_id, _condition, trial in treatment_cases
            if trial == 0
        }
    )
    rows = []
    condition_tables: dict[str, Any] = {}
    for identity in task_ids:
        domain, task_id = identity.split(":", 1)
        row: dict[str, Any] = {
            "pair_id": identity,
            "domain": domain,
            "task_id": task_id,
        }
        for condition in protocol.EVALUATION_CONDITIONS:
            key = (domain, task_id, condition, 0)
            tr = treatment_cases[key]["success"]
            co = control_cases[key]["success"]
            row[f"{condition}_treatment_success"] = tr
            row[f"{condition}_control_success"] = co
            row[f"{condition}_delta"] = tr - co
        rows.append(row)
    for condition in protocol.EVALUATION_CONDITIONS:
        pairs = [
            (
                row[f"{condition}_treatment_success"],
                row[f"{condition}_control_success"],
            )
            for row in rows
        ]
        n11 = sum(tr == 1 and co == 1 for tr, co in pairs)
        n10 = sum(tr == 1 and co == 0 for tr, co in pairs)
        n01 = sum(tr == 0 and co == 1 for tr, co in pairs)
        n00 = sum(tr == 0 and co == 0 for tr, co in pairs)
        discordant = n10 + n01
        exact = mcnemar_exact(n10, n01)
        if condition == "clean":
            exact["interpretation"] = (
                "two-sided retention diagnostic only; not a noninferiority test"
            )
            exact["exact_one_sided_treatment_greater_p"] = None
        else:
            exact["interpretation"] = (
                "exact paired error-condition test; exploratory only"
            )
        condition_tables[condition] = {
            "n11": n11,
            "n10_treatment_win_b": n10,
            "n01_control_win_c": n01,
            "n00": n00,
            "paired_delta": (n10 - n01) / 21,
            "mcnemar_binomial": exact,
            "discordant_treatment_win_probability": {
                "estimate": n10 / discordant if discordant else None,
                "clopper_pearson_95": clopper_pearson(n10, discordant),
            },
        }
    bootstrap = stratified_bootstrap(
        rows, treatment=treatment["arm"], control=control["arm"]
    )
    error_delta = condition_tables["error"]["paired_delta"]
    clean_delta = condition_tables["clean"]["paired_delta"]
    return {
        "treatment": treatment["arm"],
        "control": control["arm"],
        "independent_unit": "task_id",
        "task_count": 21,
        "conditions": condition_tables,
        "difference_in_differences": error_delta - clean_delta,
        "stratified_task_paired_bootstrap_95": bootstrap,
        "bootstrap": {
            "seed": _comparison_seed(treatment["arm"], control["arm"]),
            "replicates": protocol.TASK_BOOTSTRAP_REPLICATES,
            "strata": {"retail": 15, "airline": 6},
            "overall_tasks_equally_weighted": True,
        },
        "task_paired_rows": rows,
    }


def summarize(
    *,
    split_manifest_path: Path,
    evaluation_manifest_path: Path,
    dynamic_audit_path: Path,
    checkpoint_registry_path: Path,
    deadline_receipt_path: Path,
    core_complete_receipt_path: Path,
    extension_admission_path: Path,
    results_root: Path,
) -> dict[str, Any]:
    deadline_receipt = read_json(deadline_receipt_path)
    protocol.validate_deadline_receipt(deadline_receipt)
    phase_receipt_hashes, static_protocol_hashes = (
        validate_phase_hardware_receipts(core_complete_receipt_path)
    )
    split = evaluation.load_split_manifest(split_manifest_path)
    manifest = evaluation.load_manifest(
        evaluation_manifest_path,
        split_manifest=split,
        split_manifest_sha256=sha256_file(split_manifest_path),
    )
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 21:
        raise SummaryError("evaluation manifest must contain exactly 21 tasks")
    expected_rows = {
        f"{row['domain']}:{row['task_id']}": row for row in rows
    }
    expected_ids = set(expected_rows)
    dynamic = load_complete_dynamic_audit(
        dynamic_audit_path,
        manifest_path=evaluation_manifest_path,
        split_manifest_path=split_manifest_path,
        expected_source_split="derived_validation",
        expected_task_ids=expected_ids,
    )
    registry = evaluation.load_checkpoint_registry(
        checkpoint_registry_path,
        expected_profile=protocol.REGISTRY_PROFILE,
    )
    if registry.get("screen_evaluator") != evaluation.V5_3_12H_USER_JUDGE:
        raise SummaryError("registry does not pin the 14B screen evaluator")
    if (
        registry["training_data_provenance"]["dynamic_audits"]["validation"]
        != dynamic
    ):
        raise SummaryError("registry/dynamic-validation audit drift")
    registry_sha = sha256_file(checkpoint_registry_path)
    manifest_sha = sha256_file(evaluation_manifest_path)
    split_sha = sha256_file(split_manifest_path)
    loaded = {
        arm: load_arm_raw(
            arm=arm,
            arm_dir=results_root / "evaluation" / arm,
            expected_rows=expected_rows,
            registry=registry,
            registry_sha256=registry_sha,
            evaluation_manifest_sha256=manifest_sha,
            split_manifest_sha256=split_sha,
            dynamic_identity=dynamic,
        )
        for arm in protocol.CORE_EVAL_ARMS
    }
    core_receipt = read_json(core_complete_receipt_path)
    core_body = {
        key: value
        for key, value in core_receipt.items()
        if key != "canonical_sha256"
    }
    receipt_evidence = core_receipt.get("evaluation_evidence")
    recomputed_evidence = {
        arm: {
            str(shard): {
                "contract_sha256": loaded[arm]["contract_sha256"][
                    f"run_contract.shard-{shard:03d}-of-003.json"
                ],
                "result_sha256": {
                    name: digest
                    for name, digest in loaded[arm][
                        "result_sha256"
                    ].items()
                    if f"shard-{shard:03d}-of-003" in name
                },
                "strict_judge_mapping_sha256": loaded[arm][
                    "strict_judge_mapping_sha256"
                ][f"run_contract.shard-{shard:03d}-of-003.json"],
                "task_ids": loaded[arm]["tasks_by_shard"][str(shard)],
                "case_count": 2
                * len(loaded[arm]["tasks_by_shard"][str(shard)]),
            }
            for shard in range(3)
        }
        for arm in protocol.CORE_EVAL_ARMS
    }
    if (
        core_receipt.get("protocol")
        != f"{protocol.PROTOCOL}:core_complete_v1"
        or core_receipt.get("core_arms") != list(protocol.CORE_EVAL_ARMS)
        or core_receipt.get("evaluation_shards") != 3
        or core_receipt.get("exact_raw_case_count")
        != protocol.CORE_ROLLOUTS
        or core_receipt.get("expected_raw_case_count")
        != protocol.CORE_ROLLOUTS
        or core_receipt.get("checkpoint_registry_sha256") != registry_sha
        or core_receipt.get("evaluation_manifest_sha256") != manifest_sha
        or core_receipt.get("deadline_receipt_sha256")
        != sha256_file(deadline_receipt_path)
        or core_receipt.get("static_protocol_artifact_sha256")
        != static_protocol_hashes
        or core_receipt.get("phase_hardware_receipt_sha256")
        != {
            phase: phase_receipt_hashes[phase]
            for phase in (
                "train_entry",
                "train_complete",
                "registry_entry",
                "registry_complete",
                "evaluate_entry",
            )
        }
        or core_receipt.get("metric_values_read") is not False
        or core_receipt.get("official_test_used") is not False
        or receipt_evidence != recomputed_evidence
        or core_receipt.get("canonical_sha256")
        != protocol.canonical_sha256(core_body)
    ):
        raise SummaryError("core completion receipt/raw binding drift")
    schedule = canonical(loaded["base_model"]["tasks_by_shard"])
    if any(
        canonical(loaded[arm]["tasks_by_shard"]) != schedule
        for arm in protocol.CORE_EVAL_ARMS
    ):
        raise SummaryError("core arm shard schedules differ")
    arm_summaries = {
        arm: arm_metrics(loaded[arm], expected_rows)
        for arm in protocol.CORE_EVAL_ARMS
    }
    comparisons = {
        "repair_50_minus_perfect_success": compare(
            loaded["repair_50"], loaded["perfect_success"]
        ),
        "repair_50_minus_base_model": compare(
            loaded["repair_50"], loaded["base_model"]
        ),
        "perfect_success_minus_base_model": compare(
            loaded["perfect_success"], loaded["base_model"]
        ),
    }
    directional = protocol.directional_gate(
        perfect_error_successes=arm_summaries["perfect_success"][
            "conditions"
        ]["error"]["successes"],
        repair_error_successes=arm_summaries["repair_50"]["conditions"][
            "error"
        ]["successes"],
        perfect_clean_successes=arm_summaries["perfect_success"][
            "conditions"
        ]["clean"]["successes"],
        repair_clean_successes=arm_summaries["repair_50"]["conditions"][
            "clean"
        ]["successes"],
    )
    r50_base = comparisons["repair_50_minus_base_model"]["conditions"]
    absolute_observed = (
        r50_base["error"]["paired_delta"] > 0
        and r50_base["clean"]["paired_delta"] >= -(1 / 21)
    )

    admission = read_json(extension_admission_path)
    expected_admission = protocol.extension_admission(
        deadline_receipt,
        core_completed_at=datetime.fromisoformat(
            core_receipt["completed_at_utc"]
        ),
        core_complete=True,
    )
    expected_admission["core_complete_receipt_sha256"] = sha256_file(
        core_complete_receipt_path
    )
    expected_admission["deadline_receipt_sha256"] = sha256_file(
        deadline_receipt_path
    )
    expected_admission["canonical_sha256"] = protocol.canonical_sha256(
        expected_admission
    )
    if admission != expected_admission:
        raise SummaryError("extension admission receipt drift")
    extension: dict[str, Any] = {
        "authorized": admission.get("extension_authorized"),
        "admission_receipt_sha256": sha256_file(extension_admission_path),
        "result_values_used_for_admission": False,
    }
    if admission.get("extension_authorized") is True:
        extension_loaded: dict[str, Any] = {}
        incomplete = []
        for arm in protocol.EXTENSION_EVAL_ARMS:
            try:
                extension_loaded[arm] = load_arm_raw(
                    arm=arm,
                    arm_dir=results_root / "evaluation" / arm,
                    expected_rows=expected_rows,
                    registry=registry,
                    registry_sha256=registry_sha,
                    evaluation_manifest_sha256=manifest_sha,
                    split_manifest_sha256=split_sha,
                    dynamic_identity=dynamic,
                )
            except (OSError, RuntimeError, SummaryError) as error:
                incomplete.append({"arm": arm, "error": str(error)})
        if incomplete:
            extension.update(
                {"status": "AUTHORIZED_BUT_INCOMPLETE", "incomplete": incomplete}
            )
        else:
            extension.update(
                {
                    "status": "COMPLETE",
                    "arms": {
                        arm: arm_metrics(extension_loaded[arm], expected_rows)
                        for arm in protocol.EXTENSION_EVAL_ARMS
                    },
                    "comparisons": {
                        "failure_raw_minus_perfect_success": compare(
                            extension_loaded["failure_raw"],
                            loaded["perfect_success"],
                        ),
                        "repair_100_minus_perfect_success": compare(
                            extension_loaded["repair_100"],
                            loaded["perfect_success"],
                        ),
                    },
                }
            )
    else:
        extension["status"] = "NOT_AUTHORIZED_BY_HOUR_9_RULE"

    provenance = {
        "split_manifest_sha256": split_sha,
        "evaluation_manifest_sha256": manifest_sha,
        "dynamic_audit_sha256": dynamic["sha256"],
        "checkpoint_registry_sha256": registry_sha,
        "deadline_receipt_sha256": sha256_file(deadline_receipt_path),
        "core_complete_receipt_sha256": sha256_file(
            core_complete_receipt_path
        ),
        "phase_hardware_receipt_sha256": phase_receipt_hashes,
        "static_protocol_artifact_sha256": static_protocol_hashes,
        "core_contract_sha256": {
            arm: loaded[arm]["contract_sha256"]
            for arm in protocol.CORE_EVAL_ARMS
        },
        "core_result_sha256": {
            arm: loaded[arm]["result_sha256"]
            for arm in protocol.CORE_EVAL_ARMS
        },
        "raw_recomputation": True,
        "hand_entered_counts_accepted": False,
    }
    provenance["canonical_input_binding_sha256"] = protocol.canonical_sha256(
        provenance
    )
    return {
        "protocol": SUMMARY_PROTOCOL,
        "status": "CORE_COMPLETE",
        "primary_metric": "tau2_official_composite_end_to_end_task_success",
        "independent_unit": "task_id",
        "core_raw_cases": 126,
        "official_test": {
            "status": "SEALED",
            "used": False,
        },
        "arms": arm_summaries,
        "paired_comparisons": comparisons,
        "directional_gate": directional,
        "claim_classification": {
            "relative_directional_positive_among_sft_arms": (
                directional["status"] == "DIRECTIONAL_POSITIVE_SCREEN"
            ),
            "absolute_improvement_over_base_observed_secondary": (
                absolute_observed
            ),
            "absolute_improvement_is_primary_gate": False,
            "gate_is_statistical_significance": False,
            "allowed_language": (
                "Repair-50 is directionally better than Perfect-success in "
                "this one-seed derived-validation screen"
                if directional["status"] == "DIRECTIONAL_POSITIVE_SCREEN"
                else "the preregistered exploratory direction was not met"
            ),
            "recovery_training_improves_agent_claim_allowed": False,
        },
        "extension": extension,
        "statistics": {
            "confidence_level": protocol.CONFIDENCE_LEVEL,
            "bootstrap_replicates": protocol.TASK_BOOTSTRAP_REPLICATES,
            "bootstrap_base_seed": protocol.TASK_BOOTSTRAP_SEED,
            "mcnemar_exact": True,
            "discordant_clopper_pearson": True,
            "replay_metric_role": "diagnostic_only",
        },
        "provenance": provenance,
        "claim_boundary": dict(protocol.CLAIM_BOUNDARY),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--dynamic-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--deadline-receipt", type=Path, required=True)
    parser.add_argument("--core-complete-receipt", type=Path, required=True)
    parser.add_argument("--extension-admission", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = summarize(
            split_manifest_path=args.split_manifest.resolve(),
            evaluation_manifest_path=args.evaluation_manifest.resolve(),
            dynamic_audit_path=args.dynamic_audit.resolve(),
            checkpoint_registry_path=args.checkpoint_registry.resolve(),
            deadline_receipt_path=args.deadline_receipt.resolve(),
            core_complete_receipt_path=(
                args.core_complete_receipt.resolve()
            ),
            extension_admission_path=args.extension_admission.resolve(),
            results_root=args.results_root.resolve(),
        )
    except Exception as error:
        incomplete = {
            "protocol": SUMMARY_PROTOCOL,
            "status": "INCOMPLETE_NO_CLAIM",
            "error_type": type(error).__name__,
            "error": str(error),
            "partial_metrics_reported": False,
            "official_test_used": False,
            "claim_boundary": dict(protocol.CLAIM_BOUNDARY),
        }
        # Never destroy an existing complete immutable summary because a
        # later diagnostic invocation supplied incomplete or tampered inputs.
        if not args.output.resolve().exists():
            atomic_write(args.output.resolve(), incomplete)
        print(json.dumps(incomplete, indent=2), file=sys.stderr)
        raise SystemExit(20) from error
    output = args.output.resolve()
    if output.exists():
        existing = load_json(output)
        if existing != summary:
            raise SummaryError("existing immutable summary differs from recomputation")
    else:
        atomic_write(output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
