#!/usr/bin/env python3
"""Prepare the leakage-safe V5 Stage-1 controlled SFT mechanism screen.

The builder consumes paired clean/error τ² generation results from the frozen
inner-train split.  It emits four deterministic 512-row training schedules:

* ``perfect_success``: flawless successful trajectories;
* ``failure_raw``: the same recovery trajectories, deliberately supervising
  the one controlled injected failure;
* ``repair_50``: an equal supervised-token mixture of flawless and
  repair-only examples;
* ``repair_100``: recovery trajectories with the failure masked.

``repair_25`` and ``repair_75`` are optional.  Every assistant tool call is
linked to its adjacent tool result before it can become a label.  Failed final
trajectories never become SFT examples.  Tool-call IDs are rewritten to
condition-neutral sequential IDs.  The official test split is never exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit


SEED = 20260722
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TEACHER_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
TEACHER_REVISION = "539535859b135b0244c91f3e59816150c8056698"
USER_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
USER_JUDGE_REVISION = "b25037543e9394b818fdfca67ab2a00ecc7dd641"
SCHEDULE_ROWS = 512
CORE_ARMS = ("perfect_success", "failure_raw", "repair_50", "repair_100")
OPTIONAL_ARMS = ("repair_25", "repair_75")
ARM_RATIOS = {
    "perfect_success": 0.0,
    "failure_raw": 1.0,
    "repair_25": 0.25,
    "repair_50": 0.50,
    "repair_75": 0.75,
    "repair_100": 1.0,
}
SYSTEM_INSTRUCTION = """\
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""
FAULT_PROTOCOL = "v5_multifamily_readonly_faults_v1"
PUBLISHED_STAGE0_SPLIT_SHA256 = (
    "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
)
RAW_RESULT_RE = re.compile(
    r"^(retail|airline)_(clean|error)"
    r"(?:\.shard-(?P<shard>\d{3})-of-(?P<total>\d{3}))?\.json$"
)
GENERATION_RESULT_RE = re.compile(
    r"^(retail|airline)_(clean|error)"
    r"\.shard-(?P<shard>\d{3})-of-(?P<total>\d{3})\.json$"
)


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_clean_tracked_source() -> None:
    root = Path(__file__).resolve().parents[1]
    for command, label in (
        (["git", "diff", "--quiet", "--"], "unstaged"),
        (["git", "diff", "--cached", "--quiet", "--"], "staged"),
    ):
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"repository contains {label} tracked source drift")


def require_clean_tracked_checkout(path: Path, *, label: str) -> None:
    for command, drift_kind in (
        (["git", "-C", str(path), "diff", "--quiet", "--"], "unstaged"),
        (
            ["git", "-C", str(path), "diff", "--cached", "--quiet", "--"],
            "staged",
        ),
    ):
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{label} contains {drift_kind} tracked source drift"
            )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical(row) + "\n" for row in rows), encoding="utf-8"
    )


def _function_call(call: Any, where: str) -> tuple[str, dict[str, Any], str]:
    if not isinstance(call, dict):
        raise RuntimeError(f"{where}: tool call must be an object")
    function = call.get("function")
    if function is not None:
        if not isinstance(function, dict):
            raise RuntimeError(f"{where}: malformed function call")
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{where}: invalid JSON arguments") from error
    call_id = call.get("id")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(arguments, dict)
        or not isinstance(call_id, str)
        or not call_id
    ):
        raise RuntimeError(f"{where}: call requires id/name/object arguments")
    return name, arguments, call_id


def _tool_link_id(message: dict[str, Any]) -> str | None:
    value = message.get("tool_call_id", message.get("id"))
    return value if isinstance(value, str) and value else None


def _is_injected(message: dict[str, Any]) -> bool:
    raw = message.get("raw_data") or {}
    return bool(
        raw.get("v5_stage1_injected_fault")
        or raw.get("v5_stage0_injected_fault")
    )


def _run_key(domain: str, simulation: dict[str, Any]) -> tuple[str, str, str]:
    task_id = str(simulation.get("task_id"))
    trial = simulation.get("trial")
    if trial is None:
        trial = simulation.get("seed")
    if task_id in ("", "None") or trial is None:
        raise RuntimeError("Every simulation requires task_id and trial (or seed)")
    return domain, task_id, str(trial)


def _reward_is_one(simulation: dict[str, Any]) -> bool:
    value = (simulation.get("reward_info") or {}).get("reward")
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 1.0


def validate_fault_protocol(
    payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, str]]]:
    protocol = payload.get("fault_protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("protocol") != FAULT_PROTOCOL
    ):
        raise RuntimeError(f"manifest lacks frozen fault protocol {FAULT_PROTOCOL}")
    guarantees = protocol.get("guarantees")
    required_guarantees = {
        "tool_type": "READ",
        "expected_tool_error": True,
        "expected_state_mutation": False,
        "per_task_unique_invalid_parameter": True,
        "database_absence_checked_at_pinned_tau2_commit": True,
    }
    if not isinstance(guarantees, dict) or any(
        guarantees.get(key) != value
        for key, value in required_guarantees.items()
    ):
        raise RuntimeError("fault protocol safety guarantees drift")
    raw_families = protocol.get("families")
    if not isinstance(raw_families, dict) or set(raw_families) != {
        "retail",
        "airline",
    }:
        raise RuntimeError("fault protocol domains drift")
    catalog: dict[str, dict[str, dict[str, str]]] = {}
    for domain in ("retail", "airline"):
        rows = raw_families.get(domain)
        if not isinstance(rows, list) or len(rows) < 2:
            raise RuntimeError(f"{domain}: fewer than two fault families")
        catalog[domain] = {}
        tools: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f"{domain}: malformed fault family")
            family = row.get("fault_family")
            tool_name = row.get("tool_name")
            invalid_key = row.get("invalid_argument_key")
            if not all(
                isinstance(value, str) and value
                for value in (family, tool_name, invalid_key)
            ):
                raise RuntimeError(f"{domain}: incomplete fault family")
            if family in catalog[domain]:
                raise RuntimeError(f"{domain}: duplicate fault family {family}")
            catalog[domain][family] = {
                "tool_name": tool_name,
                "invalid_argument_key": invalid_key,
            }
            tools.add(tool_name)
        if len(tools) < 2:
            raise RuntimeError(f"{domain}: fault catalog has fewer than two tools")
    return catalog


def validate_fault_descriptor(
    fault: Any,
    *,
    domain: str,
    where: str,
    catalog: dict[str, dict[str, dict[str, str]]],
) -> str:
    if not isinstance(fault, dict):
        raise RuntimeError(f"{where}: fault descriptor must be an object")
    family = fault.get("fault_family")
    specification = catalog.get(domain, {}).get(family)
    if specification is None:
        raise RuntimeError(f"{where}: unsupported fault family {family!r}")
    required = {
        "tool_name": specification["tool_name"],
        "tool_type": "READ",
        "invalid_argument_key": specification["invalid_argument_key"],
        "expected_tool_error": True,
        "expected_state_mutation": False,
    }
    for field, value in required.items():
        if fault.get(field) != value:
            raise RuntimeError(f"{where}: fault field {field} drift")
    arguments = fault.get("arguments")
    invalid_key = specification["invalid_argument_key"]
    if not isinstance(arguments, dict) or not isinstance(
        arguments.get(invalid_key), str
    ):
        raise RuntimeError(f"{where}: fault invalid parameter missing")
    if type(fault.get("on_reference_path")) is not bool:
        raise RuntimeError(f"{where}: on_reference_path must be boolean")
    relevance = fault.get("fault_relevance")
    if relevance not in {
        "reference_path_or_operation_aligned",
        "domain_plausible_fallback",
    }:
        raise RuntimeError(f"{where}: invalid fault_relevance")
    if fault["on_reference_path"] and relevance != (
        "reference_path_or_operation_aligned"
    ):
        raise RuntimeError(f"{where}: reference-path fault cannot be fallback")
    return canonical({invalid_key: arguments[invalid_key]})


def validate_multifault_manifest(
    payload: dict[str, Any],
    *,
    expected_protocol: str,
    expected_source_split: str,
    expected_task_ids: set[str],
    split_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    if payload.get("protocol") != expected_protocol:
        raise RuntimeError(
            f"multi-fault manifest protocol drift: {payload.get('protocol')!r}"
        )
    if payload.get("source_split") != expected_source_split:
        raise RuntimeError("multi-fault manifest source split drift")
    if payload.get("split_manifest_sha256") != split_manifest_sha256:
        raise RuntimeError("multi-fault manifest split SHA drift")
    if payload.get("official_test_used") is not False:
        raise RuntimeError("multi-fault manifest opens official test")
    if payload.get("official_test_sealed") is not True:
        raise RuntimeError("multi-fault manifest lacks official-test seal")
    if payload.get("scientific_claim_allowed") is not False:
        raise RuntimeError("multi-fault validation is not a final scientific claim")
    catalog = validate_fault_protocol(payload)
    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("paired_task_count") != len(rows):
        raise RuntimeError("multi-fault manifest row count drift")
    by_pair: dict[str, dict[str, Any]] = {}
    invalid_identities: set[str] = set()
    tool_call_ids: set[str] = set()
    families_by_domain: dict[str, set[str]] = {
        "retail": set(),
        "airline": set(),
    }
    tools_by_domain: dict[str, set[str]] = {
        "retail": set(),
        "airline": set(),
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"multi-fault row {index} is not an object")
        domain = row.get("domain")
        task_id = str(row.get("task_id"))
        pair = f"{domain}:{task_id}"
        if domain not in {"retail", "airline"} or row.get("pair_id") != pair:
            raise RuntimeError(f"multi-fault row {index} identity drift")
        if row.get("source_split") != expected_source_split:
            raise RuntimeError(f"{pair}: source split drift")
        if pair in by_pair:
            raise RuntimeError(f"duplicate multi-fault task {pair}")
        clean = row.get("clean_condition") or {}
        error = row.get("error_condition") or {}
        if clean.get("inject_error") is not False or error.get("inject_error") is not True:
            raise RuntimeError(f"{pair}: invalid paired conditions")
        identity = validate_fault_descriptor(
            error,
            domain=domain,
            where=pair,
            catalog=catalog,
        )
        if identity in invalid_identities:
            raise RuntimeError("multi-fault invalid parameters are not task-unique")
        invalid_identities.add(identity)
        call_id = error.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id or call_id in tool_call_ids:
            raise RuntimeError(f"{pair}: invalid/duplicate tool_call_id")
        tool_call_ids.add(call_id)
        families_by_domain[domain].add(error["fault_family"])
        tools_by_domain[domain].add(error["tool_name"])
        by_pair[pair] = row
    if set(by_pair) != expected_task_ids:
        raise RuntimeError(
            "multi-fault manifest task coverage drift; "
            f"missing={sorted(expected_task_ids - set(by_pair))}, "
            f"extra={sorted(set(by_pair) - expected_task_ids)}"
        )
    for domain in ("retail", "airline"):
        if len(families_by_domain[domain]) < 2 or len(tools_by_domain[domain]) < 2:
            raise RuntimeError(f"{domain}: multi-fault diversity is insufficient")
    return by_pair


def load_split(path: Path, *, strict_counts: bool = True) -> dict[str, Any]:
    if strict_counts and sha256_file(path) != PUBLISHED_STAGE0_SPLIT_SHA256:
        raise RuntimeError(
            "historical Stage-0 split differs from its published SHA-256"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "v5_stage0_tau2_end_to_end":
        raise RuntimeError("unexpected frozen split protocol")
    guarantees = payload.get("guarantees") or {}
    if guarantees.get("validation_derived_from_official_train_only") is not True:
        raise RuntimeError("split does not certify validation-from-train")
    if guarantees.get("official_test_task_content_exported") is not False:
        raise RuntimeError("split does not certify the official-test seal")
    domains = payload.get("domains")
    if not isinstance(domains, dict) or set(domains) != {"retail", "airline"}:
        raise RuntimeError("split domains must be exactly retail and airline")
    expected_counts = {
        "retail": (59, 15, 40),
        "airline": (24, 6, 20),
    }
    for domain in ("retail", "airline"):
        row = domains.get(domain) or {}
        inner = {str(value) for value in row.get("inner_train_ids") or []}
        validation = {str(value) for value in row.get("validation_ids") or []}
        sealed = {str(value) for value in row.get("sealed_test_ids") or []}
        if not inner or not validation or not sealed:
            raise RuntimeError(f"{domain}: incomplete frozen split")
        if inner & validation or inner & sealed or validation & sealed:
            raise RuntimeError(f"{domain}: frozen split overlap")
        if strict_counts and (len(inner), len(validation), len(sealed)) != expected_counts[domain]:
            raise RuntimeError(
                f"{domain}: frozen split count drift; "
                f"observed={(len(inner), len(validation), len(sealed))}, "
                f"expected={expected_counts[domain]}"
            )
    return payload


def _raw_paths(raw_dir: Path, domain: str, condition: str) -> list[Path]:
    single = raw_dir / f"{domain}_{condition}.json"
    shards = sorted(raw_dir.glob(f"{domain}_{condition}.shard-*-of-*.json"))
    if single.exists() and shards:
        raise RuntimeError(f"{domain}/{condition}: both single and sharded results exist")
    paths = [single] if single.exists() else shards
    if not paths:
        raise RuntimeError(f"{domain}/{condition}: no generation results")
    return paths


def load_raw(
    raw_dir: Path,
    split: dict[str, Any],
    *,
    expected_trials: int,
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    list[Path],
]:
    by_condition: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {
        "clean": {},
        "error": {},
    }
    paths: list[Path] = []
    for domain in ("retail", "airline"):
        allowed = {
            str(value) for value in split["domains"][domain]["inner_train_ids"]
        }
        for condition in ("clean", "error"):
            for path in _raw_paths(raw_dir, domain, condition):
                paths.append(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                simulations = payload.get("simulations")
                if not isinstance(simulations, list):
                    raise RuntimeError(f"{path}: simulations must be a list")
                for simulation in simulations:
                    if not isinstance(simulation, dict):
                        raise RuntimeError(f"{path}: simulation must be an object")
                    key = _run_key(domain, simulation)
                    if key[1] not in allowed:
                        raise RuntimeError(f"{path}: task {key[:2]} is not inner-train")
                    if key in by_condition[condition]:
                        raise RuntimeError(f"duplicate {condition} run: {key}")
                    by_condition[condition][key] = simulation
        observed = Counter(
            key[1] for key in by_condition["clean"] if key[0] == domain
        )
        observed_error = Counter(
            key[1] for key in by_condition["error"] if key[0] == domain
        )
        if set(observed) != allowed or set(observed_error) != allowed:
            raise RuntimeError(f"{domain}: raw results do not cover every inner-train task")
        if any(value != expected_trials for value in observed.values()):
            raise RuntimeError(f"{domain}: clean trial count is not {expected_trials}")
        if observed != observed_error:
            raise RuntimeError(f"{domain}: clean/error trial coverage differs")
    if set(by_condition["clean"]) != set(by_condition["error"]):
        raise RuntimeError("clean/error run keys are not paired")
    for key in by_condition["clean"]:
        clean_simulation = by_condition["clean"][key]
        error_simulation = by_condition["error"][key]
        if clean_simulation.get("trial") != error_simulation.get("trial"):
            raise RuntimeError(f"clean/error actual trial mismatch: {key}")
        clean_seed = clean_simulation.get("seed")
        error_seed = error_simulation.get("seed")
        if clean_seed is None or error_seed is None or clean_seed != error_seed:
            raise RuntimeError(f"clean/error actual seed mismatch: {key}")
    return by_condition["clean"], by_condition["error"], sorted(set(paths))


def validate_generation_contracts(
    *,
    raw_dir: Path,
    split_manifest: Path,
    split: dict[str, Any],
    generation_manifest: Path,
    expected_trials: int,
    expected_seed: int,
    expected_source_commit: str,
    dynamic_audit_identity: dict[str, Any],
    expected_generation_source_commit: str | None = None,
    expected_shards: int = 4,
    observed_source_commit: str | None = None,
) -> dict[str, Any]:
    if observed_source_commit is None:
        require_clean_tracked_source()
        observed_source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD^{commit}"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if observed_source_commit != expected_source_commit:
        raise RuntimeError(
            "local source commit drift: "
            f"HEAD={observed_source_commit}, expected={expected_source_commit}"
        )
    if expected_generation_source_commit is None:
        expected_generation_source_commit = expected_source_commit
    manifest = json.loads(generation_manifest.read_text(encoding="utf-8"))
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("generation manifest lacks rows")
    expected_tasks = {
        f"{domain}:{task_id}"
        for domain in ("retail", "airline")
        for task_id in split["domains"][domain]["inner_train_ids"]
    }
    validate_multifault_manifest(
        manifest,
        expected_protocol="v5_stage1_multifault_data_construction",
        expected_source_split="derived_inner_train",
        expected_task_ids=expected_tasks,
        split_manifest_sha256=sha256_file(split_manifest),
    )
    observed_manifest_tasks = {
        f"{row.get('domain')}:{row.get('task_id')}" for row in rows
    }
    if observed_manifest_tasks != expected_tasks or len(rows) != len(expected_tasks):
        raise RuntimeError("generation manifest is not exactly the 83 inner-train tasks")
    paths = sorted(raw_dir.glob("run_contract.shard-*-of-*.json"))
    if len(paths) != expected_shards:
        raise RuntimeError(
            f"expected {expected_shards} generation run contracts, found {len(paths)}"
        )
    actual_result_paths = {
        path.name: path
        for path in raw_dir.iterdir()
        if path.is_file() and RAW_RESULT_RE.fullmatch(path.name)
    }
    indices: set[int] = set()
    task_union: set[str] = set()
    common: dict[str, Any] | None = None
    source_commits: set[str] = set()
    declared_result_hashes: dict[str, str] = {}
    for path in paths:
        contract = json.loads(path.read_text(encoding="utf-8"))
        if contract.get("protocol") != "v5_stage1_inner_train_generation_run":
            raise RuntimeError(f"{path}: generation contract protocol drift")
        if contract.get("status") != "COMPLETE":
            raise RuntimeError(f"{path}: generation contract is not COMPLETE")
        if (
            contract.get("source_split") != "derived_inner_train"
            or contract.get("official_test_used") is not False
        ):
            raise RuntimeError(f"{path}: unsafe generation source split")
        index = contract.get("shard_index")
        if (
            type(index) is not int
            or contract.get("num_shards") != expected_shards
            or index in indices
        ):
            raise RuntimeError(f"{path}: invalid/duplicate shard identity")
        indices.add(index)
        result_hashes = contract.get("result_sha256")
        if not isinstance(result_hashes, dict) or not result_hashes:
            raise RuntimeError(f"{path}: COMPLETE contract lacks result_sha256")
        for filename, declared_sha in sorted(result_hashes.items()):
            match = (
                GENERATION_RESULT_RE.fullmatch(filename)
                if isinstance(filename, str)
                else None
            )
            if match is None:
                raise RuntimeError(
                    f"{path}: invalid generation result filename {filename!r}"
                )
            if (
                int(match.group("shard")) != index
                or int(match.group("total")) != expected_shards
            ):
                raise RuntimeError(
                    f"{path}: result filename/shard identity drift: {filename}"
                )
            if filename in declared_result_hashes:
                raise RuntimeError(
                    f"{path}: result filename declared by multiple contracts: "
                    f"{filename}"
                )
            if (
                not isinstance(declared_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", declared_sha) is None
            ):
                raise RuntimeError(
                    f"{path}: invalid result SHA-256 for {filename}"
                )
            result_path = actual_result_paths.get(filename)
            if result_path is None:
                raise RuntimeError(f"{path}: declared result file is missing: {filename}")
            observed_sha = sha256_file(result_path)
            if observed_sha != declared_sha:
                raise RuntimeError(
                    f"{path}: generation result SHA drift for {filename}"
                )
            declared_result_hashes[filename] = declared_sha
        if contract.get("manifest_sha256") != sha256_file(generation_manifest):
            raise RuntimeError(f"{path}: generation manifest hash drift")
        if contract.get("split_manifest_sha256") != sha256_file(split_manifest):
            raise RuntimeError(f"{path}: split manifest hash drift")
        if contract.get("dynamic_audit_identity") != dynamic_audit_identity:
            raise RuntimeError(f"{path}: dynamic audit identity drift")
        if contract.get("fault_protocol") != manifest.get("fault_protocol"):
            raise RuntimeError(f"{path}: fault protocol drift")
        if contract.get("tau2_commit") != manifest.get("tau2_commit"):
            raise RuntimeError(f"{path}: tau2 commit drift")
        if contract.get("source_files") != manifest.get("source_files"):
            raise RuntimeError(f"{path}: source-file provenance drift")
        if contract.get("num_trials") != expected_trials:
            raise RuntimeError(f"{path}: trial count drift")
        tasks = contract.get("task_ids")
        if not isinstance(tasks, list) or len(tasks) != len(set(tasks)):
            raise RuntimeError(f"{path}: invalid contract task IDs")
        if task_union & set(tasks):
            raise RuntimeError(f"{path}: shard task overlap")
        task_union.update(tasks)
        frozen = {
            "teacher": contract.get("teacher"),
            "user": contract.get("user"),
            "judge": contract.get("judge"),
            "decoding": contract.get("decoding"),
        }
        if common is None:
            common = frozen
        elif frozen != common:
            raise RuntimeError("generation model/decoding contracts differ across shards")
        if (contract.get("decoding") or {}).get("seed") != expected_seed:
            raise RuntimeError(f"{path}: generation seed drift")
        source_commit = contract.get("source_commit")
        if (
            not isinstance(source_commit, str)
            or len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)
        ):
            raise RuntimeError(f"{path}: invalid source commit")
        if source_commit != expected_generation_source_commit:
            raise RuntimeError(f"{path}: source commit differs from frozen invocation")
        source_commits.add(source_commit)
        for role in ("teacher", "user", "judge"):
            revision = (contract.get(role) or {}).get("revision")
            if (
                not isinstance(revision, str)
                or len(revision) != 40
                or any(character not in "0123456789abcdef" for character in revision)
            ):
                raise RuntimeError(f"{path}: invalid {role} revision")
        teacher = contract.get("teacher") or {}
        user = contract.get("user") or {}
        judge = contract.get("judge") or {}
        if (
            teacher.get("model") != TEACHER_MODEL
            or teacher.get("revision") != TEACHER_REVISION
            or teacher.get("mode") != "ground_truth"
        ):
            raise RuntimeError(f"{path}: frozen teacher contract drift")
        if (
            user.get("model") != USER_JUDGE_MODEL
            or user.get("revision") != USER_JUDGE_REVISION
            or judge.get("model") != USER_JUDGE_MODEL
            or judge.get("revision") != USER_JUDGE_REVISION
        ):
            raise RuntimeError(f"{path}: frozen user/judge contract drift")
        expected_decoding = {
            "temperature": 0,
            "max_tokens": 512,
            "max_steps": 60,
            "task_timeout_seconds": 900,
            "seed": SEED,
        }
        if contract.get("decoding") != expected_decoding:
            raise RuntimeError(f"{path}: frozen decoding contract drift")
    if indices != set(range(expected_shards)) or task_union != expected_tasks:
        raise RuntimeError("generation contracts do not partition all inner-train tasks")
    if len(source_commits) != 1:
        raise RuntimeError("generation source commits differ")
    declared_names = set(declared_result_hashes)
    actual_names = set(actual_result_paths)
    if declared_names != actual_names:
        raise RuntimeError(
            "generation result declaration set drift: "
            f"missing={sorted(declared_names - actual_names)}, "
            f"undeclared={sorted(actual_names - declared_names)}"
        )
    return {
        "contract_files": {str(path): sha256_file(path) for path in paths},
        "result_files": {
            str(actual_result_paths[name]): declared_result_hashes[name]
            for name in sorted(declared_result_hashes)
        },
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "dynamic_audit_identity": dynamic_audit_identity,
        "source_commit": next(iter(source_commits)),
        "task_union": len(task_union),
        "shards": expected_shards,
        "frozen_roles_and_decoding": common,
    }


def normalize_tool_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise RuntimeError("tool schema must be an object")
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        result = deepcopy(schema)
    elif isinstance(schema.get("name"), str):
        result = {"type": "function", "function": deepcopy(schema)}
    else:
        raise RuntimeError("unsupported tool schema")
    function = result["function"]
    if not isinstance(function.get("name"), str) or not function["name"]:
        raise RuntimeError("tool schema lacks a name")
    return result


def load_tau2_contexts(tau2_root: Path) -> dict[str, dict[str, Any]]:
    require_clean_tracked_checkout(tau2_root, label="tau2 checkout")
    source = tau2_root.resolve() / "src"
    if not source.is_dir():
        raise RuntimeError(f"tau2 source directory missing: {source}")
    sys.path.insert(0, str(source))
    try:
        from tau2.domains.airline.environment import get_environment as airline_env
        from tau2.domains.retail.environment import get_environment as retail_env

        environments = {"retail": retail_env(), "airline": airline_env()}
        result = {}
        for domain, environment in environments.items():
            schemas = sorted(
                [normalize_tool_schema(tool.openai_schema) for tool in environment.get_tools()],
                key=lambda item: item["function"]["name"],
            )
            policy = environment.get_policy()
            if not isinstance(policy, str) or not policy or not schemas:
                raise RuntimeError(f"{domain}: empty policy/tool schemas")
            result[domain] = {"policy": policy, "tool_schemas": schemas}
        return result
    finally:
        if sys.path and sys.path[0] == str(source):
            sys.path.pop(0)


def system_message(context: dict[str, Any]) -> dict[str, Any]:
    content = (
        "<instructions>\n"
        + SYSTEM_INSTRUCTION
        + "\n</instructions>\n<policy>\n"
        + context["policy"]
        + "\n</policy>"
    )
    return {"role": "system", "content": content}


def analyze_simulation(
    simulation: dict[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    messages = simulation.get("messages")
    if not isinstance(messages, list) or not messages:
        # tau2 may persist an empty rollout when inference terminates before the
        # first message (for example, after a context-window rejection). Such a
        # rollout has no evidence that can safely become an SFT label.
        return {
            "messages": [],
            "outcomes": [],
            "first_user": None,
            "eligible": False,
            "reason": "invalid_simulation_no_messages",
            "final_success": False,
            "failed": [],
            "injected": [],
            "repair_index": None,
        }
    outcomes: list[dict[str, Any]] = []
    tool_result_indices: set[int] = set()
    first_user = next(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        None,
    )
    if first_user is None:
        raise RuntimeError("simulation has no user message")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RuntimeError("message must be an object")
        calls = message.get("tool_calls") or []
        if calls:
            if message.get("role") != "assistant" or len(calls) != 1:
                return {
                    "messages": messages,
                    "outcomes": [],
                    "first_user": first_user,
                    "eligible": False,
                    "reason": "invalid_simulation_non_single_tool_call",
                    "final_success": False,
                    "failed": [],
                    "injected": [],
                    "repair_index": None,
                }
            if message.get("content") not in (None, ""):
                raise RuntimeError("assistant may not mix text and tool call")
            name, arguments, call_id = _function_call(
                calls[0], f"messages[{index}].tool_calls[0]"
            )
            result_index = index + 1
            if result_index >= len(messages):
                raise RuntimeError("assistant tool call lacks adjacent result")
            result = messages[result_index]
            if result.get("role") != "tool":
                raise RuntimeError("assistant tool result is not adjacent")
            if _tool_link_id(result) != call_id:
                raise RuntimeError("tool call/result id mismatch")
            if result.get("name") not in (None, name):
                raise RuntimeError("tool call/result name mismatch")
            if type(result.get("error")) is not bool:
                raise RuntimeError("tool result requires explicit boolean error")
            tool_result_indices.add(result_index)
            outcomes.append(
                {
                    "assistant_index": index,
                    "result_index": result_index,
                    "error": result["error"],
                    "injected": _is_injected(message),
                    "name": name,
                    "arguments": arguments,
                    "call_id": call_id,
                }
            )
        elif message.get("role") == "tool":
            if index not in tool_result_indices:
                raise RuntimeError("orphan/non-adjacent tool result")
    injected = [row for row in outcomes if row["injected"]]
    failed = [row for row in outcomes if row["error"]]
    successful = [row for row in outcomes if not row["error"]]
    final_success = _reward_is_one(simulation)
    if condition == "clean":
        eligible = final_success and not failed and not injected
        reason = "eligible" if eligible else (
            "final_failure" if not final_success else "clean_contains_failure_or_injection"
        )
        repair_index = None
    else:
        injected_failure = (
            len(injected) == 1 and injected[0]["error"] and len(failed) == 1
        )
        later_successes = (
            [
                row
                for row in successful
                if injected and row["assistant_index"] > injected[0]["result_index"]
            ]
            if injected
            else []
        )
        eligible = final_success and injected_failure and bool(later_successes)
        if not final_success:
            reason = "final_failure"
        elif not injected_failure:
            reason = "not_exactly_one_controlled_failure"
        elif not later_successes:
            reason = "no_verified_post_error_repair"
        else:
            reason = "eligible"
        repair_index = later_successes[0]["assistant_index"] if later_successes else None
    return {
        "messages": messages,
        "outcomes": outcomes,
        "first_user": first_user,
        "eligible": eligible,
        "reason": reason,
        "final_success": final_success,
        "failed": failed,
        "injected": injected,
        "repair_index": repair_index,
    }


def neutralize_messages(
    messages: list[dict[str, Any]], context: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    output = [system_message(context)]
    index_map: dict[int, int] = {}
    call_ids: dict[str, str] = {}
    next_call = 1
    for old_index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        compact: dict[str, Any] = {"role": role, "content": message.get("content")}
        calls = message.get("tool_calls") or []
        if calls:
            name, arguments, original_id = _function_call(
                calls[0], f"messages[{old_index}].tool_calls[0]"
            )
            if original_id not in call_ids:
                call_ids[original_id] = f"call_{next_call:04d}"
                next_call += 1
            compact["tool_calls"] = [
                {
                    "id": call_ids[original_id],
                    "name": name,
                    "arguments": arguments,
                }
            ]
        if role == "tool":
            original_id = _tool_link_id(message)
            if original_id is None or original_id not in call_ids:
                raise RuntimeError("tool result cannot be neutralized without linked call")
            compact["tool_call_id"] = call_ids[original_id]
            compact["error"] = message["error"]
            if message.get("name") is not None:
                compact["name"] = message["name"]
        index_map[old_index] = len(output)
        output.append(compact)
    return output, index_map


def _template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        item = {"role": message["role"], "content": message.get("content") or ""}
        if message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": deepcopy(call.get("arguments") or {}),
                    },
                }
                for call in message["tool_calls"]
            ]
        if message["role"] == "tool":
            item["tool_call_id"] = message["tool_call_id"]
        normalized.append(item)
    return normalized


def token_contract(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    label_mask: list[bool],
    tool_schemas: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized = _template_messages(messages)

    def ids(prefix: list[dict[str, Any]], generation: bool) -> list[int]:
        value = tokenizer.apply_chat_template(
            prefix,
            tools=tool_schemas,
            tokenize=True,
            add_generation_prompt=generation,
        )
        if isinstance(value, dict):
            value = value.get("input_ids")
        if not isinstance(value, list) or any(type(token) is not int for token in value):
            raise RuntimeError("tokenizer did not return a flat integer list")
        return value

    full = ids(normalized, False)
    spans = []
    for index, selected in enumerate(label_mask):
        if not selected:
            continue
        before = ids(normalized[:index], True)
        through = ids(normalized[: index + 1], False)
        if len(before) >= len(through) or through[: len(before)] != before:
            raise RuntimeError(f"chat template is not prefix-stable at message {index}")
        if full[: len(through)] != through:
            raise RuntimeError(f"future message changes target prefix at message {index}")
        spans.append(
            {
                "message_index": index,
                "token_start": len(before),
                "token_end": len(through),
            }
        )
    supervised = sum(row["token_end"] - row["token_start"] for row in spans)
    if supervised <= 0:
        raise RuntimeError("example has no supervised tokens")
    return {
        "sequence_tokens": len(full),
        "supervised_tokens": supervised,
        "label_spans": spans,
    }


def make_source_example(
    *,
    domain: str,
    task_id: str,
    trial: str,
    pair_id: str,
    analysis: dict[str, Any],
    context: dict[str, Any],
    variant: str,
    fault: dict[str, Any],
    tokenizer: Any,
    seed: int,
) -> dict[str, Any]:
    messages, index_map = neutralize_messages(analysis["messages"], context)
    labels = [False] * len(messages)
    outcomes = analysis["outcomes"]
    for outcome in outcomes:
        new_index = index_map[outcome["assistant_index"]]
        if not outcome["error"]:
            labels[new_index] = True
        elif variant == "failure_raw" and outcome["injected"]:
            labels[new_index] = True
    selected = [index for index, value in enumerate(labels) if value]
    if not selected:
        raise RuntimeError(f"{pair_id}/{variant}: no valid labels")
    last = max(selected)
    # A selected tool call always retains its adjacent, ID-matched result for
    # reverse auditing.  Nothing after that result is retained.  A selected
    # assistant text message remains the final message.
    last_old = next(
        old_index for old_index, new_index in index_map.items() if new_index == last
    )
    if analysis["messages"][last_old].get("tool_calls"):
        result_old = last_old + 1
        if result_old not in index_map:
            raise RuntimeError("selected final tool call lost its linked result")
        last = index_map[result_old]
    messages = messages[: last + 1]
    labels = labels[: last + 1]
    failed_indices = [
        index_map[row["assistant_index"]]
        for row in analysis["failed"]
        if row["assistant_index"] in index_map and index_map[row["assistant_index"]] <= last
    ]
    injected_index = (
        index_map[analysis["injected"][0]["assistant_index"]]
        if analysis["injected"]
        else None
    )
    if variant == "failure_raw":
        if len(failed_indices) != 1 or not labels[failed_indices[0]]:
            raise RuntimeError("failure_raw must supervise exactly one controlled failure")
    elif any(labels[index] for index in failed_indices):
        raise RuntimeError(f"{variant}: failed call unexpectedly supervised")
    if variant == "repair_masked":
        if injected_index is None or not any(
            value and index > injected_index for index, value in enumerate(labels)
        ):
            raise RuntimeError("repair-only example lacks supervised post-error repair")
    source = "perfect_success" if variant == "perfect_success" else "failure_rich"
    source_id = f"{pair_id}:{variant}"
    schemas = context["tool_schemas"]
    schema_hash = sha256_text(canonical(schemas))
    contract = token_contract(tokenizer, messages, labels, schemas)
    return {
        "id": source_id,
        "messages": messages,
        "label_mask": labels,
        "metadata": {
            "arm": variant,
            "source": source,
            "domain": domain,
            "task_id": task_id,
            "trial": trial,
            "seed": seed,
            "pair_id": pair_id,
            "source_example_id": source_id,
            "source_split": "inner_train",
            "fit_split": "unassigned",
            "fault_family": fault["fault_family"],
            "fault_tool_name": fault["tool_name"],
            "fault_relevance": fault["fault_relevance"],
            "fault_on_reference_path": fault["on_reference_path"],
            "failed_assistant_message_indices": failed_indices,
            "injected_failed_assistant_message_index": injected_index,
            "tool_schemas": schemas,
            "tool_schemas_sha256": schema_hash,
            "call_id_policy": "condition_neutral_sequential",
            "full_trajectory_reward": 1.0,
            "source_trajectory_sha256": sha256_text(
                canonical(analysis["messages"])
            ),
            "official_test_used": False,
        },
        "token_contract": contract,
    }


def deterministic_order(values: Iterable[Any], seed: int, salt: str) -> list[Any]:
    return sorted(
        values,
        key=lambda value: (
            sha256_text(f"{seed}:{salt}:{canonical(value)}"),
            canonical(value),
        ),
    )


def take_to_budget(
    rows: list[dict[str, Any]],
    target: int,
    *,
    token_field: str = "supervised_tokens",
) -> list[dict[str, Any]]:
    """Deterministically take whole rows without exceeding a token target.

    This small helper is intentionally strict about the zero endpoint because
    the predecessor Stage-0 builder accidentally selected one row for a
    ``target=0`` mixture.
    """
    if target < 0:
        raise ValueError("token target must be non-negative")
    if target == 0:
        return []
    selected: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        size = row["token_contract"][token_field]
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError("token budget row has invalid token count")
        if total + size > target:
            continue
        selected.append(row)
        total += size
    return selected


def load_validation_manifest(
    path: Path,
    *,
    split: dict[str, Any],
    split_manifest_path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        f"{domain}:{task_id}"
        for domain in ("retail", "airline")
        for task_id in split["domains"][domain]["validation_ids"]
    }
    validate_multifault_manifest(
        payload,
        expected_protocol="v5_stage1_sft_causal_validation",
        expected_source_split="derived_validation",
        expected_task_ids=expected,
        split_manifest_sha256=sha256_file(split_manifest_path),
    )
    if len(expected) != 21:
        raise RuntimeError(f"expected 21 validation tasks, found {len(expected)}")
    observed_counts = {
        domain: sum(row["domain"] == domain for row in payload["rows"])
        for domain in ("retail", "airline")
    }
    if payload.get("domain_counts") != observed_counts:
        raise RuntimeError("validation multi-fault domain counts drift")
    return payload


def source_clone(source: dict[str, Any], *, arm: str, slot: int, fit_split: str) -> dict[str, Any]:
    row = deepcopy(source)
    row["id"] = f"{arm}:schedule:{slot:04d}"
    row["metadata"]["arm"] = arm
    row["metadata"]["fit_split"] = fit_split
    row["metadata"]["source_example_id"] = source["id"]
    return row


def task_schedule(task_ids: list[str], rows: int, seed: int) -> list[str]:
    schedule: list[str] = []
    cycle = 0
    while len(schedule) < rows:
        schedule.extend(deterministic_order(task_ids, seed, f"task-cycle:{cycle}"))
        cycle += 1
    return schedule[:rows]


def choose_candidates(
    task_ids: list[str],
    candidates: dict[str, list[dict[str, Any]]],
    *,
    target_tokens: float,
    target_sequence: float,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used_tokens = 0
    used_sequence = 0
    suffix_token_min = [0] * (len(task_ids) + 1)
    suffix_token_max = [0] * (len(task_ids) + 1)
    suffix_sequence_min = [0] * (len(task_ids) + 1)
    suffix_sequence_max = [0] * (len(task_ids) + 1)
    for index in range(len(task_ids) - 1, -1, -1):
        options = candidates[task_ids[index]]
        token_values = [
            row["token_contract"]["supervised_tokens"] for row in options
        ]
        sequence_values = [
            row["token_contract"]["sequence_tokens"] for row in options
        ]
        suffix_token_min[index] = suffix_token_min[index + 1] + min(token_values)
        suffix_token_max[index] = suffix_token_max[index + 1] + max(token_values)
        suffix_sequence_min[index] = (
            suffix_sequence_min[index + 1] + min(sequence_values)
        )
        suffix_sequence_max[index] = (
            suffix_sequence_max[index + 1] + max(sequence_values)
        )

    def interval_violation(value: float, lower: float, upper: float) -> float:
        if value < lower:
            return lower - value
        if value > upper:
            return value - upper
        return 0.0

    for position, task_id in enumerate(task_ids):
        options = deterministic_order(
            candidates[task_id], seed, f"{salt}:{position}:{task_id}"
        )
        remaining = len(task_ids) - position
        token_ideal = (target_tokens - used_tokens) / remaining
        sequence_ideal = (target_sequence - used_sequence) / remaining
        selected = min(
            options,
            key=lambda row: (
                100.0
                * interval_violation(
                    target_tokens
                    - used_tokens
                    - row["token_contract"]["supervised_tokens"],
                    suffix_token_min[position + 1],
                    suffix_token_max[position + 1],
                )
                / max(1.0, target_tokens)
                + 100.0
                * interval_violation(
                    target_sequence
                    - used_sequence
                    - row["token_contract"]["sequence_tokens"],
                    suffix_sequence_min[position + 1],
                    suffix_sequence_max[position + 1],
                )
                / max(1.0, target_sequence)
                +
                abs(row["token_contract"]["supervised_tokens"] - token_ideal)
                / max(1.0, abs(token_ideal))
                + abs(row["token_contract"]["sequence_tokens"] - sequence_ideal)
                / max(1.0, abs(sequence_ideal)),
                sha256_text(f"{seed}:{salt}:{position}:{row['id']}"),
            ),
        )
        chosen.append(selected)
        used_tokens += selected["token_contract"]["supervised_tokens"]
        used_sequence += selected["token_contract"]["sequence_tokens"]
    return chosen


def endpoint_bounds(
    task_ids: list[str], candidates: dict[str, list[dict[str, Any]]], field: str
) -> tuple[int, int]:
    lower = 0
    upper = 0
    for task_id in task_ids:
        values = [row["token_contract"][field] for row in candidates[task_id]]
        lower += min(values)
        upper += max(values)
    return lower, upper


def mixture_schedule(
    task_ids: list[str],
    clean: dict[str, list[dict[str, Any]]],
    recovery: dict[str, list[dict[str, Any]]],
    *,
    ratio: float,
    target_tokens: int,
    target_sequence: int,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    ranked_positions = deterministic_order(
        list(range(len(task_ids))), seed, f"{salt}:source-positions"
    )
    mean_clean = sum(
        row["token_contract"]["supervised_tokens"]
        for rows in clean.values()
        for row in rows
    ) / sum(len(rows) for rows in clean.values())
    mean_recovery = sum(
        row["token_contract"]["supervised_tokens"]
        for rows in recovery.values()
        for row in rows
    ) / sum(len(rows) for rows in recovery.values())
    estimate = int(
        round(
            ratio
            * len(task_ids)
            * mean_clean
            / max(1e-9, (1.0 - ratio) * mean_recovery + ratio * mean_clean)
        )
    )
    candidate_counts = range(
        max(1, estimate - 32), min(len(task_ids) - 1, estimate + 32) + 1
    )
    best: tuple[float, list[dict[str, Any]]] | None = None
    for recovery_count in candidate_counts:
        recovery_positions = set(ranked_positions[:recovery_count])
        clean_positions = [
            index for index in range(len(task_ids)) if index not in recovery_positions
        ]
        rec_positions = [
            index for index in range(len(task_ids)) if index in recovery_positions
        ]
        clean_rows = choose_candidates(
            [task_ids[index] for index in clean_positions],
            clean,
            target_tokens=target_tokens * (1.0 - ratio),
            target_sequence=target_sequence * len(clean_positions) / len(task_ids),
            seed=seed,
            salt=f"{salt}:clean:{recovery_count}",
        )
        rec_rows = choose_candidates(
            [task_ids[index] for index in rec_positions],
            recovery,
            target_tokens=target_tokens * ratio,
            target_sequence=target_sequence * len(rec_positions) / len(task_ids),
            seed=seed,
            salt=f"{salt}:recovery:{recovery_count}",
        )
        rows_by_position = {
            **dict(zip(clean_positions, clean_rows)),
            **dict(zip(rec_positions, rec_rows)),
        }
        rows = [rows_by_position[index] for index in range(len(task_ids))]
        rec_tokens = sum(
            row["token_contract"]["supervised_tokens"] for row in rec_rows
        )
        total_tokens = sum(
            row["token_contract"]["supervised_tokens"] for row in rows
        )
        total_sequence = sum(
            row["token_contract"]["sequence_tokens"] for row in rows
        )
        score = (
            abs(total_tokens - target_tokens) / max(1, target_tokens)
            + abs(total_sequence - target_sequence) / max(1, target_sequence)
            + 4.0
            * abs(rec_tokens / max(1, total_tokens) - ratio)
        )
        if best is None or score < best[0]:
            best = score, rows
    if best is None:
        raise RuntimeError(f"{salt}: no feasible mixture schedule")
    return best[1]


def prepare(
    *,
    split_manifest: Path,
    raw_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    tokenizer_name: str,
    tokenizer_revision: str,
    domain_contexts: dict[str, dict[str, Any]],
    seed: int = SEED,
    schedule_rows: int = SCHEDULE_ROWS,
    expected_trials: int = 3,
    slots_per_task: int = 3,
    min_distinct_tasks: int = 40,
    min_paired_slots: int = 120,
    include_optional_mixtures: bool = False,
    token_tolerance: float = 0.01,
    sequence_tolerance: float = 0.02,
    strict_split_counts: bool = True,
    generation_manifest: Path | None = None,
    validation_manifest: Path | None = None,
    generation_dynamic_audit: Path | None = None,
    validation_dynamic_audit: Path | None = None,
    strict_generation_contracts: bool = True,
    expected_source_commit: str | None = None,
    expected_generation_source_commit: str | None = None,
) -> dict[str, Any]:
    if not tokenizer_revision:
        raise RuntimeError("A pinned 7B tokenizer revision is required")
    if schedule_rows <= 0 or slots_per_task < 2:
        raise ValueError("schedule_rows must be positive and slots_per_task >= 2")
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise RuntimeError(f"output directory must be absent or empty: {output_dir}")
    split = load_split(split_manifest, strict_counts=strict_split_counts)
    if generation_manifest is None:
        raise RuntimeError("Stage-1 preparation requires --generation-manifest")
    if validation_manifest is None:
        raise RuntimeError("Stage-1 preparation requires --validation-manifest")
    if generation_dynamic_audit is None:
        raise RuntimeError(
            "Stage-1 preparation requires --generation-dynamic-audit"
        )
    if validation_dynamic_audit is None:
        raise RuntimeError(
            "Stage-1 preparation requires --validation-dynamic-audit"
        )
    generation_payload = json.loads(
        generation_manifest.read_text(encoding="utf-8")
    )
    generation_task_ids = {
        f"{domain}:{task_id}"
        for domain in ("retail", "airline")
        for task_id in split["domains"][domain]["inner_train_ids"]
    }
    generation_rows = validate_multifault_manifest(
        generation_payload,
        expected_protocol="v5_stage1_multifault_data_construction",
        expected_source_split="derived_inner_train",
        expected_task_ids=generation_task_ids,
        split_manifest_sha256=sha256_file(split_manifest),
    )
    validated_validation_manifest = load_validation_manifest(
        validation_manifest,
        split=split,
        split_manifest_path=split_manifest,
    )
    generation_dynamic_audit_identity = load_complete_dynamic_audit(
        generation_dynamic_audit,
        manifest_path=generation_manifest,
        split_manifest_path=split_manifest,
        expected_source_split="derived_inner_train",
        expected_task_ids=set(generation_task_ids),
    )
    validation_dynamic_audit_identity = load_complete_dynamic_audit(
        validation_dynamic_audit,
        manifest_path=validation_manifest,
        split_manifest_path=split_manifest,
        expected_source_split="derived_validation",
        expected_task_ids={
            f"{row['domain']}:{row['task_id']}"
            for row in validated_validation_manifest["rows"]
        },
    )
    generation_contract_audit = None
    if strict_generation_contracts:
        if expected_source_commit is None:
            raise RuntimeError("formal preparation requires --expected-source-commit")
        generation_contract_audit = validate_generation_contracts(
            raw_dir=raw_dir,
            split_manifest=split_manifest,
            split=split,
            generation_manifest=generation_manifest,
            expected_trials=expected_trials,
            expected_seed=seed,
            expected_source_commit=expected_source_commit,
            expected_generation_source_commit=expected_generation_source_commit,
            dynamic_audit_identity=generation_dynamic_audit_identity,
        )
    clean_raw, error_raw, raw_paths = load_raw(
        raw_dir, split, expected_trials=expected_trials
    )
    exclusions = Counter()
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    master_rows = []
    for domain, task_id, trial in sorted(clean_raw):
        fault = generation_rows[f"{domain}:{task_id}"]["error_condition"]
        clean_analysis = analyze_simulation(clean_raw[(domain, task_id, trial)], condition="clean")
        recovery_analysis = analyze_simulation(
            error_raw[(domain, task_id, trial)], condition="error"
        )
        if not clean_analysis["eligible"] or not recovery_analysis["eligible"]:
            exclusions[f"clean:{clean_analysis['reason']}"] += (
                not clean_analysis["eligible"]
            )
            exclusions[f"recovery:{recovery_analysis['reason']}"] += (
                not recovery_analysis["eligible"]
            )
            continue
        injected = recovery_analysis["injected"]
        if len(injected) != 1 or (
            injected[0]["name"],
            canonical(injected[0]["arguments"]),
        ) != (
            fault["tool_name"],
            canonical(fault["arguments"]),
        ):
            raise RuntimeError(
                f"{domain}:{task_id}:{trial}: injected fault differs from "
                "multi-fault manifest"
            )
        pair_id = f"{domain}:{task_id}:{trial}"
        context = domain_contexts[domain]
        clean = make_source_example(
            domain=domain,
            task_id=task_id,
            trial=trial,
            pair_id=pair_id,
            analysis=clean_analysis,
            context=context,
            variant="perfect_success",
            fault=fault,
            tokenizer=tokenizer,
            seed=seed,
        )
        raw = make_source_example(
            domain=domain,
            task_id=task_id,
            trial=trial,
            pair_id=pair_id,
            analysis=recovery_analysis,
            context=context,
            variant="failure_raw",
            fault=fault,
            tokenizer=tokenizer,
            seed=seed,
        )
        repair = make_source_example(
            domain=domain,
            task_id=task_id,
            trial=trial,
            pair_id=pair_id,
            analysis=recovery_analysis,
            context=context,
            variant="repair_masked",
            fault=fault,
            tokenizer=tokenizer,
            seed=seed,
        )
        master = {
            "pair_id": pair_id,
            "domain": domain,
            "task_id": task_id,
            "trial": trial,
            "source_split": "inner_train",
            "perfect_success": clean,
            "failure_raw": raw,
            "repair_masked": repair,
        }
        by_task[f"{domain}:{task_id}"].append(master)
    eligible_tasks = sorted(
        task for task, rows in by_task.items() if len(rows) >= slots_per_task
    )
    if len(eligible_tasks) < min_distinct_tasks:
        raise RuntimeError(
            f"only {len(eligible_tasks)} tasks have {slots_per_task} eligible paired slots; "
            f"need {min_distinct_tasks}"
        )
    capped: dict[str, list[dict[str, Any]]] = {}
    for task in eligible_tasks:
        capped[task] = deterministic_order(
            by_task[task], seed, f"paired-slots:{task}"
        )[:slots_per_task]
        master_rows.extend(capped[task])
    if len(master_rows) < min_paired_slots:
        raise RuntimeError(
            f"only {len(master_rows)} eligible paired slots; need {min_paired_slots}"
        )

    # One source slot per task is held out for inner-train validation loss.
    train_slots = {task: rows[:-1] for task, rows in capped.items()}
    validation_slots = {task: rows[-1] for task, rows in capped.items()}
    perfect_by_task = {
        task: [row["perfect_success"] for row in rows]
        for task, rows in train_slots.items()
    }
    raw_by_task = {
        task: [row["failure_raw"] for row in rows]
        for task, rows in train_slots.items()
    }
    repair_by_task = {
        task: [row["repair_masked"] for row in rows]
        for task, rows in train_slots.items()
    }
    scheduled_tasks = task_schedule(eligible_tasks, schedule_rows, seed)
    token_bounds = [
        endpoint_bounds(scheduled_tasks, pool, "supervised_tokens")
        for pool in (perfect_by_task, raw_by_task, repair_by_task)
    ]
    sequence_bounds = [
        endpoint_bounds(scheduled_tasks, pool, "sequence_tokens")
        for pool in (perfect_by_task, raw_by_task, repair_by_task)
    ]
    token_low, token_high = max(row[0] for row in token_bounds), min(
        row[1] for row in token_bounds
    )
    sequence_low, sequence_high = max(row[0] for row in sequence_bounds), min(
        row[1] for row in sequence_bounds
    )
    if token_low > token_high:
        raise RuntimeError("fixed task multiset has no common supervised-token budget")
    if sequence_low > sequence_high:
        raise RuntimeError("fixed task multiset has no common nonpadding-token budget")
    target_tokens = (token_low + token_high) // 2
    target_sequence = (sequence_low + sequence_high) // 2

    source_schedules = {
        "perfect_success": choose_candidates(
            scheduled_tasks,
            perfect_by_task,
            target_tokens=target_tokens,
            target_sequence=target_sequence,
            seed=seed,
            salt="perfect_success",
        ),
        "failure_raw": choose_candidates(
            scheduled_tasks,
            raw_by_task,
            target_tokens=target_tokens,
            target_sequence=target_sequence,
            seed=seed,
            salt="failure_raw",
        ),
        "repair_100": choose_candidates(
            scheduled_tasks,
            repair_by_task,
            target_tokens=target_tokens,
            target_sequence=target_sequence,
            seed=seed,
            salt="repair_100",
        ),
    }
    requested_arms = list(CORE_ARMS)
    if include_optional_mixtures:
        requested_arms.extend(OPTIONAL_ARMS)
    for arm in requested_arms:
        if arm.startswith("repair_") and arm not in source_schedules:
            source_schedules[arm] = mixture_schedule(
                scheduled_tasks,
                perfect_by_task,
                repair_by_task,
                ratio=ARM_RATIOS[arm],
                target_tokens=target_tokens,
                target_sequence=target_sequence,
                seed=seed,
                salt=arm,
            )

    arm_audit: dict[str, Any] = {}
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    task_multisets = {}
    for arm in requested_arms:
        rows = [
            source_clone(source, arm=arm, slot=index, fit_split="train_schedule")
            for index, source in enumerate(source_schedules[arm])
        ]
        arm_rows[arm] = rows
        total_tokens = sum(row["token_contract"]["supervised_tokens"] for row in rows)
        sequence_tokens = sum(row["token_contract"]["sequence_tokens"] for row in rows)
        recovery_tokens = sum(
            row["token_contract"]["supervised_tokens"]
            for row in rows
            if row["metadata"]["source"] == "failure_rich"
        )
        ratio = recovery_tokens / total_tokens
        if abs(ratio - ARM_RATIOS[arm]) > token_tolerance:
            raise RuntimeError(
                f"{arm}: recovery token ratio {ratio:.6f} exceeds tolerance"
            )
        failed_labels = 0
        for row in rows:
            failed = row["metadata"]["failed_assistant_message_indices"]
            failed_labels += sum(row["label_mask"][index] for index in failed)
        if arm == "failure_raw" and failed_labels != len(rows):
            raise RuntimeError("failure_raw must label exactly one controlled failure per row")
        if arm != "failure_raw" and failed_labels:
            raise RuntimeError(f"{arm}: failed action is positively supervised")
        task_multisets[arm] = Counter(
            (row["metadata"]["domain"], row["metadata"]["task_id"])
            for row in rows
        )
        arm_audit[arm] = {
            "rows": len(rows),
            "supervised_tokens": total_tokens,
            "nonpadding_tokens": sequence_tokens,
            "recovery_supervised_tokens": recovery_tokens,
            "recovery_supervised_token_ratio": ratio,
            "recovery_slot_fraction": sum(
                row["metadata"]["source"] == "failure_rich" for row in rows
            )
            / len(rows),
            "controlled_failed_action_labels": failed_labels,
            "fault_family_counts": dict(
                sorted(
                    Counter(
                        row["metadata"]["fault_family"] for row in rows
                    ).items()
                )
            ),
            "fault_relevance_counts": dict(
                sorted(
                    Counter(
                        row["metadata"]["fault_relevance"] for row in rows
                    ).items()
                )
            ),
        }
    reference_tasks = task_multisets[CORE_ARMS[0]]
    if any(task_multisets[arm] != reference_tasks for arm in CORE_ARMS[1:]):
        raise RuntimeError("core arm task_id multisets differ")
    core_tokens = [arm_audit[arm]["supervised_tokens"] for arm in CORE_ARMS]
    core_sequence = [arm_audit[arm]["nonpadding_tokens"] for arm in CORE_ARMS]
    if (max(core_tokens) - min(core_tokens)) / min(core_tokens) > token_tolerance:
        raise RuntimeError("core arm supervised-token budgets differ by more than 1%")
    if (max(core_sequence) - min(core_sequence)) / min(core_sequence) > sequence_tolerance:
        raise RuntimeError("core arm nonpadding-token budgets differ by more than 2%")

    validation_loss = []
    for task in eligible_tasks:
        slot = validation_slots[task]
        for source in (slot["perfect_success"], slot["repair_masked"]):
            row = source_clone(
                source,
                arm="validation_loss",
                slot=len(validation_loss),
                fit_split="validation_loss",
            )
            validation_loss.append(row)
    train_source_ids = {
        row["id"]
        for arm in requested_arms
        for row in source_schedules[arm]
    }
    validation_source_ids = {
        row["metadata"]["source_example_id"] for row in validation_loss
    }
    if train_source_ids & validation_source_ids:
        raise RuntimeError("training and validation-loss source examples overlap")
    validation_output = validated_validation_manifest
    validation_rows = validation_output["rows"]

    # All eligibility, leakage, and budget checks have passed.  Only now may
    # files appear in the requested output directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "paired_master_pool.jsonl", master_rows)
    for arm, rows in arm_rows.items():
        path = output_dir / "arms" / arm / "train.jsonl"
        write_jsonl(path, rows)
        arm_audit[arm]["sha256"] = sha256_file(path)
    write_jsonl(output_dir / "validation_loss.jsonl", validation_loss)
    write_json(output_dir / "validation_manifest.json", validation_output)

    audit = {
        "status": "PASS",
        "protocol": "v5_stage1_sft_causal",
        "claim_scope": "multi_fault_family_post_fault_robustness_screen",
        "semantic_repair_claim_allowed": False,
        "seed": seed,
        "model_tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "split_manifest_sha256": sha256_file(split_manifest),
        "official_test_used": False,
        "derived_validation_used_for_supervision": False,
        "raw_files": {str(path): sha256_file(path) for path in raw_paths},
        "generation_contracts": generation_contract_audit,
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "input_validation_manifest_sha256": sha256_file(validation_manifest),
        "dynamic_audits": {
            "generation": generation_dynamic_audit_identity,
            "validation": validation_dynamic_audit_identity,
        },
        "raw_paired_runs": len(clean_raw),
        "eligible_distinct_tasks": len(eligible_tasks),
        "paired_master_slots": len(master_rows),
        "slots_per_task": slots_per_task,
        "exclusions": dict(sorted(exclusions.items())),
        "train_schedule_rows_per_arm": schedule_rows,
        "shared_target_supervised_tokens": target_tokens,
        "shared_target_nonpadding_tokens": target_sequence,
        "core_task_id_multiset_equal": True,
        "train_validation_source_overlap": 0,
        "label_guarantees": {
            "per_tool_call_adjacent_outcome_verified": True,
            "final_failure_never_positive_sft": True,
            "raw_labels_only_controlled_failed_action": True,
            "repair_failure_is_context_not_label": True,
            "repair_has_verified_post_error_success": True,
            "causal_prefix_tokenization_verified": True,
            "call_ids_condition_neutral": True,
            "official_test_and_derived_validation_label_leakage": 0,
        },
        "arms": arm_audit,
        "validation_loss": {
            "rows": len(validation_loss),
            "source_split": "inner_train",
            "sha256": sha256_file(output_dir / "validation_loss.jsonl"),
        },
        "validation_manifest": {
            "rows": len(validation_rows),
            "sha256": sha256_file(output_dir / "validation_manifest.json"),
        },
    }
    write_json(output_dir / "audit.json", audit)
    hash_paths = [
        output_dir / "paired_master_pool.jsonl",
        output_dir / "validation_loss.jsonl",
        output_dir / "validation_manifest.json",
        output_dir / "audit.json",
        *[
            output_dir / "arms" / arm / "train.jsonl"
            for arm in requested_arms
        ],
    ]
    hashes = {
        str(path.relative_to(output_dir)): sha256_file(path) for path in hash_paths
    }
    write_json(output_dir / "hashes.json", hashes)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--generation-dynamic-audit", type=Path, required=True)
    parser.add_argument("--validation-dynamic-audit", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-generation-source-commit")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=MODEL)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--schedule-rows", type=int, default=SCHEDULE_ROWS)
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--slots-per-task", type=int, default=3)
    parser.add_argument("--min-distinct-tasks", type=int, default=40)
    parser.add_argument("--min-paired-slots", type=int, default=120)
    parser.add_argument("--include-optional-mixtures", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = {
        "--tokenizer": (args.tokenizer, MODEL),
        "--seed": (args.seed, SEED),
        "--schedule-rows": (args.schedule_rows, SCHEDULE_ROWS),
        "--expected-trials": (args.expected_trials, 3),
        "--slots-per-task": (args.slots_per_task, 3),
        "--min-distinct-tasks": (args.min_distinct_tasks, 40),
        "--min-paired-slots": (args.min_paired_slots, 120),
    }
    drift = [
        f"{name}={observed!r} (expected {expected!r})"
        for name, (observed, expected) in frozen.items()
        if observed != expected
    ]
    if drift:
        raise RuntimeError("formal Stage-1 constants are frozen: " + "; ".join(drift))
    if args.tokenizer_revision != MODEL_REVISION:
        raise RuntimeError(
            "--tokenizer-revision drift: "
            f"expected frozen {MODEL_REVISION}, got {args.tokenizer_revision}"
        )
    if (
        len(args.expected_source_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_source_commit
        )
    ):
        raise RuntimeError("--expected-source-commit must be a full lowercase commit")
    if args.expected_generation_source_commit is not None and (
        len(args.expected_generation_source_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_generation_source_commit
        )
    ):
        raise RuntimeError(
            "--expected-generation-source-commit must be a full lowercase commit"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    contexts = load_tau2_contexts(args.tau2_root)
    audit = prepare(
        split_manifest=args.split_manifest.resolve(),
        generation_manifest=args.generation_manifest.resolve(),
        validation_manifest=args.validation_manifest.resolve(),
        generation_dynamic_audit=args.generation_dynamic_audit.resolve(),
        validation_dynamic_audit=args.validation_dynamic_audit.resolve(),
        expected_source_commit=args.expected_source_commit,
        expected_generation_source_commit=args.expected_generation_source_commit,
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        domain_contexts=contexts,
        seed=args.seed,
        schedule_rows=args.schedule_rows,
        expected_trials=args.expected_trials,
        slots_per_task=args.slots_per_task,
        min_distinct_tasks=args.min_distinct_tasks,
        min_paired_slots=args.min_paired_slots,
        include_optional_mixtures=args.include_optional_mixtures,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
