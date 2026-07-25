#!/usr/bin/env python3
"""Execute frozen V5 faults and prove that each is a failing READ operation.

The historical Stage-0 smoke manifest remains supported.  For the isolated
Stage-1 multi-fault manifests, this verifier additionally checks the pinned
tau2 commit and source-file hashes, the fault-family catalog, globally unique
invalid parameters, and exact historical split binding before executing every
row.  The environment's agent and user database hashes must remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


FAULT_PROTOCOL = "v5_multifamily_readonly_faults_v1"
DYNAMIC_AUDIT_PROTOCOL = "v5_stage1_dynamic_injection_audit"
SPLIT_PROTOCOL = "v5_stage0_tau2_end_to_end"
MULTIFAULT_PROTOCOLS = {
    "v5_stage1_multifault_data_construction",
    "v5_stage1_sft_causal_validation",
}
MULTIFAULT_SPLITS = {
    "v5_stage1_multifault_data_construction": (
        "derived_inner_train",
        "inner_train_ids",
        83,
    ),
    "v5_stage1_sft_causal_validation": (
        "derived_validation",
        "validation_ids",
        21,
    ),
}
PUBLISHED_STAGE0_SPLIT_SHA256 = (
    "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
)
DOMAINS = ("retail", "airline")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_tracked_checkout(path: Path) -> None:
    for command, label in (
        (["git", "-C", str(path), "diff", "--quiet", "--"], "unstaged"),
        (
            ["git", "-C", str(path), "diff", "--cached", "--quiet", "--"],
            "staged",
        ),
    ):
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"tau2 checkout contains {label} tracked drift")


def configure_tau2_path(tau2_root: Path) -> None:
    source = tau2_root.resolve() / "src"
    if not source.exists():
        raise FileNotFoundError(f"tau2 source directory not found: {source}")
    sys.path.insert(0, str(source))


def validate_source_provenance(
    *,
    tau2_root: Path,
    manifest: dict[str, Any],
    split_manifest: Path | None,
) -> None:
    declared_commit = manifest.get("tau2_commit")
    if (
        not isinstance(declared_commit, str)
        or COMMIT_RE.fullmatch(declared_commit) is None
        or git_commit(tau2_root) != declared_commit
    ):
        raise RuntimeError("local tau2 commit differs from multi-fault manifest")
    if split_manifest is None:
        raise RuntimeError(
            "multi-fault verification requires --split-manifest"
        )
    split_sha = sha256_file(split_manifest)
    if split_sha != PUBLISHED_STAGE0_SPLIT_SHA256:
        raise RuntimeError("historical Stage-0 split SHA drift")
    if manifest.get("split_manifest_sha256") != split_sha:
        raise RuntimeError("multi-fault manifest is not bound to this split")
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    if not isinstance(split, dict) or split.get("protocol") != SPLIT_PROTOCOL:
        raise RuntimeError("historical split protocol drift")
    guarantees = split.get("guarantees")
    if (
        not isinstance(guarantees, dict)
        or guarantees.get("validation_derived_from_official_train_only")
        is not True
        or guarantees.get("official_test_task_content_exported") is not False
        or guarantees.get("official_test_used_for_model_or_smoke_selection")
        is not False
    ):
        raise RuntimeError("historical split safety guarantees drift")

    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != set(DOMAINS):
        raise RuntimeError("multi-fault manifest lacks source-file provenance")
    for domain in DOMAINS:
        expected = source_files.get(domain)
        if not isinstance(expected, dict):
            raise RuntimeError(f"{domain}: malformed source-file provenance")
        observed = {
            "tasks_json_sha256": sha256_file(
                tau2_root / "data" / "tau2" / "domains" / domain / "tasks.json"
            ),
            "db_json_sha256": sha256_file(
                tau2_root / "data" / "tau2" / "domains" / domain / "db.json"
            ),
            "tools_py_sha256": sha256_file(
                tau2_root
                / "src"
                / "tau2"
                / "domains"
                / domain
                / "tools.py"
            ),
        }
        if any(
            not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
            for value in expected.values()
        ) or observed != expected:
            raise RuntimeError(f"{domain}: local tau2 source bytes drift")


def validate_multifault_manifest(
    *,
    tau2_root: Path,
    manifest: dict[str, Any],
    split_manifest: Path | None,
) -> None:
    protocol = manifest.get("protocol")
    if protocol not in MULTIFAULT_PROTOCOLS:
        raise RuntimeError("unsupported Stage-1 multi-fault manifest protocol")
    validate_source_provenance(
        tau2_root=tau2_root,
        manifest=manifest,
        split_manifest=split_manifest,
    )
    assert split_manifest is not None
    expected_source_split, split_field, expected_count = MULTIFAULT_SPLITS[protocol]
    if manifest.get("source_split") != expected_source_split:
        raise RuntimeError("multi-fault manifest source_split drift")
    if manifest.get("official_test_used") is not False:
        raise RuntimeError("multi-fault manifest opened official test")
    if manifest.get("official_test_sealed") is not True:
        raise RuntimeError("multi-fault manifest lacks official-test seal")
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    domains = split.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(DOMAINS):
        raise RuntimeError("historical split domains drift")
    expected_ids: set[str] = set()
    sealed_ids: set[str] = set()
    for domain in DOMAINS:
        domain_split = domains.get(domain)
        if not isinstance(domain_split, dict):
            raise RuntimeError(f"historical split lacks {domain}")
        source_ids = domain_split.get(split_field)
        sealed = domain_split.get("sealed_test_ids")
        if (
            not isinstance(source_ids, list)
            or not isinstance(sealed, list)
            or any(not isinstance(value, (str, int)) for value in source_ids)
            or any(not isinstance(value, (str, int)) for value in sealed)
        ):
            raise RuntimeError(f"historical split {domain} task IDs drift")
        expected_ids.update(f"{domain}:{value}" for value in source_ids)
        sealed_ids.update(f"{domain}:{value}" for value in sealed)
    if len(expected_ids) != expected_count:
        raise RuntimeError("historical split expected task count drift")
    fault_protocol = manifest.get("fault_protocol")
    if (
        not isinstance(fault_protocol, dict)
        or fault_protocol.get("protocol") != FAULT_PROTOCOL
        or fault_protocol.get("claim_scope")
        != "multi_fault_family_post_fault_robustness_screen"
        or fault_protocol.get("family_level_inference") != "descriptive_only"
    ):
        raise RuntimeError("multi-fault protocol metadata drift")
    guarantees = fault_protocol.get("guarantees")
    required_guarantees = {
        "tool_type": "READ",
        "expected_tool_error": True,
        "expected_state_mutation": False,
        "per_task_unique_invalid_parameter": True,
        "database_absence_checked_at_pinned_tau2_commit": True,
    }
    if not isinstance(guarantees, dict) or any(
        guarantees.get(field) != value
        for field, value in required_guarantees.items()
    ):
        raise RuntimeError("multi-fault safety guarantees drift")

    raw_catalog = fault_protocol.get("families")
    if not isinstance(raw_catalog, dict) or set(raw_catalog) != set(DOMAINS):
        raise RuntimeError("multi-fault family catalog domains drift")
    catalog: dict[str, dict[str, tuple[str, str]]] = {}
    for domain in DOMAINS:
        family_rows = raw_catalog.get(domain)
        if not isinstance(family_rows, list) or len(family_rows) < 2:
            raise RuntimeError(f"{domain}: insufficient fault families")
        catalog[domain] = {}
        for row in family_rows:
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
                raise RuntimeError(f"{domain}: duplicate fault family")
            catalog[domain][family] = (tool_name, invalid_key)

    rows = manifest.get("rows")
    if (
        not isinstance(rows, list)
        or manifest.get("paired_task_count") != len(rows)
        or not rows
    ):
        raise RuntimeError("multi-fault manifest row count drift")
    invalid_parameters: set[str] = set()
    call_ids: set[str] = set()
    pair_ids: set[str] = set()
    families = {domain: set() for domain in DOMAINS}
    tools = {domain: set() for domain in DOMAINS}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("multi-fault row must be an object")
        domain = row.get("domain")
        task_id = str(row.get("task_id"))
        pair_id = f"{domain}:{task_id}"
        if (
            domain not in DOMAINS
            or row.get("pair_id") != pair_id
            or pair_id in pair_ids
        ):
            raise RuntimeError("multi-fault task identity drift")
        pair_ids.add(pair_id)
        error = row.get("error_condition")
        if not isinstance(error, dict):
            raise RuntimeError(f"{pair_id}: missing error condition")
        family = error.get("fault_family")
        specification = catalog[domain].get(family)
        if specification is None:
            raise RuntimeError(f"{pair_id}: unknown fault family")
        tool_name, invalid_key = specification
        if (
            error.get("inject_error") is not True
            or error.get("tool_name") != tool_name
            or error.get("tool_type") != "READ"
            or error.get("invalid_argument_key") != invalid_key
            or error.get("expected_tool_error") is not True
            or error.get("expected_state_mutation") is not False
        ):
            raise RuntimeError(f"{pair_id}: fault descriptor drift")
        arguments = error.get("arguments")
        if not isinstance(arguments, dict) or not isinstance(
            arguments.get(invalid_key), str
        ):
            raise RuntimeError(f"{pair_id}: invalid parameter missing")
        invalid_identity = canonical({invalid_key: arguments[invalid_key]})
        if invalid_identity in invalid_parameters:
            raise RuntimeError("invalid fault parameter reused across tasks")
        invalid_parameters.add(invalid_identity)
        call_id = error.get("tool_call_id")
        if (
            not isinstance(call_id, str)
            or not call_id
            or call_id in call_ids
        ):
            raise RuntimeError(f"{pair_id}: duplicate/invalid tool call ID")
        call_ids.add(call_id)
        if error.get("fault_relevance") not in {
            "reference_path_or_operation_aligned",
            "domain_plausible_fallback",
        } or type(error.get("on_reference_path")) is not bool:
            raise RuntimeError(f"{pair_id}: fault relevance metadata drift")
        families[domain].add(family)
        tools[domain].add(tool_name)
    leaked = pair_ids & sealed_ids
    if leaked:
        raise RuntimeError(
            f"official-test task leaked into multi-fault manifest: {sorted(leaked)[:5]}"
        )
    if pair_ids != expected_ids or len(rows) != expected_count:
        raise RuntimeError(
            "multi-fault manifest task coverage drift; "
            f"missing={sorted(expected_ids - pair_ids)[:5]}, "
            f"extra={sorted(pair_ids - expected_ids)[:5]}"
        )
    for domain in DOMAINS:
        if len(families[domain]) < 2 or len(tools[domain]) < 2:
            raise RuntimeError(f"{domain}: insufficient multi-fault coverage")


def verify(
    tau2_root: Path,
    manifest_path: Path,
    output_path: Path,
    split_manifest: Path | None = None,
) -> dict[str, Any]:
    require_clean_tracked_checkout(tau2_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("manifest must be a JSON object")
    is_multifault = manifest.get("fault_protocol") is not None
    if is_multifault:
        validate_multifault_manifest(
            tau2_root=tau2_root,
            manifest=manifest,
            split_manifest=split_manifest,
        )

    configure_tau2_path(tau2_root)
    try:
        from tau2.data_model.message import ToolCall
        from tau2.environment.toolkit import ToolType
        from tau2.registry import registry
        from tau2.runner.build import build_environment

        task_cache: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        for expected in manifest["rows"]:
            domain = expected["domain"]
            if domain not in task_cache:
                task_cache[domain] = {
                    str(task.id): task
                    for task in registry.get_tasks_loader(domain)(None)
                }
            task = task_cache[domain][str(expected["task_id"])]
            environment = build_environment(domain)
            state = task.initial_state
            environment.set_state(
                state.initialization_data if state else None,
                state.initialization_actions if state else None,
                deepcopy(state.message_history)
                if state and state.message_history
                else [],
            )
            injection = expected["error_condition"]
            tool_name = injection["tool_name"]
            tool_type = environment.tools.tool_type(tool_name)
            mutates_state = environment.tools.tool_mutates_state(tool_name)
            db_before = environment.get_db_hash()
            user_db_before = environment.get_user_db_hash()
            response = environment.get_response(
                ToolCall(
                    id=injection["tool_call_id"],
                    name=tool_name,
                    arguments=injection["arguments"],
                    requestor="assistant",
                )
            )
            db_after = environment.get_db_hash()
            user_db_after = environment.get_user_db_hash()
            rows.append(
                {
                    "pair_id": expected["pair_id"],
                    "fault_family": injection.get("fault_family"),
                    "fault_relevance": injection.get("fault_relevance"),
                    "tool_name": tool_name,
                    "arguments_sha256": hashlib.sha256(
                        canonical(injection["arguments"]).encode("utf-8")
                    ).hexdigest(),
                    "runtime_tool_type": tool_type.value,
                    "runtime_tool_mutates_state": mutates_state,
                    "tool_error_observed": response.error,
                    "agent_database_before": db_before,
                    "agent_database_after": db_after,
                    "agent_database_unchanged": db_before == db_after,
                    "user_database_before": user_db_before,
                    "user_database_after": user_db_after,
                    "user_database_unchanged": user_db_before == user_db_after,
                    "tool_response": response.content,
                }
            )
            if tool_type != ToolType.READ or mutates_state:
                raise RuntimeError(
                    f"{expected['pair_id']}: injected tool is not non-mutating READ"
                )
    finally:
        source = str(tau2_root.resolve() / "src")
        if sys.path and sys.path[0] == source:
            sys.path.pop(0)

    passed = (
        len(rows) == manifest["paired_task_count"]
        and all(row["runtime_tool_type"] == "read" for row in rows)
        and all(not row["runtime_tool_mutates_state"] for row in rows)
        and all(row["tool_error_observed"] for row in rows)
        and all(row["agent_database_unchanged"] for row in rows)
        and all(row["user_database_unchanged"] for row in rows)
    )
    result = {
        "protocol": (
            DYNAMIC_AUDIT_PROTOCOL if is_multifault else manifest["protocol"]
        ),
        "manifest_protocol": manifest["protocol"],
        "fault_protocol": (
            manifest.get("fault_protocol", {}).get("protocol")
            if is_multifault
            else None
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "split_manifest_sha256": (
            sha256_file(split_manifest) if split_manifest is not None else None
        ),
        "status": (
            "COMPLETE"
            if is_multifault and passed
            else ("PASS" if passed else "FAIL")
        ),
        "source_split": manifest.get("source_split"),
        "official_test_used": False if is_multifault else None,
        "official_test_sealed": True if is_multifault else None,
        "verified_injections": len(rows),
        "all_runtime_tools_read_only": all(
            row["runtime_tool_type"] == "read"
            and not row["runtime_tool_mutates_state"]
            for row in rows
        ),
        "all_tool_errors_observed": all(
            row["tool_error_observed"] for row in rows
        ),
        "all_agent_databases_unchanged": all(
            row["agent_database_unchanged"] for row in rows
        ),
        "all_user_databases_unchanged": all(
            row["user_database_unchanged"] for row in rows
        ),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(
        args.tau2_root.resolve(),
        args.manifest.resolve(),
        args.output.resolve(),
        (
            args.split_manifest.resolve()
            if args.split_manifest is not None
            else None
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] not in {"PASS", "COMPLETE"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
