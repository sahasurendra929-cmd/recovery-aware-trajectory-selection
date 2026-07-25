"""Shared frozen multi-fault fixtures for Stage-1 unit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FAULT_PROTOCOL = "v5_multifamily_readonly_faults_v1"
CLAIM_SCOPE = "multi_fault_family_post_fault_robustness_screen"
TAU2_COMMIT = "c" * 40
SOURCE_FILES = {
    "retail": {
        "tasks_json_sha256": "1" * 64,
        "db_json_sha256": "2" * 64,
        "tools_py_sha256": "3" * 64,
    },
    "airline": {
        "tasks_json_sha256": "4" * 64,
        "db_json_sha256": "5" * 64,
        "tools_py_sha256": "6" * 64,
    },
}
FAMILIES = {
    "retail": (
        {
            "fault_family": "retail_missing_user",
            "tool_name": "lookup_user",
            "invalid_argument_key": "user_id",
        },
        {
            "fault_family": "retail_missing_order",
            "tool_name": "lookup_order",
            "invalid_argument_key": "order_id",
        },
    ),
    "airline": (
        {
            "fault_family": "airline_missing_reservation",
            "tool_name": "lookup_reservation",
            "invalid_argument_key": "reservation_id",
        },
        {
            "fault_family": "airline_missing_flight",
            "tool_name": "lookup_flight",
            "invalid_argument_key": "flight_number",
        },
    ),
}


def fault_protocol() -> dict[str, Any]:
    return {
        "protocol": FAULT_PROTOCOL,
        "claim_scope": CLAIM_SCOPE,
        "family_level_inference": "descriptive_only",
        "guarantees": {
            "tool_type": "READ",
            "expected_tool_error": True,
            "expected_state_mutation": False,
            "per_task_unique_invalid_parameter": True,
            "database_absence_checked_at_pinned_tau2_commit": True,
        },
        "families": {
            domain: [dict(row) for row in rows]
            for domain, rows in FAMILIES.items()
        },
    }


def fault_descriptor(
    domain: str,
    task_id: str | int,
    *,
    source_split: str,
) -> dict[str, Any]:
    text_id = str(task_id)
    try:
        family_index = int(text_id) % len(FAMILIES[domain])
    except ValueError:
        family_index = sum(text_id.encode("utf-8")) % len(FAMILIES[domain])
    specification = FAMILIES[domain][family_index]
    invalid_key = specification["invalid_argument_key"]
    invalid_value = f"missing-{source_split}-{domain}-{text_id}"
    return {
        "inject_error": True,
        "fault_family": specification["fault_family"],
        "tool_name": specification["tool_name"],
        "tool_type": "READ",
        "tool_call_id": f"fault-{source_split}-{domain}-{text_id}",
        "invalid_argument_key": invalid_key,
        "arguments": {invalid_key: invalid_value},
        "expected_tool_error": True,
        "expected_state_mutation": False,
        "on_reference_path": family_index == 0,
        "fault_relevance": (
            "reference_path_or_operation_aligned"
            if family_index == 0
            else "domain_plausible_fallback"
        ),
    }


def multifault_row(
    domain: str,
    task_id: str | int,
    *,
    source_split: str,
) -> dict[str, Any]:
    text_id = str(task_id)
    return {
        "pair_id": f"{domain}:{text_id}",
        "domain": domain,
        "task_id": text_id,
        "source_split": source_split,
        "clean_condition": {"inject_error": False},
        "error_condition": fault_descriptor(
            domain,
            text_id,
            source_split=source_split,
        ),
    }


def multifault_rows(
    task_ids: dict[str, Iterable[str | int]],
    *,
    source_split: str,
) -> list[dict[str, Any]]:
    return [
        multifault_row(domain, task_id, source_split=source_split)
        for domain in ("retail", "airline")
        for task_id in task_ids[domain]
    ]


def multifault_manifest(
    rows: list[dict[str, Any]],
    *,
    protocol: str,
    source_split: str,
    split_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "seed": 20260722,
        "training": False,
        "scientific_claim_allowed": False,
        "source_split": source_split,
        "official_test_used": False,
        "official_test_sealed": True,
        "split_manifest_sha256": split_sha256,
        "tau2_commit": TAU2_COMMIT,
        "source_files": {
            domain: dict(values) for domain, values in SOURCE_FILES.items()
        },
        "fault_protocol": fault_protocol(),
        "paired_task_count": len(rows),
        "domain_counts": {
            domain: sum(row["domain"] == domain for row in rows)
            for domain in ("retail", "airline")
        },
        "rows": rows,
    }


def write_dynamic_audit(
    path: Path,
    *,
    manifest_path: Path,
    split_manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for row in manifest["rows"]:
        token = hashlib.sha256(row["pair_id"].encode("utf-8")).hexdigest()
        rows.append(
            {
                "pair_id": row["pair_id"],
                "runtime_tool_type": "read",
                "runtime_tool_mutates_state": False,
                "tool_error_observed": True,
                "agent_database_before": f"agent-{token}",
                "agent_database_after": f"agent-{token}",
                "agent_database_unchanged": True,
                "user_database_before": f"user-{token}",
                "user_database_after": f"user-{token}",
                "user_database_unchanged": True,
            }
        )
    payload = {
        "protocol": "v5_stage1_dynamic_injection_audit",
        "manifest_protocol": manifest["protocol"],
        "fault_protocol": manifest["fault_protocol"]["protocol"],
        "status": "COMPLETE",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(
            split_manifest_path.read_bytes()
        ).hexdigest(),
        "source_split": manifest["source_split"],
        "official_test_used": False,
        "official_test_sealed": True,
        "verified_injections": len(rows),
        "all_runtime_tools_read_only": True,
        "all_tool_errors_observed": True,
        "all_agent_databases_unchanged": True,
        "all_user_databases_unchanged": True,
        "rows": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload
