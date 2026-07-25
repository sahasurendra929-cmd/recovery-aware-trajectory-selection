#!/usr/bin/env python3
"""Fail-closed validation for V5 Stage-1 dynamic injection audits."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


DYNAMIC_AUDIT_PROTOCOL = "v5_stage1_dynamic_injection_audit"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_complete_dynamic_audit(
    path: Path,
    *,
    manifest_path: Path,
    split_manifest_path: Path,
    expected_source_split: str,
    expected_task_ids: set[str],
) -> dict[str, Any]:
    """Validate a dynamic audit and return its immutable identity fields."""

    if not path.is_file():
        raise RuntimeError(f"dynamic audit file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("dynamic audit must be a JSON object")
    if payload.get("protocol") != DYNAMIC_AUDIT_PROTOCOL:
        raise RuntimeError("dynamic audit protocol drift")
    if payload.get("status") != "COMPLETE":
        raise RuntimeError("dynamic audit is not COMPLETE")

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise RuntimeError("dynamic-audit manifest must be a JSON object")
    manifest_sha = sha256_file(manifest_path)
    split_sha = sha256_file(split_manifest_path)
    if payload.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("dynamic audit manifest SHA drift")
    if payload.get("manifest_protocol") != manifest_payload.get("protocol"):
        raise RuntimeError("dynamic audit manifest protocol drift")
    fault_protocol = manifest_payload.get("fault_protocol")
    if (
        not isinstance(fault_protocol, dict)
        or payload.get("fault_protocol") != fault_protocol.get("protocol")
    ):
        raise RuntimeError("dynamic audit fault protocol drift")
    if payload.get("split_manifest_sha256") != split_sha:
        raise RuntimeError("dynamic audit split SHA drift")
    if payload.get("source_split") != expected_source_split:
        raise RuntimeError("dynamic audit source split drift")
    if payload.get("official_test_used") is not False:
        raise RuntimeError("dynamic audit opened official test")
    if payload.get("official_test_sealed") is not True:
        raise RuntimeError("dynamic audit lacks official-test seal")

    rows = payload.get("rows")
    if (
        not isinstance(rows, list)
        or payload.get("verified_injections") != len(rows)
        or len(rows) != len(expected_task_ids)
    ):
        raise RuntimeError("dynamic audit row count drift")
    observed_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"dynamic audit row {index} is not an object")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in observed_ids:
            raise RuntimeError("dynamic audit has invalid/duplicate pair IDs")
        observed_ids.add(pair_id)
        if (
            row.get("runtime_tool_type") != "read"
            or row.get("runtime_tool_mutates_state") is not False
            or row.get("tool_error_observed") is not True
            or row.get("agent_database_before") is None
            or row.get("agent_database_after") is None
            or row.get("agent_database_unchanged") is not True
            or row.get("user_database_before") is None
            or row.get("user_database_after") is None
            or row.get("user_database_unchanged") is not True
            or row.get("agent_database_before")
            != row.get("agent_database_after")
            or row.get("user_database_before")
            != row.get("user_database_after")
        ):
            raise RuntimeError(f"{pair_id}: dynamic audit safety evidence failed")
    if observed_ids != expected_task_ids:
        raise RuntimeError(
            "dynamic audit task coverage drift; "
            f"missing={sorted(expected_task_ids - observed_ids)[:5]}, "
            f"extra={sorted(observed_ids - expected_task_ids)[:5]}"
        )

    required_aggregate_flags = (
        "all_runtime_tools_read_only",
        "all_tool_errors_observed",
        "all_agent_databases_unchanged",
        "all_user_databases_unchanged",
    )
    if any(payload.get(field) is not True for field in required_aggregate_flags):
        raise RuntimeError("dynamic audit aggregate safety evidence failed")

    audit_sha = sha256_file(path)
    if SHA256_RE.fullmatch(audit_sha) is None:
        raise RuntimeError("dynamic audit SHA-256 is invalid")
    return {
        "protocol": DYNAMIC_AUDIT_PROTOCOL,
        "sha256": audit_sha,
        "manifest_sha256": manifest_sha,
        "split_manifest_sha256": split_sha,
        "source_split": expected_source_split,
        "verified_injections": len(rows),
        "official_test_used": False,
        "official_test_sealed": True,
    }
