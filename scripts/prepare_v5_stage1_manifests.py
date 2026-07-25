#!/usr/bin/env python3
"""Build isolated multi-fault Stage-1 train/validation manifests.

The historical Stage-0 split is read-only input.  This script never rewrites
it; both new manifests bind its exact SHA-256 and keep official-test IDs sealed.
Every task receives one deterministic, database-absent parameter for a pinned
read-only tau2 tool.  Family-level results remain descriptive because each task
is assigned only one family.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SEED = 20260722
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
PUBLISHED_STAGE0_SPLIT_SHA256 = (
    "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
)
SPLIT_PROTOCOL = "v5_stage0_tau2_end_to_end"
GENERATION_PROTOCOL = "v5_stage1_multifault_data_construction"
VALIDATION_PROTOCOL = "v5_stage1_sft_causal_validation"
FAULT_PROTOCOL = "v5_multifamily_readonly_faults_v1"
DOMAINS = ("retail", "airline")
EXPECTED_COUNTS = {
    "retail": {"inner_train": 59, "validation": 15, "sealed_test": 40},
    "airline": {"inner_train": 24, "validation": 6, "sealed_test": 20},
}
FAULT_FAMILIES = {
    "retail": (
        {
            "fault_family": "retail_missing_user_email",
            "tool_name": "find_user_id_by_email",
            "invalid_argument_key": "email",
            "database_index": "user_emails",
        },
        {
            "fault_family": "retail_missing_user_id",
            "tool_name": "get_user_details",
            "invalid_argument_key": "user_id",
            "database_index": "users",
        },
        {
            "fault_family": "retail_missing_order_id",
            "tool_name": "get_order_details",
            "invalid_argument_key": "order_id",
            "database_index": "orders",
        },
        {
            "fault_family": "retail_missing_product_id",
            "tool_name": "get_product_details",
            "invalid_argument_key": "product_id",
            "database_index": "products",
        },
    ),
    "airline": (
        {
            "fault_family": "airline_missing_reservation_id",
            "tool_name": "get_reservation_details",
            "invalid_argument_key": "reservation_id",
            "database_index": "reservations",
        },
        {
            "fault_family": "airline_missing_user_id",
            "tool_name": "get_user_details",
            "invalid_argument_key": "user_id",
            "database_index": "users",
        },
        {
            "fault_family": "airline_missing_flight_number",
            "tool_name": "get_flight_status",
            "invalid_argument_key": "flight_number",
            "database_index": "flights",
        },
    ),
}

PINNED_TAU2_SOURCE_PATHS = tuple(
    path
    for domain in DOMAINS
    for path in (
        f"data/tau2/domains/{domain}/tasks.json",
        f"data/tau2/domains/{domain}/db.json",
        f"src/tau2/domains/{domain}/tools.py",
    )
)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_clean_pinned_tau2_sources(tau2_root: Path) -> None:
    """Reject staged/unstaged drift in every file hashed into the protocol."""

    subprocess.run(
        [
            "git",
            "-C",
            str(tau2_root),
            "ls-files",
            "--error-unmatch",
            *PINNED_TAU2_SOURCE_PATHS,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for extra in ((), ("--cached",)):
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(tau2_root),
                "diff",
                *extra,
                "--quiet",
                "HEAD",
                "--",
                *PINNED_TAU2_SOURCE_PATHS,
            ],
            check=False,
        )
        if completed.returncode != 0:
            kind = "staged" if extra else "unstaged"
            raise RuntimeError(
                f"pinned tau2 source files contain {kind} byte drift"
            )


def deterministic_order(
    values: list[str], *, seed: int, salt: str
) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            sha256_bytes(f"{seed}:{salt}:{value}".encode("utf-8")),
            value,
        ),
    )


def fault_protocol_manifest() -> dict[str, Any]:
    return {
        "protocol": FAULT_PROTOCOL,
        "claim_scope": "multi_fault_family_post_fault_robustness_screen",
        "family_level_inference": "descriptive_only",
        "guarantees": {
            "tool_type": "READ",
            "expected_tool_error": True,
            "expected_state_mutation": False,
            "per_task_unique_invalid_parameter": True,
            "database_absence_checked_at_pinned_tau2_commit": True,
        },
        "families": {
            domain: [
                {
                    "fault_family": row["fault_family"],
                    "tool_name": row["tool_name"],
                    "invalid_argument_key": row["invalid_argument_key"],
                }
                for row in FAULT_FAMILIES[domain]
            ]
            for domain in DOMAINS
        },
    }


def validate_read_only_source(tau2_root: Path) -> None:
    for domain in DOMAINS:
        source_path = (
            tau2_root / "src" / "tau2" / "domains" / domain / "tools.py"
        )
        source = source_path.read_text(encoding="utf-8")
        for specification in FAULT_FAMILIES[domain]:
            tool_name = specification["tool_name"]
            pattern = (
                r"@is_tool\(ToolType\.READ\)\s+"
                rf"def\s+{re.escape(tool_name)}\s*\("
            )
            if re.search(pattern, source) is None:
                raise RuntimeError(
                    f"{domain}.{tool_name} is not pinned as ToolType.READ"
                )


def _candidate_invalid_value(
    *,
    family: str,
    seed: int,
    phase: str,
    domain: str,
    task_id: str,
    nonce: int,
) -> str:
    digest = sha256_bytes(
        f"{seed}:{phase}:{domain}:{task_id}:{family}:{nonce}".encode("utf-8")
    )
    number = int(digest[:16], 16)
    if family == "retail_missing_user_email":
        return f"user{digest[:12]}@example.com"
    if family in {"retail_missing_user_id", "airline_missing_user_id"}:
        return f"user_{digest[:16]}"
    if family == "retail_missing_order_id":
        return f"#W{number % 10_000_000:07d}"
    if family == "retail_missing_product_id":
        return f"{number % 10_000_000_000:010d}"
    if family == "airline_missing_reservation_id":
        return digest[:6].upper()
    if family == "airline_missing_flight_number":
        return f"HAT{301 + number % 699:03d}"
    raise RuntimeError(f"unsupported fault family: {family}")


def _database_values(
    specification: dict[str, str], database: dict[str, Any]
) -> set[str]:
    index = specification["database_index"]
    if index == "user_emails":
        return {
            str(user["email"]).lower()
            for user in database.get("users", {}).values()
        }
    values = database.get(index)
    if not isinstance(values, dict):
        raise RuntimeError(f"database lacks {index}")
    return {str(value) for value in values}


def task_reference_tools(task: dict[str, Any]) -> set[str]:
    criteria = task.get("evaluation_criteria") or {}
    return {
        str(action.get("name"))
        for action in criteria.get("actions") or []
        if action.get("requestor", "assistant") == "assistant"
        and isinstance(action.get("name"), str)
    }


def operation_aligned_tools(
    domain: str, reference_tools: set[str]
) -> set[str]:
    aligned = set(reference_tools)
    if domain == "retail":
        if any(
            token in tool
            for tool in reference_tools
            for token in ("order", "return", "exchange", "cancel")
        ):
            aligned.add("get_order_details")
        if any("user" in tool or "address" in tool for tool in reference_tools):
            aligned.update(
                {"get_user_details", "find_user_id_by_email"}
            )
        if any(
            token in tool
            for tool in reference_tools
            for token in ("product", "item", "exchange", "return")
        ):
            aligned.add("get_product_details")
    else:
        if any("reservation" in tool for tool in reference_tools):
            aligned.add("get_reservation_details")
        if any(
            token in tool
            for tool in reference_tools
            for token in ("flight", "book")
        ):
            aligned.add("get_flight_status")
        if any("user" in tool for tool in reference_tools):
            aligned.add("get_user_details")
    return aligned


def choose_family(
    *,
    specifications: tuple[dict[str, str], ...],
    aligned_tools: set[str],
    counts: Counter,
    seed: int,
    phase: str,
    domain: str,
    task_id: str,
) -> dict[str, str]:
    relevant = [
        row for row in specifications if row["tool_name"] in aligned_tools
    ]
    candidates = relevant or list(specifications)
    return min(
        candidates,
        key=lambda row: (
            counts[row["fault_family"]],
            sha256_bytes(
                (
                    f"{seed}:{phase}:{domain}:{task_id}:"
                    f"{row['fault_family']}"
                ).encode("utf-8")
            ),
            row["fault_family"],
        ),
    )


def build_fault_rows(
    *,
    domain: str,
    task_ids: list[str],
    phase: str,
    tasks: dict[str, dict[str, Any]],
    database: dict[str, Any],
    seed: int,
    used_parameters: set[str],
) -> list[dict[str, Any]]:
    specifications = FAULT_FAMILIES[domain]
    family_counts: Counter = Counter()
    rows: list[dict[str, Any]] = []
    ordered = deterministic_order(
        task_ids,
        seed=seed,
        salt=f"{phase}:{domain}:fault-assignment",
    )
    for task_id in ordered:
        reference_tools = task_reference_tools(tasks[task_id])
        aligned_tools = operation_aligned_tools(domain, reference_tools)
        specification = choose_family(
            specifications=specifications,
            aligned_tools=aligned_tools,
            counts=family_counts,
            seed=seed,
            phase=phase,
            domain=domain,
            task_id=task_id,
        )
        family = specification["fault_family"]
        invalid_key = specification["invalid_argument_key"]
        existing = _database_values(specification, database)
        nonce = 0
        while True:
            invalid_value = _candidate_invalid_value(
                family=family,
                seed=seed,
                phase=phase,
                domain=domain,
                task_id=task_id,
                nonce=nonce,
            )
            lookup = (
                invalid_value.lower()
                if specification["database_index"] == "user_emails"
                else invalid_value
            )
            identity = canonical({invalid_key: invalid_value})
            if lookup not in existing and identity not in used_parameters:
                used_parameters.add(identity)
                break
            nonce += 1
            if nonce > 100:
                raise RuntimeError(
                    f"could not construct absent parameter for {domain}:{task_id}"
                )
        arguments = {invalid_key: invalid_value}
        if family == "airline_missing_flight_number":
            arguments["date"] = "2026-08-01"
        rows.append(
            {
                "pair_id": f"{domain}:{task_id}",
                "domain": domain,
                "task_id": task_id,
                "source_split": phase,
                "clean_condition": {"inject_error": False},
                "error_condition": {
                    "inject_error": True,
                    "fault_family": family,
                    "tool_name": specification["tool_name"],
                    "tool_type": "READ",
                    "tool_call_id": (
                        f"v5-stage1-multifault-{phase}-{domain}-{task_id}"
                    ),
                    "invalid_argument_key": invalid_key,
                    "arguments": arguments,
                    "expected_tool_error": True,
                    "expected_state_mutation": False,
                    "on_reference_path": (
                        specification["tool_name"] in reference_tools
                    ),
                    "fault_relevance": (
                        "reference_path_or_operation_aligned"
                        if specification["tool_name"] in aligned_tools
                        else "domain_plausible_fallback"
                    ),
                },
            }
        )
        family_counts[family] += 1
    families = {row["error_condition"]["fault_family"] for row in rows}
    tools = {row["error_condition"]["tool_name"] for row in rows}
    if len(rows) != len(task_ids) or len(families) < 2 or len(tools) < 2:
        raise RuntimeError(f"{phase}/{domain} lacks multi-fault coverage")
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(rows),
        "fault_family_counts": dict(
            sorted(
                Counter(
                    row["error_condition"]["fault_family"] for row in rows
                ).items()
            )
        ),
        "tool_counts": dict(
            sorted(
                Counter(
                    row["error_condition"]["tool_name"] for row in rows
                ).items()
            )
        ),
        "on_reference_path": sum(
            row["error_condition"]["on_reference_path"] for row in rows
        ),
        "fault_relevance_counts": dict(
            sorted(
                Counter(
                    row["error_condition"]["fault_relevance"]
                    for row in rows
                ).items()
            )
        ),
    }


def prepare(
    *,
    tau2_root: Path,
    split_manifest_path: Path,
    output_dir: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise RuntimeError(f"output directory must be absent or empty: {output_dir}")
    observed_commit = git_commit(tau2_root)
    if observed_commit != TAU2_COMMIT:
        raise RuntimeError(
            f"tau2 commit drift: expected {TAU2_COMMIT}, got {observed_commit}"
        )
    split = load_json(split_manifest_path)
    split_sha = sha256_file(split_manifest_path)
    if split_sha != PUBLISHED_STAGE0_SPLIT_SHA256:
        raise RuntimeError(
            "historical Stage-0 split SHA drift: "
            f"expected {PUBLISHED_STAGE0_SPLIT_SHA256}, got {split_sha}"
        )
    if split.get("protocol") != SPLIT_PROTOCOL:
        raise RuntimeError("unexpected historical split protocol")
    guarantees = split.get("guarantees") or {}
    if guarantees.get("validation_derived_from_official_train_only") is not True:
        raise RuntimeError("historical split lacks validation provenance")
    if guarantees.get("official_test_task_content_exported") is not False:
        raise RuntimeError("historical split does not seal official test")
    require_clean_pinned_tau2_sources(tau2_root)
    validate_read_only_source(tau2_root)

    domains_root = tau2_root / "data" / "tau2" / "domains"
    generation_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    used_parameters: set[str] = set()
    source_files: dict[str, dict[str, str]] = {}
    for domain in DOMAINS:
        split_domain = split["domains"][domain]
        inner = [str(value) for value in split_domain["inner_train_ids"]]
        validation = [str(value) for value in split_domain["validation_ids"]]
        sealed = [str(value) for value in split_domain["sealed_test_ids"]]
        counts = EXPECTED_COUNTS[domain]
        if (
            len(inner),
            len(validation),
            len(sealed),
        ) != (
            counts["inner_train"],
            counts["validation"],
            counts["sealed_test"],
        ):
            raise RuntimeError(f"{domain}: historical split count drift")
        if set(inner) & set(validation) or set(inner) & set(sealed):
            raise RuntimeError(f"{domain}: historical split overlap")
        tasks_path = domains_root / domain / "tasks.json"
        database_path = domains_root / domain / "db.json"
        tasks = {
            str(task["id"]): task for task in load_json(tasks_path)
        }
        database = load_json(database_path)
        if not set(inner) | set(validation) <= set(tasks):
            raise RuntimeError(f"{domain}: task IDs missing from pinned tasks")
        source_files[domain] = {
            "tasks_json_sha256": sha256_file(tasks_path),
            "db_json_sha256": sha256_file(database_path),
            "tools_py_sha256": sha256_file(
                tau2_root / "src" / "tau2" / "domains" / domain / "tools.py"
            ),
        }
        generation_rows.extend(
            build_fault_rows(
                domain=domain,
                task_ids=inner,
                phase="derived_inner_train",
                tasks=tasks,
                database=database,
                seed=seed,
                used_parameters=used_parameters,
            )
        )
        validation_rows.extend(
            build_fault_rows(
                domain=domain,
                task_ids=validation,
                phase="derived_validation",
                tasks=tasks,
                database=database,
                seed=seed,
                used_parameters=used_parameters,
            )
        )
    if len(generation_rows) != 83 or len(validation_rows) != 21:
        raise RuntimeError("Stage-1 manifest task counts drift")

    fault_protocol = fault_protocol_manifest()
    common = {
        "seed": seed,
        "training": False,
        "scientific_claim_allowed": False,
        "split_manifest_sha256": split_sha,
        "tau2_commit": observed_commit,
        "source_files": source_files,
        "fault_protocol": fault_protocol,
    }
    generation_manifest = {
        "protocol": GENERATION_PROTOCOL,
        **common,
        "source_split": "derived_inner_train",
        "official_test_used": False,
        "official_test_sealed": True,
        "paired_task_count": len(generation_rows),
        "domain_counts": {
            domain: sum(row["domain"] == domain for row in generation_rows)
            for domain in DOMAINS
        },
        "rows": generation_rows,
    }
    validation_manifest = {
        "protocol": VALIDATION_PROTOCOL,
        **common,
        "source_split": "derived_validation",
        "official_test_used": False,
        "official_test_sealed": True,
        "paired_task_count": len(validation_rows),
        "domain_counts": {
            domain: sum(row["domain"] == domain for row in validation_rows)
            for domain in DOMAINS
        },
        "rows": validation_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = output_dir / "generation_manifest.json"
    validation_path = output_dir / "validation_manifest.json"
    generation_path.write_text(
        json.dumps(generation_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(validation_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audit = {
        "status": "PASS",
        "protocol": FAULT_PROTOCOL,
        "claim_scope": fault_protocol["claim_scope"],
        "family_level_inference": "descriptive_only",
        "historical_split_sha256": split_sha,
        "official_test_used": False,
        "invalid_parameter_count": len(used_parameters),
        "invalid_parameters_unique": len(used_parameters) == 104,
        "generation": {
            domain: summarize_rows(
                [row for row in generation_rows if row["domain"] == domain]
            )
            for domain in DOMAINS
        },
        "validation": {
            domain: summarize_rows(
                [row for row in validation_rows if row["domain"] == domain]
            )
            for domain in DOMAINS
        },
        "generation_manifest_sha256": sha256_file(generation_path),
        "validation_manifest_sha256": sha256_file(validation_path),
    }
    if audit["invalid_parameters_unique"] is not True:
        raise RuntimeError("Stage-1 invalid parameters are not globally unique")
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    hashes = {
        "generation_manifest.json": sha256_file(generation_path),
        "validation_manifest.json": sha256_file(validation_path),
        "audit.json": sha256_file(audit_path),
    }
    (output_dir / "hashes.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        raise RuntimeError(f"Stage-1 seed is frozen at {SEED}")
    result = prepare(
        tau2_root=args.tau2_root.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
