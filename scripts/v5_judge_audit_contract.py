#!/usr/bin/env python3
"""Bind strict natural-language judge evidence to exact tau2 simulations.

The strict judge writes one audit JSON file in tau2's per-simulation
``llm_debug`` directory for every nonempty natural-language assertion set.
This module reconstructs that expected call set from result JSON bytes and
fails closed unless there is exactly one valid audit for every such
simulation, with no missing, duplicate, misplaced, or extra audit file.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence


STRICT_JUDGE_PROTOCOL = "v5_strict_nl_judge_v1"
EVIDENCE_PROTOCOL = "v5_strict_nl_judge_evidence_v1"
AUDIT_GLOB = "strict_nl_judge_audit_*.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StrictJudgeEvidenceError(RuntimeError):
    """Strict-judge evidence is missing, ambiguous, malformed, or unbound."""


def canonical_json(value: Any) -> str:
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


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictJudgeEvidenceError(
            f"{label} is not readable JSON: {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise StrictJudgeEvidenceError(f"{label} must be a JSON object: {path}")
    return payload


def _safe_path_component(value: Any, *, label: str, where: Path) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise StrictJudgeEvidenceError(f"{where}: simulation {label} is absent")
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or "\x00" in text
    ):
        raise StrictJudgeEvidenceError(
            f"{where}: simulation {label} is not a safe path component"
        )
    return text


def _expected_outcomes(
    simulation: dict[str, Any],
    *,
    result_path: Path,
    simulation_index: int,
) -> list[str]:
    reward_info = simulation.get("reward_info")
    if reward_info is None:
        return []
    if not isinstance(reward_info, dict):
        raise StrictJudgeEvidenceError(
            f"{result_path}: simulation[{simulation_index}] reward_info "
            "must be an object or null"
        )
    rows = reward_info.get("nl_assertions")
    if rows is None or rows == []:
        return []
    if not isinstance(rows, list):
        raise StrictJudgeEvidenceError(
            f"{result_path}: simulation[{simulation_index}] nl_assertions "
            "must be an array or null"
        )
    outcomes: list[str] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise StrictJudgeEvidenceError(
                f"{result_path}: simulation[{simulation_index}] "
                f"nl_assertions[{row_index}] must be an object"
            )
        outcome = row.get("nl_assertion")
        if not isinstance(outcome, str) or not outcome:
            raise StrictJudgeEvidenceError(
                f"{result_path}: simulation[{simulation_index}] "
                f"nl_assertions[{row_index}] lacks nl_assertion"
            )
        outcomes.append(outcome)
    if len(outcomes) != len(set(outcomes)):
        raise StrictJudgeEvidenceError(
            f"{result_path}: simulation[{simulation_index}] contains duplicate "
            "natural-language assertions"
        )
    return outcomes


def _validate_audit(
    path: Path,
    *,
    expected_outcomes: list[str],
    maximum_content_attempts: int,
) -> dict[str, Any]:
    payload = _load_object(path, label="strict-judge audit")
    attempts = payload.get("attempts")
    if (
        payload.get("protocol") != STRICT_JUDGE_PROTOCOL
        or payload.get("status") != "PASS"
        or payload.get("expected_outcomes") != expected_outcomes
        or not isinstance(attempts, list)
        or not attempts
        or payload.get("attempt_count") != len(attempts)
        or len(attempts) > maximum_content_attempts
    ):
        raise StrictJudgeEvidenceError(
            f"strict-judge audit protocol/status/outcomes drift: {path}"
        )

    accepted_attempts: list[int] = []
    format_rejections = 0
    for expected_attempt, row in enumerate(attempts, start=1):
        if not isinstance(row, dict) or row.get("attempt") != expected_attempt:
            raise StrictJudgeEvidenceError(
                f"strict-judge audit attempt sequence drift: {path}"
            )
        required = {
            "call_name",
            "raw_content",
            "raw_content_utf8_bytes",
            "raw_content_sha256",
            "raw_provider_response",
            "schema_status",
            "schema_error",
        }
        if not required <= set(row):
            raise StrictJudgeEvidenceError(
                f"strict-judge audit attempt fields are incomplete: {path}"
            )
        if row["call_name"] != f"nl_assertions_eval_strict_attempt_{expected_attempt}":
            raise StrictJudgeEvidenceError(
                f"strict-judge audit call name drift: {path}"
            )
        raw_content = row["raw_content"]
        if raw_content is not None and not isinstance(raw_content, str):
            raise StrictJudgeEvidenceError(
                f"strict-judge raw content has invalid type: {path}"
            )
        encoded = (raw_content or "").encode("utf-8")
        if (
            row["raw_content_utf8_bytes"] != len(encoded)
            or row["raw_content_sha256"]
            != hashlib.sha256(encoded).hexdigest()
        ):
            raise StrictJudgeEvidenceError(
                f"strict-judge raw content hash/length drift: {path}"
            )
        status = row["schema_status"]
        if status == "PASS":
            if row["schema_error"] is not None:
                raise StrictJudgeEvidenceError(
                    f"accepted strict-judge attempt records an error: {path}"
                )
            accepted_attempts.append(expected_attempt)
        elif status == "REJECTED":
            if not isinstance(row["schema_error"], str) or not row[
                "schema_error"
            ]:
                raise StrictJudgeEvidenceError(
                    f"rejected strict-judge attempt lacks an error: {path}"
                )
            format_rejections += 1
        else:
            raise StrictJudgeEvidenceError(
                f"strict-judge schema status drift: {path}"
            )
    if (
        accepted_attempts != [len(attempts)]
        or format_rejections != len(attempts) - 1
    ):
        raise StrictJudgeEvidenceError(
            f"strict-judge audit must end with exactly one accepted attempt: {path}"
        )
    return {
        "attempt_count": len(attempts),
        "accepted_attempt": accepted_attempts[0],
        "format_rejections": format_rejections,
    }


def validate_strict_judge_evidence(
    result_paths: Sequence[Path],
    *,
    audit_paths: Sequence[Path] | None = None,
    maximum_content_attempts: int = 2,
) -> dict[str, Any]:
    """Return a canonical, hash-bound per-simulation judge evidence map."""

    if (
        isinstance(maximum_content_attempts, bool)
        or not isinstance(maximum_content_attempts, int)
        or maximum_content_attempts < 1
    ):
        raise ValueError("maximum_content_attempts must be a positive integer")
    results = [Path(path).resolve() for path in result_paths]
    if not results or len(results) != len(set(results)):
        raise StrictJudgeEvidenceError(
            "result paths must be a nonempty unique collection"
        )
    parents = {path.parent for path in results}
    if len(parents) != 1:
        raise StrictJudgeEvidenceError(
            "strict-judge result files must share one output directory"
        )
    output_dir = next(iter(parents))
    for path in results:
        if not path.is_file():
            raise StrictJudgeEvidenceError(f"result file is absent: {path}")

    discovered: set[Path] = set()
    log_roots: dict[Path, Path] = {}
    for result_path in results:
        log_root = output_dir / "logs" / result_path.stem
        log_roots[result_path] = log_root
        if log_root.is_dir():
            discovered.update(
                path.resolve()
                for path in log_root.rglob(AUDIT_GLOB)
                if path.is_file()
            )

    if audit_paths is not None:
        supplied_list = [Path(path).resolve() for path in audit_paths]
        if len(supplied_list) != len(set(supplied_list)):
            raise StrictJudgeEvidenceError(
                "duplicate strict-judge audit path was supplied"
            )
        supplied = set(supplied_list)
        if supplied != discovered:
            raise StrictJudgeEvidenceError(
                "supplied strict-judge audit set differs from result log trees: "
                f"missing={sorted(str(path) for path in discovered - supplied)}, "
                f"extra={sorted(str(path) for path in supplied - discovered)}"
            )

    calls: list[dict[str, Any]] = []
    used_audits: set[Path] = set()
    call_keys: set[tuple[str, str, str]] = set()
    for result_path in sorted(results, key=lambda path: path.name):
        payload = _load_object(result_path, label="tau2 result")
        simulations = payload.get("simulations")
        if not isinstance(simulations, list):
            raise StrictJudgeEvidenceError(
                f"tau2 result lacks simulations array: {result_path}"
            )
        for simulation_index, simulation in enumerate(simulations):
            if not isinstance(simulation, dict):
                raise StrictJudgeEvidenceError(
                    f"{result_path}: simulation[{simulation_index}] "
                    "must be an object"
                )
            outcomes = _expected_outcomes(
                simulation,
                result_path=result_path,
                simulation_index=simulation_index,
            )
            if not outcomes:
                continue
            task_id = _safe_path_component(
                simulation.get("task_id"),
                label="task_id",
                where=result_path,
            )
            simulation_id = _safe_path_component(
                simulation.get("id"),
                label="id",
                where=result_path,
            )
            call_key = (result_path.name, task_id, simulation_id)
            if call_key in call_keys:
                raise StrictJudgeEvidenceError(
                    f"duplicate strict-judge simulation identity: {call_key}"
                )
            call_keys.add(call_key)
            expected_dir = (
                log_roots[result_path]
                / "artifacts"
                / f"task_{task_id}"
                / f"sim_{simulation_id}"
                / "llm_debug"
            )
            candidates = sorted(
                path.resolve()
                for path in expected_dir.glob(AUDIT_GLOB)
                if path.is_file()
            )
            if len(candidates) != 1:
                raise StrictJudgeEvidenceError(
                    "expected exactly one strict-judge audit for "
                    f"{result_path.name}/task_{task_id}/sim_{simulation_id}; "
                    f"found {len(candidates)}"
                )
            audit_path = candidates[0]
            if audit_path in used_audits:
                raise StrictJudgeEvidenceError(
                    f"strict-judge audit reused across simulations: {audit_path}"
                )
            used_audits.add(audit_path)
            audit = _validate_audit(
                audit_path,
                expected_outcomes=outcomes,
                maximum_content_attempts=maximum_content_attempts,
            )
            try:
                relative_audit = audit_path.relative_to(output_dir)
            except ValueError as error:
                raise StrictJudgeEvidenceError(
                    f"strict-judge audit escaped output directory: {audit_path}"
                ) from error
            calls.append(
                {
                    "result_file": result_path.name,
                    "result_sha256": sha256_file(result_path),
                    "task_id": task_id,
                    "simulation_id": simulation_id,
                    "trial": simulation.get("trial"),
                    "seed": simulation.get("seed"),
                    "expected_outcomes": outcomes,
                    "audit_path": relative_audit.as_posix(),
                    "audit_sha256": sha256_file(audit_path),
                    **audit,
                }
            )

    extra = discovered - used_audits
    if extra:
        raise StrictJudgeEvidenceError(
            "strict-judge audit files exist without a matching nonempty "
            f"nl_assertions simulation: {sorted(str(path) for path in extra)}"
        )
    calls.sort(
        key=lambda row: (
            row["result_file"],
            row["task_id"],
            row["simulation_id"],
        )
    )
    mapping_sha256 = hashlib.sha256(
        canonical_json(calls).encode("utf-8")
    ).hexdigest()
    return {
        "protocol": EVIDENCE_PROTOCOL,
        "status": "PASS",
        "expected_calls": len(calls),
        "observed_unique_pass_audits": len(used_audits),
        "missing_calls": 0,
        "extra_audits": 0,
        "duplicate_audits": 0,
        "maximum_content_attempts": maximum_content_attempts,
        "calls": calls,
        "canonical_mapping_sha256": mapping_sha256,
    }
