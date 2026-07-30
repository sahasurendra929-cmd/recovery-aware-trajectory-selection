#!/usr/bin/env python3
"""Build the one-time, fail-closed V6.10 official-test unseal receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from build_v6_checkpoint_registry import PROTOCOL as CHECKPOINT_PROTOCOL
    from build_v6_10_release_manifest import PROTOCOL as RELEASE_PROTOCOL
    from summarize_v6_task_clusters import (
        ARMS,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        CLEAN_NONINFERIORITY_MARGIN,
        EVALUATION_SEEDS,
        TRAINING_SEED,
    )
except ModuleNotFoundError:
    from scripts.build_v6_checkpoint_registry import (
        PROTOCOL as CHECKPOINT_PROTOCOL,
    )
    from scripts.build_v6_10_release_manifest import (
        PROTOCOL as RELEASE_PROTOCOL,
    )
    from scripts.summarize_v6_task_clusters import (
        ARMS,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        CLEAN_NONINFERIORITY_MARGIN,
        EVALUATION_SEEDS,
        TRAINING_SEED,
    )


PROTOCOL = "v6_10_official_test_unseal_v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_ARTIFACT_LABELS = {
    "release_manifest",
    "formal_freeze_manifest",
    "formal_candidate_pool_hash",
    "scoring_audit",
    "selector_flawless_only",
    "selector_random_20260806",
    "selector_random_20260807",
    "selector_random_20260808",
    "selector_full_proposed",
    "materialization_flawless_only",
    "materialization_random_20260806",
    "materialization_random_20260807",
    "materialization_random_20260808",
    "materialization_full_proposed",
    "training_manifest_flawless_only",
    "training_manifest_random_stratified",
    "training_manifest_full_proposed",
    "checkpoint_registry",
}
SELECTOR_LABELS = {
    "selector_flawless_only": ("flawless_only", None),
    "selector_random_20260806": ("random_stratified", 20260806),
    "selector_random_20260807": ("random_stratified", 20260807),
    "selector_random_20260808": ("random_stratified", 20260808),
    "selector_full_proposed": ("full_proposed", None),
}
MATERIALIZATION_LABELS = {
    "materialization_flawless_only": ("flawless_only", None),
    "materialization_random_20260806": ("random_stratified", 20260806),
    "materialization_random_20260807": ("random_stratified", 20260807),
    "materialization_random_20260808": ("random_stratified", 20260808),
    "materialization_full_proposed": ("full_proposed", None),
}
TRAINING_LABELS = {
    "training_manifest_flawless_only": "flawless_only",
    "training_manifest_random_stratified": "random_stratified",
    "training_manifest_full_proposed": "full_proposed",
}


class UnsealError(RuntimeError):
    """The V6.10 official-test unseal contract was violated."""


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
        raise UnsealError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise UnsealError(f"expected JSON object: {path}")
    return value


def git_identity(source_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status.strip())


def _official_test_still_sealed(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "official_test_used" and item is not False:
                return False
            if key == "official_test_sealed" and item is not True:
                return False
            if not _official_test_still_sealed(item):
                return False
    elif isinstance(value, list):
        return all(_official_test_still_sealed(item) for item in value)
    return True


def _split_population(split: Mapping[str, Any]) -> dict[str, Any]:
    domains = split.get("domains")
    if (
        split.get("protocol") != "v5_stage0_tau2_end_to_end"
        or not isinstance(domains, Mapping)
    ):
        raise UnsealError("split manifest protocol drift")
    expected = {"retail": 40, "airline": 20}
    task_ids: list[str] = []
    domain_hashes: dict[str, str] = {}
    for domain, count in expected.items():
        payload = domains.get(domain)
        ids = (
            payload.get("sealed_test_ids")
            if isinstance(payload, Mapping)
            else None
        )
        normalized = [str(value) for value in ids] if isinstance(ids, list) else []
        if len(normalized) != count or len(set(normalized)) != count:
            raise UnsealError(f"{domain}: official task population drift")
        identities = [f"{domain}:{value}" for value in normalized]
        task_ids.extend(identities)
        domain_hashes[domain] = hashlib.sha256(
            canonical(identities).encode("utf-8")
        ).hexdigest()
    if len(task_ids) != 60 or len(set(task_ids)) != 60:
        raise UnsealError("official task population must contain 60 tasks")
    return {
        "task_count": 60,
        "domain_counts": expected,
        "domain_task_ids_sha256": domain_hashes,
        "all_task_ids_sha256": hashlib.sha256(
            canonical(task_ids).encode("utf-8")
        ).hexdigest(),
        "task_content_exported_before_unseal": False,
    }


def _validate_bound_json(
    *,
    label: str,
    path: Path,
    source_commit: str,
) -> dict[str, Any]:
    value = read_json(path)
    if not _official_test_still_sealed(value):
        raise UnsealError(f"{label}: official test was already used/unsealed")
    artifact_commit = value.get("source_commit")
    if artifact_commit is not None and artifact_commit != source_commit:
        raise UnsealError(f"{label}: source commit drift")
    if label == "release_manifest" and (
        value.get("protocol") != RELEASE_PROTOCOL
        or value.get("status") != "PASS"
        or value.get("source_commit") != source_commit
    ):
        raise UnsealError("release manifest is not a source-bound PASS")
    if label == "formal_freeze_manifest" and (
        value.get("protocol") != "v6_candidate_pair_freeze_manifest_v1"
        or value.get("status") != "FROZEN"
        or (value.get("gate") or {}).get("status")
        != "FORMAL_SELECTION_AUTHORIZED"
    ):
        raise UnsealError("formal candidate pool is not selection-authorized")
    if label == "checkpoint_registry" and (
        value.get("protocol") != CHECKPOINT_PROTOCOL
        or value.get("source_commit") != source_commit
        or value.get("registered_arms") != list(ARMS)
        or value.get("registered_training_seeds") != [TRAINING_SEED]
    ):
        raise UnsealError("checkpoint registry identity drift")
    if label == "scoring_audit" and (
        value.get("protocol") != "v6_causal_recovery_candidate_scoring_v1"
        or value.get("status") != "PASS"
    ):
        raise UnsealError("scoring audit did not pass")
    if label in SELECTOR_LABELS:
        selector, seed = SELECTOR_LABELS[label]
        if (
            value.get("protocol") != "v6_matched_task_selector_manifests_v1"
            or value.get("selector") != selector
            or value.get("selection_seed") != seed
        ):
            raise UnsealError(f"{label}: selector identity drift")
    if label in MATERIALIZATION_LABELS:
        selector, seed = MATERIALIZATION_LABELS[label]
        if (
            value.get("protocol") != "v6_sft_materialization_v1"
            or value.get("status") != "PASS"
            or value.get("selector") != selector
            or value.get("selection_seed") != seed
        ):
            raise UnsealError(f"{label}: materialization identity drift")
    if label in TRAINING_LABELS and (
        value.get("protocol") != "v6_directional_sft_training_v1"
        or value.get("status") != "PASS"
        or value.get("mode") != "formal"
        or value.get("arm") != TRAINING_LABELS[label]
    ):
        raise UnsealError(f"{label}: training identity drift")
    return value


def _validate_cross_bindings(
    payloads: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
) -> None:
    pool_hash_path = paths["formal_candidate_pool_hash"]
    try:
        pool_hash = pool_hash_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise UnsealError("formal candidate pool hash is unreadable") from error
    if SHA256_RE.fullmatch(pool_hash) is None:
        raise UnsealError("formal candidate pool hash is invalid")
    selector_hashes: dict[str, str] = {}
    for label, (selector, seed) in SELECTOR_LABELS.items():
        manifest = payloads[label]
        if manifest.get("candidate_pool_sha256") != pool_hash:
            raise UnsealError(f"{label}: candidate pool binding drift")
        selector_hashes[label] = sha256_file(paths[label])
        materialization_label = next(
            name
            for name, identity in MATERIALIZATION_LABELS.items()
            if identity == (selector, seed)
        )
        materialization = payloads[materialization_label]
        if (
            materialization.get("candidate_pool_sha256") != pool_hash
            or materialization.get("selector_manifest_sha256")
            != selector_hashes[label]
        ):
            raise UnsealError(
                f"{materialization_label}: selector/pool binding drift"
            )
    registry = payloads["checkpoint_registry"]
    entries = registry.get("entries")
    if not isinstance(entries, Mapping) or set(entries) != set(ARMS):
        raise UnsealError("checkpoint registry entries drift")
    training_selector_labels = {
        "flawless_only": "selector_flawless_only",
        "random_stratified": "selector_random_20260806",
        "full_proposed": "selector_full_proposed",
    }
    for training_label, arm in TRAINING_LABELS.items():
        training = payloads[training_label]
        entry = entries.get(arm)
        if (
            not isinstance(entry, Mapping)
            or sha256_file(paths[training_label])
            != entry.get("run_manifest_sha256")
            or training.get("selector_manifest_sha256")
            != selector_hashes[training_selector_labels[arm]]
            or training.get("candidate_pool_sha256") != pool_hash
            or entry.get("selector_manifest_sha256")
            != training.get("selector_manifest_sha256")
            or entry.get("candidate_pool_sha256") != pool_hash
        ):
            raise UnsealError(f"{training_label}: registry binding drift")


def parse_bound_artifacts(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if (
            not separator
            or label not in REQUIRED_ARTIFACT_LABELS
            or label in result
            or not raw_path
        ):
            raise UnsealError(f"invalid bound artifact argument: {value!r}")
        result[label] = Path(raw_path).expanduser().resolve()
    missing = REQUIRED_ARTIFACT_LABELS - set(result)
    extra = set(result) - REQUIRED_ARTIFACT_LABELS
    if missing or extra:
        raise UnsealError(
            f"bound artifact set drift: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return result


def build(
    *,
    source_root: Path,
    tau2_root: Path,
    split_manifest: Path,
    evaluator: Path,
    summarizer: Path,
    bound_artifacts: Mapping[str, Path],
    output: Path,
    expected_source_commit: str,
    expected_tau2_commit: str,
) -> dict[str, Any]:
    if output.exists():
        raise UnsealError(f"refusing to overwrite {output}")
    if (
        COMMIT_RE.fullmatch(expected_source_commit) is None
        or COMMIT_RE.fullmatch(expected_tau2_commit) is None
    ):
        raise UnsealError("source and Tau2 commits must be full lowercase commits")
    source_commit, source_clean = git_identity(source_root)
    tau2_commit, tau2_clean = git_identity(tau2_root)
    if source_commit != expected_source_commit or not source_clean:
        raise UnsealError("source checkout is dirty or at the wrong commit")
    if tau2_commit != expected_tau2_commit or not tau2_clean:
        raise UnsealError("Tau2 checkout is dirty or at the wrong commit")
    expected_evaluator = (source_root / "scripts/run_v6_end_to_end_eval.py").resolve()
    expected_summarizer = (
        source_root / "scripts/summarize_v6_task_clusters.py"
    ).resolve()
    if evaluator.resolve() != expected_evaluator or not evaluator.is_file():
        raise UnsealError("evaluator is not the tracked frozen executable")
    if summarizer.resolve() != expected_summarizer or not summarizer.is_file():
        raise UnsealError("summarizer is not the tracked frozen executable")
    if set(bound_artifacts) != REQUIRED_ARTIFACT_LABELS:
        raise UnsealError("bound artifact label set drift")
    artifact_rows: dict[str, dict[str, Any]] = {}
    artifact_payloads: dict[str, dict[str, Any]] = {}
    for label in sorted(bound_artifacts):
        path = bound_artifacts[label].resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise UnsealError(f"{label}: bound artifact is missing/empty")
        if label != "formal_candidate_pool_hash":
            artifact_payloads[label] = _validate_bound_json(
                label=label,
                path=path,
                source_commit=source_commit,
            )
        artifact_rows[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    _validate_cross_bindings(
        artifact_payloads,
        {label: path.resolve() for label, path in bound_artifacts.items()},
    )
    split = read_json(split_manifest)
    population = _split_population(split)
    payload = {
        "protocol": PROTOCOL,
        "status": "UNSEALED_ONCE",
        "source_commit": source_commit,
        "source_clean_before_unseal": True,
        "tau2_commit": tau2_commit,
        "tau2_clean_before_unseal": True,
        "split_manifest": {
            "path": str(split_manifest.resolve()),
            "sha256": sha256_file(split_manifest),
        },
        "official_test_population": population,
        "bound_artifacts": artifact_rows,
        "evaluator_path": str(evaluator.resolve()),
        "evaluator_sha256": sha256_file(evaluator),
        "summarizer_path": str(summarizer.resolve()),
        "summarizer_sha256": sha256_file(summarizer),
        "training_seed": TRAINING_SEED,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "clean_noninferiority_margin": CLEAN_NONINFERIORITY_MARGIN,
        "primary_contrast": "full_proposed_minus_random_stratified",
        "official_test_access_count_before_unseal": 0,
        "official_test_access_count": 1,
        "official_test_used_for_selection": False,
        "changes_after_unseal": "FORBIDDEN",
        "unsealed_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
    }
    payload["receipt_sha256"] = hashlib.sha256(
        canonical(payload).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--summarizer", type=Path, required=True)
    parser.add_argument(
        "--bound-artifact",
        action="append",
        default=[],
        help="Required label=/absolute/path binding; repeat for every label.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-tau2-commit", required=True)
    args = parser.parse_args()
    payload = build(
        source_root=args.source_root.resolve(),
        tau2_root=args.tau2_root.resolve(),
        split_manifest=args.split_manifest.resolve(),
        evaluator=args.evaluator.resolve(),
        summarizer=args.summarizer.resolve(),
        bound_artifacts=parse_bound_artifacts(args.bound_artifact),
        output=args.output.resolve(),
        expected_source_commit=args.expected_source_commit,
        expected_tau2_commit=args.expected_tau2_commit,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt": str(args.output.resolve()),
                "receipt_sha256": payload["receipt_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
