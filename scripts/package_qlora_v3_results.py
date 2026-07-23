#!/usr/bin/env python3
"""Create a minimal, fail-closed QLoRA V3 upload package.

The packager intentionally uses an explicit allowlist.  It includes enough
metadata to audit selection, training, evaluation, and aggregation, while
excluding raw trajectories, processed SFT/test JSONL, caches, environments,
and smoke-test artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARM = "constrained_recovery"
BUILD_PROTOCOL = "qlora_v3_constrained_recovery"
EVALUATION_PROTOCOL = "qlora_v3"
FORMAL_EXAMPLES = 959
FROZEN_SOURCE_TAG = "v3-frozen-20260723"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = checkpoint_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def source_commit() -> str:
    def rev_parse(reference: str) -> str:
        completed = subprocess.run(
            ["git", "rev-parse", f"{reference}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        value = completed.stdout.strip()
        require(
            bool(re.fullmatch(r"[0-9a-f]{40}", value)),
            f"git {reference} is not a full commit SHA",
        )
        return value

    head = rev_parse("HEAD")
    frozen = rev_parse(f"refs/tags/{FROZEN_SOURCE_TAG}")
    require(
        head == frozen,
        f"package source HEAD {head} differs from frozen tag {FROZEN_SOURCE_TAG} ({frozen})",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    require(
        not status.stdout.strip(),
        "tracked source/config files differ from the frozen tag; refusing "
        "misleading provenance",
    )
    return head


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_source_file(path: Path, label: str) -> Path:
    require(path.exists(), f"missing required {label}: {path}")
    require(not path.is_symlink(), f"refusing symlinked {label}: {path}")
    require(path.is_file(), f"required {label} is not a regular file: {path}")
    require(path.stat().st_size > 0, f"required {label} is empty: {path}")
    return path


def reject_nonfinite(value: Any, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite number in {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite(item, f"{location}[{index}]")


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    require_source_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON in {label} {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must contain a JSON object: {path}")
    reject_nonfinite(value, label)
    return value


def load_json_list(path: Path, label: str) -> list[Any]:
    require_source_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON in {label} {path}: {exc}") from exc
    require(isinstance(value, list), f"{label} must contain a JSON list: {path}")
    reject_nonfinite(value, label)
    return value


def load_training_log(path: Path) -> list[dict[str, Any]]:
    """Accept non-finite diagnostics but never non-finite objective losses."""
    require_source_file(path, "training log")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON in training log {path}: {exc}") from exc
    require(isinstance(value, list) and bool(value), "training log must be a non-empty list")
    for index, entry in enumerate(value):
        require(isinstance(entry, dict), f"training log row {index} is not an object")
        for key, item in entry.items():
            if (
                isinstance(item, float)
                and not math.isfinite(item)
                and "loss" in key.lower()
            ):
                raise RuntimeError(
                    f"non-finite objective loss in training log[{index}].{key}"
                )
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    require_source_file(path, label)
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"invalid UTF-8 in {label} {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        require(bool(line.strip()), f"blank line in {label} {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid JSON in {label} {path}:{line_number}: {exc}"
            ) from exc
        require(
            isinstance(row, dict),
            f"{label} row is not a JSON object: {path}:{line_number}",
        )
        reject_nonfinite(row, f"{label}[{line_number - 1}]")
        rows.append(row)
    return rows


def validate_output_target(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    require(not output_dir.is_symlink(), f"output directory may not be a symlink: {output_dir}")
    require(output_dir.is_dir(), f"output path exists but is not a directory: {output_dir}")
    require(
        not any(output_dir.iterdir()),
        f"output directory is not empty; refusing to mix packages: {output_dir}",
    )


def validate_inputs(
    results_root: Path,
    analysis_dir: Path,
    selection_dir: Path,
    processed_dir: Path,
    config_path: Path,
) -> tuple[list[tuple[Path, Path]], dict[str, Any]]:
    formal_dir = results_root / ARM
    checkpoint_dir = formal_dir / "checkpoint_final"

    comparison = load_json_object(analysis_dir / "comparison.json", "V3 comparison")
    require(comparison.get("valid") is True, "comparison.json does not declare valid=true")
    require(comparison.get("protocol") == EVALUATION_PROTOCOL, "comparison protocol is not qlora_v3")
    errors = comparison.get("errors")
    require(errors == [], f"comparison.json contains blocking errors: {errors!r}")
    reference = comparison.get("v2_reference")
    require(isinstance(reference, dict), "comparison.json lacks v2_reference metadata")
    reference_status = reference.get("status")
    require(
        reference_status == "compatible",
        "comparison.json must contain a compatible audited V2 Random reference",
    )
    judgement = comparison.get("direction_judgement")
    require(
        isinstance(judgement, dict) and judgement.get("allowed") is True,
        "comparison.json does not permit a paired V2/V3 direction judgement",
    )

    metrics_path = formal_dir / "metrics.json"
    contract_path = formal_dir / "metrics.contract.json"
    predictions_path = formal_dir / "metrics.predictions.jsonl"
    metrics = load_json_object(metrics_path, "V3 metrics")
    contract = load_json_object(contract_path, "V3 evaluation contract")
    predictions = load_jsonl(predictions_path, "V3 predictions")
    for label, value in (
        ("metrics protocol", metrics.get("protocol")),
        ("contract protocol", contract.get("protocol")),
    ):
        require(value == EVALUATION_PROTOCOL, f"{label} is not qlora_v3: {value!r}")
    for label, document in (("metrics", metrics), ("contract", contract)):
        require(
            document.get("formal_test_examples") == FORMAL_EXAMPLES,
            f"{label} formal_test_examples is not {FORMAL_EXAMPLES}",
        )
        require(
            document.get("evaluated_examples") == FORMAL_EXAMPLES,
            f"{label} evaluated_examples is not {FORMAL_EXAMPLES}",
        )
        require(document.get("limited") is False, f"{label} is marked as limited")
    require(
        len(predictions) == FORMAL_EXAMPLES,
        f"predictions must contain exactly {FORMAL_EXAMPLES} rows; found {len(predictions)}",
    )
    prediction_ids = [row.get("example_id") for row in predictions]
    require(
        all(isinstance(example_id, str) and example_id for example_id in prediction_ids),
        "every prediction must have a non-empty string example_id",
    )
    require(
        len(set(prediction_ids)) == FORMAL_EXAMPLES,
        "predictions contain duplicate example_id values",
    )
    require(
        metrics.get("predictions_sha256") == sha256_file(predictions_path),
        "metrics predictions_sha256 does not match metrics.predictions.jsonl",
    )
    require(
        metrics.get("checkpoint_fingerprint") == contract.get("checkpoint_fingerprint"),
        "metrics and evaluation contract checkpoint fingerprints differ",
    )
    recorded_fingerprint = metrics.get("checkpoint_fingerprint")
    require(
        isinstance(recorded_fingerprint, str) and bool(recorded_fingerprint),
        "evaluation checkpoint fingerprint is missing",
    )

    resume_path = formal_dir / "metrics.resume_recovery.jsonl"
    resume_count = metrics.get("resume_recovery_events")
    resume_embedded = metrics.get("resume_recovery_audit")
    require(
        type(resume_count) is int and resume_count >= 0,
        "metrics resume_recovery_events must be a non-negative integer",
    )
    require(isinstance(resume_embedded, list), "metrics resume_recovery_audit must be a list")
    optional_files: list[tuple[Path, Path]] = []
    if resume_path.exists():
        resume_rows = load_jsonl(resume_path, "resume-recovery audit")
        require(
            len(resume_rows) == resume_count,
            "resume-recovery file count does not match metrics",
        )
        require(
            resume_rows == resume_embedded,
            "resume-recovery file does not match metrics embedded audit",
        )
        optional_files.append(
            (resume_path, Path("results/qlora_v3") / ARM / resume_path.name)
        )
    else:
        require(
            resume_count == 0 and resume_embedded == [],
            "metrics records resume recovery but the audit JSONL is missing",
        )

    adapter_config = load_json_object(
        checkpoint_dir / "adapter_config.json", "adapter configuration"
    )
    require(bool(adapter_config), "adapter_config.json is empty")
    adapter_model = require_source_file(
        checkpoint_dir / "adapter_model.safetensors", "adapter weights"
    )
    require(adapter_model.stat().st_size > 16, "adapter weights are implausibly small")
    require(
        checkpoint_fingerprint(checkpoint_dir) == recorded_fingerprint,
        "packaged adapter does not match the checkpoint evaluated in metrics",
    )

    run_manifest = load_json_object(formal_dir / "run_manifest.json", "training run manifest")
    require(run_manifest.get("protocol") == "qlora_v2", "formal V3 training protocol is not frozen qlora_v2")
    require(run_manifest.get("smoke_test") is False, "formal training manifest is marked as smoke-test")
    training_metrics = load_json_object(
        formal_dir / "training_metrics.json", "training metrics"
    )
    require(bool(training_metrics), "training_metrics.json is empty")
    training_log = load_training_log(formal_dir / "training_log.json")
    training_audit = load_json_object(
        formal_dir / "training_artifact_audit.json", "training artifact audit"
    )
    require(training_audit.get("status") == "PASS", "training artifact audit is not PASS")
    audited_checkpoint = training_audit.get("checkpoint_files")
    require(
        isinstance(audited_checkpoint, dict),
        "training artifact audit lacks checkpoint_files",
    )
    for path in (
        checkpoint_dir / "adapter_config.json",
        checkpoint_dir / "adapter_model.safetensors",
    ):
        entry = audited_checkpoint.get(path.name)
        require(
            isinstance(entry, dict),
            f"training artifact audit lacks {path.name}",
        )
        require(
            entry.get("bytes") == path.stat().st_size,
            f"training artifact audit byte count drift for {path.name}",
        )
        require(
            entry.get("sha256") == sha256_file(path),
            f"training artifact audit SHA-256 drift for {path.name}",
        )

    commands = load_json_list(results_root / "commands.json", "execution command audit")
    completed = [
        row
        for row in commands
        if isinstance(row, dict)
        and row.get("status") == "complete"
        and row.get("returncode") == 0
    ]
    require(bool(completed), "commands.json contains no successfully completed command")
    preflight = load_json_object(
        results_root / "preflight_environment.json", "CUDA preflight audit"
    )
    require(preflight.get("bf16_supported") is True, "preflight does not confirm BF16 support")

    selection_manifest_path = selection_dir / f"{ARM}_manifest.json"
    selection_manifest = load_json_object(selection_manifest_path, "selection manifest")
    require(
        selection_manifest.get("protocol") == BUILD_PROTOCOL,
        "selection manifest protocol is not qlora_v3_constrained_recovery",
    )
    build_summary = load_json_object(processed_dir / "build_summary.json", "V3 build summary")
    require(
        build_summary.get("protocol") == BUILD_PROTOCOL,
        "build summary protocol is not qlora_v3_constrained_recovery",
    )
    contract_audit = load_json_object(
        processed_dir / "contract_audit.json", "processed-data contract audit"
    )
    require(contract_audit.get("status") == "PASS", "processed-data contract audit is not PASS")
    require(
        build_summary.get("hashes", {}).get("manifest_json")
        == sha256_file(selection_manifest_path),
        "build summary does not hash the packaged selection manifest",
    )
    require(
        contract_audit.get("build_summary_sha256")
        == sha256_file(processed_dir / "build_summary.json"),
        "contract audit does not hash the packaged build summary",
    )
    require(
        contract_audit.get("manifest_sha256") == sha256_file(selection_manifest_path),
        "contract audit does not hash the packaged selection manifest",
    )

    comparison_groups = comparison.get("v3", {}).get("groups")
    metric_groups = metrics.get("groups")
    require(
        isinstance(comparison_groups, dict) and isinstance(metric_groups, dict),
        "comparison or metrics lacks V3 group summaries",
    )
    for group in (
        "overall",
        "non_recovery",
        "recovery",
        "agent_initiated",
        "user_assisted",
    ):
        comparison_group = comparison_groups.get(group)
        metric_group = metric_groups.get(group)
        require(
            isinstance(comparison_group, dict) and isinstance(metric_group, dict),
            f"missing {group} group in comparison or metrics",
        )
        for key in ("examples", "tasks", "micro", "task_macro"):
            require(
                comparison_group.get(key) == metric_group.get(key),
                f"comparison/metrics drift for V3 {group}.{key}",
            )

    config_file = require_source_file(config_path, "V3 configuration")
    try:
        config_text = config_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"invalid UTF-8 in V3 configuration {config_file}: {exc}") from exc
    require(
        "protocol: qlora_v3_constrained_recovery" in config_text,
        "V3 configuration does not declare the frozen build protocol",
    )

    required_files = [
        (
            checkpoint_dir / "adapter_config.json",
            Path("results/qlora_v3") / ARM / "checkpoint_final/adapter_config.json",
        ),
        (
            checkpoint_dir / "adapter_model.safetensors",
            Path("results/qlora_v3") / ARM / "checkpoint_final/adapter_model.safetensors",
        ),
        *[
            (
                formal_dir / name,
                Path("results/qlora_v3") / ARM / name,
            )
            for name in (
                "run_manifest.json",
                "training_metrics.json",
                "training_log.json",
                "training_artifact_audit.json",
                "metrics.json",
                "metrics.contract.json",
                "metrics.predictions.jsonl",
            )
        ],
        (results_root / "commands.json", Path("results/qlora_v3/commands.json")),
        (
            results_root / "preflight_environment.json",
            Path("results/qlora_v3/preflight_environment.json"),
        ),
        *[
            (analysis_dir / name, Path("results/analysis_v3") / name)
            for name in ("comparison.json", "comparison.csv", "comparison.md")
        ],
        (
            selection_manifest_path,
            Path("results/selection_v3") / selection_manifest_path.name,
        ),
        (
            processed_dir / "build_summary.json",
            Path("data/processed/qlora_v3/build_summary.json"),
        ),
        (
            processed_dir / "contract_audit.json",
            Path("data/processed/qlora_v3/contract_audit.json"),
        ),
        (config_path, Path("configs/qlora_v3.yaml")),
        *optional_files,
    ]
    destinations: set[Path] = set()
    for source, destination in required_files:
        require_source_file(source, f"packaged artifact {destination.as_posix()}")
        require(
            destination not in destinations,
            f"duplicate package destination: {destination.as_posix()}",
        )
        destinations.add(destination)
    return required_files, {
        "comparison": comparison,
        "reference_status": reference_status,
        "reference_note": reference.get("note"),
    }


def copy_allowlist(
    stage_dir: Path,
    files: list[tuple[Path, Path]],
) -> None:
    for source, destination in files:
        target = stage_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(
            target.stat().st_size == source.stat().st_size
            and sha256_file(target) == sha256_file(source),
            f"copy verification failed for {destination.as_posix()}",
        )


def write_readme(stage_dir: Path, metadata: dict[str, Any]) -> None:
    comparison = metadata["comparison"]
    judgement = comparison.get("direction_judgement")
    judgement_label = (
        judgement.get("label") if isinstance(judgement, dict) else "not_available"
    )
    reference_status = metadata["reference_status"]
    compatible = reference_status == "compatible"
    note = metadata.get("reference_note")
    lines = [
        "# QLoRA V3 RTX 5060 audit package",
        "",
        "This package contains the minimal allowlisted artifacts needed to audit "
        "the constrained-recovery V3 selection, formal QLoRA training, complete "
        "959-example evaluation, and aggregate report.",
        "",
        "## Claim boundary",
        "",
        "V3 is an exploratory diagnostic of offline next-tool-call imitation on "
        "the already inspected V2 test set. It is not an end-to-end Agent-success "
        "evaluation, not executable tool success, and not confirmatory or "
        "paper-final evidence.",
        "",
        f"- Frozen source commit: `{metadata['source_commit']}`",
        "",
        "## Reference status",
        "",
        f"- V2 reference status: `{reference_status}`",
        f"- Compatible paired V2 reference: `{'yes' if compatible else 'no'}`",
        f"- Direction label: `{judgement_label}`",
    ]
    if isinstance(note, str) and note:
        lines.append(f"- Aggregator note: {note}")
    lines.extend(
        [
            "",
            "## Deliberate exclusions",
            "",
            "No raw trajectories, processed train/validation/test JSONL, virtual "
            "environment, model/tokenizer cache, or smoke-test checkpoint is "
            "included. `UPLOAD_MANIFEST.json` hashes every packaged file except "
            "itself.",
            "",
        ]
    )
    (stage_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_upload_manifest(stage_dir: Path, commit: str) -> dict[str, Any]:
    manifest_path = stage_dir / "UPLOAD_MANIFEST.json"
    entries: list[dict[str, Any]] = []
    for path in sorted(stage_dir.rglob("*")):
        if path == manifest_path or path.is_dir():
            continue
        require(not path.is_symlink(), f"staged package unexpectedly contains a symlink: {path}")
        relative = path.relative_to(stage_dir).as_posix()
        entries.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    require(bool(entries), "refusing to create an empty upload package")
    payload = {
        "protocol": "qlora_v3_upload_package",
        "source_commit": commit,
        "generated_at_utc": utc_now(),
        "manifest_excludes": ["UPLOAD_MANIFEST.json"],
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_staged_manifest(stage_dir: Path, payload: dict[str, Any]) -> None:
    manifest = load_json_object(stage_dir / "UPLOAD_MANIFEST.json", "upload manifest")
    require(manifest == payload, "serialized upload manifest changed after writing")
    entries = manifest.get("files")
    require(isinstance(entries, list), "upload manifest files is not a list")
    observed_paths: set[str] = set()
    total_bytes = 0
    for entry in entries:
        require(isinstance(entry, dict), "upload manifest contains a non-object file entry")
        relative = entry.get("relative_path")
        require(isinstance(relative, str) and relative, "manifest relative_path is invalid")
        require(relative != "UPLOAD_MANIFEST.json", "manifest must exclude itself")
        require(relative not in observed_paths, f"duplicate manifest path: {relative}")
        observed_paths.add(relative)
        path = stage_dir / relative
        require_source_file(path, f"staged artifact {relative}")
        require(path.stat().st_size == entry.get("bytes"), f"byte count changed for {relative}")
        require(sha256_file(path) == entry.get("sha256"), f"SHA-256 changed for {relative}")
        total_bytes += path.stat().st_size
    actual = {
        path.relative_to(stage_dir).as_posix()
        for path in stage_dir.rglob("*")
        if path.is_file() and path.name != "UPLOAD_MANIFEST.json"
    }
    require(observed_paths == actual, "upload manifest does not cover exactly the staged files")
    require(manifest.get("file_count") == len(actual), "manifest file_count is incorrect")
    require(manifest.get("total_bytes") == total_bytes, "manifest total_bytes is incorrect")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/qlora_v3"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/analysis_v3"))
    parser.add_argument("--selection-dir", type=Path, default=Path("results/selection_v3"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/qlora_v3"))
    parser.add_argument("--config", type=Path, default=Path("configs/qlora_v3.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/qlora_v3/rtx5060"),
    )
    args = parser.parse_args()

    validate_output_target(args.output_dir)
    files, metadata = validate_inputs(
        args.results_root,
        args.analysis_dir,
        args.selection_dir,
        args.processed_dir,
        args.config,
    )
    metadata["source_commit"] = source_commit()

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.staging-",
            dir=args.output_dir.parent,
        )
    )
    try:
        copy_allowlist(stage_dir, files)
        staged_files, staged_metadata = validate_inputs(
            stage_dir / "results/qlora_v3",
            stage_dir / "results/analysis_v3",
            stage_dir / "results/selection_v3",
            stage_dir / "data/processed/qlora_v3",
            stage_dir / "configs/qlora_v3.yaml",
        )
        require(
            {destination for _, destination in staged_files}
            == {destination for _, destination in files},
            "staged package allowlist changed during verification",
        )
        require(
            staged_metadata["comparison"] == metadata["comparison"],
            "staged comparison changed during copy",
        )
        staged_metadata["source_commit"] = metadata["source_commit"]
        write_readme(stage_dir, staged_metadata)
        manifest = write_upload_manifest(stage_dir, metadata["source_commit"])
        verify_staged_manifest(stage_dir, manifest)
        if args.output_dir.exists():
            args.output_dir.rmdir()
        os.replace(stage_dir, args.output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir),
                "file_count_excluding_manifest": manifest["file_count"],
                "total_bytes_excluding_manifest": manifest["total_bytes"],
                "upload_manifest": str(args.output_dir / "UPLOAD_MANIFEST.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
