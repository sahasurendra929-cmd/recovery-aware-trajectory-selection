from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_v6_checkpoint_registry as checkpoint
from scripts import build_v6_official_unseal_receipt as unseal
from scripts import build_v6_10_release_manifest as release


SOURCE_COMMIT = "a" * 40
TAU2_COMMIT = "b" * 40


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def fixture_tree(root: Path) -> tuple[dict[str, Path], Path, Path, Path]:
    artifacts: dict[str, Path] = {}
    pool_hash = "c" * 64
    pool_path = root / "formal_candidate_pool_hash.txt"
    pool_path.write_text(pool_hash + "\n", encoding="ascii")
    artifacts["formal_candidate_pool_hash"] = pool_path
    for label in ("release_manifest", "formal_freeze_manifest", "scoring_audit"):
        path = root / f"{label}.json"
        payload = {
            "source_commit": SOURCE_COMMIT,
            "official_test_used": False,
            "official_test_sealed": True,
        }
        if label == "release_manifest":
            payload.update({"protocol": release.PROTOCOL, "status": "PASS"})
        elif label == "formal_freeze_manifest":
            payload.update(
                {
                    "protocol": "v6_candidate_pair_freeze_manifest_v1",
                    "status": "FROZEN",
                    "gate": {"status": "FORMAL_SELECTION_AUTHORIZED"},
                }
            )
        else:
            payload.update(
                {
                    "protocol": "v6_causal_recovery_candidate_scoring_v1",
                    "status": "PASS",
                }
            )
        write_json(path, payload)
        artifacts[label] = path
    selector_hashes: dict[str, str] = {}
    for label, (selector, seed) in unseal.SELECTOR_LABELS.items():
        path = root / f"{label}.json"
        write_json(
            path,
            {
                "protocol": "v6_matched_task_selector_manifests_v1",
                "selector": selector,
                "selection_seed": seed,
                "candidate_pool_sha256": pool_hash,
                "official_test_used": False,
            },
        )
        artifacts[label] = path
        selector_hashes[label] = unseal.sha256_file(path)
    for label, (selector, seed) in unseal.MATERIALIZATION_LABELS.items():
        selector_label = next(
            name
            for name, identity in unseal.SELECTOR_LABELS.items()
            if identity == (selector, seed)
        )
        path = root / f"{label}.json"
        write_json(
            path,
            {
                "protocol": "v6_sft_materialization_v1",
                "status": "PASS",
                "selector": selector,
                "selection_seed": seed,
                "selector_manifest_sha256": selector_hashes[selector_label],
                "candidate_pool_sha256": pool_hash,
                "official_test_used": False,
            },
        )
        artifacts[label] = path
    arm_selector_labels = {
        "flawless_only": "selector_flawless_only",
        "random_stratified": "selector_random_20260806",
        "full_proposed": "selector_full_proposed",
    }
    entries = {}
    for label, arm in unseal.TRAINING_LABELS.items():
        selector_hash = selector_hashes[arm_selector_labels[arm]]
        path = root / f"{label}.json"
        write_json(
            path,
            {
                "protocol": "v6_directional_sft_training_v1",
                "status": "PASS",
                "source_commit": SOURCE_COMMIT,
                "mode": "formal",
                "arm": arm,
                "selector_manifest_sha256": selector_hash,
                "candidate_pool_sha256": pool_hash,
                "official_test_used": False,
            },
        )
        artifacts[label] = path
        entries[arm] = {
            "run_manifest_sha256": unseal.sha256_file(path),
            "selector_manifest_sha256": selector_hash,
            "candidate_pool_sha256": pool_hash,
        }
    registry_path = root / "checkpoint_registry.json"
    write_json(
        registry_path,
        {
            "protocol": checkpoint.PROTOCOL,
            "source_commit": SOURCE_COMMIT,
            "registered_arms": list(unseal.ARMS),
            "registered_training_seeds": [unseal.TRAINING_SEED],
            "entries": entries,
            "official_test_used": False,
            "official_test_sealed": True,
        },
    )
    artifacts["checkpoint_registry"] = registry_path
    split = root / "split.json"
    write_json(
        split,
        {
            "protocol": "v5_stage0_tau2_end_to_end",
            "domains": {
                "retail": {"sealed_test_ids": list(range(40))},
                "airline": {"sealed_test_ids": list(range(20))},
            },
        },
    )
    source = root / "source"
    (source / "scripts").mkdir(parents=True)
    evaluator = source / "scripts/run_v6_end_to_end_eval.py"
    summarizer = source / "scripts/summarize_v6_task_clusters.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    summarizer.write_text("# summarizer\n", encoding="utf-8")
    tau2 = root / "tau2"
    tau2.mkdir()
    return artifacts, split, source, tau2


def test_unseal_binds_every_frozen_input_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, split, source, tau2 = fixture_tree(tmp_path)

    def identity(path: Path) -> tuple[str, bool]:
        return (SOURCE_COMMIT, True) if path == source else (TAU2_COMMIT, True)

    monkeypatch.setattr(unseal, "git_identity", identity)
    output = tmp_path / "official_test_unseal_receipt.json"
    result = unseal.build(
        source_root=source,
        tau2_root=tau2,
        split_manifest=split,
        evaluator=source / "scripts/run_v6_end_to_end_eval.py",
        summarizer=source / "scripts/summarize_v6_task_clusters.py",
        bound_artifacts=artifacts,
        output=output,
        expected_source_commit=SOURCE_COMMIT,
        expected_tau2_commit=TAU2_COMMIT,
    )
    assert result["status"] == "UNSEALED_ONCE"
    assert result["official_test_access_count"] == 1
    assert result["official_test_population"]["task_count"] == 60
    assert set(result["bound_artifacts"]) == unseal.REQUIRED_ARTIFACT_LABELS
    assert output.is_file()
    with pytest.raises(unseal.UnsealError, match="overwrite"):
        unseal.build(
            source_root=source,
            tau2_root=tau2,
            split_manifest=split,
            evaluator=source / "scripts/run_v6_end_to_end_eval.py",
            summarizer=source / "scripts/summarize_v6_task_clusters.py",
            bound_artifacts=artifacts,
            output=output,
            expected_source_commit=SOURCE_COMMIT,
            expected_tau2_commit=TAU2_COMMIT,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dirty", "dirty"),
        ("used", "already used"),
        ("gate", "not selection-authorized"),
        ("checkpoint", "identity drift"),
        ("cross_binding", "selector/pool binding drift"),
    ],
)
def test_unseal_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    artifacts, split, source, tau2 = fixture_tree(tmp_path)
    clean = mutation != "dirty"

    def identity(path: Path) -> tuple[str, bool]:
        return (
            (SOURCE_COMMIT, clean)
            if path == source
            else (TAU2_COMMIT, True)
        )

    monkeypatch.setattr(unseal, "git_identity", identity)
    if mutation == "used":
        value = unseal.read_json(artifacts["scoring_audit"])
        value["official_test_used"] = True
        write_json(artifacts["scoring_audit"], value)
    elif mutation == "gate":
        value = unseal.read_json(artifacts["formal_freeze_manifest"])
        value["gate"]["status"] = "SCREEN_ONLY"
        write_json(artifacts["formal_freeze_manifest"], value)
    elif mutation == "checkpoint":
        value = unseal.read_json(artifacts["checkpoint_registry"])
        value["registered_arms"] = ["full_proposed"]
        write_json(artifacts["checkpoint_registry"], value)
    elif mutation == "cross_binding":
        value = unseal.read_json(
            artifacts["materialization_full_proposed"]
        )
        value["selector_manifest_sha256"] = "d" * 64
        write_json(artifacts["materialization_full_proposed"], value)
    with pytest.raises(unseal.UnsealError, match=message):
        unseal.build(
            source_root=source,
            tau2_root=tau2,
            split_manifest=split,
            evaluator=source / "scripts/run_v6_end_to_end_eval.py",
            summarizer=source / "scripts/summarize_v6_task_clusters.py",
            bound_artifacts=artifacts,
            output=tmp_path / "receipt.json",
            expected_source_commit=SOURCE_COMMIT,
            expected_tau2_commit=TAU2_COMMIT,
        )


def test_bound_artifact_arguments_are_exact(tmp_path: Path) -> None:
    values = [
        f"{label}={tmp_path / (label + '.json')}"
        for label in sorted(unseal.REQUIRED_ARTIFACT_LABELS)
    ]
    assert set(unseal.parse_bound_artifacts(values)) == (
        unseal.REQUIRED_ARTIFACT_LABELS
    )
    with pytest.raises(unseal.UnsealError, match="set drift"):
        unseal.parse_bound_artifacts(values[:-1])
    with pytest.raises(unseal.UnsealError, match="invalid"):
        unseal.parse_bound_artifacts(values + [values[0]])
