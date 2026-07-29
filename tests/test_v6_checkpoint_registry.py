import json
from pathlib import Path

import pytest

from scripts import build_v6_checkpoint_registry as registry
from scripts import train_v6_directional_sft as trainer


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def make_run(root: Path, arm: str, source_commit: str) -> None:
    run = root / registry.RUN_DIRS[arm] / f"seed_{trainer.TRAIN_SEED}"
    checkpoint = run / "checkpoint_final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    adapter_hashes = {
        name: registry.sha256_file(checkpoint / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    hashes = {
        "selector_manifest_sha256": "1" * 64,
        "train_file_sha256": "2" * 64,
        "candidate_pool_sha256": "3" * 64,
        "hyperparameters_sha256": "4" * 64,
    }
    manifest = {
        "protocol": trainer.PROTOCOL,
        "status": "PASS",
        "source_commit": source_commit,
        "arm": arm,
        "mode": "formal",
        "official_test_used": False,
        "identities": {
            "model": trainer.MODEL_ID,
            "model_revision": trainer.MODEL_REVISION,
            "tokenizer_revision": trainer.TOKENIZER_REVISION,
        },
        "hyperparameters": {"train_seed": trainer.TRAIN_SEED},
        "adapter_files_sha256": adapter_hashes,
        **hashes,
    }
    audit = {"training_finite": True, "official_test_used": False}
    metrics = {
        "protocol": trainer.PROTOCOL,
        "arm": arm,
        "mode": "formal",
        "finite_audit": {"status": "PASS"},
    }
    write_json(run / "audit.json", audit)
    write_json(run / "training_metrics.json", metrics)
    write_json(run / "run_manifest.json", manifest)
    write_json(
        run / "files_sha256.json",
        {
            "audit.json": registry.sha256_file(run / "audit.json"),
            "training_metrics.json": registry.sha256_file(
                run / "training_metrics.json"
            ),
            "run_manifest.json": registry.sha256_file(run / "run_manifest.json"),
            "checkpoint_final/adapter_config.json": adapter_hashes[
                "adapter_config.json"
            ],
            "checkpoint_final/adapter_model.safetensors": adapter_hashes[
                "adapter_model.safetensors"
            ],
        },
    )


def test_build_registry_and_fail_closed_on_checkpoint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    training = tmp_path / "training"
    for arm in trainer.ARMS:
        make_run(training, arm, commit)
    monkeypatch.setattr(registry, "current_commit", lambda: commit)
    output = tmp_path / "checkpoint_registry.json"
    result = registry.build(
        training_root=training,
        output=output,
        source_commit=commit,
        user_judge_model="Qwen/Qwen2.5-14B-Instruct-AWQ",
        user_judge_revision="b" * 40,
        user_judge_alias="openai/v6-user-judge",
    )
    assert result["protocol"] == registry.PROTOCOL
    assert result["registered_arms"] == list(trainer.ARMS)
    assert result["official_test_sealed"] is True
    assert output.is_file()

    output.unlink()
    checkpoint = (
        training
        / registry.RUN_DIRS["full_proposed"]
        / f"seed_{trainer.TRAIN_SEED}"
        / "checkpoint_final"
        / "adapter_model.safetensors"
    )
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(registry.RegistryError, match="checkpoint bytes"):
        registry.build(
            training_root=training,
            output=output,
            source_commit=commit,
            user_judge_model="Qwen/Qwen2.5-14B-Instruct-AWQ",
            user_judge_revision="b" * 40,
            user_judge_alias="openai/v6-user-judge",
        )


def test_rejects_cross_arm_hyperparameter_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "c" * 40
    training = tmp_path / "training"
    for arm in trainer.ARMS:
        make_run(training, arm, commit)
    monkeypatch.setattr(registry, "current_commit", lambda: commit)
    run = (
        training
        / registry.RUN_DIRS["random_stratified"]
        / f"seed_{trainer.TRAIN_SEED}"
    )
    manifest = registry.read_json(run / "run_manifest.json")
    manifest["hyperparameters_sha256"] = "9" * 64
    write_json(run / "run_manifest.json", manifest)
    files = registry.read_json(run / "files_sha256.json")
    files["run_manifest.json"] = registry.sha256_file(run / "run_manifest.json")
    write_json(run / "files_sha256.json", files)
    with pytest.raises(registry.RegistryError, match="differ across"):
        registry.build(
            training_root=training,
            output=tmp_path / "registry.json",
            source_commit=commit,
            user_judge_model="Qwen/Qwen2.5-14B-Instruct-AWQ",
            user_judge_revision="d" * 40,
            user_judge_alias="openai/v6-user-judge",
        )
