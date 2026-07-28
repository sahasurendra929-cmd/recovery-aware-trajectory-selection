from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import build_v5_5_confirmation_freeze as freeze
from scripts import prepare_v5_5_natural_pairs as natural
from scripts import run_v5_5_full as controller
from scripts import run_v5_5_end_to_end_eval as evaluator
from scripts import train_v5_sft_causal as trainer
from scripts import v5_5_full_protocol as full_protocol
from scripts import v5_5_protocol as pair_protocol


def test_v5_5_context_capacity_covers_frozen_reference_rows():
    assert trainer.MAX_SEQUENCE_TOKENS == 8192
    assert trainer.V5_5_MAX_SEQUENCE_TOKENS == 10240


def test_manifest_source_check_rejects_supervised_suffix_tampering():
    messages = [
        {"role": "user", "content": "help"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "name": "get_order_details", "arguments": {"order_id": "#W1"}}]},
        {"role": "tool", "id": "c", "content": "{}", "error": False},
    ]
    source = {
        "task_identity": "retail:1",
        "domain": "retail",
        "task_id": "1",
        "messages": messages,
        "calls": [messages[1]["tool_calls"][0]],
        "selected_message_index": 1,
        "action_index": 0,
        "source_simulation_sha256": "a" * 64,
    }
    pair = {
        "task_identity": "retail:1",
        "domain": "retail",
        "task_id": "1",
        "source_simulation_sha256": "a" * 64,
        "source_call_sequence": source["calls"],
        "selected_action_index": 0,
        "clean_prefix": messages[:1],
        "supervised_messages": messages[1:],
        "supervised_suffix_sha256": pair_protocol.semantic_sha256(messages[1:]),
    }
    assert all(natural.manifest_source_checks(pair, source).values())
    pair["supervised_messages"] = pair["supervised_messages"][:-1]
    checks = natural.manifest_source_checks(pair, source)
    assert checks["stored_supervised_suffix_matches_manifest"] is False


def test_blackwell_service_commands_force_vllm_eager():
    registry = {
        "user_judge": {"model": "judge", "revision": "r", "model_id": "openai/judge"},
        "registered_arms": ["perfect_success"],
        "registered_training_seeds": [20260805],
        "entries": {"perfect_success": {"20260805": {"model_id": "openai/a", "checkpoint": {"path": "/tmp/a"}}}},
        "base_model": "base",
        "base_model_revision": "r",
        "base_model_alias": "openai/base",
    }
    args = SimpleNamespace(serve_python=Path("python"))
    for user_judge in (False, True):
        command = controller.service_command(
            args, gpu=0, port=8001, registry=registry, user_judge=user_judge
        )
        assert "--enforce-eager" in command
        assert "--enable-auto-tool-choice" in command
        assert command[command.index("--tool-call-parser") + 1] == "hermes"


def test_user_judge_port_avoids_managed_container_endpoint(tmp_path: Path):
    args = SimpleNamespace(
        results_root=tmp_path,
        serve_python=Path("python"),
        tau2_root=tmp_path / "tau2",
        split_manifest=tmp_path / "split.json",
        validation_manifest=tmp_path / "validation.json",
        protocol_audit=tmp_path / "audit.json",
        registry=tmp_path / "registry.json",
        source_commit="f" * 40,
    )
    command, _ = controller.evaluation_command(
        args,
        arm="perfect_success",
        training_seed=20260805,
        evaluation_seed=20260805,
        shard=0,
    )
    endpoint = command[command.index("--user-api-base") + 1]
    evaluation_source = command[
        command.index("--evaluation-source-commit") + 1
    ]
    assert controller.USER_JUDGE_PORT == 8201
    assert endpoint == "http://127.0.0.1:8201/v1"
    assert evaluation_source == args.source_commit


def test_service_readiness_allows_shared_storage_cold_start():
    assert controller.SERVICE_READY_TIMEOUT_SECONDS == 1800.0


def test_base_diagnostic_grid_is_single_seed_and_does_not_train():
    arms, seeds = controller.selected_grid("base-diagnostic")
    assert arms == ("base_control",)
    assert seeds == (full_protocol.TRAINING_SEEDS[0],)
    assert controller.selected_evaluation_seeds("base-diagnostic") == (
        full_protocol.EVALUATION_SEEDS[0],
    )
    assert controller.selected_evaluation_seeds(
        "reference-screen"
    ) == full_protocol.EVALUATION_SEEDS


def test_low_lr_screen_changes_only_training_learning_rate(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "hashes.json").write_text(
        json.dumps(
            {
                "arms/perfect_success/train.jsonl": "a" * 64,
                "validation_loss.jsonl": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        train_python=Path("python"),
        data_root=data_root,
        base_model_revision="c" * 40,
        source_commit="d" * 40,
        local_files_only=True,
        experiment_mode="low-lr-screen",
    )
    arms, seeds = controller.selected_grid(args.experiment_mode)
    assert arms == controller.SCREEN_ARMS
    assert seeds == (full_protocol.TRAINING_SEEDS[0],)
    assert (
        controller.selected_evaluation_seeds(args.experiment_mode)
        == full_protocol.EVALUATION_SEEDS
    )
    command = controller.training_command(
        args,
        arm=controller.SCREEN_ARMS[0],
        seed=seeds[0],
        mode="formal",
        output=tmp_path / "training",
    )
    assert command[command.index("--learning-rate") + 1] == str(
        controller.LOW_LR
    )


def test_reduced_exposure_screen_keeps_low_lr_and_halves_steps(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "hashes.json").write_text(
        json.dumps(
            {
                "arms/perfect_success/train.jsonl": "a" * 64,
                "validation_loss.jsonl": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        train_python=Path("python"),
        data_root=data_root,
        base_model_revision="c" * 40,
        source_commit="d" * 40,
        local_files_only=True,
        experiment_mode="reduced-exposure-screen",
    )
    arms, seeds = controller.selected_grid(args.experiment_mode)
    assert arms == controller.SCREEN_ARMS
    assert seeds == (full_protocol.TRAINING_SEEDS[0],)
    assert (
        controller.selected_evaluation_seeds(args.experiment_mode)
        == full_protocol.EVALUATION_SEEDS
    )
    command = controller.training_command(
        args,
        arm=controller.SCREEN_ARMS[0],
        seed=seeds[0],
        mode="formal",
        output=tmp_path / "training",
    )
    assert command[command.index("--learning-rate") + 1] == str(
        controller.LOW_LR
    )
    assert command[command.index("--formal-steps") + 1] == str(
        controller.REDUCED_EXPOSURE_STEPS
    )


def test_base_diagnostic_uses_revision_pinned_registry_base(tmp_path: Path):
    audit = tmp_path / "audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    _write(
        registry,
        {
            "protocol": evaluator.REGISTRY_PROTOCOL,
            "design_protocol": full_protocol.PROTOCOL,
            "design_version": full_protocol.DESIGN_VERSION,
            "official_test_used": False,
            "official_test_sealed": True,
            "registered_arms": ["perfect_success"],
            "registered_training_seeds": [20260805],
            "entries": {},
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "base_model_revision": "a" * 40,
            "base_model_alias": "openai/v55-base",
            "data": {
                "pair_mode": "reference",
                "claim_level": "diagnostic_only",
                "audit_path": str(audit),
                "audit_sha256": evaluator.sha256_file(audit),
            },
        },
    )
    loaded, entry = evaluator.load_registry(
        registry,
        arm="base_control",
        training_seed=20260805,
    )
    assert loaded["base_model_alias"] == "openai/v55-base"
    assert entry == {
        "trainer_arm": "base_control",
        "paper_arm": "base_unadapted",
        "training_seed": 20260805,
        "model_id": "openai/v55-base",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "base_model_revision": "a" * 40,
        "checkpoint": None,
        "diagnostic_only": True,
    }


def test_evaluation_inputs_fail_before_service_start(tmp_path: Path):
    args = SimpleNamespace(
        split_manifest=tmp_path / "split.json",
        validation_manifest=tmp_path / "validation.json",
        protocol_audit=tmp_path / "audit.json",
        registry=tmp_path / "registry.json",
    )
    for path in (
        args.split_manifest,
        args.validation_manifest,
        args.protocol_audit,
        args.registry,
    ):
        path.write_text("{}", encoding="utf-8")
    controller.require_evaluation_inputs(args)

    args.validation_manifest.unlink()
    try:
        controller.require_evaluation_inputs(args)
    except RuntimeError as error:
        assert "validation_manifest=" in str(error)
        assert "before service startup" in str(error)
    else:
        raise AssertionError("missing validation manifest was accepted")


def test_completed_evaluation_batch_requires_audited_outputs(tmp_path: Path):
    outputs = [tmp_path / f"shard-{shard}" for shard in range(3)]
    source_commit = "a" * 40
    for shard, output in enumerate(outputs):
        output.mkdir()
        (output / "rows.jsonl").write_text(
            "".join(
                json.dumps({"task_id": f"{output.name}-{index}"}) + "\n"
                for index in range(14)
            ),
            encoding="utf-8",
        )
        common = {
            "status": "PASS",
            "arm": "repair_50",
            "training_seed": 20260805,
            "evaluation_seed": 20260817,
            "official_test_used": False,
        }
        _write(output / "metrics.json", {**common, "rows": 14})
        _write(
            output / "run_contract.json",
            {
                **common,
                "status": "COMPLETE",
                "evaluation_source_commit": source_commit,
            },
        )
        for domain in ("retail", "airline"):
            for condition in ("clean", "error"):
                _write(
                    output
                    / (
                        f"{domain}_{condition}.shard-"
                        f"{shard:03d}-of-003.json"
                    ),
                    {"simulations": [{}]},
                )
        (output.parent / f"{output.name}.console.log").write_text(
            "complete\n",
            encoding="utf-8",
        )
    kwargs = {
        "evaluation_source_commit": source_commit,
        "arm": "repair_50",
        "training_seed": 20260805,
        "evaluation_seed": 20260817,
    }
    assert controller.evaluation_batch_complete(outputs, **kwargs)

    contract = json.loads(
        (outputs[0] / "run_contract.json").read_text(encoding="utf-8")
    )
    contract["status"] = "PASS"
    _write(outputs[0] / "run_contract.json", contract)
    assert not controller.evaluation_batch_complete(outputs, **kwargs)
    contract["status"] = "COMPLETE"
    _write(outputs[0] / "run_contract.json", contract)
    assert controller.evaluation_batch_complete(outputs, **kwargs)

    (outputs[1] / "rows.jsonl").write_text("", encoding="utf-8")
    assert not controller.evaluation_batch_complete(outputs, **kwargs)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_confirmation_freeze_requires_natural_paper_gate(tmp_path: Path):
    summary = tmp_path / "summary.json"
    registry = tmp_path / "registry.json"
    audit = tmp_path / "audit.json"
    _write(summary, {"protocol": "v5_5_task_cluster_summary_v1", "status": "PASS", "official_test_used": False, "statistics": {"selection": {"selected_arm": "repair_50", "selected_paper_arm": "r50", "positive_screen": True}}})
    _write(registry, {"protocol": "v5_5_checkpoint_registry_v1", "official_test_used": False, "official_test_sealed": True, "entries": {"perfect_success": {}, "repair_50": {}}})
    _write(audit, {"protocol": "v5_5_natural_counterfactual_pairs_v1", "status": "PASS_TRAINING_AUTHORIZED", "official_test_used": False, "official_test_sealed": True, "observed": {"paper_target_tasks_reached": True, "paper_target_pairs_reached": True}})
    payload = freeze.build(summary, registry, audit)
    assert payload["comparison"] == ["r0_perfect", "r50"]
    assert payload["official_test_used"] is False
