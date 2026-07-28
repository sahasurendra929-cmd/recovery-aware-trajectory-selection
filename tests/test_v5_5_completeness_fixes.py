from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import build_v5_5_confirmation_freeze as freeze
from scripts import prepare_v5_5_natural_pairs as natural
from scripts import run_v5_5_full as controller
from scripts import train_v5_sft_causal as trainer
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


def test_user_judge_port_avoids_managed_container_endpoint(tmp_path: Path):
    args = SimpleNamespace(
        results_root=tmp_path,
        serve_python=Path("python"),
        tau2_root=tmp_path / "tau2",
        split_manifest=tmp_path / "split.json",
        validation_manifest=tmp_path / "validation.json",
        protocol_audit=tmp_path / "audit.json",
        registry=tmp_path / "registry.json",
    )
    command, _ = controller.evaluation_command(
        args,
        arm="perfect_success",
        training_seed=20260805,
        evaluation_seed=20260805,
        shard=0,
    )
    endpoint = command[command.index("--user-api-base") + 1]
    assert controller.USER_JUDGE_PORT == 8201
    assert endpoint == "http://127.0.0.1:8201/v1"


def test_service_readiness_allows_shared_storage_cold_start():
    assert controller.SERVICE_READY_TIMEOUT_SECONDS == 1800.0


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
