from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import prepare_v5_5_full_sft as data
from scripts import train_v5_sft_causal as trainer
from scripts import v5_5_protocol as pair_protocol


class PrefixStableTokenizer:
    """Small tokenizer double that preserves chat-template prefix semantics."""

    roles = {"system": 11, "user": 12, "assistant": 13, "tool": 14}

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is True
        tokens = [1, len(tools)]
        for message in messages:
            tokens.append(self.roles[message["role"]])
            payload = json.dumps(
                {
                    key: value
                    for key, value in message.items()
                    if key != "role"
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            tokens.extend(20 + byte for byte in payload)
            tokens.append(2)
        if add_generation_prompt:
            tokens.append(self.roles["assistant"])
        return tokens


def tool_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]


def valid_pair(index: int) -> dict:
    correct = {
        "id": f"correct-{index}",
        "name": "get_order_details",
        "arguments": {"order_id": f"#W{index:03d}"},
        "requestor": "assistant",
    }
    injected = deepcopy(correct)
    injected["id"] = f"failed-{index}"
    injected["arguments"]["order_id"] += "X"
    failed = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [injected],
        },
        {
            "role": "tool",
            "id": injected["id"],
            "content": "Error: not found",
            "error": True,
        },
    ]
    supervised = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [correct],
        },
        {
            "role": "tool",
            "id": correct["id"],
            "content": '{"status":"ok"}',
            "error": False,
        },
        {"role": "assistant", "content": "I found the order."},
    ]
    prefix: list[dict] = []
    return {
        "protocol": pair_protocol.PROTOCOL,
        "pair_id": f"retail:{index}:counterfactual:1",
        "task_identity": f"retail:{index}",
        "domain": "retail",
        "task_id": str(index),
        "mutation_variant": 0,
        "identifier_key": "order_id",
        "task_context": {"description": f"task {index}"},
        "clean_call": correct,
        "injected_call": injected,
        "injected_result": failed[1],
        "correction_call": correct,
        "correction_result": supervised[1],
        "clean_prefix": prefix,
        "recovery_prompt": failed,
        "failed_event": failed,
        "supervised_messages": supervised,
        "supervision_starts_after_error": True,
        "clean_future_present_in_recovery_prompt": False,
        "clean_prefix_sha256": pair_protocol.semantic_sha256(prefix),
        "recovery_prefix_sha256": pair_protocol.semantic_sha256(prefix),
        "error_event_sha256": pair_protocol.semantic_sha256(failed),
        "supervised_suffix_sha256": pair_protocol.semantic_sha256(supervised),
        "clean_end_state_matches_reference": True,
        "recovery_end_state_matches_reference": True,
        "clean_agent_db_hash": f"agent-{index}",
        "clean_user_db_hash": None,
        "recovery_agent_db_hash": f"agent-{index}",
        "recovery_user_db_hash": None,
        "reference_actions_are_natural_dialogue": False,
        "independent_environment_replay_pass": False,
        "official_test_used": False,
    }


@pytest.fixture
def contexts():
    context = {"policy": "frozen policy", "tool_schemas": tool_schemas()}
    return {"retail": context, "airline": context}


def test_materialization_masks_failure_and_matches_targets(contexts):
    pairs = data.materialize_source_pairs(
        [valid_pair(1)],
        pair_mode="reference",
        contexts=contexts,
        tokenizer=PrefixStableTokenizer(),
        independent_replay_authorized=True,
    )
    assert len(pairs) == 1
    clean = pairs[0]["clean"]
    recovery = pairs[0]["recovery"]
    assert clean["token_contract"]["supervised_tokens"] == recovery[
        "token_contract"
    ]["supervised_tokens"]
    failed = recovery["metadata"]["failed_assistant_message_indices"]
    assert len(failed) == 1
    assert recovery["label_mask"][failed[0]] is False
    assert trainer.validate_row(
        clean, arm="perfect_success", split="train"
    )
    assert trainer.validate_row(
        recovery, arm="repair_100", split="train"
    )


def test_four_row_blocks_realize_exact_token_doses(contexts):
    sources = data.materialize_source_pairs(
        [valid_pair(1), valid_pair(2)],
        pair_mode="reference",
        contexts=contexts,
        tokenizer=PrefixStableTokenizer(),
        independent_replay_authorized=True,
    )
    schedules, audits = data.build_arm_schedules(
        sources,
        seed=20260805,
    )
    assert set(schedules) == set(data.ARM_MAP)
    assert {len(rows) for rows in schedules.values()} == {512}
    assert {
        arm: audit["realized_recovery_supervised_token_ratio"]
        for arm, audit in audits.items()
    } == {
        "perfect_success": 0.0,
        "repair_25": 0.25,
        "repair_50": 0.5,
        "repair_75": 0.75,
        "repair_100": 1.0,
    }
    exposure = {
        arm: [row["metadata"]["pair_id"] for row in rows[::4]]
        for arm, rows in schedules.items()
    }
    assert len({tuple(value) for value in exposure.values()}) == 1


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_complete_bundle_binds_into_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contexts,
):
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        "".join(json.dumps(valid_pair(index), sort_keys=True) + "\n" for index in (1, 2)),
        encoding="utf-8",
    )
    pair_audit = tmp_path / "pair_audit.json"
    _write(
        pair_audit,
        {
            "protocol": pair_protocol.PROTOCOL,
            "status": "PASS_TRAINING_AUTHORIZED",
            "checks": {"official_test_sealed": True},
        },
    )
    pair_manifest = tmp_path / "pair_manifest.json"
    _write(
        pair_manifest,
        {
            "protocol": pair_protocol.PROTOCOL,
            "official_test_used": False,
        },
    )
    monkeypatch.setattr(data.v5, "load_tau2_contexts", lambda _: contexts)
    output = tmp_path / "processed"
    audit = data.prepare(
        tau2_root=tmp_path,
        pairs_path=pairs_path,
        pair_audit_path=pair_audit,
        pair_manifest_path=pair_manifest,
        output_dir=output,
        tokenizer=PrefixStableTokenizer(),
        tokenizer_name="fake",
        tokenizer_revision="a" * 40,
        pair_mode="reference",
        strict=False,
    )
    assert audit["status"] == "PASS"
    train_file = output / "arms" / "repair_50" / "train.jsonl"
    validation_file = output / "validation_loss.jsonl"
    provenance = trainer.validate_training_data_provenance(
        arm="repair_50",
        train_file=train_file,
        validation_file=validation_file,
        data_audit_path=output / "audit.json",
        data_hashes_path=output / "hashes.json",
        expected_train_sha256=hashlib.sha256(train_file.read_bytes()).hexdigest(),
        expected_validation_sha256=hashlib.sha256(
            validation_file.read_bytes()
        ).hexdigest(),
    )
    assert provenance["design_version"] == "5.5-full"
    assert provenance["design_provenance"]["pair_mode"] == "reference"
    assert provenance["official_test_used"] is False
