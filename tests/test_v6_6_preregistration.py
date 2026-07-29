from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import prepare_v6_candidate_registry as registry
from scripts import run_v6_candidate_generation as generation


ROOT = Path(__file__).resolve().parents[1]


def args(**overrides):
    values = {
        "teacher_model": "teacher",
        "teacher_revision": "teacher-revision",
        "teacher_api_base": "http://teacher.invalid/v1",
        "user_model": "user",
        "user_revision": "user-revision",
        "user_api_base": "http://user.invalid/v1",
        "judge_model": "judge",
        "judge_revision": "judge-revision",
        "judge_api_base": "http://judge.invalid/v1",
        "max_tokens": 512,
        "max_steps": 60,
        "timeout": 900.0,
        "clean_attempts": 2,
        "recovery_attempts": 2,
        "clean_agent_mode": "single_turn_user_reference_replay",
        "recovery_continuation_mode": "deterministic_reference_completion",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_v6_6_preregistration_freezes_only_clean_prefix_delta():
    config = yaml.safe_load(
        (ROOT / "configs" / "v6_6_single_turn_clean_prefix.yaml").read_text(
            encoding="utf-8"
        )
    )
    delta = config["frozen_delta"]
    assert config["protocol"] == "v6_6_single_turn_user_reference_replay_v1"
    assert config["protocol"] == registry.V6_6_SINGLE_TURN_CLEAN_PREFIX_PROTOCOL
    assert config["protocol"] in registry.ALLOWED_DESIGN_PROTOCOLS
    assert registry._config_contract(config)["official_test_checks"] == {
        "used_false": True,
        "sealed_true": True,
        "content_not_exported": True,
    }
    assert (
        delta["only_scientific_change"]
        == "single_turn_user_prefix_before_reference_clean_replay"
    )
    assert delta["task_ids_changed"] is False
    assert delta["registry_or_error_branches_changed"] is False
    assert delta["gates_changed"] is False
    assert delta["selectors_or_weights_changed"] is False
    assert delta["models_or_revisions_changed"] is False
    assert delta["seeds_or_decoding_changed"] is False
    assert delta["deterministic_recovery_completion_changed"] is False
    assert delta["training_or_evaluation_changed"] is False
    assert config["official_test"] == {
        "sealed": True,
        "used": False,
        "unseal_rule": (
            "once only after protocol, pool, scores, manifests, training, "
            "checkpoints, and model-selection decision are frozen and hashed"
        ),
    }


def test_single_turn_prefix_retains_only_greeting_and_first_user_response():
    prefix = generation.single_turn_clean_prefix(
        [
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "I need help with an order."},
        ]
    )
    assert [row["role"] for row in prefix] == ["assistant", "user"]
    assert generation.first_assistant_tool_index(prefix) is None


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "assistant", "content": "hello"}],
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "lookup"}],
            },
            {"role": "user", "content": "hello"},
        ],
        [
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "result"},
        ],
    ],
)
def test_single_turn_prefix_fails_closed_on_shape_or_tool_leakage(messages):
    with pytest.raises(generation.V6GenerationError):
        generation.single_turn_clean_prefix(messages)


def test_semantic_contract_binds_single_turn_budget_and_mode():
    contract = generation.semantic_generation_contract(
        args(),
        continuation_seeds=[20260806, 20260807, 20260808],
        design_protocol="v6_6_single_turn_user_reference_replay_v1",
    )
    assert contract["clean_agent_mode"] == "single_turn_user_reference_replay"
    assert contract["clean_prefix_max_steps"] == 2
    assert contract["clean_prefix_stops_before_assistant_tool_action"] is True
    assert (
        contract["clean_agent_name"]
        == "single_turn_user_then_deterministic_tau2_reference_replay"
    )
    assert contract["recovery_continuation_mode"] == (
        "deterministic_reference_completion"
    )
    assert contract["gold_clean_future_visible"] is False


def test_generator_accepts_v6_6_registry_protocol():
    payload = {
        "protocol": registry.REGISTRY_PROTOCOL,
        "design_protocol": registry.V6_6_SINGLE_TURN_CLEAN_PREFIX_PROTOCOL,
        "selection_unit": "candidate_pair",
        "grouping_unit": "choice_set",
        "official_test_used": False,
        "official_test_sealed": True,
        "official_test_task_content_exported": False,
        "official_test_identity_overlap_count": 0,
        "structural_eligibility_sha256": (
            generation.protocol.STRUCTURAL_ELIGIBILITY_SHA256
        ),
        "phase_registry": {
            "pilot": {"task_ids": ["retail:104"]},
            "formal": {"task_ids": ["retail:104"]},
        },
        "candidate_pairs": [
            {
                "candidate_pair_id": "v6:pilot:retail:104:pair:1",
                "choice_set_id": "v6:pilot:retail:104:choice",
                "phase": "pilot",
                "partition": "arm_train",
                "task_identity": "retail:104",
                "domain": "retail",
                "task_id": "104",
                "prefix_sha256": "a" * 64,
                "environment_snapshot_sha256": "b" * 64,
                "branches": [{}, {}],
                "official_test_used": False,
            }
        ],
    }
    payload["candidate_pairs"][0]["candidate_pair_sha256"] = generation.sha256(
        payload["candidate_pairs"][0]
    )
    payload["registry_sha256"] = generation.sha256(payload)
    assert generation.verify_registry(payload) == payload["registry_sha256"]
