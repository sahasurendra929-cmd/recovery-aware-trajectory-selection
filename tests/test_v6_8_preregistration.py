import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import prepare_v6_candidate_registry as registry
from scripts import run_v6_candidate_generation as generation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v6_8_explicit_assertion_renderer.yaml"


def test_v6_8_config_freezes_only_explicit_renderer_delta():
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert payload["protocol"] == "v6_8_explicit_assertion_renderer_v1"
    delta = payload["frozen_delta"]
    assert (
        delta["only_scientific_change"]
        == "explicit_first_person_user_direct_renderer_v2"
    )
    assert delta["task_ids_changed"] is False
    assert delta["registry_or_error_branches_changed"] is False
    assert delta["gates_changed"] is False
    assert delta["models_revisions_seeds_or_decoding_changed"] is False
    assert delta["selectors_weights_training_or_evaluation_changed"] is False
    assert payload["smoke"]["tasks"] == ["retail:16", "retail:104"]


def test_v6_8_protocol_is_registry_allowlisted():
    assert (
        registry.V6_8_EXPLICIT_ASSERTION_RENDERER_PROTOCOL
        in registry.ALLOWED_DESIGN_PROTOCOLS
    )


def test_retail_16_assertion_is_explicitly_user_direct():
    rendered = generation.explicit_user_direct_assertion(
        "Agent should tell the user the total refund amount is $8,276.23."
    )
    assert rendered == (
        "I am telling you directly: the total refund amount is $8,276.23."
    )


def test_retail_104_assertion_is_explicitly_user_direct():
    rendered = generation.explicit_user_direct_assertion(
        "Agent should provide the tracking number 286422338955."
    )
    assert rendered == (
        "I am providing this directly to you: "
        "the tracking number 286422338955."
    )


@pytest.mark.parametrize(
    "assertion",
    [
        "Agent should not approve the return.",
        "Agent should cancel the pending order.",
        "Agent communicates that the refund is pending.",
        "Agent updates the delivery address.",
        "For this reservation Agent charges $20.",
        "Check that Agent clearly identifies that the order is pending.",
        "Check that agent correctly adds the bag.",
    ],
)
def test_v6_8_reuses_v6_7_supported_assertion_table(assertion):
    rendered = generation.explicit_user_direct_assertion(assertion)
    assert rendered.startswith("I am confirming this directly to you: ")
    assert "Agent should" not in rendered
    assert "Check that" not in rendered


def test_v6_8_unknown_meta_assertion_fails_closed():
    with pytest.raises(generation.V6GenerationError):
        generation.explicit_user_direct_assertion(
            "Agent unexpectedly contemplates the request."
        )


def test_v6_8_completion_omits_generic_boilerplate_and_is_deduplicated():
    criteria = SimpleNamespace(
        communicate_info=["8276.23"],
        nl_assertions=[
            "Agent should tell the user the total refund amount is $8,276.23.",
            "Agent should tell the user the total refund amount is $8,276.23.",
        ],
    )
    message = generation.deterministic_completion_message(
        criteria,
        renderer="explicit_user_direct_v2",
        fallback="The request is complete.",
    )
    assert "The requested work is complete." not in message["content"]
    assert message["content"].splitlines() == [
        "I am providing the requested information directly to you: 8276.23.",
        "I am telling you directly: the total refund amount is $8,276.23.",
    ]


def test_v6_8_semantic_contract_binds_renderer():
    args = argparse.Namespace(
        teacher_model="teacher",
        teacher_revision="teacher-revision",
        teacher_api_base="http://teacher/v1",
        user_model="user",
        user_revision="user-revision",
        user_api_base="http://user/v1",
        judge_model="judge",
        judge_revision="judge-revision",
        judge_api_base="http://judge/v1",
        max_tokens=512,
        max_steps=60,
        timeout=900.0,
        clean_attempts=2,
        clean_agent_mode="single_turn_user_reference_replay",
        recovery_attempts=2,
        recovery_continuation_mode="deterministic_reference_completion",
        completion_renderer="explicit_user_direct_v2",
    )
    contract = generation.semantic_generation_contract(
        args,
        design_protocol="v6_8_explicit_assertion_renderer_v1",
        continuation_seeds=(20260806, 20260807, 20260808),
    )
    assert contract["completion_renderer"] == "explicit_user_direct_v2"
