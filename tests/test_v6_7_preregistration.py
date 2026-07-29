from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml

from scripts import prepare_v6_candidate_registry as registry
from scripts import run_v6_candidate_generation as generation


ROOT = Path(__file__).resolve().parents[1]

REGISTERED_ASSERTION_SHAPES = [
    "Agent should tell the user the total refund amount is $8,276.23.",
    "Agent should provide the tracking number 286422338955.",
    "Agent should not approve the cancellation.",
    "Agent should not cancel reservation 9HBUV8 since it does not meet requirements.",
    "Agent should not offer any compensation.",
    "Agent should cancel reservation K1NW8N.",
    "Agent should book a reservation with the following flights: HAT023 and HAT204.",
    "Agent should exchange items in both orders.",
    "Agent should modify address for the order",
    "Agent should realize that flight was not cancel and not in business class.",
    "Agent communicates that total gift card balance is $327.",
    "Agent communicated that the $1786 will be charged to the mastercard.",
    "Agent mentions that total sum on gift cards is $327.",
    "Agent updates reservation OBUT9V return flights to HAT290 and HAT175.",
    "Agent assigns payment to gift_card_6276644.",
    "Agent add 2 free baggages to reservation HXDUBJ.",
    "Agent does not allow change to business class for only one leg.",
    "Agent does not offer compensation before the user asks.",
    "Agent cancels reservation K1NW8N.",
    "Agent books a round-trip reservation from JFK to SFO.",
    "Agent charges $500 on certificate_3765853.",
    "Agent verifies that the flight was delayed.",
    "For this reservation Agent charges $500 on a certificate.",
    "Check that Agent clearly identifies that all passengers must share a cabin.",
    "Check that agent correctly adds 2 checked bags for free.",
    "Reservation FQ8APE is updated to economy.",
]


def contract_args():
    return SimpleNamespace(
        teacher_model="teacher",
        teacher_revision="teacher-revision",
        teacher_api_base="http://teacher.invalid/v1",
        user_model="user",
        user_revision="user-revision",
        user_api_base="http://user.invalid/v1",
        judge_model="judge",
        judge_revision="judge-revision",
        judge_api_base="http://judge.invalid/v1",
        max_tokens=512,
        max_steps=60,
        timeout=900.0,
        clean_attempts=2,
        recovery_attempts=2,
        clean_agent_mode="single_turn_user_reference_replay",
        recovery_continuation_mode="deterministic_reference_completion",
        completion_renderer="natural_direct_v1",
    )


def test_v6_7_preregistration_and_registry_contract():
    config = yaml.safe_load(
        (ROOT / "configs" / "v6_7_natural_clean_completion.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["protocol"] == registry.V6_7_NATURAL_CLEAN_COMPLETION_PROTOCOL
    assert config["protocol"] in registry.ALLOWED_DESIGN_PROTOCOLS
    assert (
        config["frozen_delta"]["only_scientific_change"]
        == "natural_language_renderer_for_deterministic_completion"
    )
    assert (
        config["frozen_delta"][
            "same_renderer_for_clean_and_recovery_confirmation"
        ]
        is True
    )
    assert registry._config_contract(config)["official_test_checks"] == {
        "used_false": True,
        "sealed_true": True,
        "content_not_exported": True,
    }


@pytest.mark.parametrize("assertion", REGISTERED_ASSERTION_SHAPES)
def test_registered_assertion_shapes_become_direct_speech(assertion):
    rendered = generation.naturalize_completion_assertion(assertion)
    assert rendered
    assert not re.search(r"\b(?:Agent|agent|should)\b|Check that", rendered)


def test_retail_104_completion_is_direct_and_keeps_tracking_number():
    criteria = SimpleNamespace(
        communicate_info=["286422338955"],
        nl_assertions=["Agent should provide the tracking number 286422338955."],
    )
    message = generation.deterministic_completion_message(
        criteria,
        renderer="natural_direct_v1",
        fallback="unused",
    )
    assert message["role"] == "assistant"
    assert "Requested information: 286422338955." in message["content"]
    assert "Here is the tracking number 286422338955." in message["content"]
    assert "Agent should" not in message["content"]


def test_unknown_meta_assertion_fails_closed():
    with pytest.raises(
        generation.V6GenerationError, match="unsupported meta-level"
    ):
        generation.naturalize_completion_assertion(
            "Agent should contemplate an unregistered action."
        )


def test_semantic_contract_binds_shared_natural_renderer():
    contract = generation.semantic_generation_contract(
        contract_args(),
        continuation_seeds=[20260806, 20260807, 20260808],
        design_protocol=registry.V6_7_NATURAL_CLEAN_COMPLETION_PROTOCOL,
    )
    assert contract["completion_renderer"] == "natural_direct_v1"
    assert contract["clean_agent_mode"] == "single_turn_user_reference_replay"
    assert contract["recovery_continuation_mode"] == (
        "deterministic_reference_completion"
    )
