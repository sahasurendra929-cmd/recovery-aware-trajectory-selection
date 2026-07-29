from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts import prepare_v6_candidate_registry as registry
from scripts import run_v6_candidate_generation as generation


ROOT = Path(__file__).resolve().parents[1]


def test_v6_9_config_freezes_single_delta():
    payload = yaml.safe_load(
        (ROOT / "configs" / "v6_9_complementizer_normalization.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload["protocol"] == "v6_9_complementizer_normalization_v1"
    assert payload["frozen_delta"]["only_scientific_change"] == (
        "strip_one_leading_that_after_tell_user_prefix"
    )
    assert payload["frozen_delta"]["task_ids_changed"] is False
    assert payload["smoke"]["tasks"] == [
        "retail:19",
        "retail:16",
        "retail:104",
    ]


def test_v6_9_protocol_is_registry_allowlisted():
    assert (
        registry.V6_9_COMPLEMENTIZER_NORMALIZATION_PROTOCOL
        in registry.ALLOWED_DESIGN_PROTOCOLS
    )


def test_v6_9_strips_one_leading_complementizer_for_retail_19():
    first = generation.explicit_user_direct_assertion(
        "Agent should tell the user that returning the water bottle gives "
        "a refund of $54.04.",
        strip_leading_that=True,
    )
    second = generation.explicit_user_direct_assertion(
        "Agent should tell the user that exchanging the pet bed and office "
        "chair saves $41.64 total.",
        strip_leading_that=True,
    )
    assert first == (
        "I am telling you directly: returning the water bottle gives "
        "a refund of $54.04."
    )
    assert second == (
        "I am telling you directly: exchanging the pet bed and office "
        "chair saves $41.64 total."
    )


def test_v6_8_renderer_remains_reproducible():
    assertion = "Agent should tell the user that the refund is $20."
    assert generation.explicit_user_direct_assertion(assertion) == (
        "I am telling you directly: that the refund is $20."
    )


def test_v6_9_completion_uses_normalized_assertion():
    criteria = SimpleNamespace(
        communicate_info=["54.04"],
        nl_assertions=[
            "Agent should tell the user that returning the water bottle "
            "gives a refund of $54.04."
        ],
    )
    message = generation.deterministic_completion_message(
        criteria,
        renderer="explicit_user_direct_v3",
        fallback="unused",
    )
    assert "directly: that " not in message["content"]
    assert "returning the water bottle gives a refund of $54.04." in (
        message["content"]
    )
