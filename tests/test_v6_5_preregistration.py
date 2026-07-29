from argparse import Namespace
from pathlib import Path

import yaml

from scripts import prepare_v6_candidate_registry as registry
from scripts import run_v6_candidate_generation as generation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v6_5_reference_completion.yaml"


def test_v6_5_preregisters_one_delta_and_keeps_frozen_statistics():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"] == registry.V6_5_REFERENCE_COMPLETION_PROTOCOL
    assert config["design_version"] == "6.5"
    assert config["supersedes"]["results_must_not_be_merged"] is True
    assert config["frozen_delta"]["only_scientific_change"] == (
        "forced_correction_plus_all_other_reference_actions_once"
    )
    assert config["selectors"]["full_proposed"]["weights"] == {
        "causal": 0.50,
        "hardness": 0.25,
        "coverage": 0.25,
    }
    frozen = config["training_and_evaluation"]
    assert frozen["clean_noninferiority_margin"] == -0.05
    assert frozen["task_cluster_bootstrap_replicates"] == 10000


def test_reference_completion_mode_is_bound_into_semantic_contract():
    args = Namespace(
        judge_model=None,
        judge_revision=None,
        judge_api_base=None,
        user_model="user",
        user_revision="user-rev",
        user_api_base="http://user/v1",
        teacher_model="teacher",
        teacher_revision="teacher-rev",
        teacher_api_base="http://teacher/v1",
        max_tokens=512,
        max_steps=60,
        timeout=900.0,
        clean_attempts=2,
        recovery_attempts=2,
        clean_agent_mode="deterministic_reference_replay",
        recovery_continuation_mode="deterministic_reference_completion",
    )
    contract = generation.semantic_generation_contract(
        args,
        continuation_seeds=(20260806, 20260807, 20260808),
        design_protocol=registry.V6_5_REFERENCE_COMPLETION_PROTOCOL,
    )
    assert contract["recovery_continuation_mode"] == (
        "deterministic_reference_completion"
    )
    assert contract["design_protocol"] == (
        registry.V6_5_REFERENCE_COMPLETION_PROTOCOL
    )


class FakeAction:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments

    def model_dump(self, mode: str) -> dict:
        assert mode == "json"
        return {"name": self.name, "arguments": self.arguments, "requestor": "assistant"}


def test_completion_contract_has_unique_match_requirement():
    forced = {"name": "book", "arguments": {"id": "new"}, "requestor": "assistant"}
    actions = [
        FakeAction("cancel", {"id": "old"}),
        FakeAction("book", {"id": "new"}),
    ]
    matches = [
        index
        for index, action in enumerate(actions)
        if generation.call_semantics(action.model_dump(mode="json"))
        == generation.call_semantics(forced)
    ]
    assert matches == [1]
    remaining = [action.name for index, action in enumerate(actions) if index != matches[0]]
    assert remaining == ["cancel"]

    duplicate = [actions[1], actions[1]]
    duplicate_matches = [
        action
        for action in duplicate
        if generation.call_semantics(action.model_dump(mode="json"))
        == generation.call_semantics(forced)
    ]
    assert len(duplicate_matches) == 2
