from argparse import Namespace
from pathlib import Path

import yaml

from scripts import prepare_v6_candidate_registry as registry
from scripts import run_v6_candidate_generation as generation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v6_3_deterministic_clean_replay.yaml"


def test_v6_3_is_explicitly_versioned_and_preserves_recovery_teacher():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"] == registry.V6_3_DETERMINISTIC_CLEAN_REPLAY_PROTOCOL
    assert config["design_version"] == "6.3"
    assert config["supersedes"]["results_must_not_be_merged"] is True
    assert config["frozen_delta"]["only_scientific_change"] == (
        "deterministic_reference_action_clean_replay"
    )
    assert config["models"]["trajectory_teacher"]["model"] == (
        "Qwen/Qwen2.5-72B-Instruct-AWQ"
    )


def test_deterministic_clean_mode_is_bound_into_semantic_contract():
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
    )
    contract = generation.semantic_generation_contract(
        args,
        continuation_seeds=(20260821,),
        design_protocol=registry.V6_3_DETERMINISTIC_CLEAN_REPLAY_PROTOCOL,
    )
    assert contract["clean_agent_mode"] == "deterministic_reference_replay"
    assert contract["clean_agent_name"] == "deterministic_tau2_reference_replay"
    assert contract["agent_name"] == generation.AGENT_NAME
