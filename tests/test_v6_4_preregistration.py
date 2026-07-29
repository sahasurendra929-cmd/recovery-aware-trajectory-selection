from argparse import Namespace
from pathlib import Path

import yaml

from scripts import prepare_v6_candidate_registry as registry
from scripts import run_v6_candidate_generation as generation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v6_4_reference_tail_recovery.yaml"


def test_v6_4_is_explicitly_versioned_and_keeps_selector_contract():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"] == registry.V6_4_REFERENCE_TAIL_RECOVERY_PROTOCOL
    assert config["design_version"] == "6.4"
    assert config["supersedes"]["results_must_not_be_merged"] is True
    assert config["frozen_delta"]["only_scientific_change"] == (
        "forced_correction_plus_deterministic_reference_tail"
    )
    assert config["selectors"]["registry"]["full_proposed"]["weights"] == {
        "causal": 0.50,
        "hardness": 0.25,
        "coverage": 0.25,
    }


def test_reference_tail_mode_is_bound_into_semantic_contract():
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
        recovery_continuation_mode="deterministic_reference_tail",
    )
    contract = generation.semantic_generation_contract(
        args,
        continuation_seeds=(20260806, 20260807, 20260808),
        design_protocol=registry.V6_4_REFERENCE_TAIL_RECOVERY_PROTOCOL,
    )
    assert contract["recovery_continuation_mode"] == (
        "deterministic_reference_tail"
    )
    assert contract["clean_agent_mode"] == "deterministic_reference_replay"
