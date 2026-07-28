from __future__ import annotations

import pytest

from scripts import v5_5_full_protocol as protocol


def test_full_design_is_internally_consistent():
    protocol.validate_design()
    summary = protocol.frozen_summary()
    assert summary["arms"] == {
        "r0_perfect": 0.0,
        "r25": 0.25,
        "r50": 0.5,
        "r75": 0.75,
        "r100_recovery": 1.0,
    }
    assert summary["data_gate"]["minimum_natural_pairs"] == 48
    assert summary["data_gate"]["paper_target_natural_pairs"] == 60
    assert summary["evaluation"]["sealed_test_tasks"] == 60
    assert summary["official_test_used_for_selection"] is False


def test_primary_estimand_is_difference_from_perfect_only():
    success = {"r0_perfect": 0.20, "r50": 0.35}
    assert protocol.primary_estimand(success, "r50") == pytest.approx(0.15)
    with pytest.raises(protocol.FullProtocolError):
        protocol.primary_estimand(success, "r0_perfect")


def test_clean_noninferiority_uses_five_point_margin():
    assert protocol.clean_noninferiority(
        {"r0_perfect": 0.40, "r50": 0.35}, "r50"
    )
    assert not protocol.clean_noninferiority(
        {"r0_perfect": 0.40, "r50": 0.349}, "r50"
    )


def test_validation_utility_penalizes_only_excess_clean_damage():
    within_margin = protocol.validation_utility(
        clean_success=0.36,
        error_success=0.30,
        baseline_clean_success=0.40,
    )
    beyond_margin = protocol.validation_utility(
        clean_success=0.30,
        error_success=0.30,
        baseline_clean_success=0.40,
    )
    assert within_margin == pytest.approx(0.30)
    assert beyond_margin == pytest.approx(0.25)
