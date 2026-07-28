from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_v5_5_manifest import prepare


def test_real_pinned_tau2_structural_preflight(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    tau2_root = root / "data" / "raw" / "tau2-bench"
    if not tau2_root.is_dir():
        pytest.skip("pinned tau2 checkout is an external preflight dependency")
    output = tmp_path / "manifest.json"
    payload = prepare(
        tau2_root,
        root / "artifacts" / "v5_stage0" / "manifests" / "split_manifest.json",
        output,
        execute_tools=False,
    )
    assert payload["eligible_tasks"] == 45
    assert payload["structurally_eligible_tasks"] == 45
    assert payload["eligible_domain_counts"] == {"airline": 12, "retail": 33}
    assert payload["registered_tasks"] == 24
    assert payload["registered_pairs"] == 48
    assert payload["tool_execution_preflight_performed"] is False
    assert payload["official_test_used"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == payload
