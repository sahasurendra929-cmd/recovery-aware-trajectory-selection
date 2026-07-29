from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v6_1_72b_teacher.yaml"


def test_v6_1_changes_only_frozen_teacher_identity():
    original = yaml.safe_load(
        (ROOT / "configs" / "v6_causal_recovery_selection.yaml").read_text()
    )
    revised = yaml.safe_load(CONFIG.read_text())

    assert revised["design_version"] == "6.1"
    assert revised["supersedes"]["results_must_not_be_merged"] is True
    assert revised["models"]["trajectory_teacher"] == {
        "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "revision": "698703eae6604af048a3d2f509995dc302088217",
        "serving": "vllm_awq_tensor_parallel_2",
    }

    revised_for_comparison = dict(revised)
    for key in ("experiment_name", "protocol", "design_version", "supersedes", "frozen_delta"):
        revised_for_comparison.pop(key)
    original_for_comparison = dict(original)
    for key in ("experiment_name", "protocol", "design_version"):
        original_for_comparison.pop(key)

    revised_for_comparison["models"] = dict(revised_for_comparison["models"])
    original_for_comparison["models"] = dict(original_for_comparison["models"])
    revised_for_comparison["models"].pop("trajectory_teacher")
    original_for_comparison["models"].pop("trajectory_teacher")

    assert revised_for_comparison == original_for_comparison

