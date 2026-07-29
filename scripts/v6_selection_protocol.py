#!/usr/bin/env python3
"""Frozen contract for V6 causal recovery trajectory selection.

V6 treats data generation as a fixed candidate-pool construction step and
data selection as the scientific intervention.  A ``candidate_pair`` is the
atomic selection unit and contains two counterfactual sibling error branches
from one shared task prefix and environment snapshot.  A ``choice_set`` is
only a registry grouping at least three candidate pairs available for the
same task/prefix.  In the primary matched-task experiment every selector must
choose exactly one candidate pair per task.

The causal score is environment grounded.  For two sibling errors ``i`` and
``j``, only the first recovery action is exchanged.  All four cells then use
the same frozen continuation policy, decoding contract, seed set and rollout
budget:

    kappa(i, j) = 0.5 * [(R(i, a_i) + R(j, a_j))
                         - (R(i, a_j) + R(j, a_i))]

This avoids the invalid operation of replaying an entire error-specific future
under a different error.  Official-test content is sealed throughout pool
construction, scoring, selector development and model selection.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL = "v6_causal_recovery_selection_v1"
DESIGN_VERSION = "6.0"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

# Reuse the outcome-independent V5.3 partition exactly.  The five structural
# exclusions are removed before the remaining 78 inner-train tasks are split.
PARTITION_SEED = 20260722
PARTITION_SALT = "v5.3-loss-validation"
PARTITION_SHA256 = (
    "c90255a2cf4100d8cf5b5fcf3e684144ed1a393fad1c214058146721c4ed6818"
)
SOURCE_INNER_TRAIN_TASKS = 83
ARM_TRAIN_TASKS = 70
LOSS_VALIDATION_TASKS = 8
EXCLUDED_TASK_IDS = (
    "airline:0",
    "airline:10",
    "airline:28",
    "airline:34",
    "retail:24",
)
LOSS_VALIDATION_IDS = (
    "airline:15",
    "airline:47",
    "retail:22",
    "retail:23",
    "retail:52",
    "retail:63",
    "retail:96",
    "retail:103",
)

# The pilot registry is selected before V6 generation outcomes are observed.
PILOT_TASK_IDS = (
    "airline:21",
    "airline:40",
    "airline:33",
    "airline:12",
    "airline:4",
    "airline:14",
    "retail:16",
    "retail:98",
    "retail:47",
    "retail:31",
    "retail:92",
    "retail:99",
    "retail:11",
    "retail:104",
    "retail:107",
    "retail:37",
    "retail:21",
    "retail:19",
    "retail:35",
    "retail:30",
    "retail:1",
    "retail:87",
    "retail:72",
    "retail:7",
)
PILOT_TASKS = 24
PILOT_RETAIL_TASKS = 18
PILOT_AIRLINE_TASKS = 6
PILOT_RANK_SEED = 20260806
PILOT_REGISTRY_SHA256 = (
    "5170b139c6307a3f46e29ab9db2573b864181c7b8ff85e27452e14f591bf27c4"
)
PILOT_MIN_TASKS_WITH_THREE_PAIRS = 12
PILOT_MIN_ACCEPTED_PAIRS = 48

FORMAL_MIN_TASKS_WITH_THREE_PAIRS = 48
FORMAL_MIN_ACCEPTED_PAIRS = 144
SCREEN_MIN_TASKS_WITH_THREE_PAIRS = 40
MIN_PAIRS_PER_TASK = 3
MIN_DOMAINS = 2
MIN_ERROR_FAMILIES = 3

# Identifiability is a property of the candidate pool, not a result to tune.
KAPPA_LOW_MAX = 0.25
KAPPA_HIGH_MIN = 0.75
MIN_KAPPA_BUCKET_FRACTION = 0.20
MAX_KAPPA_BUCKET_FRACTION = 0.80
MIN_TEACHER_ACTION_SWITCH_ACCURACY = 0.80
MIN_TEACHER_OVER_ERROR_BLIND = 0.20

RANDOM_STRATIFIED_SEEDS = (20260806, 20260807, 20260808)
SELECTION_SEED = RANDOM_STRATIFIED_SEEDS[0]
DIRECTIONAL_SCREEN_ARMS = (
    "flawless_only",
    "random_stratified",
    "full_proposed",
)
DIRECTIONAL_SCREEN_TRAINING_SEEDS = (20260722,)
DIRECTIONAL_SCREEN_EVALUATION_SEEDS = (20260722, 20260723, 20260724)
SELECTION_UNIT = "candidate_pair"
GROUPING_UNIT = "choice_set"
ACCOUNTING_UNIT = "sibling_branch"
BUDGET_UNIT = "supervised_target_tokens"
MAX_ARM_TOKEN_SPREAD_FRACTION = 0.005
PRIMARY_MATCHED_PAIRS_PER_TASK = 1
SECONDARY_GLOBAL_MAX_PAIRS_PER_TASK = 2
HARDNESS_WINSOR_LIMITS = (0.05, 0.95)
LOW_KAPPA_THRESHOLD = KAPPA_LOW_MAX

SELECTORS = (
    "random",
    "shortest",
    "coverage_only",
    "hardness_only",
    "causal_only",
    "causal_hardness",
    "full_proposed",
)
PRIMARY_SELECTOR = "full_proposed"
PRIMARY_BASELINE = "random"
FULL_OBJECTIVE_WEIGHTS = {
    "causal": 0.50,
    "hardness": 0.25,
    "coverage": 0.25,
}
COVERAGE_WEIGHTS = {
    "domain": 1.0,
    "failed_tool": 1.0,
    "error_family": 1.0,
    "corrective_action": 1.0,
    "recovery_length_bin": 0.5,
    "recovery_mode": 0.5,
}
COVERAGE_FIELDS = tuple(COVERAGE_WEIGHTS)

FORCED_FIRST_CELLS = (
    "q_e1_a1",
    "q_e1_a2",
    "q_e2_a1",
    "q_e2_a2",
)
COMMON_CONTINUATION_FIELDS = (
    "continuation_policy_sha256",
    "continuation_seed_set_sha256",
    "decoding_sha256",
    "rollout_budget",
)

PAIR_TRUE_QUALITY_FIELDS = (
    "real_error_executed",
    "matched_recovery_replay_success",
    "cross_replay_complete",
    "no_future_leakage",
    "independent_replay_audited",
)
PAIR_FALSE_QUALITY_FIELDS = (
    "official_test_used",
)
PAIR_ZERO_QUALITY_FIELDS = ("failed_positive_labels",)
BRANCH_TRUE_AUDIT_FIELDS = (
    "identifier_field_allowlisted",
    "identifier_mutation_type_preserving",
    "actual_tool_error",
    "state_unchanged_after_error",
    "corrective_first_action_is_reference_call",
    "matched_task_success",
)

# Structural eligibility is checked without rollout outcomes.  A reference
# READ or WRITE call may be used only when it contains one of these strictly
# allowlisted identifier keys.  The canonical corrective family is
# ``tool_name::argument_key``; no synthetic family may be invented.
STRUCTURAL_PROTOCOL = "v6_safe_on_error_structural_eligibility_v1"
STRUCTURAL_ELIGIBILITY_SHA256 = (
    "002b5a2c5d83a4dab6dc8ce398d81bb8541bf7bfc6b25c04e62ce9ed179587f7"
)
LEGACY_STRUCTURAL_EXPECTED_SHA256 = (
    "93cc882757b677c049e725a96540244d9988e79a326eb42a659a342729904dc3"
)
STRUCTURAL_ELIGIBLE_COUNTS = {"retail": 48, "airline": 12}
ACTION_IDENTIFIABILITY_PROTOCOL = (
    "v6_distinct_reference_action_eligibility_v1"
)
ACTION_IDENTIFIABLE_SHA256 = (
    "fd47afd115fbe341ec2432533231545c24e0cf583b745e17f4862979263922aa"
)
ACTION_IDENTIFIABLE_COUNTS = {"retail": 39, "airline": 11}
ACTION_IDENTIFIABILITY_EXCLUSIONS = (
    "airline:11",
    "retail:75",
    "retail:82",
    "retail:83",
    "retail:84",
    "retail:85",
    "retail:89",
    "retail:93",
    "retail:105",
    "retail:106",
)
ACTION_IDENTIFIABILITY_EXCLUSION_REASON = (
    "FEWER_THAN_TWO_DISTINCT_REFERENCE_ACTIONS"
)
SAFE_ON_ERROR_IDENTIFIER_KEYS: dict[str, tuple[str, ...]] = {
    "retail": (
        "email",
        "item_id",
        "order_id",
        "payment_method_id",
        "product_id",
        "user_id",
    ),
    "airline": (
        "flight_number",
        "payment_id",
        "reservation_id",
        "user_id",
    ),
}
STRUCTURAL_REASON_CODES = (
    "ELIGIBLE",
    "TASK_NOT_IN_ARM_TRAIN",
    "NO_COMMON_PREFIX",
    "NO_COMMON_ENVIRONMENT_SNAPSHOT",
    "FEWER_THAN_TWO_SAFE_ON_ERROR_ACTION_FAMILIES",
    "DUPLICATE_ACTION_FAMILY",
    "IDENTIFIER_FIELD_NOT_ALLOWLISTED",
    "IDENTIFIER_MUTATION_NOT_TYPE_PRESERVING",
    "IDENTIFIER_MUTATION_DID_NOT_CHANGE_VALUE",
    "FEWER_THAN_TWO_DISTINCT_REFERENCE_ACTIONS",
)
RUNTIME_REASON_CODES = (
    "REAL_ERROR_NOT_OBSERVED",
    "ERROR_CHANGED_ENVIRONMENT_STATE",
    "CORRECTIVE_FIRST_ACTION_NOT_MATCHED_REFERENCE",
    "MATCHED_RECOVERY_FAILED",
    "CROSS_REPLAY_INCOMPLETE",
    "FUTURE_LEAKAGE",
    "FAILED_POSITIVE_LABEL",
    "INDEPENDENT_REPLAY_MISSING",
    "OFFICIAL_TEST_ACCESSED",
)


class V6ProtocolError(RuntimeError):
    """The frozen V6 design or its evidence was violated."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def canonical_reference_action_key(
    tool_name: str, arguments: Mapping[str, Any]
) -> str:
    """Hash the complete reference tool call, not merely ``tool::key``."""
    if not isinstance(tool_name, str) or not tool_name:
        raise V6ProtocolError("reference tool name is empty")
    if not isinstance(arguments, Mapping) or not arguments:
        raise V6ProtocolError("reference action arguments are empty")
    if any(not isinstance(key, str) or not key for key in arguments):
        raise V6ProtocolError("reference action argument keys must be strings")
    return sha256({"name": tool_name, "arguments": dict(arguments)})


def validate_constants() -> None:
    """Fail closed if a preregistered design constant drifts."""
    if SOURCE_INNER_TRAIN_TASKS != (
        ARM_TRAIN_TASKS + LOSS_VALIDATION_TASKS + len(EXCLUDED_TASK_IDS)
    ):
        raise V6ProtocolError("70/8/five-exclusion partition count drift")
    if len(EXCLUDED_TASK_IDS) != 5 or len(set(EXCLUDED_TASK_IDS)) != 5:
        raise V6ProtocolError("the structural exclusion set must contain five tasks")
    if len(LOSS_VALIDATION_IDS) != 8 or len(set(LOSS_VALIDATION_IDS)) != 8:
        raise V6ProtocolError("loss-validation set must contain eight tasks")
    if set(EXCLUDED_TASK_IDS) & set(LOSS_VALIDATION_IDS):
        raise V6ProtocolError("excluded and loss-validation tasks overlap")
    if len(PILOT_TASK_IDS) != PILOT_TASKS or len(set(PILOT_TASK_IDS)) != PILOT_TASKS:
        raise V6ProtocolError("pilot registry must contain 24 unique tasks")
    domain_counts = Counter(value.split(":", 1)[0] for value in PILOT_TASK_IDS)
    if domain_counts != {
        "retail": PILOT_RETAIL_TASKS,
        "airline": PILOT_AIRLINE_TASKS,
    }:
        raise V6ProtocolError("pilot registry must be 18 retail / 6 airline")
    if set(PILOT_TASK_IDS) & (
        set(EXCLUDED_TASK_IDS)
        | set(LOSS_VALIDATION_IDS)
        | set(ACTION_IDENTIFIABILITY_EXCLUSIONS)
    ):
        raise V6ProtocolError(
            "pilot contains partition- or action-identifiability-excluded tasks"
        )
    if sha256(list(PILOT_TASK_IDS)) != PILOT_REGISTRY_SHA256:
        raise V6ProtocolError("pilot registry digest drift")
    for domain in ("airline", "retail"):
        observed = [
            task for task in PILOT_TASK_IDS if task.startswith(f"{domain}:")
        ]
        expected = sorted(
            observed,
            key=lambda task: hashlib.sha256(
                f"{PROTOCOL}|pilot|{PILOT_RANK_SEED}|{task}".encode("utf-8")
            ).hexdigest(),
        )
        if observed != expected:
            raise V6ProtocolError(f"{domain}: pilot domain-hash order drift")
    if PILOT_MIN_TASKS_WITH_THREE_PAIRS != 12:
        raise V6ProtocolError("pilot qualifying-task gate drift")
    if PILOT_MIN_ACCEPTED_PAIRS != 48:
        raise V6ProtocolError("pilot accepted-pair gate drift")
    if (
        FORMAL_MIN_TASKS_WITH_THREE_PAIRS != 48
        or FORMAL_MIN_ACCEPTED_PAIRS != 144
        or SCREEN_MIN_TASKS_WITH_THREE_PAIRS != 40
    ):
        raise V6ProtocolError("formal/screen population gate drift")
    if not (
        0 <= KAPPA_LOW_MAX < KAPPA_HIGH_MIN <= 1
        and MIN_KAPPA_BUCKET_FRACTION == 0.20
        and MAX_KAPPA_BUCKET_FRACTION == 0.80
    ):
        raise V6ProtocolError("kappa identifiability thresholds drift")
    if (
        MIN_TEACHER_ACTION_SWITCH_ACCURACY != 0.80
        or MIN_TEACHER_OVER_ERROR_BLIND != 0.20
    ):
        raise V6ProtocolError("teacher/oracle identifiability gate drift")
    if (
        SELECTION_UNIT != "candidate_pair"
        or GROUPING_UNIT != "choice_set"
        or ACCOUNTING_UNIT != "sibling_branch"
    ):
        raise V6ProtocolError("selection/accounting unit drift")
    if (
        len(RANDOM_STRATIFIED_SEEDS) != 3
        or len(set(RANDOM_STRATIFIED_SEEDS)) != 3
        or HARDNESS_WINSOR_LIMITS != (0.05, 0.95)
    ):
        raise V6ProtocolError("random/hardness normalization contract drift")
    if len(SELECTORS) != len(set(SELECTORS)) or PRIMARY_SELECTOR not in SELECTORS:
        raise V6ProtocolError("selector registry drift")
    if (
        len(DIRECTIONAL_SCREEN_ARMS) != 3
        or len(set(DIRECTIONAL_SCREEN_ARMS)) != 3
        or DIRECTIONAL_SCREEN_ARMS[1] != "random_stratified"
        or DIRECTIONAL_SCREEN_TRAINING_SEEDS != (20260722,)
        or DIRECTIONAL_SCREEN_EVALUATION_SEEDS
        != (20260722, 20260723, 20260724)
    ):
        raise V6ProtocolError("directional-screen arm/seed contract drift")
    if not math.isclose(sum(FULL_OBJECTIVE_WEIGHTS.values()), 1.0):
        raise V6ProtocolError("full selector weights must sum to one")
    if set(COVERAGE_FIELDS) != set(COVERAGE_WEIGHTS):
        raise V6ProtocolError("coverage feature registry drift")
    if STRUCTURAL_ELIGIBLE_COUNTS != {"retail": 48, "airline": 12}:
        raise V6ProtocolError("structural eligibility counts drift")
    if ACTION_IDENTIFIABLE_COUNTS != {"retail": 39, "airline": 11}:
        raise V6ProtocolError("action-identifiable counts drift")
    if (
        len(ACTION_IDENTIFIABILITY_EXCLUSIONS) != 10
        or len(set(ACTION_IDENTIFIABILITY_EXCLUSIONS)) != 10
        or ACTION_IDENTIFIABILITY_EXCLUSION_REASON
        != "FEWER_THAN_TWO_DISTINCT_REFERENCE_ACTIONS"
    ):
        raise V6ProtocolError("action-identifiability exclusion contract drift")
    if SAFE_ON_ERROR_IDENTIFIER_KEYS != {
        "retail": (
            "email",
            "item_id",
            "order_id",
            "payment_method_id",
            "product_id",
            "user_id",
        ),
        "airline": (
            "flight_number",
            "payment_id",
            "reservation_id",
            "user_id",
        ),
    }:
        raise V6ProtocolError("safe-on-error identifier allowlist drift")


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V6ProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise V6ProtocolError(f"{name} must be in [0,1]")
    return result


def forced_first_kappa(
    q_e1_a1: float,
    q_e1_a2: float,
    q_e2_a1: float,
    q_e2_a2: float,
) -> float:
    """Compute the registered two-error forced-first causal contrast."""
    e1_a1 = _probability(q_e1_a1, "q_e1_a1")
    e1_a2 = _probability(q_e1_a2, "q_e1_a2")
    e2_a1 = _probability(q_e2_a1, "q_e2_a1")
    e2_a2 = _probability(q_e2_a2, "q_e2_a2")
    return 0.5 * ((e1_a1 + e2_a2) - (e1_a2 + e2_a1))


def forced_first_common_continuation_kappa(
    cells: Mapping[str, Mapping[str, Any]],
) -> float:
    """Validate four replay cells and compute kappa.

    Only the first recovery action differs.  The continuation policy and every
    rollout-control field must be byte-identical across the four cells.
    """
    if set(cells) != set(FORCED_FIRST_CELLS):
        raise V6ProtocolError(
            f"forced-first cells must be exactly {FORCED_FIRST_CELLS}"
        )
    common: tuple[Any, ...] | None = None
    rewards: dict[str, float] = {}
    for name in FORCED_FIRST_CELLS:
        cell = cells[name]
        if cell.get("forced_first_only") is not True:
            raise V6ProtocolError(f"{name}: more than the first action was forced")
        if cell.get("gold_suffix_visible") is not False:
            raise V6ProtocolError(f"{name}: gold recovery suffix was visible")
        identity = tuple(cell.get(field) for field in COMMON_CONTINUATION_FIELDS)
        if any(value in (None, "") for value in identity):
            raise V6ProtocolError(f"{name}: incomplete continuation contract")
        if common is None:
            common = identity
        elif identity != common:
            raise V6ProtocolError("four cells do not share one continuation contract")
        rewards[name] = _probability(cell.get("task_success"), f"{name}.task_success")
    return forced_first_kappa(
        rewards["q_e1_a1"],
        rewards["q_e1_a2"],
        rewards["q_e2_a1"],
        rewards["q_e2_a2"],
    )


def hardness_margin(
    correct_action_key: str,
    mean_logprob_by_action: Mapping[str, float],
) -> float:
    """Base-model hardness: strongest wrong sibling minus matched action.

    Log probabilities must be length-normalized over the canonical first tool
    call.  Larger values mean the frozen base model prefers a sibling's wrong
    recovery action and the example has greater teaching value.
    """
    if not isinstance(correct_action_key, str) or not correct_action_key:
        raise V6ProtocolError("correct action key is empty")
    if correct_action_key not in mean_logprob_by_action:
        raise V6ProtocolError("correct action is absent from hardness choices")
    if len(mean_logprob_by_action) < 2:
        raise V6ProtocolError("hardness requires at least one sibling negative")
    values: dict[str, float] = {}
    for action, value in mean_logprob_by_action.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise V6ProtocolError(f"{action}: mean log-probability is not numeric")
        score = float(value)
        if not math.isfinite(score) or score > 0:
            raise V6ProtocolError(f"{action}: invalid mean log-probability")
        values[action] = score
    strongest_wrong = max(
        value for action, value in values.items() if action != correct_action_key
    )
    return strongest_wrong - values[correct_action_key]


def supervised_token_cost(candidate_pairs: Sequence[Mapping[str, Any]]) -> int:
    """Exact selector cost under the pinned student tokenizer."""
    total = 0
    for candidate_pair in candidate_pairs:
        value = candidate_pair.get("c_sup")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise V6ProtocolError("each candidate pair needs a positive integer c_sup")
        total += value
    return total


def nonpadding_token_cost(candidate_pairs: Sequence[Mapping[str, Any]]) -> int:
    """Audited compute accounting; selection itself is budgeted by ``c_sup``."""
    total = 0
    for candidate_pair in candidate_pairs:
        value = candidate_pair.get("c_nonpad")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise V6ProtocolError(
                "each candidate pair needs a positive integer c_nonpad"
            )
        if value < int(candidate_pair.get("c_sup", 0)):
            raise V6ProtocolError("c_nonpad cannot be smaller than c_sup")
        total += value
    return total


def _feature_values(candidate_pair: Mapping[str, Any], field: str) -> set[str]:
    values: set[str] = set()
    direct = candidate_pair.get(field)
    if isinstance(direct, str) and direct:
        values.add(direct)
    for branch in candidate_pair.get("branches", []):
        value = branch.get(field)
        if isinstance(value, str) and value:
            values.add(value)
    return values


def coverage_counts(
    candidate_pairs: Iterable[Mapping[str, Any]],
) -> Counter[tuple[str, str]]:
    """Count each feature value at most once per selected candidate pair."""
    counts: Counter[tuple[str, str]] = Counter()
    for candidate_pair in candidate_pairs:
        for field in COVERAGE_FIELDS:
            for value in _feature_values(candidate_pair, field):
                counts[(field, value)] += 1
    return counts


def coverage_value(candidate_pairs: Iterable[Mapping[str, Any]]) -> float:
    """Frozen diminishing-return coverage objective."""
    counts = coverage_counts(candidate_pairs)
    return sum(
        COVERAGE_WEIGHTS[field] * math.log1p(count)
        for (field, _), count in counts.items()
    )


def coverage_marginal_gain(
    selected: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> float:
    return coverage_value([*selected, candidate]) - coverage_value(selected)


def structural_eligibility(candidate_pair: Mapping[str, Any]) -> dict[str, Any]:
    """Outcome-free check for two safe-on-error reference-action families."""
    branches = candidate_pair.get("branches")
    well_formed = isinstance(branches, list) and len(branches) == 2
    if not well_formed:
        return {
            "eligible": False,
            "reason_codes": ["FEWER_THAN_TWO_SAFE_ON_ERROR_ACTION_FAMILIES"],
            "checks": {"exactly_two_sibling_branches": False},
        }
    task_identity = candidate_pair.get("task_identity")
    domain = candidate_pair.get("domain")
    allowed_keys = set(SAFE_ON_ERROR_IDENTIFIER_KEYS.get(str(domain), ()))
    family_values: list[str] = []
    reference_action_keys: list[str] = []
    checks: dict[str, bool] = {
        "exactly_two_sibling_branches": True,
        "task_is_arm_train": (
            candidate_pair.get("partition") == "arm_train"
            and task_identity not in EXCLUDED_TASK_IDS
            and task_identity not in LOSS_VALIDATION_IDS
        ),
        "task_not_action_identifiability_excluded": (
            task_identity not in ACTION_IDENTIFIABILITY_EXCLUSIONS
        ),
        "common_prefix_present": bool(candidate_pair.get("prefix_sha256")),
        "common_environment_snapshot_present": bool(
            candidate_pair.get("environment_snapshot_sha256")
        ),
    }
    branch_actions_allowed = True
    identifier_fields_allowed = True
    mutations_type_preserving = True
    mutations_changed = True
    full_reference_actions_valid = True
    for branch in branches:
        tool = branch.get("tool_name")
        identifier_key = branch.get("identifier_key")
        if (
            not isinstance(tool, str)
            or not tool
            or identifier_key not in allowed_keys
            or branch.get("reference_action_kind") not in {"READ", "WRITE"}
        ):
            branch_actions_allowed = False
            continue
        family = f"{tool}::{identifier_key}"
        family_values.append(family)
        arguments = branch.get("reference_arguments")
        try:
            expected_action_key = canonical_reference_action_key(tool, arguments)
        except V6ProtocolError:
            full_reference_actions_valid = False
        else:
            reference_action_keys.append(expected_action_key)
            if branch.get("reference_action_key") != expected_action_key:
                full_reference_actions_valid = False
        if (
            branch.get("corrective_family") != family
            or branch.get("identifier_field_allowlisted") is not True
        ):
            identifier_fields_allowed = False
        original = branch.get("original_identifier")
        mutated = branch.get("mutated_identifier")
        if (
            branch.get("identifier_mutation_type_preserving") is not True
            or type(original) is not type(mutated)
        ):
            mutations_type_preserving = False
        if original == mutated:
            mutations_changed = False
    checks.update(
        {
            "two_safe_on_error_reference_actions": (
                branch_actions_allowed and len(family_values) == 2
            ),
            "two_distinct_corrective_families": len(set(family_values)) == 2,
            "full_reference_actions_canonical": (
                full_reference_actions_valid
                and len(reference_action_keys) == 2
            ),
            "two_distinct_full_reference_actions": (
                len(set(reference_action_keys)) == 2
            ),
            "identifier_fields_allowlisted": identifier_fields_allowed,
            "identifier_mutations_type_preserving": mutations_type_preserving,
            "identifier_mutations_change_value": mutations_changed,
        }
    )
    reason_by_check = {
        "task_is_arm_train": "TASK_NOT_IN_ARM_TRAIN",
        "task_not_action_identifiability_excluded": (
            "FEWER_THAN_TWO_DISTINCT_REFERENCE_ACTIONS"
        ),
        "common_prefix_present": "NO_COMMON_PREFIX",
        "common_environment_snapshot_present": "NO_COMMON_ENVIRONMENT_SNAPSHOT",
        "two_safe_on_error_reference_actions": (
            "FEWER_THAN_TWO_SAFE_ON_ERROR_ACTION_FAMILIES"
        ),
        "full_reference_actions_canonical": (
            "FEWER_THAN_TWO_DISTINCT_REFERENCE_ACTIONS"
        ),
        "two_distinct_full_reference_actions": (
            "FEWER_THAN_TWO_DISTINCT_REFERENCE_ACTIONS"
        ),
        "identifier_fields_allowlisted": "IDENTIFIER_FIELD_NOT_ALLOWLISTED",
        "identifier_mutations_type_preserving": (
            "IDENTIFIER_MUTATION_NOT_TYPE_PRESERVING"
        ),
        "identifier_mutations_change_value": (
            "IDENTIFIER_MUTATION_DID_NOT_CHANGE_VALUE"
        ),
    }
    reasons = list(
        dict.fromkeys(
            reason_by_check[name]
            for name, passed in checks.items()
            if not passed and name in reason_by_check
        )
    )
    return {
        "eligible": not reasons,
        "reason_codes": ["ELIGIBLE"] if not reasons else reasons,
        "checks": checks,
    }


def audit_candidate_pair(candidate_pair: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute structural/runtime eligibility without producer acceptance."""
    branches = candidate_pair.get("branches")
    if not isinstance(branches, list) or len(branches) != 2:
        return {"well_formed": False}
    structural = structural_eligibility(candidate_pair)
    quality = candidate_pair.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    true_quality = all(
        quality.get(field) is True for field in PAIR_TRUE_QUALITY_FIELDS
    )
    false_quality = all(
        quality.get(field) is False for field in PAIR_FALSE_QUALITY_FIELDS
    )
    zero_quality = all(
        isinstance(quality.get(field), int)
        and not isinstance(quality.get(field), bool)
        and quality.get(field) == 0
        for field in PAIR_ZERO_QUALITY_FIELDS
    )
    branch_audits = all(
        branch.get(field) is True
        for branch in branches
        for field in BRANCH_TRUE_AUDIT_FIELDS
    )
    branch_ids = [branch.get("branch_id") for branch in branches]
    error_events = [branch.get("error_event_sha256") for branch in branches]
    actions = [branch.get("first_recovery_action_key") for branch in branches]
    references = [branch.get("reference_action_key") for branch in branches]
    failed_injection_state_unchanged = all(
        isinstance(branch.get(before), str)
        and bool(branch.get(before))
        and branch.get(before) == branch.get(after)
        for branch in branches
        for before, after in (
            ("agent_db_hash_before_error", "agent_db_hash_after_error"),
            ("user_db_hash_before_error", "user_db_hash_after_error"),
        )
    )
    try:
        cost_valid = (
            supervised_token_cost([candidate_pair]) > 0
            and nonpadding_token_cost([candidate_pair]) > 0
        )
    except V6ProtocolError:
        cost_valid = False

    checks = {
        "well_formed": True,
        "candidate_pair_identity_present": (
            isinstance(candidate_pair.get("candidate_pair_id"), str)
            and bool(candidate_pair.get("candidate_pair_id"))
        ),
        "choice_set_identity_present": (
            isinstance(candidate_pair.get("choice_set_id"), str)
            and bool(candidate_pair.get("choice_set_id"))
        ),
        "task_identity_present": (
            isinstance(candidate_pair.get("task_identity"), str)
            and ":" in str(candidate_pair.get("task_identity"))
        ),
        "structurally_eligible": structural["eligible"] is True,
        "unique_branch_ids": (
            all(isinstance(value, str) and value for value in branch_ids)
            and len(branch_ids) == len(set(branch_ids))
        ),
        "two_distinct_real_error_events": (
            all(isinstance(value, str) and value for value in error_events)
            and len(error_events) == len(set(error_events))
        ),
        "two_distinct_first_actions": (
            all(isinstance(value, str) and value for value in actions)
            and len(actions) == len(set(actions))
        ),
        "corrective_first_actions_match_references": (
            all(isinstance(value, str) and value for value in references)
            and actions == references
        ),
        "all_branch_runtime_gates_true": branch_audits,
        "failed_injection_agent_and_user_state_unchanged": (
            failed_injection_state_unchanged
        ),
        "all_pair_quality_gates_true": true_quality,
        "all_pair_forbidden_flags_false": false_quality,
        "failed_positive_label_count_zero": zero_quality,
        "positive_exact_token_cost": cost_valid,
    }
    cells = candidate_pair.get("forced_first_cells")
    try:
        computed_kappa = forced_first_common_continuation_kappa(cells)
        declared = candidate_pair.get("kappa")
        kappa_valid = (
            isinstance(declared, (int, float))
            and not isinstance(declared, bool)
            and math.isclose(float(declared), computed_kappa, abs_tol=1e-12)
        )
    except (V6ProtocolError, TypeError):
        kappa_valid = False
    checks["forced_first_common_continuation_kappa_valid"] = kappa_valid
    return checks


def candidate_pair_is_accepted(candidate_pair: Mapping[str, Any]) -> bool:
    return all(audit_candidate_pair(candidate_pair).values())


def audit_choice_set(choice_set: Mapping[str, Any]) -> dict[str, bool]:
    """A choice set only groups candidate pairs; it is not selected itself."""
    candidate_pairs = choice_set.get("candidate_pairs")
    if not isinstance(candidate_pairs, list) or not candidate_pairs:
        return {"well_formed": False}
    pair_ids = [row.get("candidate_pair_id") for row in candidate_pairs]
    task = choice_set.get("task_identity")
    prefix = choice_set.get("prefix_sha256")
    snapshot = choice_set.get("environment_snapshot_sha256")
    checks = {
        "well_formed": True,
        "all_candidate_pairs_accepted": all(
            candidate_pair_is_accepted(row) for row in candidate_pairs
        ),
        "unique_candidate_pair_ids": (
            all(isinstance(value, str) and value for value in pair_ids)
            and len(pair_ids) == len(set(pair_ids))
        ),
        "same_task": {
            row.get("task_identity") for row in candidate_pairs
        } == {task},
        "same_prefix": {
            row.get("prefix_sha256") for row in candidate_pairs
        } == {prefix},
        "same_environment_snapshot": {
            row.get("environment_snapshot_sha256") for row in candidate_pairs
        } == {snapshot},
        "choice_set_id_matches": {
            row.get("choice_set_id") for row in candidate_pairs
        } == {choice_set.get("choice_set_id")},
    }
    return checks


def choice_set_is_accepted(choice_set: Mapping[str, Any]) -> bool:
    return all(audit_choice_set(choice_set).values())


def identifiability_gate(
    kappas: Sequence[float],
    *,
    teacher_action_switch_accuracy: float,
    error_blind_action_switch_accuracy: float,
) -> dict[str, Any]:
    """Check that the pool has both informative and uninformative candidates."""
    if not kappas:
        return {
            "status": "FAIL_CLOSED",
            "checks": {"nonempty_kappa_population": False},
        }
    checked: list[float] = []
    for value in kappas:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise V6ProtocolError("kappa values must be numeric")
        score = float(value)
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise V6ProtocolError("kappa values must be in [-1,1]")
        checked.append(score)
    n = len(checked)
    low = sum(value <= KAPPA_LOW_MAX for value in checked) / n
    high = sum(value >= KAPPA_HIGH_MIN for value in checked) / n
    middle = 1.0 - low - high
    teacher = _probability(
        teacher_action_switch_accuracy, "teacher_action_switch_accuracy"
    )
    blind = _probability(
        error_blind_action_switch_accuracy,
        "error_blind_action_switch_accuracy",
    )
    buckets = {"low": low, "middle": middle, "high": high}
    checks = {
        "nonempty_kappa_population": True,
        "at_least_20pct_kappa_high": high >= MIN_KAPPA_BUCKET_FRACTION,
        "at_least_20pct_kappa_low": low >= MIN_KAPPA_BUCKET_FRACTION,
        "no_single_kappa_bucket_above_80pct": (
            max(buckets.values()) <= MAX_KAPPA_BUCKET_FRACTION + 1e-12
        ),
        "teacher_action_switch_at_least_80pct": (
            teacher >= MIN_TEACHER_ACTION_SWITCH_ACCURACY
        ),
        "teacher_beats_error_blind_by_20pp": (
            teacher - blind >= MIN_TEACHER_OVER_ERROR_BLIND - 1e-12
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
        "checks": checks,
        "kappa_bucket_fraction": buckets,
        "teacher_action_switch_accuracy": teacher,
        "error_blind_action_switch_accuracy": blind,
        "teacher_minus_error_blind": teacher - blind,
    }


def _accepted_pool_summary(
    choice_sets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    accepted = [row for row in choice_sets if choice_set_is_accepted(row)]
    candidate_pair_ids: list[str] = []
    pairs_by_task: Counter[str] = Counter()
    domains: set[str] = set()
    error_families: set[str] = set()
    kappas: list[float] = []
    for choice_set in accepted:
        task = str(choice_set["task_identity"])
        domains.add(str(choice_set.get("domain", task.split(":", 1)[0])))
        for candidate_pair in choice_set["candidate_pairs"]:
            candidate_pair_ids.append(str(candidate_pair["candidate_pair_id"]))
            pairs_by_task[task] += 1
            kappas.append(float(candidate_pair["kappa"]))
            for branch in candidate_pair["branches"]:
                error_families.add(str(branch["error_family"]))
    qualifying_tasks = {
        task for task, count in pairs_by_task.items() if count >= MIN_PAIRS_PER_TASK
    }
    qualifying_pairs = sum(pairs_by_task[task] for task in qualifying_tasks)
    return {
        "accepted_choice_sets": len(accepted),
        "accepted_candidate_pairs": len(candidate_pair_ids),
        "unique_candidate_pair_ids": (
            len(candidate_pair_ids) == len(set(candidate_pair_ids))
        ),
        "pairs_by_task": dict(sorted(pairs_by_task.items())),
        "qualifying_task_ids": sorted(qualifying_tasks),
        "tasks_with_at_least_three_pairs": len(qualifying_tasks),
        "pairs_from_qualifying_tasks": qualifying_pairs,
        "domains": sorted(domains),
        "error_families": sorted(error_families),
        "kappas": kappas,
    }


def pilot_gate(
    choice_sets: Sequence[Mapping[str, Any]],
    *,
    registered_task_ids: Sequence[str],
    teacher_action_switch_accuracy: float,
    error_blind_action_switch_accuracy: float,
) -> dict[str, Any]:
    """Apply the fixed 24-task GO/NO-GO gate."""
    validate_constants()
    summary = _accepted_pool_summary(choice_sets)
    identification = identifiability_gate(
        summary["kappas"],
        teacher_action_switch_accuracy=teacher_action_switch_accuracy,
        error_blind_action_switch_accuracy=error_blind_action_switch_accuracy,
    )
    checks = {
        "registered_population_is_frozen_24": (
            tuple(registered_task_ids) == PILOT_TASK_IDS
        ),
        "at_least_12_tasks_with_three_pairs": (
            summary["tasks_with_at_least_three_pairs"]
            >= PILOT_MIN_TASKS_WITH_THREE_PAIRS
        ),
        "at_least_48_accepted_pairs": (
            summary["accepted_candidate_pairs"] >= PILOT_MIN_ACCEPTED_PAIRS
        ),
        "both_domains": len(summary["domains"]) >= MIN_DOMAINS,
        "at_least_three_error_families": (
            len(summary["error_families"]) >= MIN_ERROR_FAMILIES
        ),
        "unique_candidate_pair_ids": summary["unique_candidate_pair_ids"],
        "identifiability": identification["status"] == "PASS",
    }
    return {
        "protocol": PROTOCOL,
        "phase": "pilot",
        "status": "GO_FORMAL_POOL" if all(checks.values()) else "STOP_NO_GO",
        "checks": checks,
        "pool": summary,
        "identifiability": identification,
        "official_test_used": False,
    }


def formal_pool_gate(
    choice_sets: Sequence[Mapping[str, Any]],
    *,
    teacher_action_switch_accuracy: float,
    error_blind_action_switch_accuracy: float,
) -> dict[str, Any]:
    """Authorize formal selection, screen-only use, or a full stop."""
    validate_constants()
    summary = _accepted_pool_summary(choice_sets)
    identification = identifiability_gate(
        summary["kappas"],
        teacher_action_switch_accuracy=teacher_action_switch_accuracy,
        error_blind_action_switch_accuracy=error_blind_action_switch_accuracy,
    )
    common_checks = {
        "both_domains": len(summary["domains"]) >= MIN_DOMAINS,
        "at_least_three_error_families": (
            len(summary["error_families"]) >= MIN_ERROR_FAMILIES
        ),
        "unique_candidate_pair_ids": summary["unique_candidate_pair_ids"],
        "identifiability": identification["status"] == "PASS",
    }
    tasks = summary["tasks_with_at_least_three_pairs"]
    pairs = summary["pairs_from_qualifying_tasks"]
    formal_counts = (
        tasks >= FORMAL_MIN_TASKS_WITH_THREE_PAIRS
        and pairs >= FORMAL_MIN_ACCEPTED_PAIRS
    )
    screen_counts = SCREEN_MIN_TASKS_WITH_THREE_PAIRS <= tasks < (
        FORMAL_MIN_TASKS_WITH_THREE_PAIRS
    )
    if all(common_checks.values()) and formal_counts:
        status = "FORMAL_SELECTION_AUTHORIZED"
    elif all(common_checks.values()) and screen_counts:
        status = "SCREEN_ONLY"
    else:
        status = "STOP_INSUFFICIENT_POOL"
    return {
        "protocol": PROTOCOL,
        "phase": "formal_pool",
        "status": status,
        "checks": {
            **common_checks,
            "at_least_48_tasks_with_three_pairs": (
                tasks >= FORMAL_MIN_TASKS_WITH_THREE_PAIRS
            ),
            "at_least_144_pairs_from_qualifying_tasks": (
                pairs >= FORMAL_MIN_ACCEPTED_PAIRS
            ),
            "screen_band_40_to_47_tasks": screen_counts,
        },
        "pool": summary,
        "identifiability": identification,
        "official_test_used": False,
    }


def deterministic_random_key(
    candidate_pair_id: str, *, seed: int = SELECTION_SEED
) -> str:
    if not isinstance(candidate_pair_id, str) or not candidate_pair_id:
        raise V6ProtocolError("candidate_pair_id is empty")
    if seed not in RANDOM_STRATIFIED_SEEDS:
        raise V6ProtocolError("random selector seed is outside the frozen registry")
    return hashlib.sha256(
        f"{PROTOCOL}|random|{seed}|{candidate_pair_id}".encode("utf-8")
    ).hexdigest()


def selection_should_stop(
    *,
    selected_cost: int,
    token_budget: int,
    remaining_candidate_pair_costs: Sequence[int],
) -> bool:
    """Stop only when no complete candidate pair fits the remaining budget."""
    if (
        isinstance(selected_cost, bool)
        or isinstance(token_budget, bool)
        or not isinstance(selected_cost, int)
        or not isinstance(token_budget, int)
        or selected_cost < 0
        or token_budget <= 0
        or selected_cost > token_budget
    ):
        raise V6ProtocolError("invalid selector budget state")
    for cost in remaining_candidate_pair_costs:
        if isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0:
            raise V6ProtocolError("candidate-pair costs must be positive integers")
    remaining = token_budget - selected_cost
    return not any(cost <= remaining for cost in remaining_candidate_pair_costs)


def audit_arm_token_budgets(
    costs_by_arm: Mapping[str, int],
    *,
    token_budget: int,
) -> dict[str, Any]:
    """Require equal realized recovery-token exposure within 0.5%."""
    if not costs_by_arm:
        raise V6ProtocolError("no arm budgets supplied")
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget <= 0:
        raise V6ProtocolError("token_budget must be a positive integer")
    costs: dict[str, int] = {}
    for arm, value in costs_by_arm.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise V6ProtocolError(f"{arm}: invalid realized token cost")
        costs[arm] = value
    maximum = max(costs.values())
    minimum = min(costs.values())
    spread = (maximum - minimum) / token_budget
    checks = {
        "no_arm_exceeds_budget": maximum <= token_budget,
        "cross_arm_spread_at_most_0_5pct": (
            spread <= MAX_ARM_TOKEN_SPREAD_FRACTION + 1e-12
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
        "checks": checks,
        "budget_unit": BUDGET_UNIT,
        "token_budget": token_budget,
        "realized_costs": costs,
        "spread_fraction_of_budget": spread,
    }


def audit_matched_task_selection(
    selected_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_task_ids: Sequence[str],
) -> dict[str, Any]:
    """Primary experiment: every selector chooses one pair for every task."""
    expected = set(expected_task_ids)
    if not expected or len(expected) != len(tuple(expected_task_ids)):
        raise V6ProtocolError("expected matched-task registry is empty or duplicated")
    checks: dict[str, bool] = {}
    manifests: dict[str, Any] = {}
    for arm, rows in selected_by_arm.items():
        ids = [row.get("candidate_pair_id") for row in rows]
        tasks = [row.get("task_identity") for row in rows]
        arm_checks = {
            "registered_selector": arm in SELECTORS,
            "one_candidate_pair_per_task": (
                len(rows) == len(expected)
                and len(tasks) == len(set(tasks))
                and set(tasks) == expected
            ),
            "unique_candidate_pair_ids": (
                all(isinstance(value, str) and value for value in ids)
                and len(ids) == len(set(ids))
            ),
            "all_pairs_protocol_accepted": all(
                candidate_pair_is_accepted(row) for row in rows
            ),
        }
        checks[f"{arm}:valid"] = all(arm_checks.values())
        manifests[arm] = {
            "checks": arm_checks,
            "candidate_pair_ids": ids,
            "task_ids": tasks,
        }
    return {
        "status": "PASS" if checks and all(checks.values()) else "FAIL_CLOSED",
        "checks": checks,
        "arms": manifests,
        "pairs_per_task_per_arm": PRIMARY_MATCHED_PAIRS_PER_TASK,
        "official_test_used": False,
    }


def frozen_summary() -> dict[str, Any]:
    validate_constants()
    return {
        "protocol": PROTOCOL,
        "design_version": DESIGN_VERSION,
        "tau2_commit": TAU2_COMMIT,
        "partition": {
            "source_inner_train_tasks": SOURCE_INNER_TRAIN_TASKS,
            "excluded_task_ids": list(EXCLUDED_TASK_IDS),
            "arm_train_tasks": ARM_TRAIN_TASKS,
            "loss_validation_task_ids": list(LOSS_VALIDATION_IDS),
            "partition_sha256": PARTITION_SHA256,
        },
        "structural_eligibility": {
            "protocol": STRUCTURAL_PROTOCOL,
            "canonical_sha256": STRUCTURAL_ELIGIBILITY_SHA256,
            "legacy_expected_sha256": LEGACY_STRUCTURAL_EXPECTED_SHA256,
            "legacy_hash_match": False,
            "legacy_mismatch_acknowledged": True,
            "eligible_counts": STRUCTURAL_ELIGIBLE_COUNTS,
            "safe_on_error_identifier_keys": {
                domain: list(keys)
                for domain, keys in SAFE_ON_ERROR_IDENTIFIER_KEYS.items()
            },
            "minimum_distinct_corrective_families_per_task": 2,
            "corrective_family": "tool_name::argument_key",
            "wide_family_screen_only": True,
            "action_identifiability": {
                "protocol": ACTION_IDENTIFIABILITY_PROTOCOL,
                "canonical_sha256": ACTION_IDENTIFIABLE_SHA256,
                "eligible_counts": ACTION_IDENTIFIABLE_COUNTS,
                "excluded_task_ids": list(
                    ACTION_IDENTIFIABILITY_EXCLUSIONS
                ),
                "exclusion_reason": (
                    ACTION_IDENTIFIABILITY_EXCLUSION_REASON
                ),
                "gate": "at_least_two_distinct_canonical_full_reference_calls",
            },
            "failed_injection_agent_and_user_db_unchanged": True,
            "selection_uses_rollout_outcomes": False,
        },
        "selection": {
            "selection_unit": SELECTION_UNIT,
            "grouping_unit": GROUPING_UNIT,
            "accounting_unit": ACCOUNTING_UNIT,
            "selectors": list(SELECTORS),
            "random_stratified_seeds": list(RANDOM_STRATIFIED_SEEDS),
            "directional_screen_arms": list(DIRECTIONAL_SCREEN_ARMS),
            "directional_screen_training_seeds": list(
                DIRECTIONAL_SCREEN_TRAINING_SEEDS
            ),
            "directional_screen_evaluation_seeds": list(
                DIRECTIONAL_SCREEN_EVALUATION_SEEDS
            ),
            "hardness_winsor_limits": list(HARDNESS_WINSOR_LIMITS),
            "primary_selector": PRIMARY_SELECTOR,
            "primary_baseline": PRIMARY_BASELINE,
            "primary_matched_candidate_pairs_per_task": (
                PRIMARY_MATCHED_PAIRS_PER_TASK
            ),
            "budget_unit": BUDGET_UNIT,
            "maximum_arm_token_spread_fraction": (
                MAX_ARM_TOKEN_SPREAD_FRACTION
            ),
        },
        "pilot": {
            "tasks": PILOT_TASKS,
            "retail": PILOT_RETAIL_TASKS,
            "airline": PILOT_AIRLINE_TASKS,
            "task_ids": list(PILOT_TASK_IDS),
            "registry_sha256": PILOT_REGISTRY_SHA256,
            "minimum_tasks_with_three_pairs": (
                PILOT_MIN_TASKS_WITH_THREE_PAIRS
            ),
            "minimum_accepted_pairs": PILOT_MIN_ACCEPTED_PAIRS,
        },
        "formal": {
            "minimum_tasks_with_three_pairs": (
                FORMAL_MIN_TASKS_WITH_THREE_PAIRS
            ),
            "minimum_accepted_pairs": FORMAL_MIN_ACCEPTED_PAIRS,
            "screen_only_task_range": [40, 47],
        },
        "causal_score": {
            "intervention": "forced_first_action_only",
            "continuation": "same_frozen_policy_seed_set_decoding_and_budget",
            "formula": "0.5*((R_ii+R_jj)-(R_ij+R_ji))",
        },
        "official_test_used": False,
        "official_test_sealed": True,
        "test_unseal_rule": (
            "once_after_protocol_pool_scores_manifests_training_and_model_selection_are_frozen"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(frozen_summary(), indent=2, sort_keys=True))
