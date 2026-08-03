"""Pure-Python reference implementation of the CDCV decision core.

This module deliberately separates model scoring from verification. A caller
must supply candidate-score distributions for the base context and reviewed
interventions. The module then computes behavioral features, calibrates a
low-parameter reliability score, and selects COMMIT, CLARIFY, or
ABSTAIN_ESCALATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, log
from typing import Iterable, Mapping, Protocol, Sequence


ProbabilityVector = Mapping[str, float]


class Action(str, Enum):
    COMMIT = "COMMIT"
    CLARIFY = "CLARIFY"
    ABSTAIN_ESCALATE = "ABSTAIN_ESCALATE"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize(scores: ProbabilityVector) -> dict[str, float]:
    if len(scores) < 2:
        raise ValueError("at least two candidate scores are required")
    if any(float(value) < 0 for value in scores.values()):
        raise ValueError("candidate scores must be non-negative")
    total = sum(float(value) for value in scores.values())
    if total <= 0:
        raise ValueError("candidate scores must have positive mass")
    return {str(key): float(value) / total for key, value in scores.items()}


def normalized_entropy(scores: ProbabilityVector) -> float:
    probabilities = normalize(scores)
    entropy = -sum(p * log(p) for p in probabilities.values() if p > 0)
    return entropy / log(len(probabilities))


def jensen_shannon_divergence(
    first: ProbabilityVector, second: ProbabilityVector
) -> float:
    p = normalize(first)
    q = normalize(second)
    if set(p) != set(q):
        raise ValueError("all probe distributions must use the same candidates")
    midpoint = {key: 0.5 * (p[key] + q[key]) for key in p}

    def kl(left: Mapping[str, float], right: Mapping[str, float]) -> float:
        return sum(
            value * (log(value / right[key], 2))
            for key, value in left.items()
            if value > 0
        )

    return _clip01(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint))


def _top_sense(scores: ProbabilityVector) -> str:
    probabilities = normalize(scores)
    return max(probabilities, key=probabilities.get)


@dataclass(frozen=True)
class PreservingProbe:
    scores: ProbabilityVector
    validity_weight: float = 1.0


@dataclass(frozen=True)
class ChangingProbe:
    scores: ProbabilityVector
    target_sense_id: str
    validity_weight: float = 1.0


@dataclass(frozen=True)
class VerificationFeatures:
    base_sense_id: str
    top_two_margin: float
    normalized_entropy: float
    preservation_invariance: float
    preservation_probability: float
    mean_preserving_jsd: float
    targeted_response: float
    mean_changing_target_margin: float
    intervention_validity: float
    context_completeness: float
    context_conflict: bool


def _validated_weight(value: float) -> float:
    weight = float(value)
    if not 0 <= weight <= 1:
        raise ValueError("validity weights must be within [0, 1]")
    return weight


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if not values or total <= 0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total


def compute_verification_features(
    base_scores: ProbabilityVector,
    preserving: Sequence[PreservingProbe],
    changing: Sequence[ChangingProbe],
    *,
    context_completeness: float,
    context_conflict: bool,
) -> VerificationFeatures:
    """Compute prespecified features from base and intervention scores.

    Validity weights are fixed human-review weights. They must not be changed
    after inspecting model outcomes. The caller must select a symmetric probe
    branch using ``base_sense``; no sealed reference sense or action is an
    input to this function.
    """

    base = normalize(base_scores)
    base_sense = _top_sense(base)
    ordered = sorted(base.values(), reverse=True)
    margin = ordered[0] - ordered[1]

    preserving_weights = [_validated_weight(probe.validity_weight) for probe in preserving]
    preserving_distributions = [normalize(probe.scores) for probe in preserving]
    for distribution in preserving_distributions:
        if set(distribution) != set(base):
            raise ValueError("preserving probe candidate set differs from base")
    invariance = _weighted_mean(
        [float(_top_sense(distribution) == base_sense) for distribution in preserving_distributions],
        preserving_weights,
    )
    retention = _weighted_mean(
        [distribution[base_sense] for distribution in preserving_distributions],
        preserving_weights,
    )
    preserving_jsd = _weighted_mean(
        [jensen_shannon_divergence(base, distribution) for distribution in preserving_distributions],
        preserving_weights,
    )

    changing_weights = [_validated_weight(probe.validity_weight) for probe in changing]
    changing_distributions = [normalize(probe.scores) for probe in changing]
    for probe, distribution in zip(changing, changing_distributions):
        if set(distribution) != set(base):
            raise ValueError("changing probe candidate set differs from base")
        if probe.target_sense_id not in base:
            raise ValueError("changing target is absent from the candidate set")
        if probe.target_sense_id == base_sense:
            raise ValueError("changing target must contrast with the base prediction")
    targeted_response = _weighted_mean(
        [
            float(_top_sense(distribution) == probe.target_sense_id)
            for probe, distribution in zip(changing, changing_distributions)
        ],
        changing_weights,
    )
    changing_margin = _weighted_mean(
        [
            distribution[probe.target_sense_id] - distribution[base_sense]
            for probe, distribution in zip(changing, changing_distributions)
        ],
        changing_weights,
    )

    all_weights = preserving_weights + changing_weights
    validity = sum(all_weights) / len(all_weights) if all_weights else 0.0
    return VerificationFeatures(
        base_sense_id=base_sense,
        top_two_margin=_clip01(margin),
        normalized_entropy=_clip01(normalized_entropy(base)),
        preservation_invariance=_clip01(invariance),
        preservation_probability=_clip01(retention),
        mean_preserving_jsd=_clip01(preserving_jsd),
        targeted_response=_clip01(targeted_response),
        mean_changing_target_margin=max(-1.0, min(1.0, changing_margin)),
        intervention_validity=_clip01(validity),
        context_completeness=_clip01(context_completeness),
        context_conflict=bool(context_conflict),
    )


class Calibrator(Protocol):
    def predict(self, raw_score: float) -> float: ...


@dataclass(frozen=True)
class IdentityCalibrator:
    def predict(self, raw_score: float) -> float:
        return _clip01(raw_score)


@dataclass(frozen=True)
class _IsoBlock:
    low: float
    high: float
    weight: float
    positive_rate: float


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Stepwise isotonic probability mapping fitted by pair-adjacent violators."""

    blocks: tuple[_IsoBlock, ...]

    @classmethod
    def fit(
        cls,
        raw_scores: Sequence[float],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "IsotonicCalibrator":
        if len(raw_scores) != len(labels) or not raw_scores:
            raise ValueError("non-empty score and label arrays must have equal length")
        if any(label not in (0, 1, False, True) for label in labels):
            raise ValueError("isotonic labels must be binary")

        if sample_weights is None:
            weights = [1.0] * len(raw_scores)
        else:
            if len(sample_weights) != len(raw_scores):
                raise ValueError("sample weights must match the score array length")
            weights = [float(weight) for weight in sample_weights]
            if any(not isfinite(weight) or weight <= 0 for weight in weights):
                raise ValueError("sample weights must be finite and strictly positive")

        ordered = sorted(
            (float(score), int(label), weight)
            for score, label, weight in zip(raw_scores, labels, weights)
        )
        if any(not isfinite(score) for score, _, _ in ordered):
            raise ValueError("isotonic scores must be finite")

        # Pool equal score values before pair-adjacent-violators. Without this
        # step, tied scores create overlapping singleton blocks and prediction
        # at the shared boundary depends on arbitrary label ordering.
        mutable: list[list[float]] = []
        for score, label, weight in ordered:
            if mutable and score == mutable[-1][1]:
                block = mutable[-1]
                combined_weight = block[2] + weight
                block[3] = (
                    block[3] * block[2] + float(label) * weight
                ) / combined_weight
                block[2] = combined_weight
            else:
                mutable.append([score, score, weight, float(label)])

        pooled: list[list[float]] = []
        for block in mutable:
            pooled.append(block)
            while len(pooled) >= 2 and pooled[-2][3] > pooled[-1][3]:
                right = pooled.pop()
                left = pooled.pop()
                weight = left[2] + right[2]
                rate = (left[3] * left[2] + right[3] * right[2]) / weight
                pooled.append([left[0], right[1], weight, rate])
        return cls(tuple(_IsoBlock(*block) for block in pooled))

    def predict(self, raw_score: float) -> float:
        score = float(raw_score)
        if not self.blocks:
            raise ValueError("calibrator has no blocks")
        for block in self.blocks:
            if score <= block.high:
                return _clip01(block.positive_rate)
        return _clip01(self.blocks[-1].positive_rate)


@dataclass(frozen=True)
class AnswerScenario:
    """Development/calibration prior scenario, never a sealed evaluator label."""

    probability: float
    resulting_scores: ProbabilityVector


@dataclass(frozen=True)
class ClarificationQuestion:
    question_id: str
    context_slot: str
    scenarios: tuple[AnswerScenario, ...]
    interaction_cost: float = 0.0
    privacy_cost: float = 0.0
    compute_cost: float = 0.0
    approved: bool = True
    sensitive: bool = False

    def expected_information_gain(self, base_scores: ProbabilityVector) -> float:
        if not self.scenarios:
            return float("-inf")
        base = normalize(base_scores)
        probabilities = [float(item.probability) for item in self.scenarios]
        if any(value < 0 for value in probabilities) or sum(probabilities) <= 0:
            raise ValueError("answer scenario probabilities must have positive mass")
        resulting = [normalize(item.resulting_scores) for item in self.scenarios]
        if any(set(scores) != set(base) for scores in resulting):
            raise ValueError("answer scenarios must use the base candidate set")
        total = sum(probabilities)
        expected_entropy = sum(
            probability / total * normalized_entropy(scores)
            for probability, scores in zip(probabilities, resulting)
        )
        return normalized_entropy(base) - expected_entropy


@dataclass(frozen=True)
class GateState:
    card_valid: bool = True
    interventions_valid: bool = True
    prohibited_attribute_requested: bool = False
    out_of_domain: bool = False
    candidate_set_complete: bool = True
    missing_discriminating_slots: tuple[str, ...] = ()
    questions_remaining: int = 1
    compute_budget_available: bool = True


def _default_weights() -> dict[str, float]:
    return {
        "base_confidence": 0.25,
        "preservation": 0.25,
        "targeted_change": 0.25,
        "context_quality": 0.25,
    }


@dataclass(frozen=True)
class ControllerConfig:
    commit_threshold: float = 0.80
    minimum_intervention_validity: float = 0.80
    minimum_preservation_invariance: float = 1.00
    minimum_targeted_response: float = 1.00
    minimum_question_utility: float = 0.0
    interaction_cost_weight: float = 0.10
    privacy_cost_weight: float = 1.00
    compute_cost_weight: float = 0.10
    reliability_weights: Mapping[str, float] = field(default_factory=_default_weights)
    other_unlisted_id: str = "OTHER_UNLISTED"

    def __post_init__(self) -> None:
        if not 0 <= self.commit_threshold <= 1:
            raise ValueError("commit threshold must be within [0, 1]")
        if not 0 <= self.minimum_intervention_validity <= 1:
            raise ValueError("minimum intervention validity must be within [0, 1]")
        if not 0 <= self.minimum_preservation_invariance <= 1:
            raise ValueError("minimum preservation invariance must be within [0, 1]")
        if not 0 <= self.minimum_targeted_response <= 1:
            raise ValueError("minimum targeted response must be within [0, 1]")
        if not self.reliability_weights or sum(self.reliability_weights.values()) <= 0:
            raise ValueError("reliability weights must have positive total mass")
        if any(
            not isfinite(float(value)) or float(value) < 0
            for value in self.reliability_weights.values()
        ):
            raise ValueError("reliability weights must be finite and non-negative")
        if any(
            not isfinite(float(value)) or float(value) < 0
            for value in (
                self.interaction_cost_weight,
                self.privacy_cost_weight,
                self.compute_cost_weight,
            )
        ):
            raise ValueError("question-cost weights must be finite and non-negative")


@dataclass(frozen=True)
class GateDecision:
    action: Action
    estimated_safe_commit_probability: float
    raw_reliability: float
    sense_id: str | None = None
    question_id: str | None = None
    context_slot: str | None = None
    question_utility: float | None = None
    reason_code: str = ""


@dataclass
class GatePolicy:
    config: ControllerConfig = field(default_factory=ControllerConfig)
    calibrator: Calibrator = field(default_factory=IdentityCalibrator)

    def commit_eligibility(
        self, features: VerificationFeatures, state: GateState
    ) -> tuple[bool, str]:
        """Return the label-blind hard eligibility mask used before ranking.

        Reliability thresholds are deliberately excluded. No matched-coverage
        analysis may override a false result from this method.
        """

        failures = (
            (not state.card_valid, "INVALID_CONTEXT_CARD"),
            (not state.interventions_valid, "INVALID_INTERVENTION"),
            (state.prohibited_attribute_requested, "PROTECTED_ATTRIBUTE_POLICY"),
            (state.out_of_domain, "OUT_OF_DOMAIN"),
            (not state.candidate_set_complete, "CANDIDATE_COVERAGE_FAILURE"),
            (
                features.base_sense_id == self.config.other_unlisted_id,
                "OTHER_UNLISTED_SELECTED",
            ),
            (features.context_conflict, "CONTEXT_CONFLICT"),
            (not state.compute_budget_available, "BUDGET_EXHAUSTED"),
            (
                bool(state.missing_discriminating_slots),
                "UNRESOLVED_MISSING_CONTEXT",
            ),
            (
                features.intervention_validity
                < self.config.minimum_intervention_validity,
                "LOW_INTERVENTION_VALIDITY",
            ),
            (
                features.preservation_invariance
                < self.config.minimum_preservation_invariance,
                "PRESERVATION_FAILURE",
            ),
            (
                features.targeted_response
                < self.config.minimum_targeted_response,
                "TARGETED_RESPONSE_FAILURE",
            ),
        )
        for condition, reason in failures:
            if condition:
                return False, reason
        return True, "ELIGIBLE"

    def raw_reliability(self, features: VerificationFeatures) -> float:
        components = {
            "base_confidence": 0.5
            * (features.top_two_margin + 1.0 - features.normalized_entropy),
            "preservation": (
                features.preservation_invariance
                + features.preservation_probability
                + 1.0
                - features.mean_preserving_jsd
            )
            / 3.0,
            "targeted_change": 0.5
            * (
                features.targeted_response
                + 0.5 * (features.mean_changing_target_margin + 1.0)
            ),
            "context_quality": min(
                features.intervention_validity, features.context_completeness
            ),
        }
        weights = self.config.reliability_weights
        unknown = set(weights).difference(components)
        if unknown:
            raise ValueError(f"unknown reliability feature weights: {sorted(unknown)}")
        total = sum(float(value) for value in weights.values())
        return _clip01(
            sum(float(weight) * components[name] for name, weight in weights.items())
            / total
        )

    def question_utility(
        self, question: ClarificationQuestion, base_scores: ProbabilityVector
    ) -> float:
        if not question.approved or question.sensitive:
            return float("-inf")
        return (
            question.expected_information_gain(base_scores)
            - self.config.interaction_cost_weight * question.interaction_cost
            - self.config.privacy_cost_weight * question.privacy_cost
            - self.config.compute_cost_weight * question.compute_cost
        )

    def select_question(
        self,
        questions: Iterable[ClarificationQuestion],
        base_scores: ProbabilityVector,
        missing_slots: Sequence[str],
    ) -> tuple[ClarificationQuestion | None, float]:
        allowed = set(missing_slots)
        ranked = [
            (self.question_utility(question, base_scores), question)
            for question in questions
            if question.context_slot in allowed
        ]
        if not ranked:
            return None, float("-inf")
        utility, question = max(ranked, key=lambda item: (item[0], item[1].question_id))
        return question, utility

    def decide(
        self,
        features: VerificationFeatures,
        state: GateState,
        base_scores: ProbabilityVector,
        questions: Iterable[ClarificationQuestion] = (),
    ) -> GateDecision:
        raw = self.raw_reliability(features)
        estimated = _clip01(self.calibrator.predict(raw))

        eligible, eligibility_reason = self.commit_eligibility(features, state)
        has_missing_discriminator = bool(state.missing_discriminating_slots)
        if not eligible and eligibility_reason != "UNRESOLVED_MISSING_CONTEXT":
            return GateDecision(
                Action.ABSTAIN_ESCALATE,
                estimated,
                raw,
                reason_code=eligibility_reason,
            )

        can_commit = (
            eligible
            and estimated >= self.config.commit_threshold
        )
        if can_commit:
            return GateDecision(
                Action.COMMIT,
                estimated,
                raw,
                sense_id=features.base_sense_id,
                reason_code="CALIBRATED_COMMIT",
            )

        if has_missing_discriminator and state.questions_remaining > 0:
            question, utility = self.select_question(
                questions, base_scores, state.missing_discriminating_slots
            )
            if question is not None and utility >= self.config.minimum_question_utility:
                return GateDecision(
                    Action.CLARIFY,
                    estimated,
                    raw,
                    question_id=question.question_id,
                    context_slot=question.context_slot,
                    question_utility=utility,
                    reason_code="TARGETED_INFORMATION_GAIN",
                )

        reason = "LOW_CALIBRATED_RELIABILITY"
        if has_missing_discriminator:
            reason = "UNRESOLVED_MISSING_CONTEXT"
        elif features.intervention_validity < self.config.minimum_intervention_validity:
            reason = "LOW_INTERVENTION_VALIDITY"
        return GateDecision(Action.ABSTAIN_ESCALATE, estimated, raw, reason_code=reason)


@dataclass(frozen=True)
class CoverageSelection:
    selected_indices: tuple[int, ...]
    requested_coverage: float
    achieved_coverage: float
    eligible_coverage: float
    target_unattainable: bool


def common_feasible_coverage(
    requested_coverage: float, eligibility_masks: Sequence[Sequence[bool]]
) -> float:
    """Cap a calibration-selected target at every method's eligible coverage."""

    requested = float(requested_coverage)
    if not 0 <= requested <= 1:
        raise ValueError("requested coverage must be within [0, 1]")
    if not eligibility_masks or any(not mask for mask in eligibility_masks):
        raise ValueError("at least one non-empty eligibility mask is required")
    feasible = min(sum(bool(value) for value in mask) / len(mask) for mask in eligibility_masks)
    return min(requested, feasible)


def select_eligible_at_coverage(
    scores: Sequence[float],
    eligible: Sequence[bool],
    tie_break_hashes: Sequence[str],
    target_coverage: float,
) -> CoverageSelection:
    """Select a label-blind top-score fraction without overriding hard masks."""

    if not scores or len(scores) != len(eligible) or len(scores) != len(tie_break_hashes):
        raise ValueError("scores, eligibility, and tie hashes must be non-empty and aligned")
    target = float(target_coverage)
    if not 0 <= target <= 1:
        raise ValueError("target coverage must be within [0, 1]")
    if any(not isfinite(float(score)) for score in scores):
        raise ValueError("coverage-ranking scores must be finite")
    if any(not str(value) for value in tie_break_hashes):
        raise ValueError("tie-break hashes must be non-empty")

    count = len(scores)
    target_count = int(target * count)
    ranked = sorted(
        (index for index, allowed in enumerate(eligible) if allowed),
        key=lambda index: (-float(scores[index]), str(tie_break_hashes[index])),
    )
    selected = tuple(ranked[:target_count])
    eligible_count = len(ranked)
    return CoverageSelection(
        selected_indices=selected,
        requested_coverage=target,
        achieved_coverage=len(selected) / count,
        eligible_coverage=eligible_count / count,
        target_unattainable=eligible_count < target_count,
    )


def route_if_beneficial(
    *,
    context_complete: bool,
    context_conflict: bool,
    privacy_allowed: bool,
    expected_small_correctness: float,
    expected_large_correctness: float,
    normalized_route_cost: float,
    cost_weight: float,
) -> bool:
    """Return an internal routing decision; this is not an external action."""

    if not context_complete or context_conflict or not privacy_allowed:
        return False
    benefit = float(expected_large_correctness) - float(expected_small_correctness)
    return benefit > float(cost_weight) * max(0.0, float(normalized_route_cost))
