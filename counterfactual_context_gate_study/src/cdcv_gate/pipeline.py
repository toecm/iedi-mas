"""Gold-free orchestration for the CDCV-Gate reference implementation.

This provider-neutral module is a protocol implementation, not empirical
evidence. DEMO mode accepts invented development fixtures only. SEALED mode
requires an authorized, access-controlled process and accepted attestations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Mapping, Protocol, Sequence

from .contracts import (
    OTHER_UNLISTED,
    PERMITTED_CONTEXT_SLOTS,
    PROTECTED_IDENTITY_FIELDS,
    SEALED_LABEL_FIELDS,
    apply_context_patch,
    sha256_json,
    validate_context_card,
)
from .core import (
    Action,
    AnswerScenario,
    ChangingProbe,
    ClarificationQuestion,
    GateDecision,
    GatePolicy,
    GateState,
    PreservingProbe,
    VerificationFeatures,
    compute_verification_features,
    normalize,
    normalized_entropy,
    route_if_beneficial,
)


class AttestationStatus(str, Enum):
    DEMO_ONLY = "DEMO_ONLY"
    ACCEPTED = "ACCEPTED"


def _hash_ok(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _all_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            found.add(str(key))
            found.update(_all_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(_all_keys(nested))
    return found


def _top(scores: Mapping[str, float], order: Sequence[str]) -> str:
    return max(order, key=lambda candidate: float(scores[candidate]))


@dataclass(frozen=True)
class ResourceUse:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    peak_memory_mb: float | None = None
    cost_usd: float | None = 0.0

    def __post_init__(self) -> None:
        if self.calls < 0 or self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("resource counters must be non-negative")
        if not isfinite(float(self.latency_ms)) or self.latency_ms < 0:
            raise ValueError("latency must be finite and non-negative")
        for name, value in (
            ("peak_memory_mb", self.peak_memory_mb),
            ("cost_usd", self.cost_usd),
        ):
            if value is not None and (not isfinite(float(value)) or value < 0):
                raise ValueError(f"{name} must be null or finite and non-negative")

    def __add__(self, other: "ResourceUse") -> "ResourceUse":
        memory = [
            value
            for value in (self.peak_memory_mb, other.peak_memory_mb)
            if value is not None
        ]
        cost = (
            None
            if self.cost_usd is None or other.cost_usd is None
            else self.cost_usd + other.cost_usd
        )
        return ResourceUse(
            self.calls + other.calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.latency_ms + other.latency_ms,
            max(memory) if memory else None,
            cost,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResourceEvent:
    stage: str
    resources: ResourceUse


@dataclass(frozen=True)
class BudgetEnvelope:
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.max_calls <= 0:
            raise ValueError("at least one call must be allocated")
        if self.max_input_tokens < 0 or self.max_output_tokens < 0:
            raise ValueError("token ceilings must be non-negative")
        if self.max_cost_usd is not None and (
            not isfinite(float(self.max_cost_usd)) or self.max_cost_usd < 0
        ):
            raise ValueError("cost ceiling must be null or finite and non-negative")

    def as_allocated_resource(self, observed: ResourceUse) -> ResourceUse:
        return ResourceUse(
            calls=self.max_calls,
            input_tokens=self.max_input_tokens,
            output_tokens=self.max_output_tokens,
            latency_ms=observed.latency_ms,
            peak_memory_mb=observed.peak_memory_mb,
            cost_usd=(
                self.max_cost_usd
                if self.max_cost_usd is not None
                else observed.cost_usd
            ),
        )


@dataclass
class BudgetLedger:
    envelope: BudgetEnvelope
    events: list[ResourceEvent] = field(default_factory=list)

    @property
    def consumed(self) -> ResourceUse:
        total = ResourceUse()
        for event in self.events:
            total = total + event.resources
        return total

    @property
    def remaining_calls(self) -> int:
        return self.envelope.max_calls - self.consumed.calls

    def charge(self, stage: str, resources: ResourceUse) -> None:
        after = self.consumed + resources
        if after.calls > self.envelope.max_calls:
            raise RuntimeError("scorer-call budget exceeded")
        if after.input_tokens > self.envelope.max_input_tokens:
            raise RuntimeError("input-token budget exceeded")
        if after.output_tokens > self.envelope.max_output_tokens:
            raise RuntimeError("output-token budget exceeded")
        if self.envelope.max_cost_usd is not None:
            if after.cost_usd is None:
                raise RuntimeError("cost is unknown under a finite cost ceiling")
            if after.cost_usd > self.envelope.max_cost_usd:
                raise RuntimeError("cost budget exceeded")
        self.events.append(ResourceEvent(stage, resources))


@dataclass(frozen=True)
class ScoringRequest:
    utterance: str
    candidate_ids: tuple[str, ...]
    candidate_definitions: Mapping[str, str]
    context_card: Mapping[str, object]
    seed: int
    opaque_call_id: str

    def __post_init__(self) -> None:
        forbidden = _all_keys(self.context_card).intersection(
            SEALED_LABEL_FIELDS | PROTECTED_IDENTITY_FIELDS
        )
        if forbidden:
            raise ValueError(f"forbidden scorer context keys: {sorted(forbidden)}")
        if set(self.context_card).difference({"scope", "variety_cue", "fields"}):
            raise ValueError("scorer context is not an allow-listed projection")


@dataclass(frozen=True)
class ScoreResponse:
    scores: Mapping[str, float]
    resources: ResourceUse = field(default_factory=lambda: ResourceUse(calls=1))
    prompt_hash: str | None = None


class CandidateScorer(Protocol):
    model_id: str

    def score(self, request: ScoringRequest) -> ScoreResponse: ...


@dataclass(frozen=True)
class CandidateSet:
    candidate_ids: tuple[str, ...]
    definitions: Mapping[str, str]
    source: str

    def __post_init__(self) -> None:
        if len(self.candidate_ids) != 3 or len(set(self.candidate_ids)) != 3:
            raise ValueError("exactly three distinct candidates are required")
        if self.candidate_ids.count(OTHER_UNLISTED) != 1:
            raise ValueError("exactly one OTHER_UNLISTED candidate is required")
        if set(self.definitions) != set(self.candidate_ids):
            raise ValueError("candidate definitions must match candidate IDs")


class CandidateGenerator(Protocol):
    generator_id: str

    def generate(self, utterance: str) -> CandidateSet: ...


@dataclass(frozen=True)
class FixedCandidateProvider:
    generator_id: str
    candidate_set: CandidateSet

    def generate(self, utterance: str) -> CandidateSet:
        if not str(utterance).strip():
            raise ValueError("utterance is required")
        return self.candidate_set


@dataclass
class ScriptedScorer:
    """Offline deterministic test adapter, never an empirical baseline."""

    model_id: str
    responses: Sequence[Mapping[str, float]]
    resource_per_call: ResourceUse = field(
        default_factory=lambda: ResourceUse(
            calls=1,
            input_tokens=24,
            output_tokens=3,
            latency_ms=1.0,
            peak_memory_mb=1.0,
            cost_usd=0.0,
        )
    )
    _cursor: int = field(default=0, init=False, repr=False)
    seen_requests: list[ScoringRequest] = field(default_factory=list, init=False)

    def score(self, request: ScoringRequest) -> ScoreResponse:
        if self._cursor >= len(self.responses):
            raise RuntimeError("scripted scorer has no remaining response")
        scores = normalize(self.responses[self._cursor])
        self._cursor += 1
        if set(scores) != set(request.candidate_ids):
            raise ValueError("scorer changed the candidate set")
        self.seen_requests.append(request)
        return ScoreResponse(
            scores=scores,
            resources=self.resource_per_call,
            prompt_hash=sha256_json(
                {
                    "adapter": "ScriptedScorer/2",
                    "model_id": self.model_id,
                    "candidate_ids": request.candidate_ids,
                    "candidate_definitions": request.candidate_definitions,
                    "context_card": request.context_card,
                }
            ),
        )


@dataclass(frozen=True)
class ProbeContract:
    probe_id: str
    probe_type: str
    source_candidate_id: str
    changed_slots: tuple[str, ...]
    context_patch: Mapping[str, Mapping[str, object]]
    source_context_hash: str
    result_context_hash: str
    target_candidate_id: str | None = None
    scenario_answer_id: str | None = None
    validity_weight: float = 1.0
    review_status: AttestationStatus = AttestationStatus.DEMO_ONLY
    safety_status: AttestationStatus = AttestationStatus.DEMO_ONLY
    protected_attribute_manipulated: bool = False
    gloss_leakage_detected: bool = False

    def __post_init__(self) -> None:
        if self.probe_type not in {"PRESERVING", "MEANING_CHANGING"}:
            raise ValueError("unknown probe type")
        if not self.changed_slots or not set(self.changed_slots).issubset(
            PERMITTED_CONTEXT_SLOTS
        ):
            raise ValueError("probe changes a prohibited or empty slot set")
        if set(self.context_patch) != set(self.changed_slots):
            raise ValueError("probe patch must exactly match changed slots")
        if self.probe_type == "PRESERVING" and self.target_candidate_id is not None:
            raise ValueError("preserving probe cannot name a target candidate")
        if self.probe_type == "MEANING_CHANGING" and not self.target_candidate_id:
            raise ValueError("meaning-changing probe requires a target candidate")
        if not _hash_ok(self.source_context_hash) or not _hash_ok(
            self.result_context_hash
        ):
            raise ValueError("probe context hashes must be SHA-256 values")
        if not 0 <= float(self.validity_weight) <= 1:
            raise ValueError("probe validity weight must be within [0, 1]")
        forbidden = _all_keys(self.context_patch).intersection(
            SEALED_LABEL_FIELDS | PROTECTED_IDENTITY_FIELDS
        )
        if forbidden:
            raise ValueError(f"forbidden probe keys: {sorted(forbidden)}")


@dataclass(frozen=True)
class CandidateBranch:
    source_candidate_id: str
    preserving: ProbeContract
    meaning_changing: ProbeContract


def branch_manifest_hash(branches: Mapping[str, CandidateBranch]) -> str:
    return sha256_json(
        {candidate: asdict(branches[candidate]) for candidate in sorted(branches)}
    )


def candidate_set_manifest_hash(
    candidate_ids: Sequence[str], definitions: Mapping[str, str]
) -> str:
    return sha256_json({"order": tuple(candidate_ids), "definitions": definitions})


@dataclass(frozen=True)
class ScenarioReference:
    answer_id: str
    prior_probability: float
    probe_id: str

    def __post_init__(self) -> None:
        if not self.answer_id or not self.probe_id:
            raise ValueError("scenario answer and probe IDs are required")
        if not isfinite(float(self.prior_probability)) or self.prior_probability < 0:
            raise ValueError("scenario prior must be finite and non-negative")


@dataclass(frozen=True)
class QuestionContract:
    question_id: str
    context_slot: str
    scenarios: tuple[ScenarioReference, ...]
    interaction_cost: float = 0.0
    privacy_cost: float = 0.0
    compute_cost: float = 0.0
    approved: bool = True
    sensitive: bool = False
    leadingness_median: float = 1.0
    review_status: AttestationStatus = AttestationStatus.DEMO_ONLY
    safety_status: AttestationStatus = AttestationStatus.DEMO_ONLY

    def __post_init__(self) -> None:
        if self.context_slot not in PERMITTED_CONTEXT_SLOTS:
            raise ValueError("question requests a prohibited slot")
        for value in (
            self.interaction_cost,
            self.privacy_cost,
            self.compute_cost,
            self.leadingness_median,
        ):
            if not isfinite(float(value)) or value < 0:
                raise ValueError("question costs and ratings must be non-negative")

    @property
    def manifest_hash(self) -> str:
        return sha256_json(asdict(self))

    def materialize(
        self,
        results: Mapping[str, tuple[ProbeContract, Mapping[str, float]]],
    ) -> ClarificationQuestion:
        if not self.approved or self.sensitive or self.leadingness_median > 2:
            raise ValueError("question is not approved")
        if len(self.scenarios) < 2:
            raise ValueError("question requires at least two scenarios")
        if len({item.answer_id for item in self.scenarios}) != len(self.scenarios):
            raise ValueError("question answer IDs must be unique")
        if abs(sum(item.prior_probability for item in self.scenarios) - 1.0) > 1e-9:
            raise ValueError("question scenario priors must sum to one")
        answers = []
        for scenario in self.scenarios:
            if scenario.probe_id not in results:
                raise ValueError("question references an unavailable probe")
            probe, scores = results[scenario.probe_id]
            if probe.scenario_answer_id != scenario.answer_id:
                raise ValueError("answer ID is not bound to the referenced probe")
            if probe.changed_slots != (self.context_slot,):
                raise ValueError("question scenario changes a different slot")
            answers.append(AnswerScenario(scenario.prior_probability, scores))
        return ClarificationQuestion(
            self.question_id,
            self.context_slot,
            tuple(answers),
            self.interaction_cost,
            self.privacy_cost,
            self.compute_cost,
            True,
            False,
        )


def question_bank_manifest_hash(
    bank: Mapping[str, tuple[QuestionContract, ...]]
) -> str:
    return sha256_json(
        {
            candidate: [asdict(question) for question in bank[candidate]]
            for candidate in sorted(bank)
        }
    )


@dataclass(frozen=True)
class ContractAttestation:
    manifest_id: str
    case_id: str
    family_id: str
    context_card_hash: str
    candidate_set_hash: str
    review_status: AttestationStatus
    schema_status: AttestationStatus
    cross_record_status: AttestationStatus
    safety_status: AttestationStatus
    required_context_slots: tuple[str, ...]
    intervention_manifest_hash: str
    question_manifest_hash: str
    reviewed_value_manifest_hash: str
    integrity_manifest_hash: str
    constructed_without_sealed_labels: bool = True
    protected_attribute_manipulated: bool = False
    gloss_leakage_detected: bool = False
    variety_fixed: bool = True

    def __post_init__(self) -> None:
        if not self.required_context_slots:
            raise ValueError("attestation requires at least one context slot")
        if not set(self.required_context_slots).issubset(PERMITTED_CONTEXT_SLOTS):
            raise ValueError("attestation requires a prohibited slot")
        if len(set(self.required_context_slots)) != len(self.required_context_slots):
            raise ValueError("required slots must be unique")
        if not all(
            _hash_ok(value)
            for value in (
                self.context_card_hash,
                self.candidate_set_hash,
                self.intervention_manifest_hash,
                self.question_manifest_hash,
                self.reviewed_value_manifest_hash,
                self.integrity_manifest_hash,
            )
        ):
            raise ValueError("attestation manifest hashes must be SHA-256 values")

    def payload(self) -> dict:
        value = asdict(self)
        value.pop("integrity_manifest_hash")
        return value

    def computed_integrity_hash(self) -> str:
        return sha256_json(self.payload())

    def errors(self, sealed: bool) -> list[str]:
        errors: list[str] = []
        if sealed and any(
            status != AttestationStatus.ACCEPTED
            for status in (
                self.review_status,
                self.schema_status,
                self.cross_record_status,
                self.safety_status,
            )
        ):
            errors.append("sealed execution requires ACCEPTED attestation statuses")
        if not self.constructed_without_sealed_labels:
            errors.append("attestation used sealed labels during construction")
        if self.protected_attribute_manipulated:
            errors.append("attestation manipulates a protected identity attribute")
        if self.gloss_leakage_detected:
            errors.append("attestation reports gloss leakage")
        if not self.variety_fixed:
            errors.append("attestation does not fix the episode variety")
        return errors


def attestation_integrity_manifest_hash(
    attestation: ContractAttestation,
) -> str:
    return attestation.computed_integrity_hash()


@dataclass(frozen=True)
class ReleasedAnswer:
    case_id: str
    question_id: str
    answer_id: str
    context_slot: str
    context_patch: Mapping[str, Mapping[str, object]]
    question_manifest_hash: str
    answer_manifest_hash: str
    validation_status: AttestationStatus
    safety_status: AttestationStatus
    post_answer_branch_manifest_hash: str
    post_answer_branches: Mapping[str, CandidateBranch]

    def __post_init__(self) -> None:
        if not all(
            _hash_ok(value)
            for value in (
                self.question_manifest_hash,
                self.answer_manifest_hash,
                self.post_answer_branch_manifest_hash,
            )
        ):
            raise ValueError("released-answer manifest hashes must be SHA-256 values")
        forbidden = _all_keys(self.context_patch).intersection(
            SEALED_LABEL_FIELDS | PROTECTED_IDENTITY_FIELDS
        )
        if forbidden:
            raise ValueError(f"forbidden released-answer keys: {sorted(forbidden)}")

    def payload(self) -> dict:
        return {
            "case_id": self.case_id,
            "question_id": self.question_id,
            "answer_id": self.answer_id,
            "context_slot": self.context_slot,
            "context_patch": self.context_patch,
            "question_manifest_hash": self.question_manifest_hash,
            "validation_status": self.validation_status,
            "safety_status": self.safety_status,
            "post_answer_branch_manifest_hash": (
                self.post_answer_branch_manifest_hash
            ),
            "post_answer_branches": {
                key: asdict(self.post_answer_branches[key])
                for key in sorted(self.post_answer_branches)
            },
        }

    def computed_manifest_hash(self) -> str:
        return sha256_json(self.payload())

    def apply(self, card: Mapping[str, object]) -> dict:
        if set(self.context_patch) != {self.context_slot}:
            raise ValueError("answer patch must modify only the requested slot")
        value = self.context_patch[self.context_slot]
        if value.get("provenance") != "standardized_clarification":
            raise ValueError("answer patch has invalid provenance")
        return apply_context_patch(card, (self.context_slot,), self.context_patch)


def released_answer_manifest_hash(answer: ReleasedAnswer) -> str:
    return answer.computed_manifest_hash()


class ClarificationAnswerBroker(Protocol):
    def release(
        self, case_id: str, question: QuestionContract
    ) -> ReleasedAnswer | None: ...


@dataclass(frozen=True)
class StaticDemoAnswerBroker:
    answers: Mapping[tuple[str, str], ReleasedAnswer]

    def release(
        self, case_id: str, question: QuestionContract
    ) -> ReleasedAnswer | None:
        return self.answers.get((case_id, question.question_id))


@dataclass(frozen=True)
class RuntimeEpisode:
    case_id: str
    family_id: str
    split: str
    utterance: str
    candidate_ids: tuple[str, ...]
    candidate_definitions: Mapping[str, str]
    context_card: Mapping[str, object]
    branches: Mapping[str, CandidateBranch]
    attestation: ContractAttestation
    questions_by_candidate: Mapping[str, tuple[QuestionContract, ...]] = field(
        default_factory=dict
    )
    out_of_domain: bool = False

    def __post_init__(self) -> None:
        if self.split not in {"development", "calibration", "sealed_test"}:
            raise ValueError("unsupported split")

    @property
    def core_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            candidate
            for candidate in self.candidate_ids
            if candidate != OTHER_UNLISTED
        )


@dataclass(frozen=True)
class RoutingOption:
    expected_small_correctness: float
    expected_large_correctness: float
    normalized_route_cost: float
    cost_weight: float
    privacy_allowed: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("expected_small_correctness", self.expected_small_correctness),
            ("expected_large_correctness", self.expected_large_correctness),
        ):
            if not isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        for name, value in (
            ("normalized_route_cost", self.normalized_route_cost),
            ("cost_weight", self.cost_weight),
        ):
            if not isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class VerificationPass:
    pass_name: str
    model_id: str
    base_scores: Mapping[str, float]
    selected_branch: str | None
    probes: tuple[ProbeContract, ...]
    probe_scores: Mapping[str, Mapping[str, float]]
    prompt_hashes: tuple[str, ...]
    features: VerificationFeatures
    decision: GateDecision


@dataclass(frozen=True)
class RunResult:
    episode: RuntimeEpisode
    initial_decision: GateDecision
    final_decision: GateDecision
    passes: tuple[VerificationPass, ...]
    resource_events: tuple[ResourceEvent, ...]
    budget_envelope: BudgetEnvelope
    selected_question: QuestionContract | None = None
    answer_applied: bool = False
    applied_answer_id: str | None = None
    applied_answer_hash: str | None = None
    routed: bool = False
    routing_predicted_benefit: float | None = None
    contract_errors: tuple[str, ...] = ()

    @property
    def consumed(self) -> ResourceUse:
        total = ResourceUse()
        for event in self.resource_events:
            total = total + event.resources
        return total


@dataclass(frozen=True)
class EqualBudgetResult:
    mean_scores: Mapping[str, float]
    sample_agreement: float
    resource_events: tuple[ResourceEvent, ...]
    budget_envelope: BudgetEnvelope


def _project_context(card: Mapping[str, object]) -> dict:
    errors = validate_context_card(card)
    if errors:
        raise ValueError(f"invalid context card: {errors}")
    variety = card.get("variety_cue", {})
    fields = card.get("fields", {})
    projected_fields: dict[str, dict] = {}
    if isinstance(fields, Mapping):
        for slot in sorted(PERMITTED_CONTEXT_SLOTS):
            value = fields.get(slot)
            if isinstance(value, Mapping):
                projected_fields[slot] = {
                    key: value[key]
                    for key in (
                        "value",
                        "provenance",
                        "confidence",
                        "retain_after_episode",
                    )
                    if key in value
                }
    projected = {
        "scope": card.get("scope"),
        "variety_cue": (
            {
                key: variety[key]
                for key in ("value", "provenance", "retain_after_episode")
                if key in variety
            }
            if isinstance(variety, Mapping)
            else {}
        ),
        "fields": projected_fields,
    }
    forbidden = _all_keys(projected).intersection(
        SEALED_LABEL_FIELDS | PROTECTED_IDENTITY_FIELDS
    )
    if forbidden:
        raise ValueError(f"forbidden context projection keys: {sorted(forbidden)}")
    return projected


def _hard_conflict(card: Mapping[str, object]) -> bool:
    conflicts = card.get("conflicts", [])
    return isinstance(conflicts, list) and any(
        isinstance(item, Mapping) and item.get("severity") == "hard_stop"
        for item in conflicts
    )


def _completeness(
    card: Mapping[str, object], required_slots: Sequence[str]
) -> float:
    fields = card.get("fields", {})
    if not required_slots:
        return 1.0
    if not isinstance(fields, Mapping):
        return 0.0
    present = sum(
        1
        for slot in required_slots
        if isinstance(fields.get(slot), Mapping)
        and fields[slot].get("value") is not None
        and fields[slot].get("provenance") != "missing"
    )
    return present / len(required_slots)


def _missing(card: Mapping[str, object], required: Sequence[str]) -> tuple[str, ...]:
    fields = card.get("fields", {})
    result = []
    for slot in required:
        value = fields.get(slot) if isinstance(fields, Mapping) else None
        if (
            not isinstance(value, Mapping)
            or value.get("value") is None
            or value.get("provenance") == "missing"
        ):
            result.append(slot)
    return tuple(result)


def _branch_errors(
    card: Mapping[str, object],
    candidate_ids: Sequence[str],
    branches: Mapping[str, CandidateBranch],
    *,
    sealed: bool = False,
) -> list[str]:
    errors: list[str] = []
    core = tuple(
        candidate for candidate in candidate_ids if candidate != OTHER_UNLISTED
    )
    if len(core) != 2:
        return ["exactly two core candidates are required"]
    if set(branches) != set(core):
        errors.append("probe branches are not symmetric across core candidates")
    source_hash = sha256_json(card)
    seen: set[str] = set()
    for candidate in core:
        branch = branches.get(candidate)
        if branch is None:
            continue
        if branch.source_candidate_id != candidate:
            errors.append("branch source differs from branch key")
        other = next(value for value in core if value != candidate)
        preserving = branch.preserving
        changing = branch.meaning_changing
        if preserving.probe_type != "PRESERVING":
            errors.append("preserving branch contains the wrong probe type")
        if preserving.source_candidate_id != candidate:
            errors.append("preserving probe source differs from branch source")
        if changing.probe_type != "MEANING_CHANGING":
            errors.append("changing branch contains the wrong probe type")
        if changing.source_candidate_id != candidate:
            errors.append("changing probe source differs from branch source")
        if changing.target_candidate_id != other:
            errors.append("changing probe does not target the other core candidate")
        for probe in (preserving, changing):
            if probe.probe_id in seen:
                errors.append("probe IDs must be unique across branches")
            seen.add(probe.probe_id)
            if probe.source_context_hash != source_hash:
                errors.append(f"probe {probe.probe_id!r} source hash mismatch")
            if sealed and (
                probe.review_status != AttestationStatus.ACCEPTED
                or probe.safety_status != AttestationStatus.ACCEPTED
            ):
                errors.append(f"probe {probe.probe_id!r} is not accepted")
            if probe.protected_attribute_manipulated:
                errors.append(f"probe {probe.probe_id!r} manipulates protected identity")
            if probe.gloss_leakage_detected:
                errors.append(f"probe {probe.probe_id!r} contains gloss leakage")
            try:
                result = apply_context_patch(
                    card, probe.changed_slots, probe.context_patch
                )
            except ValueError as exc:
                errors.append(f"probe {probe.probe_id!r} is invalid: {exc}")
                continue
            if probe.result_context_hash != sha256_json(result):
                errors.append(f"probe {probe.probe_id!r} result hash mismatch")
    return errors


def validate_episode_contract(episode: RuntimeEpisode) -> list[str]:
    errors = validate_context_card(episode.context_card)
    if len(episode.candidate_ids) != 3 or len(set(episode.candidate_ids)) != 3:
        errors.append("episode requires exactly three distinct candidates")
    if episode.candidate_ids.count(OTHER_UNLISTED) != 1:
        errors.append("episode requires exactly one OTHER_UNLISTED")
    if set(episode.candidate_definitions) != set(episode.candidate_ids):
        errors.append("candidate definitions do not match candidate IDs")
    errors.extend(
        _branch_errors(episode.context_card, episode.candidate_ids, episode.branches)
    )
    if episode.attestation.case_id != episode.case_id:
        errors.append("attestation belongs to a different case")
    if episode.attestation.family_id != episode.family_id:
        errors.append("attestation belongs to a different family")
    if episode.attestation.context_card_hash != sha256_json(episode.context_card):
        errors.append("attested context-card hash mismatch")
    if episode.attestation.candidate_set_hash != candidate_set_manifest_hash(
        episode.candidate_ids, episode.candidate_definitions
    ):
        errors.append("attested candidate-set hash mismatch")
    if (
        episode.attestation.integrity_manifest_hash
        != episode.attestation.computed_integrity_hash()
    ):
        errors.append("attestation integrity-manifest hash mismatch")
    if (
        episode.attestation.intervention_manifest_hash
        != branch_manifest_hash(episode.branches)
    ):
        errors.append("attested intervention manifest hash mismatch")
    if (
        episode.attestation.question_manifest_hash
        != question_bank_manifest_hash(episode.questions_by_candidate)
    ):
        errors.append("attested question manifest hash mismatch")

    core = set(episode.core_candidate_ids)
    if set(episode.questions_by_candidate).difference(core):
        errors.append("question bank names a non-core branch")
    question_ids: set[str] = set()
    for candidate, questions in episode.questions_by_candidate.items():
        branch = episode.branches.get(candidate)
        if branch is None:
            continue
        probes = {
            branch.preserving.probe_id: branch.preserving,
            branch.meaning_changing.probe_id: branch.meaning_changing,
        }
        for question in questions:
            if question.question_id in question_ids:
                errors.append("question IDs must be globally unique")
            question_ids.add(question.question_id)
            if question.context_slot not in episode.attestation.required_context_slots:
                errors.append("question slot is not a required context slot")
            if len(question.scenarios) < 2:
                errors.append("question requires at least two scenarios")
            if (
                abs(
                    sum(
                        float(scenario.prior_probability)
                        for scenario in question.scenarios
                    )
                    - 1.0
                )
                > 1e-9
            ):
                errors.append("question scenario priors must sum to one")
            if len({item.answer_id for item in question.scenarios}) != len(
                question.scenarios
            ):
                errors.append("question answer IDs must be unique")
            for scenario in question.scenarios:
                probe = probes.get(scenario.probe_id)
                if probe is None:
                    errors.append("question scenario crosses candidate branches")
                    continue
                if probe.changed_slots != (question.context_slot,):
                    errors.append("question scenario changes the wrong slot")
                if probe.scenario_answer_id != scenario.answer_id:
                    errors.append("question answer is not bound to its probe")
    return errors


def _fallback_features(
    scores: Mapping[str, float],
    order: Sequence[str],
    card: Mapping[str, object],
    required_slots: Sequence[str],
) -> VerificationFeatures:
    normalized = normalize(scores)
    ranked = sorted(normalized.values(), reverse=True)
    return VerificationFeatures(
        base_sense_id=_top(normalized, order),
        top_two_margin=ranked[0] - ranked[1],
        normalized_entropy=normalized_entropy(normalized),
        preservation_invariance=0.0,
        preservation_probability=0.0,
        mean_preserving_jsd=1.0,
        targeted_response=0.0,
        mean_changing_target_margin=-1.0,
        intervention_validity=0.0,
        context_completeness=_completeness(card, required_slots),
        context_conflict=_hard_conflict(card),
    )


def _contract_admission_errors(
    episode: RuntimeEpisode,
    *,
    run_mode: str,
    sealed_execution_authorized: bool,
    trusted_attestation_hashes: frozenset[str],
) -> list[str]:
    if run_mode not in {"DEMO", "SEALED"}:
        return ["run mode must be DEMO or SEALED"]
    sealed = run_mode == "SEALED"
    errors = list(validate_episode_contract(episode))
    if not sealed and episode.split != "development":
        errors.append("DEMO mode accepts development episodes only")
    if sealed:
        if not sealed_execution_authorized:
            errors.append("SEALED mode requires an authorized process")
        if episode.split != "sealed_test":
            errors.append("SEALED mode accepts sealed_test episodes only")
        if (
            episode.attestation.integrity_manifest_hash
            not in trusted_attestation_hashes
        ):
            errors.append("attestation is absent from the frozen trusted manifest")
    errors.extend(episode.attestation.errors(sealed))
    errors.extend(
        _branch_errors(
            episode.context_card,
            episode.candidate_ids,
            episode.branches,
            sealed=sealed,
        )
    )
    if sealed:
        for questions in episode.questions_by_candidate.values():
            for question in questions:
                if (
                    question.review_status != AttestationStatus.ACCEPTED
                    or question.safety_status != AttestationStatus.ACCEPTED
                ):
                    errors.append(f"question {question.question_id!r} is not accepted")
    return list(dict.fromkeys(errors))


class CDCVRunner:
    """Initial verification, one clarification repair, and one optional route."""

    def __init__(
        self,
        policy: GatePolicy,
        *,
        budget_envelope: BudgetEnvelope,
        run_mode: str = "DEMO",
        sealed_execution_authorized: bool = False,
        trusted_attestation_hashes: frozenset[str] | None = None,
        trusted_answer_manifest_hashes: frozenset[str] | None = None,
    ) -> None:
        if run_mode not in {"DEMO", "SEALED"}:
            raise ValueError("run mode must be DEMO or SEALED")
        if run_mode == "SEALED" and not sealed_execution_authorized:
            raise PermissionError("sealed execution requires an authorized process")
        if run_mode == "SEALED" and not trusted_attestation_hashes:
            raise PermissionError(
                "sealed execution requires a frozen trusted-attestation manifest"
            )
        if budget_envelope.max_calls != 9:
            raise ValueError("the CDCV call cap is exactly nine")
        self.policy = policy
        self.budget_envelope = budget_envelope
        self.run_mode = run_mode
        self.sealed_execution_authorized = sealed_execution_authorized
        self.trusted_attestation_hashes = trusted_attestation_hashes or frozenset()
        self.trusted_answer_manifest_hashes = (
            trusted_answer_manifest_hashes or frozenset()
        )

    def _preflight(
        self, episode: RuntimeEpisode
    ) -> tuple[str | None, tuple[str, ...]]:
        errors = _contract_admission_errors(
            episode,
            run_mode=self.run_mode,
            sealed_execution_authorized=self.sealed_execution_authorized,
            trusted_attestation_hashes=self.trusted_attestation_hashes,
        )
        if errors:
            return "INVALID_RUNTIME_CONTRACT", tuple(dict.fromkeys(errors))
        if episode.out_of_domain:
            return "OUT_OF_DOMAIN", ()
        if _hard_conflict(episode.context_card):
            return "CONTEXT_CONFLICT", ()
        return None, ()

    @staticmethod
    def _score(
        scorer: CandidateScorer,
        episode: RuntimeEpisode,
        card: Mapping[str, object],
        ledger: BudgetLedger,
        *,
        pass_name: str,
        call_index: int,
        stage: str,
        seed: int,
    ) -> tuple[Mapping[str, float], str]:
        if ledger.remaining_calls < 1:
            raise RuntimeError("no scorer calls remain")
        request = ScoringRequest(
            utterance=episode.utterance,
            candidate_ids=episode.candidate_ids,
            candidate_definitions=dict(episode.candidate_definitions),
            context_card=_project_context(card),
            seed=seed,
            opaque_call_id=sha256_json(
                {"case_id": episode.case_id, "pass": pass_name, "call": call_index}
            ),
        )
        response = scorer.score(request)
        if response.resources.calls != 1:
            raise ValueError("each scorer response must charge one call")
        scores = normalize(response.scores)
        if set(scores) != set(episode.candidate_ids):
            raise ValueError("scorer changed the frozen candidate set")
        ordered = {
            candidate: scores[candidate] for candidate in episode.candidate_ids
        }
        prompt_hash = response.prompt_hash or sha256_json(
            {
                "contract": "CandidateScorer/2",
                "model_id": scorer.model_id,
                "request": asdict(request),
            }
        )
        if not _hash_ok(prompt_hash):
            raise ValueError("prompt hash must be a SHA-256 value")
        ledger.charge(stage, response.resources)
        return ordered, prompt_hash

    def _pass(
        self,
        episode: RuntimeEpisode,
        card: Mapping[str, object],
        branches: Mapping[str, CandidateBranch],
        questions: Mapping[str, tuple[QuestionContract, ...]],
        scorer: CandidateScorer,
        ledger: BudgetLedger,
        *,
        pass_name: str,
        base_stage: str,
        probe_stage: str,
        missing_slots: tuple[str, ...],
        questions_remaining: int,
        seed: int,
    ) -> VerificationPass:
        errors = _branch_errors(
            card,
            episode.candidate_ids,
            branches,
            sealed=self.run_mode == "SEALED",
        )
        if errors:
            raise ValueError(f"invalid pass-specific branch bundle: {errors}")
        if ledger.remaining_calls < 3:
            raise RuntimeError("an atomic verification pass requires three calls")
        base_scores, base_hash = self._score(
            scorer,
            episode,
            card,
            ledger,
            pass_name=pass_name,
            call_index=0,
            stage=base_stage,
            seed=seed,
        )
        selected = _top(base_scores, episode.candidate_ids)
        if selected == OTHER_UNLISTED:
            features = _fallback_features(
                base_scores,
                episode.candidate_ids,
                card,
                episode.attestation.required_context_slots,
            )
            decision = self.policy.decide(
                features,
                GateState(
                    missing_discriminating_slots=missing_slots,
                    questions_remaining=questions_remaining,
                ),
                base_scores,
            )
            return VerificationPass(
                pass_name,
                scorer.model_id,
                base_scores,
                None,
                (),
                {},
                (base_hash,),
                features,
                decision,
            )

        branch = branches[selected]
        preserving_card = apply_context_patch(
            card, branch.preserving.changed_slots, branch.preserving.context_patch
        )
        changing_card = apply_context_patch(
            card,
            branch.meaning_changing.changed_slots,
            branch.meaning_changing.context_patch,
        )
        preserving_scores, preserving_hash = self._score(
            scorer,
            episode,
            preserving_card,
            ledger,
            pass_name=pass_name,
            call_index=1,
            stage=probe_stage,
            seed=seed + 1,
        )
        changing_scores, changing_hash = self._score(
            scorer,
            episode,
            changing_card,
            ledger,
            pass_name=pass_name,
            call_index=2,
            stage=probe_stage,
            seed=seed + 2,
        )
        results = {
            branch.preserving.probe_id: (branch.preserving, preserving_scores),
            branch.meaning_changing.probe_id: (
                branch.meaning_changing,
                changing_scores,
            ),
        }
        features = compute_verification_features(
            base_scores,
            (PreservingProbe(preserving_scores, branch.preserving.validity_weight),),
            (
                ChangingProbe(
                    changing_scores,
                    str(branch.meaning_changing.target_candidate_id),
                    branch.meaning_changing.validity_weight,
                ),
            ),
            context_completeness=_completeness(
                card, episode.attestation.required_context_slots
            ),
            context_conflict=_hard_conflict(card),
        )
        materialized = [
            question.materialize(results)
            for question in questions.get(selected, ())
            if question.context_slot in missing_slots
        ]
        decision = self.policy.decide(
            features,
            GateState(
                missing_discriminating_slots=missing_slots,
                questions_remaining=questions_remaining,
            ),
            base_scores,
            materialized,
        )
        return VerificationPass(
            pass_name,
            scorer.model_id,
            base_scores,
            selected,
            (branch.preserving, branch.meaning_changing),
            {key: scores for key, (_, scores) in results.items()},
            (base_hash, preserving_hash, changing_hash),
            features,
            decision,
        )

    def _release(
        self,
        episode: RuntimeEpisode,
        question: QuestionContract,
        answer: ReleasedAnswer,
        card: Mapping[str, object],
    ) -> tuple[dict, tuple[str, ...]]:
        errors: list[str] = []
        if answer.case_id != episode.case_id:
            errors.append("answer belongs to a different case")
        if answer.question_id != question.question_id:
            errors.append("answer belongs to a different question")
        if answer.context_slot != question.context_slot:
            errors.append("answer targets a different slot")
        if answer.question_manifest_hash != question.manifest_hash:
            errors.append("question manifest hash mismatch")
        if answer.answer_id not in {item.answer_id for item in question.scenarios}:
            errors.append("answer is outside the approved domain")
        if answer.answer_manifest_hash != answer.computed_manifest_hash():
            errors.append("answer manifest hash mismatch")
        if answer.post_answer_branch_manifest_hash != branch_manifest_hash(
            answer.post_answer_branches
        ):
            errors.append("post-answer branch manifest hash mismatch")
        if self.run_mode == "SEALED" and (
            answer.validation_status != AttestationStatus.ACCEPTED
            or answer.safety_status != AttestationStatus.ACCEPTED
        ):
            errors.append("sealed answer is not accepted")
        if (
            self.run_mode == "SEALED"
            and answer.answer_manifest_hash not in self.trusted_answer_manifest_hashes
        ):
            errors.append("answer is absent from the frozen trusted manifest")
        if errors:
            return dict(card), tuple(errors)
        try:
            updated = answer.apply(card)
        except ValueError as exc:
            return dict(card), (f"answer patch rejected: {exc}",)
        errors.extend(
            _branch_errors(
                updated,
                episode.candidate_ids,
                answer.post_answer_branches,
                sealed=self.run_mode == "SEALED",
            )
        )
        return updated, tuple(errors)

    def run(
        self,
        episode: RuntimeEpisode,
        scorer: CandidateScorer,
        *,
        answer_broker: ClarificationAnswerBroker | None = None,
        large_scorer: CandidateScorer | None = None,
        routing: RoutingOption | None = None,
        seed: int = 0,
    ) -> RunResult:
        ledger = BudgetLedger(self.budget_envelope)
        reason, contract_errors = self._preflight(episode)
        if reason is not None:
            decision = GateDecision(
                Action.ABSTAIN_ESCALATE, 0.0, 0.0, reason_code=reason
            )
            return RunResult(
                episode,
                decision,
                decision,
                (),
                (),
                self.budget_envelope,
                contract_errors=contract_errors,
            )

        card = dict(episode.context_card)
        branches = episode.branches
        missing_slots = _missing(
            card, episode.attestation.required_context_slots
        )
        initial = self._pass(
            episode,
            card,
            branches,
            episode.questions_by_candidate,
            scorer,
            ledger,
            pass_name="initial",
            base_stage="base",
            probe_stage="probes",
            missing_slots=missing_slots,
            questions_remaining=1,
            seed=seed,
        )
        passes = [initial]
        final = initial.decision
        selected_question = None
        answer_applied = False
        applied_answer_id = None
        applied_answer_hash = None

        if initial.decision.action == Action.CLARIFY:
            selected_question = next(
                (
                    question
                    for question in episode.questions_by_candidate.get(
                        str(initial.selected_branch), ()
                    )
                    if question.question_id == initial.decision.question_id
                ),
                None,
            )
            ledger.charge("question_selection", ResourceUse())
            released = (
                answer_broker.release(episode.case_id, selected_question)
                if answer_broker is not None and selected_question is not None
                else None
            )
            if selected_question is None or released is None:
                final = GateDecision(
                    Action.ABSTAIN_ESCALATE,
                    initial.decision.estimated_safe_commit_probability,
                    initial.decision.raw_reliability,
                    reason_code="CLARIFICATION_UNRESOLVED",
                )
            else:
                updated, release_errors = self._release(
                    episode, selected_question, released, card
                )
                if release_errors:
                    contract_errors += release_errors
                    final = GateDecision(
                        Action.ABSTAIN_ESCALATE,
                        initial.decision.estimated_safe_commit_probability,
                        initial.decision.raw_reliability,
                        reason_code="CLARIFICATION_RELEASE_REJECTED",
                    )
                else:
                    card = updated
                    branches = released.post_answer_branches
                    answer_applied = True
                    applied_answer_id = released.answer_id
                    applied_answer_hash = released.answer_manifest_hash
                    missing_slots = _missing(
                        card, episode.attestation.required_context_slots
                    )
                    repaired = self._pass(
                        episode,
                        card,
                        branches,
                        {},
                        scorer,
                        ledger,
                        pass_name="post_question",
                        base_stage="post_question",
                        probe_stage="post_question",
                        missing_slots=missing_slots,
                        questions_remaining=0,
                        seed=seed + 100,
                    )
                    passes.append(repaired)
                    final = repaired.decision

        routed = False
        predicted_benefit = None
        context_complete = not _missing(
            card, episode.attestation.required_context_slots
        )
        if (
            final.action == Action.ABSTAIN_ESCALATE
            and large_scorer is not None
            and routing is not None
            and context_complete
            and ledger.remaining_calls >= 3
            and route_if_beneficial(
                context_complete=context_complete,
                context_conflict=_hard_conflict(card),
                privacy_allowed=routing.privacy_allowed,
                expected_small_correctness=routing.expected_small_correctness,
                expected_large_correctness=routing.expected_large_correctness,
                normalized_route_cost=routing.normalized_route_cost,
                cost_weight=routing.cost_weight,
            )
        ):
            routed_pass = self._pass(
                episode,
                card,
                branches,
                {},
                large_scorer,
                ledger,
                pass_name="routing",
                base_stage="routing",
                probe_stage="routing",
                missing_slots=(),
                questions_remaining=0,
                seed=seed + 200,
            )
            passes.append(routed_pass)
            final = routed_pass.decision
            routed = True
            predicted_benefit = (
                routing.expected_large_correctness
                - routing.expected_small_correctness
                - routing.cost_weight * routing.normalized_route_cost
            )

        return RunResult(
            episode,
            initial.decision,
            final,
            tuple(passes),
            tuple(ledger.events),
            self.budget_envelope,
            selected_question=selected_question,
            answer_applied=answer_applied,
            applied_answer_id=applied_answer_id,
            applied_answer_hash=applied_answer_hash,
            routed=routed,
            routing_predicted_benefit=predicted_benefit,
            contract_errors=contract_errors,
        )


def run_equal_budget_structured_context(
    episode: RuntimeEpisode,
    scorer: CandidateScorer,
    *,
    budget_envelope: BudgetEnvelope,
    run_mode: str = "DEMO",
    sealed_execution_authorized: bool = False,
    trusted_attestation_hashes: frozenset[str] | None = None,
    seed: int = 0,
) -> EqualBudgetResult:
    """Repeat the unchanged card under an explicit call and token envelope."""

    if run_mode == "SEALED" and not sealed_execution_authorized:
        raise PermissionError("sealed control requires an authorized process")
    if run_mode == "SEALED" and not trusted_attestation_hashes:
        raise PermissionError(
            "sealed control requires a frozen trusted-attestation manifest"
        )
    errors = _contract_admission_errors(
        episode,
        run_mode=run_mode,
        sealed_execution_authorized=sealed_execution_authorized,
        trusted_attestation_hashes=trusted_attestation_hashes or frozenset(),
    )
    if errors or episode.out_of_domain or _hard_conflict(episode.context_card):
        raise ValueError("episode is ineligible for the equal-budget control")
    ledger = BudgetLedger(budget_envelope)
    responses: list[Mapping[str, float]] = []
    for index in range(budget_envelope.max_calls):
        scores, _ = CDCVRunner._score(
            scorer,
            episode,
            episode.context_card,
            ledger,
            pass_name="equal_budget_context",
            call_index=index,
            stage="base",
            seed=seed + index,
        )
        responses.append(scores)
    mean = {
        candidate: sum(item[candidate] for item in responses) / len(responses)
        for candidate in episode.candidate_ids
    }
    tops = [_top(item, episode.candidate_ids) for item in responses]
    mode = max(
        episode.candidate_ids,
        key=lambda candidate: tops.count(candidate),
    )
    return EqualBudgetResult(
        mean_scores=normalize(mean),
        sample_agreement=tops.count(mode) / len(tops),
        resource_events=tuple(ledger.events),
        budget_envelope=budget_envelope,
    )


def _verification_dict(features: VerificationFeatures) -> dict:
    return {
        "top_two_margin": features.top_two_margin,
        "normalized_entropy": features.normalized_entropy,
        "preservation_invariance": features.preservation_invariance,
        "preservation_probability": features.preservation_probability,
        "mean_preserving_jsd_base2": features.mean_preserving_jsd,
        "targeted_response": features.targeted_response,
        "mean_changing_target_margin": features.mean_changing_target_margin,
        "intervention_validity": features.intervention_validity,
        "context_completeness": features.context_completeness,
        "conflict_severity": "hard_stop" if features.context_conflict else "none",
    }


def build_prediction_record(
    result: RunResult,
    policy: GatePolicy,
    *,
    run_id: str,
    system_id: str,
    code_commit: str,
    timestamp_utc: str | None = None,
) -> dict:
    """Build a complete gold-free audit record for one system decision."""

    episode = result.episode
    initial_pass = result.passes[0] if result.passes else None
    final_pass = result.passes[-1] if result.passes else None
    base_scores = (
        dict(initial_pass.base_scores)
        if initial_pass is not None
        else {
            candidate: 1.0 / len(episode.candidate_ids)
            for candidate in episode.candidate_ids
        }
    )
    initial_features = (
        initial_pass.features
        if initial_pass is not None
        else _fallback_features(
            base_scores,
            episode.candidate_ids,
            episode.context_card,
            episode.attestation.required_context_slots,
        )
    )
    probe_rows: list[dict] = []
    if initial_pass is not None:
        for probe in initial_pass.probes:
            probe_rows.append(
                {
                    "probe_id": probe.probe_id,
                    "probe_type": probe.probe_type,
                    "source_candidate_id": probe.source_candidate_id,
                    "target_candidate_id": probe.target_candidate_id,
                    "scores": dict(initial_pass.probe_scores[probe.probe_id]),
                    "validity_weight": probe.validity_weight,
                }
            )
    pass_trace = [
        {
            "pass_name": item.pass_name,
            "model_id": item.model_id,
            "base_scores": dict(item.base_scores),
            "selected_branch": item.selected_branch,
            "prompt_hashes": list(item.prompt_hashes),
            "probe_scores_hash": sha256_json(item.probe_scores),
            "verification": _verification_dict(item.features),
            "action": item.decision.action.value,
            "reason_code": item.decision.reason_code,
        }
        for item in result.passes
    ]
    consumed = result.consumed
    if (
        consumed.calls > result.budget_envelope.max_calls
        or consumed.input_tokens > result.budget_envelope.max_input_tokens
        or consumed.output_tokens > result.budget_envelope.max_output_tokens
    ):
        raise ValueError("consumed resources exceed the allocated budget")

    selected = result.selected_question
    clarification_trace = None
    if result.initial_decision.action == Action.CLARIFY:
        clarification_trace = {
            "question_id": result.initial_decision.question_id or "UNRESOLVED",
            "context_slot": result.initial_decision.context_slot or "UNRESOLVED",
            "question_utility": float(
                result.initial_decision.question_utility or 0.0
            ),
            "scenario_manifest_hash": (
                selected.manifest_hash
                if selected is not None
                else sha256_json("NO_MANIFEST")
            ),
            "answer_id": result.applied_answer_id,
            "answer_patch_hash": (
                result.applied_answer_hash if result.answer_applied else None
            ),
            "post_question_verification_hash": (
                sha256_json(asdict(result.passes[1].features))
                if result.answer_applied and len(result.passes) > 1
                else None
            ),
        }

    controller_hash = sha256_json(asdict(policy.config))
    calibration_hash = sha256_json(
        {
            "class": type(policy.calibrator).__name__,
            "fitted_state": asdict(policy.calibrator),
        }
    )
    candidate_payload = {
        "order": episode.candidate_ids,
        "definitions": episode.candidate_definitions,
    }
    runtime_payload = {
        "case_id": episode.case_id,
        "family_id": episode.family_id,
        "split": episode.split,
        "utterance_hash": sha256_json(episode.utterance),
        "candidate_set": candidate_payload,
        "context_card": episode.context_card,
        "branches": {
            key: asdict(episode.branches[key])
            for key in sorted(episode.branches)
        },
        "questions": {
            key: [asdict(value) for value in episode.questions_by_candidate[key]]
            for key in sorted(episode.questions_by_candidate)
        },
        "attestation": asdict(episode.attestation),
    }
    prompt_hashes = [
        prompt_hash
        for verification_pass in result.passes
        for prompt_hash in verification_pass.prompt_hashes
    ]
    verification = _verification_dict(initial_features)
    verification.update(
        {
            "estimated_safe_commit_probability": (
                result.initial_decision.estimated_safe_commit_probability
            ),
            "allocated_budget_remaining": max(
                0, result.budget_envelope.max_calls - consumed.calls
            ),
        }
    )
    stage_rows = [
        {"stage": event.stage, "resources": event.resources.to_dict()}
        for event in result.resource_events
    ]
    timestamp = timestamp_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "run_id": run_id,
        "system_id": system_id,
        "case_id": episode.case_id,
        "family_id": episode.family_id,
        "source_model_id": (
            initial_pass.model_id if initial_pass else "NO_SCORER_CALLED"
        ),
        "runtime_manifest_hash": sha256_json(runtime_payload),
        "context_card_hash": sha256_json(episode.context_card),
        "intervention_bundle_hash": branch_manifest_hash(episode.branches),
        "candidate_set_hash": sha256_json(candidate_payload),
        "prompt_hash": sha256_json(prompt_hashes),
        "candidate_order_hash": sha256_json(episode.candidate_ids),
        "controller_hash": controller_hash,
        "calibration_manifest_hash": calibration_hash,
        "code_commit": code_commit,
        "pricing_snapshot_hash": None,
        "base_scores": base_scores,
        "probe_scores": probe_rows,
        "pass_trace": pass_trace,
        "verification": verification,
        "initial_action": result.initial_decision.action.value,
        "final_action": result.final_decision.action.value,
        "committed_sense_id": result.final_decision.sense_id,
        "clarification_trace": clarification_trace,
        "routing_trace": {
            "routed": result.routed,
            "target_model_id": (
                final_pass.model_id if result.routed and final_pass else None
            ),
            "predicted_benefit": result.routing_predicted_benefit,
            "pre_route_action": (
                result.passes[-2].decision.action.value
                if result.routed and len(result.passes) >= 2
                else None
            ),
            "post_route_verification_hash": (
                sha256_json(asdict(result.passes[-1].features))
                if result.routed
                else None
            ),
        },
        "reason_code": result.final_decision.reason_code,
        "resource_use": {
            "allocated": result.budget_envelope.as_allocated_resource(
                consumed
            ).to_dict(),
            "consumed": consumed.to_dict(),
            "stages": stage_rows,
        },
        "timestamp_utc": timestamp,
    }
