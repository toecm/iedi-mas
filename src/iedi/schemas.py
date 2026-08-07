from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Route(str, Enum):
    LOCAL = "local"
    FLASH = "flash"
    PRO = "pro"
    HUMAN_REVIEW = "human_review"


class FeedbackAction(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    SUGGEST = "suggest_update"


class FeedbackState(str, Enum):
    PROPOSED = "proposed"
    USER_VALIDATED = "user_validated"
    IPFS_PINNED = "ipfs_pinned"
    CHAIN_CONFIRMED = "chain_confirmed"
    APPROVED = "approved"
    MERGED = "merged"
    MERGED_AND_INDEXED = "merged_and_indexed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class AcousticAffect:
    """Versioned acoustic evidence produced locally, never an implicit claim of accuracy."""

    label: str | None = None
    confidence: float | None = None
    extractor_id: str | None = None
    extractor_version: str | None = None
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("acoustic confidence must be in [0, 1]")


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_s: float
    end_s: float
    speaker_id: str | None
    asr_confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("segment text is required")
        if self.start_s < 0 or self.end_s < self.start_s:
            raise ValueError("invalid segment timestamps")
        if self.asr_confidence is not None and not 0.0 <= self.asr_confidence <= 1.0:
            raise ValueError("ASR confidence must be in [0, 1]")


@dataclass(frozen=True)
class CodebookEntry:
    entry_id: str
    concept_id: str
    text: str
    dialect: str
    universal_gloss: str
    intent: str
    sociolinguistic_tags: tuple[str, ...]
    tone_categories: tuple[str, ...] = ()
    linguistic_contexts: tuple[str, ...] = ()
    pragmatic_analysis: str = ""
    surface_forms: tuple[str, ...] = ()
    syntax_patterns: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    counterexamples: tuple[str, ...] = ()
    speaker_roles: tuple[str, ...] = ()
    persona_ids: tuple[str, ...] = ()
    priority: int = 0
    source_type: str = "human"
    source_reference: str = ""
    reviewed_by: tuple[str, ...] = ()
    review_status: str = "pending"
    version: int = 1
    supersedes_entry_id: str | None = None
    audio_uri: str | None = None
    audio_sha256: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        required = {
            "entry_id": self.entry_id,
            "concept_id": self.concept_id,
            "text": self.text,
            "dialect": self.dialect,
            "universal_gloss": self.universal_gloss,
            "intent": self.intent,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"missing required codebook fields: {', '.join(missing)}")
        if not self.sociolinguistic_tags:
            raise ValueError("sociolinguistic_tags must contain at least one tag")
        if self.version < 1:
            raise ValueError("version must be positive")
        if self.review_status not in {"approved", "pending", "rejected", "superseded"}:
            raise ValueError(f"unsupported review_status: {self.review_status}")
        if self.review_status == "approved" and not self.reviewed_by:
            raise ValueError("approved codebook entries must record at least one reviewer")

    @property
    def all_surface_forms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.text, *self.surface_forms, *self.examples)))


@dataclass(frozen=True)
class PragmaticRule:
    rule_id: str
    trigger: str
    preferred_entry_id: str
    interpretation: str
    tone: str | None = None
    speaker_role: str | None = None
    context_condition: str | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        if not all((self.rule_id.strip(), self.trigger.strip(), self.preferred_entry_id.strip())):
            raise ValueError("rule_id, trigger and preferred_entry_id are required")


@dataclass(frozen=True)
class MatchEvidence:
    entry: CodebookEntry
    score: float
    method: str
    persona_priority: bool = False
    rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("match score must be in [0, 1]")


@dataclass(frozen=True)
class InterpretationRequest:
    utterance: str
    active_persona_ids: tuple[str, ...] = ()
    conversation_context: tuple[str, ...] = ()
    speaker_id: str | None = None
    speaker_role: str | None = None
    supplied_tone: str | None = None
    supplied_context: str | None = None
    acoustic_affect: AcousticAffect | None = None
    asr_confidence: float | None = None
    network_available: bool = True
    latency_budget_ms: float | None = None
    cost_budget_usd: float | None = None
    risk_score: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not self.utterance.strip():
            raise ValueError("utterance is required")
        if len(self.utterance) > 4096:
            raise ValueError("utterance exceeds the 4096-character limit")
        if len(self.conversation_context) > 20:
            raise ValueError("conversation context exceeds 20 turns")
        if any(len(turn) > 4096 for turn in self.conversation_context):
            raise ValueError("conversation context turn exceeds 4096 characters")
        if len(self.active_persona_ids) > 16:
            raise ValueError("at most 16 active personas are allowed")
        if self.asr_confidence is not None and not 0.0 <= self.asr_confidence <= 1.0:
            raise ValueError("ASR confidence must be in [0, 1]")
        if self.latency_budget_ms is not None and self.latency_budget_ms <= 0:
            raise ValueError("latency budget must be positive")
        if self.cost_budget_usd is not None and self.cost_budget_usd < 0:
            raise ValueError("cost budget cannot be negative")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")


@dataclass(frozen=True)
class RoutingFeatures:
    top_score: float
    second_score: float
    score_margin: float
    plausible_senses: int
    persona_conflict: bool
    context_complete: bool
    asr_confidence: float | None
    polysemous_surface: bool
    risk_score: float = 0.0


@dataclass(frozen=True)
class RouteDecision:
    requested_route: Route
    used_route: Route
    ambiguity_score: float
    reasons: tuple[str, ...]
    policy_version: str
    model_id: str | None = None
    cold_start: bool = False
    fallback_from: Route | None = None
    fallback_reason: str | None = None
    degraded: bool = False


@dataclass(frozen=True)
class CandidateInterpretation:
    candidate_id: str
    entry_id: str | None
    dialect: str
    clarification: str
    intent: str
    tone_category: str
    linguistic_context: str
    pragmatic_analysis: str
    sociolinguistic_tags: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = (
            self.candidate_id,
            self.dialect,
            self.clarification,
            self.intent,
            self.tone_category,
            self.linguistic_context,
            self.pragmatic_analysis,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("candidate interpretation has missing required fields")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be in [0, 1]")


@dataclass(frozen=True)
class ModelCallRecord:
    model_id: str
    route: Route
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    response_id: str | None = None
    status: str = "success"
    error_reason: str | None = None


@dataclass(frozen=True)
class TimingTrace:
    edge_asr_ms: float = 0.0
    edge_retrieval_ms: float = 0.0
    serialization_ms: float = 0.0
    estimated_uplink_transfer_ms: float = 0.0
    observed_api_round_trip_ms: float = 0.0
    response_parse_ms: float = 0.0
    end_to_end_ms: float = 0.0


@dataclass(frozen=True)
class InterpretationResult:
    request_id: str
    candidates: tuple[CandidateInterpretation, ...]
    decision: RouteDecision
    retrieved_entry_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    profile_hashes: tuple[str, ...]
    model_calls: tuple[ModelCallRecord, ...] = ()
    payload_bytes: int = 0
    raw_audio_bytes: int | None = None
    timing: TimingTrace = field(default_factory=TimingTrace)
    needs_human_review: bool = False
    dataset_version: str = "unversioned"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision"]["requested_route"] = self.decision.requested_route.value
        value["decision"]["used_route"] = self.decision.used_route.value
        if self.decision.fallback_from is not None:
            value["decision"]["fallback_from"] = self.decision.fallback_from.value
        for call in value["model_calls"]:
            call["route"] = call["route"].value if isinstance(call["route"], Route) else call["route"]
        return value


@dataclass(frozen=True)
class FeedbackEvent:
    request_id: str
    action: FeedbackAction
    actor_id: str
    candidate: CandidateInterpretation | None = None
    corrected_entry: CodebookEntry | None = None
    source_result_sha256: str | None = None
    signature: str | None = None
    state: FeedbackState = FeedbackState.PROPOSED
    event_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)


def require_string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array of strings")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if len(result) != len(value):
        raise ValueError(f"{field_name} cannot contain empty values")
    return result
