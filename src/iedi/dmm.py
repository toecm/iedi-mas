from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, replace
from time import monotonic, perf_counter
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .codebook import Codebook
from .providers import (
    InvalidProviderResponse,
    ModelProvider,
    ProviderAuthenticationError,
    ProviderError,
    ProviderQuotaError,
    ProviderResponse,
)
from .schemas import (
    CandidateInterpretation,
    InterpretationRequest,
    MatchEvidence,
    ModelCallRecord,
    Route,
    RouteDecision,
    RoutingFeatures,
)


MODEL_OUTPUT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "entry_id": {"type": ["string", "null"]},
                    "dialect": {"type": "string"},
                    "clarification": {"type": "string"},
                    "intent": {"type": "string"},
                    "tone_category": {"type": "string"},
                    "linguistic_context": {"type": "string"},
                    "pragmatic_analysis": {"type": "string"},
                    "sociolinguistic_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "entry_id",
                    "dialect",
                    "clarification",
                    "intent",
                    "tone_category",
                    "linguistic_context",
                    "pragmatic_analysis",
                    "sociolinguistic_tags",
                    "confidence",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class DMMPolicy:
    version: str = "dmm-2.0"
    local_threshold: float = 0.80
    local_margin: float = 0.05
    plausible_threshold: float = 0.72
    pro_ambiguity_threshold: float = 0.58
    pro_risk_threshold: float = 0.70
    pro_min_latency_budget_ms: float = 900.0
    cold_start_requests: int = 0
    low_model_confidence: float = 0.55
    flash_model_id: str = "gemini-2.5-flash"
    pro_model_id: str = "gemini-2.5-pro"
    temperature: float = 0.1
    timeout_s: float = 30.0
    max_candidates: int = 3
    allow_quality_escalation: bool = True
    retryable_pro_fallback: bool = True
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: float = 30.0
    estimated_flash_call_cost_usd: float | None = None
    estimated_pro_call_cost_usd: float | None = None
    input_cost_per_million: Mapping[str, float] | None = None
    output_cost_per_million: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        for name in (
            "local_threshold",
            "local_margin",
            "plausible_threshold",
            "pro_ambiguity_threshold",
            "pro_risk_threshold",
            "low_model_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.cold_start_requests < 0:
            raise ValueError("cold_start_requests cannot be negative")
        if self.max_candidates not in {1, 2, 3}:
            raise ValueError("max_candidates must be 1, 2 or 3")
        for name in ("estimated_flash_call_cost_usd", "estimated_pro_call_cost_usd"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")


class DMMExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DMMOutcome:
    candidates: tuple[CandidateInterpretation, ...]
    decision: RouteDecision
    calls: tuple[ModelCallRecord, ...]
    needs_human_review: bool = False
    response_parse_ms: float = 0.0


class AmbiguityEstimator:
    """Transparent feature-based ambiguity score; no model chooses its own lane."""

    def score(self, features: RoutingFeatures) -> tuple[float, tuple[str, ...]]:
        reasons: list[str] = []
        uncertainty = 1.0 - features.top_score
        margin_uncertainty = 1.0 - min(features.score_margin / 0.20, 1.0)
        polysemy = min(max(features.plausible_senses - 1, 0) / 2.0, 1.0)

        if features.top_score < 0.80:
            reasons.append("below_retrieval_threshold")
        if features.score_margin < 0.05 and features.plausible_senses > 1:
            reasons.append("low_match_margin")
        if features.plausible_senses > 1 or features.polysemous_surface:
            reasons.append("multiple_codebook_senses")
        if features.persona_conflict:
            reasons.append("persona_conflict")
        if not features.context_complete and features.polysemous_surface:
            reasons.append("missing_disambiguating_context")
        if features.asr_confidence is not None and features.asr_confidence < 0.75:
            reasons.append("low_asr_confidence")

        score = (
            0.30 * uncertainty
            + 0.25 * margin_uncertainty
            + 0.20 * polysemy
            + 0.10 * float(features.persona_conflict)
            + 0.10 * float(not features.context_complete and features.polysemous_surface)
            + 0.05 * (1.0 - (features.asr_confidence if features.asr_confidence is not None else 1.0))
        )
        if features.polysemous_surface and features.plausible_senses > 1:
            score += 0.15
        score = max(score, features.risk_score * 0.80)
        return min(max(score, 0.0), 1.0), tuple(dict.fromkeys(reasons))


class DynamicModelManager:
    def __init__(
        self,
        provider: ModelProvider,
        policy: DMMPolicy | None = None,
        *,
        estimator: AmbiguityEstimator | None = None,
    ) -> None:
        self.provider = provider
        self.policy = policy or DMMPolicy()
        self.estimator = estimator or AmbiguityEstimator()
        self._lock = threading.Lock()
        self._request_counts: dict[str, int] = {}
        self._pro_failures = 0
        self._pro_opened_at: float | None = None

    def interpret(
        self,
        *,
        request: InterpretationRequest,
        evidence: tuple[MatchEvidence, ...],
        features: RoutingFeatures,
        codebook: Codebook,
        task: str = "interpret",
        require_three: bool = False,
    ) -> DMMOutcome:
        ambiguity, ambiguity_reasons = self.estimator.score(features)
        decision = self._decide(
            request=request,
            features=features,
            ambiguity=ambiguity,
            ambiguity_reasons=ambiguity_reasons,
            task=task,
        )

        if decision.used_route is Route.LOCAL:
            # A LOCAL decision is possible only for one authoritative sense.  Do not
            # leak lower-ranked fuzzy matches into the answer merely because they
            # were useful while estimating ambiguity.
            candidates = _local_candidates(evidence[:1], 1)
            return DMMOutcome(candidates, decision, ())
        if decision.used_route is Route.HUMAN_REVIEW:
            return DMMOutcome((), decision, (), needs_human_review=True)

        prompt = self._build_prompt(
            request=request,
            evidence=evidence,
            codebook=codebook,
            ambiguity=ambiguity,
            require_three=require_three,
        )
        calls: list[ModelCallRecord] = []
        response_parse_ms = 0.0

        try:
            response = self._call(decision.used_route, prompt)
            calls.append(self._record_call(decision.used_route, response))
            self._record_pro_success(decision.used_route)
            parse_started = perf_counter()
            candidates = _parse_candidates(
                response.data,
                require_three=require_three,
                max_candidates=self.policy.max_candidates,
                allowed_entry_ids={item.entry.entry_id for item in evidence},
                codebook=codebook,
            )
            response_parse_ms += (perf_counter() - parse_started) * 1000.0
            low_confidence_flash = (
                decision.used_route is Route.FLASH
                and self.policy.allow_quality_escalation
                and candidates
                and max(candidate.confidence for candidate in candidates)
                < self.policy.low_model_confidence
            )
            escalation_has_budget = self._quality_escalation_fits_budget(request, calls)
            pro_available = self._pro_is_available()
            if low_confidence_flash and pro_available and escalation_has_budget:
                try:
                    pro_response = self._call(Route.PRO, prompt)
                    calls.append(self._record_call(Route.PRO, pro_response))
                    self._record_pro_success(Route.PRO)
                    parse_started = perf_counter()
                    candidates = _parse_candidates(
                        pro_response.data,
                        require_three=require_three,
                        max_candidates=self.policy.max_candidates,
                        allowed_entry_ids={item.entry.entry_id for item in evidence},
                        codebook=codebook,
                    )
                    response_parse_ms += (perf_counter() - parse_started) * 1000.0
                    decision = replace(
                        decision,
                        requested_route=Route.FLASH,
                        used_route=Route.PRO,
                        model_id=self.policy.pro_model_id,
                        reasons=(*decision.reasons, "flash_low_confidence_escalation"),
                        fallback_from=Route.FLASH,
                        fallback_reason="low_confidence",
                    )
                except ProviderAuthenticationError as exc:
                    calls.append(self._failed_call(Route.PRO, exc.reason, exc))
                    raise DMMExecutionError(
                        "Pro escalation authentication failed; refusing to hide it behind Flash"
                    ) from exc
                except InvalidProviderResponse as exc:
                    calls.append(self._failed_call(Route.PRO, exc.reason, exc))
                    raise DMMExecutionError(
                        "Pro escalation response failed schema validation; refusing silent substitution"
                    ) from exc
                except ProviderError as exc:
                    self._record_pro_failure(Route.PRO)
                    calls.append(self._failed_call(Route.PRO, exc.reason, exc))
                    decision = replace(
                        decision,
                        requested_route=Route.FLASH,
                        used_route=Route.FLASH,
                        model_id=self.policy.flash_model_id,
                        reasons=(*decision.reasons, "quality_escalation_failed"),
                        fallback_from=Route.PRO,
                        fallback_reason=exc.reason,
                        degraded=True,
                    )
                    return DMMOutcome(
                        candidates,
                        decision,
                        tuple(calls),
                        needs_human_review=True,
                        response_parse_ms=response_parse_ms,
                    )
            elif low_confidence_flash:
                reason = (
                    "quality_escalation_budget_blocked"
                    if not escalation_has_budget
                    else "quality_escalation_circuit_blocked"
                )
                decision = replace(
                    decision,
                    degraded=True,
                    reasons=(*decision.reasons, reason),
                )
                return DMMOutcome(
                    candidates,
                    decision,
                    tuple(calls),
                    needs_human_review=True,
                    response_parse_ms=response_parse_ms,
                )
            if (
                candidates
                and max(candidate.confidence for candidate in candidates)
                < self.policy.low_model_confidence
            ):
                decision = replace(
                    decision,
                    degraded=True,
                    reasons=(*decision.reasons, "model_low_confidence_requires_review"),
                )
                return DMMOutcome(
                    candidates,
                    decision,
                    tuple(calls),
                    needs_human_review=True,
                    response_parse_ms=response_parse_ms,
                )
            return DMMOutcome(
                candidates,
                decision,
                tuple(calls),
                response_parse_ms=response_parse_ms,
            )
        except ProviderQuotaError as exc:
            calls.append(self._failed_call(decision.used_route, exc.reason, exc))
            self._record_pro_failure(decision.used_route, force_open=True)
            if decision.used_route is Route.PRO and self.policy.retryable_pro_fallback:
                return self._fallback_to_flash(
                    prompt=prompt,
                    decision=decision,
                    calls=calls,
                    reason=exc.reason,
                    require_three=require_three,
                    allowed_entry_ids={item.entry.entry_id for item in evidence},
                    codebook=codebook,
                )
            return self._failed_outcome(decision, calls, exc.reason)
        except ProviderAuthenticationError as exc:
            raise DMMExecutionError(
                "model authentication failed; refusing silent lane substitution"
            ) from exc
        except InvalidProviderResponse as exc:
            raise DMMExecutionError(
                "model response failed schema validation; refusing silent lane substitution"
            ) from exc
        except ProviderError as exc:
            calls.append(self._failed_call(decision.used_route, exc.reason, exc))
            self._record_pro_failure(decision.used_route)
            if (
                decision.used_route is Route.PRO
                and exc.retryable
                and self.policy.retryable_pro_fallback
            ):
                return self._fallback_to_flash(
                    prompt=prompt,
                    decision=decision,
                    calls=calls,
                    reason=exc.reason,
                    require_three=require_three,
                    allowed_entry_ids={item.entry.entry_id for item in evidence},
                    codebook=codebook,
                )
            return self._failed_outcome(decision, calls, exc.reason)

    def _decide(
        self,
        *,
        request: InterpretationRequest,
        features: RoutingFeatures,
        ambiguity: float,
        ambiguity_reasons: tuple[str, ...],
        task: str,
    ) -> RouteDecision:
        unambiguous_local = (
            features.top_score >= self.policy.local_threshold
            and features.score_margin >= self.policy.local_margin
            and features.plausible_senses == 1
            and not features.persona_conflict
            and not features.polysemous_surface
            and features.risk_score < self.policy.pro_risk_threshold
        )
        if task != "initial_profile_setup" and unambiguous_local:
            return RouteDecision(
                requested_route=Route.LOCAL,
                used_route=Route.LOCAL,
                ambiguity_score=ambiguity,
                reasons=("authoritative_unique_codebook_match",),
                policy_version=self.policy.version,
            )

        if not request.network_available:
            return RouteDecision(
                requested_route=Route.HUMAN_REVIEW,
                used_route=Route.HUMAN_REVIEW,
                ambiguity_score=ambiguity,
                reasons=(*ambiguity_reasons, "network_unavailable"),
                policy_version=self.policy.version,
                degraded=True,
            )

        scope = "|".join(request.active_persona_ids) or "global"
        request_number = self._reserve_request(scope)
        cold_start = request_number <= self.policy.cold_start_requests
        pro_budget_allowed = (
            request.latency_budget_ms is None
            or request.latency_budget_ms >= self.policy.pro_min_latency_budget_ms
        )
        pro_cost_allowed = (
            request.cost_budget_usd is None
            or (
                self.policy.estimated_pro_call_cost_usd is not None
                and self.policy.estimated_pro_call_cost_usd <= request.cost_budget_usd
            )
        )
        wants_pro = (
            task == "initial_profile_setup"
            or cold_start
            or ambiguity >= self.policy.pro_ambiguity_threshold
            or features.risk_score >= self.policy.pro_risk_threshold
        )

        reasons = list(ambiguity_reasons)
        if task == "initial_profile_setup":
            reasons.append("initial_profile_setup")
        if cold_start:
            reasons.append("persona_cold_start")
        if features.risk_score >= self.policy.pro_risk_threshold:
            reasons.append("high_risk")
        if not pro_budget_allowed:
            reasons.append("latency_budget_requires_flash")
        if not pro_cost_allowed:
            reasons.append("cost_budget_requires_flash")

        if wants_pro and pro_budget_allowed and pro_cost_allowed and self._pro_is_available():
            route = Route.PRO
            model_id = self.policy.pro_model_id
        else:
            route = Route.FLASH
            model_id = self.policy.flash_model_id
            if wants_pro and not self._pro_is_available():
                reasons.append("pro_circuit_open")

        if not reasons:
            reasons.append("routine_low_ambiguity")
        return RouteDecision(
            requested_route=route,
            used_route=route,
            ambiguity_score=ambiguity,
            reasons=tuple(dict.fromkeys(reasons)),
            policy_version=self.policy.version,
            model_id=model_id,
            cold_start=cold_start,
        )

    def _build_prompt(
        self,
        *,
        request: InterpretationRequest,
        evidence: Iterable[MatchEvidence],
        codebook: Codebook,
        ambiguity: float,
        require_three: bool,
    ) -> str:
        evidence_payload = [
            {
                "entry_id": item.entry.entry_id,
                "dialect": item.entry.dialect,
                "universal_gloss": item.entry.universal_gloss,
                "intent": item.entry.intent,
                "tone_categories": item.entry.tone_categories,
                "linguistic_contexts": item.entry.linguistic_contexts,
                "pragmatic_analysis": item.entry.pragmatic_analysis,
                "sociolinguistic_tags": item.entry.sociolinguistic_tags,
                "score": item.score,
                "method": item.method,
            }
            for item in evidence
        ]
        payload = {
            "instruction": (
                "Interpret the intra-English dialect utterance using only culturally reviewed "
                "codebook evidence when available. Preserve uncertainty and do not invent cultural facts."
            ),
            "candidate_count": 3 if require_three else f"1-{self.policy.max_candidates}",
            "utterance": request.utterance,
            "speaker_id": request.speaker_id,
            "speaker_role": request.speaker_role,
            "supplied_tone": request.supplied_tone,
            "supplied_context": request.supplied_context,
            "conversation_context": request.conversation_context[-5:],
            "acoustic_affect": asdict(request.acoustic_affect)
            if request.acoustic_affect is not None
            else None,
            "ambiguity_score": ambiguity,
            "retrieved_evidence": evidence_payload,
            "full_persona_context": json.loads(
                codebook.render_persona_context(request.active_persona_ids)
            ),
            "output_rule": (
                "Return distinct alternatives when ambiguity remains. Confidence is epistemic, not style."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _call(self, route: Route, prompt: str) -> ProviderResponse:
        model_id = self.policy.pro_model_id if route is Route.PRO else self.policy.flash_model_id
        return self.provider.generate(
            model_id=model_id,
            prompt=prompt,
            response_schema=MODEL_OUTPUT_SCHEMA,
            temperature=self.policy.temperature,
            timeout_s=self.policy.timeout_s,
        )

    def _record_call(self, route: Route, response: ProviderResponse) -> ModelCallRecord:
        model_id = self.policy.pro_model_id if route is Route.PRO else self.policy.flash_model_id
        return ModelCallRecord(
            model_id=response.model_version or model_id,
            route=route,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            estimated_cost_usd=self._estimate_cost(model_id, response),
            response_id=response.response_id,
        )

    def _failed_call(
        self, route: Route, reason: str, error: Exception | None = None
    ) -> ModelCallRecord:
        model_id = self.policy.pro_model_id if route is Route.PRO else self.policy.flash_model_id
        return ModelCallRecord(
            model_id=model_id,
            route=route,
            latency_ms=float(getattr(error, "latency_ms", 0.0)),
            status="failed",
            error_reason=reason,
        )

    def _estimate_cost(self, model_id: str, response: ProviderResponse) -> float | None:
        input_rates = self.policy.input_cost_per_million or {}
        output_rates = self.policy.output_cost_per_million or {}
        if response.input_tokens is None or response.output_tokens is None:
            return None
        if model_id not in input_rates or model_id not in output_rates:
            return None
        return (
            response.input_tokens * input_rates[model_id]
            + response.output_tokens * output_rates[model_id]
        ) / 1_000_000.0

    def _fallback_to_flash(
        self,
        *,
        prompt: str,
        decision: RouteDecision,
        calls: list[ModelCallRecord],
        reason: str,
        require_three: bool,
        allowed_entry_ids: set[str],
        codebook: Codebook,
    ) -> DMMOutcome:
        try:
            response = self._call(Route.FLASH, prompt)
            calls.append(self._record_call(Route.FLASH, response))
            parse_started = perf_counter()
            candidates = _parse_candidates(
                response.data,
                require_three=require_three,
                max_candidates=self.policy.max_candidates,
                allowed_entry_ids=allowed_entry_ids,
                codebook=codebook,
            )
            response_parse_ms = (perf_counter() - parse_started) * 1000.0
        except ProviderError as exc:
            calls.append(self._failed_call(Route.FLASH, exc.reason, exc))
            return self._failed_outcome(decision, calls, f"{reason};flash_{exc.reason}")
        fallback_decision = replace(
            decision,
            used_route=Route.FLASH,
            model_id=self.policy.flash_model_id,
            fallback_from=Route.PRO,
            fallback_reason=reason,
            degraded=True,
            reasons=(*decision.reasons, "pro_to_flash_fallback"),
        )
        return DMMOutcome(
            candidates,
            fallback_decision,
            tuple(calls),
            response_parse_ms=response_parse_ms,
        )

    def _failed_outcome(
        self,
        decision: RouteDecision,
        calls: Iterable[ModelCallRecord],
        reason: str,
    ) -> DMMOutcome:
        failed = replace(
            decision,
            used_route=Route.HUMAN_REVIEW,
            model_id=None,
            fallback_reason=reason,
            degraded=True,
            reasons=(*decision.reasons, "model_lane_failed"),
        )
        return DMMOutcome((), failed, tuple(calls), needs_human_review=True)

    def _reserve_request(self, scope: str) -> int:
        with self._lock:
            value = self._request_counts.get(scope, 0) + 1
            self._request_counts[scope] = value
            return value

    def request_count(self, scope: str = "global") -> int:
        with self._lock:
            return self._request_counts.get(scope, 0)

    def _pro_is_available(self) -> bool:
        with self._lock:
            if self._pro_opened_at is None:
                return True
            if monotonic() - self._pro_opened_at >= self.policy.circuit_breaker_cooldown_s:
                self._pro_opened_at = None
                self._pro_failures = 0
                return True
            return False

    def _record_pro_failure(self, route: Route, *, force_open: bool = False) -> None:
        if route is not Route.PRO:
            return
        with self._lock:
            self._pro_failures += 1
            if force_open or self._pro_failures >= self.policy.circuit_breaker_failures:
                self._pro_opened_at = monotonic()

    def _record_pro_success(self, route: Route) -> None:
        """A successful Pro call closes the consecutive-failure circuit."""

        if route is not Route.PRO:
            return
        with self._lock:
            self._pro_failures = 0
            self._pro_opened_at = None

    def _quality_escalation_fits_budget(
        self,
        request: InterpretationRequest,
        calls: Iterable[ModelCallRecord],
    ) -> bool:
        """Apply the request's remaining latency and total-cost budgets to Flash→Pro.

        Explicit budgets are fail-closed when a required cost estimate is missing;
        otherwise the DMM could claim to respect a budget it cannot calculate.
        """

        calls = tuple(calls)
        if request.latency_budget_ms is not None:
            elapsed_ms = sum(max(call.latency_ms, 0.0) for call in calls)
            if request.latency_budget_ms - elapsed_ms < self.policy.pro_min_latency_budget_ms:
                return False

        if request.cost_budget_usd is not None:
            spent = 0.0
            for call in calls:
                if call.estimated_cost_usd is not None:
                    spent += call.estimated_cost_usd
                elif call.route is Route.FLASH:
                    if self.policy.estimated_flash_call_cost_usd is None:
                        return False
                    spent += self.policy.estimated_flash_call_cost_usd
            if self.policy.estimated_pro_call_cost_usd is None:
                return False
            if spent + self.policy.estimated_pro_call_cost_usd > request.cost_budget_usd:
                return False
        return True


def _local_candidates(
    evidence: Iterable[MatchEvidence], max_candidates: int
) -> tuple[CandidateInterpretation, ...]:
    candidates: list[CandidateInterpretation] = []
    for item in evidence:
        entry = item.entry
        candidates.append(
            CandidateInterpretation(
                candidate_id=str(uuid4()),
                entry_id=entry.entry_id,
                dialect=entry.dialect,
                clarification=entry.universal_gloss,
                intent=entry.intent,
                tone_category=entry.tone_categories[0] if entry.tone_categories else "unspecified",
                linguistic_context=(
                    entry.linguistic_contexts[0]
                    if entry.linguistic_contexts
                    else "codebook-defined context"
                ),
                pragmatic_analysis=entry.pragmatic_analysis or entry.universal_gloss,
                sociolinguistic_tags=entry.sociolinguistic_tags,
                confidence=item.score,
                evidence=(entry.entry_id, item.method, *item.rule_ids),
            )
        )
        if len(candidates) >= max_candidates:
            break
    return tuple(candidates)


def _parse_candidates(
    payload: Mapping[str, Any],
    *,
    require_three: bool,
    max_candidates: int = 3,
    allowed_entry_ids: set[str] | None = None,
    codebook: Codebook | None = None,
) -> tuple[CandidateInterpretation, ...]:
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise InvalidProviderResponse("candidates must be a non-empty array")
    if len(raw_candidates) > max_candidates:
        raise InvalidProviderResponse(f"at most {max_candidates} candidates are allowed")
    if require_three and len(raw_candidates) != 3:
        raise InvalidProviderResponse("polysemous route requires exactly three candidates")

    candidates: list[CandidateInterpretation] = []
    seen_meanings: set[str] = set()
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise InvalidProviderResponse("candidate must be an object")
        try:
            meaning_key = str(raw["clarification"]).strip().casefold()
            if meaning_key in seen_meanings:
                raise InvalidProviderResponse("candidate meanings must be distinct")
            seen_meanings.add(meaning_key)
            tags = raw["sociolinguistic_tags"]
            evidence = raw["evidence"]
            if not isinstance(tags, list) or not isinstance(evidence, list):
                raise TypeError("tags and evidence must be arrays")
            entry_id = str(raw["entry_id"]) if raw["entry_id"] is not None else None
            if entry_id is not None and allowed_entry_ids is not None:
                if entry_id not in allowed_entry_ids:
                    raise InvalidProviderResponse(
                        "candidate entry_id is not grounded in retrieved evidence"
                    )
            if entry_id is not None and codebook is not None:
                grounded_entry = codebook.get_entry(entry_id)
                if grounded_entry.review_status != "approved":
                    raise InvalidProviderResponse("candidate entry_id is not approved")
            candidates.append(
                CandidateInterpretation(
                    candidate_id=str(uuid4()),
                    entry_id=entry_id,
                    dialect=str(raw["dialect"]),
                    clarification=str(raw["clarification"]),
                    intent=str(raw["intent"]),
                    tone_category=str(raw["tone_category"]),
                    linguistic_context=str(raw["linguistic_context"]),
                    pragmatic_analysis=str(raw["pragmatic_analysis"]),
                    sociolinguistic_tags=tuple(str(item) for item in tags),
                    confidence=float(raw["confidence"]),
                    evidence=tuple(str(item) for item in evidence),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, InvalidProviderResponse):
                raise
            raise InvalidProviderResponse(f"invalid candidate: {exc}") from exc
    return tuple(candidates)
