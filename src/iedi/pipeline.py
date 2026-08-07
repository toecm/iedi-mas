from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Mapping

from .codebook import Codebook, normalize_text
from .dmm import DMMPolicy, DynamicModelManager
from .edge import NetworkProfile, estimate_transfer_ms, make_semantic_payload
from .providers import ModelProvider
from .schemas import (
    CodebookEntry,
    InterpretationRequest,
    InterpretationResult,
    RoutingFeatures,
    TimingTrace,
)


@dataclass(frozen=True)
class PaperProfile:
    paper_id: str
    description: str
    dmm_policy: DMMPolicy
    require_active_persona: bool = False
    return_three_for_unresolved_polysemy: bool = False
    use_acoustic_affect: bool = False
    trust_gate_required: bool = False
    require_ipfs_verification: bool = False
    require_finalized_chain_commitment: bool = False
    claim_requirements: Mapping[str, bool] = field(default_factory=dict)

    def missing_claim_evidence(self, evidence: Mapping[str, bool]) -> tuple[str, ...]:
        return tuple(
            name
            for name, required in self.claim_requirements.items()
            if required and not bool(evidence.get(name, False))
        )

    def require_claim_evidence(self, evidence: Mapping[str, bool]) -> None:
        missing = self.missing_claim_evidence(evidence)
        if missing:
            raise ValueError(
                f"{self.paper_id} empirical/operational claim evidence is missing: "
                + ", ".join(missing)
            )


PAPER_PROFILES: Mapping[str, PaperProfile] = {
    "paper2": PaperProfile(
        paper_id="paper2",
        description="KICS 2025: >=80% local retrieval; unmatched or ambiguous input to adapted LLM",
        dmm_policy=DMMPolicy(
            version="paper2-hybrid-v1",
            local_threshold=0.80,
            local_margin=0.05,
            pro_ambiguity_threshold=0.50,
            cold_start_requests=0,
        ),
    ),
    "paper3": PaperProfile(
        paper_id="paper3",
        description="KICS Winter 2026: persona-first Pro/Flash DMM and three polysemy options",
        dmm_policy=DMMPolicy(
            version="paper3-persona-dmm-v1",
            local_threshold=0.80,
            local_margin=0.05,
            pro_ambiguity_threshold=0.48,
            cold_start_requests=0,
        ),
        require_active_persona=True,
        return_three_for_unresolved_polysemy=True,
    ),
    "paper4": PaperProfile(
        paper_id="paper4",
        description="JCCI 2026: edge semantic payload with route-specific measurements",
        dmm_policy=DMMPolicy(
            version="paper4-edge-cloud-dmm-v1",
            local_threshold=0.80,
            local_margin=0.05,
            pro_ambiguity_threshold=0.58,
            cold_start_requests=0,
        ),
    ),
    "paper5": PaperProfile(
        paper_id="paper5",
        description="CA-IEDI 2026: context-aware MAS, DMM, affect and fail-closed trust gate",
        dmm_policy=DMMPolicy(
            version="paper5-ca-iedi-dmm-v1",
            local_threshold=0.80,
            local_margin=0.05,
            pro_ambiguity_threshold=0.58,
            cold_start_requests=50,
        ),
        require_active_persona=True,
        return_three_for_unresolved_polysemy=True,
        use_acoustic_affect=True,
        trust_gate_required=True,
        require_ipfs_verification=True,
        require_finalized_chain_commitment=True,
    ),
}


class IEDIPipeline:
    def __init__(
        self,
        *,
        codebook: Codebook,
        dmm: DynamicModelManager,
        profile: PaperProfile,
    ) -> None:
        self._codebook = codebook
        self._codebook_lock = threading.RLock()
        self.dmm = dmm
        self.profile = profile
        if profile.return_three_for_unresolved_polysemy and dmm.policy.max_candidates < 3:
            raise ValueError("polysemy profile requires DMM max_candidates >= 3")

    @property
    def codebook(self) -> Codebook:
        with self._codebook_lock:
            return self._codebook

    def append_codebook_entry(self, entry: CodebookEntry) -> str:
        """Atomically materialize and publish a new immutable retrieval snapshot."""

        with self._codebook_lock:
            try:
                existing = self._codebook.get_entry(entry.entry_id)
            except KeyError:
                existing = None
            if existing is not None:
                if existing != entry:
                    raise ValueError(
                        f"entry_id already exists with different content: {entry.entry_id}"
                    )
                return self._codebook.version_hash
            updated = self._codebook.append_version(entry)
            self._codebook = updated
            return updated.version_hash

    def interpret(
        self,
        request: InterpretationRequest,
        *,
        raw_audio_bytes: int | None = None,
        edge_asr_ms: float = 0.0,
        network_profile: NetworkProfile | None = None,
        task: str = "interpret",
    ) -> InterpretationResult:
        started = perf_counter()
        with self._codebook_lock:
            codebook = self._codebook
        if self.profile.require_active_persona and not request.active_persona_ids:
            raise ValueError(f"{self.profile.paper_id} requires a validated active persona")
        missing_affect = self.profile.use_acoustic_affect and not _valid_acoustic_evidence(
            request
        )

        retrieval_started = perf_counter()
        evidence = codebook.search(request, top_k=5)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000.0
        features = self._routing_features(request, evidence, codebook)

        payload = make_semantic_payload(request, evidence)
        unresolved_polysemy = (
            self.profile.return_three_for_unresolved_polysemy
            and features.polysemous_surface
            and not features.context_complete
        )
        require_three = unresolved_polysemy and features.plausible_senses >= 3
        curation_gap = unresolved_polysemy and features.plausible_senses < 3
        outcome = self.dmm.interpret(
            request=request,
            evidence=evidence,
            features=features,
            codebook=codebook,
            task=task,
            require_three=require_three,
        )
        if curation_gap:
            outcome = replace(
                outcome,
                decision=replace(
                    outcome.decision,
                    reasons=(
                        *outcome.decision.reasons,
                        "insufficient_curated_polysemy_senses",
                    ),
                    degraded=True,
                ),
                needs_human_review=True,
            )

        uplink_ms = 0.0
        if network_profile is not None and outcome.calls:
            uplink_ms = estimate_transfer_ms(payload.size_bytes, network_profile.uplink_kbps)
        api_round_trip_ms = sum(call.latency_ms for call in outcome.calls)
        total_ms = (perf_counter() - started) * 1000.0 + edge_asr_ms
        profiles = codebook.active_personas(request.active_persona_ids)

        return InterpretationResult(
            request_id=request.request_id,
            candidates=outcome.candidates,
            decision=outcome.decision,
            retrieved_entry_ids=tuple(item.entry.entry_id for item in evidence),
            profile_ids=tuple(profile.profile_id for profile in profiles),
            profile_hashes=tuple(profile.content_hash for profile in profiles),
            model_calls=outcome.calls,
            payload_bytes=payload.size_bytes,
            raw_audio_bytes=raw_audio_bytes,
            timing=TimingTrace(
                edge_asr_ms=edge_asr_ms,
                edge_retrieval_ms=retrieval_ms,
                serialization_ms=payload.serialization_ms,
                estimated_uplink_transfer_ms=uplink_ms,
                observed_api_round_trip_ms=api_round_trip_ms,
                response_parse_ms=outcome.response_parse_ms,
                end_to_end_ms=total_ms,
            ),
            needs_human_review=outcome.needs_human_review or missing_affect or curation_gap,
            dataset_version=codebook.version_hash,
        )

    def _routing_features(
        self, request: InterpretationRequest, evidence: tuple, codebook: Codebook
    ) -> RoutingFeatures:
        top_score = evidence[0].score if evidence else 0.0
        second_score = evidence[1].score if len(evidence) > 1 else 0.0
        margin = max(top_score - second_score, 0.0)
        plausible = [
            item for item in evidence if item.score >= self.dmm.policy.plausible_threshold
        ]
        semantic_senses = {
            (item.entry.concept_id, item.entry.universal_gloss.casefold()) for item in plausible
        }
        persona_meanings = {
            item.entry.universal_gloss.casefold()
            for item in plausible
            if item.persona_priority
        }
        generic_meanings = {
            item.entry.universal_gloss.casefold()
            for item in plausible
            if not item.persona_priority
        }
        persona_conflict = bool(persona_meanings and generic_meanings - persona_meanings)
        context_complete = _has_relevant_context(request, plausible)
        return RoutingFeatures(
            top_score=top_score,
            second_score=second_score,
            score_margin=margin,
            plausible_senses=len(semantic_senses),
            persona_conflict=persona_conflict,
            context_complete=context_complete,
            asr_confidence=request.asr_confidence,
            polysemous_surface=codebook.is_polysemous_surface(request.utterance),
            risk_score=request.risk_score,
        )


def build_pipeline(
    paper_id: str,
    *,
    codebook: Codebook,
    provider: ModelProvider,
    policy_override: DMMPolicy | None = None,
    config_path: str | Path | None = None,
) -> IEDIPipeline:
    try:
        profile = PAPER_PROFILES[paper_id]
    except KeyError as exc:
        raise ValueError(f"unsupported paper profile: {paper_id}") from exc
    if config_path is not None:
        profile = load_paper_profile(config_path, expected_paper_id=paper_id)
    if policy_override is not None:
        profile = PaperProfile(
            paper_id=profile.paper_id,
            description=profile.description,
            dmm_policy=policy_override,
            require_active_persona=profile.require_active_persona,
            return_three_for_unresolved_polysemy=profile.return_three_for_unresolved_polysemy,
            use_acoustic_affect=profile.use_acoustic_affect,
            trust_gate_required=profile.trust_gate_required,
            require_ipfs_verification=profile.require_ipfs_verification,
            require_finalized_chain_commitment=profile.require_finalized_chain_commitment,
            claim_requirements=profile.claim_requirements,
        )
    dmm = DynamicModelManager(provider=provider, policy=profile.dmm_policy)
    return IEDIPipeline(codebook=codebook, dmm=dmm, profile=profile)


def load_paper_profile(
    path: str | Path, *, expected_paper_id: str | None = None
) -> PaperProfile:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("paper configuration must be an object")
    paper_id = str(raw.get("paper_id", ""))
    if expected_paper_id is not None and paper_id != expected_paper_id:
        raise ValueError(
            f"configuration is for {paper_id!r}, expected {expected_paper_id!r}"
        )
    try:
        base = PAPER_PROFILES[paper_id]
    except KeyError as exc:
        raise ValueError(f"unsupported paper profile: {paper_id}") from exc

    aliases = {
        "ambiguity_margin": "local_margin",
        "require_validated_persona": "require_active_persona",
        "return_three_for_unresolved_polysemy": "return_three_for_unresolved_polysemy",
        "require_acoustic_evidence_for_affect_claim": "use_acoustic_affect",
        "require_ipfs_verification": "require_ipfs_verification",
        "require_finalized_chain_commitment": "require_finalized_chain_commitment",
    }
    policy_fields = set(DMMPolicy.__dataclass_fields__)
    policy_values = {
        key: value for key, value in raw.items() if key in policy_fields
    }
    if "ambiguity_margin" in raw:
        policy_values["local_margin"] = raw["ambiguity_margin"]
    policy = replace(base.dmm_policy, **policy_values)

    raw_claim_requirements = raw.get("claim_requirements", {})
    if not isinstance(raw_claim_requirements, dict):
        raise ValueError("claim_requirements must be an object of boolean evidence flags")
    profile_values = {
        "require_active_persona": base.require_active_persona,
        "return_three_for_unresolved_polysemy": base.return_three_for_unresolved_polysemy,
        "use_acoustic_affect": base.use_acoustic_affect,
        "trust_gate_required": base.trust_gate_required,
        "require_ipfs_verification": base.require_ipfs_verification,
        "require_finalized_chain_commitment": base.require_finalized_chain_commitment,
        "claim_requirements": {
            str(name): bool(required)
            for name, required in raw_claim_requirements.items()
        },
    }
    for source_key, destination_key in aliases.items():
        if source_key in raw and destination_key in profile_values:
            profile_values[destination_key] = bool(raw[source_key])
    return PaperProfile(
        paper_id=paper_id,
        description=str(raw.get("architecture", base.description)),
        dmm_policy=policy,
        **profile_values,
    )


def _valid_acoustic_evidence(request: InterpretationRequest) -> bool:
    affect = request.acoustic_affect
    if affect is None:
        return False
    return bool(
        affect.label
        and affect.extractor_id
        and affect.extractor_version
        and affect.confidence is not None
        and math.isfinite(affect.confidence)
        and affect.features
        and all(math.isfinite(float(value)) for value in affect.features.values())
    )


def _has_relevant_context(request: InterpretationRequest, plausible: list) -> bool:
    """Treat context as complete only when it resolves against curated evidence."""

    if any(item.rule_ids for item in plausible):
        return True
    descriptors: set[str] = set()
    for item in plausible:
        entry = item.entry
        descriptors.update(
            normalize_text(value)
            for value in (
                *entry.tone_categories,
                *entry.linguistic_contexts,
                *entry.speaker_roles,
                entry.intent,
                entry.pragmatic_analysis,
            )
            if value
        )
    signals = [
        request.supplied_tone or "",
        request.supplied_context or "",
        request.speaker_role or "",
        *request.conversation_context,
    ]
    if (
        request.acoustic_affect is not None
        and request.acoustic_affect.label
        and request.acoustic_affect.confidence is not None
        and request.acoustic_affect.confidence >= 0.70
    ):
        signals.append(request.acoustic_affect.label)
    normalized_signals = [normalize_text(value) for value in signals if value]
    return any(
        descriptor == signal
        or descriptor in signal
        or signal in descriptor
        for descriptor in descriptors
        for signal in normalized_signals
        if descriptor and signal
    )
