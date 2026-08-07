from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping

from .schemas import InterpretationRequest, MatchEvidence


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    uplink_kbps: float
    downlink_kbps: float
    base_rtt_ms: float
    jitter_ms: float = 0.0
    packet_loss_percent: float = 0.0
    emulator: str | None = None

    def __post_init__(self) -> None:
        if self.uplink_kbps <= 0 or self.downlink_kbps <= 0:
            raise ValueError("network bandwidth must be positive")
        if self.base_rtt_ms < 0 or self.jitter_ms < 0:
            raise ValueError("latency values cannot be negative")
        if not 0.0 <= self.packet_loss_percent <= 100.0:
            raise ValueError("packet loss must be in [0, 100]")


@dataclass(frozen=True)
class SerializedPayload:
    body: bytes
    serialization_ms: float

    @property
    def size_bytes(self) -> int:
        return len(self.body)


def make_semantic_payload(
    request: InterpretationRequest,
    evidence: Iterable[MatchEvidence],
    *,
    max_evidence: int = 3,
) -> SerializedPayload:
    """Create deterministic UTF-8 JSON; size is the actual serialized byte length."""

    started = perf_counter()
    compact: dict[str, Any] = {
        "v": 1,
        "id": request.request_id,
        "u": request.utterance,
        "n": request.network_available,
        "k": request.risk_score,
    }
    if request.speaker_id:
        compact["s"] = request.speaker_id
    if request.speaker_role:
        compact["r"] = request.speaker_role
    if request.supplied_tone:
        compact["t"] = request.supplied_tone
    if request.supplied_context:
        compact["c"] = request.supplied_context
    if request.active_persona_ids:
        compact["p"] = list(request.active_persona_ids)
    if request.conversation_context:
        compact["h"] = list(request.conversation_context)
    if request.asr_confidence is not None:
        compact["q"] = request.asr_confidence
    if request.cost_budget_usd is not None:
        compact["b"] = request.cost_budget_usd
    if request.latency_budget_ms is not None:
        compact["m"] = request.latency_budget_ms
    if request.acoustic_affect is not None:
        affect = {
            "l": request.acoustic_affect.label,
            "q": request.acoustic_affect.confidence,
            "x": request.acoustic_affect.extractor_id,
            "v": request.acoustic_affect.extractor_version,
            "f": dict(request.acoustic_affect.features),
        }
        compact["a"] = {key: value for key, value in affect.items() if value is not None}

    grounded = [item.entry.entry_id for item in evidence][:max_evidence]
    if grounded:
        compact["e"] = grounded

    body = json.dumps(
        compact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return SerializedPayload(body, (perf_counter() - started) * 1000.0)


def estimate_transfer_ms(size_bytes: int, uplink_kbps: float) -> float:
    if size_bytes < 0 or uplink_kbps <= 0:
        raise ValueError("invalid payload size or bandwidth")
    return (size_bytes * 8.0) / (uplink_kbps * 1000.0) * 1000.0


def payload_reduction(raw_audio_bytes: int, semantic_payload_bytes: int) -> float:
    if raw_audio_bytes <= 0 or semantic_payload_bytes < 0:
        raise ValueError("invalid raw audio or semantic payload size")
    return 1.0 - semantic_payload_bytes / raw_audio_bytes


def network_profile_dict(profile: NetworkProfile) -> Mapping[str, Any]:
    return asdict(profile)
