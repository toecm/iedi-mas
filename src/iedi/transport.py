from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Protocol
from urllib.request import Request, urlopen

from .edge import NetworkProfile, estimate_transfer_ms, make_semantic_payload
from .pipeline import IEDIPipeline
from .schemas import AcousticAffect, InterpretationRequest


class EdgeCloudProtocolError(RuntimeError):
    pass


class EdgeCloudTransport(Protocol):
    def send(self, body: bytes) -> tuple[Mapping[str, Any], float]: ...


@dataclass(frozen=True)
class EdgeCloudObservation:
    result: Mapping[str, Any]
    wire_payload_bytes: int
    serialization_ms: float
    estimated_transfer_ms: float
    observed_gateway_round_trip_ms: float
    raw_audio_bytes: int | None

    @property
    def payload_reduction(self) -> float | None:
        if self.raw_audio_bytes is None:
            return None
        return 1.0 - self.wire_payload_bytes / self.raw_audio_bytes


class CloudInterpretationService:
    """Cloud-side handler. Raw audio is neither accepted nor deserialized."""

    def __init__(self, pipeline: IEDIPipeline, *, max_payload_bytes: int = 16_384) -> None:
        self.pipeline = pipeline
        self.max_payload_bytes = max_payload_bytes

    def handle(self, body: bytes) -> Mapping[str, Any]:
        request = deserialize_semantic_request(body, max_payload_bytes=self.max_payload_bytes)
        return self.pipeline.interpret(request).to_dict()


class LoopbackEdgeCloudTransport:
    """Protocol test adapter; it does not constitute a network deployment."""

    def __init__(self, service: CloudInterpretationService) -> None:
        self.service = service

    def send(self, body: bytes) -> tuple[Mapping[str, Any], float]:
        started = perf_counter()
        result = self.service.handle(body)
        return result, (perf_counter() - started) * 1000.0


class HttpEdgeCloudTransport:
    def __init__(self, endpoint: str, *, timeout_s: float = 60.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def send(self, body: bytes) -> tuple[Mapping[str, Any], float]:
        started = perf_counter()
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            response_body = response.read()
        elapsed_ms = (perf_counter() - started) * 1000.0
        try:
            parsed = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EdgeCloudProtocolError("cloud response is not valid UTF-8 JSON") from exc
        if not isinstance(parsed, Mapping):
            raise EdgeCloudProtocolError("cloud response root must be an object")
        return parsed, elapsed_ms


class EdgeInterpretationClient:
    def __init__(self, transport: EdgeCloudTransport) -> None:
        self.transport = transport

    def interpret(
        self,
        request: InterpretationRequest,
        *,
        raw_audio_bytes: int | None = None,
        network_profile: NetworkProfile | None = None,
    ) -> EdgeCloudObservation:
        payload = make_semantic_payload(request, ())
        result, observed_rtt_ms = self.transport.send(payload.body)
        estimated_ms = (
            estimate_transfer_ms(payload.size_bytes, network_profile.uplink_kbps)
            if network_profile is not None
            else 0.0
        )
        return EdgeCloudObservation(
            result=result,
            wire_payload_bytes=payload.size_bytes,
            serialization_ms=payload.serialization_ms,
            estimated_transfer_ms=estimated_ms,
            observed_gateway_round_trip_ms=observed_rtt_ms,
            raw_audio_bytes=raw_audio_bytes,
        )


def deserialize_semantic_request(
    body: bytes, *, max_payload_bytes: int = 16_384
) -> InterpretationRequest:
    if len(body) > max_payload_bytes:
        raise EdgeCloudProtocolError("semantic payload exceeds configured maximum")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EdgeCloudProtocolError("semantic payload is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise EdgeCloudProtocolError("semantic payload root must be an object")
    forbidden = {"audio", "audio_bytes", "raw_audio", "waveform"}
    if forbidden & set(payload):
        raise EdgeCloudProtocolError("raw audio is forbidden at the semantic boundary")
    allowed = {
        "v",
        "id",
        "u",
        "s",
        "r",
        "t",
        "c",
        "p",
        "h",
        "q",
        "b",
        "m",
        "n",
        "k",
        "a",
        "e",
    }
    unexpected = set(payload) - allowed
    if unexpected:
        raise EdgeCloudProtocolError(f"unexpected semantic fields: {sorted(unexpected)}")
    if payload.get("v", 1) != 1:
        raise EdgeCloudProtocolError("unsupported semantic wire-schema version")

    raw_affect = payload.get("a")
    affect = None
    if raw_affect is not None:
        if not isinstance(raw_affect, Mapping):
            raise EdgeCloudProtocolError("acoustic evidence must be an object")
        raw_features = raw_affect.get("f", {})
        if not isinstance(raw_features, Mapping) or not all(
            isinstance(name, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            for name, value in raw_features.items()
        ):
            raise EdgeCloudProtocolError("acoustic features must map strings to numbers")
        affect = AcousticAffect(
            label=str(raw_affect["l"]) if raw_affect.get("l") is not None else None,
            confidence=float(raw_affect["q"]) if raw_affect.get("q") is not None else None,
            extractor_id=str(raw_affect["x"]) if raw_affect.get("x") is not None else None,
            extractor_version=(
                str(raw_affect["v"]) if raw_affect.get("v") is not None else None
            ),
            features={str(name): float(value) for name, value in raw_features.items()},
        )

    personas = payload.get("p", [])
    history = payload.get("h", [])
    if not isinstance(personas, list) or not all(isinstance(item, str) for item in personas):
        raise EdgeCloudProtocolError("persona IDs must be an array of strings")
    if not isinstance(history, list) or not all(isinstance(item, str) for item in history):
        raise EdgeCloudProtocolError("conversation history must be an array of strings")
    request_id = str(payload.get("id", "")).strip()
    if not request_id:
        raise EdgeCloudProtocolError("request ID is required")
    network_available = payload.get("n", True)
    if not isinstance(network_available, bool):
        raise EdgeCloudProtocolError("network availability must be boolean")
    return InterpretationRequest(
        utterance=str(payload.get("u", "")),
        active_persona_ids=tuple(personas),
        conversation_context=tuple(history),
        speaker_id=str(payload["s"]) if payload.get("s") is not None else None,
        speaker_role=str(payload["r"]) if payload.get("r") is not None else None,
        supplied_tone=str(payload["t"]) if payload.get("t") is not None else None,
        supplied_context=str(payload["c"]) if payload.get("c") is not None else None,
        acoustic_affect=affect,
        asr_confidence=float(payload["q"]) if payload.get("q") is not None else None,
        cost_budget_usd=float(payload["b"]) if payload.get("b") is not None else None,
        latency_budget_ms=float(payload["m"]) if payload.get("m") is not None else None,
        network_available=network_available,
        risk_score=float(payload.get("k", 0.0)),
        request_id=request_id,
    )


def create_fastapi_app(service: CloudInterpretationService) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the cloud extra: pip install -e .[cloud]") from exc

    app = FastAPI(title="IEDI semantic interpretation service")

    async def interpret(request) -> Mapping[str, Any]:
        body = await request.body()
        try:
            return await run_in_threadpool(service.handle, body)
        except EdgeCloudProtocolError as exc:
            status = 413 if "exceeds configured maximum" in str(exc) else 422
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # FastAPI needs the concrete Request class. Assign it after function creation so
    # postponed annotations do not leave an unresolvable local-name string.
    interpret.__annotations__["request"] = FastAPIRequest
    app.post("/v1/interpret")(interpret)

    return app
