from __future__ import annotations

import json

import pytest

from iedi.edge import NetworkProfile, make_semantic_payload
from iedi.pipeline import build_pipeline
from iedi.schemas import AcousticAffect, InterpretationRequest
from iedi.transport import (
    CloudInterpretationService,
    EdgeCloudProtocolError,
    EdgeInterpretationClient,
    LoopbackEdgeCloudTransport,
    create_fastapi_app,
    deserialize_semantic_request,
)


def test_edge_cloud_protocol_transmits_semantics_not_audio(codebook, fake_provider) -> None:
    pipeline = build_pipeline("paper4", codebook=codebook, provider=fake_provider)
    client = EdgeInterpretationClient(
        LoopbackEdgeCloudTransport(CloudInterpretationService(pipeline))
    )
    request = InterpretationRequest(
        "wahala",
        speaker_id="SPEAKER_00",
        conversation_context=("prior turn",),
        asr_confidence=0.91,
    )
    observation = client.interpret(
        request,
        raw_audio_bytes=96 * 1024,
        network_profile=NetworkProfile("emulated", 10_000, 20_000, 45),
    )
    payload = make_semantic_payload(request, ())
    decoded = json.loads(payload.body)
    assert observation.wire_payload_bytes == len(payload.body)
    assert observation.payload_reduction is not None
    assert "audio" not in decoded and "raw_audio" not in decoded
    assert observation.result["decision"]["used_route"] == "local"
    assert observation.result["request_id"] == request.request_id


def test_cloud_boundary_rejects_raw_audio() -> None:
    with pytest.raises(EdgeCloudProtocolError, match="raw audio"):
        deserialize_semantic_request(b'{"id":"1","u":"hello","raw_audio":"AAAA"}')


def test_cloud_boundary_rejects_unexpected_fields() -> None:
    with pytest.raises(EdgeCloudProtocolError, match="unexpected"):
        deserialize_semantic_request(b'{"id":"1","u":"hello","claim":"45 bytes"}')


def test_semantic_wire_round_trip_preserves_routing_fields() -> None:
    request = InterpretationRequest(
        "I beg",
        active_persona_ids=("ng-en-v1",),
        conversation_context=("turn one", "turn two", "turn three", "turn four"),
        speaker_id="SPEAKER_01",
        speaker_role="older relative",
        supplied_tone="Frustrated",
        supplied_context="response challenging a preceding claim",
        acoustic_affect=AcousticAffect(
            label="Frustrated",
            confidence=0.83,
            extractor_id="fixture-extractor",
            extractor_version="1.2.0",
            features={"pitch_median_hz": 176.5, "rms_mean": 0.12},
        ),
        asr_confidence=0.91,
        network_available=False,
        latency_budget_ms=350.0,
        cost_budget_usd=0.004,
        risk_score=0.8,
        request_id="round-trip-1",
    )
    restored = deserialize_semantic_request(make_semantic_payload(request, ()).body)
    assert restored == request


def test_fastapi_factory_registers_semantic_endpoint(codebook, fake_provider) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    pipeline = build_pipeline("paper4", codebook=codebook, provider=fake_provider)
    app = create_fastapi_app(CloudInterpretationService(pipeline))
    assert any(route.path == "/v1/interpret" for route in app.routes)
    client = TestClient(app)
    response = client.post(
        "/v1/interpret",
        content=make_semantic_payload(InterpretationRequest("wahala"), ()).body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    assert response.json()["decision"]["used_route"] == "local"
    rejected = client.post(
        "/v1/interpret",
        content=b'{"v":1,"id":"x","u":"hello","raw_audio":"AAAA"}',
        headers={"content-type": "application/json"},
    )
    assert rejected.status_code == 422
