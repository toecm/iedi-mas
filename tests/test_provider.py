from __future__ import annotations

from dataclasses import dataclass

import pytest

from iedi.providers import (
    GoogleGenAIProvider,
    InvalidProviderResponse,
    ProviderQuotaError,
)


@dataclass
class Usage:
    prompt_token_count: int = 10
    candidates_token_count: int = 4
    total_token_count: int = 14


class Response:
    parsed = {"candidates": []}
    text = '{"candidates":[]}'
    usage_metadata = Usage()
    response_id = "response-1"
    model_version = "gemini-2.5-flash-001"


class FakeModels:
    def __init__(self, *, error=None, response=None):
        self.error = error
        self.response = response or Response()
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, models):
        self.models = models


def test_google_adapter_requests_strict_json_schema_without_live_call() -> None:
    models = FakeModels()
    provider = GoogleGenAIProvider(client=FakeClient(models))
    response = provider.generate(
        model_id="gemini-2.5-flash",
        prompt="test",
        response_schema={"type": "object"},
        temperature=0.1,
        timeout_s=5,
    )
    assert models.kwargs["model"] == "gemini-2.5-flash"
    assert models.kwargs["config"]["response_mime_type"] == "application/json"
    assert models.kwargs["config"]["response_json_schema"] == {"type": "object"}
    assert response.total_tokens == 14
    assert response.model_version == "gemini-2.5-flash-001"


def test_google_adapter_classifies_quota_errors() -> None:
    error = RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")
    provider = GoogleGenAIProvider(client=FakeClient(FakeModels(error=error)))
    with pytest.raises(ProviderQuotaError) as raised:
        provider.generate(
            model_id="gemini-2.5-pro",
            prompt="test",
            response_schema={"type": "object"},
            temperature=0.1,
            timeout_s=5,
        )
    assert raised.value.latency_ms >= 0


def test_google_adapter_rejects_non_object_root() -> None:
    response = Response()
    response.parsed = ["not", "an", "object"]
    provider = GoogleGenAIProvider(client=FakeClient(FakeModels(response=response)))
    with pytest.raises(InvalidProviderResponse):
        provider.generate(
            model_id="gemini-2.5-pro",
            prompt="test",
            response_schema={"type": "object"},
            temperature=0.1,
            timeout_s=5,
        )
