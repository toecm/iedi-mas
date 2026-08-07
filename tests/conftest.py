from __future__ import annotations

import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping

import pytest

from iedi.codebook import Codebook
from iedi.providers import ProviderResponse


def candidate_payload(
    clarification: str = "interpreted meaning",
    *,
    entry_id: str | None = None,
    confidence: float = 0.82,
    tone: str = "Neutral",
) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "dialect": "Nigerian English",
        "clarification": clarification,
        "intent": "communicate meaning",
        "tone_category": tone,
        "linguistic_context": "conversation",
        "pragmatic_analysis": "context-sensitive interpretation",
        "sociolinguistic_tags": ["informal"],
        "confidence": confidence,
        "evidence": [entry_id or "model-analysis"],
    }


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: dict[str, Mapping[str, Any]] = {}
        self.errors: dict[str, deque[Exception]] = defaultdict(deque)
        self._lock = threading.Lock()

    def queue_error(self, model_id: str, error: Exception) -> None:
        self.errors[model_id].append(error)

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        response_schema: Mapping[str, Any],
        temperature: float,
        timeout_s: float,
    ) -> ProviderResponse:
        with self._lock:
            self.calls.append(
                {
                    "model_id": model_id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "timeout_s": timeout_s,
                }
            )
            if self.errors[model_id]:
                raise self.errors[model_id].popleft()
            data = self.responses.get(
                model_id,
                {"candidates": [candidate_payload()]},
            )
        return ProviderResponse(
            data=data,
            latency_ms=12.5,
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            response_id=f"fake-{len(self.calls)}",
            model_version=model_id,
        )


@pytest.fixture
def codebook() -> Codebook:
    path = Path(__file__).parents[1] / "data" / "codebook.demo.json"
    return Codebook.from_json(path)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
