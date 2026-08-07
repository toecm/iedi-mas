from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, Protocol


class ProviderError(RuntimeError):
    retryable = False
    reason = "provider_error"


class ProviderAuthenticationError(ProviderError):
    reason = "authentication"


class ProviderQuotaError(ProviderError):
    retryable = True
    reason = "quota"


class ProviderTimeoutError(ProviderError):
    retryable = True
    reason = "timeout"


class ProviderServiceError(ProviderError):
    retryable = True
    reason = "service"


class InvalidProviderResponse(ProviderError):
    reason = "invalid_schema"


@dataclass(frozen=True)
class ProviderResponse:
    data: Mapping[str, Any]
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    response_id: str | None = None
    model_version: str | None = None
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        response_schema: Mapping[str, Any],
        temperature: float,
        timeout_s: float,
    ) -> ProviderResponse: ...


class OfflineFixtureProvider:
    """Deterministic schema fixture for notebooks/tests; never evaluation evidence.

    It echoes reviewed retrieval evidence from the DMM prompt and performs no model
    inference.  ``model_version`` is prefixed with ``offline-fixture`` so telemetry
    cannot be confused with a live Gemini result.
    """

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        response_schema: Mapping[str, Any],
        temperature: float,
        timeout_s: float,
    ) -> ProviderResponse:
        del response_schema, temperature, timeout_s
        request = json.loads(prompt)
        evidence = request.get("retrieved_evidence", [])
        requested_count = request.get("candidate_count")
        count = 3 if requested_count == 3 else 1
        selected = evidence[:count]
        if requested_count == 3 and len(selected) != 3:
            raise InvalidProviderResponse(
                "offline fixture will not fabricate missing curated senses"
            )
        candidates = []
        for item in selected:
            tones = item.get("tone_categories") or ["Unspecified"]
            contexts = item.get("linguistic_contexts") or ["codebook-defined context"]
            candidates.append(
                {
                    "entry_id": item.get("entry_id"),
                    "dialect": item.get("dialect") or "Unspecified",
                    "clarification": item.get("universal_gloss") or "Unresolved",
                    "intent": item.get("intent") or "requires human review",
                    "tone_category": tones[0],
                    "linguistic_context": contexts[0],
                    "pragmatic_analysis": item.get("pragmatic_analysis")
                    or "No reviewed pragmatic analysis",
                    "sociolinguistic_tags": list(item.get("sociolinguistic_tags") or []),
                    "confidence": min(float(item.get("score", 0.0)), 0.5),
                    "evidence": [item.get("entry_id"), "offline-fixture-echo"],
                }
            )
        if not candidates:
            candidates.append(
                {
                    "entry_id": None,
                    "dialect": "Unresolved",
                    "clarification": "No reviewed codebook evidence; human review required",
                    "intent": "unresolved",
                    "tone_category": "Unspecified",
                    "linguistic_context": "insufficient evidence",
                    "pragmatic_analysis": "Offline fixture performs no inference",
                    "sociolinguistic_tags": ["unresolved"],
                    "confidence": 0.0,
                    "evidence": ["offline-fixture-no-model-call"],
                }
            )
        return ProviderResponse(
            data={"candidates": candidates},
            latency_ms=0.0,
            response_id="offline-fixture",
            model_version=f"offline-fixture::{model_id}",
        )


class GoogleGenAIProvider:
    """Current Google Gen AI SDK adapter with strict structured output."""

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ProviderAuthenticationError(
                "set GEMINI_API_KEY (or GOOGLE_API_KEY) before constructing the Gemini provider"
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError("install the 'gemini' extra: pip install -e .[gemini]") from exc
        self._client = genai.Client(api_key=resolved_key)

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        response_schema: Mapping[str, Any],
        temperature: float,
        timeout_s: float,
    ) -> ProviderResponse:
        started = perf_counter()
        try:
            response = self._client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": dict(response_schema),
                    "temperature": temperature,
                    "http_options": {"timeout": int(timeout_s * 1000)},
                },
            )
        except Exception as exc:  # SDK exception hierarchy has changed across releases.
            classified = _classify_google_error(exc)
            classified.latency_ms = (perf_counter() - started) * 1000.0
            raise classified from exc

        elapsed_ms = (perf_counter() - started) * 1000.0
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            try:
                parsed = json.loads(response.text)
            except (AttributeError, TypeError, json.JSONDecodeError) as exc:
                invalid = InvalidProviderResponse("Gemini did not return valid JSON")
                invalid.latency_ms = (perf_counter() - started) * 1000.0
                raise invalid from exc
        if not isinstance(parsed, Mapping):
            invalid = InvalidProviderResponse("Gemini response root must be an object")
            invalid.latency_ms = (perf_counter() - started) * 1000.0
            raise invalid

        usage = getattr(response, "usage_metadata", None)
        return ProviderResponse(
            data=dict(parsed),
            latency_ms=elapsed_ms,
            input_tokens=_int_attr(usage, "prompt_token_count"),
            output_tokens=_int_attr(usage, "candidates_token_count"),
            total_tokens=_int_attr(usage, "total_token_count"),
            response_id=getattr(response, "response_id", None),
            model_version=getattr(response, "model_version", None),
        )


def _int_attr(value: Any, name: str) -> int | None:
    raw = getattr(value, name, None)
    return int(raw) if raw is not None else None


def _classify_google_error(exc: Exception) -> ProviderError:
    message = str(exc).casefold()
    code = getattr(exc, "code", None)
    status_code = getattr(exc, "status_code", None)
    numeric_code = code if isinstance(code, int) else status_code

    if numeric_code in {401, 403} or any(
        marker in message for marker in ("unauthenticated", "permission_denied", "api key")
    ):
        return ProviderAuthenticationError(str(exc))
    if numeric_code == 429 or any(
        marker in message for marker in ("resource_exhausted", "quota", "rate limit")
    ):
        return ProviderQuotaError(str(exc))
    if numeric_code in {408, 504} or "timeout" in message or "deadline" in message:
        return ProviderTimeoutError(str(exc))
    if numeric_code is not None and 500 <= int(numeric_code) <= 599:
        return ProviderServiceError(str(exc))
    if any(marker in message for marker in ("service unavailable", "temporarily unavailable")):
        return ProviderServiceError(str(exc))
    return ProviderError(str(exc))
