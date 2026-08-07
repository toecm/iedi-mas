from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from iedi.dmm import DMMExecutionError, DMMPolicy
from iedi.edge import NetworkProfile
from iedi.pipeline import build_pipeline
from iedi.providers import (
    ProviderAuthenticationError,
    ProviderQuotaError,
    ProviderServiceError,
)
from iedi.schemas import InterpretationRequest, Route
from iedi.schemas import AcousticAffect

from conftest import candidate_payload


def test_unique_authoritative_match_stays_local(codebook, fake_provider) -> None:
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    result = pipeline.interpret(
        InterpretationRequest("wahala", active_persona_ids=("ng-en-v1",))
    )
    assert result.decision.used_route is Route.LOCAL
    assert result.model_calls == ()
    assert fake_provider.calls == []
    assert len(result.candidates) == 1
    assert result.candidates[0].entry_id == "ng-wahala-1"


def test_initial_profile_setup_uses_pro_even_for_known_phrase(codebook, fake_provider) -> None:
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    result = pipeline.interpret(
        InterpretationRequest("wahala", active_persona_ids=("ng-en-v1",)),
        task="initial_profile_setup",
    )
    assert result.decision.used_route is Route.PRO
    assert "initial_profile_setup" in result.decision.reasons


def test_routine_low_ambiguity_unresolved_uses_flash(codebook, fake_provider) -> None:
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    result = pipeline.interpret(
        InterpretationRequest("wh", active_persona_ids=("ng-en-v1",))
    )
    assert result.decision.used_route is Route.FLASH
    assert [call["model_id"] for call in fake_provider.calls] == ["gemini-2.5-flash"]


def test_unresolved_polysemy_uses_pro_and_marks_curation_gap(codebook, fake_provider) -> None:
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    result = pipeline.interpret(
        InterpretationRequest("I beg", active_persona_ids=("ng-en-v1",))
    )
    assert result.decision.used_route is Route.PRO
    assert "multiple_codebook_senses" in result.decision.reasons
    assert "insufficient_curated_polysemy_senses" in result.decision.reasons
    assert result.needs_human_review
    assert fake_provider.calls[0]["model_id"] == "gemini-2.5-pro"
    prompt = json.loads(fake_provider.calls[0]["prompt"])
    persona = prompt["full_persona_context"][0]
    assert persona["cultural_context"]
    assert persona["pragmatic_rules"][0]["rule_id"]
    assert {entry["entry_id"] for entry in persona["entries"]} >= {
        "ng-i-beg-please",
        "ng-i-beg-seriously",
    }


def test_persona_profile_is_required_before_paper3_analysis(codebook, fake_provider) -> None:
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    with pytest.raises(ValueError, match="requires a validated active persona"):
        pipeline.interpret(InterpretationRequest("I beg"))
    assert fake_provider.calls == []


def test_supplied_context_can_resolve_to_flash(codebook, fake_provider) -> None:
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    result = pipeline.interpret(
        InterpretationRequest(
            "I beg",
            active_persona_ids=("ng-en-v1",),
            supplied_tone="Casual",
            supplied_context="discourse marker used to soften commands",
        )
    )
    assert result.decision.used_route is Route.FLASH
    assert fake_provider.calls[0]["model_id"] == "gemini-2.5-flash"


def test_pro_quota_falls_back_once_and_records_it(codebook, fake_provider) -> None:
    fake_provider.queue_error("gemini-2.5-pro", ProviderQuotaError("quota"))
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    result = pipeline.interpret(
        InterpretationRequest("I beg", active_persona_ids=("ng-en-v1",))
    )
    assert [call["model_id"] for call in fake_provider.calls] == [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    assert result.decision.used_route is Route.FLASH
    assert result.decision.fallback_from is Route.PRO
    assert result.decision.fallback_reason == "quota"
    assert result.decision.degraded
    assert [call.status for call in result.model_calls] == ["failed", "success"]


def test_authentication_error_does_not_silently_switch_lanes(codebook, fake_provider) -> None:
    fake_provider.queue_error("gemini-2.5-pro", ProviderAuthenticationError("bad key"))
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    with pytest.raises(DMMExecutionError, match="authentication"):
        pipeline.interpret(
            InterpretationRequest("I beg", active_persona_ids=("ng-en-v1",))
        )
    assert [call["model_id"] for call in fake_provider.calls] == ["gemini-2.5-pro"]


def test_cost_budget_can_select_flash_for_otherwise_pro_request(codebook, fake_provider) -> None:
    policy = DMMPolicy(
        pro_ambiguity_threshold=0.58,
        cold_start_requests=0,
        estimated_flash_call_cost_usd=0.001,
        estimated_pro_call_cost_usd=0.02,
        allow_quality_escalation=False,
    )
    pipeline = build_pipeline(
        "paper3", codebook=codebook, provider=fake_provider, policy_override=policy
    )
    result = pipeline.interpret(
        InterpretationRequest(
            "I beg",
            active_persona_ids=("ng-en-v1",),
            cost_budget_usd=0.005,
        )
    )
    assert result.decision.used_route is Route.FLASH
    assert "cost_budget_requires_flash" in result.decision.reasons


def test_flash_quality_escalation_respects_remaining_latency_and_cost(
    codebook, fake_provider
) -> None:
    fake_provider.responses["gemini-2.5-flash"] = {
        "candidates": [candidate_payload(confidence=0.2)]
    }
    policy = DMMPolicy(
        cold_start_requests=0,
        estimated_flash_call_cost_usd=0.001,
        estimated_pro_call_cost_usd=0.02,
    )
    pipeline = build_pipeline(
        "paper3", codebook=codebook, provider=fake_provider, policy_override=policy
    )
    result = pipeline.interpret(
        InterpretationRequest(
            "wh",
            active_persona_ids=("ng-en-v1",),
            latency_budget_ms=100,
            cost_budget_usd=0.005,
        )
    )
    assert [call["model_id"] for call in fake_provider.calls] == ["gemini-2.5-flash"]
    assert "quality_escalation_budget_blocked" in result.decision.reasons
    assert result.needs_human_review


def test_explicit_high_risk_signal_reaches_pro_branch(codebook, fake_provider) -> None:
    policy = DMMPolicy(cold_start_requests=0, allow_quality_escalation=False)
    pipeline = build_pipeline(
        "paper3", codebook=codebook, provider=fake_provider, policy_override=policy
    )
    result = pipeline.interpret(
        InterpretationRequest(
            "wh", active_persona_ids=("ng-en-v1",), risk_score=0.9
        )
    )
    assert result.decision.used_route is Route.PRO
    assert "high_risk" in result.decision.reasons


def test_irrelevant_history_does_not_resolve_polysemy(codebook, fake_provider) -> None:
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    result = pipeline.interpret(
        InterpretationRequest(
            "I beg",
            active_persona_ids=("ng-en-v1",),
            conversation_context=("the weather is warm",),
        )
    )
    assert result.decision.used_route is Route.PRO
    assert "missing_disambiguating_context" in result.decision.reasons


def test_successful_pro_call_resets_consecutive_failure_counter(
    codebook, fake_provider
) -> None:
    policy = DMMPolicy(
        cold_start_requests=0,
        circuit_breaker_failures=2,
        allow_quality_escalation=False,
    )
    pipeline = build_pipeline(
        "paper3", codebook=codebook, provider=fake_provider, policy_override=policy
    )
    request = InterpretationRequest("I beg", active_persona_ids=("ng-en-v1",))
    fake_provider.queue_error("gemini-2.5-pro", ProviderServiceError("first"))
    pipeline.interpret(request)
    assert pipeline.interpret(request).decision.used_route is Route.PRO
    fake_provider.queue_error("gemini-2.5-pro", ProviderServiceError("second"))
    pipeline.interpret(request)
    final = pipeline.interpret(request)
    assert final.decision.used_route is Route.PRO
    assert "pro_circuit_open" not in final.decision.reasons


def test_quota_failure_opens_pro_circuit_for_next_request(codebook, fake_provider) -> None:
    fake_provider.queue_error("gemini-2.5-pro", ProviderQuotaError("quota"))
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    first = pipeline.interpret(
        InterpretationRequest("I beg", active_persona_ids=("ng-en-v1",))
    )
    assert first.decision.fallback_reason == "quota"
    fake_provider.calls.clear()
    second = pipeline.interpret(
        InterpretationRequest("I beg", active_persona_ids=("ng-en-v1",))
    )
    assert second.decision.used_route is Route.FLASH
    assert "pro_circuit_open" in second.decision.reasons
    assert [call["model_id"] for call in fake_provider.calls] == ["gemini-2.5-flash"]


def test_flash_failure_returns_explicit_human_review(codebook, fake_provider) -> None:
    fake_provider.queue_error("gemini-2.5-flash", ProviderServiceError("unavailable"))
    policy = DMMPolicy(
        pro_ambiguity_threshold=0.58,
        estimated_pro_call_cost_usd=1.0,
        allow_quality_escalation=False,
    )
    pipeline = build_pipeline(
        "paper3", codebook=codebook, provider=fake_provider, policy_override=policy
    )
    result = pipeline.interpret(
        InterpretationRequest(
            "I beg", active_persona_ids=("ng-en-v1",), cost_budget_usd=0.01
        )
    )
    assert result.decision.used_route is Route.HUMAN_REVIEW
    assert result.needs_human_review
    assert result.candidates == ()
    assert result.model_calls[0].status == "failed"


def test_invalid_structured_response_fails_visibly(codebook, fake_provider) -> None:
    fake_provider.responses["gemini-2.5-pro"] = {"candidates": [{"dialect": "NgE"}]}
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    with pytest.raises(DMMExecutionError, match="schema validation"):
        pipeline.interpret(
            InterpretationRequest("I beg", active_persona_ids=("ng-en-v1",))
        )


def test_model_candidate_entry_id_must_be_grounded_in_retrieval(
    codebook, fake_provider
) -> None:
    fake_provider.responses["gemini-2.5-flash"] = {
        "candidates": [candidate_payload(entry_id="unretrieved-entry")]
    }
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    with pytest.raises(DMMExecutionError, match="schema validation"):
        pipeline.interpret(
            InterpretationRequest("wh", active_persona_ids=("ng-en-v1",))
        )


def test_empty_acoustic_object_does_not_satisfy_paper5_evidence(
    codebook, fake_provider
) -> None:
    policy = DMMPolicy(cold_start_requests=0, allow_quality_escalation=False)
    pipeline = build_pipeline(
        "paper5", codebook=codebook, provider=fake_provider, policy_override=policy
    )
    result = pipeline.interpret(
        InterpretationRequest(
            "wahala",
            active_persona_ids=("ng-en-v1",),
            acoustic_affect=AcousticAffect(),
        )
    )
    assert result.needs_human_review


def test_quality_escalation_authentication_failure_is_not_hidden(
    codebook, fake_provider
) -> None:
    fake_provider.responses["gemini-2.5-flash"] = {
        "candidates": [candidate_payload(confidence=0.2)]
    }
    fake_provider.queue_error("gemini-2.5-pro", ProviderAuthenticationError("no pro access"))
    pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
    with pytest.raises(DMMExecutionError, match="authentication"):
        pipeline.interpret(
            InterpretationRequest("wh", active_persona_ids=("ng-en-v1",))
        )


def test_paper5_high_ambiguity_still_uses_pro_after_request_50(codebook, fake_provider) -> None:
    policy = DMMPolicy(
        version="test-paper5",
        local_threshold=0.80,
        local_margin=0.05,
        pro_ambiguity_threshold=0.58,
        cold_start_requests=50,
        allow_quality_escalation=False,
    )
    pipeline = build_pipeline(
        "paper5", codebook=codebook, provider=fake_provider, policy_override=policy
    )
    routine = InterpretationRequest(
        "wh",
        active_persona_ids=("ng-en-v1",),
        supplied_tone="Neutral",
        acoustic_affect=None,
    )
    for _ in range(50):
        pipeline.interpret(routine)
    fake_provider.calls.clear()
    result = pipeline.interpret(
        InterpretationRequest(
            "I beg",
            active_persona_ids=("ng-en-v1",),
            acoustic_affect=None,
        )
    )
    assert result.decision.used_route is Route.PRO
    assert not result.decision.cold_start
    assert fake_provider.calls[0]["model_id"] == "gemini-2.5-pro"


def test_counter_is_thread_safe(codebook, fake_provider) -> None:
    policy = DMMPolicy(cold_start_requests=0, allow_quality_escalation=False)
    pipeline = build_pipeline(
        "paper3", codebook=codebook, provider=fake_provider, policy_override=policy
    )

    def run(index: int) -> None:
        pipeline.interpret(
            InterpretationRequest(
                f"unseen routine wording {index}",
                active_persona_ids=("ng-en-v1",),
                supplied_context="explicit context",
                latency_budget_ms=100,
            )
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(40)))
    assert pipeline.dmm.request_count("ng-en-v1") == 40


def test_wire_payload_and_route_latency_are_measured_separately(codebook, fake_provider) -> None:
    pipeline = build_pipeline("paper4", codebook=codebook, provider=fake_provider)
    result = pipeline.interpret(
        InterpretationRequest("unseen phrase", latency_budget_ms=2000),
        raw_audio_bytes=96 * 1024,
        network_profile=NetworkProfile("4g-emulated", 10_000, 20_000, 45),
    )
    assert result.payload_bytes > 0
    assert result.timing.estimated_uplink_transfer_ms > 0
    assert result.timing.observed_api_round_trip_ms == 12.5
    assert result.timing.end_to_end_ms >= result.timing.serialization_ms
    assert json.loads(fake_provider.calls[0]["prompt"])["utterance"] == "unseen phrase"
