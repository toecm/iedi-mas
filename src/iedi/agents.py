from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar

from .audio import AudioInputResult, InputAgent
from .edge import NetworkProfile
from .pipeline import IEDIPipeline
from .schemas import FeedbackEvent, InterpretationRequest, InterpretationResult
from .trust import TrustGate, TrustReceipt


@dataclass(frozen=True)
class UXInterpretation:
    result: InterpretationResult
    view: dict[str, Any]


T = TypeVar("T")


@dataclass
class _Envelope(Generic[T]):
    payload: T
    options: dict[str, Any]
    future: asyncio.Future


class InterpretationAgent:
    def __init__(self, pipeline: IEDIPipeline) -> None:
        self.pipeline = pipeline

    def process(self, request: InterpretationRequest, **options: Any) -> InterpretationResult:
        return self.pipeline.interpret(request, **options)


class TrustAgent:
    def __init__(self, gate: TrustGate) -> None:
        self.gate = gate

    def process(self, event: FeedbackEvent) -> TrustReceipt:
        return self.gate.process(event)


class UXAgent:
    """Presentation boundary; UI frameworks consume this stable view model."""

    def present(self, result: InterpretationResult) -> UXInterpretation:
        return UXInterpretation(
            result=result,
            view={
                "request_id": result.request_id,
                "route": result.decision.used_route.value,
                "model": result.decision.model_id,
                "ambiguity_score": result.decision.ambiguity_score,
                "route_reasons": list(result.decision.reasons),
                "fallback_reason": result.decision.fallback_reason,
                "payload_bytes": result.payload_bytes,
                "needs_human_review": result.needs_human_review,
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "dialect": candidate.dialect,
                        "clarification": candidate.clarification,
                        "tone": candidate.tone_category,
                        "context": candidate.linguistic_context,
                        "pragmatics": candidate.pragmatic_analysis,
                        "confidence": candidate.confidence,
                    }
                    for candidate in result.candidates
                ],
            },
        )


class MultiAgentRuntime:
    """Typed in-process actor runtime for the four paper-named agents.

    It is deliberately described as in-process; deployment across services requires a
    transport adapter and is not implied by this class.
    """

    def __init__(
        self,
        *,
        input_agent: InputAgent,
        interpretation_agent: InterpretationAgent,
        trust_agent: TrustAgent,
        ux_agent: UXAgent | None = None,
        max_queue_size: int = 100,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self.input_agent = input_agent
        self.interpretation_agent = interpretation_agent
        self.trust_agent = trust_agent
        self.ux_agent = ux_agent or UXAgent()
        self._input_queue: asyncio.Queue[_Envelope[Path] | None] = asyncio.Queue(max_queue_size)
        self._interpretation_queue: asyncio.Queue[_Envelope[InterpretationRequest] | None] = (
            asyncio.Queue(max_queue_size)
        )
        self._ux_queue: asyncio.Queue[_Envelope[InterpretationResult] | None] = asyncio.Queue(
            max_queue_size
        )
        self._trust_queue: asyncio.Queue[_Envelope[FeedbackEvent] | None] = asyncio.Queue(
            max_queue_size
        )
        self._tasks: list[asyncio.Task] = []
        self._accepting = False

    async def __aenter__(self) -> "MultiAgentRuntime":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.stop()

    async def start(self) -> None:
        if self._tasks:
            return
        self._accepting = True
        self._tasks = [
            asyncio.create_task(self._input_loop(), name="AgentInput"),
            asyncio.create_task(self._interpretation_loop(), name="AgentInterpretation"),
            asyncio.create_task(self._ux_loop(), name="AgentUX"),
            asyncio.create_task(self._trust_loop(), name="AgentTrust"),
        ]

    async def stop(self) -> None:
        if not self._tasks:
            return
        self._accepting = False
        # Drain work in dependency order. Interpretation may enqueue UX work, so its
        # queue must finish before the UX sentinel is sent.
        await self._input_queue.join()
        await self._interpretation_queue.join()
        await self._ux_queue.join()
        await self._trust_queue.join()
        for queue in (
            self._input_queue,
            self._interpretation_queue,
            self._ux_queue,
            self._trust_queue,
        ):
            await queue.put(None)
        results = await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError("an agent worker terminated unexpectedly") from failures[0]

    async def process_audio(self, audio_path: str | Path) -> AudioInputResult:
        self._require_started()
        future = asyncio.get_running_loop().create_future()
        await self._input_queue.put(_Envelope(Path(audio_path), {}, future))
        return await future

    async def interpret_audio(
        self,
        audio_path: str | Path,
        *,
        active_persona_ids: tuple[str, ...],
        speaker_roles: Mapping[str, str] | None = None,
        supplied_context: str | None = None,
        network_profile: NetworkProfile | None = None,
    ) -> tuple[UXInterpretation, ...]:
        """Run the concrete Input → Interpretation → UX path for every ASR segment."""

        audio = await self.process_audio(audio_path)
        roles = speaker_roles or {}
        presented: list[UXInterpretation] = []
        for segment in audio.segments:
            presented.append(
                await self.interpret(
                    InterpretationRequest(
                        segment.text,
                        active_persona_ids=active_persona_ids,
                        speaker_id=segment.speaker_id,
                        speaker_role=roles.get(segment.speaker_id or ""),
                        supplied_context=supplied_context,
                        acoustic_affect=audio.acoustic_affect,
                        asr_confidence=segment.asr_confidence,
                    ),
                    raw_audio_bytes=audio.raw_audio_bytes,
                    edge_asr_ms=audio.asr_latency_ms,
                    network_profile=network_profile,
                )
            )
        return tuple(presented)

    async def interpret(
        self,
        request: InterpretationRequest,
        *,
        raw_audio_bytes: int | None = None,
        edge_asr_ms: float = 0.0,
        network_profile: NetworkProfile | None = None,
        task: str = "interpret",
    ) -> UXInterpretation:
        self._require_started()
        future = asyncio.get_running_loop().create_future()
        await self._interpretation_queue.put(
            _Envelope(
                request,
                {
                    "raw_audio_bytes": raw_audio_bytes,
                    "edge_asr_ms": edge_asr_ms,
                    "network_profile": network_profile,
                    "task": task,
                },
                future,
            )
        )
        return await future

    async def submit_feedback(self, event: FeedbackEvent) -> TrustReceipt:
        self._require_started()
        future = asyncio.get_running_loop().create_future()
        await self._trust_queue.put(_Envelope(event, {}, future))
        return await future

    def _require_started(self) -> None:
        if not self._tasks or not self._accepting:
            raise RuntimeError("start MultiAgentRuntime before submitting work")

    async def _input_loop(self) -> None:
        while True:
            envelope = await self._input_queue.get()
            try:
                if envelope is None:
                    return
                result = await asyncio.to_thread(
                    self.input_agent.process_audio, envelope.payload
                )
                _set_result_if_pending(envelope.future, result)
            except Exception as exc:
                _set_exception_if_pending(envelope.future, exc)
            finally:
                self._input_queue.task_done()

    async def _interpretation_loop(self) -> None:
        while True:
            envelope = await self._interpretation_queue.get()
            try:
                if envelope is None:
                    return
                result = await asyncio.to_thread(
                    self.interpretation_agent.process,
                    envelope.payload,
                    **envelope.options,
                )
                await self._ux_queue.put(_Envelope(result, {}, envelope.future))
            except Exception as exc:
                _set_exception_if_pending(envelope.future, exc)
            finally:
                self._interpretation_queue.task_done()

    async def _ux_loop(self) -> None:
        while True:
            envelope = await self._ux_queue.get()
            try:
                if envelope is None:
                    return
                _set_result_if_pending(
                    envelope.future, self.ux_agent.present(envelope.payload)
                )
            except Exception as exc:
                _set_exception_if_pending(envelope.future, exc)
            finally:
                self._ux_queue.task_done()

    async def _trust_loop(self) -> None:
        while True:
            envelope = await self._trust_queue.get()
            try:
                if envelope is None:
                    return
                result = await asyncio.to_thread(self.trust_agent.process, envelope.payload)
                _set_result_if_pending(envelope.future, result)
            except Exception as exc:
                _set_exception_if_pending(envelope.future, exc)
            finally:
                self._trust_queue.task_done()


def _set_result_if_pending(future: asyncio.Future, value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception_if_pending(future: asyncio.Future, error: Exception) -> None:
    if not future.done():
        future.set_exception(error)
