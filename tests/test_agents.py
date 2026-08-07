from __future__ import annotations

import asyncio
from pathlib import Path

from iedi.agents import InterpretationAgent, MultiAgentRuntime, TrustAgent, UXAgent
from iedi.audio import ASRSegment, InputAgent
from iedi.pipeline import build_pipeline
from iedi.provenance import InMemoryChain, InMemoryIPFS
from iedi.schemas import InterpretationRequest, Route
from iedi.trust import HashChainAuditLog, LocalAppendOnlyEntryStore, TrustGate
from iedi.trust import PipelineIndexUpdater, StaticValidatorAuthorizer


class FakeASR:
    def transcribe(self, audio_path):
        return (ASRSegment("wahala", 0, 1, 0.9),)


class FakeDiarizer:
    def assign_speakers(self, audio_path, segments):
        return ("SPEAKER_00",)


def test_four_agents_exchange_typed_messages(tmp_path, codebook, fake_provider) -> None:
    async def scenario():
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fixture")
        pipeline = build_pipeline("paper3", codebook=codebook, provider=fake_provider)
        gate = TrustGate(
            store=LocalAppendOnlyEntryStore(
                tmp_path / "entries.jsonl", baseline_entries=codebook.entries
            ),
            audit_log=HashChainAuditLog(tmp_path / "audit.jsonl"),
            authorizer=StaticValidatorAuthorizer({"demo-validator"}),
            ipfs=InMemoryIPFS(),
            chain=InMemoryChain(),
            index_updater=PipelineIndexUpdater(pipeline),
        )
        runtime = MultiAgentRuntime(
            input_agent=InputAgent(
                asr=FakeASR(), diarizer=FakeDiarizer(), require_diarization=True
            ),
            interpretation_agent=InterpretationAgent(pipeline),
            trust_agent=TrustAgent(gate),
            ux_agent=UXAgent(),
        )
        async with runtime:
            (presented,) = await runtime.interpret_audio(
                audio_path,
                active_persona_ids=("ng-en-v1",),
            )
        assert presented.result.decision.used_route is Route.LOCAL
        assert presented.view["route"] == "local"
        assert presented.view["candidates"][0]["clarification"]
        # Stop drains downstream UX work and the same runtime can be restarted.
        await runtime.start()
        restarted = await runtime.interpret(
            InterpretationRequest("wahala", active_persona_ids=("ng-en-v1",))
        )
        await runtime.stop()
        assert restarted.view["route"] == "local"

    asyncio.run(scenario())
