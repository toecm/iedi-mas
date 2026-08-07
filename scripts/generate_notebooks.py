from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    (NOTEBOOKS / name).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


BOOTSTRAP = """from pathlib import Path
import os
import sys

search_roots = (Path.cwd(), *Path.cwd().parents, Path("/content/iedi-mas"))
ROOT = next((path for path in search_roots if (path / "src" / "iedi").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Repository not found. Clone it and install with: pip install -e .[gemini]")
sys.path.insert(0, str(ROOT / "src"))

from iedi.codebook import Codebook
from iedi.providers import GoogleGenAIProvider, OfflineFixtureProvider
from iedi.pipeline import build_pipeline
from iedi.schemas import InterpretationRequest

codebook = Codebook.from_json(ROOT / "data" / "codebook.demo.json")
# OfflineFixtureProvider only echoes reviewed evidence; it is never empirical evidence.
# Set IEDI_LIVE_GEMINI=1 and GEMINI_API_KEY to exercise the real 2.5 Flash/Pro adapter.
LIVE_GEMINI = os.getenv("IEDI_LIVE_GEMINI") == "1"
provider = GoogleGenAIProvider() if LIVE_GEMINI else OfflineFixtureProvider()
print("provider:", "live Gemini" if LIVE_GEMINI else "offline schema fixture")
"""


def main() -> None:
    write_notebook(
        "Paper_2_IUUY.ipynb",
        [
            markdown(
                "# Paper 2 — IUUY hybrid\n\n"
                "Implements ≥80% top-two retrieval and explicit unmatched/ambiguous fallback. "
                "A fine-tuned/RLHF claim additionally requires hashed checkpoint, training-log, evaluation, deployment, and preference artifacts; this notebook does not invent them. "
                "For Colab, clone the repository and run `%pip install -e .[gemini,audio,diarization,evaluation]` first.\n"
            ),
            code(BOOTSTRAP + '\npipeline = build_pipeline("paper2", codebook=codebook, provider=provider, config_path=ROOT / "configs" / "paper2.json")\n'),
            code(
                "request = InterpretationRequest(\n"
                "    utterance=\"How far?\",\n"
                "    speaker_id=\"SPEAKER_00\",\n"
                ")\n"
                "result = pipeline.interpret(request)\n"
                "result.to_dict()\n"
            ),
            markdown(
                "For audio, construct `InputAgent` with `WhisperASR` and `PyannoteDiarizer`, "
                "using `require_diarization=True`. Missing diarization then fails visibly instead of returning a constant speaker label.\n"
            ),
        ],
    )

    write_notebook(
        "Paper_3_IEDI_MAS.ipynb",
        [
            markdown(
                "# Paper 3 — Persona-aware IEDI DMM\n\n"
                "The full schema-approved demonstration persona is loaded before interpretation and the centralized DMM chooses Local, Flash or Pro from ambiguity evidence. Native-speaker validation remains external evidence.\n"
            ),
            code(BOOTSTRAP + '\npipeline = build_pipeline("paper3", codebook=codebook, provider=provider, config_path=ROOT / "configs" / "paper3.json")\n'),
            code(
                "ambiguous = InterpretationRequest(\n"
                "    utterance=\"I beg\",\n"
                "    active_persona_ids=(\"ng-en-v1\",),\n"
                ")\n"
                "result = pipeline.interpret(ambiguous)\n"
                "result.to_dict()\n"
            ),
            code(
                "context_resolved = InterpretationRequest(\n"
                "    utterance=\"I beg\",\n"
                "    active_persona_ids=(\"ng-en-v1\",),\n"
                "    supplied_tone=\"Casual\",\n"
                "    supplied_context=\"discourse marker used to soften commands\",\n"
                ")\n"
                "pipeline.interpret(context_resolved).to_dict()\n"
            ),
            markdown(
                "The paper asks for three senses but supplies two. Until a qualified annotator approves a third, the result is marked for human review rather than fabricating cultural ground truth.\n"
            ),
        ],
    )

    write_notebook(
        "Paper_4_JCCI.ipynb",
        [
            markdown(
                "# Paper 4 — Edge/cloud CA-IEDI measurement\n\n"
                "Measures deterministic UTF-8 wire bytes and keeps estimated transfer time separate from observed gateway round-trip time. The default loopback is a protocol dry run, not a 4G experiment.\n"
            ),
            code(BOOTSTRAP + '\npipeline = build_pipeline("paper4", codebook=codebook, provider=provider, config_path=ROOT / "configs" / "paper4.json")\n'),
            code(
                "from iedi.edge import NetworkProfile\n"
                "from iedi.transport import (\n"
                "    CloudInterpretationService, EdgeInterpretationClient,\n"
                "    LoopbackEdgeCloudTransport,\n"
                ")\n\n"
                "network = NetworkProfile(\n"
                "    name=\"documented-4g-emulation\",\n"
                "    uplink_kbps=10_000,\n"
                "    downlink_kbps=20_000,\n"
                "    base_rtt_ms=45,\n"
                "    emulator=\"replace with actual emulator/hardware metadata\",\n"
                ")\n"
                "# Loopback exercises the exact protocol. Replace it with\n"
                "# HttpEdgeCloudTransport for a separately deployed cloud service.\n"
                "cloud = CloudInterpretationService(pipeline)\n"
                "edge = EdgeInterpretationClient(LoopbackEdgeCloudTransport(cloud))\n"
                "observation = edge.interpret(\n"
                "    InterpretationRequest(\"An unresolved phrase\"),\n"
                "    raw_audio_bytes=96 * 1024,  # hypothetical baseline, not a measured file\n"
                "    network_profile=network,\n"
                ")\n"
                "{\n"
                "    \"route\": observation.result[\"decision\"][\"used_route\"],\n"
                "    \"wire_bytes\": observation.wire_payload_bytes,\n"
                "    \"reduction\": observation.payload_reduction,\n"
                "    \"estimated_transfer_ms\": observation.estimated_transfer_ms,\n"
                "    \"observed_gateway_rtt_ms\": observation.observed_gateway_round_trip_ms,\n"
                "}\n"
            ),
            markdown(
                "Run repeated trials and retain raw records before comparing with the paper. Target byte/latency values are not embedded as successful assertions.\n"
            ),
        ],
    )

    write_notebook(
        "Paper_5_CA_IEDI.ipynb",
        [
            markdown(
                "# Paper 5 — CA-IEDI actual DMM and trust gate\n\n"
                "Uses the ambiguity/cold-start DMM, bounded four-agent runtime, acoustic-evidence interface, and fail-closed provenance state machine. The default provenance/authorizer are explicit offline fixtures, not a deployed DAO.\n"
            ),
            code(BOOTSTRAP + '\npipeline = build_pipeline("paper5", codebook=codebook, provider=provider, config_path=ROOT / "configs" / "paper5.json")\n'),
            code(
                "import tempfile\n"
                "from dataclasses import replace\n\n"
                "from iedi.agents import InterpretationAgent, MultiAgentRuntime, TrustAgent, UXAgent\n"
                "from iedi.audio import InputAgent, LibrosaAcousticExtractor, PyannoteDiarizer, WhisperASR\n"
                "from iedi.provenance import InMemoryChain, InMemoryIPFS\n"
                "from iedi.schemas import FeedbackAction, FeedbackEvent\n"
                "from iedi.trust import (\n"
                "    HashChainAuditLog, LocalAppendOnlyEntryStore,\n"
                "    StaticValidatorAuthorizer, TrustGate, interpretation_result_sha256,\n"
                ")\n\n"
                "# Ephemeral fixtures exercise state transitions without claiming public persistence.\n"
                "runtime_tmp = tempfile.TemporaryDirectory()\n"
                "runtime_dir = Path(runtime_tmp.name)\n"
                "trust_gate = TrustGate.for_pipeline(\n"
                "    pipeline,\n"
                "    store=LocalAppendOnlyEntryStore(\n"
                "        runtime_dir / \"entries.jsonl\", baseline_entries=pipeline.codebook.entries\n"
                "    ),\n"
                "    audit_log=HashChainAuditLog(runtime_dir / \"audit.jsonl\"),\n"
                "    authorizer=StaticValidatorAuthorizer({\"demo-validator\"}),\n"
                "    ipfs=InMemoryIPFS(),\n"
                "    chain=InMemoryChain(),\n"
                ")\n"
                "runtime = MultiAgentRuntime(\n"
                "    input_agent=InputAgent(\n"
                "        asr=WhisperASR(), diarizer=PyannoteDiarizer(),\n"
                "        acoustic_extractor=LibrosaAcousticExtractor(), require_diarization=True,\n"
                "    ),\n"
                "    interpretation_agent=InterpretationAgent(pipeline),\n"
                "    trust_agent=TrustAgent(trust_gate),\n"
                "    ux_agent=UXAgent(),\n"
                ")\n"
            ),
            code(
                "async def run_text_feedback_demo():\n"
                "    # No acoustic evidence is fabricated: Paper 5 correctly marks review required.\n"
                "    request = InterpretationRequest(\n"
                "        utterance=\"wahala\", active_persona_ids=(\"ng-en-v1\",),\n"
                "        request_id=\"paper5-demo-request\",\n"
                "    )\n"
                "    async with runtime:\n"
                "        presented = await runtime.interpret(request)\n"
                "        original = pipeline.codebook.get_entry(\"ng-wahala-1\")\n"
                "        correction = replace(\n"
                "            original, entry_id=\"ng-wahala-2-demo\", version=2,\n"
                "            supersedes_entry_id=original.entry_id,\n"
                "            universal_gloss=\"human-reviewed demonstration correction\",\n"
                "            reviewed_by=(\"demo-validator\",),\n"
                "            source_type=\"offline-demonstration\",\n"
                "            created_at=\"2026-08-07T00:00:00+00:00\",\n"
                "        )\n"
                "        event = FeedbackEvent(\n"
                "            request_id=request.request_id, action=FeedbackAction.ACCEPT,\n"
                "            actor_id=\"demo-validator\", candidate=presented.result.candidates[0],\n"
                "            corrected_entry=correction,\n"
                "            source_result_sha256=interpretation_result_sha256(presented.result),\n"
                "        )\n"
                "        receipt = await runtime.submit_feedback(event)\n"
                "        updated = await runtime.interpret(request)\n"
                "    return presented, receipt, updated\n\n"
                "before, receipt, after = await run_text_feedback_demo()\n"
                "{\n"
                "    \"route\": before.view[\"route\"],\n"
                "    \"missing_affect_requires_review\": before.result.needs_human_review,\n"
                "    \"feedback_state\": receipt.state.value,\n"
                "    \"indexed_version\": receipt.indexed_dataset_version,\n"
                "    \"updated_entry\": after.result.candidates[0].entry_id,\n"
                "}\n"
            ),
            markdown(
                "For real audio, install the audio/diarization extras and call `runtime.interpret_audio(...)`; that path executes ASR → diarization → acoustic extraction → interpretation → UX. In-memory provenance and a static allow-list prove state-machine behavior only. They are not evidence of IPFS durability, blockchain finality, authenticated community governance, or poisoning prevention.\n"
            ),
        ],
    )


if __name__ == "__main__":
    main()
