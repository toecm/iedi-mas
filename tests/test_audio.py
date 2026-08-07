from __future__ import annotations

from pathlib import Path

import pytest

from iedi.audio import ASRSegment, InputAgent


class FakeASR:
    def transcribe(self, audio_path: Path):
        return (
            ASRSegment("Hello", 0.0, 1.0, 0.9),
            ASRSegment("How far?", 1.0, 2.0, 0.8),
        )


class FakeDiarizer:
    def assign_speakers(self, audio_path: Path, segments):
        return ("SPEAKER_00", "SPEAKER_01")


def test_input_agent_retains_real_speaker_assignments(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"not-real-audio-needed-by-fakes")
    result = InputAgent(
        asr=FakeASR(), diarizer=FakeDiarizer(), require_diarization=True
    ).process_audio(audio)
    assert [segment.speaker_id for segment in result.segments] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert result.segments[1].start_s == 1.0
    assert result.audio_sha256
    assert result.raw_audio_bytes == len(b"not-real-audio-needed-by-fakes")


def test_required_diarization_cannot_fall_back_to_constant_label(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"audio")
    with pytest.raises(RuntimeError, match="diarization is required"):
        InputAgent(asr=FakeASR(), require_diarization=True).process_audio(audio)
