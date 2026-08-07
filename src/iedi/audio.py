from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Protocol, Sequence

from .schemas import AcousticAffect, TranscriptSegment


@dataclass(frozen=True)
class ASRSegment:
    text: str
    start_s: float
    end_s: float
    confidence: float | None = None


class ASRBackend(Protocol):
    def transcribe(self, audio_path: str | Path) -> Sequence[ASRSegment]: ...


class SpeakerDiarizer(Protocol):
    def assign_speakers(
        self, audio_path: str | Path, segments: Sequence[ASRSegment]
    ) -> Sequence[str]: ...


class AcousticFeatureExtractor(Protocol):
    def extract(self, audio_path: str | Path) -> AcousticAffect: ...


@dataclass(frozen=True)
class AudioInputResult:
    segments: tuple[TranscriptSegment, ...]
    audio_sha256: str
    raw_audio_bytes: int
    asr_latency_ms: float
    acoustic_affect: AcousticAffect | None


class InputAgent:
    """Edge-side ASR, diarization and acoustic extraction.

    Speaker IDs are never replaced with a fabricated constant. A paper profile that
    claims diarization should construct this agent with require_diarization=True.
    """

    def __init__(
        self,
        *,
        asr: ASRBackend,
        diarizer: SpeakerDiarizer | None = None,
        acoustic_extractor: AcousticFeatureExtractor | None = None,
        require_diarization: bool = False,
    ) -> None:
        self.asr = asr
        self.diarizer = diarizer
        self.acoustic_extractor = acoustic_extractor
        self.require_diarization = require_diarization

    def process_audio(self, audio_path: str | Path) -> AudioInputResult:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        if not raw:
            raise ValueError("audio file is empty")

        started = perf_counter()
        asr_segments = tuple(self.asr.transcribe(path))
        asr_latency_ms = (perf_counter() - started) * 1000.0
        if not asr_segments:
            raise ValueError("ASR returned no transcript segments")

        if self.diarizer is None:
            if self.require_diarization:
                raise RuntimeError("speaker diarization is required but no diarizer is configured")
            speaker_ids: Sequence[str | None] = [None] * len(asr_segments)
        else:
            assigned = tuple(self.diarizer.assign_speakers(path, asr_segments))
            if len(assigned) != len(asr_segments):
                raise ValueError("diarizer must return one speaker ID per ASR segment")
            if any(not speaker.strip() for speaker in assigned):
                raise ValueError("diarizer returned an empty speaker ID")
            speaker_ids = assigned

        segments = tuple(
            TranscriptSegment(
                text=segment.text,
                start_s=segment.start_s,
                end_s=segment.end_s,
                speaker_id=speaker_id,
                asr_confidence=segment.confidence,
            )
            for segment, speaker_id in zip(asr_segments, speaker_ids)
        )
        affect = self.acoustic_extractor.extract(path) if self.acoustic_extractor else None
        return AudioInputResult(
            segments=segments,
            audio_sha256=hashlib.sha256(raw).hexdigest(),
            raw_audio_bytes=len(raw),
            asr_latency_ms=asr_latency_ms,
            acoustic_affect=affect,
        )


class WhisperASR:
    def __init__(
        self,
        *,
        model_name: str = "turbo",
        device: str | None = None,
        initial_prompt: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.initial_prompt = initial_prompt
        self._model = model

    def _load(self) -> Any:
        if self._model is None:
            try:
                import whisper
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("install the 'audio' extra: pip install -e .[audio]") from exc
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe(self, audio_path: str | Path) -> Sequence[ASRSegment]:
        result = self._load().transcribe(
            str(audio_path),
            initial_prompt=self.initial_prompt,
            verbose=False,
        )
        segments: list[ASRSegment] = []
        for item in result.get("segments", []):
            average_log_probability = item.get("avg_logprob")
            confidence = None
            if average_log_probability is not None:
                confidence = min(max(math.exp(float(average_log_probability)), 0.0), 1.0)
            segments.append(
                ASRSegment(
                    text=str(item["text"]).strip(),
                    start_s=float(item["start"]),
                    end_s=float(item["end"]),
                    confidence=confidence,
                )
            )
        return segments


class PyannoteDiarizer:
    def __init__(
        self,
        *,
        model_id: str = "pyannote/speaker-diarization-3.1",
        token: str | None = None,
        pipeline: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.token = token
        self._pipeline = pipeline

    def _load(self) -> Any:
        if self._pipeline is None:
            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "install the 'diarization' extra: pip install -e .[diarization]"
                ) from exc
            self._pipeline = Pipeline.from_pretrained(self.model_id, token=self.token)
        return self._pipeline

    def assign_speakers(
        self, audio_path: str | Path, segments: Sequence[ASRSegment]
    ) -> Sequence[str]:
        diarization = self._load()(str(audio_path))
        turns: list[tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((float(turn.start), float(turn.end), str(speaker)))
        if not turns:
            raise ValueError("diarization returned no speaker turns")

        assigned: list[str] = []
        for segment in segments:
            midpoint = (segment.start_s + segment.end_s) / 2.0
            containing = [turn for turn in turns if turn[0] <= midpoint <= turn[1]]
            if containing:
                assigned.append(containing[0][2])
                continue
            nearest = min(turns, key=lambda turn: min(abs(midpoint - turn[0]), abs(midpoint - turn[1])))
            assigned.append(nearest[2])
        return assigned


class LibrosaAcousticExtractor:
    """Extracts versioned local acoustic evidence; labeling remains an explicit adapter."""

    def __init__(
        self,
        *,
        labeler: Callable[[dict[str, float]], tuple[str, float]] | None = None,
        extractor_version: str = "librosa-basic-v1",
    ) -> None:
        self.labeler = labeler
        self.extractor_version = extractor_version

    def extract(self, audio_path: str | Path) -> AcousticAffect:
        try:
            import librosa
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'audio' extra: pip install -e .[audio]") from exc

        waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
        if waveform.size == 0:
            raise ValueError("cannot extract acoustic features from empty waveform")
        rms = librosa.feature.rms(y=waveform)[0]
        zero_crossing = librosa.feature.zero_crossing_rate(waveform)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]
        pitch = librosa.yin(waveform, fmin=65, fmax=500, sr=sample_rate)
        finite_pitch = pitch[np.isfinite(pitch)]
        features = {
            "duration_s": float(len(waveform) / sample_rate),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "zero_crossing_rate_mean": float(np.mean(zero_crossing)),
            "spectral_centroid_mean_hz": float(np.mean(spectral_centroid)),
            "pitch_median_hz": float(np.median(finite_pitch)) if finite_pitch.size else 0.0,
            "pitch_std_hz": float(np.std(finite_pitch)) if finite_pitch.size else 0.0,
        }
        label = None
        confidence = None
        if self.labeler is not None:
            label, confidence = self.labeler(features)
        return AcousticAffect(
            label=label,
            confidence=confidence,
            extractor_id="librosa-basic-acoustic-features",
            extractor_version=self.extractor_version,
            features=features,
        )
