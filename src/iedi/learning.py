from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .codebook import canonical_json
from .schemas import CandidateInterpretation, CodebookEntry, utc_now


@dataclass(frozen=True)
class PreferenceExample:
    request_id: str
    prompt: str
    chosen: Mapping[str, Any]
    rejected: tuple[Mapping[str, Any], ...]
    corrected_entry_id: str | None
    actor_id: str
    dataset_version: str
    example_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_feedback(
        cls,
        *,
        request_id: str,
        prompt: str,
        chosen: CandidateInterpretation,
        rejected: tuple[CandidateInterpretation, ...],
        corrected_entry: CodebookEntry | None,
        actor_id: str,
        dataset_version: str,
    ) -> "PreferenceExample":
        return cls(
            request_id=request_id,
            prompt=prompt,
            chosen=asdict(chosen),
            rejected=tuple(asdict(candidate) for candidate in rejected),
            corrected_entry_id=corrected_entry.entry_id if corrected_entry else None,
            actor_id=actor_id,
            dataset_version=dataset_version,
        )


class PreferenceDataset:
    """Tamper-evident preference export; collecting it is not itself RLHF."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._head_hash = "0" * 64
        self._example_hashes: dict[str, str] = {}
        if self.path.exists():
            self.verify()

    def append(self, example: PreferenceExample) -> str:
        body = canonical_json(asdict(example))
        digest = hashlib.sha256(body).hexdigest()
        with self._lock:
            existing = self._example_hashes.get(example.example_id)
            if existing is not None:
                if existing != digest:
                    raise ValueError("preference example ID was reused with different content")
                return existing
            unsigned_record = {
                "example_sha256": digest,
                "example": asdict(example),
                "parent_hash": self._head_hash,
            }
            record_hash = hashlib.sha256(canonical_json(unsigned_record)).hexdigest()
            record = {**unsigned_record, "record_hash": record_hash}
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._head_hash = record_hash
            self._example_hashes[example.example_id] = digest
        return digest

    def verify(self) -> bool:
        expected_parent = "0" * 64
        examples: dict[str, str] = {}
        if not self.path.exists():
            self._head_hash = expected_parent
            self._example_hashes = examples
            return True
        with self._lock, self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_hash = str(record.pop("record_hash", ""))
                if record.get("parent_hash") != expected_parent:
                    raise ValueError(f"broken preference parent hash at line {line_number}")
                if hashlib.sha256(canonical_json(record)).hexdigest() != record_hash:
                    raise ValueError(f"tampered preference record at line {line_number}")
                example = record.get("example", {})
                digest = hashlib.sha256(canonical_json(example)).hexdigest()
                if digest != record.get("example_sha256"):
                    raise ValueError(f"tampered preference example at line {line_number}")
                example_id = str(example.get("example_id", ""))
                if not example_id or example_id in examples:
                    raise ValueError(f"duplicate/empty preference example ID at line {line_number}")
                examples[example_id] = digest
                expected_parent = record_hash
        self._head_hash = expected_parent
        self._example_hashes = examples
        return True

    @property
    def file_sha256(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ModelManifest:
    base_model: str
    adapter_or_checkpoint_uri: str
    training_dataset_hash: str
    training_run_id: str
    training_method: str
    evaluation_run_id: str
    deployed_model_version: str
    promoted_at: str
    checkpoint_sha256: str
    training_log_sha256: str
    evaluation_artifact_sha256: str
    deployment_receipt_sha256: str

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError("model manifest must be an object")
        return cls(**{field_name: str(raw.get(field_name, "")) for field_name in cls.__dataclass_fields__})

    def validate(self) -> None:
        missing = [
            field_name
            for field_name, value in asdict(self).items()
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(f"incomplete model manifest: {', '.join(missing)}")
        for field_name in (
            "training_dataset_hash",
            "checkpoint_sha256",
            "training_log_sha256",
            "evaluation_artifact_sha256",
            "deployment_receipt_sha256",
        ):
            if not re.fullmatch(r"[0-9a-fA-F]{64}", getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a SHA-256 digest")

    @property
    def declares_fine_tuning_method(self) -> bool:
        self.validate()
        return self.training_method.casefold() in {
            "sft",
            "lora-sft",
            "dpo",
            "rlhf-ppo",
            "rlhf",
        }

    @property
    def declares_rlhf_method(self) -> bool:
        self.validate()
        return self.training_method.casefold() in {"rlhf-ppo", "rlhf"}


def verify_training_trace(
    manifest: ModelManifest,
    *,
    preference_dataset: PreferenceDataset,
    require_rlhf: bool,
    checkpoint_path: str | Path,
    training_log_path: str | Path,
    evaluation_artifact_path: str | Path,
    deployment_receipt_path: str | Path,
) -> None:
    manifest.validate()
    preference_dataset.verify()
    if manifest.training_dataset_hash != preference_dataset.file_sha256:
        raise ValueError("model manifest does not match the preference dataset artifact")
    if require_rlhf and not manifest.declares_rlhf_method:
        raise ValueError(
            "the training artifact is not an RLHF/PPO run; describe it as SFT or preference optimization"
        )
    if not require_rlhf and not manifest.declares_fine_tuning_method:
        raise ValueError("training method does not substantiate a fine-tuned-model claim")
    artifacts = {
        "checkpoint_sha256": Path(checkpoint_path),
        "training_log_sha256": Path(training_log_path),
        "evaluation_artifact_sha256": Path(evaluation_artifact_path),
        "deployment_receipt_sha256": Path(deployment_receipt_path),
    }
    for field_name, path in artifacts.items():
        if not path.is_file():
            raise ValueError(f"missing training evidence artifact: {path}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != getattr(manifest, field_name):
            raise ValueError(f"{field_name} does not match artifact: {path}")
