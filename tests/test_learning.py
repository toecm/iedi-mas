from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from iedi.learning import (
    ModelManifest,
    PreferenceDataset,
    PreferenceExample,
    verify_training_trace,
)


def test_collecting_preferences_is_not_automatically_rlhf(tmp_path: Path) -> None:
    dataset = PreferenceDataset(tmp_path / "preferences.jsonl")
    dataset.append(
        PreferenceExample(
            request_id="r1",
            prompt="I beg",
            chosen={"clarification": "please"},
            rejected=({"clarification": "literal begging"},),
            corrected_entry_id="ng-i-beg-please",
            actor_id="validator",
            dataset_version="v1",
        )
    )
    artifacts = {}
    for name in ("checkpoint", "training", "evaluation", "deployment"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(f"verified-{name}".encode())
        artifacts[name] = path
    manifest = ModelManifest(
        base_model="base",
        adapter_or_checkpoint_uri="hf://checkpoint",
        training_dataset_hash=dataset.file_sha256,
        training_run_id="run-1",
        training_method="sft",
        evaluation_run_id="eval-1",
        deployed_model_version="model-v2",
        promoted_at="2026-08-07T00:00:00Z",
        checkpoint_sha256=hashlib.sha256(artifacts["checkpoint"].read_bytes()).hexdigest(),
        training_log_sha256=hashlib.sha256(artifacts["training"].read_bytes()).hexdigest(),
        evaluation_artifact_sha256=hashlib.sha256(
            artifacts["evaluation"].read_bytes()
        ).hexdigest(),
        deployment_receipt_sha256=hashlib.sha256(
            artifacts["deployment"].read_bytes()
        ).hexdigest(),
    )
    evidence = {
        "checkpoint_path": artifacts["checkpoint"],
        "training_log_path": artifacts["training"],
        "evaluation_artifact_path": artifacts["evaluation"],
        "deployment_receipt_path": artifacts["deployment"],
    }
    verify_training_trace(
        manifest, preference_dataset=dataset, require_rlhf=False, **evidence
    )
    with pytest.raises(ValueError, match="not an RLHF"):
        verify_training_trace(
            manifest, preference_dataset=dataset, require_rlhf=True, **evidence
        )


def test_preference_dataset_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "preferences.jsonl"
    dataset = PreferenceDataset(path)
    dataset.append(
        PreferenceExample(
            request_id="r1",
            prompt="I beg",
            chosen={"clarification": "please"},
            rejected=(),
            corrected_entry_id=None,
            actor_id="validator",
            dataset_version="v1",
        )
    )
    path.write_text(path.read_text(encoding="utf-8").replace("please", "tampered"), encoding="utf-8")
    with pytest.raises(ValueError, match="tampered"):
        PreferenceDataset(path)
