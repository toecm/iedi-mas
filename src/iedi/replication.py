from __future__ import annotations

import hashlib
import io
import os
from typing import Any, Mapping, Protocol

from .codebook import canonical_json


class RecordPublisher(Protocol):
    def publish(
        self,
        *,
        payload_sha256: str,
        payload: Mapping[str, Any],
        chain_reference: str | None,
    ) -> str: ...


class HuggingFaceRecordPublisher:
    """Publishes record-content-addressed files and returns the resulting commit ID."""

    def __init__(
        self,
        *,
        repo_id: str,
        token: str | None = None,
        revision: str = "main",
        api: Any | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.token = token or os.getenv("HF_TOKEN")
        self.revision = revision
        if api is not None:
            self.api = api
            return
        if not self.token:
            raise ValueError("HF_TOKEN is required")
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the hf extra: pip install -e .[hf]") from exc
        self.api = HfApi(token=self.token)

    def publish(
        self,
        *,
        payload_sha256: str,
        payload: Mapping[str, Any],
        chain_reference: str | None,
    ) -> str:
        observed_payload_hash = hashlib.sha256(canonical_json(dict(payload))).hexdigest()
        if observed_payload_hash != payload_sha256:
            raise ValueError("payload_sha256 does not match the canonical payload")
        body = canonical_json(
            {
                "payload": dict(payload),
                "payload_sha256": payload_sha256,
                "chain_reference": chain_reference,
            }
        )
        record_sha256 = hashlib.sha256(body).hexdigest()
        commit = self.api.upload_file(
            path_or_fileobj=io.BytesIO(body),
            path_in_repo=f"records/{payload_sha256}/{record_sha256}.json",
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            commit_message=f"Add approved IEDID record {payload_sha256[:12]}",
        )
        reference = getattr(commit, "oid", None) or getattr(commit, "commit_url", None)
        if not reference:
            raise RuntimeError("Hugging Face upload returned no immutable commit reference")
        return str(reference)
