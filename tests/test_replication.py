from __future__ import annotations

from dataclasses import dataclass
import hashlib

from iedi.codebook import canonical_json
from iedi.replication import HuggingFaceRecordPublisher


@dataclass
class Commit:
    oid: str = "commit-123"


class FakeHFAPI:
    def __init__(self) -> None:
        self.kwargs = None

    def upload_file(self, **kwargs):
        self.kwargs = kwargs
        return Commit()


def test_hf_publisher_uses_immutable_content_hash_path() -> None:
    api = FakeHFAPI()
    publisher = HuggingFaceRecordPublisher(repo_id="owner/iedid", api=api)
    payload = {"entry": {"text": "I beg"}}
    payload_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
    reference = publisher.publish(
        payload_sha256=payload_hash,
        payload=payload,
        chain_reference="tx-1",
    )
    assert reference == "commit-123"
    assert api.kwargs["path_in_repo"].startswith(f"records/{payload_hash}/")
    assert api.kwargs["path_in_repo"].endswith(".json")
    assert api.kwargs["repo_type"] == "dataset"
    assert b"tx-1" in api.kwargs["path_or_fileobj"].getvalue()
