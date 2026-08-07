from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from iedi.provenance import InMemoryChain, InMemoryIPFS
from iedi.pipeline import build_pipeline
from iedi.schemas import CodebookEntry, FeedbackAction, FeedbackEvent, FeedbackState
from iedi.trust import (
    HashChainAuditLog,
    LocalAppendOnlyEntryStore,
    PipelineIndexUpdater,
    StaticValidatorAuthorizer,
    TrustGate,
    TrustGateError,
    TrustPolicy,
)


SOURCE_HASH = "a" * 64
PAYLOAD_HASH = "b" * 64


def entry(
    entry_id: str = "new-1",
    tone: str = "Friendly",
    *,
    reviewer: str = "validator-1",
) -> CodebookEntry:
    return CodebookEntry(
        entry_id=entry_id,
        concept_id="new-concept",
        text="A new phrase",
        dialect="Nigerian English",
        universal_gloss="a reviewed meaning",
        intent="communicate",
        sociolinguistic_tags=("informal",),
        tone_categories=(tone,),
        linguistic_contexts=("peer conversation",),
        pragmatic_analysis="human-reviewed pragmatics",
        reviewed_by=(reviewer,),
        review_status="approved",
        created_at="2026-01-01T00:00:00+00:00",
    )


class RecordingIndex:
    def __init__(self, version: str = "dataset-v2") -> None:
        self.version = version
        self.calls: list[tuple[CodebookEntry, str]] = []
        self._indexed: set[tuple[str, str]] = set()

    def update(self, item: CodebookEntry, *, dataset_record_hash: str) -> str:
        identity = (item.entry_id, dataset_record_hash)
        if identity not in self._indexed:
            self.calls.append((item, dataset_record_hash))
            self._indexed.add(identity)
        return self.version


def feedback(
    item: CodebookEntry | None = None,
    *,
    action: FeedbackAction = FeedbackAction.ACCEPT,
    actor: str = "validator-1",
    event_id: str = "event-1",
) -> FeedbackEvent:
    return FeedbackEvent(
        request_id="request-1",
        action=action,
        actor_id=actor,
        corrected_entry=item,
        source_result_sha256=SOURCE_HASH,
        event_id=event_id,
        created_at="2026-01-01T00:00:01+00:00",
    )


def gate(
    tmp_path: Path,
    *,
    chain=None,
    index=None,
    policy: TrustPolicy | None = None,
) -> tuple[TrustGate, LocalAppendOnlyEntryStore, RecordingIndex | None]:
    store = LocalAppendOnlyEntryStore(tmp_path / "entries.jsonl")
    updater = RecordingIndex() if index is ... else index
    return (
        TrustGate(
            store=store,
            audit_log=HashChainAuditLog(tmp_path / "audit.jsonl"),
            authorizer=StaticValidatorAuthorizer({"validator-1", "v1"}),
            ipfs=InMemoryIPFS(),
            chain=chain or InMemoryChain(),
            index_updater=updater,
            policy=policy,
        ),
        store,
        updater,
    )


def test_feedback_merges_only_after_verified_ipfs_chain_and_index(tmp_path: Path) -> None:
    trust, store, updater = gate(tmp_path, index=...)
    proposed = entry()
    receipt = trust.process(feedback(proposed))
    assert receipt.state is FeedbackState.MERGED_AND_INDEXED
    assert receipt.cid and receipt.chain_reference and receipt.dataset_record_hash
    assert receipt.indexed_dataset_version == "dataset-v2"
    assert store.contains(proposed)
    assert updater is not None and len(updater.calls) == 1
    assert trust.audit_log.verify()


def test_without_index_updater_reports_merged_not_indexed(tmp_path: Path) -> None:
    trust, _, _ = gate(
        tmp_path,
        index=None,
        policy=TrustPolicy(require_index_update=False),
    )
    receipt = trust.process(feedback(entry()))
    assert receipt.state is FeedbackState.MERGED
    assert receipt.indexed_dataset_version is None


def test_required_index_updater_fails_before_external_side_effects(tmp_path: Path) -> None:
    chain = InMemoryChain()
    trust, store, _ = gate(tmp_path, chain=chain, index=None)
    with pytest.raises(TrustGateError, match="index updater"):
        trust.process(feedback(entry()))
    assert chain.commitments == []
    assert store.head_hash == "0" * 64


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (replace(feedback(entry()), state=FeedbackState.APPROVED), "PROPOSED"),
        (replace(feedback(entry()), source_result_sha256=None), "source_result_sha256"),
        (replace(feedback(entry()), source_result_sha256="not-a-digest"), "source_result_sha256"),
        (feedback(entry(reviewer="someone-else")), "reviewed_by"),
        (feedback(entry(), actor="unknown"), "not authorized"),
    ],
)
def test_feedback_requires_provenance_authorization_and_reviewer_binding(
    tmp_path: Path,
    event: FeedbackEvent,
    message: str,
) -> None:
    trust, _, _ = gate(tmp_path, index=...)
    with pytest.raises(TrustGateError, match=message):
        trust.process(event)


def test_processing_same_event_is_idempotent_across_restart(tmp_path: Path) -> None:
    chain = InMemoryChain()
    updater = RecordingIndex()
    proposed = feedback(entry())
    trust, store, _ = gate(tmp_path, chain=chain, index=updater)
    first = trust.process(proposed)
    audit_lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()

    fresh_updater = RecordingIndex()
    restarted = TrustGate(
        store=LocalAppendOnlyEntryStore(tmp_path / "entries.jsonl"),
        audit_log=HashChainAuditLog(tmp_path / "audit.jsonl"),
        authorizer=StaticValidatorAuthorizer({"validator-1"}),
        ipfs=trust.ipfs,
        chain=chain,
        index_updater=fresh_updater,
    )
    second = restarted.process(proposed)
    assert second == first
    assert len(chain.commitments) == 1
    assert len(updater.calls) == 1
    assert len(fresh_updater.calls) == 1
    assert (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines() == audit_lines
    assert store.contains(entry())


def test_completed_replay_rejects_index_version_mismatch(tmp_path: Path) -> None:
    trust, _, _ = gate(tmp_path, index=...)
    proposed = feedback(entry())
    trust.process(proposed)
    restarted = TrustGate(
        store=LocalAppendOnlyEntryStore(tmp_path / "entries.jsonl"),
        audit_log=HashChainAuditLog(tmp_path / "audit.jsonl"),
        authorizer=StaticValidatorAuthorizer({"validator-1"}),
        ipfs=trust.ipfs,
        chain=trust.chain,
        index_updater=RecordingIndex("different-version"),
    )
    with pytest.raises(TrustGateError, match="does not match"):
        restarted.process(proposed)


def test_chain_parent_tracks_payload_commitments_not_local_record_hashes(tmp_path: Path) -> None:
    chain = InMemoryChain()
    trust, store, _ = gate(tmp_path, chain=chain, index=...)
    first = trust.process(
        feedback(entry("first", "Friendly"), event_id="event-first")
    )
    second = trust.process(
        feedback(entry("second", "Serious"), event_id="event-second")
    )

    assert chain.commitments[0]["parent_sha256"] == "0" * 64
    assert chain.commitments[1]["parent_sha256"] == first.payload_sha256
    assert store.commitment_head == second.payload_sha256
    assert store.head_hash == second.dataset_record_hash
    assert store.commitment_head != store.head_hash


def test_audit_event_state_key_is_idempotent_and_immutable(tmp_path: Path) -> None:
    audit = HashChainAuditLog(tmp_path / "audit.jsonl")
    first = audit.append({"state": "approved", "value": 1}, idempotency_key="e:approved")
    assert audit.append(
        {"state": "approved", "value": 1}, idempotency_key="e:approved"
    ) == first
    assert len((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(TrustGateError, match="idempotency conflict"):
        audit.append({"state": "approved", "value": 2}, idempotency_key="e:approved")


def test_duplicate_key_includes_tone_and_preserves_polysemy(tmp_path: Path) -> None:
    trust, store, _ = gate(tmp_path, index=...)
    trust.process(feedback(entry("friendly", "Friendly"), event_id="event-friendly"))
    trust.process(feedback(entry("frustrated", "Frustrated"), event_id="event-frustrated"))
    assert store.contains(entry("anything", "Friendly"))
    assert store.contains(entry("anything", "Frustrated"))

    with pytest.raises(TrustGateError, match="duplicate"):
        trust.process(feedback(entry("duplicate", "Friendly"), event_id="event-duplicate"))


def test_store_merge_is_idempotent_but_compare_and_swap_rejects_stale_head(
    tmp_path: Path,
) -> None:
    store = LocalAppendOnlyEntryStore(tmp_path / "entries.jsonl")
    initial = entry()
    first_hash = store.merge(
        initial,
        payload_sha256=PAYLOAD_HASH,
        cid="cid-1",
        chain_reference="tx-1",
        event_id="event-1",
        expected_parent_hash="0" * 64,
    )
    assert store.merge(
        initial,
        payload_sha256=PAYLOAD_HASH,
        cid="different-retry-cid",
        chain_reference="different-retry-tx",
        event_id="event-1",
        expected_parent_hash="0" * 64,
    ) == first_hash

    with pytest.raises(TrustGateError, match="head changed"):
        store.merge(
            entry("other", "Serious"),
            payload_sha256="c" * 64,
            cid="cid-2",
            chain_reference="tx-2",
            event_id="event-2",
            expected_parent_hash="0" * 64,
        )


def test_independent_store_instances_refresh_before_compare_and_swap(tmp_path: Path) -> None:
    path = tmp_path / "entries.jsonl"
    first_store = LocalAppendOnlyEntryStore(path)
    stale_store = LocalAppendOnlyEntryStore(path)
    first_store.merge(
        entry("first", "Friendly"),
        payload_sha256="1" * 64,
        cid=None,
        chain_reference=None,
        event_id="event-first",
        expected_parent_hash="0" * 64,
    )
    with pytest.raises(TrustGateError, match="head changed"):
        stale_store.merge(
            entry("second", "Serious"),
            payload_sha256="2" * 64,
            cid=None,
            chain_reference=None,
            event_id="event-second",
            expected_parent_hash="0" * 64,
        )


def test_store_enforces_entry_identity_and_linear_version_chain(tmp_path: Path) -> None:
    store = LocalAppendOnlyEntryStore(tmp_path / "entries.jsonl")
    original = entry()
    store.merge(
        original,
        payload_sha256=PAYLOAD_HASH,
        cid=None,
        chain_reference=None,
        event_id="event-1",
        expected_parent_hash=store.head_hash,
    )

    with pytest.raises(TrustGateError, match="entry_id already exists"):
        store.merge(
            replace(original, intent="different"),
            payload_sha256="c" * 64,
            cid=None,
            chain_reference=None,
            event_id="event-2",
            expected_parent_hash=store.head_hash,
        )

    successor = replace(
        original,
        entry_id="new-2",
        version=2,
        supersedes_entry_id=original.entry_id,
        pragmatic_analysis="corrected analysis",
    )
    store.merge(
        successor,
        payload_sha256="d" * 64,
        cid=None,
        chain_reference=None,
        event_id="event-2",
        expected_parent_hash=store.head_hash,
    )

    sibling = replace(successor, entry_id="new-2b")
    with pytest.raises(TrustGateError, match="already has a successor"):
        store.merge(
            sibling,
            payload_sha256="e" * 64,
            cid=None,
            chain_reference=None,
            event_id="event-3",
            expected_parent_hash=store.head_hash,
        )

    bad_version = replace(
        successor,
        entry_id="new-3",
        version=4,
        supersedes_entry_id=successor.entry_id,
    )
    with pytest.raises(TrustGateError, match="exactly one"):
        store.merge(
            bad_version,
            payload_sha256="f" * 64,
            cid=None,
            chain_reference=None,
            event_id="event-4",
            expected_parent_hash=store.head_hash,
        )


def test_feedback_can_supersede_a_validated_baseline_entry(tmp_path: Path) -> None:
    baseline = entry("base-1")
    store = LocalAppendOnlyEntryStore(
        tmp_path / "entries.jsonl",
        baseline_entries=(baseline,),
    )
    assert store.head_hash == "0" * 64
    assert store.commitment_head == "0" * 64
    assert store.contains(baseline)

    successor = replace(
        baseline,
        entry_id="base-2",
        version=2,
        supersedes_entry_id=baseline.entry_id,
        pragmatic_analysis="reviewed correction",
    )
    trust = TrustGate(
        store=store,
        audit_log=HashChainAuditLog(tmp_path / "audit.jsonl"),
        authorizer=StaticValidatorAuthorizer({"validator-1"}),
        ipfs=InMemoryIPFS(),
        chain=InMemoryChain(),
        index_updater=RecordingIndex(),
    )
    receipt = trust.process(feedback(successor))
    assert receipt.state is FeedbackState.MERGED_AND_INDEXED
    assert store.find_by_entry_id(successor.entry_id) is not None

    restarted = LocalAppendOnlyEntryStore(
        tmp_path / "entries.jsonl",
        baseline_entries=(baseline,),
    )
    assert restarted.find_by_entry_id(successor.entry_id) is not None
    assert restarted.commitment_head == receipt.payload_sha256


def test_chain_failure_is_fail_closed(tmp_path: Path) -> None:
    class FailingChain:
        def anchor(self, **kwargs):
            raise RuntimeError("chain unavailable")

    trust, store, _ = gate(tmp_path, chain=FailingChain(), index=...)
    proposed = entry()
    with pytest.raises(TrustGateError, match="chain unavailable"):
        trust.process(feedback(proposed))
    assert not store.contains(proposed)
    assert not (tmp_path / "entries.jsonl").exists()
    assert trust.audit_log.verify()


def test_reject_is_authorized_source_bound_and_audited_without_mutation(tmp_path: Path) -> None:
    trust, store, _ = gate(tmp_path, index=...)
    receipt = trust.process(feedback(action=FeedbackAction.REJECT))
    assert receipt.state is FeedbackState.REJECTED
    assert store.head_hash == "0" * 64
    assert trust.audit_log.verify()


def test_required_replication_failure_blocks_active_merge(tmp_path: Path) -> None:
    class FailingPublisher:
        def publish(self, **kwargs):
            raise RuntimeError("HF unavailable")

    store = LocalAppendOnlyEntryStore(tmp_path / "entries.jsonl")
    trust = TrustGate(
        store=store,
        audit_log=HashChainAuditLog(tmp_path / "audit.jsonl"),
        authorizer=StaticValidatorAuthorizer({"validator-1"}),
        ipfs=InMemoryIPFS(),
        chain=InMemoryChain(),
        publisher=FailingPublisher(),
        index_updater=RecordingIndex(),
        policy=TrustPolicy(require_hf_replication=True),
    )
    proposed = entry()
    with pytest.raises(TrustGateError, match="HF unavailable"):
        trust.process(feedback(proposed))
    assert not store.contains(proposed)


def test_pipeline_index_updater_delegates_and_requires_version() -> None:
    class Pipeline:
        def __init__(self) -> None:
            self.items: list[CodebookEntry] = []

        def append_codebook_entry(self, item: CodebookEntry) -> str:
            self.items.append(item)
            return "materialized-v2"

    pipeline = Pipeline()
    updater = PipelineIndexUpdater(pipeline)
    assert updater.index(entry(), dataset_record_hash="f" * 64) == "materialized-v2"
    assert pipeline.items == [entry()]


def test_paper5_factory_enforces_profile_provenance_and_live_index(
    tmp_path: Path, codebook, fake_provider
) -> None:
    pipeline = build_pipeline(
        "paper5",
        codebook=codebook,
        provider=fake_provider,
        config_path=Path(__file__).parents[1] / "configs" / "paper5.json",
    )
    trust = TrustGate.for_pipeline(
        pipeline,
        store=LocalAppendOnlyEntryStore(
            tmp_path / "entries.jsonl", baseline_entries=codebook.entries
        ),
        audit_log=HashChainAuditLog(tmp_path / "audit.jsonl"),
        authorizer=StaticValidatorAuthorizer({"validator-1"}),
        ipfs=InMemoryIPFS(),
        chain=InMemoryChain(),
    )
    assert trust.policy.require_ipfs
    assert trust.policy.require_chain
    assert isinstance(trust.index_updater, PipelineIndexUpdater)
