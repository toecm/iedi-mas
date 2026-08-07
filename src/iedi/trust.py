from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from .codebook import canonical_json, normalize_text
from .provenance import ChainSink, IPFSSink
from .replication import RecordPublisher
from .schemas import (
    CodebookEntry,
    FeedbackAction,
    FeedbackEvent,
    FeedbackState,
    InterpretationResult,
)


_ZERO_HASH = "0" * 64
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class TrustGateError(RuntimeError):
    pass


def _lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _exclusive_file_lock(path: Path):
    """Small cross-platform advisory lock for local JSONL coordination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":  # pragma: no cover - platform-specific branch
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - exercised by Linux CI
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def interpretation_result_sha256(result: InterpretationResult) -> str:
    """Canonical digest used to bind feedback to the displayed interpretation."""

    return hashlib.sha256(canonical_json(result.to_dict())).hexdigest()


def duplicate_key(entry: CodebookEntry) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        normalize_text(entry.text),
        normalize_text(entry.dialect),
        normalize_text(entry.universal_gloss),
        tuple(sorted(normalize_text(tone) for tone in entry.tone_categories)),
    )


class FeedbackAuthorizer(Protocol):
    """Authenticate the actor and authorize the proposed feedback event.

    Production implementations should verify the event signature against an
    authenticated validator identity.  The trust gate deliberately does not
    infer authentication from a caller-supplied ``actor_id``.
    """

    def authorize(self, event: FeedbackEvent) -> bool: ...


@dataclass(frozen=True, init=False)
class StaticValidatorAuthorizer:
    """Explicit offline/test authorizer backed by a fixed validator allow-list.

    This is useful for notebooks and deterministic tests; it is not a
    cryptographic signature verifier and should not be presented as one.
    """

    validator_ids: frozenset[str]

    def __init__(self, validator_ids: Iterable[str]) -> None:
        values = frozenset(value.strip() for value in validator_ids if value.strip())
        if not values:
            raise ValueError("at least one validator identity is required")
        object.__setattr__(self, "validator_ids", values)

    def authorize(self, event: FeedbackEvent) -> bool:
        return event.actor_id in self.validator_ids


class IndexUpdater(Protocol):
    """Update the active retrieval index and return its new dataset version."""

    def update(
        self,
        entry: CodebookEntry,
        *,
        dataset_record_hash: str,
    ) -> str: ...


class _AppendablePipeline(Protocol):
    def append_codebook_entry(self, entry: CodebookEntry) -> str: ...


@dataclass
class PipelineIndexUpdater:
    """Adapter for a pipeline that atomically swaps in an appended codebook."""

    pipeline: _AppendablePipeline

    def update(
        self,
        entry: CodebookEntry,
        *,
        dataset_record_hash: str,
    ) -> str:
        del dataset_record_hash  # The pipeline versions the materialized codebook itself.
        version = self.pipeline.append_codebook_entry(entry)
        if not str(version).strip():
            raise TrustGateError("index updater returned no dataset version")
        return str(version)

    # A descriptive alias is retained for direct notebook use.
    def index(
        self,
        entry: CodebookEntry,
        *,
        dataset_record_hash: str,
    ) -> str:
        return self.update(entry, dataset_record_hash=dataset_record_hash)


@dataclass(frozen=True)
class StoredEntryRecord:
    entry: CodebookEntry
    payload_sha256: str
    cid: str | None
    chain_reference: str | None
    replica_reference: str | None
    event_id: str | None
    parent_hash: str
    record_hash: str


class AppendOnlyEntryStore(Protocol):
    @property
    def head_hash(self) -> str: ...

    @property
    def commitment_head(self) -> str: ...

    def contains(self, entry: CodebookEntry) -> bool: ...

    def find_by_entry_id(self, entry_id: str) -> StoredEntryRecord | None: ...

    def refresh(self) -> None: ...

    def merge(
        self,
        entry: CodebookEntry,
        *,
        payload_sha256: str,
        cid: str | None,
        chain_reference: str | None,
        replica_reference: str | None,
        event_id: str,
        expected_parent_hash: str,
    ) -> str: ...


class LocalAppendOnlyEntryStore:
    """Hash-chained JSONL store with immutable IDs and linear version chains."""

    def __init__(
        self,
        path: str | Path,
        *,
        baseline_entries: Iterable[CodebookEntry] = (),
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._active_keys: dict[tuple[str, str, str, tuple[str, ...]], str] = {}
        self._all_keys: set[tuple[str, str, str, tuple[str, ...]]] = set()
        self._entries_by_id: dict[str, CodebookEntry] = {}
        self._records_by_id: dict[str, StoredEntryRecord] = {}
        self._superseded_by: dict[str, str] = {}
        self._head_hash = _ZERO_HASH
        self._commitment_head = _ZERO_HASH
        self._baseline_entries = tuple(baseline_entries)
        self._load_baseline(self._baseline_entries)
        if self.path.exists():
            self._load()

    @property
    def head_hash(self) -> str:
        with self._lock:
            return self._head_hash

    @property
    def commitment_head(self) -> str:
        """Latest committed payload digest, matching PureChain's finalized head."""

        with self._lock:
            return self._commitment_head

    def contains(self, entry: CodebookEntry) -> bool:
        with self._lock:
            return duplicate_key(entry) in self._all_keys

    def find_by_entry_id(self, entry_id: str) -> StoredEntryRecord | None:
        with self._lock:
            return self._records_by_id.get(entry_id)

    def refresh(self) -> None:
        with self._lock, _exclusive_file_lock(_lock_path(self.path)):
            self._reload_unlocked()

    def merge(
        self,
        entry: CodebookEntry,
        *,
        payload_sha256: str,
        cid: str | None,
        chain_reference: str | None,
        replica_reference: str | None = None,
        event_id: str = "",
        expected_parent_hash: str,
    ) -> str:
        if not _SHA256_RE.fullmatch(payload_sha256):
            raise TrustGateError("payload_sha256 must be a 64-character hexadecimal digest")
        if not _SHA256_RE.fullmatch(expected_parent_hash):
            raise TrustGateError("expected_parent_hash must be a 64-character hexadecimal digest")

        with self._lock, _exclusive_file_lock(_lock_path(self.path)):
            self._reload_unlocked()
            existing_entry = self._entries_by_id.get(entry.entry_id)
            if existing_entry is not None:
                existing = self._records_by_id.get(entry.entry_id)
                if (
                    existing is not None
                    and existing.entry == entry
                    and existing.payload_sha256 == payload_sha256.lower()
                    and existing.event_id == (event_id or None)
                ):
                    # A retry after an uncertain write is safe even though the head
                    # has advanced beyond the caller's original compare-and-swap value.
                    return existing.record_hash
                raise TrustGateError(f"entry_id already exists with different content: {entry.entry_id}")

            if expected_parent_hash != self._head_hash:
                raise TrustGateError(
                    "dataset head changed before merge; submit a new feedback event against the new head"
                )

            self._validate_new_entry(entry)
            key = duplicate_key(entry)
            key_owner = self._active_keys.get(key)
            if key_owner is not None and key_owner != entry.supersedes_entry_id:
                raise TrustGateError("exact utterance/dialect/gloss/tone duplicate")

            record_body = {
                "entry": asdict(entry),
                "payload_sha256": payload_sha256.lower(),
                "cid": cid,
                "chain_reference": chain_reference,
                "replica_reference": replica_reference,
                "event_id": event_id or None,
                "parent_hash": self._head_hash,
            }
            record_hash = hashlib.sha256(canonical_json(record_body)).hexdigest()
            persisted = dict(record_body, record_hash=record_hash)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(persisted, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            stored = StoredEntryRecord(
                entry=entry,
                payload_sha256=payload_sha256.lower(),
                cid=cid,
                chain_reference=chain_reference,
                replica_reference=replica_reference,
                event_id=event_id or None,
                parent_hash=self._head_hash,
                record_hash=record_hash,
            )
            self._register(stored)
            self._head_hash = record_hash
            self._commitment_head = payload_sha256.lower()
            return record_hash

    def _reload_unlocked(self) -> None:
        self._active_keys.clear()
        self._all_keys.clear()
        self._entries_by_id.clear()
        self._records_by_id.clear()
        self._superseded_by.clear()
        self._head_hash = _ZERO_HASH
        self._commitment_head = _ZERO_HASH
        self._load_baseline(self._baseline_entries)
        if self.path.exists():
            self._load()

    def _validate_new_entry(self, entry: CodebookEntry) -> None:
        predecessor_id = entry.supersedes_entry_id
        if predecessor_id is None:
            if entry.version != 1:
                raise TrustGateError("an initial entry must have version 1")
            return

        predecessor = self._entries_by_id.get(predecessor_id)
        if predecessor is None:
            raise TrustGateError(f"superseded entry does not exist: {predecessor_id}")
        if predecessor_id in self._superseded_by:
            raise TrustGateError(f"entry already has a successor: {predecessor_id}")
        if entry.concept_id != predecessor.concept_id:
            raise TrustGateError("a successor must preserve concept_id")
        if normalize_text(entry.dialect) != normalize_text(predecessor.dialect):
            raise TrustGateError("a successor must preserve dialect identity")
        if entry.version != predecessor.version + 1:
            raise TrustGateError("a successor version must increment its predecessor by exactly one")

    def _register(self, stored: StoredEntryRecord) -> None:
        entry = stored.entry
        self._entries_by_id[entry.entry_id] = entry
        self._records_by_id[entry.entry_id] = stored
        self._all_keys.add(duplicate_key(entry))
        if entry.supersedes_entry_id is not None:
            predecessor = self._entries_by_id[entry.supersedes_entry_id]
            predecessor_key = duplicate_key(predecessor)
            if self._active_keys.get(predecessor_key) == predecessor.entry_id:
                del self._active_keys[predecessor_key]
            self._superseded_by[predecessor.entry_id] = entry.entry_id
        self._active_keys[duplicate_key(entry)] = entry.entry_id

    def _load_baseline(self, entries: tuple[CodebookEntry, ...]) -> None:
        """Load a validated materialized base view without adding audit records."""

        for entry in entries:
            if entry.review_status not in {"approved", "superseded"}:
                raise TrustGateError(
                    f"baseline entry is not validated: {entry.entry_id} ({entry.review_status})"
                )
            if entry.entry_id in self._entries_by_id:
                raise TrustGateError(f"duplicate baseline entry_id: {entry.entry_id}")
            self._entries_by_id[entry.entry_id] = entry

        # Validate identity/version relationships after indexing every ID so a
        # caller need not provide the baseline in topological order.
        for entry in entries:
            predecessor_id = entry.supersedes_entry_id
            if predecessor_id is None:
                if entry.version != 1:
                    raise TrustGateError("an initial baseline entry must have version 1")
                continue
            predecessor = self._entries_by_id.get(predecessor_id)
            if predecessor is None:
                raise TrustGateError(
                    f"baseline superseded entry does not exist: {predecessor_id}"
                )
            if predecessor_id in self._superseded_by:
                raise TrustGateError(f"baseline entry has multiple successors: {predecessor_id}")
            if entry.concept_id != predecessor.concept_id:
                raise TrustGateError("a baseline successor must preserve concept_id")
            if normalize_text(entry.dialect) != normalize_text(predecessor.dialect):
                raise TrustGateError("a baseline successor must preserve dialect identity")
            if entry.version != predecessor.version + 1:
                raise TrustGateError(
                    "a baseline successor version must increment its predecessor by exactly one"
                )
            self._superseded_by[predecessor_id] = entry.entry_id

        for entry in entries:
            key = duplicate_key(entry)
            self._all_keys.add(key)
            if entry.entry_id in self._superseded_by or entry.review_status != "approved":
                continue
            if key in self._active_keys:
                raise TrustGateError("duplicate active semantic key in baseline")
            self._active_keys[key] = entry.entry_id

    def _load(self) -> None:
        expected_parent = _ZERO_HASH
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                record_hash = str(raw.pop("record_hash"))
                if raw.get("parent_hash") != expected_parent:
                    raise TrustGateError(f"broken parent hash at line {line_number}")
                calculated = hashlib.sha256(canonical_json(raw)).hexdigest()
                if calculated != record_hash:
                    raise TrustGateError(f"tampered entry store at line {line_number}")
                entry = _entry_from_mapping(raw["entry"])
                stored = StoredEntryRecord(
                    entry=entry,
                    payload_sha256=str(raw["payload_sha256"]),
                    cid=raw.get("cid"),
                    chain_reference=raw.get("chain_reference"),
                    replica_reference=raw.get("replica_reference"),
                    event_id=raw.get("event_id"),
                    parent_hash=str(raw["parent_hash"]),
                    record_hash=record_hash,
                )
                try:
                    if entry.entry_id in self._entries_by_id:
                        raise TrustGateError(f"duplicate entry_id: {entry.entry_id}")
                    self._validate_new_entry(entry)
                    key_owner = self._active_keys.get(duplicate_key(entry))
                    if key_owner is not None and key_owner != entry.supersedes_entry_id:
                        raise TrustGateError("duplicate active semantic key")
                    self._register(stored)
                except TrustGateError as exc:
                    raise TrustGateError(f"invalid entry store at line {line_number}: {exc}") from exc
                expected_parent = record_hash
        self._head_hash = expected_parent
        self._commitment_head = (
            self._records_by_id[next(reversed(self._records_by_id))].payload_sha256
            if self._records_by_id
            else _ZERO_HASH
        )


@dataclass(frozen=True)
class AuditedEvent:
    event: Mapping[str, Any]
    record_hash: str


class HashChainAuditLog:
    """Hash-chained state log with one content-bound record per event/state key."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._head_hash = _ZERO_HASH
        self._by_idempotency_key: dict[str, AuditedEvent] = {}
        if self.path.exists():
            self._verify_unlocked()

    @property
    def head_hash(self) -> str:
        with self._lock:
            return self._head_hash

    def get(self, idempotency_key: str) -> AuditedEvent | None:
        with self._lock:
            return self._by_idempotency_key.get(idempotency_key)

    def append(
        self,
        event: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> str:
        materialized = dict(event)
        with self._lock, _exclusive_file_lock(_lock_path(self.path)):
            self._verify_unlocked()
            if idempotency_key:
                existing = self._by_idempotency_key.get(idempotency_key)
                if existing is not None:
                    if canonical_json(existing.event) != canonical_json(materialized):
                        raise TrustGateError(
                            f"audit idempotency conflict for {idempotency_key}"
                        )
                    return existing.record_hash

            body = {
                "event": materialized,
                "idempotency_key": idempotency_key,
                "parent_hash": self._head_hash,
            }
            record_hash = hashlib.sha256(canonical_json(body)).hexdigest()
            persisted = dict(body, record_hash=record_hash)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(persisted, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._head_hash = record_hash
            if idempotency_key:
                self._by_idempotency_key[idempotency_key] = AuditedEvent(
                    event=materialized,
                    record_hash=record_hash,
                )
            return record_hash

    def verify(self) -> bool:
        with self._lock:
            self._verify_unlocked()
        return True

    def refresh(self) -> None:
        with self._lock, _exclusive_file_lock(_lock_path(self.path)):
            self._verify_unlocked()

    def _verify_unlocked(self) -> None:
        expected_parent = _ZERO_HASH
        keys: dict[str, AuditedEvent] = {}
        if not self.path.exists():
            self._head_hash = expected_parent
            self._by_idempotency_key = keys
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                record_hash = str(raw.pop("record_hash"))
                if raw.get("parent_hash") != expected_parent:
                    raise TrustGateError(f"broken audit parent at line {line_number}")
                calculated = hashlib.sha256(canonical_json(raw)).hexdigest()
                if calculated != record_hash:
                    raise TrustGateError(f"tampered audit record at line {line_number}")
                key = raw.get("idempotency_key")
                if key:
                    if key in keys:
                        raise TrustGateError(
                            f"duplicate audit idempotency key at line {line_number}: {key}"
                        )
                    keys[str(key)] = AuditedEvent(
                        event=dict(raw["event"]),
                        record_hash=record_hash,
                    )
                expected_parent = record_hash
        self._head_hash = expected_parent
        self._by_idempotency_key = keys


@dataclass(frozen=True)
class TrustPolicy:
    require_ipfs: bool = True
    require_chain: bool = True
    verify_ipfs_content: bool = True
    require_hf_replication: bool = False
    require_index_update: bool = True


@dataclass(frozen=True)
class TrustReceipt:
    event_id: str
    state: FeedbackState
    payload_sha256: str | None = None
    cid: str | None = None
    chain_reference: str | None = None
    replica_reference: str | None = None
    dataset_record_hash: str | None = None
    indexed_dataset_version: str | None = None
    audit_record_hash: str | None = None


class TrustGate:
    """Fail-closed, resumable feedback state machine.

    Persistent dataset mutation happens only after the configured provenance
    requirements.  ``MERGED_AND_INDEXED`` is emitted only after an injected
    index updater returns a concrete dataset version.
    """

    def __init__(
        self,
        *,
        store: AppendOnlyEntryStore,
        audit_log: HashChainAuditLog,
        authorizer: FeedbackAuthorizer,
        policy: TrustPolicy | None = None,
        ipfs: IPFSSink | None = None,
        chain: ChainSink | None = None,
        publisher: RecordPublisher | None = None,
        index_updater: IndexUpdater | None = None,
    ) -> None:
        self.store = store
        self.audit_log = audit_log
        self.authorizer = authorizer
        self.policy = policy or TrustPolicy()
        self.ipfs = ipfs
        self.chain = chain
        self.publisher = publisher
        self.index_updater = index_updater
        self._lock = threading.Lock()
        audit_path = getattr(audit_log, "path", None)
        self._process_lock_path = (
            Path(audit_path).with_name(f"{Path(audit_path).name}.workflow.lock")
            if audit_path is not None
            else None
        )

    @classmethod
    def for_pipeline(
        cls,
        pipeline: _AppendablePipeline,
        *,
        store: AppendOnlyEntryStore,
        audit_log: HashChainAuditLog,
        authorizer: FeedbackAuthorizer,
        ipfs: IPFSSink | None = None,
        chain: ChainSink | None = None,
        publisher: RecordPublisher | None = None,
        require_hf_replication: bool = False,
    ) -> "TrustGate":
        """Compose a gate from enforced paper-profile requirements and live index."""

        profile = getattr(pipeline, "profile", None)
        if profile is None or not bool(getattr(profile, "trust_gate_required", False)):
            raise TrustGateError("the pipeline profile does not require a trust gate")
        policy = TrustPolicy(
            require_ipfs=bool(getattr(profile, "require_ipfs_verification", False)),
            require_chain=bool(
                getattr(profile, "require_finalized_chain_commitment", False)
            ),
            verify_ipfs_content=True,
            require_hf_replication=require_hf_replication,
            require_index_update=True,
        )
        return cls(
            store=store,
            audit_log=audit_log,
            authorizer=authorizer,
            policy=policy,
            ipfs=ipfs,
            chain=chain,
            publisher=publisher,
            index_updater=PipelineIndexUpdater(pipeline),
        )

    def process(self, event: FeedbackEvent) -> TrustReceipt:
        # Serializing one gate instance avoids duplicate external side effects;
        # the store CAS still protects independent processes/gate instances.
        process_lock = (
            _exclusive_file_lock(self._process_lock_path)
            if self._process_lock_path is not None
            else nullcontext()
        )
        with self._lock, process_lock:
            return self._process_locked(event)

    def _process_locked(self, event: FeedbackEvent) -> TrustReceipt:
        refresh_store = getattr(self.store, "refresh", None)
        if refresh_store is not None:
            refresh_store()
        refresh_audit = getattr(self.audit_log, "refresh", None)
        if refresh_audit is not None:
            refresh_audit()
        self._validate_proposal(event)

        if event.action is FeedbackAction.REJECT:
            rejected = replace(event, state=FeedbackState.REJECTED)
            audit_hash = self._audit(rejected)
            return TrustReceipt(event.event_id, FeedbackState.REJECTED, audit_record_hash=audit_hash)

        entry = event.corrected_entry
        if entry is None:
            raise TrustGateError("accepted or suggested feedback requires a complete corrected entry")
        if entry.review_status != "approved":
            raise TrustGateError("only a human-approved entry can pass the trust gate")
        if event.actor_id not in entry.reviewed_by:
            raise TrustGateError("the authenticated actor must be recorded in corrected_entry.reviewed_by")
        if self.policy.require_index_update and self.index_updater is None:
            raise TrustGateError("an index updater is required by trust policy")

        completed = self._completed_receipt(event)
        if completed is not None:
            return completed

        validated_record = self.audit_log.get(_state_key(event, FeedbackState.USER_VALIDATED))
        if validated_record is not None:
            self._assert_same_event(event, validated_record.event, FeedbackState.USER_VALIDATED)
            payload_sha256 = str(validated_record.event["payload_sha256"])
            expected_parent = str(validated_record.event["expected_parent_hash"])
            parent_commitment_hash = str(validated_record.event["parent_commitment_hash"])
            payload = _feedback_payload(event, entry, parent_commitment_hash)
            if hashlib.sha256(canonical_json(payload)).hexdigest() != payload_sha256:
                raise TrustGateError("validated payload no longer matches its audit commitment")
        else:
            expected_parent = self.store.head_hash
            parent_commitment_hash = self.store.commitment_head
            payload = _feedback_payload(event, entry, parent_commitment_hash)
            payload_sha256 = hashlib.sha256(canonical_json(payload)).hexdigest()
            self._audit(
                replace(event, state=FeedbackState.USER_VALIDATED),
                payload_sha256=payload_sha256,
                expected_parent_hash=expected_parent,
                parent_commitment_hash=parent_commitment_hash,
            )

        cid: str | None = None
        chain_reference: str | None = None
        replica_reference: str | None = None
        try:
            if self.policy.require_ipfs:
                pinned = self.audit_log.get(_state_key(event, FeedbackState.IPFS_PINNED))
                if pinned is not None:
                    self._assert_same_event(event, pinned.event, FeedbackState.IPFS_PINNED)
                    cid = str(pinned.event["cid"])
                else:
                    if self.ipfs is None:
                        raise TrustGateError("IPFS is required but no sink is configured")
                    cid = self.ipfs.pin(payload)
                    if not cid or cid.casefold().startswith(("ipfs_fail", "local-log")):
                        raise TrustGateError("IPFS pin did not return a valid CID")
                    self._audit(
                        replace(event, state=FeedbackState.IPFS_PINNED),
                        payload_sha256=payload_sha256,
                        cid=cid,
                    )
                if self.policy.verify_ipfs_content:
                    if self.ipfs is None or not self.ipfs.verify(cid, payload_sha256):
                        raise TrustGateError("IPFS content does not match the committed payload")

            if self.policy.require_chain:
                confirmed = self.audit_log.get(_state_key(event, FeedbackState.CHAIN_CONFIRMED))
                if confirmed is not None:
                    self._assert_same_event(event, confirmed.event, FeedbackState.CHAIN_CONFIRMED)
                    chain_reference = str(confirmed.event["chain_reference"])
                    cid = cid or str(confirmed.event["cid"])
                else:
                    if self.chain is None:
                        raise TrustGateError("chain confirmation is required but no sink is configured")
                    if not cid:
                        raise TrustGateError("chain anchoring requires a verified CID")
                    chain_reference = self.chain.anchor(
                        payload_sha256=payload_sha256,
                        cid=cid,
                        parent_sha256=parent_commitment_hash,
                    )
                    if not chain_reference:
                        raise TrustGateError("chain transaction was not finalized")
                    self._audit(
                        replace(event, state=FeedbackState.CHAIN_CONFIRMED),
                        payload_sha256=payload_sha256,
                        cid=cid,
                        chain_reference=chain_reference,
                    )

            approved = self.audit_log.get(_state_key(event, FeedbackState.APPROVED))
            if approved is not None:
                self._assert_same_event(event, approved.event, FeedbackState.APPROVED)
                replica_reference = _optional_string(approved.event.get("replica_reference"))
                cid = cid or _optional_string(approved.event.get("cid"))
                chain_reference = chain_reference or _optional_string(
                    approved.event.get("chain_reference")
                )
            else:
                if self.policy.require_hf_replication:
                    if self.publisher is None:
                        raise TrustGateError(
                            "Hugging Face replication is required but no publisher is configured"
                        )
                    replica_reference = self.publisher.publish(
                        payload_sha256=payload_sha256,
                        payload=payload,
                        chain_reference=chain_reference,
                    )
                    if not replica_reference:
                        raise TrustGateError("Hugging Face replication returned no commit reference")

                self._audit(
                    replace(event, state=FeedbackState.APPROVED),
                    payload_sha256=payload_sha256,
                    cid=cid,
                    chain_reference=chain_reference,
                    replica_reference=replica_reference,
                )

            dataset_hash = self.store.merge(
                entry,
                payload_sha256=payload_sha256,
                cid=cid,
                chain_reference=chain_reference,
                replica_reference=replica_reference,
                event_id=event.event_id,
                expected_parent_hash=expected_parent,
            )
            merged = replace(event, state=FeedbackState.MERGED)
            merged_audit_hash = self._audit(
                merged,
                payload_sha256=payload_sha256,
                cid=cid,
                chain_reference=chain_reference,
                replica_reference=replica_reference,
                dataset_record_hash=dataset_hash,
            )

            if self.index_updater is None:
                return TrustReceipt(
                    event_id=event.event_id,
                    state=FeedbackState.MERGED,
                    payload_sha256=payload_sha256,
                    cid=cid,
                    chain_reference=chain_reference,
                    replica_reference=replica_reference,
                    dataset_record_hash=dataset_hash,
                    audit_record_hash=merged_audit_hash,
                )

            indexed_version = self._update_index(entry, dataset_hash)
            if not str(indexed_version).strip():
                raise TrustGateError("index updater returned no dataset version")
            final = replace(event, state=FeedbackState.MERGED_AND_INDEXED)
            audit_hash = self._audit(
                final,
                payload_sha256=payload_sha256,
                cid=cid,
                chain_reference=chain_reference,
                replica_reference=replica_reference,
                dataset_record_hash=dataset_hash,
                indexed_dataset_version=str(indexed_version),
            )
            return TrustReceipt(
                event_id=event.event_id,
                state=FeedbackState.MERGED_AND_INDEXED,
                payload_sha256=payload_sha256,
                cid=cid,
                chain_reference=chain_reference,
                replica_reference=replica_reference,
                dataset_record_hash=dataset_hash,
                indexed_dataset_version=str(indexed_version),
                audit_record_hash=audit_hash,
            )
        except Exception as exc:
            failed_event = replace(event, state=FeedbackState.FAILED)
            failed_payload = _event_dict(failed_event) | {
                "payload_sha256": payload_sha256,
                "error_type": type(exc).__name__,
            }
            try:
                self.audit_log.append(
                    failed_payload,
                    idempotency_key=_state_key(event, FeedbackState.FAILED),
                )
            except TrustGateError:
                # Preserve the operational failure if a previous failed attempt
                # already owns the event/state audit key.
                pass
            if isinstance(exc, TrustGateError):
                raise
            raise TrustGateError(str(exc)) from exc

    def _validate_proposal(self, event: FeedbackEvent) -> None:
        if event.state is not FeedbackState.PROPOSED:
            raise TrustGateError("only feedback in the PROPOSED state can enter the trust gate")
        if not event.actor_id.strip():
            raise TrustGateError("feedback actor identity is required")
        if not event.source_result_sha256 or not _SHA256_RE.fullmatch(event.source_result_sha256):
            raise TrustGateError(
                "source_result_sha256 must be a 64-character hexadecimal digest"
            )
        if not bool(self.authorizer.authorize(event)):
            raise TrustGateError("feedback actor is not authorized")

    def _audit(self, event: FeedbackEvent, **extra: Any) -> str:
        return self.audit_log.append(
            _event_dict(event) | extra,
            idempotency_key=_state_key(event, event.state),
        )

    def _completed_receipt(self, event: FeedbackEvent) -> TrustReceipt | None:
        final = self.audit_log.get(_state_key(event, FeedbackState.MERGED_AND_INDEXED))
        if final is not None:
            self._assert_same_event(event, final.event, FeedbackState.MERGED_AND_INDEXED)
            self._assert_store_matches_completion(event, final.event)
            if self.index_updater is not None:
                entry = event.corrected_entry
                if entry is None:  # guarded by caller; retained for type safety
                    raise TrustGateError("indexed feedback has no corrected entry")
                dataset_hash = str(final.event["dataset_record_hash"])
                audited_version = _optional_string(
                    final.event.get("indexed_dataset_version")
                )
                if audited_version is None:
                    raise TrustGateError(
                        "indexed audit record does not contain a dataset version"
                    )
                current_version = self._update_index(entry, dataset_hash)
                if current_version != audited_version:
                    raise TrustGateError(
                        "current index version does not match the audited indexed version"
                    )
            return _receipt_from_audit(event.event_id, FeedbackState.MERGED_AND_INDEXED, final)

        merged = self.audit_log.get(_state_key(event, FeedbackState.MERGED))
        if merged is None:
            return None
        self._assert_same_event(event, merged.event, FeedbackState.MERGED)
        self._assert_store_matches_completion(event, merged.event)
        if self.index_updater is None:
            if self.policy.require_index_update:
                raise TrustGateError("an index updater is required by trust policy")
            return _receipt_from_audit(event.event_id, FeedbackState.MERGED, merged)

        entry = event.corrected_entry
        if entry is None:  # guarded by caller; retained for type safety
            raise TrustGateError("merged feedback has no corrected entry")
        dataset_hash = str(merged.event["dataset_record_hash"])
        indexed_version = self._update_index(entry, dataset_hash)
        if not str(indexed_version).strip():
            raise TrustGateError("index updater returned no dataset version")
        final_event = replace(event, state=FeedbackState.MERGED_AND_INDEXED)
        final_hash = self._audit(
            final_event,
            payload_sha256=merged.event.get("payload_sha256"),
            cid=merged.event.get("cid"),
            chain_reference=merged.event.get("chain_reference"),
            replica_reference=merged.event.get("replica_reference"),
            dataset_record_hash=dataset_hash,
            indexed_dataset_version=str(indexed_version),
        )
        return TrustReceipt(
            event_id=event.event_id,
            state=FeedbackState.MERGED_AND_INDEXED,
            payload_sha256=_optional_string(merged.event.get("payload_sha256")),
            cid=_optional_string(merged.event.get("cid")),
            chain_reference=_optional_string(merged.event.get("chain_reference")),
            replica_reference=_optional_string(merged.event.get("replica_reference")),
            dataset_record_hash=dataset_hash,
            indexed_dataset_version=str(indexed_version),
            audit_record_hash=final_hash,
        )

    def _update_index(self, entry: CodebookEntry, dataset_hash: str) -> str:
        if self.index_updater is None:
            raise TrustGateError("an index updater is required by trust policy")
        update = getattr(self.index_updater, "update", None)
        if update is None:
            # Transitional compatibility for the earliest local prototype of
            # this interface; new implementations should expose update().
            update = getattr(self.index_updater, "index", None)
        if update is None:
            raise TrustGateError("index updater does not implement update()")
        try:
            version = update(entry, dataset_record_hash=dataset_hash)
        except TrustGateError:
            raise
        except Exception as exc:
            raise TrustGateError(str(exc)) from exc
        if not str(version).strip():
            raise TrustGateError("index updater returned no dataset version")
        return str(version)

    def _assert_store_matches_completion(
        self,
        event: FeedbackEvent,
        audited: Mapping[str, Any],
    ) -> None:
        entry = event.corrected_entry
        if entry is None:
            raise TrustGateError("completed feedback has no corrected entry")
        stored = self.store.find_by_entry_id(entry.entry_id)
        if stored is None:
            raise TrustGateError("audit reports a merge that is absent from the entry store")
        if stored.event_id != event.event_id or stored.record_hash != audited.get(
            "dataset_record_hash"
        ):
            raise TrustGateError("audit completion does not match the entry store record")

    @staticmethod
    def _assert_same_event(
        event: FeedbackEvent,
        audited: Mapping[str, Any],
        state: FeedbackState,
    ) -> None:
        expected = _event_dict(replace(event, state=state))
        actual = {key: audited.get(key) for key in expected}
        if canonical_json(actual) != canonical_json(expected):
            raise TrustGateError("event_id is already bound to different feedback content")


def _state_key(event: FeedbackEvent, state: FeedbackState) -> str:
    return f"{event.event_id}:{state.value}"


def _feedback_payload(
    event: FeedbackEvent,
    entry: CodebookEntry,
    parent_commitment_hash: str,
) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "request_id": event.request_id,
        "source_result_sha256": event.source_result_sha256.lower()
        if event.source_result_sha256
        else None,
        "signature": event.signature,
        "actor_id": event.actor_id,
        "action": event.action.value,
        "entry": asdict(entry),
        "parent_commitment_hash": parent_commitment_hash,
    }


def _receipt_from_audit(
    event_id: str,
    state: FeedbackState,
    audited: AuditedEvent,
) -> TrustReceipt:
    return TrustReceipt(
        event_id=event_id,
        state=state,
        payload_sha256=_optional_string(audited.event.get("payload_sha256")),
        cid=_optional_string(audited.event.get("cid")),
        chain_reference=_optional_string(audited.event.get("chain_reference")),
        replica_reference=_optional_string(audited.event.get("replica_reference")),
        dataset_record_hash=_optional_string(audited.event.get("dataset_record_hash")),
        indexed_dataset_version=_optional_string(
            audited.event.get("indexed_dataset_version")
        ),
        audit_record_hash=audited.record_hash,
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value)
    return rendered if rendered else None


def _event_dict(event: FeedbackEvent) -> dict[str, Any]:
    value = asdict(event)
    value["action"] = event.action.value
    value["state"] = event.state.value
    return value


def _entry_from_mapping(raw: Mapping[str, Any]) -> CodebookEntry:
    sequence_fields = {
        "sociolinguistic_tags",
        "tone_categories",
        "linguistic_contexts",
        "surface_forms",
        "syntax_patterns",
        "examples",
        "counterexamples",
        "speaker_roles",
        "persona_ids",
        "reviewed_by",
    }
    values = dict(raw)
    for field_name in sequence_fields:
        values[field_name] = tuple(values.get(field_name, ()))
    return CodebookEntry(**values)
