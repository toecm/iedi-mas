from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Any, Mapping, Protocol

from .codebook import canonical_json


class IPFSSink(Protocol):
    def pin(self, payload: Mapping[str, Any]) -> str: ...

    def verify(self, cid: str, expected_sha256: str) -> bool: ...


class ChainSink(Protocol):
    def anchor(self, *, payload_sha256: str, cid: str, parent_sha256: str) -> str: ...


@dataclass
class InMemoryIPFS:
    objects: dict[str, bytes]

    def __init__(self) -> None:
        self.objects = {}

    def pin(self, payload: Mapping[str, Any]) -> str:
        body = canonical_json(payload)
        digest = hashlib.sha256(body).hexdigest()
        cid = f"bafy-test-{digest}"
        self.objects[cid] = body
        return cid

    def verify(self, cid: str, expected_sha256: str) -> bool:
        body = self.objects.get(cid)
        return body is not None and hashlib.sha256(body).hexdigest() == expected_sha256


@dataclass
class InMemoryChain:
    commitments: list[dict[str, str]]

    def __init__(self) -> None:
        self.commitments = []
        self._by_commitment: dict[tuple[str, str, str], str] = {}
        self._head = "0" * 64
        self._lock = threading.Lock()

    def anchor(self, *, payload_sha256: str, cid: str, parent_sha256: str) -> str:
        key = (payload_sha256, cid, parent_sha256)
        with self._lock:
            existing = self._by_commitment.get(key)
            if existing is not None:
                return existing
            if parent_sha256 != self._head:
                raise RuntimeError("chain commitment has a stale parent")
            reference = hashlib.sha256(
                canonical_json(
                    {
                        "payload_sha256": payload_sha256,
                        "cid": cid,
                        "parent_sha256": parent_sha256,
                    }
                )
            ).hexdigest()
            self.commitments.append(
                {
                    "reference": reference,
                    "payload_sha256": payload_sha256,
                    "cid": cid,
                    "parent_sha256": parent_sha256,
                }
            )
            self._by_commitment[key] = reference
            self._head = payload_sha256
            return reference


class PinataIPFS:
    def __init__(
        self,
        *,
        jwt: str | None = None,
        gateway_base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.jwt = jwt or os.getenv("PINATA_JWT")
        if not self.jwt:
            raise ValueError("PINATA_JWT is required")
        self.gateway_base_url = (gateway_base_url or os.getenv("PINATA_GATEWAY", "")).rstrip("/")
        self.timeout_s = timeout_s

    def pin(self, payload: Mapping[str, Any]) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the provenance extra") from exc
        response = requests.post(
            "https://api.pinata.cloud/pinning/pinJSONToIPFS",
            headers={"Authorization": f"Bearer {self.jwt}"},
            json={"pinataContent": dict(payload)},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        cid = str(response.json().get("IpfsHash", "")).strip()
        if not cid:
            raise RuntimeError("Pinata response did not include IpfsHash")
        return cid

    def verify(self, cid: str, expected_sha256: str) -> bool:
        if not self.gateway_base_url:
            raise RuntimeError("PINATA_GATEWAY is required for content verification")
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the provenance extra") from exc
        response = requests.get(
            f"{self.gateway_base_url}/ipfs/{cid}",
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        return hashlib.sha256(canonical_json(payload)).hexdigest() == expected_sha256


class Web3PureChain:
    """Adapter for contracts/PureChainRegistry.sol; waits for receipt and finalization."""

    def __init__(
        self,
        *,
        web3: Any,
        contract: Any,
        sender: str,
        confirmations: int = 1,
        finality_timeout_s: float = 60.0,
        poll_interval_s: float = 0.5,
    ) -> None:
        if confirmations < 1:
            raise ValueError("at least one confirmation is required")
        if finality_timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("finality timeout and poll interval must be positive")
        self.web3 = web3
        self.contract = contract
        self.sender = sender
        self.confirmations = confirmations
        self.finality_timeout_s = finality_timeout_s
        self.poll_interval_s = poll_interval_s

    def anchor(self, *, payload_sha256: str, cid: str, parent_sha256: str) -> str:
        payload_hash = bytes.fromhex(payload_sha256)
        parent_hash = bytes.fromhex(parent_sha256)
        if len(payload_hash) != 32 or len(parent_hash) != 32 or not cid.strip():
            raise ValueError("payload, parent and CID are required for chain anchoring")
        proposal_id = self.contract.functions.computeProposalId(
            payload_hash, cid, parent_hash
        ).call()

        exists = bool(self.contract.functions.proposalExists(proposal_id).call())
        if exists:
            proposal = self.contract.functions.getProposal(proposal_id).call()
            if not _proposal_matches(proposal, payload_hash, parent_hash, cid):
                raise RuntimeError("existing PureChain proposal does not match commitment")
        else:
            transaction_hash = self.contract.functions.propose(
                proposal_id,
                payload_hash,
                cid,
                parent_hash,
            ).transact({"from": self.sender})
            self._wait_for_success(transaction_hash, "proposal")

        if not bool(self.contract.functions.isFinalized(proposal_id).call()):
            already_approved = bool(
                self.contract.functions.hasApproved(proposal_id, self.sender).call()
            )
            if not already_approved:
                transaction_hash = self.contract.functions.approve(proposal_id).transact(
                    {"from": self.sender}
                )
                self._wait_for_success(transaction_hash, "approval")

        deadline = monotonic() + self.finality_timeout_s
        while not bool(self.contract.functions.isFinalized(proposal_id).call()):
            if monotonic() >= deadline:
                raise TimeoutError(
                    "PureChain quorum was not finalized before the configured timeout"
                )
            sleep(min(self.poll_interval_s, max(deadline - monotonic(), 0.0)))

        proposal = self.contract.functions.getProposal(proposal_id).call()
        if not _proposal_matches(proposal, payload_hash, parent_hash, cid):
            raise RuntimeError("finalized PureChain proposal changed unexpectedly")
        finalized_block = int(proposal[5])
        target_block = finalized_block + self.confirmations - 1
        while int(self.web3.eth.block_number) < target_block:
            if monotonic() >= deadline:
                raise TimeoutError(
                    "PureChain commitment lacks the configured block confirmations"
                )
            sleep(min(self.poll_interval_s, max(deadline - monotonic(), 0.0)))
        return _hex_value(proposal_id)

    def _wait_for_success(self, transaction_hash: Any, operation: str) -> Any:
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(
                transaction_hash, timeout=self.finality_timeout_s
            )
        except TypeError:  # small fake clients and older Web3 releases
            receipt = self.web3.eth.wait_for_transaction_receipt(transaction_hash)
        status = receipt.status if hasattr(receipt, "status") else receipt["status"]
        if int(status) != 1:
            raise RuntimeError(f"PureChain {operation} transaction reverted")
        return receipt


def _proposal_matches(
    proposal: Any, payload_hash: bytes, parent_hash: bytes, cid: str
) -> bool:
    return (
        bytes(proposal[0]) == payload_hash
        and bytes(proposal[1]) == parent_hash
        and str(proposal[2]) == cid
    )


def _hex_value(value: Any) -> str:
    if hasattr(value, "hex"):
        rendered = value.hex()
        return rendered if str(rendered).startswith("0x") else f"0x{rendered}"
    return f"0x{bytes(value).hex()}"
