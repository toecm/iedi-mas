from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from iedi.provenance import InMemoryChain, Web3PureChain


def test_in_memory_chain_is_idempotent_and_rejects_stale_parent() -> None:
    chain = InMemoryChain()
    first = chain.anchor(payload_sha256="a" * 64, cid="cid-a", parent_sha256="0" * 64)
    assert chain.anchor(
        payload_sha256="a" * 64, cid="cid-a", parent_sha256="0" * 64
    ) == first
    assert len(chain.commitments) == 1
    with pytest.raises(RuntimeError, match="stale parent"):
        chain.anchor(payload_sha256="b" * 64, cid="cid-b", parent_sha256="0" * 64)


class _Action:
    def __init__(self, *, call=None, transact=None):
        self._call = call
        self._transact = transact

    def call(self):
        return self._call() if callable(self._call) else self._call

    def transact(self, options):
        return self._transact(options)


class _FakeContractFunctions:
    def __init__(self) -> None:
        self.proposals = {}
        self.approved = set()
        self.propose_count = 0
        self.finality_checks = 0

    def computeProposalId(self, payload, cid, parent):
        proposal_id = hashlib.sha256(payload + cid.encode() + parent).digest()
        return _Action(call=proposal_id)

    def proposalExists(self, proposal_id):
        return _Action(call=lambda: proposal_id in self.proposals)

    def propose(self, proposal_id, payload, cid, parent):
        def transact(options):
            self.propose_count += 1
            self.proposals[proposal_id] = [payload, parent, cid, 1, False, 0]
            self.approved.add((proposal_id, options["from"]))
            return b"proposal-tx"

        return _Action(transact=transact)

    def hasApproved(self, proposal_id, sender):
        return _Action(call=lambda: (proposal_id, sender) in self.approved)

    def approve(self, proposal_id):
        def transact(options):
            proposal = self.proposals[proposal_id]
            proposal[3] += 1
            self.approved.add((proposal_id, options["from"]))
            return b"approval-tx"

        return _Action(transact=transact)

    def isFinalized(self, proposal_id):
        def call():
            self.finality_checks += 1
            proposal = self.proposals[proposal_id]
            # Simulate a second operational validator approving asynchronously.
            if self.finality_checks >= 3:
                proposal[3] = 2
                proposal[4] = True
                proposal[5] = 10
            return proposal[4]

        return _Action(call=call)

    def getProposal(self, proposal_id):
        return _Action(call=lambda: tuple(self.proposals[proposal_id]))


def test_web3_adapter_waits_for_quorum_and_retry_does_not_repropose() -> None:
    functions = _FakeContractFunctions()
    contract = SimpleNamespace(functions=functions)
    eth = SimpleNamespace(
        block_number=11,
        wait_for_transaction_receipt=lambda tx, timeout=None: SimpleNamespace(status=1),
    )
    adapter = Web3PureChain(
        web3=SimpleNamespace(eth=eth),
        contract=contract,
        sender="validator-1",
        confirmations=2,
        finality_timeout_s=0.1,
        poll_interval_s=0.001,
    )
    kwargs = {
        "payload_sha256": "a" * 64,
        "cid": "bafy-example",
        "parent_sha256": "0" * 64,
    }
    first = adapter.anchor(**kwargs)
    assert first.startswith("0x")
    assert adapter.anchor(**kwargs) == first
    assert functions.propose_count == 1


def test_contract_enforces_canonical_linear_commitments() -> None:
    source = (
        Path(__file__).parents[1] / "contracts" / "PureChainRegistry.sol"
    ).read_text(encoding="utf-8")
    assert "proposalId == computeProposalId" in source
    assert 'require(parentHash == finalizedHead, "stale parent")' in source
    assert 'require(!validators[initialValidators[i]], "duplicate validator")' in source
    assert "finalizedHead = proposal.payloadHash" in source
