// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title PureChainRegistry
/// @notice Validator-quorum registry for versioned IEDID/IPFS commitments.
/// @dev This is a reference contract. A claim of deployment requires a real address,
///      chain ID, verified bytecode, transaction receipts, and operational validators.
contract PureChainRegistry {
    struct Proposal {
        bytes32 payloadHash;
        bytes32 parentHash;
        string cid;
        uint256 approvals;
        uint256 finalizedBlock;
        bool exists;
        bool finalized;
    }

    uint256 public immutable quorum;
    uint256 public immutable validatorCount;
    bytes32 public finalizedHead;
    mapping(address => bool) public validators;
    mapping(bytes32 => Proposal) private proposals;
    mapping(bytes32 => mapping(address => bool)) public hasApproved;

    event CommitmentProposed(
        bytes32 indexed proposalId,
        bytes32 indexed payloadHash,
        bytes32 indexed parentHash,
        string cid,
        address proposer
    );
    event CommitmentApproved(bytes32 indexed proposalId, address indexed validator, uint256 approvals);
    event CommitmentFinalized(bytes32 indexed proposalId, bytes32 indexed payloadHash, string cid);

    modifier onlyValidator() {
        require(validators[msg.sender], "not validator");
        _;
    }

    constructor(address[] memory initialValidators, uint256 requiredApprovals) {
        require(initialValidators.length > 0, "validators required");
        require(requiredApprovals > 0 && requiredApprovals <= initialValidators.length, "invalid quorum");
        quorum = requiredApprovals;
        uint256 uniqueValidators = 0;
        for (uint256 i = 0; i < initialValidators.length; i++) {
            require(initialValidators[i] != address(0), "zero validator");
            require(!validators[initialValidators[i]], "duplicate validator");
            validators[initialValidators[i]] = true;
            uniqueValidators += 1;
        }
        require(requiredApprovals <= uniqueValidators, "quorum exceeds unique validators");
        validatorCount = uniqueValidators;
    }

    function computeProposalId(bytes32 payloadHash, string memory cid, bytes32 parentHash)
        public
        pure
        returns (bytes32)
    {
        return keccak256(abi.encode(payloadHash, cid, parentHash));
    }

    function propose(
        bytes32 proposalId,
        bytes32 payloadHash,
        string calldata cid,
        bytes32 parentHash
    ) external onlyValidator {
        require(!proposals[proposalId].exists, "proposal exists");
        require(payloadHash != bytes32(0), "payload hash required");
        require(bytes(cid).length > 0, "cid required");
        require(parentHash == finalizedHead, "stale parent");
        require(
            proposalId == computeProposalId(payloadHash, cid, parentHash),
            "non-canonical proposal id"
        );

        proposals[proposalId] = Proposal({
            payloadHash: payloadHash,
            parentHash: parentHash,
            cid: cid,
            approvals: 0,
            finalizedBlock: 0,
            exists: true,
            finalized: false
        });
        emit CommitmentProposed(proposalId, payloadHash, parentHash, cid, msg.sender);
        _approve(proposalId, msg.sender);
    }

    function approve(bytes32 proposalId) external onlyValidator {
        _approve(proposalId, msg.sender);
    }

    function _approve(bytes32 proposalId, address validator) internal {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.exists, "unknown proposal");
        require(!proposal.finalized, "already finalized");
        require(!hasApproved[proposalId][validator], "already approved");
        require(proposal.parentHash == finalizedHead, "stale parent");

        hasApproved[proposalId][validator] = true;
        proposal.approvals += 1;
        emit CommitmentApproved(proposalId, validator, proposal.approvals);

        if (proposal.approvals >= quorum) {
            proposal.finalized = true;
            proposal.finalizedBlock = block.number;
            finalizedHead = proposal.payloadHash;
            emit CommitmentFinalized(proposalId, proposal.payloadHash, proposal.cid);
        }
    }

    function proposalExists(bytes32 proposalId) external view returns (bool) {
        return proposals[proposalId].exists;
    }

    function isFinalized(bytes32 proposalId) external view returns (bool) {
        return proposals[proposalId].finalized;
    }

    function getProposal(bytes32 proposalId)
        external
        view
        returns (
            bytes32 payloadHash,
            bytes32 parentHash,
            string memory cid,
            uint256 approvals,
            bool finalized,
            uint256 finalizedBlock
        )
    {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.exists, "unknown proposal");
        return (
            proposal.payloadHash,
            proposal.parentHash,
            proposal.cid,
            proposal.approvals,
            proposal.finalized,
            proposal.finalizedBlock
        );
    }
}
