from __future__ import annotations

from pathlib import Path

import pytest

from iedi.pipeline import load_paper_profile


def test_paper_configs_drive_model_and_policy_values() -> None:
    root = Path(__file__).parents[1]
    paper3 = load_paper_profile(root / "configs" / "paper3.json")
    assert paper3.require_active_persona
    assert paper3.return_three_for_unresolved_polysemy
    assert paper3.dmm_policy.flash_model_id == "gemini-2.5-flash"
    assert paper3.dmm_policy.pro_model_id == "gemini-2.5-pro"
    assert paper3.dmm_policy.local_margin == 0.05

    paper5 = load_paper_profile(root / "configs" / "paper5.json")
    assert paper5.dmm_policy.cold_start_requests == 50
    assert paper5.use_acoustic_affect
    assert paper5.trust_gate_required
    assert paper5.require_ipfs_verification
    assert paper5.require_finalized_chain_commitment
    assert "deployed_quorum_contract" in paper5.missing_claim_evidence({})
    with pytest.raises(ValueError, match="claim evidence is missing"):
        paper5.require_claim_evidence({})
