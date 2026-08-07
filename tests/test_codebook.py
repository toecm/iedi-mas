from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from iedi.codebook import Codebook
from iedi.schemas import CodebookEntry, InterpretationRequest


def test_demo_codebook_has_versioned_personas(codebook: Codebook) -> None:
    assert len(codebook.entries) == 9
    assert set(codebook.personas) == {
        "ng-en-v1",
        "am-en-v1",
        "ko-en-v1",
        "ind-en-v1",
        "id-en-pending-v1",
    }
    assert codebook.personas["id-en-pending-v1"].review_status == "pending"
    assert all(profile.content_hash for profile in codebook.personas.values())


def test_invalid_string_rule_fails_before_inference() -> None:
    path = Path(__file__).parents[1] / "data" / "codebook.demo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["personas"][0]["pragmatic_rules"] = ["Be concise"]
    with pytest.raises(ValueError, match="objects, not strings"):
        Codebook.from_dict(payload)


def test_polysemy_is_preserved_without_context(codebook: Codebook) -> None:
    request = InterpretationRequest("I beg", active_persona_ids=("ng-en-v1",))
    results = codebook.search(request)
    assert results[0].score == 1.0
    assert results[1].score == 1.0
    assert {results[0].entry.entry_id, results[1].entry.entry_id} == {
        "ng-i-beg-please",
        "ng-i-beg-seriously",
    }
    assert codebook.is_polysemous_surface("I beg")


def test_validated_rule_prioritizes_before_final_selection(codebook: Codebook) -> None:
    request = InterpretationRequest(
        "I beg",
        active_persona_ids=("ng-en-v1",),
        supplied_tone="Casual",
        supplied_context="discourse marker used to soften commands",
    )
    results = codebook.search(request)
    assert results[0].entry.entry_id == "ng-i-beg-please"
    assert results[0].method == "persona_rule"
    assert results[1].method == "contextually_deprioritized"
    assert results[0].score > results[1].score


def test_arbitrary_nested_quantifier_is_rejected() -> None:
    path = Path(__file__).parents[1] / "data" / "codebook.demo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["syntax_patterns"] = ["(a+)+"]
    with pytest.raises(ValueError, match="unsafe or unsupported regex"):
        Codebook.from_dict(payload)


def test_unbounded_wildcard_regex_is_rejected() -> None:
    path = Path(__file__).parents[1] / "data" / "codebook.demo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["syntax_patterns"] = ["start.*end"]
    with pytest.raises(ValueError, match="unsafe or unsupported regex"):
        Codebook.from_dict(payload)


def test_edit_appends_version_and_retains_superseded_history(codebook: Codebook) -> None:
    previous = codebook.get_entry("ng-wahala-1")
    replacement = CodebookEntry(
        **{
            **previous.__dict__,
            "entry_id": "ng-wahala-2",
            "universal_gloss": "reviewed updated gloss",
            "version": 2,
            "supersedes_entry_id": previous.entry_id,
        }
    )
    updated = codebook.append_version(replacement)
    assert updated.get_entry(previous.entry_id).review_status == "superseded"
    assert updated.get_entry(replacement.entry_id).review_status == "approved"
    assert len(updated.entries) == len(codebook.entries) + 1
    result = updated.search(
        InterpretationRequest("wahala", active_persona_ids=("ng-en-v1",))
    )
    assert result[0].entry.entry_id == replacement.entry_id
    assert replacement.entry_id in updated.personas["ng-en-v1"].entry_ids
    assert previous.entry_id not in updated.personas["ng-en-v1"].entry_ids
