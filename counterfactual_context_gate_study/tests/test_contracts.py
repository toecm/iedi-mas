import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdcv_gate.contracts import (  # noqa: E402
    OTHER_UNLISTED,
    apply_context_patch,
    freeze_prediction_hash,
    join_labels_after_prediction_freeze,
    sha256_json,
    validate_context_card,
    validate_intervention_bundle_integrity,
    validate_runtime_episode_gold_free,
)


def _field(value):
    return {
        "value": value,
        "provenance": "benchmark_assignment",
        "confidence": 1.0,
        "retain_after_episode": False,
    }


def _context_card():
    return {
        "episode_id": "CASE_001",
        "mode": "benchmark",
        "scope": "current_interaction_only",
        "expires_at": None,
        "variety_cue": {
            "value": "reviewed_demo_variety",
            "provenance": "experimentally_supplied",
            "retain_after_episode": False,
        },
        "fields": {
            "relationship_role": _field("colleagues"),
            "setting": _field("project room"),
            "formality": _field("informal"),
            "discourse_goal": _field("request assistance"),
            "preceding_speech_act": _field("offer of help"),
            "situation": _field("shared task"),
        },
        "conflicts": [],
    }


def _integrity_fixture():
    source = _context_card()
    candidates = [
        {"candidate_id": "sense_a", "definition": "first reviewed meaning"},
        {"candidate_id": "sense_b", "definition": "second reviewed meaning"},
        {"candidate_id": OTHER_UNLISTED, "definition": "none of the above"},
    ]
    runtime = {
        "case_id": "CASE_001",
        "family_id": "FAMILY_001",
        "utterance_hash": "a" * 64,
        "candidate_set_hash": sha256_json(candidates),
        "context_card_hash": sha256_json(source),
        "candidate_senses": candidates,
    }
    resulting_contexts = {}

    def probe(probe_id, source_candidate, value, target=None):
        result = apply_context_patch(
            source,
            ("discourse_goal",),
            {"discourse_goal": _field(value)},
        )
        result_hash = sha256_json(result)
        resulting_contexts[result_hash] = result
        item = {
            "intervention_id": probe_id,
            "source_context_hash": runtime["context_card_hash"],
            "result_context_hash": result_hash,
            "changed_slots": ["discourse_goal"],
        }
        if target is not None:
            item["target_candidate_id"] = target
        return item

    bundle = {
        "family_id": runtime["family_id"],
        "base_case_id": runtime["case_id"],
        "utterance_hash": runtime["utterance_hash"],
        "candidate_set_hash": runtime["candidate_set_hash"],
        "variety_cue_fixed": True,
        "constructed_without_sealed_reference_action": True,
        "validation": {"status": "accepted"},
        "candidate_branches": [
            {
                "source_candidate_id": "sense_a",
                "preserving": [probe("preserve_a", "sense_a", "request help politely")],
                "meaning_changing": [
                    probe("change_a_to_b", "sense_a", "contrastive goal b", "sense_b")
                ],
            },
            {
                "source_candidate_id": "sense_b",
                "preserving": [probe("preserve_b", "sense_b", "contrastive goal b politely")],
                "meaning_changing": [
                    probe("change_b_to_a", "sense_b", "request help directly", "sense_a")
                ],
            },
        ],
    }
    return runtime, bundle, source, resulting_contexts


class RecursiveAccessControlTests(unittest.TestCase):
    def test_context_card_rejects_nested_sealed_and_protected_keys(self):
        card = _context_card()
        card["fields"]["situation"]["value"] = {
            "safe_summary": "shared task",
            "nested": [
                {"gold_action": "COMMIT"},
                {"profile": {"race": "prohibited"}},
            ],
        }

        errors = validate_context_card(card)

        self.assertTrue(any("gold_action" in error for error in errors))
        self.assertTrue(any("race" in error for error in errors))

    def test_runtime_episode_rejects_nested_sealed_and_protected_keys(self):
        runtime = {
            "candidate_senses": [
                {
                    "candidate_id": "sense_a",
                    "definition": "first",
                    "metadata": [{"reference_sense_id": "sense_a"}],
                },
                {
                    "candidate_id": "sense_b",
                    "definition": "second",
                    "metadata": {"persona": {"religion": "prohibited"}},
                },
                {"candidate_id": OTHER_UNLISTED, "definition": "other"},
            ]
        }

        errors = validate_runtime_episode_gold_free(runtime)

        self.assertTrue(any("reference_sense_id" in error for error in errors))
        self.assertTrue(any("religion" in error for error in errors))


class CanonicalInterventionHashTests(unittest.TestCase):
    def test_valid_bundle_uses_canonical_source_and_result_hashes(self):
        runtime, bundle, source, resulting_contexts = _integrity_fixture()

        self.assertEqual(
            validate_intervention_bundle_integrity(
                runtime, bundle, source, resulting_contexts
            ),
            [],
        )

    def test_rejects_source_context_when_canonical_hash_changes(self):
        runtime, bundle, source, resulting_contexts = _integrity_fixture()
        source["fields"]["setting"]["value"] = "changed after hashing"

        errors = validate_intervention_bundle_integrity(
            runtime, bundle, source, resulting_contexts
        )

        self.assertTrue(any("canonical source context" in error for error in errors))

    def test_rejects_result_context_when_canonical_hash_changes(self):
        runtime, bundle, source, resulting_contexts = _integrity_fixture()
        result_hash = bundle["candidate_branches"][0]["preserving"][0][
            "result_context_hash"
        ]
        resulting_contexts[result_hash]["fields"]["setting"]["value"] = (
            "changed after hashing"
        )

        errors = validate_intervention_bundle_integrity(
            runtime, bundle, source, resulting_contexts
        )

        self.assertTrue(any("result hash is not canonical" in error for error in errors))

    def test_result_context_recursively_rejects_protected_data(self):
        runtime, bundle, source, resulting_contexts = _integrity_fixture()
        probe = bundle["candidate_branches"][0]["preserving"][0]
        original_hash = probe["result_context_hash"]
        result = deepcopy(resulting_contexts.pop(original_hash))
        result["fields"]["situation"]["value"] = {"profile": {"ethnicity": "x"}}
        new_hash = sha256_json(result)
        resulting_contexts[new_hash] = result
        probe["result_context_hash"] = new_hash

        errors = validate_intervention_bundle_integrity(
            runtime, bundle, source, resulting_contexts
        )

        self.assertTrue(any("ethnicity" in error for error in errors))


class PredictionFreezeJoinTests(unittest.TestCase):
    def test_rejects_duplicate_prediction_case_ids(self):
        predictions = [{"case_id": "A"}, {"case_id": "A"}]

        with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
            join_labels_after_prediction_freeze(
                predictions,
                [{"case_id": "A"}],
                frozen_prediction_hash=freeze_prediction_hash(predictions),
            )

    def test_rejects_duplicate_label_case_ids(self):
        predictions = [{"case_id": "A"}]

        with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
            join_labels_after_prediction_freeze(
                predictions,
                [{"case_id": "A"}, {"case_id": "A"}],
                frozen_prediction_hash=freeze_prediction_hash(predictions),
            )

    def test_rejects_missing_or_extra_label_cases(self):
        predictions = [{"case_id": "A"}, {"case_id": "B"}]

        with self.assertRaisesRegex(ValueError, "case sets must match exactly"):
            join_labels_after_prediction_freeze(
                predictions,
                [{"case_id": "A"}, {"case_id": "C"}],
                frozen_prediction_hash=freeze_prediction_hash(predictions),
            )


if __name__ == "__main__":
    unittest.main()
