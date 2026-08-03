import importlib.util
import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_package", ROOT / "scripts" / "validate_package.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PackageIntegrityTests(unittest.TestCase):
    def test_package_integrity(self):
        self.assertEqual(MODULE.validate(), [])

    def test_test_case_arithmetic(self):
        design = MODULE.load_json("config/study_design.json")
        self.assertEqual(design["splits"]["sealed_test"]["families"], 150)
        self.assertEqual(design["cases_per_retained_family"]["total"], 8)
        self.assertEqual(design["sealed_test_cases"], 1200)

    def test_design_rejects_action_coverage_cap_drift(self):
        design = MODULE.load_json("config/study_design.json")
        runtime_schema = MODULE.load_json("data/schemas/runtime_episode.schema.json")
        design["primary_task"]["maximum_action_appropriate_coverage"] = 0.80

        def fake_load_json(relative):
            if relative == "config/study_design.json":
                return design
            if relative == "data/schemas/runtime_episode.schema.json":
                return runtime_schema
            return MODULE.load_json(relative)

        errors = []
        with patch.object(MODULE, "load_json", side_effect=fake_load_json):
            MODULE.check_design(errors)
        self.assertTrue(any("coverage cap" in error for error in errors))

    def test_safety_and_evidence_locks(self):
        design = MODULE.load_json("config/study_design.json")
        self.assertTrue(
            design["intervention_policy"]["protected_identity_interventions_prohibited"]
        )
        self.assertFalse(design["iedid"]["sealed_test_gold_allowed"])
        self.assertFalse(
            design["result_policy"]["synthetic_results_as_empirical_evidence"]
        )
        self.assertFalse(design["application_pilot"]["confirmatory_inference_allowed"])
        self.assertFalse(design["application_pilot"]["reuse_confirmatory_test_for_tuning"])
        self.assertFalse(
            design["minimum_public_release"][
                "release_before_ethics_license_consent_and_community_review"
            ]
        )

    def test_runtime_view_has_no_gold_labels(self):
        design = MODULE.load_json("config/study_design.json")
        schema = MODULE.load_json("data/schemas/runtime_episode.schema.json")
        self.assertFalse(
            design["intervention_policy"]["sealed_reference_labels_visible_to_inference"]
        )
        forbidden = {
            "reference_action",
            "reference_sense_id",
            "acceptable_clarification_slots",
            "case_type",
        }
        self.assertTrue(forbidden.isdisjoint(schema["properties"]))

    def test_rewritten_notebook_is_output_free_and_pinned(self):
        notebook = json.loads(
            (ROOT / "notebooks" / "CA_IEDI_0803.ipynb").read_text(encoding="utf-8")
        )
        metadata = notebook["metadata"]["cdcv_gate"]
        self.assertEqual(
            metadata["upstream_commit"],
            "5cff1e509efb09c24f9ac7e30075b6a131ee6fbc",
        )
        self.assertEqual(metadata["run_mode_default"], "DEMO")
        self.assertTrue(
            all(
                cell.get("outputs", []) == [] and cell.get("execution_count") is None
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )

    def test_checked_in_notebook_equals_builder_document(self):
        notebook = json.loads(
            (ROOT / "notebooks" / "CA_IEDI_0803.ipynb").read_text(encoding="utf-8")
        )
        errors = []
        expected = MODULE._load_builder_notebook(errors)
        self.assertEqual(errors, [])
        self.assertEqual(notebook, expected)

    def test_root_notebook_compatibility_mirror_matches(self):
        mirror_path = ROOT.parent / "CA_IEDI_0803.ipynb"
        if not mirror_path.is_file():
            self.skipTest("package is not installed beside a repository mirror")
        canonical = json.loads(
            (ROOT / "notebooks" / "CA_IEDI_0803.ipynb").read_text(encoding="utf-8")
        )
        mirror = json.loads(mirror_path.read_text(encoding="utf-8"))
        self.assertEqual(mirror, canonical)

    def test_builder_equality_check_rejects_notebook_drift(self):
        expected = {"nbformat": 4, "cells": []}
        drifted = {"nbformat": 4, "cells": [{"cell_type": "markdown"}]}
        errors = []
        MODULE.check_notebook_matches_builder(drifted, expected, errors)
        self.assertTrue(any("differs" in error for error in errors))

    def test_provenance_chronology_rejects_retrieval_before_commit(self):
        provenance = {
            "source_commit_timestamp_utc": "2026-08-03T14:13:15Z",
            "retrieved_for_audit_utc": "2026-08-03T14:13:14Z",
        }
        errors = []
        MODULE.check_provenance_chronology(provenance, errors)
        self.assertTrue(any("predates" in error for error in errors))

    def test_provenance_chronology_requires_explicit_utc(self):
        provenance = {
            "source_commit_timestamp_utc": "2026-08-03T14:13:15",
            "retrieved_for_audit_utc": "not-a-date",
        }
        errors = []
        MODULE.check_provenance_chronology(provenance, errors)
        self.assertTrue(any("explicitly use UTC" in error for error in errors))
        self.assertTrue(any("valid ISO-8601" in error for error in errors))

    @staticmethod
    def _code_cell(source, *, outputs=None, execution_count=None):
        return {
            "cell_type": "code",
            "metadata": {},
            "source": source.splitlines(keepends=True),
            "outputs": [] if outputs is None else outputs,
            "execution_count": execution_count,
        }

    def test_notebook_static_check_accepts_explicit_read_only_locks(self):
        cell = self._code_cell(
            "def require(condition, message):\n"
            "    if not condition:\n"
            "        raise RuntimeError(message)\n"
            "RUN_MODE = 'DEMO'\n"
            "RESULTS_LOCKED = True\n"
            "require(RUN_MODE == 'DEMO', 'mode')\n"
            "require(RESULTS_LOCKED, 'results')\n"
            "from pathlib import Path\n"
            "text = Path('fixture.json').read_text(encoding='utf-8')\n"
        )
        errors = []
        MODULE.check_notebook_code_cells([cell], errors)
        self.assertEqual(errors, [])

    def test_notebook_static_check_rejects_assert_network_and_writes(self):
        cell = self._code_cell(
            "def require(condition, message):\n"
            "    if not condition:\n"
            "        raise RuntimeError(message)\n"
            "RUN_MODE = 'DEMO'\n"
            "RESULTS_LOCKED = True\n"
            "require(RUN_MODE == 'DEMO', 'mode')\n"
            "require(RESULTS_LOCKED, 'results')\n"
            "assert RESULTS_LOCKED\n"
            "import requests\n"
            "from pathlib import Path\n"
            "Path('leak.txt').write_text('x')\n"
        )
        errors = []
        MODULE.check_notebook_code_cells([cell], errors)
        self.assertTrue(any("uses assert" in error for error in errors))
        self.assertTrue(any("requests" in error for error in errors))
        self.assertTrue(any("write_text" in error for error in errors))

    def test_notebook_static_check_rejects_saved_execution_state(self):
        cell = self._code_cell(
            "def require(condition, message):\n"
            "    if not condition:\n"
            "        raise RuntimeError(message)\n"
            "RUN_MODE = 'DEMO'\n"
            "RESULTS_LOCKED = True\n"
            "require(RUN_MODE == 'DEMO', 'mode')\n"
            "require(RESULTS_LOCKED, 'results')\n",
            outputs=[{"output_type": "stream", "name": "stdout", "text": "x"}],
            execution_count=1,
        )
        errors = []
        MODULE.check_notebook_code_cells([cell], errors)
        self.assertTrue(any("execution count" in error for error in errors))
        self.assertTrue(any("empty outputs" in error for error in errors))

    def test_notebook_static_check_rejects_non_explicit_lock_helper(self):
        cell = self._code_cell(
            "RUN_MODE = 'DEMO'\nRESULTS_LOCKED = True\nassert RESULTS_LOCKED\n"
        )
        errors = []
        MODULE.check_notebook_code_cells([cell], errors)
        self.assertTrue(any("explicit raise" in error for error in errors))
        self.assertTrue(any("RUN_MODE lock" in error for error in errors))
        self.assertTrue(any("RESULTS_LOCKED lock" in error for error in errors))

    def test_notebook_smoke_runner_executes_all_cells(self):
        executor_spec = importlib.util.spec_from_file_location(
            "execute_notebook", ROOT / "scripts" / "execute_notebook.py"
        )
        executor = importlib.util.module_from_spec(executor_spec)
        assert executor_spec.loader is not None
        executor_spec.loader.exec_module(executor)
        namespace = executor.execute(ROOT / "notebooks" / "CA_IEDI_0803.ipynb")
        summary = namespace["smoke_summary"]
        self.assertFalse(summary["sealed_results_present"])
        self.assertEqual(summary["commit_path_calls"], 3)
        self.assertEqual(summary["clarify_repair_path_calls"], 6)
        self.assertEqual(summary["repair_route_path_calls"], 9)
        previous = Path.cwd()
        try:
            os.chdir(ROOT / "notebooks")
            nested_namespace = executor.execute(ROOT / "notebooks" / "CA_IEDI_0803.ipynb")
        finally:
            os.chdir(previous)
        self.assertEqual(nested_namespace["smoke_summary"]["commit_path_calls"], 3)

    def _integrity_fixture(self):
        runtime = {
            "case_id": "case-1",
            "family_id": "family-1",
            "utterance_hash": "a" * 64,
            "candidate_set_hash": "b" * 64,
            "context_card_hash": "c" * 64,
            "candidate_senses": [
                {"candidate_id": "sense_a"},
                {"candidate_id": "sense_b"},
                {"candidate_id": "OTHER_UNLISTED"},
            ],
        }
        source = {
            "variety_cue": {"value": "community_resource"},
            "fields": {"discourse_goal": {"value": "request"}},
        }
        runtime["context_card_hash"] = MODULE.canonical_sha256(source)
        results = {}

        def probe(probe_id, result_character, target=None):
            result = {
                "variety_cue": deepcopy(source["variety_cue"]),
                "fields": {
                    "discourse_goal": {
                        "value": f"{probe_id}-{result_character}"
                    }
                },
            }
            result_hash = MODULE.canonical_sha256(result)
            results[result_hash] = result
            value = {
                "intervention_id": probe_id,
                "source_context_hash": runtime["context_card_hash"],
                "result_context_hash": result_hash,
                "changed_slots": ["discourse_goal"],
            }
            if target is not None:
                value["target_candidate_id"] = target
            return value

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
                    "preserving": [probe("pa", "d")],
                    "meaning_changing": [probe("cab", "e", "sense_b")],
                },
                {
                    "source_candidate_id": "sense_b",
                    "preserving": [probe("pb", "f")],
                    "meaning_changing": [probe("cba", "1", "sense_a")],
                },
            ],
        }
        return runtime, bundle, source, results

    def test_cross_record_intervention_integrity_accepts_symmetric_bundle(self):
        runtime, bundle, source, results = self._integrity_fixture()
        self.assertEqual(
            MODULE.validate_intervention_bundle_integrity(
                runtime, bundle, source, results
            ),
            [],
        )

    def test_cross_record_intervention_integrity_rejects_duplicate_branch(self):
        runtime, bundle, source, results = self._integrity_fixture()
        bundle["candidate_branches"][1]["source_candidate_id"] = "sense_a"
        errors = MODULE.validate_intervention_bundle_integrity(
            runtime, bundle, source, results
        )
        self.assertTrue(any("not exactly symmetric" in error for error in errors))

    def test_cross_record_intervention_integrity_rejects_variety_change(self):
        runtime, bundle, source, results = self._integrity_fixture()
        first_hash = bundle["candidate_branches"][0]["preserving"][0][
            "result_context_hash"
        ]
        results[first_hash]["variety_cue"] = {"value": "changed"}
        errors = MODULE.validate_intervention_bundle_integrity(
            runtime, bundle, source, results
        )
        self.assertTrue(any("fixed variety" in error for error in errors))

    def test_cross_record_intervention_integrity_rejects_noncanonical_hashes(self):
        runtime, bundle, source, results = self._integrity_fixture()
        runtime["context_card_hash"] = "0" * 64
        errors = MODULE.validate_intervention_bundle_integrity(
            runtime, bundle, source, results
        )
        self.assertTrue(any("context-card hash is not canonical" in error for error in errors))

        runtime, bundle, source, results = self._integrity_fixture()
        first_hash = bundle["candidate_branches"][0]["preserving"][0][
            "result_context_hash"
        ]
        results[first_hash]["fields"]["discourse_goal"]["value"] = "tampered"
        errors = MODULE.validate_intervention_bundle_integrity(
            runtime, bundle, source, results
        )
        self.assertTrue(any("result hash is not canonical" in error for error in errors))

    def test_primary_clarification_manifest_reuses_probes_and_sums_priors(self):
        question = {
            "question_id": "q1",
            "family_id": "f1",
            "candidate_set_hash": "a" * 64,
            "context_slot": "discourse_goal",
            "answer_domain": [{"answer_id": "a"}, {"answer_id": "b"}],
        }
        manifest = {
            "question_id": "q1",
            "family_id": "f1",
            "candidate_set_hash": "a" * 64,
            "context_slot": "discourse_goal",
            "mode": "PRIMARY_REUSE_ONLY",
            "additional_model_calls": 0,
            "scenarios": [
                {
                    "answer_id": "a",
                    "prior_probability": 0.5,
                    "score_source": {
                        "source_type": "REUSED_PROBE_SCORES",
                        "probe_id": "p1",
                    },
                },
                {
                    "answer_id": "b",
                    "prior_probability": 0.5,
                    "score_source": {
                        "source_type": "REUSED_PROBE_SCORES",
                        "probe_id": "p2",
                    },
                },
            ],
        }
        self.assertEqual(
            MODULE.validate_clarification_scenario_manifest(
                question, manifest, {"p1", "p2"}
            ),
            [],
        )
        manifest["scenarios"][1]["prior_probability"] = 0.4
        errors = MODULE.validate_clarification_scenario_manifest(
            question, manifest, {"p1", "p2"}
        )
        self.assertTrue(any("sum to one" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
