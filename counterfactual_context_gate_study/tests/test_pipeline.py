import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdcv_gate import (  # noqa: E402
    Action,
    AttestationStatus,
    BudgetEnvelope,
    CandidateBranch,
    CDCVRunner,
    ContractAttestation,
    ControllerConfig,
    GatePolicy,
    IdentityCalibrator,
    ProbeContract,
    QuestionContract,
    ReleasedAnswer,
    RoutingOption,
    RuntimeEpisode,
    ScenarioReference,
    ScriptedScorer,
    StaticDemoAnswerBroker,
    apply_context_patch,
    attestation_integrity_manifest_hash,
    branch_manifest_hash,
    build_prediction_record,
    candidate_set_manifest_hash,
    question_bank_manifest_hash,
    released_answer_manifest_hash,
    run_equal_budget_structured_context,
    sha256_json,
    validate_episode_contract,
)


FULL_BUDGET = BudgetEnvelope(9, 216, 27, 0.0)
THREE_CALL_BUDGET = BudgetEnvelope(3, 72, 9, 0.0)


def context_field(value, provenance="benchmark_assignment"):
    return {
        "value": value,
        "provenance": provenance,
        "confidence": 1.0 if value is not None else 0.0,
        "retain_after_episode": False,
    }


def make_card(*, missing_goal=False, missing_setting=False, conflict=False):
    card = {
        "episode_id": "DEMO_CASE_001",
        "mode": "benchmark",
        "scope": "current_interaction_only",
        "expires_at": None,
        "variety_cue": {
            "value": "invented_demo_resource",
            "provenance": "experimentally_supplied",
            "retain_after_episode": False,
        },
        "fields": {
            "relationship_role": context_field("colleagues"),
            "setting": (
                context_field(None, "missing")
                if missing_setting
                else context_field("project room")
            ),
            "formality": context_field("informal"),
            "discourse_goal": (
                context_field(None, "missing")
                if missing_goal
                else context_field("request assistance")
            ),
            "preceding_speech_act": context_field("offer of help"),
            "situation": context_field("shared task in progress"),
        },
        "conflicts": [],
    }
    if conflict:
        card["conflicts"].append(
            {
                "slot": "discourse_goal",
                "code": "DEMO_CONFLICT",
                "severity": "hard_stop",
            }
        )
    return card


def make_branches(card, *, suffix="initial", wrong_question_slot=False):
    source_hash = sha256_json(card)

    def probe(probe_id, kind, source, value, answer_id, target=None):
        slot = "setting" if wrong_question_slot else "discourse_goal"
        patch = {slot: context_field(value)}
        result = apply_context_patch(card, (slot,), patch)
        return ProbeContract(
            probe_id=f"{probe_id}_{suffix}",
            probe_type=kind,
            source_candidate_id=source,
            target_candidate_id=target,
            changed_slots=(slot,),
            context_patch=patch,
            source_context_hash=source_hash,
            result_context_hash=sha256_json(result),
            scenario_answer_id=answer_id,
            validity_weight=1.0,
        )

    branch_a = CandidateBranch(
        "sense_a",
        probe(
            "probe_a_same",
            "PRESERVING",
            "sense_a",
            "request assistance",
            "assistance",
        ),
        probe(
            "probe_a_to_b",
            "MEANING_CHANGING",
            "sense_a",
            "request repayment",
            "repayment",
            "sense_b",
        ),
    )
    branch_b = CandidateBranch(
        "sense_b",
        probe(
            "probe_b_same",
            "PRESERVING",
            "sense_b",
            "request repayment",
            "repayment",
        ),
        probe(
            "probe_b_to_a",
            "MEANING_CHANGING",
            "sense_b",
            "request assistance",
            "assistance",
            "sense_a",
        ),
    )
    return {"sense_a": branch_a, "sense_b": branch_b}


def make_questions(branches, *, context_slot="discourse_goal"):
    return {
        "sense_a": (
            QuestionContract(
                "q_goal_a",
                context_slot,
                (
                    ScenarioReference(
                        "assistance", 0.5, branches["sense_a"].preserving.probe_id
                    ),
                    ScenarioReference(
                        "repayment",
                        0.5,
                        branches["sense_a"].meaning_changing.probe_id,
                    ),
                ),
                interaction_cost=0.05,
            ),
        ),
        "sense_b": (
            QuestionContract(
                "q_goal_b",
                context_slot,
                (
                    ScenarioReference(
                        "repayment", 0.5, branches["sense_b"].preserving.probe_id
                    ),
                    ScenarioReference(
                        "assistance",
                        0.5,
                        branches["sense_b"].meaning_changing.probe_id,
                    ),
                ),
                interaction_cost=0.05,
            ),
        ),
    }


def make_episode(
    *,
    missing_goal=False,
    conflict=False,
    missing_setting=False,
    split="development",
    wrong_question_slot=False,
):
    card = make_card(
        missing_goal=missing_goal,
        missing_setting=missing_setting,
        conflict=conflict,
    )
    branches = make_branches(
        card, suffix="initial", wrong_question_slot=wrong_question_slot
    )
    questions = make_questions(
        branches, context_slot="discourse_goal"
    )
    attestation_draft = ContractAttestation(
        manifest_id="DEMO_ATTESTATION_001",
        case_id="DEMO_CASE_001",
        family_id="DEMO_FAMILY_001",
        context_card_hash=sha256_json(card),
        candidate_set_hash=candidate_set_manifest_hash(
            ("sense_a", "sense_b", "OTHER_UNLISTED"),
            {
                "sense_a": "A request for assistance with the shared task.",
                "sense_b": "A request concerning an outstanding repayment.",
                "OTHER_UNLISTED": "Neither listed candidate is adequate.",
            },
        ),
        review_status=AttestationStatus.DEMO_ONLY,
        schema_status=AttestationStatus.DEMO_ONLY,
        cross_record_status=AttestationStatus.DEMO_ONLY,
        safety_status=AttestationStatus.DEMO_ONLY,
        required_context_slots=("discourse_goal",),
        intervention_manifest_hash=branch_manifest_hash(branches),
        question_manifest_hash=question_bank_manifest_hash(questions),
        reviewed_value_manifest_hash=sha256_json("DEMO_REVIEWED_VALUES"),
        integrity_manifest_hash="0" * 64,
    )
    attestation = replace(
        attestation_draft,
        integrity_manifest_hash=attestation_integrity_manifest_hash(
            attestation_draft
        ),
    )
    return RuntimeEpisode(
        case_id="DEMO_CASE_001",
        family_id="DEMO_FAMILY_001",
        split=split,
        utterance="Could you handle that for me?",
        candidate_ids=("sense_a", "sense_b", "OTHER_UNLISTED"),
        candidate_definitions={
            "sense_a": "A request for assistance with the shared task.",
            "sense_b": "A request concerning an outstanding repayment.",
            "OTHER_UNLISTED": "Neither listed candidate is adequate.",
        },
        context_card=card,
        branches=branches,
        attestation=attestation,
        questions_by_candidate=questions,
    )


def make_answer(episode, *, answer_id="assistance", wrong_case=False):
    question = episode.questions_by_candidate["sense_a"][0]
    patch = {
        "discourse_goal": context_field(
            "request assistance" if answer_id == "assistance" else "request repayment",
            "standardized_clarification",
        )
    }
    updated = apply_context_patch(
        episode.context_card, ("discourse_goal",), patch
    )
    post_branches = make_branches(updated, suffix=f"post_{answer_id}")
    answer = ReleasedAnswer(
        case_id="WRONG_CASE" if wrong_case else episode.case_id,
        question_id=question.question_id,
        answer_id=answer_id,
        context_slot=question.context_slot,
        context_patch=patch,
        question_manifest_hash=question.manifest_hash,
        answer_manifest_hash="0" * 64,
        validation_status=AttestationStatus.DEMO_ONLY,
        safety_status=AttestationStatus.DEMO_ONLY,
        post_answer_branch_manifest_hash=branch_manifest_hash(post_branches),
        post_answer_branches=post_branches,
    )
    return replace(answer, answer_manifest_hash=released_answer_manifest_hash(answer))


def broker_for(episode, answer=None):
    question = episode.questions_by_candidate["sense_a"][0]
    return StaticDemoAnswerBroker(
        {(episode.case_id, question.question_id): answer or make_answer(episode)}
    )


def accepted_sealed_episode(*, missing_goal=False):
    episode = make_episode(
        split="sealed_test", missing_goal=missing_goal
    )
    branches = {}
    for candidate, branch in episode.branches.items():
        branches[candidate] = replace(
            branch,
            preserving=replace(
                branch.preserving,
                review_status=AttestationStatus.ACCEPTED,
                safety_status=AttestationStatus.ACCEPTED,
            ),
            meaning_changing=replace(
                branch.meaning_changing,
                review_status=AttestationStatus.ACCEPTED,
                safety_status=AttestationStatus.ACCEPTED,
            ),
        )
    questions = {
        candidate: tuple(
            replace(
                question,
                review_status=AttestationStatus.ACCEPTED,
                safety_status=AttestationStatus.ACCEPTED,
            )
            for question in bank
        )
        for candidate, bank in episode.questions_by_candidate.items()
    }
    attestation = replace(
        episode.attestation,
        review_status=AttestationStatus.ACCEPTED,
        schema_status=AttestationStatus.ACCEPTED,
        cross_record_status=AttestationStatus.ACCEPTED,
        safety_status=AttestationStatus.ACCEPTED,
        intervention_manifest_hash=branch_manifest_hash(branches),
        question_manifest_hash=question_bank_manifest_hash(questions),
        integrity_manifest_hash="0" * 64,
    )
    attestation = replace(
        attestation,
        integrity_manifest_hash=attestation_integrity_manifest_hash(attestation),
    )
    return replace(
        episode,
        branches=branches,
        questions_by_candidate=questions,
        attestation=attestation,
    )


def accepted_sealed_answer(episode):
    answer = make_answer(episode)
    branches = {}
    for candidate, branch in answer.post_answer_branches.items():
        branches[candidate] = replace(
            branch,
            preserving=replace(
                branch.preserving,
                review_status=AttestationStatus.ACCEPTED,
                safety_status=AttestationStatus.ACCEPTED,
            ),
            meaning_changing=replace(
                branch.meaning_changing,
                review_status=AttestationStatus.ACCEPTED,
                safety_status=AttestationStatus.ACCEPTED,
            ),
        )
    draft = replace(
        answer,
        validation_status=AttestationStatus.ACCEPTED,
        safety_status=AttestationStatus.ACCEPTED,
        post_answer_branch_manifest_hash=branch_manifest_hash(branches),
        post_answer_branches=branches,
        answer_manifest_hash="0" * 64,
    )
    return replace(
        draft, answer_manifest_hash=released_answer_manifest_hash(draft)
    )


GOOD_PASS = (
    {"sense_a": 0.82, "sense_b": 0.13, "OTHER_UNLISTED": 0.05},
    {"sense_a": 0.80, "sense_b": 0.15, "OTHER_UNLISTED": 0.05},
    {"sense_a": 0.12, "sense_b": 0.83, "OTHER_UNLISTED": 0.05},
)

CLARIFY_PASS = (
    {"sense_a": 0.49, "sense_b": 0.46, "OTHER_UNLISTED": 0.05},
    {"sense_a": 0.88, "sense_b": 0.08, "OTHER_UNLISTED": 0.04},
    {"sense_a": 0.07, "sense_b": 0.89, "OTHER_UNLISTED": 0.04},
)

FAILED_TARGET_PASS = (
    {"sense_a": 0.82, "sense_b": 0.13, "OTHER_UNLISTED": 0.05},
    {"sense_a": 0.80, "sense_b": 0.15, "OTHER_UNLISTED": 0.05},
    {"sense_a": 0.78, "sense_b": 0.17, "OTHER_UNLISTED": 0.05},
)


class PipelineTests(unittest.TestCase):
    def policy(self):
        return GatePolicy(
            ControllerConfig(commit_threshold=0.70), IdentityCalibrator()
        )

    def runner(self):
        return CDCVRunner(self.policy(), budget_envelope=FULL_BUDGET)

    def test_initial_commit_uses_three_calls_and_strict_projection(self):
        episode = make_episode()
        scorer = ScriptedScorer("mock-small", GOOD_PASS)
        result = self.runner().run(episode, scorer)
        self.assertEqual(result.final_decision.action, Action.COMMIT)
        self.assertEqual(result.consumed.calls, 3)
        self.assertEqual(result.passes[0].selected_branch, "sense_a")
        request = scorer.seen_requests[0]
        self.assertEqual(set(request.context_card), {"scope", "variety_cue", "fields"})
        self.assertNotIn("conflicts", request.context_card)
        self.assertFalse(hasattr(request, "reference_sense_id"))
        self.assertFalse(hasattr(request, "probe_type"))

    def test_completeness_uses_only_case_required_slots(self):
        episode = make_episode(missing_setting=True)
        result = self.runner().run(
            episode, ScriptedScorer("required-only", GOOD_PASS)
        )
        self.assertEqual(
            result.passes[0].features.context_completeness, 1.0
        )
        self.assertEqual(result.final_decision.action, Action.COMMIT)

    def test_frozen_candidate_order_breaks_exact_tie(self):
        tied = (
            {"sense_b": 0.475, "sense_a": 0.475, "OTHER_UNLISTED": 0.05},
            GOOD_PASS[1],
            GOOD_PASS[2],
        )
        result = self.runner().run(make_episode(), ScriptedScorer("tie", tied))
        self.assertEqual(result.passes[0].selected_branch, "sense_a")

    def test_other_abstains_after_one_call(self):
        scorer = ScriptedScorer(
            "mock-other",
            ({"sense_a": 0.05, "sense_b": 0.05, "OTHER_UNLISTED": 0.90},),
        )
        result = self.runner().run(make_episode(), scorer)
        self.assertEqual(result.final_decision.action, Action.ABSTAIN_ESCALATE)
        self.assertEqual(result.final_decision.reason_code, "OTHER_UNLISTED_SELECTED")
        self.assertEqual(result.consumed.calls, 1)

    def test_one_bound_answer_repairs_once_with_rebased_probes(self):
        episode = make_episode(missing_goal=True)
        scorer = ScriptedScorer("mock-small", CLARIFY_PASS + GOOD_PASS)
        result = self.runner().run(
            episode, scorer, answer_broker=broker_for(episode)
        )
        self.assertEqual(result.initial_decision.action, Action.CLARIFY)
        self.assertEqual(result.final_decision.action, Action.COMMIT)
        self.assertTrue(result.answer_applied)
        self.assertEqual(result.applied_answer_id, "assistance")
        self.assertEqual(result.consumed.calls, 6)
        self.assertEqual(len(result.passes), 2)

    def test_unreleased_answer_becomes_abstention(self):
        episode = make_episode(missing_goal=True)
        result = self.runner().run(
            episode, ScriptedScorer("mock", CLARIFY_PASS)
        )
        self.assertEqual(result.final_decision.action, Action.ABSTAIN_ESCALATE)
        self.assertEqual(
            result.final_decision.reason_code, "CLARIFICATION_UNRESOLVED"
        )
        self.assertEqual(result.consumed.calls, 3)

    def test_cross_case_and_out_of_domain_answers_are_rejected(self):
        for answer in (
            make_answer(make_episode(missing_goal=True), wrong_case=True),
            make_answer(make_episode(missing_goal=True), answer_id="unapproved"),
        ):
            episode = make_episode(missing_goal=True)
            result = self.runner().run(
                episode,
                ScriptedScorer("mock", CLARIFY_PASS),
                answer_broker=broker_for(episode, answer),
            )
            self.assertEqual(
                result.final_decision.reason_code,
                "CLARIFICATION_RELEASE_REJECTED",
            )
            self.assertEqual(result.consumed.calls, 3)

    def test_answer_manifest_and_post_answer_source_hash_are_enforced(self):
        episode = make_episode(missing_goal=True)
        valid = make_answer(episode)
        stale = replace(valid, post_answer_branches=episode.branches)
        stale = replace(
            stale, answer_manifest_hash=released_answer_manifest_hash(stale)
        )
        result = self.runner().run(
            episode,
            ScriptedScorer("mock", CLARIFY_PASS),
            answer_broker=broker_for(episode, stale),
        )
        self.assertEqual(
            result.final_decision.reason_code,
            "CLARIFICATION_RELEASE_REJECTED",
        )
        tampered = replace(valid, answer_manifest_hash="f" * 64)
        result = self.runner().run(
            episode,
            ScriptedScorer("mock", CLARIFY_PASS),
            answer_broker=broker_for(episode, tampered),
        )
        self.assertEqual(
            result.final_decision.reason_code,
            "CLARIFICATION_RELEASE_REJECTED",
        )

    def test_repair_then_single_route_honors_nine_call_cap(self):
        episode = make_episode(missing_goal=True)
        small = ScriptedScorer(
            "mock-small", CLARIFY_PASS + FAILED_TARGET_PASS
        )
        large = ScriptedScorer("mock-large", GOOD_PASS)
        result = self.runner().run(
            episode,
            small,
            answer_broker=broker_for(episode),
            large_scorer=large,
            routing=RoutingOption(0.45, 0.80, 0.20, 0.50),
        )
        self.assertTrue(result.routed)
        self.assertEqual(result.final_decision.action, Action.COMMIT)
        self.assertEqual(result.consumed.calls, 9)
        self.assertEqual(len(result.passes), 3)

    def test_route_denied_for_privacy_or_missing_context(self):
        complete = make_episode()
        denied = self.runner().run(
            complete,
            ScriptedScorer("small", FAILED_TARGET_PASS),
            large_scorer=ScriptedScorer("large", GOOD_PASS),
            routing=RoutingOption(0.45, 0.80, 0.20, 0.50, False),
        )
        self.assertFalse(denied.routed)
        self.assertEqual(denied.consumed.calls, 3)

        missing = make_episode(missing_goal=True)
        unresolved = self.runner().run(
            missing,
            ScriptedScorer("small", CLARIFY_PASS),
            large_scorer=ScriptedScorer("large", GOOD_PASS),
            routing=RoutingOption(0.45, 0.80, 0.20, 0.50),
        )
        self.assertFalse(unresolved.routed)
        self.assertEqual(unresolved.consumed.calls, 3)

    def test_routing_option_rejects_invalid_values(self):
        for values in (
            (math.nan, 0.8, 0.1, 0.5),
            (0.5, 1.1, 0.1, 0.5),
            (0.5, 0.8, -0.1, 0.5),
        ):
            with self.assertRaises(ValueError):
                RoutingOption(*values)

    def test_conflict_nested_protected_and_nested_gold_fail_before_scoring(self):
        conflict = make_episode(conflict=True)
        protected = make_episode()
        protected.context_card["fields"]["situation"]["value"] = {
            "race": "not permitted"
        }
        gold = make_episode()
        gold.context_card["fields"]["situation"]["value"] = {
            "reference_sense_id": "sense_a"
        }
        for episode in (conflict, protected, gold):
            scorer = ScriptedScorer("unused", GOOD_PASS)
            result = self.runner().run(episode, scorer)
            self.assertEqual(result.final_decision.action, Action.ABSTAIN_ESCALATE)
            self.assertEqual(result.consumed.calls, 0)
            self.assertEqual(scorer.seen_requests, [])

    def test_demo_split_allowlist_fails_closed(self):
        for split in ("calibration", "sealed_test"):
            result = self.runner().run(
                make_episode(split=split), ScriptedScorer("unused", GOOD_PASS)
            )
            self.assertEqual(result.consumed.calls, 0)
            self.assertEqual(
                result.final_decision.reason_code, "INVALID_RUNTIME_CONTRACT"
            )
        with self.assertRaises(ValueError):
            make_episode(split="sealed_test_alias")

    def test_sealed_mode_requires_authorization_and_accepted_contracts(self):
        with self.assertRaises(PermissionError):
            CDCVRunner(
                self.policy(),
                budget_envelope=FULL_BUDGET,
                run_mode="SEALED",
            )
        accepted_episode = accepted_sealed_episode()
        sealed_runner = CDCVRunner(
            self.policy(),
            budget_envelope=FULL_BUDGET,
            run_mode="SEALED",
            sealed_execution_authorized=True,
            trusted_attestation_hashes=frozenset(
                {accepted_episode.attestation.integrity_manifest_hash}
            ),
        )
        demo_contract = sealed_runner.run(
            make_episode(split="sealed_test"),
            ScriptedScorer("unused", GOOD_PASS),
        )
        self.assertEqual(demo_contract.consumed.calls, 0)
        accepted = sealed_runner.run(
            accepted_episode,
            ScriptedScorer("sealed-mock", GOOD_PASS),
        )
        self.assertEqual(accepted.final_decision.action, Action.COMMIT)
        self.assertEqual(accepted.consumed.calls, 3)

        with self.assertRaises(PermissionError):
            run_equal_budget_structured_context(
                accepted_episode,
                ScriptedScorer("unused-control", GOOD_PASS),
                budget_envelope=THREE_CALL_BUDGET,
                run_mode="SEALED",
            )
        control = run_equal_budget_structured_context(
            accepted_episode,
            ScriptedScorer("sealed-control", GOOD_PASS),
            budget_envelope=THREE_CALL_BUDGET,
            run_mode="SEALED",
            sealed_execution_authorized=True,
            trusted_attestation_hashes=frozenset(
                {accepted_episode.attestation.integrity_manifest_hash}
            ),
        )
        self.assertEqual(sum(e.resources.calls for e in control.resource_events), 3)

    def test_sealed_clarification_requires_trusted_answer_hash(self):
        episode = accepted_sealed_episode(missing_goal=True)
        answer = accepted_sealed_answer(episode)
        broker = broker_for(episode, answer)
        attestation_hashes = frozenset(
            {episode.attestation.integrity_manifest_hash}
        )
        untrusted = CDCVRunner(
            self.policy(),
            budget_envelope=FULL_BUDGET,
            run_mode="SEALED",
            sealed_execution_authorized=True,
            trusted_attestation_hashes=attestation_hashes,
        ).run(
            episode,
            ScriptedScorer("sealed-small", CLARIFY_PASS),
            answer_broker=broker,
        )
        self.assertEqual(
            untrusted.final_decision.reason_code,
            "CLARIFICATION_RELEASE_REJECTED",
        )
        trusted = CDCVRunner(
            self.policy(),
            budget_envelope=FULL_BUDGET,
            run_mode="SEALED",
            sealed_execution_authorized=True,
            trusted_attestation_hashes=attestation_hashes,
            trusted_answer_manifest_hashes=frozenset(
                {answer.answer_manifest_hash}
            ),
        ).run(
            episode,
            ScriptedScorer("sealed-small", CLARIFY_PASS + GOOD_PASS),
            answer_broker=broker,
        )
        self.assertEqual(trusted.final_decision.action, Action.COMMIT)
        self.assertEqual(trusted.consumed.calls, 6)

    def test_attestation_is_bound_to_case_card_and_candidate_set(self):
        episode = make_episode()
        with self.assertRaises(ValueError):
            replace(episode.attestation, required_context_slots=())
        for attestation in (
            replace(episode.attestation, case_id="OTHER_CASE"),
            replace(episode.attestation, context_card_hash="f" * 64),
            replace(episode.attestation, candidate_set_hash="e" * 64),
        ):
            tampered = replace(episode, attestation=attestation)
            result = self.runner().run(
                tampered, ScriptedScorer("unused", GOOD_PASS)
            )
            self.assertEqual(result.consumed.calls, 0)
            self.assertEqual(
                result.final_decision.reason_code, "INVALID_RUNTIME_CONTRACT"
            )

    def test_question_probe_slot_binding_is_enforced(self):
        episode = make_episode(missing_goal=True, wrong_question_slot=True)
        errors = validate_episode_contract(episode)
        self.assertTrue(any("wrong slot" in error for error in errors))

    def test_equal_budget_control_enforces_calls_and_tokens(self):
        episode = make_episode()
        scorer = ScriptedScorer("control", GOOD_PASS)
        result = run_equal_budget_structured_context(
            episode, scorer, budget_envelope=THREE_CALL_BUDGET
        )
        self.assertEqual(result.budget_envelope, THREE_CALL_BUDGET)
        self.assertEqual(sum(e.resources.calls for e in result.resource_events), 3)
        self.assertEqual(sum(e.resources.input_tokens for e in result.resource_events), 72)
        self.assertTrue(
            all(
                request.context_card == scorer.seen_requests[0].context_card
                for request in scorer.seen_requests
            )
        )

        full_scorer = ScriptedScorer(
            "full-control", GOOD_PASS + GOOD_PASS + GOOD_PASS
        )
        full = run_equal_budget_structured_context(
            episode, full_scorer, budget_envelope=FULL_BUDGET
        )
        self.assertEqual(sum(e.resources.calls for e in full.resource_events), 9)
        self.assertEqual(
            sum(e.resources.input_tokens for e in full.resource_events), 216
        )

    def test_token_overrun_is_rejected(self):
        tiny = BudgetEnvelope(9, 10, 27, 0.0)
        runner = CDCVRunner(self.policy(), budget_envelope=tiny)
        with self.assertRaisesRegex(RuntimeError, "input-token"):
            runner.run(make_episode(), ScriptedScorer("mock", GOOD_PASS))

    def test_prediction_record_logs_all_passes_and_valid_budget(self):
        try:
            import jsonschema
        except ImportError as exc:
            self.fail(f"jsonschema is required for the validation test: {exc}")
        episode = make_episode(missing_goal=True)
        policy = self.policy()
        result = CDCVRunner(policy, budget_envelope=FULL_BUDGET).run(
            episode,
            ScriptedScorer("small", CLARIFY_PASS + GOOD_PASS),
            answer_broker=broker_for(episode),
        )
        record = build_prediction_record(
            result,
            policy,
            run_id="DEMO_RUN",
            system_id="cdcv_one_question",
            code_commit="LOCAL_DEMO",
            timestamp_utc="2026-08-03T15:00:00Z",
        )
        schema = json.loads(
            (ROOT / "data" / "schemas" / "prediction_record.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.Draft202012Validator(schema).validate(record)
        self.assertEqual(
            [item["pass_name"] for item in record["pass_trace"]],
            ["initial", "post_question"],
        )
        self.assertEqual(len(record["pass_trace"][0]["prompt_hashes"]), 3)
        self.assertGreaterEqual(
            record["resource_use"]["allocated"]["input_tokens"],
            record["resource_use"]["consumed"]["input_tokens"],
        )
        self.assertEqual(record["clarification_trace"]["answer_id"], "assistance")


if __name__ == "__main__":
    unittest.main()
