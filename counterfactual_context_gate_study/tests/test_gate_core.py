import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdcv_gate import (  # noqa: E402
    Action,
    AnswerScenario,
    ChangingProbe,
    ClarificationQuestion,
    ControllerConfig,
    GatePolicy,
    GateState,
    IdentityCalibrator,
    IsotonicCalibrator,
    PreservingProbe,
    common_feasible_coverage,
    compute_verification_features,
    route_if_beneficial,
    select_eligible_at_coverage,
)


class VerificationFeatureTests(unittest.TestCase):
    def setUp(self):
        self.base = {"sense_a": 0.86, "sense_b": 0.09, "OTHER_UNLISTED": 0.05}
        self.preserving = (
            PreservingProbe({"sense_a": 0.82, "sense_b": 0.11, "OTHER_UNLISTED": 0.07}),
            PreservingProbe({"sense_a": 0.88, "sense_b": 0.08, "OTHER_UNLISTED": 0.04}),
        )
        self.changing = (
            ChangingProbe(
                {"sense_a": 0.08, "sense_b": 0.87, "OTHER_UNLISTED": 0.05},
                "sense_b",
            ),
            ChangingProbe(
                {"sense_a": 0.14, "sense_b": 0.80, "OTHER_UNLISTED": 0.06},
                "sense_b",
            ),
        )

    def features(self, conflict=False):
        return compute_verification_features(
            self.base,
            self.preserving,
            self.changing,
            context_completeness=1.0,
            context_conflict=conflict,
        )

    def test_dual_probe_features_reward_only_targeted_change(self):
        features = self.features()
        self.assertEqual(features.base_sense_id, "sense_a")
        self.assertEqual(features.preservation_invariance, 1.0)
        self.assertEqual(features.targeted_response, 1.0)
        self.assertGreater(features.mean_changing_target_margin, 0.6)

    def test_arbitrary_flip_is_not_appropriate_response(self):
        arbitrary = (
            ChangingProbe(
                {"sense_a": 0.05, "sense_b": 0.10, "OTHER_UNLISTED": 0.85},
                "sense_b",
            ),
        )
        features = compute_verification_features(
            self.base,
            self.preserving,
            arbitrary,
            context_completeness=1.0,
            context_conflict=False,
        )
        self.assertEqual(features.targeted_response, 0.0)

    def test_candidate_mismatch_fails(self):
        with self.assertRaises(ValueError):
            compute_verification_features(
                self.base,
                (PreservingProbe({"sense_a": 0.9, "sense_b": 0.1}),),
                self.changing,
                context_completeness=1.0,
                context_conflict=False,
            )


class ControllerTests(VerificationFeatureTests):
    def test_high_reliability_commits(self):
        policy = GatePolicy(
            ControllerConfig(commit_threshold=0.70), IdentityCalibrator()
        )
        decision = policy.decide(self.features(), GateState(), self.base)
        self.assertEqual(decision.action, Action.COMMIT)
        self.assertEqual(decision.sense_id, "sense_a")

    def test_failed_preservation_cannot_be_hidden_by_weighted_score(self):
        features = compute_verification_features(
            self.base,
            (
                PreservingProbe(
                    {"sense_a": 0.10, "sense_b": 0.85, "OTHER_UNLISTED": 0.05}
                ),
            ),
            self.changing,
            context_completeness=1.0,
            context_conflict=False,
        )
        decision = GatePolicy(ControllerConfig(commit_threshold=0.0)).decide(
            features, GateState(), self.base
        )
        self.assertEqual(decision.action, Action.ABSTAIN_ESCALATE)
        self.assertEqual(decision.reason_code, "PRESERVATION_FAILURE")

    def test_failed_targeted_response_cannot_be_hidden_by_weighted_score(self):
        features = compute_verification_features(
            self.base,
            self.preserving,
            (
                ChangingProbe(
                    {"sense_a": 0.85, "sense_b": 0.10, "OTHER_UNLISTED": 0.05},
                    "sense_b",
                ),
            ),
            context_completeness=1.0,
            context_conflict=False,
        )
        decision = GatePolicy(ControllerConfig(commit_threshold=0.0)).decide(
            features, GateState(), self.base
        )
        self.assertEqual(decision.action, Action.ABSTAIN_ESCALATE)
        self.assertEqual(decision.reason_code, "TARGETED_RESPONSE_FAILURE")

    def test_context_conflict_hard_abstains(self):
        policy = GatePolicy(ControllerConfig(commit_threshold=0.0))
        decision = policy.decide(self.features(conflict=True), GateState(), self.base)
        self.assertEqual(decision.action, Action.ABSTAIN_ESCALATE)
        self.assertEqual(decision.reason_code, "CONTEXT_CONFLICT")

    def test_protected_attribute_request_hard_abstains(self):
        policy = GatePolicy(ControllerConfig(commit_threshold=0.0))
        state = GateState(prohibited_attribute_requested=True)
        decision = policy.decide(self.features(), state, self.base)
        self.assertEqual(decision.action, Action.ABSTAIN_ESCALATE)
        self.assertEqual(decision.reason_code, "PROTECTED_ATTRIBUTE_POLICY")

    def test_other_unlisted_never_commits(self):
        base = {"sense_a": 0.05, "sense_b": 0.05, "OTHER_UNLISTED": 0.90}
        preserving = (
            PreservingProbe({"sense_a": 0.03, "sense_b": 0.04, "OTHER_UNLISTED": 0.93}),
        )
        changing = (
            ChangingProbe({"sense_a": 0.85, "sense_b": 0.10, "OTHER_UNLISTED": 0.05}, "sense_a"),
        )
        features = compute_verification_features(
            base,
            preserving,
            changing,
            context_completeness=1.0,
            context_conflict=False,
        )
        decision = GatePolicy(ControllerConfig(commit_threshold=0.0)).decide(
            features, GateState(), base
        )
        self.assertEqual(decision.action, Action.ABSTAIN_ESCALATE)
        self.assertEqual(decision.reason_code, "OTHER_UNLISTED_SELECTED")

    def test_selects_information_reducing_question(self):
        uncertain = {"sense_a": 0.49, "sense_b": 0.46, "OTHER_UNLISTED": 0.05}
        question = ClarificationQuestion(
            "q_goal",
            "discourse_goal",
            (
                AnswerScenario(0.5, {"sense_a": 0.90, "sense_b": 0.07, "OTHER_UNLISTED": 0.03}),
                AnswerScenario(0.5, {"sense_a": 0.06, "sense_b": 0.91, "OTHER_UNLISTED": 0.03}),
            ),
            interaction_cost=0.05,
            compute_cost=0.05,
        )
        state = GateState(missing_discriminating_slots=("discourse_goal",))
        policy = GatePolicy(ControllerConfig(commit_threshold=0.99))
        decision = policy.decide(self.features(), state, uncertain, (question,))
        self.assertEqual(decision.action, Action.CLARIFY)
        self.assertEqual(decision.context_slot, "discourse_goal")

    def test_no_second_question(self):
        state = GateState(
            missing_discriminating_slots=("discourse_goal",), questions_remaining=0
        )
        policy = GatePolicy(ControllerConfig(commit_threshold=0.99))
        decision = policy.decide(self.features(), state, self.base)
        self.assertEqual(decision.action, Action.ABSTAIN_ESCALATE)

    def test_hard_eligibility_mask_excludes_other_and_conflict(self):
        policy = GatePolicy(ControllerConfig(commit_threshold=0.0))
        eligible, reason = policy.commit_eligibility(
            self.features(conflict=True), GateState()
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "CONTEXT_CONFLICT")

    def test_question_scenarios_must_share_candidate_set(self):
        question = ClarificationQuestion(
            "q_bad",
            "discourse_goal",
            (AnswerScenario(1.0, {"sense_a": 0.8, "sense_b": 0.2}),),
        )
        with self.assertRaises(ValueError):
            question.expected_information_gain(self.base)


class MatchedCoverageTests(unittest.TestCase):
    def test_common_target_is_capped_by_least_eligible_method(self):
        target = common_feasible_coverage(
            0.75,
            ([True, True, True, False], [True, True, False, False]),
        )
        self.assertEqual(target, 0.5)

    def test_selection_never_overrides_ineligible_cases(self):
        selection = select_eligible_at_coverage(
            scores=[0.99, 0.80, 0.70, 0.60],
            eligible=[False, True, False, True],
            tie_break_hashes=["a", "b", "c", "d"],
            target_coverage=0.75,
        )
        self.assertEqual(set(selection.selected_indices), {1, 3})
        self.assertTrue(selection.target_unattainable)
        self.assertEqual(selection.achieved_coverage, 0.5)

    def test_ties_use_frozen_hash(self):
        selection = select_eligible_at_coverage(
            scores=[0.5, 0.5],
            eligible=[True, True],
            tie_break_hashes=["z", "a"],
            target_coverage=0.5,
        )
        self.assertEqual(selection.selected_indices, (1,))


class CalibrationAndRoutingTests(unittest.TestCase):
    def test_isotonic_pools_tied_scores(self):
        calibrator = IsotonicCalibrator.fit([0.5, 0.5], [0, 1])
        self.assertAlmostEqual(calibrator.predict(0.5), 0.5)

    def test_isotonic_tied_scores_respect_sample_weights(self):
        calibrator = IsotonicCalibrator.fit(
            [0.5, 0.5], [0, 1], sample_weights=[1.0, 3.0]
        )
        self.assertAlmostEqual(calibrator.predict(0.5), 0.75)

    def test_isotonic_rejects_invalid_sample_weights(self):
        with self.assertRaises(ValueError):
            IsotonicCalibrator.fit([0.2, 0.8], [0, 1], sample_weights=[1.0])
        with self.assertRaises(ValueError):
            IsotonicCalibrator.fit([0.2, 0.8], [0, 1], sample_weights=[1.0, 0.0])

    def test_isotonic_predictions_are_monotone(self):
        calibrator = IsotonicCalibrator.fit(
            [0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1]
        )
        predictions = [calibrator.predict(value / 100) for value in range(61)]
        self.assertEqual(predictions, sorted(predictions))

    def test_route_requires_positive_net_benefit_and_safe_context(self):
        self.assertTrue(
            route_if_beneficial(
                context_complete=True,
                context_conflict=False,
                privacy_allowed=True,
                expected_small_correctness=0.55,
                expected_large_correctness=0.80,
                normalized_route_cost=0.20,
                cost_weight=0.50,
            )
        )
        self.assertFalse(
            route_if_beneficial(
                context_complete=True,
                context_conflict=True,
                privacy_allowed=True,
                expected_small_correctness=0.55,
                expected_large_correctness=0.90,
                normalized_route_cost=0.0,
                cost_weight=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
