"""Generate the output-free CDCV-Gate protocol notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "CA_IEDI_0803.ipynb"
REPOSITORY_MIRROR = ROOT.parent / "CA_IEDI_0803.ipynb"


def _source(text: str) -> list[str]:
    return (dedent(text).strip() + "\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


CELLS = [
    markdown(
        r'''
        # CA_IEDI_0803 — CDCV-Gate protocol rewrite

        **DEMONSTRATION ONLY — RESULTS_LOCKED**

        This is a clean derivative of the
        [pinned upstream CA-IEDI notebook](https://github.com/toecm/iedi-mas/blob/5cff1e509efb09c24f9ac7e30075b6a131ee6fbc/CA_IEDI_0803.ipynb).
        The audited source is commit
        5cff1e509efb09c24f9ac7e30075b6a131ee6fbc, Git blob
        5b83ce8dbdc0e147637ef499b7c4f7deabfbb653, and UTF-8 SHA-256
        0287daf13a8a863f267a8cd50acc1e6563ab2a64c852873bf59aa561bc616eaa.

        The audited source remains preserved at the pinned commit above. This
        clean rewrite supersedes the mutable root path without copying source
        code or saved output. It removes provider credentials, mutable dataset
        writes, public UI launch, audio/ASR, IPFS, Web3, and Hardhat surfaces.

        Passing this notebook proves only that the offline reference paths
        execute. It is not evidence of accuracy, selective risk, robustness,
        fairness, latency, cost, community validity, or deployment benefit.
        '''
    ),
    markdown(
        r'''
        ## 1. Read-only environment and evidence locks

        The notebook installs nothing, opens no network connection, writes no
        file, and requests no credential. DEMO mode is restricted to invented
        development fixtures. The explicit require helper cannot be disabled by
        optimized Python execution.
        '''
    ),
    code(
        r'''
        from pathlib import Path
        from dataclasses import asdict
        import json
        import sys

        RUN_MODE = "DEMO"
        RESULTS_LOCKED = True
        UPSTREAM_COMMIT = "5cff1e509efb09c24f9ac7e30075b6a131ee6fbc"
        UPSTREAM_BLOB_SHA = "5b83ce8dbdc0e147637ef499b7c4f7deabfbb653"


        def require(condition, message):
            if not condition:
                raise RuntimeError(message)


        require(RUN_MODE == "DEMO", "reference notebook must remain in DEMO mode")
        require(RESULTS_LOCKED, "reference notebook cannot expose empirical results")

        start = Path.cwd().resolve()
        candidates = []
        for base in (start, *start.parents):
            candidates.extend(
                (
                    base,
                    base / "counterfactual_context_gate_study",
                    base / "CA-IEDI" / "counterfactual_context_gate_study",
                )
            )
        STUDY_ROOT = next(
            (
                path
                for path in candidates
                if (path / "config" / "study_design.json").is_file()
                and (path / "src" / "cdcv_gate").is_dir()
            ),
            None,
        )
        require(STUDY_ROOT is not None, "study package cannot be located")
        sys.path.insert(0, str(STUDY_ROOT / "src"))
        '''
    ),
    code(
        r'''
        from cdcv_gate import (
            Action,
            AttestationStatus,
            BudgetEnvelope,
            CandidateBranch,
            CandidateSet,
            CDCVRunner,
            ClarificationAnswerBroker,
            ContractAttestation,
            ControllerConfig,
            FixedCandidateProvider,
            GatePolicy,
            IsotonicCalibrator,
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
            common_feasible_coverage,
            freeze_prediction_hash,
            question_bank_manifest_hash,
            released_answer_manifest_hash,
            run_equal_budget_structured_context,
            select_eligible_at_coverage,
            sha256_json,
            validate_context_card,
            validate_episode_contract,
            validate_runtime_episode_gold_free,
        )

        DESIGN = json.loads(
            (STUDY_ROOT / "config" / "study_design.json").read_text(encoding="utf-8")
        )
        require(DESIGN["status"] == "PLANNED_RESULTS_LOCKED", "design status drift")
        require(
            DESIGN["result_policy"]["synthetic_results_as_empirical_evidence"] is False,
            "synthetic evidence lock drift",
        )
        require(
            DESIGN["intervention_policy"]["sealed_reference_labels_visible_to_inference"]
            is False,
            "sealed-label isolation drift",
        )
        require(
            DESIGN["runtime_probe_budget"][
                "maximum_scorer_calls_excluding_candidate_generation"
            ]
            == 9,
            "call cap drift",
        )
        require(
            DESIGN["controller"]["maximum_clarification_questions"] == 1,
            "clarification cap drift",
        )
        '''
    ),
    markdown(
        r'''
        ## 2. Invented development contracts

        The neutral sentence, candidate definitions, context values, probes, and
        questions below were created only for software testing. They are not
        IEDID rows, community-authored benchmark cases, or observations.

        Fixed candidates represent the primary task. Model-backed candidate
        generation is a planned secondary analysis and remains NOT_CONFIGURED.
        '''
    ),
    code(
        r'''
        CANDIDATES = ("sense_a", "sense_b", "OTHER_UNLISTED")
        DEFINITIONS = {
            "sense_a": "A request for assistance with the shared task.",
            "sense_b": "A request concerning an outstanding repayment.",
            "OTHER_UNLISTED": "Neither listed candidate is adequate.",
        }
        primary_candidate_provider = FixedCandidateProvider(
            "DEMO_FIXED_CANDIDATES",
            CandidateSet(CANDIDATES, DEFINITIONS, "DEMO_ONLY"),
        )
        require(
            primary_candidate_provider.generate(
                "Could you handle that for me?"
            ).candidate_ids
            == CANDIDATES,
            "fixed candidate provider drift",
        )
        MODEL_BACKED_CANDIDATE_GENERATOR_STATUS = "NOT_CONFIGURED"


        def context_field(value, provenance="benchmark_assignment"):
            return {
                "value": value,
                "provenance": provenance,
                "confidence": 1.0 if value is not None else 0.0,
                "retain_after_episode": False,
            }


        def make_card(*, missing_goal=False, conflict=False):
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
                    "setting": context_field("project room"),
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
            require(validate_context_card(card) == [], "invalid invented context card")
            return card


        def make_branches(card, suffix):
            source_hash = sha256_json(card)

            def probe(probe_id, kind, source, value, answer_id, target=None):
                patch = {"discourse_goal": context_field(value)}
                result = apply_context_patch(
                    card, ("discourse_goal",), patch
                )
                return ProbeContract(
                    probe_id=f"{probe_id}_{suffix}",
                    probe_type=kind,
                    source_candidate_id=source,
                    target_candidate_id=target,
                    changed_slots=("discourse_goal",),
                    context_patch=patch,
                    source_context_hash=source_hash,
                    result_context_hash=sha256_json(result),
                    scenario_answer_id=answer_id,
                    validity_weight=1.0,
                    review_status=AttestationStatus.DEMO_ONLY,
                    safety_status=AttestationStatus.DEMO_ONLY,
                )

            return {
                "sense_a": CandidateBranch(
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
                ),
                "sense_b": CandidateBranch(
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
                ),
            }
        '''
    ),
    code(
        r'''
        def make_questions(branches):
            return {
                "sense_a": (
                    QuestionContract(
                        "q_goal_a",
                        "discourse_goal",
                        (
                            ScenarioReference(
                                "assistance",
                                0.5,
                                branches["sense_a"].preserving.probe_id,
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
                        "discourse_goal",
                        (
                            ScenarioReference(
                                "repayment",
                                0.5,
                                branches["sense_b"].preserving.probe_id,
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


        def make_episode(*, missing_goal=False, conflict=False):
            card = make_card(missing_goal=missing_goal, conflict=conflict)
            branches = make_branches(card, "initial")
            questions = make_questions(branches)
            reviewed_value_manifest_hash = sha256_json(
                "DEMO_REVIEWED_VALUES"
            )
            draft_attestation = ContractAttestation(
                "DEMO_ATTESTATION_001",
                "DEMO_CASE_001",
                "DEMO_FAMILY_001",
                sha256_json(card),
                candidate_set_manifest_hash(CANDIDATES, DEFINITIONS),
                AttestationStatus.DEMO_ONLY,
                AttestationStatus.DEMO_ONLY,
                AttestationStatus.DEMO_ONLY,
                AttestationStatus.DEMO_ONLY,
                ("discourse_goal",),
                branch_manifest_hash(branches),
                question_bank_manifest_hash(questions),
                reviewed_value_manifest_hash,
                "0" * 64,
            )
            attestation = ContractAttestation(
                "DEMO_ATTESTATION_001",
                "DEMO_CASE_001",
                "DEMO_FAMILY_001",
                sha256_json(card),
                candidate_set_manifest_hash(CANDIDATES, DEFINITIONS),
                AttestationStatus.DEMO_ONLY,
                AttestationStatus.DEMO_ONLY,
                AttestationStatus.DEMO_ONLY,
                AttestationStatus.DEMO_ONLY,
                ("discourse_goal",),
                branch_manifest_hash(branches),
                question_bank_manifest_hash(questions),
                reviewed_value_manifest_hash,
                attestation_integrity_manifest_hash(draft_attestation),
            )
            episode = RuntimeEpisode(
                "DEMO_CASE_001",
                "DEMO_FAMILY_001",
                "development",
                "Could you handle that for me?",
                CANDIDATES,
                DEFINITIONS,
                card,
                branches,
                attestation,
                questions,
            )
            require(
                validate_episode_contract(episode) == [],
                "invented episode contract is invalid",
            )
            return episode


        def make_released_answer(episode):
            question = episode.questions_by_candidate["sense_a"][0]
            patch = {
                "discourse_goal": context_field(
                    "request assistance", "standardized_clarification"
                )
            }
            repaired_card = apply_context_patch(
                episode.context_card, ("discourse_goal",), patch
            )
            post_answer_branches = make_branches(
                repaired_card, "post_assistance"
            )
            draft = ReleasedAnswer(
                episode.case_id,
                question.question_id,
                "assistance",
                question.context_slot,
                patch,
                question.manifest_hash,
                "0" * 64,
                AttestationStatus.DEMO_ONLY,
                AttestationStatus.DEMO_ONLY,
                branch_manifest_hash(post_answer_branches),
                post_answer_branches,
            )
            return ReleasedAnswer(
                episode.case_id,
                question.question_id,
                "assistance",
                question.context_slot,
                patch,
                question.manifest_hash,
                released_answer_manifest_hash(draft),
                AttestationStatus.DEMO_ONLY,
                AttestationStatus.DEMO_ONLY,
                branch_manifest_hash(post_answer_branches),
                post_answer_branches,
            )


        def make_broker(episode, answer=None):
            question = episode.questions_by_candidate["sense_a"][0]
            broker = StaticDemoAnswerBroker(
                {
                    (episode.case_id, question.question_id): (
                        answer if answer is not None else make_released_answer(episode)
                    )
                }
            )
            broker_interface: ClarificationAnswerBroker = broker
            return broker_interface
        '''
    ),
    markdown(
        r'''
        ## 3. Gold-free runtime record

        The system scorer never receives a reference action, reference sense,
        case type, accepted clarification slot, probe relation, target candidate,
        answer label, or validator rating. The attestation is label-free and
        DEMO_ONLY; SEALED mode would reject it.
        '''
    ),
    code(
        r'''
        demo_episode = make_episode()
        runtime_record = {
            "case_id": demo_episode.case_id,
            "family_id": demo_episode.family_id,
            "split": demo_episode.split,
            "evaluation_regime": "family_held_out",
            "community_resource": "american_english",
            "utterance_private_ref": "data/private/DEMO_CASE_001.txt",
            "utterance_hash": sha256_json(demo_episode.utterance),
            "candidate_senses": [
                {
                    "candidate_id": candidate,
                    "label_private_ref": f"data/private/{candidate}.label",
                    "definition_private_ref": f"data/private/{candidate}.definition",
                }
                for candidate in CANDIDATES
            ],
            "candidate_set_hash": candidate_set_manifest_hash(
                CANDIDATES, DEFINITIONS
            ),
            "context_card_ref": "runs/demo/context_card.json",
            "context_card_hash": sha256_json(demo_episode.context_card),
            "intervention_bundle_ref": "runs/demo/intervention_bundle.json",
            "intervention_bundle_hash": branch_manifest_hash(demo_episode.branches),
            "question_bank_ref": "runs/demo/question_bank.json",
            "question_bank_hash": question_bank_manifest_hash(
                demo_episode.questions_by_candidate
            ),
            "contract_attestation_ref": "runs/demo/attestation.json",
            "contract_attestation_hash": sha256_json(asdict(demo_episode.attestation)),
            "runtime_manifest_hash": sha256_json(
                {
                    "case_id": demo_episode.case_id,
                    "family_id": demo_episode.family_id,
                    "split": demo_episode.split,
                }
            ),
        }
        require(
            validate_runtime_episode_gold_free(runtime_record) == [],
            "runtime view contains forbidden data",
        )


        def validate_schema(instance, schema_name):
            try:
                import jsonschema
            except ImportError:
                if RUN_MODE == "SEALED":
                    raise RuntimeError("jsonschema is mandatory in SEALED mode")
                return "NOT_INSTALLED_DEMO_STRUCTURAL_GUARDS"
            schema = json.loads(
                (STUDY_ROOT / "data" / "schemas" / schema_name).read_text(
                    encoding="utf-8"
                )
            )
            jsonschema.Draft202012Validator(schema).validate(instance)
            return "DRAFT_2020_12_VALID"


        runtime_schema_status = validate_schema(
            runtime_record, "runtime_episode.schema.json"
        )
        attestation_schema_status = validate_schema(
            json.loads(json.dumps(asdict(demo_episode.attestation))),
            "contract_attestation.schema.json",
        )
        '''
    ),
    markdown(
        r'''
        ## 4. Connected demonstration calibrator and scorer boundary

        The fitted isotonic object below is connected to the controller before
        any decision. Its four invented observations are a mechanics fixture,
        not a calibration estimate. ScriptedScorer is an offline adapter and is
        never registered as a real model or empirical baseline.
        '''
    ),
    code(
        r'''
        DEMO_CALIBRATION_ONLY = True
        demo_calibrator = IsotonicCalibrator.fit(
            [0.10, 0.40, 0.70, 0.95],
            [0, 0, 1, 1],
            sample_weights=[1.0, 1.0, 1.0, 1.0],
        )
        policy = GatePolicy(
            ControllerConfig(
                commit_threshold=0.70,
                minimum_preservation_invariance=1.0,
                minimum_targeted_response=1.0,
            ),
            demo_calibrator,
        )
        FULL_DEMO_BUDGET = BudgetEnvelope(9, 216, 27, 0.0)
        runner = CDCVRunner(
            policy, budget_envelope=FULL_DEMO_BUDGET, run_mode=RUN_MODE
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

        commit_scorer = ScriptedScorer("DEMO_SCRIPTED_SMALL", GOOD_PASS)
        commit_result = runner.run(make_episode(), commit_scorer, seed=11)
        require(
            commit_result.final_decision.action == Action.COMMIT,
            "commit smoke path failed",
        )
        require(commit_result.consumed.calls == 3, "commit path call drift")
        require(
            commit_result.passes[0].selected_branch == "sense_a",
            "model-prediction branch selection drift",
        )
        require(
            set(commit_scorer.seen_requests[0].context_card)
            == {"scope", "variety_cue", "fields"},
            "scorer projection drift",
        )
        require(
            not hasattr(commit_scorer.seen_requests[0], "reference_sense_id"),
            "gold label leaked into scorer request",
        )
        '''
    ),
    markdown(
        r'''
        ## 5. One targeted clarification, broker release, and repair

        Question utility reuses the already scored probe distributions, so
        selection adds no model call. Only the matching case, question, slot,
        answer-domain member, question hash, and answer hash can be released.
        The repaired pass uses a separate probe bundle whose source hashes bind
        to the repaired card.
        '''
    ),
    code(
        r'''
        clarify_episode = make_episode(missing_goal=True)
        demo_released_answer = make_released_answer(clarify_episode)
        released_answer_schema_status = validate_schema(
            json.loads(json.dumps(asdict(demo_released_answer))),
            "released_answer.schema.json",
        )
        repair_result = runner.run(
            clarify_episode,
            ScriptedScorer(
                "DEMO_SCRIPTED_SMALL", CLARIFY_PASS + GOOD_PASS
            ),
            answer_broker=make_broker(
                clarify_episode, demo_released_answer
            ),
            seed=21,
        )
        require(
            repair_result.initial_decision.action == Action.CLARIFY,
            "clarification smoke path failed",
        )
        require(
            repair_result.final_decision.action == Action.COMMIT,
            "one-question repair failed",
        )
        require(repair_result.answer_applied, "broker answer was not applied")
        require(repair_result.consumed.calls == 6, "repair path call drift")
        require(
            sum(
                event.resources.calls
                for event in repair_result.resource_events
                if event.stage == "question_selection"
            )
            == 0,
            "question selection added a model call",
        )

        unresolved_result = runner.run(
            clarify_episode,
            ScriptedScorer("DEMO_SCRIPTED_SMALL", CLARIFY_PASS),
            seed=22,
        )
        require(
            unresolved_result.final_decision.action == Action.ABSTAIN_ESCALATE,
            "unreleased answer did not abstain",
        )
        require(
            unresolved_result.final_decision.reason_code
            == "CLARIFICATION_UNRESOLVED",
            "unreleased answer reason drift",
        )
        '''
    ),
    markdown(
        r'''
        ## 6. Hard abstention, optional route, and equal budget

        Hard context conflicts stop before scoring. A larger model is called
        once only after repair remains unresolved, context is complete,
        privacy permits routing, expected net benefit is positive, and a full
        three-call verification pass remains. The control receives an explicit
        equal call/input-token/output-token allocation.
        '''
    ),
    code(
        r'''
        conflict_result = runner.run(
            make_episode(conflict=True),
            ScriptedScorer("DEMO_UNUSED", GOOD_PASS),
        )
        require(
            conflict_result.final_decision.action == Action.ABSTAIN_ESCALATE,
            "conflict path did not abstain",
        )
        require(conflict_result.consumed.calls == 0, "conflict consumed model calls")

        FAILED_TARGET_PASS = (
            {"sense_a": 0.82, "sense_b": 0.13, "OTHER_UNLISTED": 0.05},
            {"sense_a": 0.80, "sense_b": 0.15, "OTHER_UNLISTED": 0.05},
            {"sense_a": 0.78, "sense_b": 0.17, "OTHER_UNLISTED": 0.05},
        )
        routed_result = runner.run(
            clarify_episode,
            ScriptedScorer(
                "DEMO_SCRIPTED_SMALL", CLARIFY_PASS + FAILED_TARGET_PASS
            ),
            answer_broker=make_broker(clarify_episode),
            large_scorer=ScriptedScorer("DEMO_SCRIPTED_LARGE", GOOD_PASS),
            routing=RoutingOption(0.45, 0.80, 0.20, 0.50),
            seed=31,
        )
        require(routed_result.routed, "beneficial route was not used")
        require(
            routed_result.final_decision.action == Action.COMMIT,
            "routed pass did not commit",
        )
        require(routed_result.consumed.calls == 9, "nine-call cap drift")

        control_result = run_equal_budget_structured_context(
            make_episode(),
            ScriptedScorer("DEMO_SCRIPTED_CONTROL", GOOD_PASS),
            budget_envelope=BudgetEnvelope(3, 72, 9, 0.0),
            seed=41,
        )
        require(
            sum(event.resources.calls for event in control_result.resource_events)
            == 3,
            "equal-budget call allocation drift",
        )
        require(
            sum(
                event.resources.input_tokens
                for event in control_result.resource_events
            )
            == 72,
            "equal-budget input-token allocation drift",
        )

        full_budget_control = run_equal_budget_structured_context(
            make_episode(),
            ScriptedScorer(
                "DEMO_SCRIPTED_FULL_CONTROL",
                GOOD_PASS + GOOD_PASS + GOOD_PASS,
            ),
            budget_envelope=FULL_DEMO_BUDGET,
            seed=51,
        )
        require(
            sum(
                event.resources.calls
                for event in full_budget_control.resource_events
            )
            == 9,
            "full equal-budget call allocation drift",
        )
        require(
            sum(
                event.resources.input_tokens
                for event in full_budget_control.resource_events
            )
            == 216,
            "full equal-budget token allocation drift",
        )
        '''
    ),
    markdown(
        r'''
        ## 7. Calibration mechanics, prediction trace, and freeze

        Coverage values below are invented mechanics fixtures. The prediction
        record logs every verification pass and fitted calibrator state, then is
        hashed before any evaluator-only label store could be mounted. This
        notebook contains and joins no sealed label.
        '''
    ),
    code(
        r'''
        demo_common_coverage = common_feasible_coverage(
            0.75,
            ([True, True, True, False], [True, True, False, False]),
        )
        demo_selection = select_eligible_at_coverage(
            [0.9, 0.8, 0.7, 0.6],
            [False, True, False, True],
            ["a", "b", "c", "d"],
            0.75,
        )
        require(demo_common_coverage == 0.5, "coverage fixture drift")
        require(demo_selection.target_unattainable, "coverage guard drift")

        demo_record = build_prediction_record(
            repair_result,
            policy,
            run_id="DEMO_RUN_NOT_EVIDENCE",
            system_id="cdcv_one_question",
            code_commit="LOCAL_PROTOCOL_REWRITE",
            timestamp_utc="2026-08-03T15:00:00Z",
        )
        prediction_schema_status = validate_schema(
            demo_record, "prediction_record.schema.json"
        )
        require(
            [item["pass_name"] for item in demo_record["pass_trace"]]
            == ["initial", "post_question"],
            "prediction pass trace drift",
        )
        demo_prediction_hash = freeze_prediction_hash([demo_record])
        require(len(demo_prediction_hash) == 64, "prediction freeze hash drift")
        '''
    ),
    markdown(
        r'''
        ## 8. Implementation boundary and smoke summary

        IEDID may support expression discovery, development examples, profile
        schema construction, and lineage diagnostics. It is not the sealed test
        set. The frozen IEDI notebook is a lineage/coverage diagnostic, including
        a separate known-expression, unseen-context regime.

        All real external systems, community-accepted contracts, model-backed
        candidate generation, calibration artifacts, empirical token ceilings,
        sealed inference, evaluator joins, metrics, and results remain
        NOT_CONFIGURED or RESULTS_LOCKED.
        '''
    ),
    code(
        r'''
        demo_implemented = {
            "fixed_candidate_provider",
            "scripted_scorer",
            "cdcv_gate_orchestration",
            "equal_budget_control_orchestration",
        }
        external_system_status = {
            system_id: "NOT_CONFIGURED"
            for system_id in DESIGN["systems"]["required"]
        }
        require(
            external_system_status["iedi_nb_lineage_diagnostic"]
            == "NOT_CONFIGURED",
            "IEDI diagnostic must not be impersonated by the demo",
        )
        require(
            external_system_status["kics_w_recon"] == "NOT_CONFIGURED",
            "KICS-W baseline must not be impersonated by the demo",
        )

        smoke_summary = {
            "evidence_status": "DEMONSTRATION_ONLY_RESULTS_LOCKED",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_blob_sha": UPSTREAM_BLOB_SHA,
            "runtime_schema": runtime_schema_status,
            "attestation_schema": attestation_schema_status,
            "released_answer_schema": released_answer_schema_status,
            "prediction_schema": prediction_schema_status,
            "commit_path_calls": commit_result.consumed.calls,
            "clarify_repair_path_calls": repair_result.consumed.calls,
            "conflict_path_calls": conflict_result.consumed.calls,
            "repair_route_path_calls": routed_result.consumed.calls,
            "full_equal_budget_control_calls": sum(
                event.resources.calls
                for event in full_budget_control.resource_events
            ),
            "configured_demo_components": sorted(demo_implemented),
            "model_backed_candidate_generation": MODEL_BACKED_CANDIDATE_GENERATOR_STATUS,
            "sealed_results_present": False,
        }
        require(smoke_summary["commit_path_calls"] == 3, "commit summary drift")
        require(
            smoke_summary["clarify_repair_path_calls"] == 6,
            "repair summary drift",
        )
        require(smoke_summary["conflict_path_calls"] == 0, "conflict summary drift")
        require(
            smoke_summary["repair_route_path_calls"] == 9,
            "route summary drift",
        )
        require(
            smoke_summary["full_equal_budget_control_calls"] == 9,
            "equal-budget summary drift",
        )
        print(json.dumps(smoke_summary, indent=2, sort_keys=True))
        '''
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10+"},
        "cdcv_gate": {
            "evidence_status": "DEMONSTRATION_ONLY_RESULTS_LOCKED",
            "run_mode_default": "DEMO",
            "upstream_repository": "toecm/iedi-mas",
            "upstream_path": "CA_IEDI_0803.ipynb",
            "upstream_commit": "5cff1e509efb09c24f9ac7e30075b6a131ee6fbc",
            "upstream_blob_sha": "5b83ce8dbdc0e147637ef499b7c4f7deabfbb653",
            "upstream_utf8_sha256": "0287daf13a8a863f267a8cd50acc1e6563ab2a64c852873bf59aa561bc616eaa",
            "source_outputs_copied": False,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n"
    TARGET.write_text(serialized, encoding="utf-8")
    print(f"Wrote {TARGET}")
    if REPOSITORY_MIRROR.is_file():
        REPOSITORY_MIRROR.write_text(serialized, encoding="utf-8")
        print(f"Wrote compatibility mirror {REPOSITORY_MIRROR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
