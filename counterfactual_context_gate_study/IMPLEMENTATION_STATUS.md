# Reference implementation status

## Evidence status

This package is a `DEMO_ONLY` reference orchestrator for a planned study. It
contains no completed community annotation, accepted empirical intervention
bundle, sealed test execution or RQ1/RQ2 result. `RESULTS_LOCKED` remains the
only valid result status. Passing the notebook, validator or unit tests
supports code-path behavior only—not reliability, fairness, latency, novelty
or effectiveness.

The local [`notebooks/CA_IEDI_0803.ipynb`](notebooks/CA_IEDI_0803.ipynb) is a
clean rewrite linked to upstream commit
`5cff1e509efb09c24f9ac7e30075b6a131ee6fbc`; the full commit, blob and content
hash record is in
[`notebooks/UPSTREAM_PROVENANCE.json`](notebooks/UPSTREAM_PROVENANCE.json).
The notebook contains invented neutral fixtures, a deterministic fixed-
candidate provider and a scripted offline scorer. It retains no saved output
and refuses sealed-test use in demo mode.

## Implemented reference contracts and control flow

`src/cdcv_gate/core.py`, `contracts.py`, and `pipeline.py` provide typed,
gold-free orchestration for:

- normalized candidate-score vectors, margin, entropy, preservation,
  targeted-response and context-quality features;
- pair-adjacent-violators isotonic calibration and the frozen binary
  preservation/targeted-response commit guards;
- the three actions `COMMIT`, `CLARIFY` and `ABSTAIN_ESCALATE` plus a separate
  benefit- and privacy-constrained one-time route;
- frozen candidate order, including non-committable `OTHER_UNLISTED`;
- one base, one hypothesis-preserving and one hypothesis-contrast scorer call
  per verification pass;
- a `ContractAttestation` boundary for schema, cross-record, leakage,
  protected-field and required-slot declarations;
- an `ACCEPTED`-status rule for sealed execution. `DEMO_ONLY` attestations and
  intervention/question/answer records are usable only in demo execution;
- frozen trusted-attestation and trusted-answer hash allowlists, so a
  caller-created `ACCEPTED` value is not sufficient authority;
- required-slot derivation from `ContractAttestation.required_context_slots`
  and the current card, with no caller-supplied action label or “missing
  discriminating slot” hint;
- a `ClarificationAnswerBroker` that releases a `ReleasedAnswer` only for the
  emitted case/question/slot and approved answer domain;
- a `StaticDemoAnswerBroker` containing invented answers for notebook and test
  fixtures only;
- one-slot patch application followed by separately validated
  `ReleasedAnswer.post_answer_branches` whose source hashes match the repaired
  card;
  original-card probes are not reused after clarification;
- `BudgetEnvelope` enforcement for calls, input tokens, output tokens and,
  when configured, cost;
- three/six/nine-call orchestration with at most one question and one route;
- a structured-context control that receives an explicit envelope with the
  same frozen call and token allocation and the same sealed admission boundary
  as the corresponding CDCV condition;
- per-stage resource telemetry and schema-valid gold-free prediction records;
  and
- an exact frozen prediction hash before a separate evaluator may join sealed
  action and sense labels.

An `ACCEPTED` attestation or answer is an eligibility condition, not evidence
that the corresponding empirical artifact has already been collected. The
demo constructs only `DEMO_ONLY` records.

## Runtime/evaluator boundary

The scorer projection contains only the utterance/dialogue, frozen candidate
definitions and permitted structured-card fields. Sealed fields—including
`reference_action`, `reference_sense_id`, `case_type`, acceptable
clarification-slot labels, expected probe relations, target labels and
validator scores—must not enter the scorer or controller request.

The answer broker is access controlled in the real design: it releases one
independently validated answer only after a matching question is emitted. A
wrong, unavailable or out-of-domain answer does not reveal a helpful patch and
the case ends as unresolved abstention/escalation. Predictions are frozen
before labels are mounted by a separate evaluator process. Python validation
and a notebook kernel are not security boundaries.

## `NOT_CONFIGURED` empirical components

The following are deliberately absent or placeholders:

- a model-backed end-to-end candidate generator (the fixed-candidate task is
  the primary design, and the generated-candidate analysis is secondary);
- community authoring, independent validation and accepted attestation
  artifacts;
- a sealed runtime/answer-broker/evaluator process deployment;
- production model providers, frozen model/prompt manifests and score
  adapters;
- numerical empirical input/output token and cost ceilings;
- IEDI, direct-LLM, KICS-W, confidence, self-consistency, calibrated/conformal
  and other required external empirical systems;
- a family-clustered development/calibration scheduler, full telemetry backend,
  bootstrap analysis and statistical report; and
- the consented application pilot.

The empirical token ceilings cannot be inferred from the invented demo. They
must be frozen with the model and prompt manifests before execution, and the
same `BudgetEnvelope` must be allocated to the primary method and equal-budget
control.

## Calibration lock

The reference score combines four monotone groups: base confidence,
preservation, targeted change and context quality. Group weights are selected
or fitted on development data. Only the one-dimensional calibration mapping
and commit threshold use calibration families, with equal total weight per
family. The binary target is whether `COMMIT(base_prediction)` is both action-
appropriate and sense-correct; it is not ordinary sense accuracy, which is
undefined for clarify/abstain cases.

With one accepted preserving and one accepted contrast probe per pass, both
preservation invariance and targeted response must equal `1.0` before
commitment. These two guards are fixed in the protocol before data. Any later
multi-probe design requires a new version and newly frozen thresholds.
