# Confirmatory study protocol

## Objective

Evaluate whether CDCV-Gate reduces wrong committed intended-meaning selections
at matched coverage and whether its candidate-indexed counterfactual checks
provide useful calibrated behavioral-consistency signals and one-question
repair. The gate does not verify semantic truth, recover a speaker's private
ground truth, or authenticate supplied context.

## Unit and sample

The independent sampling and resampling unit is the **expression family**,
not the nested case, intervention, model call or validator rating.

| Split | Families | Families/community | Cases/family | Total cases | Purpose |
|---|---:|---:|---:|---:|---|
| Development | 30 | 10 | flexible, target 8 | about 240 | prompts, controller form, costs, question templates |
| Calibration | 60 | 20 | 8 | 480 | fit calibration and freeze thresholds |
| Sealed test | 150 | 50 | 8 | 1,200 | one-shot confirmatory evaluation |
| Total retained | 240 | 80 | — | about 1,920 | — |

The previous Monte Carlo planning exercise supports 150 test families for a
nominal pooled five-percentage-point paired effect under its stated base
assumptions. It does not power within-community superiority. If annotation
retention is about 70%, recruit approximately 345 candidate families (115 per
community) to retain 240; update the recruitment target using a blinded
validation pilot.

## Family construction

An accepted family contains one fixed ambiguous expression, two distinguishable
candidate senses, a fixed three-candidate set (the two senses plus
`OTHER_UNLISTED`), and eight cases:

1. canonical context supporting sense A;
2. canonical context supporting sense B;
3. meaning-preserving variant of A;
4. meaning-preserving variant of B;
5. minimal meaning-changing variant from A toward B;
6. minimal meaning-changing variant from B toward A;
7. underspecified case whose gold action is `CLARIFY`; and
8. contradictory or out-of-scope case whose gold action is
   `ABSTAIN_ESCALATE`.

The exact utterance is fixed within intervention pairs. Candidate order is
balanced and logged. Intervention edits are limited to approved context slots;
protected identity attributes are never edited.

For every runtime episode, an independent probe team constructs a symmetric
branch for each of the two non-`OTHER_UNLISTED` candidates without access to
the sealed base action or sense. The evaluated model selects its branch using
its own base prediction. A confirmatory branch contains one community-
validated **hypothesis-preserving probe**, designed to retain its branch
candidate, and one community-validated **hypothesis-contrast probe**, designed
to support the other listed candidate. The selected branch therefore requires
three scorer calls including the base. These relations are defined relative
to the model-selected candidate, not to the unknown intended meaning of the
base episode. Passing them is evidence of candidate-conditioned behavioral
consistency, not semantic correctness or context authenticity. The contrast
target is part of a candidate-conditioned transformation contract; it does
not reveal the correct sense for the original episode. Selecting
`OTHER_UNLISTED` always produces candidate-coverage abstention, never a
commitment.
The scorer sees only the resulting context card and common candidates, not
relation names, target IDs, validation ratings, or reference labels. The
controller applies the operator contract only after scores are returned.

A frozen cross-record validator runs before inference. It requires exactly two
distinct branches matching the two runtime non-`OTHER_UNLISTED` candidates;
requires every contrast target to be the other core candidate; verifies that
source-card hashes match, a probe does not reference itself, patches change
only declared permitted slots, and fixed variety fields do not change; and
admits only bundles that passed all acceptance rules. A failed check makes the
episode ineligible for commitment and is logged. No branch may be substituted
after model scores have been observed.

## Community authorship and validation

- Recruit community members with self-described relevant language histories;
  do not infer expertise from nationality alone.
- Compensate creators and validators at locally appropriate rates disclosed in
  the ethics materials.
- A creator cannot validate the same bundle.
- Use five independent variety-proficient validators per two-sense bundle.
- Elicit an open interpretation before showing candidate senses.
- Then collect candidate choice, action label, naturalness, contextual
  coherence, minimality, sensitivity concerns and a short rationale.
- Accept a target/action only with at least 4-of-5 convergence, median
  naturalness and coherence at least 4/5, and, for a meaning-changing edit,
  median minimality at least 4/5; require no gloss-leakage finding, no
  unresolved safety flag, and a traceable source or community-authored
  provenance statement. Accepted clarification questions additionally require
  median leadingness at most 2/5 (1 = not leading).
- Report raw agreement, entropy and Krippendorff's alpha with family-clustered
  uncertainty where appropriate. Do not turn agreement into a claim about all
  speakers.

### Annotation workload and feasibility gate

The retained design contains exactly 1,920 base episode cases
(240 families $\times$ 8). Each base episode has four symmetric probe instances
(two candidate branches $\times$ one hypothesis-preserving and one
hypothesis-contrast probe), yielding 7,680 probe instances. At five validators
per instance, probe validation alone requires 38,400 probe ratings; validating
the 1,920 base cases at the same depth requires another 9,600 ratings.

There is one gold-`CLARIFY` case per retained family, hence 240 such cases
across all splits and 150 in the sealed test. The action benchmark needs at
least one accepted question per `CLARIFY` case: 240 accepted question items or
1,200 ratings at five validators. The planned information-gain-versus-random
question contrast requires at least two accepted options per `CLARIFY` case,
so its minimum is 480 accepted question items or 2,400 ratings. If $Q$
candidate question items are submitted, including items later rejected, the
actual question-rating workload is $5Q$, with $Q\ge480$. The final $Q$, the
expected rejection/replacement allowance, time per rating, compensation rate,
platform charges, and adjudication hours must be frozen in the ethics and
budget plan before recruitment. Thus the minimum accepted-item workload for
the full planned design is 50,400 ratings (9,600 base + 38,400 probe + 2,400
question ratings), excluding rejected candidates, qualification tasks, and
adjudication.

Consent materials must separately describe expression authorship, base-case
rating, candidate-indexed probe rating, question rating, possible quotation,
retention, controlled release, and withdrawal. Recruitment cannot begin until
the institution confirms that the full workload, compensation, community
balance, privacy protections, and replacement allowance are feasible; if not,
the design is resized and the power simulation rerun before preregistration.

## Data split and leakage controls

Splits are family disjoint. Before test labels are sealed, canonicalize and
hash expressions, paraphrases, glosses and profile phrases. Audit the test
against:

- every IEDID revision and seed table used by any system;
- IUUY/IEDI/KICS-W/CA-IEDI examples and prompts;
- persona/profile files and few-shot examples;
- development and calibration families;
- question templates that reveal a sense label; and
- external retrieval indexes or model memory supplied at inference time.

Semantic near-duplicates receive blinded human adjudication. Any test family
with prohibited overlap is removed before the split hash is frozen. Training
data inside proprietary foundation models cannot be fully audited; disclose
this residual risk and include memorization-oriented probes without claiming
proof of absence.

The inference runner mounts only gold-free runtime episodes, symmetric probe
bundles, context cards, and the approved question bank. It cannot mount the
sealed-label store. Predictions are written, hashed, and made immutable before
a separate evaluator joins labels by `case_id`. Automated tests reject any
reference action, sense, case type, or acceptable clarification slot in the
runtime view.

## IEDID use

Current rows and personas are allowed only for expression discovery,
development examples, profile-schema construction and lineage baselines.
They are prohibited as sealed-test gold. Pin the exact revision, remove direct
and quasi-identifiers, confirm licenses and provenance, and record which rows
influenced each development artifact.

## Reference actions

| Evidence state | Reference action | Sense scoring |
|---|---|---|
| Context sufficient and coherent | `COMMIT(target_sense)` | committed sense must equal validated target |
| Alternate coherent context | `COMMIT(contrast_sense)` | committed sense must equal validated contrast |
| Discriminating field missing | `CLARIFY(target_slot)` | slot must be in validated acceptable set |
| Contradictory, unsafe or out of scope | `ABSTAIN_ESCALATE` | no sense commitment allowed |

## Systems

Freeze model versions, prompts, decoding, candidate ordering and resources.
Include:

1. frozen public-notebook IEDI (`IEDI-NB`), when data/license gates pass;
2. a frozen non-LLM contextual bi-encoder/cross-encoder ranker;
3. direct LLM, utterance/dialogue only;
4. `Direct-Random@C*`, a label-blind case-hash subsample used only to project
   always-answer risk to common coverage;
5. `KICS-W_RECON`, a paper-faithful persona/codebook reconstruction whose
   prompt, codebook, router omissions, and assumptions are frozen/disclosed;
6. structured context without verification;
7. separate margin/normalized-entropy gating;
8. repetition/sample-agreement gating;
9. semantic-entropy gating;
10. isotonic/logistic calibrated gating;
11. conformal-risk control as a separate method with explicit assumptions;
12. probe-only frozen aggregation using the same probe library;
13. structured-uncertainty/EVPI clarification using the same question bank and
    costs but no dual probes;
14. CDCV-Gate without clarification;
15. CDCV-Gate with one clarification; and
16. an equal-call/token structured-context control, the primary comparator.

Because CDCV-Gate makes additional calls, compare it with self-consistency and
the structured-context control under the same maximum calls and token budget.
Report both allocated and actually consumed budgets. Cache only deterministic,
system-independent inputs; never reuse a model answer across conditions.

The equal-budget context control makes three stochastic calls with the same
utterance and full structured card, averages candidate distributions, adds
sample agreement, and uses the same family-weighted calibration split; it does
not receive intervention relations. Freeze temperature, seeds, aggregation,
call cap, mean-token cap, and threshold rule. Always-clarify, oracle-slot, and
random-question policies are bounds/negative controls rather than competitive
systems.
Always-answer is reported at native coverage one; its random, label-blind
projection is the only “always-answer” matched-coverage row. Confidence gating
is reported separately as the deployable selective version.

For RQ2, a standardized answer harness supplies controlled repair evidence.
Independent validators approve an episode-specific question--slot--answer
patch. The runtime receives that patch only after it asks the matching
question; a wrong or unnecessary question receives no informative answer.
This is not a live-user experiment and does not estimate natural answer rates,
burden, or user satisfaction.

Question selection reads a frozen clarification-scenario manifest. Each
approved `case_id`--`question_id`--answer entry stores a normalized answer
prior, a reusable probe-score or hypothetical-score reference, the source
split and immutable hash of that score, and the validated context-patch
reference. Priors and references are frozen before sealed execution, and the
runtime manifest contains no reference action or sense. In the primary system,
all score references resolve to distributions already produced within the
allocated three-call probe pass (or to development/calibration-only static
tables); question ranking therefore adds zero test-time model calls. Missing
or invalid manifest entries make a question ineligible. Any exploratory live
hypothetical scoring is a separately budgeted analysis and cannot be counted
inside the primary nine-call cap.

## Prespecified bounded application pilot

The application pilot is secondary and non-confirmatory. It activates only
after the sealed benchmark analysis is complete and after a pilot-specific
sample-size or precision plan, interface build, model versions, prompts,
thresholds, outcomes, stopping rules, and missing-data rules are frozen. It
compares CDCV-Gate with one question against the structured-uncertainty
question comparator in a non-high-stakes, text-only mediation sandbox.

- Use newly consented interactions that are expression-family disjoint from
  development, calibration, and confirmatory test data.
- Do not use pilot records to tune, select, or replace any confirmatory model,
  prompt, controller, threshold, question, or result.
- Record the participant-supplied or participant-confirmed intended
  interpretation before feedback and confirmation or rejection of the final
  interpretation after at most one clarification.
- Report participant-confirmed wrong-commit burden, appropriate commitment or
  abstention, one-question repair, unnecessary questions, abandonment,
  interaction burden, latency, tokens, calls, memory, and cost.
- Require separate ethics review and consent for participation, retention,
  quotation, and release. Collect no audio, infer no protected attribute,
  create no persistent cultural profile, and trigger no high-stakes action.
- Limit inference to the frozen pilot setting, with participant- and
  expression-family-clustered uncertainty where estimable. Do not describe the
  pilot as a powered confirmatory superiority test or evidence of general
  real-world outcomes.

## Two IEDI regimes

- **Family-held-out:** remove sealed families from the lexicon; report IEDI as
  a lineage coverage/abstention diagnostic, not as a competitive selector.
- **Known expression, unseen context (secondary):** preserve a frozen mapping
  for known expressions but hold out every episode context. Label this regime
  separately and do not pool it with family-held-out evaluation.

## Outcomes

Primary:

- wrong-commit rate among committed cases (selective risk) at matched coverage.

A commitment is wrong if either (i) the reference action is `CLARIFY` or
`ABSTAIN_ESCALATE`, so no commitment was warranted, or (ii) the reference
action is `COMMIT` but the selected sense differs from the validated target.
Coverage is the proportion of all eight case types receiving `COMMIT`.
Consequently, safe full-action coverage cannot exceed 6/8 = 0.75. We also
report interpretation-only selective risk on the six commit-eligible cases and
unsupported-commit rates separately on clarify- and abstain-labelled cases.

Required secondary:

- coverage and the risk–coverage curve;
- area under the risk–coverage curve (AURC; orientation stated);
- three-way action confusion matrix;
- preservation invariance on accepted preserving pairs;
- appropriate counterfactual-flip rate on accepted changing pairs;
- clarification recovery, unnecessary-question rate and final accuracy after
  one question;
- per-community point estimates and confidence intervals;
- latency distributions, input/output tokens, model calls, peak memory and
  monetary cost under a dated price snapshot; and
- failure categories: dominant-variety default, persona overreach, context
  neglect, context overreaction, conflict miss, leakage, unsafe question and
  invalid output.

Always report action accuracy alongside conditional sense accuracy so a model
cannot look good by abstaining on everything.

Primary clarification recovery uses **all gold-`CLARIFY` cases in the evaluated
split** as its intention-to-interact denominator (150 cases in the sealed
test). The numerator requires the system to ask an acceptable matching
question, receive the one standardized answer, and then make a correct commit;
not asking, asking a nonmatching question, or failing to commit correctly
counts as failure. Conditional recovery among cases on which a matching
question was actually asked is reported only as a secondary diagnostic,
together with the matching-question rate. The unnecessary-question rate is the proportion of
sufficient or abstain-labelled cases on which a system asks. AURC integrates
selective risk over the prespecified common coverage range; lower is better,
zero-commit points are undefined and excluded, and the exact trapezoidal/tie
rule is frozen in analysis code.

## Confirmatory analysis

The primary RQ1 contrast is CDCV-Gate **without clarification** versus the
equal-budget structured-context control. RQ2 compares CDCV-Gate with one
question against matched one-question policies. Before any ranking, each
method applies its frozen hard eligibility mask. Invalid cards or probe
bundles, protected or prohibited fields, unresolved hard conflicts,
`OTHER_UNLISTED`, missing required discriminators, controllable leakage
violations, out-of-scope inputs, and hard-budget failures can never be forced
to `COMMIT`.

A common feasible target coverage is chosen on calibration, capped at 0.75,
and no greater than the minimum eligible calibration coverage of the two
primary methods. On test, each method ranks only eligible cases under its
frozen score rule without reading labels; ties are resolved by a frozen case
hash. If fewer than the frozen target fraction are eligible on test, matched
coverage is reported as unattainable for that method and observed coverage and
risk are reported without forced commitments. Full risk--coverage curves are
descriptive.

Use 10,000 family-cluster bootstrap resamples stratified by community for the
paired risk difference. Every resample preserves each method's hard eligibility
mask and recomputes ranking only among eligible resampled cases; an ineligible
case never becomes a commitment. Replicates in which the frozen target is
unattainable are flagged and summarized rather than repaired by forced
commitment. The primary success criterion is:

- the upper endpoint of the two-sided 95% confidence interval for
  `risk_CDCV - risk_control` is below zero at matched coverage; and
- at the separately frozen operating point, CDCV coverage is no more than five
  percentage points below the control.

Report the point estimate even if the criterion fails. Apply Holm correction
to the prespecified family of secondary system contrasts. Use hierarchical or
mixed-effects sensitivity models with family and case nesting, but do not let
them replace the paired primary estimand.

Per-community analyses are heterogeneity and harm-guard analyses, not powered
claims of superiority. Flag a community if the upper confidence bound permits
a prespecified practically important increase in wrong-commit risk; interpret
with community partners before release.

## Calibration and sealing

Controller form and clarification-cost weights are chosen on development.
All calibrator parameters and thresholds are fitted on the 60-family
calibration split. Then freeze:

- code commit and environment;
- data/split hashes;
- model identifiers and dates;
- prompts and candidate order seeds;
- call/token budgets;
- all estimands and confidence-interval code; and
- table shells.

Run a blinded integrity check, execute each system on the sealed test once,
and preserve raw logs. Corrections after unsealing require a dated deviation
note and a full rerun identifier.

## Ethics and governance gates

No recruitment or annotation begins before institutional ethics review or a
documented exemption. Consent must cover storage, quotation, controlled
release and withdrawal. Release deidentified structured labels where
permitted; avoid raw audio and identity-bearing text by default. Community
partners review labels, failure analyses and example quotations before public
release. The interface exposes and lets users correct interaction context; it
does not silently construct persistent cultural profiles.
The ethics submission and funding approval must include the workload and cost
feasibility calculation above, validator time estimates, locally appropriate
compensation, replacement rates for rejected items, and a stopping rule if
participant burden or budget makes the prespecified sample infeasible.

Before sealing, consent and governance review must determine the authorized
release level for each structured record. After the analysis is frozen, the
minimum reproducibility bundle contains deidentified cases and labels, probe
contracts, approved questions and standardized clarification answers, frozen
predictions, model/prompt/controller manifests, and evaluation code sufficient
to regenerate every reported table. Release authorized records publicly;
place them under controlled access only where consent, license, or community
governance requires it, and document every withheld field and reason. Raw
audio and identity-bearing source text are never public by default. Hashes and
aggregates support audit integrity but do not substitute for the structured
records required to reproduce the analyses.
