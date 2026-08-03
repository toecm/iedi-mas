# CDCV-Gate system specification

## 1. Scope

CDCV-Gate sits between an intended-meaning model and a downstream action. It
does not infer a speaker's identity or private mental state. It estimates
whether a model's candidate selection is behaviorally consistent under
controlled, candidate-indexed context probes and whether the system should
commit, clarify, or abstain. This is a **calibrated behavioral-consistency
gate**, not a verifier of semantic truth, ground-truth intended meaning, or the
authenticity of supplied context. Permitted provenance and deterministic
conflict checks can reject malformed or contradictory cards, but they do not
authenticate contextual claims. Only evaluation labels, kept outside the
runtime process, determine whether a committed interpretation was correct.

Inputs:

- an utterance and short dialogue co-text;
- a fixed candidate-sense set for the primary benchmark;
- a schema-valid, episode-specific context card;
- an `ACCEPTED`, gold-free `ContractAttestation` carrying the required context
  slots and validation-manifest hashes but no reference action or sense;
- `ACCEPTED` hypothesis-preserving and hypothesis-contrast benchmark probes,
  indexed relative to each candidate, or reviewed probe operators;
- an `ACCEPTED` clarification-question manifest and an access-controlled
  `ClarificationAnswerBroker`;
- separately `ACCEPTED` post-answer branches for each releasable clarification
  answer; and
- frozen calibration parameters and a `BudgetEnvelope` covering calls, input
  tokens, output tokens and, when applicable, cost.

Outputs:

- `COMMIT(sense_id)`;
- `CLARIFY(context_slot)`; or
- `ABSTAIN_ESCALATE(reason_code)`.

The reference package has distinct `DEMO_ONLY` and sealed acceptance states.
Invented demo attestations, interventions, questions and answers may exercise
control flow but must be rejected by sealed execution. An `ACCEPTED` status
means that a prespecified validation process and its hashes are present; it
does not expose the evaluator's action or sense label. Sealed action labels,
reference senses, case types and acceptable-clarification labels are never
available to the inference process and are joined only after predictions are
frozen by hash in a separate evaluator process. The notebook and Python type
checks are not an access-control boundary.

The planned end-to-end system first runs a frozen generator
\(G(x)=\{m_1,m_2,\texttt{OTHER\_UNLISTED}\}\) from the utterance/dialogue
before context scoring. The generated-candidate evaluation is secondary. The
primary fixed-candidate condition replaces \(G\)'s output with two validated
competing senses plus `OTHER_UNLISTED` to isolate verification and selection
from candidate recall. A missing validated sense is scored as candidate-recall
failure, not selector failure. `OTHER_UNLISTED` is never a committable sense.
The local rewrite currently provides only a deterministic fixed-candidate
demo provider. A model-backed generator and all external empirical systems are
`NOT_CONFIGURED` and cannot produce paper results.

## 2. Context-card policy

Permitted fields are linguistically relevant and interaction scoped:

- `relationship_role` (for example peer, supervisor–subordinate, service
  encounter; no names);
- `setting`;
- `formality`;
- `discourse_goal`;
- `preceding_speech_act`;
- `situation`;
- `variety_cue`, only when voluntarily self-declared or explicitly supplied
  for the interaction; and
- provenance, confidence and expiry for every field.

The production system must not infer or manipulate race, ethnicity,
nationality, religion, gender, sexuality, disability, age or other protected
identity attributes. Community affiliation is not a counterfactual switch.
Variety labels in the controlled benchmark index validated resources; they are
not immutable properties of a person.

## 3. Candidate scoring

For candidate set \(Y(x)=\{y_1,\ldots,y_J\}\), the frozen base model returns

\[
p_0(y\mid x,c), \qquad
\hat y_0=\arg\max_y p_0(y\mid x,c), \qquad
m=p_0(\hat y_0\mid x,c)-p_0(y_{(2)}\mid x,c),
\]

where \(m\) is the top-two margin. If an API exposes no calibrated token
probabilities, the implementation must use a frozen, calibration-tested score
adapter and disclose that limitation. Self-reported verbal confidence alone
is not treated as probability.

## 4. Candidate-indexed counterfactual probes

The inference process never receives the sealed reference sense or reference
action. Instead, each runtime episode provides a symmetric operator branch for
every listed non-`OTHER_UNLISTED` candidate. These branches are constructed
without access to the base-episode label. After the scorer predicts
\(\hat y_0\), the gate selects the branch indexed by its own prediction. For
candidate \(m\), the branch contains:

- \(C^+(m)=\{T^+_k(c,m)\}\): changes whose operator contract is
  `SAME_AS_BRANCH_SOURCE`, called **hypothesis-preserving probes** because they
  are designed to preserve candidate \(m\), not asserted to preserve the
  unknown meaning of the base episode; and
- \(C^-(m)=\{(T^-_k(c,m,t_k),t_k)\}\): minimal changes whose operator
  contract targets a contrasting listed candidate \(t_k\), called
  **hypothesis-contrast probes**.

The utterance, candidate set, and fixed variety cue do not change. Branch
metadata state expected transformation relations but never state which
candidate is correct in the original episode. Therefore, passing a branch
shows candidate-conditioned behavioral consistency only; it is not evidence
that the branch candidate is the speaker's true intended meaning.

The gate computes:

\[
I^+=\frac{1}{K_+}\sum_k
\mathbf 1[\hat y\{x,T^+_k(c,\hat y_0)\}=\hat y_0],
\]

\[
R^-=\frac{1}{K_-}\sum_k
\mathbf 1[\hat y\{x,T^-_k(c,\hat y_0,t_k)\}=t_k].
\]

It additionally logs mean distributional divergence from the base prediction,
schema violations, incompatible-card fields, invalid candidate IDs and
leakage flags. A raw prediction change is not a success on \(C^-\); only a
change to the branch's community-validated contrast candidate counts.

Before any branch reaches inference, a frozen cross-record validator and
`ContractAttestation` must
bind the case ID, family ID, source-card hash and ordered candidate-set hash,
and confirm that (i) the two distinct branch IDs match exactly the two runtime
non-`OTHER_UNLISTED` candidates; (ii) every contrast target differs from its
branch source and is the other listed core candidate; (iii) source context
hashes match the base card; (iv) patches alter only declared permitted slots;
(v) the variety cue and other fixed fields remain unchanged; (vi) probes do
not reuse the base case as their own probe; and (vii) every probe and branch
has passed the prespecified acceptance rules. The validator recomputes source
and patched-card hashes rather than trusting caller-supplied hashes. In sealed
execution the attestation and every referenced record must have status
`ACCEPTED`, and the attestation integrity hash must occur in the external
frozen trusted manifest; `DEMO_ONLY` artifacts and caller-created accepted
flags are ineligible. The scheduler manifest also binds the reviewed-value
manifest used to safety-check free-text field values. A failure makes the episode
ineligible for commitment and is logged; it is never repaired by selecting a
different branch after observing model scores.

## 5. Verification feature vector

The frozen controller consumes

\[
z=[m,\ 1-H(p_0),\ I^+,\ P^+,\ 1-D^+,\ R^-,\ M^-,\
V,\ U,\ \Gamma],
\]

where \(P^+\) is probability retention, \(D^+\) is base-2
Jensen--Shannon divergence, \(M^-\) is the changing-target contrast margin,
\(V\) is intervention validity, \(U\) is context completeness, and \(\Gamma\)
is deterministic conflict severity. Budget is an action constraint rather than
a correctness feature. Raw free-text identities and unapproved context fields
are never controller inputs.
Expression/gloss/prompt leakage indicators are retained as offline audit
metadata and never supplied to the learned controller, which would otherwise
be able to recognize benchmark artifacts.

The primary controller is a transparent, low-dimensional monotone score whose
form and parameters are selected/fitted on development data. The 60-family
calibration split is used only for a one-dimensional probability mapping and
operating threshold, with each family given equal total weight. A conformal or
selective alternative is reported separately and receives no guarantee unless
its assumptions hold. All forms, features, parameters, and thresholds are
frozen before test execution.

## 6. Decision policy

The reference rule is ordered to avoid abstention gaming:

1. `ABSTAIN_ESCALATE` if the card is invalid, contradictory, out of scope, or
   protected-attribute manipulation is requested. A controllable benchmark
   leakage finding invalidates the case or run offline rather than becoming a
   runtime gate feature.
2. `COMMIT(ŷ)` only if calibrated probability that committing is both action-
   appropriate and sense-correct is at least \(\tau\), hypothesis-preservation and targeted-response thresholds are satisfied,
   and no conflict/validity guard fires.
3. `CLARIFY(slot)` if one approved missing or conflicting slot has positive
   expected utility and the one-question budget remains.
4. `ABSTAIN_ESCALATE` otherwise.

The reference design uses exactly one accepted probe of each relation per
pass. Consequently, both the hypothesis-preservation invariance and targeted-
response indicators must equal `1.0` before commitment. These binary hard
guards are fixed in the protocol before data; only the reliability calibration
and operating threshold \(\tau\) are fitted on calibration families. A later
multi-probe design requires a versioned protocol and newly frozen thresholds.

Every method first applies a frozen **hard eligibility mask**. Cases with an
invalid card or probe bundle, prohibited field, unresolved hard conflict,
`OTHER_UNLISTED` base prediction, missing required discriminator, controllable
leakage violation, out-of-scope input, or exhausted hard budget are ineligible
for commitment. Ranking and thresholding operate only on eligible cases.
Ineligible cases can clarify when the policy permits or must abstain/escalate;
they are never forced to commit to meet a coverage target.

Required discriminators are derived only by comparing the current card with
`ContractAttestation.required_context_slots`. The runtime episode carries no
reference-action-derived “missing slot” field and no caller-supplied validity
booleans. Attestation acceptance, card structure, provenance, conflicts and
resource limits are checked directly.

Calibration fixes an operating threshold and a common feasible target coverage
before test execution. The target is no greater than the minimum eligible
coverage achieved on calibration by the two primary methods and no greater
than the action-appropriate ceiling. A prespecified score-ranking rule may
select the top target-coverage fraction from each method's eligible cases
without reading test labels; ties use a frozen case hash. If a method has too
few eligible test cases to reach the frozen target, its target is reported as
unattainable and its observed coverage and risk are reported without forcing
additional commitments. Full test risk--coverage curves are descriptive. No
label-dependent test threshold tuning is permitted. Family-cluster bootstrap
resampling carries the method-specific eligibility masks into every replicate
and never converts an ineligible case into a commitment.

## 7. One-question clarification

For approved questions \(Q(c)\), choose

\[
q^*=\arg\max_{q\in Q(c)}
\mathbb E_{a\sim \hat P(a\mid q,c)}
[H(p_0)-H(p_0\mid a)]
-\lambda_I C_I(q)-\lambda_P C_P(q)-\lambda_C C_C(q),
\]

subject to the question being community validated, non-leading,
non-sensitive, answerable and mapped to exactly one context slot. The
interaction, privacy and compute weights are frozen on development data.
The primary implementation reads a frozen clarification-scenario manifest.
For every approved `case_id`--`question_id`--answer scenario, the manifest
records a normalized answer prior, the reusable probe-score or precomputed
hypothetical-score reference, its source split and hash, and the resulting
context-patch reference. The answer priors and all score references are frozen
before the sealed run; test labels and reference actions are absent from the
runtime view. The primary selector uses only already scored probe distributions
grouped by context slot (or score references computed and sealed before the
test run), so question ranking consumes **zero additional model calls**. A
question without a complete valid manifest entry is ineligible. Any exploratory
selector that performs live hypothetical calls is reported separately and its
calls are added to both the declared system and comparator budgets.

Community validation applies directly to curated benchmark instances. An
online system may instantiate only a reviewed operator/template library; every
generated intervention must pass fidelity checks, and a failed check forces
abstention. The same model must not generate and approve its own probes.

In the controlled benchmark, answers are neither live-user observations nor
model generated. A `ClarificationAnswerBroker` releases one independently
validated `ReleasedAnswer` only after the system emits a matching question for
that `case_id`, `question_id` and context slot. The answer ID must belong to
the question manifest's frozen domain, and the question/answer manifest hashes
must match. A wrong, unavailable or unnecessary question receives no helpful
answer and terminates as unresolved `ABSTAIN_ESCALATE` rather than remaining a
second clarifiable state.

The released answer applies exactly one permitted-slot patch with provenance
`standardized_clarification`. It also selects the corresponding entry in
`ReleasedAnswer.post_answer_branches`. Those branches are separately
accepted and their source hashes are recomputed against the updated card.
Initial-card probe contracts are never reused after clarification. The
verifier then runs once more and must either commit, route once, or
abstain/escalate. Clarification recovery therefore estimates controlled repair
potential rather than real-user behavior. Repeated questioning is outside this
paper. `StaticDemoAnswerBroker` is only an invented-fixture implementation and
is forbidden as evidence or in sealed execution.

## 8. Escalation

Routing to a larger model is allowed only after the behavioral-consistency gate cannot
resolve uncertainty, the privacy policy permits the route and the budget has
capacity. The larger model receives the minimum necessary fields. Escalation
is logged separately from abstention and does not erase the initial failure.
Because dynamic model routing is inherited from KICS-W, it is an operational
option rather than a novelty claim.

With one base call plus one selected hypothesis-preserving and one selected
hypothesis-contrast call, a verification pass costs three scorer calls. The
primary clarification selector adds zero calls. One initial pass, one
post-answer pass, and at most one routed pass therefore remain within the
prespecified cap of nine scorer calls, excluding candidate generation. The
equal-budget primary comparator receives the same allocated call and token
cap even when CDCV-Gate consumes fewer calls, and passes through the same
split, authorization, trusted-attestation, out-of-domain and conflict
admission checks. `BudgetEnvelope` enforces the
call, input-token and output-token ceilings before each charge, and optionally
enforces monetary cost. The empirical token and cost ceilings are
`NOT_CONFIGURED`: they must be frozen with model, tokenizer, prompt and pricing
manifests before either primary system is run. Demo token counts are smoke-test
fixtures and must not be reused as empirical allocations.

## 9. Pseudocode

```text
VERIFY(episode, budget_envelope, broker):
    ledger = BudgetLedger(budget_envelope)
    if not accepted(episode.attestation) or not schema_valid(episode.card):
        return ABSTAIN_ESCALATE(POLICY)
    if sealed_label_visible_to_runtime(episode):
        return ABSTAIN_ESCALATE(ACCESS_SEPARATION)
    missing = required_slots(episode.attestation) - present_slots(episode.card)
    p0, y0 = score(project_gold_free(episode))
    if y0 == OTHER_UNLISTED:
        return ABSTAIN_ESCALATE(CANDIDATE_COVERAGE)
    branch = accepted_symmetric_branch_for(y0, episode.card)
    plus  = score_all(branch.hypothesis_preserving)
    minus = score_all(branch.hypothesis_contrast)
    z = verification_features(p0, y0, plus, minus, episode.card,
                              budget_envelope)
    action = calibrated_controller(z)
    if action is COMMIT:
        return COMMIT(y0)
    if action is CLARIFY and one_question_remaining(ledger):
        q = select_accepted_question(z, missing)
        answer = broker.release_after_matching_question(episode.case_id, q)
        if answer is unavailable_or_mismatched:
            return ABSTAIN_ESCALATE(CLARIFICATION_UNRESOLVED)
        card2 = apply_one_slot_patch(episode.card, answer)
        branches2 = accepted_post_answer_branches(answer, card2)
        return VERIFY_REPAIRED_ONCE(card2, branches2,
                                    no_further_question=true)
    if route_is_safe_and_beneficial(z, episode.card, ledger) and not ledger.routed:
        routed_scores = score_with_large_model(episode, branch)
        return VERIFY_ROUTED_ONCE(routed_scores, no_further_route=true)
    return ABSTAIN_ESCALATE(action.reason)
```

## 10. Required logs

Every decision records immutable hashes for the gold-free runtime case, context card,
attestation, intervention bundle, question/answer manifests, applied answer,
post-answer branch set, prompt, candidate order, model/version, controller,
calibration split, `BudgetEnvelope`, code commit and pricing snapshot, plus all
pass-level component scores, action, reason code, allocated/consumed calls and
tokens, latency and peak memory. The evaluator joins access-controlled labels
only after predictions are frozen.
