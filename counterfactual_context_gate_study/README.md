# Counterfactual Dialect-Context Verification study

Working paper title:

> **When to Interpret and When to Ask: Counterfactual Dialect-Context
> Verification for Selective Intended-Meaning Resolution**

Working mechanism name:

> **Counterfactual Dialect-Context Verification and Clarification Gate
> (CDCV-Gate)**

This directory is a clean, evidence-gated workspace for a new engineering
paper protocol/preprint. It does not revise the rejected CA-IEDI paper in place
or replace the separate cue-ablation study, which asks a different research
question.

## Notebook link and provenance

The local [`notebooks/CA_IEDI_0803.ipynb`](notebooks/CA_IEDI_0803.ipynb) is a
clean, protocol-aligned rewrite linked to the
[pinned upstream notebook](https://github.com/toecm/iedi-mas/blob/5cff1e509efb09c24f9ac7e30075b6a131ee6fbc/CA_IEDI_0803.ipynb),
not a copy of the mutable `main` revision. The upstream commit is
`5cff1e509efb09c24f9ac7e30075b6a131ee6fbc`, its Git blob is
`5b83ce8dbdc0e147637ef499b7c4f7deabfbb653`, and the retrieved UTF-8 content
has SHA-256
`0287daf13a8a863f267a8cd50acc1e6563ab2a64c852873bf59aa561bc616eaa`.
[`notebooks/UPSTREAM_PROVENANCE.json`](notebooks/UPSTREAM_PROVENANCE.json) is
the canonical provenance record. Upstream license status is unresolved, so
the source content must not be redistributed until that gate is cleared.

The rewrite contains no copied source code or saved source output. It removes
the source notebook's mutable dataset writes, credential interfaces, public
UI launch, audio/ASR, IPFS, Web3 and Hardhat paths. Its invented fixtures and
scripted scorer run in `DEMO_ONLY` mode and are code-path checks, not research
observations or empirical results.

## Central contribution

> We design a schema-constrained inference-time verifier that probes
> intended-meaning predictions with community-validated meaning-preserving and
> meaning-changing context interventions, then uses calibrated selective
> prediction to commit, request one targeted clarification, or abstain. We
> prespecify a held-out evaluation of that mechanism without claiming a result.

Only after the sealed study is completed may the contribution be written as
“we design and evaluate.” The manuscript contains a conspicuous result lock,
is labelled as a protocol/preprint, and presents no synthetic performance
value as evidence.

## Two research questions

**RQ1 — Selective reliability.** At matched coverage, does counterfactual
context verification reduce wrong committed interpretations compared with
always-answer, confidence-only, self-consistency and standard
context-injection baselines?

**RQ2 — Robustness and repair.** Does the proposed system preserve
interpretations under meaning-preserving changes, respond appropriately to
sense-changing contexts, and recover rejected cases using one targeted
clarification?

## Recommended design frozen in this workspace

- Three bounded English-use community resources in the United States, Nigeria
  and Korea; these do not represent global World-English coverage.
- 30 development, 60 calibration and 150 sealed-test expression families.
- 50 sealed-test families per community; 8 nested cases per family; 1,200
  sealed cases in total.
- Five independent variety-proficient validators per two-sense intervention
  bundle, with creator/validator separation and a 4-of-5 acceptance rule.
- Family-disjoint splits and a phrase/gloss/profile leakage audit.
- A fixed-candidate primary task to isolate intended-meaning selection, plus a
  clearly secondary end-to-end candidate-generation evaluation. The local
  notebook implements only a deterministic fixed-candidate demo provider;
  the empirical generator is `NOT_CONFIGURED`.
- A paired, family-clustered primary comparison against an equal-call/token
  structured-context control. Both systems must receive the same frozen
  `BudgetEnvelope`; the empirical input/output token ceilings remain
  `NOT_CONFIGURED` until model and prompt manifests are frozen.
- Frozen calibration thresholds and a sealed analysis.
- An `ACCEPTED` `ContractAttestation` before sealed inference. It identifies
  the case, family, source-card hash, ordered candidate-set hash and required
  context slots, and certifies schema, cross-record, leakage and protected-
  field checks without exposing an action or sense label.
- A `ClarificationAnswerBroker` that releases one matching `ReleasedAnswer`
  only after the system asks the corresponding validated question. Each
  answer selects separately validated post-answer probe branches rebased to
  the updated card; initial-card probes are never reused after repair.
- A secondary, consented and sandboxed text-prototype application pilot with
  frozen thresholds and descriptive feasibility outcomes; it is not a second
  confirmatory superiority test.
- A consent-qualified public-release contract requiring deidentified cases,
  labels, intervention contracts, clarification materials, predictions and
  evaluation code sufficient to regenerate the main tables. Hashes alone are
  not treated as reproducible data.

A separate design-size simulation is planning evidence only. Its synthetic
outcomes must never populate the Results section.

## Current IEDID/persona boundary

Current IEDID rows and personas may support expression discovery,
development examples, context-card schema construction and lineage baselines.
They may not constitute the sealed test set because they lack independently
validated counterfactual pairs, action labels and reliable leakage separation.
The live data also require deidentification, provenance/consent review,
license confirmation and revision pinning before any research release.

## Directory map

- `SYSTEM_SPEC.md` — implementable runtime mechanism and decision policy.
- `STUDY_PROTOCOL.md` — acquisition, annotation, splits, evaluation and
  statistical analysis.
- `CLAIM_LINEAGE.md` — inherited versus new components and prohibited claims.
- `EVIDENCE_REGISTER.md` — claim status and result-release gates.
- `config/study_design.json` — machine-readable frozen design.
- `data/schemas/` — JSON Schemas for context cards, families, cases,
  questions and prediction logs.
- `notebooks/CA_IEDI_0803.ipynb` — output-free, offline CDCV-Gate rewrite of
  the pinned public CA-IEDI notebook; demonstration only.
- `notebooks/UPSTREAM_PROVENANCE.json` — immutable upstream commit, blob,
  content hash and license-status record.
- `src/cdcv_gate/pipeline.py` — gold-free scoring, probe, clarification,
  attestation, answer-broker, routing, equal-budget and telemetry
  orchestration.
- `src/cdcv_gate/contracts.py` — interaction-scope, protected-field,
  cross-record and prediction-freeze guards.
- `manuscript/main.tex` — double-anonymous engineering-journal
  protocol/preprint draft.
- The non-anonymous manuscript title page is deliberately excluded from the
  public artifact until every named author confirms authorship and release.
- `manuscript/references.bib` — verified scholarly and lineage references.
- `scripts/validate_package.py` — dependency-free package checks.
- `tests/test_validate_package.py` — regression tests for the design locks.
- `runs/` and `data/private/` — intentionally untracked locations for model
  outputs and access-controlled annotation data.

## Runtime/evaluator separation

The inference process may receive the utterance, candidate order, structured
card, an `ACCEPTED` gold-free attestation, `ACCEPTED` intervention contracts,
question manifests and frozen model/controller resources. It must not mount
or receive `reference_action`, `reference_sense_id`, `case_type`, acceptable
clarification-slot labels or any other evaluator field. Required missing
slots are derived from `ContractAttestation.required_context_slots` and the
card, never from an action label or caller-supplied “missing slot” hint.
In sealed mode, the attestation integrity hash must also occur in a frozen
trusted manifest supplied by the external scheduler; a caller-created
`ACCEPTED` object is insufficient. The attestation binds a reviewed-value
manifest so safety review covers free-text values as well as field names.

Predictions are frozen and hashed before a separate evaluator process may
join sealed action/sense labels. The Python types and notebook are validation
layers, not a security boundary; sealed execution still requires process- and
access-level isolation.

## What is and is not implemented

The local package demonstrates the CDCV control flow with invented cases: one
three-call verification pass, one brokered clarification with a separately
validated post-answer pass, an optional one-time route, action logging, and a
structured-context control under an explicit call/token envelope. In sealed
mode, only `ACCEPTED` attestations, probe contracts, question manifests and
answer records are eligible, their hashes must occur in the corresponding
trusted manifests, and the control uses the same admission boundary.
`DEMO_ONLY` records are rejected.

No community-accepted benchmark bundle, sealed split, model-backed candidate
generator, production scorer, required external baseline suite, empirical
token/cost envelope, or RQ1/RQ2 result exists in this folder. Those components
remain `NOT_CONFIGURED`; `RESULTS_LOCKED` must remain in place.

## Immediate execution sequence

1. Obtain ethics approval/exemption and community-partner agreements.
2. Pin and sanitize the IEDID development snapshot; never expose identifiers.
3. Recruit community authors/validators and build at least 345 candidate
   families if the expected retention rate is 70%.
4. Accept 240 families, lock their splits, and publish only hashes before
   evaluation; retain for the confirmatory set only cases whose consent and
   governance status permits the minimum deidentified post-run release.
5. Freeze model versions, prompts, decoding, budgets, thresholds and the
   analysis plan.
6. Run development and calibration; inspect neither test labels nor test
   outcomes while tuning.
7. Execute all systems once on the sealed test and produce the locked tables.
8. Replace every `RESULTS_LOCKED` marker only from the auditable run manifest.
9. Run the separately labelled bounded application pilot with already frozen
   thresholds, then release the consent-authorized reproducibility bundle to a
   persistent repository. If the minimum bundle cannot be released, do not
   claim public reproducibility and reconsider the EAAI target.

## Validation

From this directory:

```powershell
python scripts/validate_package.py
python -m unittest discover -s tests -v
python scripts/execute_notebook.py notebooks/CA_IEDI_0803.ipynb
```

The local machine currently needs a LaTeX distribution with `elsarticle` to
compile the journal preprint. Package validation does not substitute for a
full LaTeX compile.
