# Claim and component lineage

**Non-review metadata:** this file contains author-identifying repository and
submission history. It must not be included in the double-anonymous manuscript
package. The review manuscript uses an anonymized artifact placeholder; the
exact pins below are retained for editors, internal audit, and restoration
after review.

This file distinguishes the new study from IUUY, IEDI, KICS-W and the
unpublished CA-IEDI submission. “New” below means new relative to this local
research lineage, not a claim of global novelty or patentability.

## Reproducibility pins

- IEDI repository: `toecm/iedi-mas`
- Public-notebook commit audited previously:
  `3b9a12dd61fd2f37eee3bc47f754008b204fad1e`
- Notebook: `IEDI_MAS Nov 28.ipynb`
- Notebook blob SHA: `a2b7ddb6cb28bb8ec5dca78f3b88bf1da1e93dc4`
- CA-IEDI integration notebook: `CA_IEDI_0803.ipynb`
- CA-IEDI integration-notebook commit:
  `5cff1e509efb09c24f9ac7e30075b6a131ee6fbc`
- CA-IEDI integration-notebook blob SHA:
  `5b83ce8dbdc0e147637ef499b7c4f7deabfbb653`
- CA-IEDI integration-notebook UTF-8 SHA-256:
  `0287daf13a8a863f267a8cd50acc1e6563ab2a64c852873bf59aa561bc616eaa`
- Pinned upstream notebook:
  <https://github.com/toecm/iedi-mas/blob/5cff1e509efb09c24f9ac7e30075b6a131ee6fbc/CA_IEDI_0803.ipynb>
- Protocol-aligned derivative and provenance record:
  `notebooks/CA_IEDI_0803.ipynb` and
  `notebooks/UPSTREAM_PROVENANCE.json`
- Upstream license status:
  **TO BE CONFIRMED before redistributing source content.** The derivative is
  a clean rewrite and copies neither source code nor source outputs verbatim.
- Audited IEDID snapshot used for aggregate development audit:
  `0bc6208b0a28f95268771df58d4f9983193c4ff9`
- Final experiment snapshot: **TO BE PINNED after license, provenance and
  deidentification gates pass.**

## Lineage table

| Component | Earlier lineage | Treatment in this paper |
|---|---|---|
| Four-role Input/Interpretation/UX/Trust architecture | IUUY | Background only; not the claimed mechanism |
| ASR, diarization, feedback UI and feedback/provenance directions | IUUY and later work | Excluded from the core experiment |
| IEDID dialect/gloss resource and three-variety focus | IEDI | Development/provenance source; not test gold and not new |
| Thresholded IEDID fuzzy path plus proposed LLM fallback | IEDI, formalizing IUUY's earlier table-plus-Gemini direction | Paper-described architecture only; executable IEDI-NB is frozen separately |
| Global RapidFuzz lookup at threshold 75 and `Unknown` below threshold | Public IEDI notebook | Executable lineage diagnostic if snapshot/license gates pass |
| Persona/codebook injection plus `Tone_Category` and `Linguistic_Context` metadata | KICS-W | Paper-faithful reconstruction and schema-design source; exact unpublished prompt/codebook assumptions must be disclosed |
| Pro--Flash dynamic-model-manager/routing concept | KICS-W | Optional escalation policy; explicitly inherited and not treated as an evaluated prior router |
| Integration of four-role/trust and persona/context/DMM components | IUUY + KICS-W; integrated in unpublished CA-IEDI | Integration record and motivation only; no prior performance claim is recycled |
| `CA_IEDI_0803.ipynb` monolithic Whisper/Gemini/IEDID/feedback/Gradio/IPFS/Web3 implementation | Unpublished CA-IEDI implementation notebook | Pinned lineage artifact only. The derivative retains no code or outputs verbatim; read-only discovery and telemetry concepts are rewritten behind typed, gold-free contracts |
| Fixed-candidate provider, scorer boundary, symmetric probe runner, `ContractAttestation`, `ClarificationAnswerBroker`/`ReleasedAnswer`, separately validated post-answer branches, `BudgetEnvelope`, one-route ledger and prediction freeze | New relative to local executable lineage | `DEMO_ONLY` reference orchestration. Accepted artifacts are required for sealed use; passing demo tests supports code behavior only, not RQ1/RQ2 |
| Model-backed candidate generation and required external empirical systems | Planned successor study | `NOT_CONFIGURED`; the notebook's deterministic fixed-candidate provider is not an end-to-end model evaluation |
| Controlled cue ablation and held-out pragmatic selection | sibling empirical-context study | Related planned study; not the present gate contribution |
| Community-validated preserving and minimal sense-changing bundles | New relative to local lineage | Benchmark contribution, conditional on completed validation |
| Joint preservation and targeted-change probes before a decision | New relative to local lineage | Core verification signal |
| Calibrated `COMMIT` / one-slot `CLARIFY` / `ABSTAIN_ESCALATE` controller | New relative to local lineage as an integrated mechanism | Core engineering contribution; individual selective-prediction and clarification methods have prior art |
| Equal-budget comparison and action-aware selective evaluation | New evaluation discipline in this lineage | Required, not optional |
| Bounded consented text-prototype pilot | New planned application evaluation | Secondary feasibility evidence only; thresholds remain frozen and no superiority claim is licensed without separate power |

## External novelty boundary

Current literature already contains selective classification, conformal or
calibrated abstention, context-counterfactual benchmarks, structured
uncertainty and value-of-information clarification. Accordingly, this paper
must not claim to invent abstention, confidence gating, context flips,
structured uncertainty, clarification selection or model routing.

The defensible combination is narrower:

> a schema-constrained runtime verifier for ambiguous expressions drawn from
> three English-use community resources that jointly tests stability under community-validated
> meaning-preserving interventions and targeted responsiveness under minimal
> meaning-changing interventions, then converts those signals into a
> calibrated commit, one-slot clarification or abstention action.

Whether that combination is globally novel must be established by a dated,
documented scholarly and patent search before submission or filing.

## Current evidence boundary

No row in this file licenses an empirical performance claim. There is no
community-accepted benchmark bundle, sealed prediction file or completed
RQ1/RQ2 analysis in the package. The local notebook uses invented
`DEMO_ONLY` records and a scripted scorer. Its traces are not observations.

In the planned sealed system, `ACCEPTED` attestations certify gold-free
contract checks only. They contain no reference action, reference sense, case
type or acceptable-question label. Required context slots come from the
attestation's predeclared schema contract. A broker may release one matching
validated answer only after the system emits the corresponding question, and
the repaired pass must use separately accepted branches hashed against the
updated card. The inference process freezes predictions before an isolated
evaluator may join action and sense labels.

Equal-budget comparison means equal frozen allocations, not merely similar
observed usage. The CDCV condition and structured-context control must receive
identical call, input-token and output-token limits through `BudgetEnvelope`.
The numerical empirical token/cost limits remain `NOT_CONFIGURED` until model,
tokenizer, prompt and pricing manifests are frozen.

## Frozen IEDI protocol correction

Removing every sealed-test expression family from the frozen IEDI lexicon
makes abstention the expected behavior, not a competitive accuracy result.
Therefore:

1. report IEDI on the family-held-out regime only as a lineage/coverage
   diagnostic; and
2. if resources permit, add a separately labeled **known expression, unseen
   context** regime whose expression-to-entry mapping and snapshot are frozen
   before evaluation.

Never pool the two regimes or imply that the known-expression regime is a
genuinely novel-expression test.
