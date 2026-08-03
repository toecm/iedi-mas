# Evidence and release register

This register is the paper's factual firewall. A claim may move from
`PLANNED` to `SUPPORTED` only when the named artifact exists, passes the
declared checks and is linked from a frozen run manifest.

| Claim | Current status | Required evidence | Release condition |
|---|---|---|---|
| CDCV-Gate is fully implemented | `PLANNED` | versioned code, unit/integration tests, prompt hashes | independent clean run reproduces all actions and logs |
| Gold-free CDCV reference orchestration is executable | `PARTLY SUPPORTED` | pinned derivative notebook, source generator, controller/contracts/pipeline modules, output-free smoke execution and unit tests | supports only the implemented deterministic paths; external providers, baselines, process isolation and empirical scheduler remain explicitly unconfigured |
| Intervention bundles are community validated | `PLANNED` | deidentified ratings, creator/validator IDs, provenance and agreement report | ethics/provenance gates pass; 4/5 sense convergence and naturalness threshold met |
| The 150-family test is genuinely held out | `PLANNED` | split manifest, family hashes, training/profile/IEDID audit | audit run before label unsealing; zero prohibited overlap or documented removal |
| Verification lowers wrong-commit risk | `UNTESTED` | paired sealed predictions and family-clustered confidence interval | upper 95% CI for prespecified risk difference is below zero at matched coverage |
| Coverage is practically preserved | `UNTESTED` | risk–coverage analysis | proposed method is compared at matched coverage and any fixed-operating-point loss is within the prespecified 5-point tolerance |
| Meaning-preserving contexts are stable | `UNTESTED` | accepted preserving pairs and predictions | point estimate and family-clustered CI reported; no threshold invented after test access |
| Meaning-changing contexts elicit the validated contrast | `UNTESTED` | accepted contrastive pairs and predictions | correct targeted-flip rate and CI reported, not raw change rate alone |
| One clarification repairs rejected cases | `UNTESTED` | validated question bank, simulated/recorded answer protocol, before/after logs | recovery, final accuracy and unnecessary-question rate all reported |
| Per-community behavior is acceptable | `UNTESTED` | 50 test families/community | descriptive CIs and prespecified harm guard reported; no underpowered superiority claim |
| Runtime/cost is practical | `UNTESTED` | measured wall-clock, calls, tokens, memory and priced-model snapshot | all systems run on common hardware/API conditions and equal-budget comparator included |
| The bounded application pilot is feasible | `PLANNED` | consented text-prototype interaction logs with frozen thresholds | participant-confirmed wrong commits, repair, burden, abandonment, latency and cost are reported descriptively with participant/family clustering |
| Main tables are independently regenerable | `PLANNED` | deidentified cases, labels, probe contracts, approved clarification materials, predictions, prompts/manifests and evaluation code | consent and governance authorize the minimum bundle and a persistent repository identifier is issued; hashes alone do not satisfy this claim |
| IEDI lineage baseline is fairly characterized | `PARTLY SUPPORTED` | pinned notebook/code and data snapshot | family-held-out result labeled coverage diagnostic; known-expression regime reported separately if used |

## Result lock

- `manuscript/main.tex` must contain `\resultslockedtrue` until every primary
  test artifact is frozen.
- No number from the separate design-size simulation may enter a Results table.
- Development and calibration performance must be labeled as such and must not
  be used to describe sealed-test effectiveness.
- A failed or null result is released under the same rule as a positive result.

## Claims that remain out of scope

The present study does not establish emotion recognition, ASR robustness,
speaker identity, cultural essence, causal effects of protected identities,
global World-English coverage, real-time performance, privacy guarantees,
blockchain correctness, user trust or patentability.
