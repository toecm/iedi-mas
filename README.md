# iedi-mas

Intra-English Dialect Interpretation Multi-Agent System: a hybrid of deterministic
case retrieval, persona-aware language-model interpretation, human validation, and
verifiable provenance. This repository retains the historical IUUY/IEDI notebooks
and adds a testable reference implementation aligned to Papers 2–5.

## Paper-aligned implementation

The legacy notebooks remain unchanged. New work lives in:

- `src/iedi/`: shared schemas, codebook retrieval, actual Dynamic Model Manager,
  edge/audio adapters, four-agent actor runtime, trust gate, and evaluation metrics;
- `data/codebook.demo.json`: schema-approved demonstration records (not community validation), plus an explicit pending Indonesian-English curation placeholder;
- `configs/`: executable routing policy plus explicit evidence requirements for claims;
- `notebooks/`: four thin, output-free paper entry points;
- `contracts/PureChainRegistry.sol`: validator-quorum commitment registry;
- `tests/`: offline acceptance tests using fake providers and provenance adapters.

Read [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) before describing a notebook as a
paper reproduction. It distinguishes implemented architecture from integrations and
empirical claims that still require credentials, deployments, human-labelled data,
or training artifacts.

The generated notebooks default to `OfflineFixtureProvider`, which only echoes
reviewed codebook evidence and labels telemetry `offline-fixture::*`. Set
`IEDI_LIVE_GEMINI=1` plus `GEMINI_API_KEY` for live calls; fixture output must never
be reported as model performance.

## Quick verification

```powershell
python -m pytest -q
```

For live Gemini calls, install the optional provider and set a secret only in the
environment—not in notebook cells or outputs:

```powershell
python -m pip install -e ".[gemini]"
$env:GEMINI_API_KEY = "..."
```

The runnable DMM currently defaults to `gemini-2.5-flash` and `gemini-2.5-pro`, and
model IDs remain configurable. Google lists both 2.5 models for shutdown on
2026-10-16, so a deployment must migrate before then. Historical 1.5/2.0 identifiers
are retained only as paper metadata because a model-version substitution reproduces
the architecture, not the original endpoint. See the official
[Gemini deprecation schedule](https://ai.google.dev/gemini-api/docs/deprecations).
