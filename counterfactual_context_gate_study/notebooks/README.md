# CA_IEDI_0803 CDCV-Gate rewrite

`CA_IEDI_0803.ipynb` is the protocol-aligned derivative of the public
[`CA_IEDI_0803.ipynb`](https://github.com/toecm/iedi-mas/blob/5cff1e509efb09c24f9ac7e30075b6a131ee6fbc/CA_IEDI_0803.ipynb).
The source revision, blob and content hashes are frozen in
`UPSTREAM_PROVENANCE.json`. The mutable `main` URL is recorded only for human
navigation and is never downloaded or executed by the derivative notebook.

## What is retained

- IEDID and personas as read-only lineage, expression-discovery and development
  resources only;
- the idea of candidate interpretation and resource telemetry;
- the IUUY/IEDI/CA-IEDI four-role lineage as historical motivation.

## What is replaced

- prose-option retrieval becomes fixed two-sense plus `OTHER_UNLISTED` scoring;
- global persona injection becomes an episode-scoped structured context card;
- model output acceptance becomes symmetric, community-reviewed
  hypothesis-preserving and hypothesis-contrast probes;
- always-return-three behavior becomes calibrated `COMMIT`, one-slot `CLARIFY`,
  or `ABSTAIN_ESCALATE`;
- unconditional cloud use becomes a benefit-, privacy- and budget-constrained
  one-time route after unresolved verification;
- approximate payload/latency text becomes typed per-stage call, token, latency,
  memory and cost telemetry;
- live data mutation becomes a gold-free immutable prediction record, frozen
  before evaluator labels may be joined.

## Intentionally removed

The derivative contains no model-provider key, package installation cell,
Whisper/audio path, Hugging Face write, Gradio public tunnel, feedback learning,
Pinata/IPFS, Web3/blockchain, Hardhat, model-list diagnostic, or upstream saved
output. The primary study is text-only; audio belongs only in the separately
prespecified non-confirmatory application pilot.

## Evidence boundary

The notebook defaults to `RUN_MODE = "DEMO"`, uses invented neutral fixtures and
a deterministic scripted adapter, and refuses sealed-test episodes. Its outputs
are code smoke tests, not empirical findings. Real sealed inference, answer
brokering and evaluation must execute in separate access-controlled processes;
one notebook kernel is not a security boundary. The external scheduler must
supply frozen trusted-attestation and trusted-answer hash manifests; status
strings created inside Python are not authority.

Regenerate the notebook after editing its source cells:

```powershell
python scripts/build_cdcv_notebook.py
```

The package notebook is canonical. When the package is checked out beside the
repository-root `CA_IEDI_0803.ipynb`, the builder updates that compatibility
mirror too, and the package validator rejects any drift between the two files.

Execute every code cell without Jupyter dependencies:

```powershell
python scripts/execute_notebook.py notebooks/CA_IEDI_0803.ipynb
```
