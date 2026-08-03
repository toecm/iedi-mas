# iedi-mas
Intra-English Dialect Interpretation Multi-Agents System: A hybrid of NLP+LLM and more. These are subsequent (a series of) upgrades of the initial system, "I Understand Understand You: A Reliable Multi-Agent Facilitator for Reducing Communication Breakdowns" (IUUY).

## Counterfactual context-verification study

[`CA_IEDI_0803.ipynb`](CA_IEDI_0803.ipynb) now provides an output-free,
offline rewrite for the **Counterfactual Dialect-Context Verification and
Clarification Gate (CDCV-Gate)**. The supporting reference implementation,
schemas, protocol, tests, and evidence boundaries are in
[`counterfactual_context_gate_study/`](counterfactual_context_gate_study/).
The package notebook is canonical; the root file is a validated compatibility
mirror for the historical repository layout.

The included examples are deterministic `DEMO_ONLY` code-path checks. They are
not empirical results. Community-validated data, sealed evaluation, and
model-backed candidate generation remain explicitly unconfigured.
