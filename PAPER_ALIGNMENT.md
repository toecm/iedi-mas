# Papers 2–5 implementation and claim boundary

This recode preserves the original notebooks as historical artifacts. The four new
notebooks use one tested core so a fix to routing, validation, or provenance applies
consistently across the paper profiles.

## Architectural mapping

| Paper | New notebook | Implemented architecture | Evidence still required |
|---|---|---|---|
| Paper 2, KICS 2025 | `notebooks/Paper_2_IUUY.ipynb` | IEDID schema with `concept_id`, dialect, gloss, intent and sociolinguistic tags; top-two retrieval at ≥0.80; explicit unmatched/near-tie/polysemy route; structured LLM adapter; timestamped speaker segments; tamper-evident preference export; entry and dialect metrics; evaluation-manifest validation | A genuinely dialect-adapted checkpoint, an executed RLHF/PPO trace if that term is retained, pyannote or equivalent diarization, 150 disjoint clips, and ten unique raw evaluator ratings per case |
| Paper 3, KICS Winter 2026 | `notebooks/Paper_3_IEDI_MAS.ipynb` | Full schema-approved persona loaded before interpretation; persona rules affect ranking; centralized ambiguity-aware Pro/Flash DMM; request-local route telemetry; three-option enforcement only when three reviewed senses exist; append-only versioning | Native-speaker validation, a curated third `I beg` sense, held-out persona/static comparison, and current-model results. Gemini 1.5 endpoints are historical, not runnable defaults |
| Paper 4, JCCI 2026 | `notebooks/Paper_4_JCCI.ipynb` | Versioned edge semantic boundary that preserves routing fields but rejects raw audio; deterministic UTF-8 JSON; actual byte length; transfer estimate separated from observed gateway round trip; route/model-specific timing and distribution utilities | Separate edge/cloud deployment, measured audio files, applied/logged cellular hardware or emulator, monolithic baseline, repeated raw trials, and confidence intervals for published target values |
| Paper 5, CA-IEDI 2026 | `notebooks/Paper_5_CA_IEDI.ipynb` | Four named agents on bounded in-process queues; local/Flash/Pro DMM; process-local model-eligible cold-start plus ambiguity rule; schema-approved personas; validated acoustic-evidence contract; authenticated-authorizer interface; resumable IPFS/chain/HF trust state machine; live atomic reindex; canonical linear quorum contract; SAER and semantic-resolution metrics | Authenticated community validators, deployed contract, IPFS/HF durability evidence, calibrated affect model, Indonesian-English community curation, Gradio/two-party UX, distributed transport, and empirical >92%/latency/bandwidth outcomes |

## What makes this DMM real

All model calls pass through `DynamicModelManager.interpret`. UI or interpretation
code cannot directly select a `generate_fast` or `generate_smart` method.

1. A unique approved codebook result with a sufficient score and margin stays local.
2. Ambiguity is calculated from retrieval confidence, top-two margin, plausible
   senses, known polysemy, persona conflict, missing context and ASR confidence.
3. Routine low-ambiguity unresolved input uses Flash.
4. High ambiguity, an explicit validated risk score, initial profile setup, or configured persona cold-start uses
   Pro when its circuit and latency budget permit.
5. Retryable Pro quota/timeout/service failures fall back once to Flash and record the
   requested route, used route and reason.
6. Authentication and schema errors fail visibly. They do not silently switch lanes.
7. A weak Flash result can escalate to Pro only when the remaining latency and total
   cost budgets permit it; both calls remain in telemetry.
8. Low-confidence Pro output is retained for inspection but marked for human review.

Paper 5's prose says ambiguity should select Pro, while its Algorithm 1 ignores the
`ambiguity_level` argument and routes the first 49 requests to Pro. The implemented
policy combines the two: deterministic local first, then Pro for high ambiguity **or**
a configurable first-50 model-eligible, per-persona, process-local cold-start; high
ambiguity continues to select Pro after request 50. Persisting that counter across
deployments is an integration requirement, not silently implied here.

## Non-negotiable terminology

- Appending a CSV or preference row is HITL data collection, not RLHF.
- A fine-tuned/RLHF claim requires a `ModelManifest` plus matching SHA-256-checked
  checkpoint, training log, preference dataset, evaluation artifact, and deployment
  receipt. A method-name string alone is rejected as evidence.
- Stored audio is not multimodal affect. The runtime requires a label, confidence,
  extractor ID/version, and finite feature map before treating acoustic evidence as
  present; cultural validity and calibration still require external evaluation.
- The asynchronous runtime is an in-process actor system. It is not a distributed MAS
  until transport and independently deployed services are demonstrated.
- IPFS is content-addressed provenance, not correctness.
- The trust gate is operational only when content retrieval verifies the committed
  hash and `PureChainRegistry` reaches validator quorum/finality. A self-transaction
  with data is not equivalent.
- Demonstration personas are schema fixtures, not proof of cultural correctness.
- Local JSONL chains are tamper-evident relative to their retained head, not globally
  immutable. Durable signed/external checkpoints remain a deployment responsibility.

## Quantitative claim corrections

The papers define semantic ambiguity error as mean cosine distance:

```text
SAER = mean(1 - cosine_similarity(hypothesis, reference))
```

Lower SAER is therefore better. The code reports both:

```text
SAER = distance
semantic resolution rho = 1 - SAER
```

The claimed “>92%” should be tested as `rho > 0.92`, equivalently `SAER < 0.08`, on
a frozen native-speaker-labelled test set. It is not a unit-test constant.

Likewise, 96 KiB → 45 B exceeds 99.9% reduction, while 96 KiB → 120 B is about
99.878%. Transfer time (`bits / uplink rate`) is reported separately from observed
network RTT; neither is labelled “4G latency” without a measured network experiment.

## Running live integrations

The default test suite performs no paid/network calls. Live Gemini, Whisper,
diarization, Pinata, Hugging Face and Web3 integrations are opt-in. Secrets must be
provided through environment variables or a managed secret store. Do not save them
in notebook cells or notebook outputs.

For a real Paper 4 boundary, run the cloud service separately and point the benchmark
client at it:

```powershell
uvicorn scripts.cloud_server:create_app --factory --host 0.0.0.0 --port 8000
python scripts/benchmark_edge_cloud.py --endpoint http://CLOUD_HOST:8000/v1/interpret `
  --utterance "An unresolved phrase" --runs 30 --raw-audio-file .\fixtures\clip.wav `
  --uplink-kbps 10000 --downlink-kbps 20000 --base-rtt-ms 45 `
  --network-name documented-profile --emulator "tc/netem configuration name" `
  --network-evidence-file .\benchmark-results\netem-config.txt `
  --output benchmark-results/paper4.jsonl
```

The client and service should run on independently documented hosts before reporting
the gateway round trip as a network measurement. The benchmark hashes supplied
audio/network evidence but does not apply a network emulator itself; its profile is
metadata and its transfer time remains a calculation.
