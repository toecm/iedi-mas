# Run artifacts

Each immutable run directory should contain:

- environment and code-commit manifest;
- model/version, prompt, controller and pricing hashes;
- data and split hashes;
- raw schema-valid prediction JSONL;
- call/token/latency/memory/cost telemetry;
- analysis command and logs; and
- generated tables/figures with SHA-256 hashes.

Do not commit provider credentials, protected benchmark text or user data.
