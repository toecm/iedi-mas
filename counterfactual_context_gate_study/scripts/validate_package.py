"""Dependency-free integrity checks for the CDCV-Gate paper package."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
import re
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = (
    "README.md",
    "SYSTEM_SPEC.md",
    "STUDY_PROTOCOL.md",
    "CLAIM_LINEAGE.md",
    "EVIDENCE_REGISTER.md",
    "config/study_design.json",
    "data/schemas/context_card.schema.json",
    "data/schemas/intervention_bundle.schema.json",
    "data/schemas/runtime_episode.schema.json",
    "data/schemas/sealed_label.schema.json",
    "data/schemas/clarification_question.schema.json",
    "data/schemas/clarification_answer.schema.json",
    "data/schemas/released_answer.schema.json",
    "data/schemas/contract_attestation.schema.json",
    "data/schemas/clarification_scenario_manifest.schema.json",
    "data/schemas/prediction_record.schema.json",
    "notebooks/CA_IEDI_0803.ipynb",
    "notebooks/UPSTREAM_PROVENANCE.json",
    "notebooks/README.md",
    "scripts/build_cdcv_notebook.py",
    "scripts/execute_notebook.py",
    "manuscript/main.tex",
    "manuscript/references.bib",
)

PERMITTED_INTERVENTION_SLOTS = {
    "relationship_role",
    "setting",
    "formality",
    "discourse_goal",
    "preceding_speech_act",
    "situation",
}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256(payload.encode("utf-8")).hexdigest()

# A source-controlled research notebook must remain an offline, read-only smoke
# test. These modules either open network/process surfaces or are model/provider
# SDKs whose presence would make an apparently offline validation ambiguous.
PROHIBITED_NOTEBOOK_IMPORT_PREFIXES = (
    "aiohttp",
    "anthropic",
    "azure",
    "boto3",
    "botocore",
    "cohere",
    "datasets",
    "ftplib",
    "fastapi",
    "firebase_admin",
    "flask",
    "google.generativeai",
    "google.genai",
    "gradio",
    "http.client",
    "httpx",
    "huggingface_hub",
    "ipfshttpclient",
    "mistralai",
    "openai",
    "os",
    "paramiko",
    "requests",
    "replicate",
    "shutil",
    "smtplib",
    "socket",
    "subprocess",
    "tempfile",
    "transformers",
    "urllib",
    "uvicorn",
    "vertexai",
    "wandb",
    "web3",
    "webbrowser",
    "whisper",
)

# Attribute calls with these terminal names create external state even when the
# receiver is aliased (for example, ``Path(...).write_text``). Read-only
# ``Path.read_text``/``read_bytes`` remain permitted.
PROHIBITED_NOTEBOOK_CALL_NAMES = {
    "__import__",
    "chmod",
    "chown",
    "compile",
    "copy",
    "copy2",
    "copyfile",
    "copytree",
    "dump",
    "eval",
    "exec",
    "get_ipython",
    "hardlink_to",
    "launch",
    "makedirs",
    "mkdir",
    "move",
    "open",
    "popen",
    "import_module",
    "input",
    "push_to_hub",
    "remove",
    "removedirs",
    "rename",
    "replace",
    "rmdir",
    "save",
    "savefig",
    "save_pretrained",
    "symlink_to",
    "system",
    "system_raw",
    "run_line_magic",
    "run_cell_magic",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_json",
    "to_parquet",
    "to_pickle",
    "touch",
    "truncate",
    "unlink",
    "upload_file",
    "urlopen",
    "urlretrieve",
    "write",
    "write_bytes",
    "write_text",
    "writelines",
}


def load_json(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_expected_files(errors: list[str]) -> None:
    for relative in EXPECTED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")


def check_design(errors: list[str]) -> None:
    design = load_json("config/study_design.json")
    implementation = design["reference_implementation"]
    if implementation["default_run_mode"] != "DEMO":
        errors.append("reference notebook must default to DEMO mode")
    if implementation["sealed_test_allowed_in_demo_mode"] is not False:
        errors.append("DEMO mode must never accept sealed-test episodes")
    if implementation["saved_notebook_outputs_allowed"] is not False:
        errors.append("saved notebook outputs must remain prohibited")
    if implementation["real_sealed_inference_requires_separate_process"] is not True:
        errors.append("sealed inference must remain process-separated from the notebook")
    required_forbidden_inputs = {
        "reference_action",
        "reference_sense_id",
        "probe_type",
        "target_candidate_id",
        "validator_scores",
    }
    if not required_forbidden_inputs.issubset(implementation["scorer_forbidden_inputs"]):
        errors.append("scorer interface is missing required gold/probe metadata bans")
    splits = design["splits"]
    communities = design["communities"]
    for split_name, split in splits.items():
        expected = split["families_per_community"] * len(communities)
        if split["families"] != expected:
            errors.append(f"{split_name}: families != communities × per-community")
    case_total = sum(
        value
        for key, value in design["cases_per_retained_family"].items()
        if key != "total"
    )
    if case_total != design["cases_per_retained_family"]["total"]:
        errors.append("cases-per-family components do not sum to total")
    expected_test_cases = (
        splits["sealed_test"]["families"]
        * design["cases_per_retained_family"]["total"]
    )
    if design["sealed_test_cases"] != expected_test_cases:
        errors.append("sealed_test_cases is inconsistent with the design")
    if design["iedid"]["sealed_test_gold_allowed"] is not False:
        errors.append("IEDID must remain prohibited as sealed-test gold")
    if design["result_policy"]["synthetic_results_as_empirical_evidence"] is not False:
        errors.append("synthetic results must remain prohibited as evidence")
    if design["controller"]["maximum_clarification_questions"] != 1:
        errors.append("confirmatory protocol permits exactly one clarification")
    if design["controller"]["minimum_preservation_invariance"] != 1.0:
        errors.append("the one-probe preservation threshold must remain 1.0")
    if design["controller"]["minimum_targeted_response"] != 1.0:
        errors.append("the one-probe targeted-response threshold must remain 1.0")
    if design["controller"]["probe_thresholds_frozen_on"] != "protocol_before_data":
        errors.append("binary one-probe behavior thresholds must be fixed before data")
    if not design["intervention_policy"]["protected_identity_interventions_prohibited"]:
        errors.append("protected-identity interventions must be prohibited")
    if design["intervention_policy"]["sealed_reference_labels_visible_to_inference"]:
        errors.append("sealed reference labels must never be visible to inference")
    if design["intervention_policy"]["branch_selection_source"] != "model_base_prediction_only":
        errors.append("runtime probe branch must be selected from the model prediction")
    if design["access_separation"]["inference_process_may_mount_sealed_labels"]:
        errors.append("inference process must not be able to mount sealed labels")
    if not design["access_separation"]["join_labels_only_after_prediction_freeze"]:
        errors.append("labels must be joined only after predictions are frozen")
    runtime_schema = load_json("data/schemas/runtime_episode.schema.json")
    runtime_properties = set(runtime_schema.get("properties", {}))
    forbidden_runtime_fields = {
        "reference_action",
        "reference_sense_id",
        "acceptable_clarification_slots",
        "case_type",
    }
    leaked = sorted(runtime_properties.intersection(forbidden_runtime_fields))
    if leaked:
        errors.append("gold fields leaked into runtime schema: " + ", ".join(leaked))
    probe_budget = design["runtime_probe_budget"]
    expected_calls = (
        1
        + probe_budget["preserving_probes_per_selected_candidate_branch"]
        + probe_budget["meaning_changing_probes_per_selected_candidate_branch"]
    )
    if probe_budget["scorer_calls_per_verification_pass_including_base"] != expected_calls:
        errors.append("per-pass scorer-call budget does not match probe cardinality")
    if probe_budget["maximum_scorer_calls_excluding_candidate_generation"] != (
        expected_calls * probe_budget["maximum_passes_with_one_clarification_and_one_route"]
    ):
        errors.append("maximum scorer-call budget does not match pass cap")
    if design["candidate_family_target_at_70pct_retention"] != 345:
        errors.append("balanced 70-percent recruitment target must be 345")
    if not design["primary_task"]["hard_eligibility_mask_required"]:
        errors.append("matched coverage must preserve the hard eligibility mask")
    primary = design["primary_task"]
    if primary["maximum_action_appropriate_coverage"] != 0.75:
        errors.append("all-case action-appropriate coverage cap must be 0.75")
    if primary["matched_coverage_target"] is not None and not (
        0
        < primary["matched_coverage_target"]
        <= primary["maximum_action_appropriate_coverage"]
    ):
        errors.append("matched coverage target must be in (0, 0.75]")
    if design["clarification_scenarios"]["primary_additional_model_calls"] != 0:
        errors.append("primary clarification selection must use zero extra model calls")
    workload = design["annotation_workload"]
    retained = sum(split["families"] for split in splits.values())
    expected_base = retained * design["cases_per_retained_family"]["total"]
    if workload["base_cases_total"] != expected_base:
        errors.append("annotation workload base-case count is inconsistent")
    expected_probes = (
        expected_base
        * workload["candidate_branches_per_case"]
        * workload["probes_per_branch"]
    )
    if workload["probe_instances_total"] != expected_probes:
        errors.append("annotation workload probe-instance count is inconsistent")
    if workload["probe_ratings_total"] != (
        expected_probes * workload["validators_per_probe"]
    ):
        errors.append("annotation workload probe-rating count is inconsistent")
    expected_questions = retained * workload["candidate_questions_per_family"]
    if workload["candidate_question_instances_total"] != expected_questions:
        errors.append("annotation workload question count is inconsistent")
    if workload["question_ratings_total"] != (
        expected_questions * workload["validators_per_question"]
    ):
        errors.append("annotation workload question-rating count is inconsistent")
    expected_minimum_ratings = (
        workload["base_case_ratings_total"]
        + workload["probe_ratings_total"]
        + workload["question_ratings_total"]
    )
    if workload["minimum_structured_ratings_total"] != expected_minimum_ratings:
        errors.append("annotation workload minimum rating total is inconsistent")
    pilot = design["application_pilot"]
    if pilot["confirmatory_inference_allowed"] is not False:
        errors.append("the bounded application pilot cannot support confirmatory inference")
    if pilot["reuse_confirmatory_test_for_tuning"] is not False:
        errors.append("the application pilot cannot tune on the confirmatory test")
    release = design["minimum_public_release"]
    if release["release_before_ethics_license_consent_and_community_review"] is not False:
        errors.append("public release must remain behind governance gates")
    if release["raw_identity_bearing_text_public_by_default"] is not False:
        errors.append("identity-bearing text cannot be public by default")
    if not release["required_artifacts"]:
        errors.append("minimum public release must name required artifacts")


def validate_intervention_bundle_integrity(
    runtime: dict,
    bundle: dict,
    source_context: dict,
    resulting_contexts: dict[str, dict],
) -> list[str]:
    """Validate cross-record symmetry and context-patch integrity.

    This function is intended for the sealed scheduler before a bundle becomes
    mountable by inference. JSON Schema alone cannot express these joins.
    """

    errors: list[str] = []
    if bundle.get("family_id") != runtime.get("family_id"):
        errors.append("bundle family_id does not match runtime episode")
    if bundle.get("base_case_id") != runtime.get("case_id"):
        errors.append("bundle base_case_id does not match runtime episode")
    if bundle.get("utterance_hash") != runtime.get("utterance_hash"):
        errors.append("bundle utterance hash does not match runtime episode")
    if bundle.get("candidate_set_hash") != runtime.get("candidate_set_hash"):
        errors.append("bundle candidate-set hash does not match runtime episode")
    if bundle.get("validation", {}).get("status") != "accepted":
        errors.append("only accepted intervention bundles may reach inference")
    if bundle.get("variety_cue_fixed") is not True:
        errors.append("bundle does not lock the variety cue")
    if bundle.get("constructed_without_sealed_reference_action") is not True:
        errors.append("bundle lacks the no-sealed-label construction lock")

    candidate_ids = [
        item.get("candidate_id") for item in runtime.get("candidate_senses", [])
    ]
    core_ids = [value for value in candidate_ids if value != "OTHER_UNLISTED"]
    if len(core_ids) != 2 or len(set(core_ids)) != 2:
        errors.append("runtime episode must contain exactly two distinct core candidates")
        return errors

    branches = bundle.get("candidate_branches", [])
    sources = [branch.get("source_candidate_id") for branch in branches]
    if len(branches) != 2 or set(sources) != set(core_ids) or len(set(sources)) != 2:
        errors.append("candidate branches are not exactly symmetric across core candidates")

    source_hash = runtime.get("context_card_hash")
    if source_hash != canonical_sha256(source_context):
        errors.append("runtime context-card hash is not canonical")
    source_fields = source_context.get("fields", {})
    source_variety = source_context.get("variety_cue")
    for branch in branches:
        source_candidate = branch.get("source_candidate_id")
        preserving = branch.get("preserving", [])
        changing = branch.get("meaning_changing", [])
        if len(preserving) != 1 or len(changing) != 1:
            errors.append(f"branch {source_candidate!r} must contain one probe of each type")
            continue
        expected_target = next((value for value in core_ids if value != source_candidate), None)
        if changing[0].get("target_candidate_id") != expected_target:
            errors.append(f"branch {source_candidate!r} does not target the other core candidate")

        for probe in preserving + changing:
            if probe.get("source_context_hash") != source_hash:
                errors.append(f"probe {probe.get('intervention_id')!r} source hash mismatch")
            declared = set(probe.get("changed_slots", []))
            if not declared or not declared.issubset(PERMITTED_INTERVENTION_SLOTS):
                errors.append(f"probe {probe.get('intervention_id')!r} changes a prohibited slot")
            result_hash = probe.get("result_context_hash")
            result = resulting_contexts.get(result_hash)
            if result is None:
                errors.append(f"probe {probe.get('intervention_id')!r} result context is unavailable")
                continue
            if result_hash != canonical_sha256(result):
                errors.append(
                    f"probe {probe.get('intervention_id')!r} result hash is not canonical"
                )
            if result.get("variety_cue") != source_variety:
                errors.append(f"probe {probe.get('intervention_id')!r} changes the fixed variety cue")
            result_fields = result.get("fields", {})
            actual = {
                slot
                for slot in set(source_fields).union(result_fields)
                if source_fields.get(slot) != result_fields.get(slot)
            }
            if actual != declared:
                errors.append(f"probe {probe.get('intervention_id')!r} declared slots do not match its patch")
    return errors


def validate_clarification_scenario_manifest(
    question: dict, manifest: dict, available_probe_ids: set[str]
) -> list[str]:
    """Validate the frozen, zero-extra-call question-utility inputs."""

    errors: list[str] = []
    for key in ("question_id", "family_id", "context_slot", "candidate_set_hash"):
        if question.get(key) != manifest.get(key):
            errors.append(f"clarification scenario {key} does not match its question")
    answer_ids = [item.get("answer_id") for item in question.get("answer_domain", [])]
    scenarios = manifest.get("scenarios", [])
    scenario_ids = [item.get("answer_id") for item in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)) or set(scenario_ids) != set(answer_ids):
        errors.append("clarification scenarios do not map one-to-one to answer IDs")
    probability_sum = sum(float(item.get("prior_probability", 0.0)) for item in scenarios)
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("clarification scenario priors must sum to one")
    if manifest.get("mode") == "PRIMARY_REUSE_ONLY":
        if manifest.get("additional_model_calls") != 0:
            errors.append("primary clarification scenarios must add zero model calls")
        for item in scenarios:
            source = item.get("score_source", {})
            if source.get("source_type") != "REUSED_PROBE_SCORES":
                errors.append("primary clarification scenario does not reuse probe scores")
            if source.get("probe_id") not in available_probe_ids:
                errors.append("clarification scenario references an unavailable probe")
    return errors


def check_schemas(errors: list[str]) -> None:
    for path in sorted((ROOT / "data" / "schemas").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append(f"{path.name}: unexpected JSON Schema dialect")
        if data.get("additionalProperties") is not False:
            errors.append(f"{path.name}: top-level additionalProperties must be false")
        try:
            import jsonschema  # type: ignore
        except ImportError:
            jsonschema = None
        if jsonschema is not None:
            try:
                jsonschema.Draft202012Validator.check_schema(data)
            except jsonschema.exceptions.SchemaError as exc:
                errors.append(f"{path.name}: invalid JSON Schema: {exc.message}")


def strip_latex_comments(text: str) -> str:
    """Remove unescaped percent comments while preserving line structure."""
    clean_lines = []
    for line in text.splitlines():
        cut = len(line)
        for index, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cut = index
                break
        clean_lines.append(line[:cut])
    return "\n".join(clean_lines)


def check_latex_structure(text: str, errors: list[str]) -> None:
    clean = strip_latex_comments(text)
    brace_depth = 0
    for index, char in enumerate(clean):
        if char not in "{}":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and clean[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 1:
            continue
        brace_depth += 1 if char == "{" else -1
        if brace_depth < 0:
            errors.append("manuscript has a closing brace without an opener")
            break
    if brace_depth != 0:
        errors.append(f"manuscript brace balance is {brace_depth}, expected 0")

    env_tokens = re.findall(r"\\(begin|end)\{([^}]+)\}", clean)
    stack: list[str] = []
    for kind, environment in env_tokens:
        if kind == "begin":
            stack.append(environment)
        elif not stack or stack[-1] != environment:
            errors.append(f"manuscript environment mismatch at \\end{{{environment}}}")
            break
        else:
            stack.pop()
    if stack:
        errors.append("manuscript has unclosed environments: " + ", ".join(stack))


def check_manuscript(errors: list[str]) -> None:
    path = ROOT / "manuscript" / "main.tex"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    required = (
        r"\resultslockedtrue",
        "RQ1---Selective reliability",
        "RQ2---Robustness and repair",
        "RESULTS_LOCKED",
        "COMMIT",
        "CLARIFY",
        "ABSTAIN",
        "equal-budget",
    )
    for marker in required:
        if marker not in text:
            errors.append(f"manuscript missing required marker: {marker}")
    if r"\resultslockedfalse" in text:
        errors.append("manuscript result lock has been disabled")
    check_latex_structure(text, errors)

    prohibited_unverified_claims = (
        ">92",
        "1050-ms",
        "45-byte",
        "poisoning prevention",
        "state-of-the-art",
    )
    lowered = text.lower()
    for phrase in prohibited_unverified_claims:
        if phrase.lower() in lowered:
            errors.append(f"manuscript contains prohibited unverified claim: {phrase}")


def check_bibliography(errors: list[str]) -> None:
    path = ROOT / "manuscript" / "references.bib"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    keys = re.findall(r"^@[A-Za-z]+\{([^,]+),", text, flags=re.MULTILINE)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        errors.append("duplicate bibliography keys: " + ", ".join(duplicates))
    manuscript_path = ROOT / "manuscript" / "main.tex"
    if manuscript_path.exists():
        manuscript = strip_latex_comments(
            manuscript_path.read_text(encoding="utf-8")
        )
        cited: set[str] = set()
        for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", manuscript):
            cited.update(key.strip() for key in group.split(",") if key.strip())
        missing = sorted(cited.difference(keys))
        if missing:
            errors.append("citation keys missing from bibliography: " + ", ".join(missing))


def _parse_utc_timestamp(value: object, field: str, errors: list[str]) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp and report malformed provenance."""

    if not isinstance(value, str) or not value.strip():
        errors.append(f"provenance {field} must be a non-empty ISO-8601 timestamp")
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        errors.append(f"provenance {field} is not a valid ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        errors.append(f"provenance {field} must explicitly use UTC")
        return None
    return parsed


def check_provenance_chronology(provenance: dict, errors: list[str]) -> None:
    """Require an auditable retrieval timestamp at or after the source commit."""

    committed = _parse_utc_timestamp(
        provenance.get("source_commit_timestamp_utc"),
        "source_commit_timestamp_utc",
        errors,
    )
    retrieved = _parse_utc_timestamp(
        provenance.get("retrieved_for_audit_utc"),
        "retrieved_for_audit_utc",
        errors,
    )
    if committed is not None and retrieved is not None and retrieved < committed:
        errors.append("provenance retrieval timestamp predates the source commit")


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _is_prohibited_import(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in PROHIBITED_NOTEBOOK_IMPORT_PREFIXES
    )


def check_notebook_code_cells(cells: list[dict], errors: list[str]) -> None:
    """Compile and statically enforce an offline, read-only notebook surface."""

    require_definition_raises = False
    run_mode_lock = False
    results_lock = False

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"notebook cell {index} must be a JSON object")
            continue
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"notebook code cell {index} retains an execution count")
        if cell.get("outputs") != []:
            errors.append(f"notebook code cell {index} must have an empty outputs list")
        raw_source = cell.get("source", [])
        if isinstance(raw_source, str):
            source = raw_source
        elif isinstance(raw_source, list) and all(
            isinstance(line, str) for line in raw_source
        ):
            source = "".join(raw_source)
        else:
            errors.append(f"notebook code cell {index} has an invalid source field")
            continue
        try:
            tree = ast.parse(source, filename=f"CA_IEDI_0803.ipynb:cell-{index}")
            compile(tree, f"CA_IEDI_0803.ipynb:cell-{index}", "exec")
        except (SyntaxError, TypeError, ValueError) as exc:
            message = getattr(exc, "msg", str(exc))
            errors.append(f"notebook code cell {index} does not compile: {message}")
            continue

        for statement in tree.body:
            if isinstance(statement, ast.FunctionDef) and statement.name == "require":
                require_definition_raises = any(
                    isinstance(descendant, ast.Raise)
                    for descendant in ast.walk(statement)
                )
            if not isinstance(statement, ast.Expr) or not isinstance(
                statement.value, ast.Call
            ):
                continue
            called = _dotted_name(statement.value.func)
            if called != "require" or not statement.value.args:
                continue
            referenced_names = {
                child.id
                for child in ast.walk(statement.value.args[0])
                if isinstance(child, ast.Name)
            }
            run_mode_lock = run_mode_lock or "RUN_MODE" in referenced_names
            results_lock = results_lock or "RESULTS_LOCKED" in referenced_names

        assert_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assert_found = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_prohibited_import(alias.name):
                        errors.append(
                            f"notebook code cell {index} imports prohibited module: {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_modules = [module] + [
                    f"{module}.{alias.name}" if module else alias.name
                    for alias in node.names
                ]
                prohibited = next(
                    (value for value in imported_modules if _is_prohibited_import(value)),
                    None,
                )
                if prohibited is not None:
                    errors.append(
                        f"notebook code cell {index} imports prohibited module: {prohibited}"
                    )
            elif isinstance(node, ast.Call):
                called = _dotted_name(node.func)
                terminal = called.rsplit(".", 1)[-1] if called else ""
                if terminal in PROHIBITED_NOTEBOOK_CALL_NAMES:
                    errors.append(
                        f"notebook code cell {index} calls prohibited primitive: {called or terminal}"
                    )
        if assert_found:
            errors.append(
                f"notebook code cell {index} uses assert; protocol locks must raise explicitly"
            )

    if not require_definition_raises:
        errors.append("notebook must define require(condition, message) with an explicit raise")
    if not run_mode_lock:
        errors.append("notebook must explicitly enforce its RUN_MODE lock with require()")
    if not results_lock:
        errors.append("notebook must explicitly enforce its RESULTS_LOCKED lock with require()")


def _load_builder_notebook(errors: list[str]) -> dict | None:
    """Load the builder's NOTEBOOK without invoking its writer or writing bytecode."""

    builder_path = ROOT / "scripts" / "build_cdcv_notebook.py"
    try:
        source = builder_path.read_text(encoding="utf-8")
        compiled = compile(source, str(builder_path), "exec")
        namespace = {
            "__file__": str(builder_path),
            "__name__": "_cdcv_notebook_builder_for_validation",
        }
        exec(compiled, namespace, namespace)
    except Exception as exc:  # surfaced as a validation error, not a traceback
        errors.append(f"notebook builder cannot be loaded read-only: {exc}")
        return None
    expected = namespace.get("NOTEBOOK")
    if not isinstance(expected, dict):
        errors.append("notebook builder does not expose an in-memory NOTEBOOK dictionary")
        return None
    return expected


def check_notebook_matches_builder(
    notebook: dict, expected_notebook: dict | None, errors: list[str]
) -> None:
    if expected_notebook is not None and notebook != expected_notebook:
        errors.append(
            "checked-in notebook differs from scripts/build_cdcv_notebook.py NOTEBOOK; "
            "regenerate it before validation"
        )


def check_repository_notebook_mirror(notebook: dict, errors: list[str]) -> None:
    mirror_path = ROOT.parent / "CA_IEDI_0803.ipynb"
    if not mirror_path.is_file():
        return
    try:
        mirror = json.loads(mirror_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"root notebook compatibility mirror is invalid JSON: {exc}")
        return
    if mirror != notebook:
        errors.append(
            "root CA_IEDI_0803.ipynb differs from the canonical package notebook"
        )


def check_notebook(errors: list[str]) -> None:
    path = ROOT / "notebooks" / "CA_IEDI_0803.ipynb"
    provenance_path = ROOT / "notebooks" / "UPSTREAM_PROVENANCE.json"
    if not path.is_file() or not provenance_path.is_file():
        return
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"notebook/provenance JSON is invalid: {exc}")
        return
    if not isinstance(notebook, dict) or not isinstance(provenance, dict):
        errors.append("notebook and provenance must each be a JSON object")
        return
    if notebook.get("nbformat") != 4:
        errors.append("rewritten notebook must use nbformat 4")
    metadata = notebook.get("metadata", {}).get("cdcv_gate", {})
    expected_commit = "5cff1e509efb09c24f9ac7e30075b6a131ee6fbc"
    expected_blob = "5b83ce8dbdc0e147637ef499b7c4f7deabfbb653"
    if provenance.get("source_commit") != expected_commit:
        errors.append("upstream notebook commit is not pinned to the audited revision")
    if provenance.get("source_git_blob_sha1") != expected_blob:
        errors.append("upstream notebook blob is not pinned to the audited revision")
    check_provenance_chronology(provenance, errors)
    if metadata.get("upstream_commit") != expected_commit:
        errors.append("notebook metadata does not match the pinned upstream commit")
    if metadata.get("upstream_blob_sha") != expected_blob:
        errors.append("notebook metadata does not match the pinned upstream blob")
    if metadata.get("evidence_status") != "DEMONSTRATION_ONLY_RESULTS_LOCKED":
        errors.append("notebook evidence status is not locked")
    if metadata.get("run_mode_default") != "DEMO":
        errors.append("rewritten notebook must default to DEMO mode")
    if metadata.get("source_outputs_copied") is not False:
        errors.append("upstream saved outputs must not be copied")

    cells = notebook.get("cells", [])
    if not isinstance(cells, list) or not cells:
        errors.append("rewritten notebook has no cells")
        return
    if not all(isinstance(cell, dict) for cell in cells):
        errors.append("rewritten notebook contains a non-object cell")
        return
    combined = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in cells
    )
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    )
    required_markers = (
        "RESULTS_LOCKED",
        'RUN_MODE = "DEMO"',
        expected_commit,
        expected_blob,
        "CDCVRunner",
        "ReleasedAnswer",
        "ClarificationAnswerBroker",
        "StaticDemoAnswerBroker",
        "run_equal_budget_structured_context",
        "freeze_prediction_hash",
        "sealed_results_present",
    )
    for marker in required_markers:
        if marker not in combined:
            errors.append(f"rewritten notebook missing required marker: {marker}")
    prohibited_code = (
        "!pip",
        "%pip",
        "GOOGLE_API_KEY",
        "HF_TOKEN",
        "PINATA_JWT",
        "METAMASK_PRIVATE_KEY",
        "import gradio",
        "import whisper",
        "import web3",
        "hf_hub_download",
        "ui.launch(",
        "requests.post(",
        "requests.get(",
    )
    for marker in prohibited_code:
        if marker in code_text:
            errors.append(f"rewritten notebook contains prohibited legacy code: {marker}")
    check_notebook_code_cells(cells, errors)

    expected_notebook = _load_builder_notebook(errors)
    check_notebook_matches_builder(notebook, expected_notebook, errors)
    check_repository_notebook_mirror(notebook, errors)


def validate() -> list[str]:
    errors: list[str] = []
    check_expected_files(errors)
    check_design(errors)
    check_schemas(errors)
    check_manuscript(errors)
    check_bibliography(errors)
    check_notebook(errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    errors = validate()
    result = {"ok": not errors, "errors": errors}
    if args.json:
        print(json.dumps(result, indent=2))
    elif errors:
        for error in errors:
            print(f"ERROR: {error}")
    else:
        print("CDCV-Gate package checks passed.")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
