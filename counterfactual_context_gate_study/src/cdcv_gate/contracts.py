"""Dependency-free contracts for the CDCV-Gate runtime boundary.

JSON Schema remains the canonical storage contract.  These helpers enforce the
cross-record and access-control rules that JSON Schema cannot express by
itself, and are safe to import from notebooks and experiment runners.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Iterable, Mapping, Sequence


OTHER_UNLISTED = "OTHER_UNLISTED"

PERMITTED_CONTEXT_SLOTS = frozenset(
    {
        "relationship_role",
        "setting",
        "formality",
        "discourse_goal",
        "preceding_speech_act",
        "situation",
    }
)

PROTECTED_IDENTITY_FIELDS = frozenset(
    {
        "age",
        "caste",
        "citizenship",
        "disability",
        "ethnicity",
        "gender",
        "gender_identity",
        "genetic_information",
        "nationality",
        "pregnancy",
        "race",
        "religion",
        "sex",
        "sexual_orientation",
    }
)

SEALED_LABEL_FIELDS = frozenset(
    {
        "reference_action",
        "reference_sense_id",
        "acceptable_clarification_slots",
        "case_type",
        "gold_action",
        "gold_sense",
        "gold_label",
    }
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_all_mapping_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            keys.update(_all_mapping_keys(item))
    return keys


def validate_runtime_episode_gold_free(runtime: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    all_keys = _all_mapping_keys(runtime)
    leaked = sorted(all_keys.intersection(SEALED_LABEL_FIELDS))
    if leaked:
        errors.append("sealed labels leaked into runtime episode: " + ", ".join(leaked))
    protected = sorted(all_keys.intersection(PROTECTED_IDENTITY_FIELDS))
    if protected:
        errors.append(
            "protected identity keys leaked into runtime episode: "
            + ", ".join(protected)
        )
    candidate_items = runtime.get("candidate_senses", [])
    if not isinstance(candidate_items, Sequence) or isinstance(candidate_items, (str, bytes)):
        errors.append("runtime candidate_senses must be an array")
        return errors
    candidate_ids = [
        item.get("candidate_id")
        for item in candidate_items
        if isinstance(item, Mapping)
    ]
    if len(candidate_ids) != 3 or len(set(candidate_ids)) != 3:
        errors.append("runtime requires exactly three distinct candidate IDs")
    if candidate_ids.count(OTHER_UNLISTED) != 1:
        errors.append("runtime requires exactly one OTHER_UNLISTED candidate")
    return errors


def validate_context_card(card: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    all_keys = _all_mapping_keys(card)
    leaked = sorted(all_keys.intersection(SEALED_LABEL_FIELDS))
    if leaked:
        errors.append("sealed/evaluator field present in context card: " + ", ".join(leaked))
    nested_protected = sorted(all_keys.intersection(PROTECTED_IDENTITY_FIELDS))
    if nested_protected:
        errors.append(
            "protected identity key present in context card: "
            + ", ".join(nested_protected)
        )
    if card.get("scope") != "current_interaction_only":
        errors.append("context card must be scoped to the current interaction")
    fields = card.get("fields", {})
    if not isinstance(fields, Mapping):
        errors.append("context card fields must be an object")
        return errors
    field_names = {str(name) for name in fields}
    prohibited = sorted(field_names.intersection(PROTECTED_IDENTITY_FIELDS))
    if prohibited:
        errors.append("protected identity field present in context card: " + ", ".join(prohibited))
    unknown = sorted(field_names.difference(PERMITTED_CONTEXT_SLOTS))
    if unknown:
        errors.append("unsupported context slot: " + ", ".join(unknown))
    variety = card.get("variety_cue")
    if variety is not None:
        if not isinstance(variety, Mapping):
            errors.append("variety cue must be null or an object")
        else:
            if variety.get("provenance") not in {"self_declared", "experimentally_supplied"}:
                errors.append("variety cue must be self-declared or experimentally supplied")
            if variety.get("retain_after_episode") is not False:
                errors.append("variety cue cannot persist after the episode")
    for slot, field in fields.items():
        if not isinstance(field, Mapping):
            errors.append(f"context field {slot!r} must be an object")
            continue
        if field.get("retain_after_episode") is not False:
            errors.append(f"context field {slot!r} cannot persist after the episode")
        if field.get("provenance") == "missing" and field.get("value") is not None:
            errors.append(f"missing context field {slot!r} must have a null value")
    hard_conflicts = [
        item
        for item in card.get("conflicts", [])
        if isinstance(item, Mapping) and item.get("severity") == "hard_stop"
    ]
    for conflict in hard_conflicts:
        if conflict.get("slot") not in PERMITTED_CONTEXT_SLOTS | {"variety_cue"}:
            errors.append("hard conflict names an unsupported slot")
    return errors


def apply_context_patch(
    card: Mapping[str, object],
    changed_slots: Iterable[str],
    patch: Mapping[str, Mapping[str, object]],
) -> dict:
    """Apply a reviewed slot patch without allowing identity/variety mutation."""

    declared = tuple(str(slot) for slot in changed_slots)
    if not declared or len(set(declared)) != len(declared):
        raise ValueError("changed slots must be non-empty and unique")
    if not set(declared).issubset(PERMITTED_CONTEXT_SLOTS):
        raise ValueError("context intervention requested a prohibited slot")
    if set(patch) != set(declared):
        raise ValueError("context patch keys must exactly equal declared changed slots")
    updated = deepcopy(dict(card))
    fields = deepcopy(dict(updated.get("fields", {})))
    for slot in declared:
        value = deepcopy(dict(patch[slot]))
        if value.get("retain_after_episode") is not False:
            raise ValueError("patched context cannot persist after the episode")
        fields[slot] = value
    updated["fields"] = fields
    errors = validate_context_card(updated)
    if errors:
        raise ValueError("; ".join(errors))
    return updated


def validate_intervention_bundle_integrity(
    runtime: Mapping[str, object],
    bundle: Mapping[str, object],
    source_context: Mapping[str, object],
    resulting_contexts: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Validate symmetric branches and cross-record context integrity."""

    errors: list[str] = []
    if bundle.get("family_id") != runtime.get("family_id"):
        errors.append("bundle family_id does not match runtime episode")
    if bundle.get("base_case_id") != runtime.get("case_id"):
        errors.append("bundle base_case_id does not match runtime episode")
    if bundle.get("utterance_hash") != runtime.get("utterance_hash"):
        errors.append("bundle utterance hash does not match runtime episode")
    if bundle.get("candidate_set_hash") != runtime.get("candidate_set_hash"):
        errors.append("bundle candidate-set hash does not match runtime episode")
    validation = bundle.get("validation", {})
    if not isinstance(validation, Mapping) or validation.get("status") != "accepted":
        errors.append("only accepted intervention bundles may reach inference")
    if bundle.get("variety_cue_fixed") is not True:
        errors.append("bundle does not lock the variety cue")
    if bundle.get("constructed_without_sealed_reference_action") is not True:
        errors.append("bundle lacks the no-sealed-label construction lock")

    candidates = runtime.get("candidate_senses", [])
    candidate_ids = [
        item.get("candidate_id")
        for item in candidates
        if isinstance(item, Mapping)
    ]
    core_ids = [value for value in candidate_ids if value != OTHER_UNLISTED]
    if len(core_ids) != 2 or len(set(core_ids)) != 2:
        errors.append("runtime episode must contain exactly two distinct core candidates")
        return errors

    branches = bundle.get("candidate_branches", [])
    if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
        errors.append("candidate branches must be an array")
        return errors
    sources = [
        branch.get("source_candidate_id")
        for branch in branches
        if isinstance(branch, Mapping)
    ]
    if len(branches) != 2 or set(sources) != set(core_ids) or len(set(sources)) != 2:
        errors.append("candidate branches are not exactly symmetric across core candidates")

    source_hash = runtime.get("context_card_hash")
    recomputed_source_hash = sha256_json(source_context)
    if source_hash != recomputed_source_hash:
        errors.append("runtime context-card hash does not match canonical source context")
    for error in validate_context_card(source_context):
        errors.append("source context is invalid: " + error)
    source_fields = source_context.get("fields", {})
    source_variety = source_context.get("variety_cue")
    for branch in branches:
        if not isinstance(branch, Mapping):
            errors.append("candidate branch must be an object")
            continue
        source_candidate = branch.get("source_candidate_id")
        preserving = branch.get("preserving", [])
        changing = branch.get("meaning_changing", [])
        if not isinstance(preserving, Sequence) or not isinstance(changing, Sequence):
            errors.append(f"branch {source_candidate!r} probe lists are invalid")
            continue
        if len(preserving) != 1 or len(changing) != 1:
            errors.append(f"branch {source_candidate!r} must contain one probe of each type")
            continue
        expected_target = next((value for value in core_ids if value != source_candidate), None)
        changing_probe = changing[0]
        if not isinstance(changing_probe, Mapping) or changing_probe.get("target_candidate_id") != expected_target:
            errors.append(f"branch {source_candidate!r} does not target the other core candidate")

        for probe in tuple(preserving) + tuple(changing):
            if not isinstance(probe, Mapping):
                errors.append("probe must be an object")
                continue
            if probe.get("source_context_hash") != source_hash:
                errors.append(f"probe {probe.get('intervention_id')!r} source hash mismatch")
            declared = set(probe.get("changed_slots", []))
            if not declared or not declared.issubset(PERMITTED_CONTEXT_SLOTS):
                errors.append(f"probe {probe.get('intervention_id')!r} changes a prohibited slot")
            result_hash = probe.get("result_context_hash")
            result = resulting_contexts.get(str(result_hash))
            if result is None:
                errors.append(f"probe {probe.get('intervention_id')!r} result context is unavailable")
                continue
            if result_hash != sha256_json(result):
                errors.append(
                    f"probe {probe.get('intervention_id')!r} result hash is not canonical"
                )
            for error in validate_context_card(result):
                errors.append(
                    f"probe {probe.get('intervention_id')!r} result context is invalid: "
                    + error
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
    question: Mapping[str, object],
    manifest: Mapping[str, object],
    available_probe_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    for key in ("question_id", "family_id", "context_slot", "candidate_set_hash"):
        if question.get(key) != manifest.get(key):
            errors.append(f"clarification scenario {key} does not match its question")
    answer_domain = question.get("answer_domain", [])
    scenarios = manifest.get("scenarios", [])
    answer_ids = [
        item.get("answer_id") for item in answer_domain if isinstance(item, Mapping)
    ]
    scenario_ids = [
        item.get("answer_id") for item in scenarios if isinstance(item, Mapping)
    ]
    if len(scenario_ids) != len(set(scenario_ids)) or set(scenario_ids) != set(answer_ids):
        errors.append("clarification scenarios do not map one-to-one to answer IDs")
    probability_sum = sum(
        float(item.get("prior_probability", 0.0))
        for item in scenarios
        if isinstance(item, Mapping)
    )
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("clarification scenario priors must sum to one")
    if manifest.get("mode") == "PRIMARY_REUSE_ONLY":
        if manifest.get("additional_model_calls") != 0:
            errors.append("primary clarification scenarios must add zero model calls")
        for item in scenarios:
            if not isinstance(item, Mapping):
                errors.append("clarification scenario must be an object")
                continue
            source = item.get("score_source", {})
            if not isinstance(source, Mapping):
                errors.append("clarification scenario score source must be an object")
                continue
            if source.get("source_type") != "REUSED_PROBE_SCORES":
                errors.append("primary clarification scenario does not reuse probe scores")
            if source.get("probe_id") not in available_probe_ids:
                errors.append("clarification scenario references an unavailable probe")
    return errors


def freeze_prediction_hash(predictions: Sequence[Mapping[str, object]]) -> str:
    """Return the immutable prediction-artifact hash used before gold access."""

    return sha256_json(list(predictions))


def join_labels_after_prediction_freeze(
    predictions: Sequence[Mapping[str, object]],
    labels: Sequence[Mapping[str, object]],
    *,
    frozen_prediction_hash: str,
) -> list[dict]:
    """Join labels only when the supplied predictions match their frozen hash."""

    if freeze_prediction_hash(predictions) != frozen_prediction_hash:
        raise ValueError("prediction artifact does not match its frozen hash")
    prediction_ids = [str(item["case_id"]) for item in predictions]
    label_ids = [str(item["case_id"]) for item in labels]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction artifact contains duplicate case IDs")
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("sealed labels contain duplicate case IDs")
    if set(prediction_ids) != set(label_ids):
        raise ValueError("prediction and sealed-label case sets must match exactly")
    label_by_case = {str(item["case_id"]): dict(item) for item in labels}
    joined: list[dict] = []
    for prediction in predictions:
        case_id = str(prediction["case_id"])
        if case_id not in label_by_case:
            raise ValueError(f"sealed label missing for case {case_id!r}")
        joined.append({"prediction": dict(prediction), "sealed_label": label_by_case[case_id]})
    return joined
