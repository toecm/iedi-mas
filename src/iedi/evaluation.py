from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    dialect: str
    audio_sha256: str
    reference_entry_id: str
    split: str = "test"


@dataclass(frozen=True)
class Prediction:
    case_id: str
    predicted_entry_ids: tuple[str, ...]
    route: str
    predicted_dialect: str | None = None


@dataclass(frozen=True)
class HumanRating:
    case_id: str
    evaluator_id: str
    fidelity: int
    naturalness: int

    def __post_init__(self) -> None:
        if not 1 <= self.fidelity <= 5 or not 1 <= self.naturalness <= 5:
            raise ValueError("human ratings must be in the inclusive range 1-5")


@dataclass(frozen=True)
class AccuracyReport:
    total: int
    correct_top1: int
    top1_accuracy: float
    recall_at_3: float
    by_dialect: Mapping[str, float]
    dialect_identification_accuracy: float | None = None


@dataclass(frozen=True)
class Paper2EvaluationReport:
    accuracy: AccuracyReport
    human_ratings: Mapping[str, float]


def validate_paper2_manifest(cases: Iterable[EvaluationCase]) -> tuple[EvaluationCase, ...]:
    values = tuple(cases)
    counts: dict[str, int] = {}
    seen_audio: set[str] = set()
    seen_cases: set[str] = set()
    for case in values:
        if case.split != "test":
            raise ValueError("Paper 2 reproduction manifest must be frozen to the test split")
        if case.audio_sha256 in seen_audio:
            raise ValueError(f"duplicate audio in evaluation manifest: {case.audio_sha256}")
        seen_audio.add(case.audio_sha256)
        if case.case_id in seen_cases:
            raise ValueError(f"duplicate case_id in evaluation manifest: {case.case_id}")
        seen_cases.add(case.case_id)
        counts[case.dialect] = counts.get(case.dialect, 0) + 1
    expected = {"Nigerian English": 50, "Korean English": 50, "American English": 50}
    if counts != expected:
        raise ValueError(f"Paper 2 requires exactly 50 clips per dialect; observed {counts}")
    return values


def evaluate_predictions(
    cases: Iterable[EvaluationCase], predictions: Iterable[Prediction]
) -> AccuracyReport:
    case_values = tuple(cases)
    prediction_values = tuple(predictions)
    if len({case.case_id for case in case_values}) != len(case_values):
        raise ValueError("evaluation cases contain duplicate case IDs")
    if len({prediction.case_id for prediction in prediction_values}) != len(
        prediction_values
    ):
        raise ValueError("predictions contain duplicate case IDs")
    case_map = {case.case_id: case for case in case_values}
    prediction_map = {prediction.case_id: prediction for prediction in prediction_values}
    if set(case_map) != set(prediction_map):
        raise ValueError("predictions must cover the evaluation manifest exactly once")
    total = len(case_map)
    if total == 0:
        raise ValueError("evaluation set cannot be empty")

    correct_top1 = 0
    correct_top3 = 0
    dialect_identity_correct = 0
    dialect_totals: dict[str, int] = {}
    entry_correct_by_dialect: dict[str, int] = {}
    for case_id, case in case_map.items():
        predicted = prediction_map[case_id].predicted_entry_ids
        dialect_totals[case.dialect] = dialect_totals.get(case.dialect, 0) + 1
        if predicted and predicted[0] == case.reference_entry_id:
            correct_top1 += 1
            entry_correct_by_dialect[case.dialect] = (
                entry_correct_by_dialect.get(case.dialect, 0) + 1
            )
        if case.reference_entry_id in predicted[:3]:
            correct_top3 += 1
        if prediction_map[case_id].predicted_dialect == case.dialect:
            dialect_identity_correct += 1
    dialect_predictions_present = [
        prediction.predicted_dialect is not None for prediction in prediction_values
    ]
    if any(dialect_predictions_present) and not all(dialect_predictions_present):
        raise ValueError("predicted_dialect must be supplied for every case or none")
    return AccuracyReport(
        total=total,
        correct_top1=correct_top1,
        top1_accuracy=correct_top1 / total,
        recall_at_3=correct_top3 / total,
        by_dialect={
            dialect: entry_correct_by_dialect.get(dialect, 0) / count
            for dialect, count in dialect_totals.items()
        },
        dialect_identification_accuracy=(
            dialect_identity_correct / total if all(dialect_predictions_present) else None
        ),
    )


def summarize_human_ratings(
    ratings: Iterable[HumanRating],
    *,
    expected_evaluators_per_case: int = 10,
    expected_case_ids: Iterable[str] | None = None,
) -> Mapping[str, float]:
    values = tuple(ratings)
    if not values:
        raise ValueError("human ratings cannot be empty")
    evaluators_by_case: dict[str, set[str]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for rating in values:
        pair = (rating.case_id, rating.evaluator_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate evaluator rating for case: {pair}")
        seen_pairs.add(pair)
        evaluators_by_case.setdefault(rating.case_id, set()).add(rating.evaluator_id)
    if expected_case_ids is not None:
        expected = set(expected_case_ids)
        if set(evaluators_by_case) != expected:
            raise ValueError("human ratings must cover the evaluation manifest exactly")
    invalid = {
        case_id: len(evaluators)
        for case_id, evaluators in evaluators_by_case.items()
        if len(evaluators) != expected_evaluators_per_case
    }
    if invalid:
        raise ValueError(f"expected {expected_evaluators_per_case} evaluators per case: {invalid}")
    return {
        "fidelity_mean": statistics.fmean(rating.fidelity for rating in values),
        "naturalness_mean": statistics.fmean(rating.naturalness for rating in values),
        "rating_count": float(len(values)),
    }


def evaluate_paper2_reproduction(
    cases: Iterable[EvaluationCase],
    predictions: Iterable[Prediction],
    ratings: Iterable[HumanRating],
) -> Paper2EvaluationReport:
    """Fail-closed Paper 2 report from a frozen 150-clip manifest and raw rows."""

    manifest = validate_paper2_manifest(cases)
    accuracy = evaluate_predictions(manifest, predictions)
    if accuracy.dialect_identification_accuracy is None:
        raise ValueError("Paper 2 reproduction requires predicted_dialect for every clip")
    human = summarize_human_ratings(
        ratings,
        expected_evaluators_per_case=10,
        expected_case_ids=(case.case_id for case in manifest),
    )
    return Paper2EvaluationReport(accuracy=accuracy, human_ratings=human)
