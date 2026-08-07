from __future__ import annotations

import math

import pytest

from iedi.edge import payload_reduction
from iedi.evaluation import (
    EvaluationCase,
    HumanRating,
    Prediction,
    evaluate_paper2_reproduction,
    evaluate_predictions,
    summarize_human_ratings,
    validate_paper2_manifest,
)
from iedi.metrics import semantic_ambiguity_error_rate, summarize


def test_saer_is_distance_and_resolution_is_one_minus_saer() -> None:
    result = semantic_ambiguity_error_rate(
        [(1.0, 0.0), (1.0, 0.0)],
        [(1.0, 0.0), (0.8, 0.6)],
    )
    assert result.saer == pytest.approx(0.1)
    assert result.semantic_resolution == pytest.approx(0.9)
    assert result.saer + result.semantic_resolution == pytest.approx(1.0)


def test_payload_reduction_does_not_round_into_false_claim() -> None:
    reduction_45 = payload_reduction(96 * 1024, 45)
    reduction_120 = payload_reduction(96 * 1024, 120)
    assert reduction_45 > 0.999
    assert reduction_120 < 0.999


def test_distribution_reports_p95_and_confidence_interval() -> None:
    report = summarize(range(1, 101))
    assert report.count == 100
    assert report.median == 50.5
    assert report.p95 == 95
    assert report.ci95_low < report.mean < report.ci95_high


def test_prediction_metrics_are_derived_from_raw_rows() -> None:
    cases = [
        EvaluationCase("1", "NgE", "hash1", "a"),
        EvaluationCase("2", "NgE", "hash2", "b"),
    ]
    report = evaluate_predictions(
        cases,
        [
            Prediction("1", ("a",), "local", predicted_dialect="NgE"),
            Prediction("2", ("x", "b"), "pro", predicted_dialect="other"),
        ],
    )
    assert report.top1_accuracy == 0.5
    assert report.recall_at_3 == 1.0
    assert report.dialect_identification_accuracy == 0.5


def test_paper2_manifest_requires_exactly_50_per_named_dialect() -> None:
    cases = []
    for dialect in ("Nigerian English", "Korean English", "American English"):
        for index in range(50):
            cases.append(
                EvaluationCase(
                    f"{dialect}-{index}",
                    dialect,
                    f"{dialect}-hash-{index}",
                    f"entry-{index}",
                )
            )
    assert len(validate_paper2_manifest(cases)) == 150

    predictions = [
        Prediction(
            case.case_id,
            (case.reference_entry_id,),
            "local",
            predicted_dialect=case.dialect,
        )
        for case in cases
    ]
    ratings = [
        HumanRating(case.case_id, f"e{index}", 5, 5)
        for case in cases
        for index in range(10)
    ]
    report = evaluate_paper2_reproduction(cases, predictions, ratings)
    assert report.accuracy.dialect_identification_accuracy == 1.0
    assert report.human_ratings["rating_count"] == 1500


def test_human_ratings_validate_count_and_range() -> None:
    ratings = [HumanRating("case", f"e{i}", 4, 5) for i in range(10)]
    report = summarize_human_ratings(ratings)
    assert report["fidelity_mean"] == 4
    assert report["naturalness_mean"] == 5
    with pytest.raises(ValueError, match="1-5"):
        HumanRating("case", "bad", 6, 1)
    with pytest.raises(ValueError, match="duplicate evaluator"):
        summarize_human_ratings([*ratings, ratings[0]])


def test_duplicate_prediction_ids_are_rejected() -> None:
    cases = [EvaluationCase("1", "NgE", "hash1", "a")]
    with pytest.raises(ValueError, match="duplicate case IDs"):
        evaluate_predictions(
            cases,
            [Prediction("1", ("a",), "local"), Prediction("1", ("a",), "local")],
        )
