from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SAERResult:
    sample_count: int
    saer: float
    semantic_resolution: float
    per_sample_distance: tuple[float, ...]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embeddings must be non-empty and have equal dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("zero-norm embeddings are not valid for cosine similarity")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def semantic_ambiguity_error_rate(
    hypotheses: Iterable[Sequence[float]],
    references: Iterable[Sequence[float]],
) -> SAERResult:
    hypothesis_list = list(hypotheses)
    reference_list = list(references)
    if len(hypothesis_list) != len(reference_list) or not hypothesis_list:
        raise ValueError("hypothesis and reference sets must have equal non-zero length")
    distances = tuple(
        1.0 - cosine_similarity(hypothesis, reference)
        for hypothesis, reference in zip(hypothesis_list, reference_list)
    )
    saer = statistics.fmean(distances)
    return SAERResult(
        sample_count=len(distances),
        saer=saer,
        semantic_resolution=1.0 - saer,
        per_sample_distance=distances,
    )


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    mean: float
    median: float
    p95: float
    standard_deviation: float
    ci95_low: float
    ci95_high: float


def summarize(values: Iterable[float]) -> DistributionSummary:
    data = sorted(float(value) for value in values)
    if not data:
        raise ValueError("at least one observation is required")
    count = len(data)
    mean = statistics.fmean(data)
    median = statistics.median(data)
    p95_index = max(0, math.ceil(0.95 * count) - 1)
    p95 = data[p95_index]
    standard_deviation = statistics.stdev(data) if count > 1 else 0.0
    half_width = 1.96 * standard_deviation / math.sqrt(count) if count > 1 else 0.0
    return DistributionSummary(
        count=count,
        mean=mean,
        median=median,
        p95=p95,
        standard_deviation=standard_deviation,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
    )


def route_breakdown(routes: Iterable[str]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        counts[route] = counts.get(route, 0) + 1
    return counts
