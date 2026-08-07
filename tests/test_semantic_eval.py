from __future__ import annotations

import pytest

from iedi.semantic_eval import (
    EmbeddingManifest,
    RecordedSAERReference,
    SAERCase,
    evaluate_recorded_saer_results,
    evaluate_saer_cases,
)
from iedi.pipeline import build_pipeline
from iedi.schemas import InterpretationRequest


class FakeEmbedder:
    model_id = "fake-e5"
    revision = "commit-sha-123"
    pooling = "mean"
    preprocessing = "NFKC-v1"
    vectors = {
        "please": (1.0, 0.0),
        "a polite request": (1.0, 0.0),
        "seriously": (0.0, 1.0),
        "disbelief": (0.0, 1.0),
    }

    def encode(self, texts):
        return [self.vectors[text] for text in texts]


def manifest() -> EmbeddingManifest:
    return EmbeddingManifest(
        model_id="fake-e5",
        revision="commit-sha-123",
        pooling="mean",
        preprocessing="NFKC-v1",
    )


def test_saer_report_pins_model_and_breaks_down_routes() -> None:
    report = evaluate_saer_cases(
        [
            SAERCase(
                "1", "NgE", "pro", "please", "a polite request", "ref-1", "a" * 64
            ),
            SAERCase(
                "2", "NgE", "flash", "seriously", "disbelief", "ref-2", "b" * 64
            ),
        ],
        embedder=FakeEmbedder(),
        embedding_manifest=manifest(),
    )
    assert report.overall.saer == pytest.approx(0.0)
    assert report.overall.semantic_resolution == pytest.approx(1.0)
    assert set(report.by_route) == {"pro", "flash"}
    assert report.embedding_manifest.revision == "commit-sha-123"


def test_correct_local_retrieval_can_be_evaluated_when_bound_to_top1_result() -> None:
    report = evaluate_saer_cases(
        [
            SAERCase(
                "1",
                "NgE",
                "local",
                "please",
                "a polite request",
                "ref-1",
                "c" * 64,
                retrieved_entry_ids=("ref-1",),
            )
        ],
        embedder=FakeEmbedder(),
        embedding_manifest=manifest(),
    )
    assert report.overall.sample_count == 1


def test_embedder_must_return_one_vector_per_case() -> None:
    class BadEmbedder(FakeEmbedder):
        def encode(self, texts):
            return [(1.0, 0.0), (1.0, 0.0)]

    with pytest.raises(ValueError, match="one vector"):
        evaluate_saer_cases(
            [
                SAERCase(
                    "1", "NgE", "pro", "please", "a polite request", "ref-1", "d" * 64
                )
            ],
            embedder=BadEmbedder(),
            embedding_manifest=manifest(),
        )


def test_recorded_saer_uses_pipeline_top1(codebook, fake_provider) -> None:
    result = build_pipeline(
        "paper3", codebook=codebook, provider=fake_provider
    ).interpret(InterpretationRequest("wahala", active_persona_ids=("ng-en-v1",)))

    class ConstantEmbedder(FakeEmbedder):
        def encode(self, texts):
            return [(1.0, 0.0) for _ in texts]

    report = evaluate_recorded_saer_results(
        [
            RecordedSAERReference(
                "case-1",
                "Nigerian English",
                result,
                result.candidates[0].clarification,
                result.candidates[0].entry_id or "unknown",
            )
        ],
        embedder=ConstantEmbedder(),
        embedding_manifest=manifest(),
    )
    assert report.overall.saer == 0.0
