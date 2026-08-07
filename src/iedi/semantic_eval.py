from __future__ import annotations

import re
import unicodedata
import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

from .metrics import SAERResult, semantic_ambiguity_error_rate
from .codebook import canonical_json
from .schemas import InterpretationResult


class SemanticEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class EmbeddingManifest:
    model_id: str
    revision: str
    pooling: str
    preprocessing: str

    def __post_init__(self) -> None:
        if not all((self.model_id, self.revision, self.pooling, self.preprocessing)):
            raise ValueError("embedding model, revision, pooling and preprocessing are required")


@dataclass(frozen=True)
class SAERCase:
    case_id: str
    dialect: str
    route: str
    hypothesis: str
    reference: str
    reference_entry_id: str
    source_result_sha256: str
    selected_candidate_rank: int = 1
    retrieved_entry_ids: tuple[str, ...] = ()
    split: str = "test"


@dataclass(frozen=True)
class GroundedSAERReport:
    embedding_manifest: EmbeddingManifest
    overall: SAERResult
    by_dialect: Mapping[str, SAERResult]
    by_route: Mapping[str, SAERResult]


@dataclass(frozen=True)
class RecordedSAERReference:
    case_id: str
    dialect: str
    result: InterpretationResult
    reference: str
    reference_entry_id: str
    split: str = "test"


def evaluate_recorded_saer_results(
    references: Iterable[RecordedSAERReference],
    *,
    embedder: SemanticEmbedder,
    embedding_manifest: EmbeddingManifest,
) -> GroundedSAERReport:
    """Evaluate the recorded top-1 output, never a caller-selected alternative."""

    cases: list[SAERCase] = []
    for item in references:
        if not item.result.candidates:
            raise ValueError(f"result for {item.case_id} has no top-1 candidate")
        result_hash = hashlib.sha256(canonical_json(item.result.to_dict())).hexdigest()
        cases.append(
            SAERCase(
                case_id=item.case_id,
                dialect=item.dialect,
                route=item.result.decision.used_route.value,
                hypothesis=item.result.candidates[0].clarification,
                reference=item.reference,
                reference_entry_id=item.reference_entry_id,
                source_result_sha256=result_hash,
                selected_candidate_rank=1,
                retrieved_entry_ids=item.result.retrieved_entry_ids,
                split=item.split,
            )
        )
    return evaluate_saer_cases(
        cases,
        embedder=embedder,
        embedding_manifest=embedding_manifest,
    )


def evaluate_saer_cases(
    cases: Iterable[SAERCase],
    *,
    embedder: SemanticEmbedder,
    embedding_manifest: EmbeddingManifest,
) -> GroundedSAERReport:
    values = tuple(cases)
    if not values:
        raise ValueError("SAER evaluation requires at least one held-out case")
    if len({case.case_id for case in values}) != len(values):
        raise ValueError("SAER cases contain duplicate case IDs")
    embedder_identity = (
        getattr(embedder, "model_id", None),
        getattr(embedder, "revision", None),
        getattr(embedder, "pooling", None),
        getattr(embedder, "preprocessing", None),
    )
    manifest_identity = (
        embedding_manifest.model_id,
        embedding_manifest.revision,
        embedding_manifest.pooling,
        embedding_manifest.preprocessing,
    )
    if embedder_identity != manifest_identity:
        raise ValueError("embedder implementation does not match the embedding manifest")
    for case in values:
        if case.split != "test":
            raise ValueError(f"case {case.case_id} is not in the frozen test split")
        if case.selected_candidate_rank != 1:
            raise ValueError(f"case {case.case_id} must evaluate the recorded top-1 candidate")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", case.source_result_sha256):
            raise ValueError(f"case {case.case_id} lacks a valid source-result hash")
        if not case.hypothesis.strip() or not case.reference.strip():
            raise ValueError(f"case {case.case_id} has empty hypothesis/reference text")

    hypothesis_vectors = embedder.encode([case.hypothesis for case in values])
    reference_vectors = embedder.encode([case.reference for case in values])
    if len(hypothesis_vectors) != len(values) or len(reference_vectors) != len(values):
        raise ValueError("embedder must return exactly one vector per SAER case")
    overall = semantic_ambiguity_error_rate(hypothesis_vectors, reference_vectors)

    indexed_vectors = list(zip(values, hypothesis_vectors, reference_vectors))
    by_dialect = _grouped_saer(indexed_vectors, lambda case: case.dialect)
    by_route = _grouped_saer(indexed_vectors, lambda case: case.route)
    return GroundedSAERReport(
        embedding_manifest=embedding_manifest,
        overall=overall,
        by_dialect=by_dialect,
        by_route=by_route,
    )


def _grouped_saer(indexed_vectors, key):
    groups: dict[str, tuple[list[Sequence[float]], list[Sequence[float]]]] = {}
    for case, hypothesis, reference in indexed_vectors:
        hypotheses, references = groups.setdefault(key(case), ([], []))
        hypotheses.append(hypothesis)
        references.append(reference)
    return {
        name: semantic_ambiguity_error_rate(hypotheses, references)
        for name, (hypotheses, references) in groups.items()
    }


class SentenceTransformerEmbedder:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        pooling: str = "model-default",
        preprocessing: str = "NFKC-v1",
        model=None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.pooling = pooling
        self.preprocessing = preprocessing
        if model is not None:
            self.model = model
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the evaluation extra") from exc
        self.model = SentenceTransformer(model_id, revision=revision)

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self.preprocessing != "NFKC-v1":
            raise ValueError(f"unsupported preprocessing policy: {self.preprocessing}")
        prepared = [" ".join(unicodedata.normalize("NFKC", text).split()) for text in texts]
        vectors = self.model.encode(prepared, normalize_embeddings=False)
        return [tuple(float(value) for value in vector) for vector in vectors]
