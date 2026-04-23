import numpy as np

from app.classifier.interface import ClassificationResult

# Prototypes stored in EmbeddingClassifier are pre-normalized unit vectors.
# Text embeddings are normalized once per classify() call.
# This allows cosine similarity to be computed as a simple dot product.


def _aggregate(similarities: list[float], method: str) -> float:
    if not similarities:
        return 0.0
    if method == "max":
        return max(similarities)
    if method == "mean":
        return sum(similarities) / len(similarities)
    raise ValueError(f"Unknown aggregation method: {method!r}")


def score(
    text_embedding: np.ndarray,
    prototype_embeddings: dict[str, list[np.ndarray]],
    min_score: float,
    min_margin: float = 0.0,
    min_score_by_label: dict[str, float] | None = None,
    min_margin_by_label: dict[str, float] | None = None,
    min_margin_by_pair: dict[str, dict[str, float]] | None = None,
    aggregation: str = "max",
) -> ClassificationResult:
    """Score text against pre-normalized category prototypes.

    OTHER policy: predicted_label = "OTHER" when top_score < min_score
    OR when margin < min_margin (ambiguous between top two categories).
    """
    _other = ClassificationResult(
        predicted_label="OTHER",
        top_label="OTHER",
        second_label="OTHER",
        top_score=0.0,
        second_score=0.0,
        margin=0.0,
        applied_min_score=min_score,
        applied_min_margin=min_margin,
    )

    if not prototype_embeddings:
        return _other

    norm = float(np.linalg.norm(text_embedding))
    if norm == 0.0:
        return _other
    text_norm = text_embedding / norm

    scores: dict[str, float] = {
        label: _aggregate(
            [float(np.dot(text_norm, p)) for p in protos],
            aggregation,
        )
        for label, protos in prototype_embeddings.items()
    }

    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    top_label = ranked[0]
    top_score = scores[top_label]
    second_label = ranked[1] if len(ranked) > 1 else top_label
    second_score = scores[second_label] if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    applied_min_score = (
        min_score_by_label.get(top_label, min_score) if min_score_by_label else min_score
    )
    applied_min_margin = (
        min_margin_by_label.get(top_label, min_margin) if min_margin_by_label else min_margin
    )
    if min_margin_by_pair:
        applied_min_margin = min_margin_by_pair.get(top_label, {}).get(
            second_label,
            applied_min_margin,
        )
    rejected_by_score = top_score < applied_min_score
    rejected_by_margin = margin < applied_min_margin
    if rejected_by_score and rejected_by_margin:
        predicted_label = "OTHER"
        reject_reason = "score_and_margin"
    elif rejected_by_score:
        predicted_label = "OTHER"
        reject_reason = "score"
    elif rejected_by_margin:
        predicted_label = "OTHER"
        reject_reason = "margin"
    else:
        predicted_label = top_label
        reject_reason = None

    return ClassificationResult(
        predicted_label=predicted_label,
        top_label=top_label,
        second_label=second_label,
        top_score=top_score,
        second_score=second_score,
        margin=margin,
        applied_min_score=applied_min_score,
        applied_min_margin=applied_min_margin,
        reject_reason=reject_reason,
    )
