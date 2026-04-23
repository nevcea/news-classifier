import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from app.classifier import scorer
from app.classifier.interface import BaseClassifier, ClassificationResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_ENCODE_BATCH_SIZE = 64


class EmbeddingClassifier(BaseClassifier):
    """Classifies text via cosine similarity against per-category prototype embeddings.

    Prototype vectors are pre-normalized at build time so that classify() uses
    efficient dot products instead of recomputing norms on every call.

    Similarity is aggregated across prototypes using a configurable strategy
    (default: max). OTHER is excluded from prototypes — it is a threshold fallback.

    All prototype phrases are embedded in a single batched encode call for efficiency.
    classify_batch() encodes multiple texts in one call; use it in preference to
    calling classify() in a loop.
    """

    def __init__(
        self,
        categories: dict[str, list[str]],
        min_score: float,
        min_margin: float,
        min_score_by_label: dict[str, float] | None = None,
        min_margin_by_label: dict[str, float] | None = None,
        min_margin_by_pair: dict[str, dict[str, float]] | None = None,
        model_name: str = _DEFAULT_MODEL,
        aggregation: str = "max",
    ) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._min_score = min_score
        self._min_margin = min_margin
        self._min_score_by_label = min_score_by_label or {}
        self._min_margin_by_label = min_margin_by_label or {}
        self._min_margin_by_pair = min_margin_by_pair or {}
        self._aggregation = aggregation
        self._prototypes = self._build_prototypes(categories)

    def _build_prototypes(self, categories: dict[str, list[str]]) -> dict[str, list[np.ndarray]]:
        """Embed and pre-normalize every prototype phrase in a single batched call.

        All phrases from all categories are encoded together so the model processes
        a single batch instead of one batch per category.
        """
        # Collect (label, phrase) pairs, skipping OTHER and empty categories.
        pairs: list[tuple[str, str]] = [
            (label, phrase)
            for label, phrases in categories.items()
            if label != "OTHER" and phrases
            for phrase in phrases
        ]
        if not pairs:
            logger.warning("No prototype phrases found in categories config")
            return {}

        labels_seq = [lbl for lbl, _ in pairs]
        phrases_seq = [phr for _, phr in pairs]

        embeddings: np.ndarray = self._model.encode(
            phrases_seq,
            batch_size=_ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        result: dict[str, list[np.ndarray]] = {}
        for label, emb in zip(labels_seq, embeddings, strict=True):
            norm = float(np.linalg.norm(emb))
            result.setdefault(label, []).append(emb / norm if norm > 0.0 else emb)

        logger.info("Built prototypes for %d categories", len(result))
        return result

    @property
    def model_info(self) -> str:
        return self._model_name

    def classify(self, text: str) -> ClassificationResult:
        embedding: np.ndarray = self._model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return scorer.score(
            embedding,
            self._prototypes,
            self._min_score,
            self._min_margin,
            self._min_score_by_label,
            self._min_margin_by_label,
            self._min_margin_by_pair,
            self._aggregation,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Encode all texts in one batched call, then score each against prototypes.

        Significantly faster than calling classify() in a loop because the embedding
        model amortises overhead (tokenisation, GPU transfer, forward pass) across the
        full batch.
        """
        if not texts:
            return []
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=_ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [
            scorer.score(
                emb,
                self._prototypes,
                self._min_score,
                self._min_margin,
                self._min_score_by_label,
                self._min_margin_by_label,
                self._min_margin_by_pair,
                self._aggregation,
            )
            for emb in embeddings
        ]
