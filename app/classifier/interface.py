from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationResult:
    predicted_label: str  # after threshold fallback; may be "OTHER"
    top_label: str  # highest-scoring label before any fallback
    second_label: str  # second highest-scoring label
    top_score: float
    second_score: float
    margin: float
    applied_min_score: float = 0.0
    applied_min_margin: float = 0.0
    reject_reason: str | None = None


class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, text: str) -> ClassificationResult: ...

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify a list of texts. Subclasses should override with a batched implementation."""
        return [self.classify(t) for t in texts]
