from unittest.mock import MagicMock, patch

import numpy as np

from app.classifier.embedding import EmbeddingClassifier


def _make_classifier(categories: dict, aggregation: str = "max", min_score: float = 0.0):
    """Instantiate EmbeddingClassifier with a mocked SentenceTransformer."""
    mock_model = MagicMock()

    def fake_encode(texts, **kwargs):
        # Return an identity-like matrix so each phrase gets a distinct vector.
        n = len(texts)
        return np.eye(n, max(n, 3))

    mock_model.encode.side_effect = fake_encode

    with patch("app.classifier.embedding.SentenceTransformer", return_value=mock_model):
        clf = EmbeddingClassifier(
            categories=categories,
            min_score=min_score,
            min_margin=0.05,
            aggregation=aggregation,
        )
    # Attach mock so tests can call classify() later with the same mock.
    clf._model = mock_model
    return clf


# --- Prototype structure ---


def test_other_excluded_from_prototypes():
    clf = _make_classifier({"AI": ["phrase"], "OTHER": ["misc"]})
    assert "OTHER" not in clf._prototypes
    assert "AI" in clf._prototypes


def test_empty_phrases_excluded():
    clf = _make_classifier({"AI": [], "CLOUD": ["cloud thing"]})
    assert "AI" not in clf._prototypes
    assert "CLOUD" in clf._prototypes


def test_per_prototype_stored_as_list():
    clf = _make_classifier({"AI": ["phrase one", "phrase two"]})
    assert isinstance(clf._prototypes["AI"], list)
    assert len(clf._prototypes["AI"]) == 2


def test_single_prototype_stored_as_list():
    clf = _make_classifier({"AI": ["only phrase"]})
    assert isinstance(clf._prototypes["AI"], list)
    assert len(clf._prototypes["AI"]) == 1


# --- Aggregation parameter ---


def test_aggregation_default_is_max():
    clf = _make_classifier({"AI": ["phrase"]})
    assert clf._aggregation == "max"


def test_aggregation_mean_stored():
    clf = _make_classifier({"AI": ["phrase"]}, aggregation="mean")
    assert clf._aggregation == "mean"


# --- model_info property ---


def test_model_info_returns_model_name():
    clf = _make_classifier({"AI": ["phrase"]})
    assert clf.model_info == "all-MiniLM-L6-v2"


# --- classify() integration ---


def test_classify_returns_classification_result():
    clf = _make_classifier({"AI": ["ai phrase"], "CLOUD": ["cloud phrase"]}, min_score=0.0)

    def fake_encode(texts, **kwargs):
        # Both categories and input text use identity-like vectors.
        return np.eye(len(texts), 3)

    clf._model.encode.side_effect = fake_encode
    result = clf.classify("some text")
    assert result.predicted_label in {"AI", "CLOUD", "OTHER"}
    assert isinstance(result.top_score, float)
    assert isinstance(result.margin, float)


def test_classify_below_threshold_returns_other():
    clf = _make_classifier({"AI": ["phrase"]}, min_score=0.99)

    def fake_encode(texts, **kwargs):
        # Prototype was built as [1, 0, 0]; return [0, 1, 0] for text → sim=0.
        n = len(texts)
        arr = np.zeros((n, 3))
        arr[:, 1] = 1.0  # orthogonal to prototype on axis 0
        return arr

    clf._model.encode.side_effect = fake_encode
    result = clf.classify("unrelated text")
    # score will be low; threshold is 0.99 → OTHER
    assert result.predicted_label == "OTHER"
    assert result.top_label != "OTHER"
