import numpy as np
import pytest

from app.classifier.scorer import score


def _unit(values: list[float]) -> np.ndarray:
    v = np.array(values, dtype=float)
    return v / np.linalg.norm(v)


# --- OTHER policy ---


def test_top_score_below_threshold_returns_other():
    text = _unit([1.0, 0.0, 0.0])
    protos = {"A": [_unit([0.0, 1.0, 0.0])]}  # orthogonal → sim=0
    result = score(text, protos, min_score=0.5)
    assert result.predicted_label == "OTHER"
    assert result.top_label == "A"  # raw winner before fallback
    assert result.reject_reason == "score"


def test_top_score_above_threshold_returns_label():
    text = _unit([1.0, 0.0, 0.0])
    protos = {"A": [_unit([1.0, 0.0, 0.0])]}  # perfect match → sim=1
    result = score(text, protos, min_score=0.5)
    assert result.predicted_label == "A"


def test_low_margin_returns_other():
    text = _unit([1.0, 0.0, 0.0])
    a = _unit([0.99, 0.01, 0.0])
    b = _unit([0.98, 0.02, 0.0])
    protos = {"A": [a], "B": [b]}
    result = score(text, protos, min_score=0.1, min_margin=0.1)
    assert result.predicted_label == "OTHER"
    assert result.margin < 0.1  # confirm margin is genuinely small
    assert result.reject_reason == "margin"


def test_per_label_threshold_override_allows_near_miss():
    text = _unit([1.0, 0.0, 0.0])
    protos = {"AI": [_unit([0.37, 0.93, 0.0])], "BUSINESS": [_unit([0.0, 1.0, 0.0])]}
    result = score(
        text,
        protos,
        min_score=0.4,
        min_margin=0.1,
        min_score_by_label={"AI": 0.35},
        min_margin_by_label={"AI": 0.0},
    )
    assert result.predicted_label == "AI"
    assert result.applied_min_score == 0.35
    assert result.applied_min_margin == 0.0
    assert result.reject_reason is None


def test_pair_specific_margin_override_allows_close_labels():
    text = _unit([1.0, 0.0, 0.0])
    protos = {
        "BUSINESS": [_unit([0.80, 0.60, 0.0])],
        "AI": [_unit([0.77, 0.64, 0.0])],
    }
    result = score(
        text,
        protos,
        min_score=0.4,
        min_margin=0.1,
        min_margin_by_label={"BUSINESS": 0.08},
        min_margin_by_pair={"BUSINESS": {"AI": 0.02}},
    )
    assert result.top_label == "BUSINESS"
    assert result.predicted_label == "BUSINESS"
    assert result.margin < 0.08
    assert result.margin >= 0.02
    assert result.applied_min_margin == 0.02


def test_empty_prototypes_returns_other():
    text = _unit([1.0, 0.0, 0.0])
    result = score(text, {}, min_score=0.0)
    assert result.predicted_label == "OTHER"
    assert result.top_score == 0.0


# --- Label selection ---


def test_correct_label_selected():
    text = _unit([1.0, 0.0, 0.0])
    protos = {
        "A": [_unit([1.0, 0.0, 0.0])],  # sim=1.0
        "B": [_unit([0.0, 1.0, 0.0])],  # sim=0.0
    }
    result = score(text, protos, min_score=0.0)
    assert result.top_label == "A"
    assert result.second_label == "B"
    assert result.top_score > result.second_score


# --- Aggregation ---


def test_max_aggregation_picks_best_prototype():
    """With max, one strong prototype wins over a uniform mean."""
    text = _unit([1.0, 0.0, 0.0])
    strong = _unit([1.0, 0.0, 0.0])  # sim=1.0
    weak = _unit([0.0, 1.0, 0.0])  # sim=0.0
    medium = _unit([0.7, 0.7, 0.0])  # sim≈0.707
    protos = {"A": [strong, weak], "B": [medium]}

    result = score(text, protos, min_score=0.0, aggregation="max")
    # max(A)=1.0, max(B)≈0.707 → A wins
    assert result.top_label == "A"


def test_mean_aggregation_averages_prototypes():
    """With mean, two mediocre prototypes can lose to one good one."""
    text = _unit([1.0, 0.0, 0.0])
    strong = _unit([1.0, 0.0, 0.0])  # sim=1.0
    weak = _unit([0.0, 1.0, 0.0])  # sim=0.0
    medium = _unit([0.7, 0.7, 0.0])  # sim≈0.707
    protos = {"A": [strong, weak], "B": [medium]}

    result = score(text, protos, min_score=0.0, aggregation="mean")
    # mean(A)=0.5, mean(B)≈0.707 → B wins
    assert result.top_label == "B"


def test_unknown_aggregation_raises():
    text = _unit([1.0, 0.0, 0.0])
    protos = {"A": [_unit([1.0, 0.0, 0.0])]}
    with pytest.raises(ValueError, match="Unknown aggregation"):
        score(text, protos, min_score=0.0, aggregation="median")
