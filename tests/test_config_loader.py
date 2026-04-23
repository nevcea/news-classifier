from unittest.mock import patch

from app.settings import loader as config_loader


def test_load_categories_returns_per_label_threshold_overrides():
    config = {
        "categories": {
            "AI": {"prototypes": ["model release"]},
            "BUSINESS": {"prototypes": ["funding round"]},
        },
        "thresholds": {
            "min_score": 0.4,
            "min_margin": 0.1,
            "min_score_by_label": {"AI": 0.37, "UNKNOWN": 0.2},
            "min_margin_by_label": {"AI": 0.07, "BUSINESS": 0.08},
            "min_margin_by_pair": {
                "BUSINESS": {"AI": 0.04, "UNKNOWN": 0.01},
                "UNKNOWN": {"AI": 0.02},
            },
        },
    }

    with patch("app.settings.loader._load_yaml", return_value=config):
        result = config_loader.load_categories()

    assert result["min_score"] == 0.4
    assert result["min_margin"] == 0.1
    assert result["min_score_by_label"] == {"AI": 0.37}
    assert result["min_margin_by_label"] == {"AI": 0.07, "BUSINESS": 0.08}
    assert result["min_margin_by_pair"] == {"BUSINESS": {"AI": 0.04}}
