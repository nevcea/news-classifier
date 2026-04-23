from app.classifier.interface import ClassificationResult
from app.pipeline.formatter import format_all, format_item


def _result(**kwargs) -> ClassificationResult:
    defaults = {
        "predicted_label": "AI",
        "top_label": "AI",
        "second_label": "DEV",
        "top_score": 0.81234,
        "second_score": 0.4,
        "margin": 0.41234,
    }
    defaults.update(kwargs)
    return ClassificationResult(**defaults)


def test_format_item_includes_label_score_margin_and_source():
    item = {"title": "Hello World", "source": "https://example.com/feed"}

    line = format_item(item, _result())

    assert "[AI" in line
    assert "score=0.812" in line
    assert "margin=0.412" in line
    assert "Hello World" in line
    assert "https://example.com/feed" in line


def test_format_all_joins_lines():
    pairs = [
        ({"title": "One", "source": "a"}, _result(predicted_label="AI")),
        ({"title": "Two", "source": "b"}, _result(predicted_label="DEV", top_label="DEV")),
    ]

    output = format_all(pairs)

    assert "One" in output
    assert "Two" in output
    assert output.count("\n") == 1
