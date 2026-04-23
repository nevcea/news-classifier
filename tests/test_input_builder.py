import pytest

from app.pipeline.input_builder import _MAX_INPUT_LEN, build_input


def test_title_only_ignores_summary():
    item = {"title": "Hello World", "summary": "Should be ignored"}
    text, used = build_input(item, mode="title_only")
    assert text == "Hello World"
    assert used is False


def test_title_plus_summary_concatenates():
    item = {"title": "Hello", "summary": "World"}
    text, used = build_input(item, mode="title_plus_summary")
    assert text == "Hello World"
    assert used is True


def test_title_plus_summary_no_summary():
    item = {"title": "Hello", "summary": ""}
    text, used = build_input(item, mode="title_plus_summary")
    assert text == "Hello"
    assert used is False


def test_missing_fields_default_to_empty():
    text, used = build_input({}, mode="title_plus_summary")
    assert text == ""
    assert used is False


def test_none_fields_handled():
    item = {"title": None, "summary": None}
    text, used = build_input(item, mode="title_plus_summary")
    assert text == ""
    assert used is False


def test_truncation_applied():
    long_title = "A" * (_MAX_INPUT_LEN + 100)
    item = {"title": long_title, "summary": ""}
    text, _ = build_input(item, mode="title_only")
    assert len(text) == _MAX_INPUT_LEN


def test_combined_truncation():
    item = {"title": "T" * 300, "summary": "S" * 300}
    text, used = build_input(item, mode="title_plus_summary")
    assert len(text) <= _MAX_INPUT_LEN
    assert used is True


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown input mode"):
        build_input({"title": "x"}, mode="full_text")
