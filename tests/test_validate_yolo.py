from pathlib import Path

import pytest

from src.data.validate_yolo import validate_label_file


@pytest.mark.parametrize(
    "content, expected_class",
    [
        ("0 0.5 0.5 0.2 0.2\n", 0),
        ("1 0.5 0.5 0.2 0.2\n", 1),
        ("2 0.5 0.5 0.2 0.2\n", 2),
        ("3 0.5 0.5 0.2 0.2\n", 3),
    ],
)
def test_valid_labels(tmp_path: Path, content: str, expected_class: int):
    label_path = tmp_path / "valid.txt"
    label_path.write_text(content, encoding="utf-8")

    counts, errors = validate_label_file(label_path)

    assert errors == []
    assert counts[expected_class] == 1


@pytest.mark.parametrize(
    "content",
    [
        "4 0.5 0.5 0.2 0.2\n",       # Unknown class
        "0 1.5 0.5 0.2 0.2\n",       # Invalid x center
        "0 0.5 0.5 -0.2 0.2\n",      # Negative width
        "0 0.5 0.5 0.2\n",           # Missing coordinate
        "0 abc 0.5 0.2 0.2\n",       # Invalid number
        "0 nan 0.5 0.2 0.2\n",       # NaN
        "0 0.95 0.5 0.2 0.2\n",      # Box exceeds boundary
    ],
)
def test_invalid_labels(tmp_path: Path, content: str):
    label_path = tmp_path / "invalid.txt"
    label_path.write_text(content, encoding="utf-8")

    counts, errors = validate_label_file(label_path)

    assert len(errors) > 0
    assert sum(counts.values()) == 0


def test_empty_label_file_is_valid(tmp_path: Path):
    """An XML-annotated frame may legitimately contain no vehicles."""

    label_path = tmp_path / "empty.txt"
    label_path.write_text("", encoding="utf-8")

    counts, errors = validate_label_file(label_path)

    assert errors == []
    assert sum(counts.values()) == 0


def test_multiple_valid_labels(tmp_path: Path):
    label_path = tmp_path / "multiple.txt"

    label_path.write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "1 0.3 0.3 0.1 0.1\n"
        "0 0.7 0.7 0.1 0.1\n",
        encoding="utf-8",
    )

    counts, errors = validate_label_file(label_path)

    assert errors == []
    assert counts[0] == 2
    assert counts[1] == 1
    assert sum(counts.values()) == 3


def test_missing_label_file(tmp_path: Path):
    label_path = tmp_path / "missing.txt"

    counts, errors = validate_label_file(label_path)

    assert len(errors) > 0
    assert sum(counts.values()) == 0