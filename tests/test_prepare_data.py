import pytest

from src.prepare_data import validate_examples


def test_validate_examples_strips_text():
    result = validate_examples([{"prompt": " p ", "chosen": " c ", "rejected": " r "}])
    assert result == [{"prompt": "p", "chosen": "c", "rejected": "r"}]


def test_validate_examples_rejects_missing_field():
    with pytest.raises(ValueError, match="missing required fields"):
        validate_examples([{"prompt": "p", "chosen": "c"}])


def test_validate_examples_rejects_empty_dataset():
    with pytest.raises(ValueError, match="empty"):
        validate_examples([])
