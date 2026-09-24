"""Tests for the Phase-A split boundary contract."""

from __future__ import annotations

import pytest

from jev_classifier.splits import duplicate_texts, validate_case, validate_split_rows


def row(case_id: str, split: str, family: str, group: str, text: str = "hello") -> dict:
    return {
        "id": case_id,
        "text": text,
        "destination": "service",
        "subqueue": "mechanical_diagnostic",
        "split": split,
        "source_group": group,
        "case_family_id": family,
        "label_source": "contract",
        "taxonomy_hash": "tax-v1",
    }


def test_split_contract_accepts_disjoint_families():
    result = validate_split_rows(
        {
            "train": [row("a", "train", "family-a", "group-a")],
            "dev_select": [row("b", "dev_select", "family-b", "group-b")],
        }
    )
    assert result["cases"] == 2
    assert result["families"] == 2


def test_split_contract_rejects_a_family_crossing_train_and_test():
    with pytest.raises(ValueError, match="crosses splits"):
        validate_split_rows(
            {
                "train": [row("a", "train", "family-a", "group-a")],
                "frozen_test": [row("b", "frozen_test", "family-a", "group-b")],
            }
        )


def test_split_contract_rejects_duplicate_case_ids():
    with pytest.raises(ValueError, match="appears in splits"):
        validate_split_rows({"train": [row("a", "train", "a", "a")], "ood": [row("a", "ood", "b", "b")]})


def test_case_validation_rejects_invalid_subqueue():
    bad = row("a", "train", "a", "a")
    bad["subqueue"] = "lease_terms"
    with pytest.raises(ValueError, match="not under"):
        validate_case(bad)


def test_duplicate_text_helper_is_normalized():
    rows = [row("a", "train", "a", "a", " Hello   world "), row("b", "train", "b", "b", "hello world")]
    assert duplicate_texts(rows) == ["b"]
