"""Versioned split contracts for generated and hand-authored decision data.

The contract is intentionally stricter than the historical JSONL rows: every row names its split,
source group, case family, and label provenance. That prevents a random row split from leaking a
paraphrase family across train, calibration, and test boundaries.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence

from . import store_profile as SP

SPLITS = ("train", "dev_select", "dev_calibrate", "frozen_test", "ood")
REQUIRED_FIELDS = (
    "id",
    "text",
    "destination",
    "subqueue",
    "split",
    "source_group",
    "case_family_id",
    "label_source",
    "taxonomy_hash",
)


def validate_case(row: Mapping[str, Any], *, allowed_splits: Iterable[str] = SPLITS) -> None:
    missing = [key for key in REQUIRED_FIELDS if key not in row]
    if missing:
        raise ValueError(f"case {row.get('id', '?')!r} is missing fields: {missing}")
    if row["split"] not in set(allowed_splits):
        raise ValueError(f"case {row['id']!r} has unknown split {row['split']!r}")
    if not str(row["id"]).strip() or not str(row["text"]).strip():
        raise ValueError(f"case {row['id']!r} needs a non-empty id and text")
    if not str(row["source_group"]).strip() or not str(row["case_family_id"]).strip():
        raise ValueError(f"case {row['id']!r} needs source_group and case_family_id")
    if not str(row["label_source"]).strip() or not str(row["taxonomy_hash"]).strip():
        raise ValueError(f"case {row['id']!r} needs label and taxonomy provenance")

    profile = SP.load()
    destination = row["destination"]
    if destination not in profile.destination_keys:
        raise ValueError(f"case {row['id']!r} has unknown destination {destination!r}")
    subqueue = row["subqueue"]
    if subqueue is not None and profile.subqueue(destination, subqueue) is None:
        raise ValueError(
            f"case {row['id']!r}: subqueue {subqueue!r} is not under destination {destination!r}"
        )


def validate_split_rows(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Validate rows and prove IDs/families/source groups do not cross split boundaries."""
    unknown = set(rows_by_split) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")
    seen: Dict[str, tuple[str, str]] = {}
    families: Dict[str, str] = {}
    groups: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for split, rows in rows_by_split.items():
        counts[split] = len(rows)
        for row in rows:
            validate_case(row, allowed_splits=(split,))
            case_id = str(row["id"])
            prior = seen.get(case_id)
            if prior is not None:
                raise ValueError(f"case id {case_id!r} appears in splits {prior[0]!r} and {split!r}")
            seen[case_id] = (split, "id")
            for field, store in (("case_family_id", families), ("source_group", groups)):
                value = str(row[field])
                prior_split = store.get(value)
                if prior_split is not None and prior_split != split:
                    raise ValueError(
                        f"{field} {value!r} crosses splits {prior_split!r} and {split!r}"
                    )
                store[value] = split
    return {"counts": counts, "cases": len(seen), "families": len(families), "source_groups": len(groups)}


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def duplicate_texts(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: Dict[str, str] = {}
    duplicates = []
    for row in rows:
        text = normalized_text(str(row["text"]))
        if text in seen:
            duplicates.append(str(row["id"]))
        else:
            seen[text] = str(row["id"])
    return duplicates
