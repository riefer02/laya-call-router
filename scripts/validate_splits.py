"""Validate versioned train/development/calibration/test JSONL splits.

Example:

    uv run python scripts/validate_splits.py \\
        --split train=data/calls/synthetic_v2/routing_train.jsonl \\
        --split frozen_test=data/calls/synthetic_v2/routing_frozen_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev_classifier.splits import validate_split_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append", required=True, help="NAME=path.jsonl")
    args = ap.parse_args()
    rows_by_split = {}
    for item in args.split:
        if "=" not in item:
            raise SystemExit(f"--split must be NAME=path, got {item!r}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if name in rows_by_split:
            raise SystemExit(f"split {name!r} was supplied twice")
        if not path.is_file():
            raise SystemExit(f"split file does not exist: {path}")
        rows_by_split[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = validate_split_rows(rows_by_split)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
