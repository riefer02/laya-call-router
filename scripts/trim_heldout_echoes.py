"""Remove training rows that echo a held-out evaluation case.

The generator's deduper refuses anything within Jaccard 0.6 of the 81 hand-labelled cases. That is
not enough, because a short query embedded in a longer sentence scores low on Jaccard: the union is
large, so "what time do you open on saturdays?" (7 words) against a 14-word training row scores
0.50 and passed the guard. The same shape hid `sev-04` in the severity data.

An eval case that is a *fragment* of a training row is not held out, and the exact-text check we
were using cannot see it. This applies the same containment rule the deduper now enforces, to
datasets that were generated before the rule existed, so the fix is reproducible rather than a
hand-edit.

    uv run python scripts/trim_heldout_echoes.py            # report only
    uv run python scripts/trim_heldout_echoes.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_classifier import synthgen  # noqa: E402

CALLS = ROOT / "data" / "calls"

# Every measured task, and the eval set its training data must not echo.
PAIRS = [
    ("synthetic.jsonl", "routing.jsonl"),
    ("severity_train.jsonl", "severity.jsonl"),
]


def load(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite the training files")
    args = ap.parse_args()

    total = 0
    for train_name, eval_name in PAIRS:
        train_path, eval_path = CALLS / train_name, CALLS / eval_name
        if not (train_path.is_file() and eval_path.is_file()):
            print(f"{train_name}: skipped (not generated)")
            continue
        rows = load(train_path)
        held_out = [r["text"] for r in load(eval_path)]
        kept = [r for r in rows if not synthgen.echoes_heldout(r["text"], held_out)]
        dropped = [r for r in rows if synthgen.echoes_heldout(r["text"], held_out)]
        total += len(dropped)
        print(f"{train_name}: {len(rows)} rows, {len(dropped)} echo a held-out case")
        for r in dropped:
            print("    drop:", r["text"][:96])
        if dropped:
            if not args.write:
                continue
            if not kept:
                raise SystemExit(f"refusing to write an empty {train_name}")
            tmp = train_path.with_suffix(train_path.suffix + ".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
            tmp.replace(train_path)
            print(f"    wrote {train_path.name} ({len(kept)} rows)")

    if not args.write and total:
        print(f"\n{total} rows would be dropped - re-run with --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
