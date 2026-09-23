"""Audit every checkpoint's packaged training data against the eval sets, and record it.

A checkpoint is scored on cases its own snapshot may contain. Nothing in the pipeline noticed that
until a second reader checked v6 by hand: its snapshot held `gen-01` inside a routing row and
`sev-04` inside two severity rows, so 0.963 destination and 18-of-18 hazard recall were measured
partly on memorisation. The files were trimmed afterwards, which fixed the repository and not the
checkpoint.

Run this after every download and commit the manifest, so a number and its provenance travel
together:

    uv run python scripts/audit_snapshots.py            # report
    uv run python scripts/audit_snapshots.py --write     # record a manifest per checkpoint

Exits non-zero if any checkpoint is unclean, so it can gate a release rather than inform one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_classifier import snapshots  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="record a manifest per checkpoint")
    args = ap.parse_args()

    reports = snapshots.audit_all()
    if not reports:
        print("no checkpoint directories with packaged training data")
        return 0

    print(f"{'checkpoint':34s} {'rows':>14s}  held-out")
    unclean = []
    for report in reports:
        rows = "/".join(str(f["rows"]) for f in report["files"].values())
        verdict = "clean" if report["held_out_clean"] else f"{len(report['leaks'])} LEAK(S)"
        print(f"  {report['snapshot']:32s} {rows:>14s}  {verdict}")
        if not report["held_out_clean"]:
            unclean.append(report)
            for leak in report["leaks"][:4]:
                print(f"      {leak['eval_id']} ({leak['eval_file']}) inside {leak['train_file']}")
            if args.write:
                path = ROOT / "models" / report["snapshot"] / snapshots.MANIFEST
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(f"      recorded {path.relative_to(ROOT)}")
        elif args.write:
            path = ROOT / "models" / report["snapshot"] / snapshots.MANIFEST
            path.write_text(json.dumps(report, indent=2) + "\n")

    print()
    if unclean:
        print(
            f"{len(unclean)} checkpoint(s) trained on data containing the cases they are scored on. "
            "Their numbers need the caveat; the trim does not repair them."
        )
        return 1
    print("every checkpoint's packaged training data is free of the cases it is scored on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
