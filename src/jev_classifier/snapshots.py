"""Did this checkpoint's training data contain the cases it is scored on?

A checkpoint cannot be called held out because *today's* files are clean. It trained on the snapshot
Kaggle packaged, and that snapshot sits on disk beside the weights. v6 was quoted at 0.963
destination and 18-of-18 hazard recall while its snapshot held `gen-01` verbatim inside a routing
row and `sev-04` inside two severity rows. The trim that removed them landed after the run, so it
repairs the repository and not the checkpoint.

Nothing caught that, because the leak test looked at `data/calls/` and the checkpoint looked at
`models/kaggle-out-v6/`. This module audits the snapshot itself, and `scripts/audit_snapshots.py`
records the result next to the weights so a run carries its own provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from . import synthgen

CALLS = Path(__file__).resolve().parents[2] / "data" / "calls"

# Which packaged training file is scored by which eval set.
PAIRS = (
    ("synthetic.jsonl", "routing.jsonl"),
    ("severity_train.jsonl", "severity.jsonl"),
)

# The acceptance split is by reply phrasing rather than by text, so it gets its own check.
ACCEPTANCE_PAIR = ("acceptance_train.jsonl", "acceptance_dev.jsonl")

MANIFEST = "snapshot_manifest.json"


def _load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_snapshot(snapshot: Path, calls: Path = CALLS) -> Dict[str, object]:
    """Everything a reader needs to know whether this snapshot's numbers are held out.

    Containment, not exact-text: a short eval case sitting inside a longer training row is a leak
    that no equality check sees, and it is the form both of v6's leaks took.
    """
    files: Dict[str, Dict[str, object]] = {}
    leaks: List[Dict[str, str]] = []

    for train_name, eval_name in PAIRS:
        train_path, eval_path = snapshot / train_name, calls / eval_name
        if not (train_path.is_file() and eval_path.is_file()):
            continue
        train = _load(train_path)
        files[train_name] = {"sha256": _sha256(train_path), "rows": len(train)}
        held_out = [_ for _ in _load(eval_path)]
        for case in held_out:
            for row in train:
                if synthgen.echoes_heldout(case.get("text", ""), [row.get("text", "")]):
                    leaks.append(
                        {
                            "eval_id": str(case.get("id", "?")),
                            "eval_file": eval_name,
                            "train_file": train_name,
                            "train_text": row.get("text", ""),
                        }
                    )

    train_name, eval_name = ACCEPTANCE_PAIR
    train_path, eval_path = snapshot / train_name, calls / eval_name
    if train_path.is_file() and eval_path.is_file():
        train, dev = _load(train_path), _load(eval_path)
        files[train_name] = {"sha256": _sha256(train_path), "rows": len(train)}
        shared = {r.get("reply_template") for r in train} & {r.get("reply_template") for r in dev}
        for phrase in sorted(shared):
            leaks.append(
                {
                    "eval_id": "reply-phrasing",
                    "eval_file": eval_name,
                    "train_file": train_name,
                    "train_text": str(phrase),
                }
            )

    return {
        "snapshot": snapshot.name,
        "files": files,
        "leaks": leaks,
        "held_out_clean": not leaks,
    }


def audit_all(models_dir: Path = None, calls: Path = CALLS) -> List[Dict[str, object]]:
    """Every checkpoint directory that ships training data, audited."""
    models_dir = models_dir or (Path(__file__).resolve().parents[2] / "models")
    out = []
    if not models_dir.is_dir():
        return out
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        if not any((child / name).is_file() for name, _ in PAIRS):
            continue
        out.append(audit_snapshot(child, calls))
    return out
