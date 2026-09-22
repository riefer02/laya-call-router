"""Build `train_items.pt` for the RLCD fine-tune from our synthetic dataset.

The trainer (`training/train_ddp.py`) is the maintainer's, unmodified. This is the only part that
is ours: turning a labelled utterance into the sequence/target pairs Laya's trainer expects.

Each training row yields **two** typed questions, mirroring exactly what the cascade asks at
inference:

  destination   a choice over all destinations in the store profile
  subqueue      a choice over the sub-queues of the *gold* destination

Targets are one-hot; calibration is handled afterwards by the trainer's temperature fitting step.

State is the caller's utterance on its own — the turn-1 distribution, which is where the
destination decision is actually made (later turns skip it once the fact is pinned).

Runs inside the Kaggle notebook (it needs the `laya` package). Locally we only use `laya-mlx`.

    python training/build_items.py data/calls/synthetic.jsonl /kaggle/working/train_items.pt
"""

from __future__ import annotations

import json
import os
import sys

import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence, render_options
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.environ.get("LAYA_MODEL_ID", "convaiinnovations/laya")


def load_profile():
    """Read the store profile, preferring an explicit path so the notebook and the runtime cannot
    drift apart."""
    candidates = [
        os.environ.get("JEV_STORE_PROFILE"),
        "/kaggle/working/store_profile.json",
        os.path.join(HERE, "store_profile.json"),
        os.path.join(HERE, "..", "config", "store_profile.json"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return json.load(open(path)), path
    raise SystemExit(
        "could not find store_profile.json; pass JEV_STORE_PROFILE or copy it next to this script"
    )


def build_item(tok, cfg, state, instructions, criteria, gold):
    """One (sequence, target) pair, or None if the option budget cannot fit the question."""
    keys = list(criteria.keys())
    if gold not in keys:
        return None
    target = [1.0 if k == gold else 0.0 for k in keys]
    q = {"t": "choice", "ins": instructions, "crit": criteria}
    n_options = len(render_options(q))
    seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != n_options:
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES["choice"],
        "target": target,
        "label": target.index(max(target)),
    }


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/calls/synthetic.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/working/train_items.pt"

    profile, profile_path = load_profile()
    destinations = {d["key"]: d.get("description", "") for d in profile["destinations"]}
    subqueues = {}
    for sub in profile.get("subqueues", []):
        subqueues.setdefault(sub["parent"], {})[sub["key"]] = sub.get("description", "")
    destination_instructions = (
        "Which part of the dealership should handle this caller? Pick where the work belongs, "
        "not the first thing the caller mentioned."
    )
    print(f"store profile: {profile_path}  ({profile.get('name')})")
    print(f"  {len(destinations)} destinations, {sum(len(v) for v in subqueues.values())} sub-queues")

    print(f"fetching tokenizer/config from {MODEL_ID} ...")
    model_dir = snapshot_download(MODEL_ID)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as fh:
        cfg = json.load(fh)
    print(f"max_len={cfg['max_len']} head_max_len={cfg['head_max_len']}")

    rows = [json.loads(line) for line in open(src) if line.strip()]
    print(f"loaded {len(rows)} labelled utterances from {src}")

    items, skipped = [], 0
    from collections import Counter

    for row in rows:
        destination, subqueue, text = row["destination"], row.get("subqueue"), row["text"]
        state = text

        it = build_item(tok, cfg, state, destination_instructions, destinations, destination)
        if it is None:
            skipped += 1
        else:
            it["task"] = "destination"
            items.append(it)

        branch = subqueues.get(destination) or {}
        if branch and subqueue:
            label = profile["destinations"]
            name = next(
                (d.get("label", d["key"]) for d in label if d["key"] == destination), destination
            )
            instructions = (
                f"This is a {name.lower()} call. What exactly does the caller want, and which "
                "sub-queue should it go to? Pick the single closest option."
            )
            it = build_item(tok, cfg, state, instructions, branch, subqueue)
            if it is None:
                skipped += 1
            else:
                it["task"] = "subqueue"
                items.append(it)

    print(f"built {len(items)} training sequences ({skipped} skipped for option-budget/unknown label)")
    print("by task:", dict(Counter(i["task"] for i in items)))

    torch.save(items, dst)
    print(f"saved {dst}")

    meta = {
        "source": src,
        "store_profile": profile_path,
        "n_utterances": len(rows),
        "n_items": len(items),
        "skipped": skipped,
        "by_task": dict(Counter(i["task"] for i in items)),
        "model_id": MODEL_ID,
        "max_len": cfg["max_len"],
        "head_max_len": cfg["head_max_len"],
    }
    with open(os.path.join(os.path.dirname(dst), "train_items_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote train_items_meta.json")


if __name__ == "__main__":
    main()
