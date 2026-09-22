"""Build `train_items.pt` for the RLCD fine-tune from our synthetic dataset.

The trainer (`training/train_ddp.py`) is the maintainer's, unmodified. This is the only part that
is ours: turning a labelled utterance into the sequence/target pair Laya's trainer expects.

Each training row yields **two** typed questions:

  department  a choice over all nine departments
  intent      a choice over the branch belonging to the *gold* department

which mirrors exactly the two questions the cascade asks at inference. Targets are one-hot;
calibration is handled afterwards by the trainer's temperature fitting step.

State is the caller's utterance on its own — the turn-1 distribution, which is where the
department decision is actually made (later turns skip it once the fact is pinned).

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


def load_taxonomy():
    with open(os.path.join(HERE, "taxonomy.json")) as fh:
        tax = json.load(fh)
    return tax["departments"], tax["intents"], tax["department_question"], tax["intent_question_template"]


def build_item(tok, cfg, state, qtype, instructions, criteria, gold):
    """One (sequence, target) pair, or None if the option budget cannot fit the question."""
    if qtype == "choice":
        keys = list(criteria.keys())
        if gold not in keys:
            return None
        target = [1.0 if k == gold else 0.0 for k in keys]
    else:
        raise ValueError(f"unsupported question type {qtype!r}")

    q = {"t": qtype, "ins": instructions, "crit": criteria}
    n_options = len(render_options(q))
    seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != n_options:
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[qtype],
        "target": target,
        "label": target.index(max(target)),
    }


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/calls/synthetic.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/working/train_items.pt"

    departments, intents, dept_q, intent_tpl = load_taxonomy()

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
    for row in rows:
        dept, intent, text = row["department"], row["intent"], row["text"]
        # The state the cascade actually passes at inference: the caller's own words.
        state = text

        it = build_item(
            tok, cfg, state, "choice", dept_q["instructions"], departments, dept
        )
        if it is None:
            skipped += 1
        else:
            it["task"] = "department"
            items.append(it)

        branch = intents.get(dept, {})
        if branch:
            instructions = intent_tpl.format(department=dept.replace("_", " "))
            it = build_item(tok, cfg, state, "choice", instructions, branch, intent)
            if it is None:
                skipped += 1
            else:
                it["task"] = "intent"
                items.append(it)

    print(f"built {len(items)} training sequences ({skipped} skipped for option-budget/unknown label)")
    from collections import Counter

    print("by task:", dict(Counter(i["task"] for i in items)))

    torch.save(items, dst)
    print(f"saved {dst}")

    meta = {
        "source": src,
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
