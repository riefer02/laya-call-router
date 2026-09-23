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


def build_noul_item(tok, cfg, state, instructions, criteria, value):
    """One yes/no item. `target` is [P(false), P(true)] and there are always two markers.

    Mirrors the reference implementation in the model bundle's own `rl_common.encode_record`: a
    noul is encoded as a two-option choice between "false: ..." and "true: ...", which is why the
    option text is what actually defines the question.
    """
    q = {"t": "noul", "ins": instructions, "crit": criteria}
    ids, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != 2:
        return None
    y = 1.0 if value else 0.0
    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES["noul"],
        "target": [1.0 - y, y],
        "label": int(y),
    }


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/calls/synthetic.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/working/train_items.pt"

    profile, profile_path = load_profile()
    destinations = {d["key"]: d.get("description", "") for d in profile["destinations"]}
    subqueues = {}
    for sub in profile.get("subqueues", []):
        subqueues.setdefault(sub["parent"], {})[sub["key"]] = sub.get("description", "")
    questions = profile.get("questions") or {}
    if "destination" not in questions or "subqueue" not in questions:
        raise SystemExit(
            "store_profile.json has no `questions` block. The training and inference instructions "
            "must be the same string, so they are defined once in the profile rather than copied "
            "into this script (a copy drifted once already). Add:\n"
            '  "questions": {"destination": "...", "subqueue": "This is a {label} call. ..."}\n'
            f"profile: {profile_path}"
        )
    destination_instructions = questions["destination"]
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

    # ---- yes/no items, if we have severity labels for them
    # These were never trained before, which is why `is_safe_to_drive` missed a fifth of the
    # stranded callers it was supposed to dispatch.
    sev_src = os.environ.get("JEV_SEVERITY") or os.path.join(os.path.dirname(src), "severity_train.jsonl")
    noul_spec = profile.get("noul") or {}
    noul_rows: list = []
    if os.path.isfile(sev_src):
        noul_rows = [json.loads(line) for line in open(sev_src) if line.strip()]
        print(f"loaded {len(noul_rows)} severity-labelled utterances from {sev_src}")
    else:
        # A missing file used to print a note and carry on, which is how a run trained 2,512 items
        # instead of 6,210 and reported nothing wrong. The dataset ships these files, so their
        # absence means the notebook did not copy them. That is a failure, not a smaller job.
        if not os.environ.get("JEV_ALLOW_CHOICE_ONLY"):
            raise SystemExit(
                f"no severity data at {sev_src}, but the dataset ships it. The notebook probably "
                "did not copy it - which is how a run silently trained choice questions only. "
                "Set JEV_ALLOW_CHOICE_ONLY=1 to train choice questions deliberately."
            )
        print(f"no severity data at {sev_src} - training choice questions only (explicitly allowed)")

    for row in noul_rows:
        for key, spec in noul_spec.items():
            if key not in row:
                continue
            crit = {"false": spec.get("false", ""), "true": spec.get("true", "")}
            it = build_noul_item(
                tok, cfg, row["text"], spec.get("instructions", key), crit, row[key]
            )
            if it is None:
                skipped += 1
            else:
                it["task"] = key
                items.append(it)

    # ---- acceptance items: the only question whose state is a whole conversation
    acc_src = os.environ.get("JEV_ACCEPTANCE") or os.path.join(
        os.path.dirname(src), "acceptance_train.jsonl"
    )
    if os.path.isfile(acc_src) and "acceptance" in questions:
        acc_rows = [json.loads(line) for line in open(acc_src) if line.strip()]
        print(f"loaded {len(acc_rows)} acceptance examples from {acc_src}")
        for row in acc_rows:
            it = build_item(
                tok, cfg, row["text"], questions["acceptance"], row["options"], row["choice"]
            )
            if it is None:
                skipped += 1
            else:
                it["task"] = "acceptance"
                items.append(it)
    else:
        if not os.environ.get("JEV_ALLOW_CHOICE_ONLY"):
            raise SystemExit(
                f"no acceptance data at {acc_src}, but the dataset ships it. The notebook probably "
                "did not copy it. Set JEV_ALLOW_CHOICE_ONLY=1 to skip it deliberately."
            )
        print(f"no acceptance data at {acc_src} - the booking flow stays untrained (explicitly allowed)")

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
            name = next(
                (
                    d.get("label", d["key"])
                    for d in profile["destinations"]
                    if d["key"] == destination
                ),
                destination,
            )
            instructions = questions["subqueue"].format(label=name)
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
