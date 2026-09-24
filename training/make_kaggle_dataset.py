"""Package the training data + training code as an uploadable Kaggle dataset.

Kaggle notebooks can't read this repo, so everything the notebook needs is copied into one folder
with a `dataset-metadata.json`. Then either upload it through the Kaggle UI, or:

    kaggle datasets create -p kaggle/jev-dealership-data

    uv run python training/make_kaggle_dataset.py --owner YOUR_KAGGLE_USERNAME
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "jev-dealership-data"

FILES = [
    ("data/calls/synthetic.jsonl", "synthetic.jsonl"),
    ("data/calls/severity_train.jsonl", "severity_train.jsonl"),
    ("data/calls/acceptance_train.jsonl", "acceptance_train.jsonl"),
    ("config/store_profile.json", "store_profile.json"),
    ("training/build_items.py", "build_items.py"),
    ("training/train_ddp.py", "train_ddp.py"),
    ("training/validation.py", "validation.py"),
    ("training/run_config.json", "run_config.json"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME", "YOUR_KAGGLE_USERNAME"))
    ap.add_argument("--slug", default="jev-dealership-routing-data")
    args = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    total = 0
    for src_rel, dst_name in FILES:
        src = ROOT / src_rel
        if not src.is_file():
            raise SystemExit(
                f"missing {src_rel}\n"
                "run: uv run python scripts/generate_training.py --per-subqueue 50"
            )
        shutil.copy(src, OUT / dst_name)
        size = (OUT / dst_name).stat().st_size
        total += size
        print(f"  {dst_name:20s} {size:>10,} bytes")

    rows = sum(1 for _ in open(OUT / "synthetic.jsonl"))
    meta = {
        "title": "Laya Call Router Dealership Routing Data",
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "apache-2.0"}],
        "description": (
            f"{rows} teacher-labelled dealership call utterances for fine-tuning Laya "
            "(destinations + sub-queues), plus the RLCD training code. Labels were produced by "
            "deepseek-flash and kept only where two independently-worded labelling passes agreed "
            "with each other and the intended target."
        ),
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\n{rows} training rows, {total/1024:.0f} KB total")
    print(f"wrote {OUT}")
    print("\nUpload it:")
    print(f"  kaggle datasets create -p {OUT.relative_to(ROOT)}")
    print("  (or drag the folder into kaggle.com/datasets/new)")
    if args.owner == "YOUR_KAGGLE_USERNAME":
        print("\n!! dataset id still has a placeholder owner — rerun with --owner YOUR_KAGGLE_USERNAME")


if __name__ == "__main__":
    main()
