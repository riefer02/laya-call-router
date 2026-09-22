"""Generate the Kaggle fine-tuning notebook.

Kept as a generator rather than a checked-in .ipynb so the cells stay readable and reviewable in
a diff — a raw notebook is a wall of escaped JSON.

    uv run python training/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebooks" / "laya_finetune_dealership_kaggle.ipynb"

MD_HEADER = """# Fine-tuning Laya for dealership call routing

Adapts the official Laya RLCD fine-tuning notebook (`NandhaKishorM/laya`, Apache-2.0) to this
project's dealership routing task. The training script itself is **unchanged** — only the data
preparation differs.

**Notebook settings (right sidebar → Notebook options):**
* Accelerator: **GPU T4 x2**
* Internet: **On**

**Before running:** upload this repo's `training/` folder and `data/calls/synthetic.jsonl` as a
Kaggle dataset and attach it (Add Data → Your datasets). The cell below looks for them in the
usual mount points and tells you if it cannot find them.
"""

MD_DATA = """## 1. Locate the uploaded data

The dataset needs `store_profile.json`, `build_items.py` and `synthetic.jsonl`. We search the common
Kaggle mount points rather than hard-coding a slug.
"""

MD_BUILD = """## 2. Build training items

Turns each labelled utterance into sequences for the two questions the cascade asks at inference:
`department` (9 options) and `intent` (the branch for the gold department). Unmodified
`train_ddp.py` then consumes `train_items.pt`.

If the option-budget warning below fires, some rows were skipped — that is the `head_max_len`
constraint, and it is reported rather than hidden.
"""

MD_TRAIN = """## 3. Fine-tune (RLCD, DDP across both T4s)

The maintainer's training script, vendored verbatim at `training/train_ddp.py`. ~4 epochs; expect
roughly 5–20 minutes depending on dataset size.
"""

MD_PACKAGE = """## 4. Package the result

Zips the fine-tuned checkpoint so it can be downloaded and evaluated **locally** with the project's
own harness — `scripts/eval.py` — rather than re-implementing the evaluation here. One eval
harness, one source of truth.
"""

MD_NEXT = """## Next: evaluate it properly

1. Download `laya-dealership-routing.zip` from the notebook output.
2. Unzip it locally into `models/laya-dealership-routing/`.
3. Run the existing evaluation against it:

```bash
uv run python scripts/eval.py --limit 0 --determinism 1
```

The held-out 81 hand-labelled cases were **never** in this training set — `build_items.py` only
reads `synthetic.jsonl`, and the generator's deduper refuses anything within Jaccard 0.6 of them.
"""


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


CHECK_GPU = '''!nvidia-smi || echo "nvidia-smi unavailable"
import os, torch

n_gpu = torch.cuda.device_count()
print(f"CUDA available: {torch.cuda.is_available()} | visible GPUs: {n_gpu}")
assert n_gpu >= 1, (
    "No GPU is attached to this run.\\n"
    "  - Set Accelerator to 'GPU T4 x2' (or a single T4) in the right sidebar, and\\n"
    "  - make sure your Kaggle account is phone-verified, which is required for accelerators.\\n"
    "If you pushed this kernel with the CLI, also pass --accelerator."
)
if n_gpu < 2:
    print("!! only 1 GPU - training will run single-process, which is slower but works")

# DDP across whatever we actually got, so a single-T4 run still completes.
NPROC = max(1, n_gpu)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(f"ready: {n_gpu} GPU(s), torchrun nproc_per_node={NPROC}")
'''

INSTALL = '''!pip install -q -U "laya>=0.1.6" "transformers>=4.48.0" safetensors huggingface_hub pyarrow pandas scipy accelerate
import laya, transformers, torch
print("laya", laya.__version__, "| transformers", transformers.__version__, "| torch", torch.__version__)
'''

FIND_DATA = '''import os, glob, shutil

WANTED = ("synthetic.jsonl", "store_profile.json", "build_items.py", "train_ddp.py")

def log_tree(root, limit=40):
    print(f"  tree under {root}:")
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            p = os.path.join(dirpath, f)
            print(f"    {os.path.relpath(p, root)}  {os.path.getsize(p):,} bytes")
            n += 1
            if n >= limit:
                print("    ...")
                return
    if n == 0:
        print("    (empty)")

print("=== /kaggle/input ===")
if os.path.isdir("/kaggle/input"):
    entries = sorted(os.listdir("/kaggle/input"))
    print("  entries:", entries or "(none)")
    for e in entries:
        log_tree(os.path.join("/kaggle/input", e), limit=10)
else:
    print("  /kaggle/input does not exist")

# Find the files anywhere under /kaggle/input rather than assuming a mount layout.
found_dir = None
for dirpath, _, filenames in os.walk("/kaggle/input"):
    if all(w in filenames for w in WANTED):
        found_dir = dirpath
        break

if found_dir is None:
    raise SystemExit(
        "Could not find all of {0} under /kaggle/input.\\n"
        "Attach the dataset to this notebook (Add Data -> andrewriefenstahl/"
        "jev-dealership-routing-data) and re-run.".format(WANTED)
    )

print(f"\\nusing data from {found_dir}")
for name in WANTED:
    shutil.copy(os.path.join(found_dir, name), os.path.join("/kaggle/working", name))
    print("  copied", name)

n = sum(1 for _ in open("/kaggle/working/synthetic.jsonl"))
print(f"\\n{n} labelled training utterances")
'''

BUILD = '''!cd /kaggle/working && python build_items.py /kaggle/working/synthetic.jsonl /kaggle/working/train_items.pt
'''

TRAIN = '''MODEL_DIR = "/kaggle/working/laya_base"
import os
from huggingface_hub import snapshot_download
snapshot_download("convaiinnovations/laya", local_dir=MODEL_DIR)
print("base checkpoint at", MODEL_DIR)

OUTPUT_DIR = "/kaggle/working/laya-dealership-routing"
cmd = (f"torchrun --standalone --nproc_per_node={NPROC} /kaggle/working/train_ddp.py "
       f"{MODEL_DIR} {OUTPUT_DIR} /kaggle/working/train_items.pt")
print("running:", cmd)
!{cmd}
'''

PACKAGE = '''import os, shutil
OUTPUT_DIR = "/kaggle/working/laya-dealership-routing"
assert os.path.isdir(OUTPUT_DIR), f"{OUTPUT_DIR} does not exist - training did not finish"
shutil.make_archive("/kaggle/working/laya-dealership-routing", "zip", OUTPUT_DIR)
for root, _, files in os.walk(OUTPUT_DIR):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"{os.path.relpath(p, OUTPUT_DIR):50s} {os.path.getsize(p):>12,} bytes")
print("\\nzip ready: /kaggle/working/laya-dealership-routing.zip")
'''


def main() -> None:
    nb = {
        "cells": [
            md(MD_HEADER),
            code(CHECK_GPU),
            code(INSTALL),
            md(MD_DATA),
            code(FIND_DATA),
            md(MD_BUILD),
            code(BUILD),
            md(MD_TRAIN),
            code(TRAIN),
            md(MD_PACKAGE),
            code(PACKAGE),
            md(MD_NEXT),
        ],
        "metadata": {
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
