# Fine-tuning Laya for dealership routing

The training set is capped by its labels, so the pipeline has a gate at each end: the teacher must
agree with held-out human labels before anything is generated, and the dataset must pass its
acceptance criteria before it is used.

## What's here

| file | role |
|---|---|
| `train_ddp.py` | the **maintainer's** RLCD trainer, vendored unmodified from the official notebook (Apache-2.0) |
| `build_items.py` | ours — turns a labelled utterance into the two typed questions the cascade asks |
| `taxonomy.json` | generated from `dealership.py`; `tests/test_taxonomy.py` fails if they drift |
| `dump_taxonomy.py` | regenerates the above |
| `make_kaggle_dataset.py` | packages the data + code as an uploadable Kaggle dataset |
| `make_notebook.py` | generates the Kaggle notebook |

## Run it

### Automated (recommended)

With the Kaggle CLI installed and a token at `~/.kaggle/access_token`:

```bash
uv tool install kaggle                 # once
uv run python training/kaggle_run.py submit --watch
```

That packages the dataset, creates/versions it, pushes the kernel as a **GPU batch run**, waits,
and downloads the model. Pushing a kernel *runs* it and consumes GPU quota — the script says so
before it does it.

### Manual

**1. Package the dataset**

```bash
uv run python training/make_kaggle_dataset.py --owner YOUR_KAGGLE_USERNAME
```

**2. Upload it** — drag `kaggle/jev-dealership-data/` into <https://kaggle.com/datasets/new>, or
`kaggle datasets create -p kaggle/jev-dealership-data`.

**3. Open the notebook** — `notebooks/laya_finetune_dealership_kaggle.ipynb` on Kaggle, set
**Accelerator: GPU T4 ×2** and **Internet: On**, attach the dataset via *Add Data*.

**4. Run all cells.** Roughly 5–20 minutes.

### ⚠️ Accelerators require phone verification

If the notebook fails at the first cell with *"No GPU is attached to this run"*, the account is
almost certainly not phone-verified — Kaggle requires it for GPUs, and it does **not** surface as
an error when you push; the kernel just starts without an accelerator. Check your quota
(`uv run python training/kaggle_run.py watch` prints the diagnosis) — if it shows hours available,
verification is the blocker. Verify at <https://www.kaggle.com/settings>, then re-run.

The notebook tolerates a single GPU (`--nproc_per_node` follows the device count), so a single T4
works — just slower.

**5. Evaluate it properly, locally**

```bash
unzip laya-dealership-routing.zip -d models/
uv run python scripts/eval.py --finetuned models/laya-dealership-routing
```

That scores the fine-tuned cascade against the **same held-out 81**, alongside the base cascade and
both LLM arms. One harness, one test set — the number that matters comes from the code you already
trust, not from a notebook re-implementation.

## Why the trainer is untouched

`train_ddp.py` is the maintainer's proven RLCD implementation: proper-scoring-rule rewards, GRPO
baselines, DDP across both T4s, rolling checkpoints, and post-training temperature fitting.
Rewriting it would only add ways to be wrong. Everything project-specific lives in `build_items.py`.

The one thing worth knowing about the data shape: each utterance produces **two** training
questions — `department` (9 options) and `intent` (the branch for the gold department) — mirroring
exactly what the cascade asks at inference. Targets are one-hot; calibration is the trainer's job.

## Known limitations

- **State is the caller's first utterance only.** The cascade re-asks `department` on later turns
  with a longer transcript; training does not cover that distribution. In practice department is
  settled on turn 1 and then pinned, so the exposed case is small.
- **`general` is the thinnest department** (44 examples, one specific intent). If real traffic is
  heavy on opening-hours/location calls, revisit it.
- **9% of utterances name their own department**, which may make the task slightly easier than a
  real switchboard.
- **Residual `other` intents are deliberately not gated** — they are learned as the branch
  fallback. See the README for why generating positives for them is the wrong move.
