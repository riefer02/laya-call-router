# Train a typed-decision model for another domain

The bundled [v7 checkpoint](../models/kaggle-out-v7/laya-dealership-routing/README.md) is ready
for inference. This directory records how the dealership fine-tune was prepared. Its reusable
pattern is: define answer choices, label examples, audit the training set against held-out cases,
train, then test both individual decisions and complete conversations.

## What is here

| File | Role |
| --- | --- |
| `build_items.py` | Converts labelled utterances into the five typed training questions used by the application. |
| `train_ddp.py` | The Laya RLCD trainer used in the Kaggle job. |
| `run_config.json` | Recorded epoch and learning-rate settings. |
| `make_kaggle_dataset.py` | Packages training files, profile, and code for Kaggle. |
| `make_notebook.py` | Builds the notebook from the recorded recipe. |
| `kaggle_run.py` | Submits and watches a Kaggle run when configured. |

The [store profile](../config/store_profile.json) defines the destination choices and question
wording in one place. `build_items.py` reads it directly, so training and inference use the same
questions. The data files are under `data/calls/`; generated candidates must pass the checks in
`scripts/generate_training.py` and `scripts/audit_snapshots.py` before their scores are cited.

## Phase-A training contract

The trainer now supports an honest split boundary without changing the RLCD objective:

- `JEV_SEED` seeds Python, NumPy, Torch, and CUDA;
- `JEV_MAX_LEN` and `JEV_HEAD_MAX_LEN` are shared by the item builder and trainer;
- `JEV_CALIBRATION_ITEMS` points at calibration items and is required when
  `JEV_REQUIRE_SPLITS=1`;
- `JEV_VALIDATION_ITEMS` points at validation items used for per-epoch proper scores and
  `checkpoint_best`;
- `JEV_STRICT_BUILD=1` refuses silently skipped training items;
- every run writes `run_manifest.json` with input hashes, task counts, seed, runtime, and the
  calibration/validation source.

The legacy fallback still samples calibration items from training data when no calibration file
is supplied, but it prints a warning and records that fallback in the manifest. A future GPU run
should set `JEV_REQUIRE_SPLITS=1`; otherwise the fallback is not an honest calibration result.


The current v7 weights are already bundled. Retraining requires a GPU environment, the training
data, and the upstream Laya dependencies. Inspect `training/run_config.json` and the generated notebook before launching a job. Set
`base_model_revision` to an immutable Hugging Face commit for a reproducible new run. A Kaggle
submission starts compute and may consume quota.

```bash
uv run python training/make_kaggle_dataset.py --owner YOUR_KAGGLE_USERNAME
uv run python training/make_notebook.py
```

Upload `kaggle/jev-dealership-data/` and run
`notebooks/laya_finetune_dealership_kaggle.ipynb` with a suitable GPU. If your Kaggle account is
configured, `uv run python training/kaggle_run.py submit --watch` automates submission and
download. The frozen v7 training files and their SHA-256 manifest are under
`models/kaggle-out-v7/` so you can inspect what that job saw. The main
`data/calls/severity_train.jsonl` has since changed; package the frozen snapshot if you want to
repeat v7 rather than train on today's data. GPU execution can still produce a different
checkpoint from the same inputs.

Once you have a checkpoint, point `JEV_MODEL` at its directory and run:

```bash
uv run python scripts/eval.py --skip-llm --finetuned PATH_TO_CHECKPOINT --out /tmp/jev-eval.json
uv run python scripts/check_demo.py
```

For another domain, change the profile and labelled data first. Check the teacher against human
labels before generating examples, keep a separate held-out set, inspect the packaged snapshot for
overlap, and judge the full conversation as well as the classifier's answer. The
[root README](../README.md) walks through those steps with the dealership example.
