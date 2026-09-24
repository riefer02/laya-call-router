# Training and ML instructions

These instructions apply to the `training/` directory and to data/model experiments elsewhere in
this repository.

## Training contract

- The trainer is an RLCD-style typed-decision fine-tune, not a general text generator.
- The current objective combines a proper-score policy-gradient term with a supervised target
  cross-entropy anchor. Do not describe it as full online RL, PPO, or GRPO without qualification.
- `training/run_config.json` is the source of truth for the notebook recipe.
- Regenerate the notebook with `uv run python training/make_notebook.py`; do not hand-edit the
  generated `.ipynb`.
- `build_items.py` and `train_ddp.py` must use the same token budget. The current recipe records
  `max_len=512` and `head_max_len=192`.
- Set and record Python, NumPy, Torch, and CUDA seeds for every new run.
- Record base-model revision, dependency versions, GPU count, optimizer updates, task counts, data
  hashes, and selected checkpoint in the run manifest.
- A checkpoint is not reproducible merely because a final config exists. Check the actual log,
  input hashes, package versions, and runtime metadata.

## Data and evaluation boundaries

The following roles are distinct:

- `train`: gradient updates only;
- `dev_select`: checkpoint, data, and hyperparameter selection;
- `dev_calibrate`: temperature and threshold fitting;
- `frozen_test`: one final report;
- `ood`: separate out-of-domain and clarification behavior.

Split by case family and source group, not by individual generated rows. Do not let paraphrases,
template variants, or the same synthetic scenario cross a split boundary.

A future run should set:

```bash
JEV_REQUIRE_SPLITS=1
JEV_CALIBRATION_ITEMS=/path/to/calibration_items.pt
JEV_VALIDATION_ITEMS=/path/to/validation_items.pt
```

The legacy calibration fallback samples training items and is not an honest held-out calibration
result. `JEV_STRICT_BUILD=1` must fail on silently skipped training items.

The 81 routing cases, 45 safety cases, and 27 scripted calls are historical/development benchmarks.
Do not use them to relabel training data or claim a new final result after observing model outputs.
A synthetic-only result must be labelled as synthetic-contract evidence.

## Label and teacher policy

- Teacher agreement is a consistency filter, not independent human ground truth.
- Preserve provider/model, prompt version, raw response, cost, and acceptance/rejection reason.
- Do not describe two prompt variants from one model as independent annotators.
- Prefer a structured business contract as the source of a synthetic label, then use models to
  render and verify the text.
- Keep ambiguous cases as clarification/abstention cases; do not force them into `other`.
- A new safety or scope policy needs its own test suite and a new evaluation boundary.

## Current Phase-A guard

`scope` and `safety_applicability` are defined in the store profile and enforced by the policy when
present. They are not yet part of the active v7 pass because v7 was not trained on those questions.
Do not add an untrained decision to the demo merely to make a test pass.

## Research and reporting

For current Laya, PyTorch, MLX, or hosted-model behavior, consult current primary documentation,
source, model cards, papers, and issue trackers. Record URLs, dates, versions/revisions, and the
specific claim supported. Separate upstream facts from local recommendations.

For every experiment:

1. state a hypothesis and success criteria;
2. freeze the relevant data and evaluation split;
3. run the smallest safe local diagnostic;
4. record negative and null findings;
5. inspect the packaged snapshot;
6. promote only if the predeclared gates pass.

Do not launch a Kaggle job or paid generation run without explicit approval.
