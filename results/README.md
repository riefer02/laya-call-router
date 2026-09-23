# Results — what each file is, and which are superseded

Raw evidence behind the numbers in the top-level README. Tracked in git deliberately: a
measurement that is not versioned cannot be audited, and gets quietly overwritten by the next run.
That already happened once — the old-taxonomy dataset report was lost this way, which is why these
are now committed.

**Ground truth is `data/calls/*.jsonl`** and the taxonomy is `config/store_profile.json`. Every
report here was produced by `scripts/eval.py`, `scripts/validate_teacher.py`,
`scripts/generality_test.py` or `scripts/generate_training.py`, and every one names the checkpoint
it used.

## Current — corrected taxonomy

| file | what it measured |
| --- | --- |
| `teacher_validation.json` | **deepseek-flash teacher gate**: 0.975 destination / 0.901 sub-queue / 0.901 joint agreement with the 81 hand labels. This is the ceiling anything distilled from flash can reach. |
| `teacher_validation_v4pro.json` | **deepseek-v4-pro teacher gate**: 0.951 / 0.864. *Worse* than flash at 3x the cost — the "buy a bigger teacher" idea, measured and rejected. |
| `eval_v3.json` | four-arm eval, 4-epoch fine-tune. destination 0.938, joint 0.864, queue 0.951. |
| `eval_v4.json` | four-arm eval, 8-epoch fine-tune. destination 0.951, joint 0.914, queue 0.963. |
| `eval_newtaxonomy_base.json` | base Laya on the corrected taxonomy, before any fine-tuning: 0.654 / 0.518. |
| `dataset_report.json` | the current synthetic set — 1395 rows, acceptance criteria, per-class counts. |
| `severity.json` | the yes/no questions, **old wording**, threshold 0.3: recall 0.778, missed 4 of 18. |
| `severity_reworded.json` | **same model, reworded question**, threshold 0.7: recall **1.000**, missed **0 of 18**. No training involved — the question asked what the caller *said* rather than whether the *vehicle* was unsafe. |
| `acceptance.json` | the mid-conversation question: overall 0.644, and `unclear` **0 of 130**. It never abstains, which is why the booking demo is flaky. |

## Superseded — old 9-department taxonomy

Kept on disk and labelled, not deleted. These measured a task that was partly wrong: tires,
detailing and towing sat beside service as if they were peer departments, and `general` was a
dumping ground absorbing two unrelated things. **Their numbers are not comparable to the current
ones** and must not be quoted as if they were.

| file | why superseded |
| --- | --- |
| `eval_finetuned_v2.json` | the best old-taxonomy fine-tune (joint 0.889). The checkpoint at `models/kaggle-out-v2/` is invalid for the current taxonomy. |
| `eval_finetuned.json` | first old-taxonomy fine-tune. |
| `eval.json` | original support-triage cascade, before the dealership domain. |
| `quality.json`, `latency.json`, `incremental_call_cost.json`, `verified_call_cost.json`, `baseline_call_cost.json` | support-triage era measurements. |

## Reading any of these

- **One case on 81 is 1.23 points.** Wilson 95% intervals are in every report; two arms whose
  intervals overlap are not distinguishable at this sample size. The LLM arms moved 2.5 points
  between consecutive runs on identical inputs.
- **`destination_accuracy` is agreement with the taxonomy labels; `queue_accuracy` is the routing
  outcome.** They differ because two labels can route to the same place (`front_desk` and
  `non_customer` both go to Front Desk). Both are reported; neither replaces the other.
- **The ground truth is single-labelled.** Where all three models agree against it (see `det-04`),
  the label is the likeliest thing to be wrong.
