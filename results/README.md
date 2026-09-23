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
| `dataset_report.json` | the synthetic set — 1395 rows, acceptance criteria, per-class counts. **Predates the containment trim**, which removed one row that echoed a held-out case; the live count is 1394 (1255 train / 139 dev). |
| `severity.json` | the yes/no questions, **old wording**, threshold 0.3: recall 0.778, missed 4 of 18. Superseded by the rewording. |
| `severity_reworded.json` | **same model, reworded question**, threshold 0.7: recall **1.000**, missed **0 of 18**. No training involved — the question asked what the caller *said* rather than whether the *vehicle* was unsafe. |
| `severity_trained.json` | the yes/no questions after they were finally **trained** (v5). `needs_human` recall **0.571 → 0.857** on genuinely held-out cases, but precision collapses (0.207) and the probabilities are saturated, so the threshold sweep is flat. |
| `severity_calibrated.json` | the **temperature experiment**. `temperature_by_options` is inherited from the base checkpoint and overrides the trainer's fitted vector — a real bug. Removing it moves a noul probability from 1.000 to **0.996**, so it is *not* the cause of the saturation. Recorded because it killed a hypothesis I had already half-written up. |
| `eval_v5.json` | four-arm eval, the multi-task 4-epoch fine-tune. destination 0.938, **joint 0.877**, call-level 0.889 — down from v4's 0.914 / 0.963. Superseded by v6: **the regression was the epoch count, not the new tasks.** |
| `eval_v6.json` | four-arm eval, **the best checkpoint** (8 epochs, all five tasks). destination 0.963, joint **0.926**, queue **0.975** — ties `deepseek-flash` on joint and beats it on queue. Call level 0.852, which is *not* a regression from training; see below. |
| `eval_v4_currentpolicy.json` | the same v4 checkpoint re-measured under the **current** safety policy. Call level **0.852**, identical to v6 — which is what proves the call-level drop was the safety rewording, not the new training tasks. `eval_v4.json` predates that policy change and its 0.963 is not comparable. |
| `severity_v6.json` | the safety questions on the 8-epoch checkpoint. Recall unchanged from v5 (`is_safe_to_drive` 1.000, `needs_human` 0.857) and precision still below untrained (0.692 vs 0.857). The fitted noul temperature is **1.0** here against v5's 3.683 — the 4-epoch head was the pathological one. |
| `eval_v7.json` | four-arm eval, **the first checkpoint with clean provenance** (v13: register-corrected data, the calibration fix, the held-out acceptance split). destination 0.914, **joint 0.876** — worse than v6. But the **call level is 0.926 against v6's 0.852**, because the dispatch hijacks halved (4 → 2). `results/severity_v7.json` and `acceptance_v7.json` come from the same run. |
| `severity_v7.json` | the register experiment's test. `is_safe_to_drive` **passed its gate**: recall held at 1.000 while false alarms fell **8 → 3** and precision 0.692 → **0.857**, recovering the untrained model's precision with the trained model's recall. `needs_human` **failed**: recall 0.857 → 0.714 (missed `sev-40`, `sev-42`). |
| `acceptance_v7.json` | the **first honest acceptance measurement** — scored on `acceptance_dev.jsonl`, held out by reply phrasing. Base 0.615 with `unclear` 0 of 33; fine-tuned **1.000**, `unclear` 33 of 33. The booking bug (never abstaining) is fixed. Caveat: the split holds out *reply phrasings*, not transcripts. |
| `eval_v6_fixedcalib.json` | ablation: v6's weights with v7's calibration behaviour (inherited map removed). Isolates the trainer fix from the data change in v7's routing regression — **it is inert** (identical misses, +0.000 joint), as theory says a temperature scale cannot change an argmax. |
| `eval_v8.json`, `severity_v8.json`, `acceptance_v8.json` | **v14, the clean additive experiment.** v6's exact data (minus the 6 contaminated rows) plus the 222 terse rows, and nothing else — built by construction so the delta is provable. It settles the confound v7 left open:

| | dest | joint | call | `is_safe` FA | `needs_human` recall |
| --- | --- | --- | --- | --- | --- |
| v6 (long data) | 0.963 | 0.926 | 0.852 | 10 | 0.857 |
| v7 (+terse, −172 long) | 0.914 | 0.876 | 0.926 | 3 | 0.714 |
| **v14 (+terse only)** | 0.938 | 0.889 | 0.889 | **3** | 0.714 |

**The terse rows did the work, not the deletions.** v14 keeps all 172 long hazards and still gets false alarms down to 3 with recall held at 1.000 — so the register hypothesis is confirmed and attributable, which is what v7 could not show. It also shows the collateral is the same addition: `needs_human` recall falls to 0.714 in both, because the 112 new terse hazards are all escalation positives and dilute that class.

**But v14 is not the better demo checkpoint.** It regresses `call-glass` (a windscreen booking → Roadside) and `tow-09`, while recovering the detailing cases. v7 gets 6 of 6 demo-relevant calls; v14 gets 5. `models/active` stays on v7. |
| `acceptance.json` | the mid-conversation question, before training: overall 0.644, and `unclear` **0 of 130**. It never abstains, which is why the booking demo is flaky. Measured on `acceptance_train.jsonl`, but the model was untrained on it, so this reads as zero-shot. |
| `acceptance_leaky.json` | ⚠️ **Do not quote.** The trained acceptance model reporting **1.000 accuracy** — scored against `acceptance_train.jsonl`, the file training is built from. Kept as the example of what a leak looks like from the inside. |

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
- **"Held out" is checked, not intended.** The eval sets come out of the same pipeline as the
  training data, and three of these reports were found reading it: the acceptance eval scored the
  training file outright, and two cases leaked by *containment* — a short eval utterance sitting
  inside a longer training row, which an exact-text check cannot see and Jaccard scores too low to
  catch. Tests now assert exact disjointness *and* containment disjointness for routing and severity,
  and phrasing disjointness for acceptance. Re-run `scripts/trim_heldout_echoes.py` if a generator
  changes.
- **Check what policy a report was measured under, not just what model.** `eval_v4.json` shows a
  call-level queue accuracy of 0.963; `eval_v4_currentpolicy.json` shows **0.852 for the same
  checkpoint**. The only difference is that the safety question was reworded and the dispatch
  threshold moved between them — and because an unsafe call's queue is *overwritten* with
  `Roadside / Towing`, that policy change silently re-routed 3 of 27 calls. Comparing across a
  policy change is the same mistake as comparing across a taxonomy change.
