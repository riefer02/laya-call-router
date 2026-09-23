# Morning review

Where the project stands, what I did overnight, and what I'd do next — with honest estimates
rather than confident-sounding ones.

Read `LEARNINGS.md` for the narrative. This is the decision document.

---

## The one-line version

**The retrain that was supposed to train the questions we never trained did that — and it cost us
routing accuracy. Then three of our measurements turned out to be reading their own training data.**

The first two GPU runs were wasted (different causes, same bug shape). The third trained all five
tasks and regressed routing. Along the way I found that the acceptance number was memorisation and
that the safety question's marquee case was in the training set. One headline survived, and it is a
real one: **`needs_human` recall went from 0.571 to 0.857 on genuinely held-out data.**

A fourth run (v12, 8 epochs, all five tasks) is training now and will tell us whether the routing
regression was the new tasks or just the epoch count.

---

## What is running now

**Kernel v12** — 8 epochs, all five questions as training targets, on the corrected data (285 rows
of echo removed across two files). It is the run that disentangles the two changes that got
confounded in v5.

Why it matters: **v5 ran 4 epochs and v4 ran 8.** v5 regressed routing, and I cannot yet say whether
that is the four new training tasks or simply stopping early. That is the same confound in
`LEARNINGS.md` that cost +2.5 points the last time we looked at a loss curve. v12 removes it.

---

## Where it stands

| metric | base | v4 (8ep, 2 tasks) | **v5 (4ep, 5 tasks)** | nano | deepseek-flash |
| --- | --- | --- | --- | --- | --- |
| destination | 0.654 | 0.951 | 0.938 | 0.963 | **0.988** |
| sub-queue | 0.518 | 0.914 | 0.877 | 0.864 | **0.914** |
| joint | 0.518 | **0.914** | 0.877 | 0.864 | **0.914** |
| queue (the outcome) | 0.667 | **0.963** | 0.938 | 0.926 | 0.951 |
| call-level queue (27 calls) | 0.852 | **0.963** | 0.889 | 0.963 | 0.963 |
| p50 latency | 21.4 ms | **21.8 ms** | 21.8 ms | 631 ms | 1455 ms |
| cost per call | $0 | **$0** | **$0** | $0.0025 | $0.0114 |

**v5 gave back 3.7 points of joint accuracy.** Whether that is the new tasks or the missing four
epochs is exactly what v12 answers. `results/eval_v5.json` is the raw report.

The safety questions, measured on 45 hand-labelled cases that are genuinely held out:

| question | v4 encoder (noul never trained) | **v5 (noul trained)** |
| --- | --- | --- |
| `is_safe_to_drive` recall | 1.000 | **1.000** (0 of 18 missed) |
| `is_safe_to_drive` precision | **0.857** | 0.667 |
| `needs_human` recall | 0.571 (3 of 7 missed) | **0.857** (1 of 7) |
| `needs_human` precision | **1.000** | 0.207 |

**Training the yes/no questions worked, and it also over-fired.** `needs_human` recall is the single
best thing to come out of the night: it goes from missing 3 of 7 escalated calls to missing 1. But
precision fell off a cliff on both questions, and I think I know why — see finding 4.

---

## The measurements that were lying

This is the part I'd want you to read even if you read nothing else, because it is the part that
was hiding real problems behind good news.

**1. The acceptance number was memorisation.** `eval_acceptance.py` scored
`acceptance_train.jsonl` — the file training is built from. The fine-tuned model reported **1.000
accuracy**, against 0.554 for the base model, and that looked like the booking demo being fixed.
It was the model reciting phrasings it had been trained on.

The split now holds out whole *reply phrasings*, because every reply is a template: a row split
would still put "Yes, {t} works for me." in both halves and measure nothing. train 281 / dev 109,
disjoint. **The acceptance question cannot be honestly scored until we retrain on the split** — so
the booking-demo claim is suspended, not made.

**2. Routing and severity leaked by containment, not by text.** The deduper refuses anything within
Jaccard ≥ 0.6 of the held-out 81. But a short query inside a longer sentence scores *low*, because
the union is large:

```
"What time do you open on Saturdays?"                     7 words
"What time do you open on Saturdays? I couldn't..."      14 words   Jaccard 0.50  → passed
```

The same shape hid `sev-04` — *"there's smoke coming from under the hood"* — inside two training
rows, and `sev-04` is the case the reworded safety question is specifically credited with rescuing.
An exact-text check cannot see either. `contains()` now joins Jaccard in the deduper, one shared
rule cleans the data, and a test asserts no eval set is a substring of a training row.

**3. The trainer's temperature fit was dead code.** `temperature_by_options` is inherited from the
base checkpoint and takes precedence over the fitted vector. Every bucket our questions occupy is in
that inherited map, so the step the code calls *"what makes the confidence usable"* never applied to
a single question. Fixed in `train_ddp.py`.

**I was wrong about what that would buy.** I predicted it would un-saturate the probabilities. I
tested it on a staged checkpoint: a noul probability moves **1.000 → 0.996**. The plumbing was
broken, but it was not the cause. The trained logits are extreme in their own right and the
trainer's chosen temperature (3.683) cannot soften them.

**4. Which leaves the likely real cause of the precision loss: the training prior.** The severity
data is capped at a **40% positive rate** and a real switchboard is nowhere near that for
`needs_human`. The model learned a prior nothing like the world, so it over-fires — and because the
probabilities are saturated, **a threshold cannot undo it**. That is the next thing to try.

---

## What I did not do

**I did not relabel `gen-08`.** Editing the test set after seeing results is grading our own
homework. The dispute is recorded; a second labeller is the fix.

**I did not tune prompts against the 81.** Frozen.

**I did not trust the 1.000.** It is kept as `results/acceptance_leaky.json` rather than deleted,
because a flattering number that was wrong is worth being able to point at.

---

## What I would do next, in order

Estimates are ranges with reasoning. "Confidence" is how sure I am the *direction* is right.

### 1. Read v12 — running now

**Does routing recover at 8 epochs with all five tasks?** If joint returns to ~0.914 the regression
was the epoch count and the multi-task training is free. If it stays at 0.877, the four new tasks
are competing with the choice questions and the training mix needs attention.
**Confidence:** high that this is the next thing to know; that is all it is.

### 2. Lower the severity positive rate toward the real base rate

**What:** regenerate or re-cap `severity_train.jsonl` below 40% — the natural rate for
`needs_human` is far lower.

**Why:** it is the explanation that fits the evidence. The trained model over-fires on both
questions (precision 0.667 and 0.207), the probabilities are saturated so no threshold helps, and a
shifted prior is the classic cause. The comment in `generate_severity.py` already predicted this and
capped at 40% — **I now think 40% was still too high.**

**Projected gain:** `is_safe_to_drive` precision **0.667 → 0.80–0.86** (recovering the v4 number)
while holding recall at 1.000.
**Cost:** cheap — the cap re-runs with `--rebalance-only`, no API calls. A regen is ~$1.30 if the
labels need to change.
**Confidence:** medium-high on direction, because it is the only lever that survives the saturation
finding.

### 3. Re-measure acceptance on the held-out split

**What:** retrain (v13) and score against `acceptance_dev.jsonl`.

**Why:** right now the number is void. The base model on the dev split is a legitimate zero-shot
reading and worth having regardless.
**Projected gain:** unknown, and that is the point — **we have never measured this.**
**Cost:** free GPU, one run. **Confidence:** high that it will be lower than 1.000 and higher than
0.554, but I will not guess the number.

### 4. A second labeller — still the ceiling on the destination number

One case (`det-04`) is missed by every model; `gen-08` is answered against our label by all three.
Where every model disagrees with the key, the key is the likeliest thing to be wrong. No model work
moves this. **+1.2 to +3.7 destination** as a *measurement correction*, not a model improvement.

### 5. Answer from the store facts — free product win

The agent should answer "what time do you open?" from `facts` instead of transferring. Data already
loaded, hours/directions is top repeatable Fixed Ops volume. ~1 hour, $0, no metric moves.

---

## What I would NOT do

- **More epochs, blindly.** v4 converged at 0.042. But note v5's loss was *still falling* at 4
  epochs on the bigger set — which is exactly why v12 exists.
- **Tune a threshold to fix the safety precision.** The sweep is flat from 0.3 to 0.8 for a reason:
  the model is confidently wrong. A threshold on a saturated distribution is a no-op.
- **A bigger teacher.** Measured: `deepseek-v4-pro` agrees with our labels *less* than flash at 3×
  the cost.
- **Trust any single run.** One case on 81 is 1.23 points; the LLM arms moved 2.5 points between
  identical runs.

---

## Open questions for you

1. **Priority: routing parity, or the safety/booking surface?** v5 bought `needs_human` recall and
   paid for it in routing. If routing parity is the headline, I should treat the extra training
   tasks as something to isolate rather than absorb.
2. **How much is the booking demo worth?** The acceptance classifier is trained and *unmeasurable*
   until we retrain on the split. That is one GPU run away, but it is another run.
3. **Do we grow the test set?** 81 cases means one case is 1.23 points. Growing to ~150 would halve
   the interval — but it is more single-labeller labels, which is the constraint we are already
   fighting.

---

*Everything above is reproducible: `results/` holds the raw reports, `results/README.md` says which
are superseded, and every number came from a script that exits non-zero when a gate fails.*
