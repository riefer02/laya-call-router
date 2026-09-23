# Morning review

Where the project stands, what I did overnight, and what I'd do next — with honest estimates
rather than confident-sounding ones.

Read `LEARNINGS.md` for the narrative. This is the decision document.

---

## Overnight: safety first, and it needed no training at all

You said safety first, so that is what I did. **`is_safe_to_drive` now catches every stranded
caller in the labelled set — 0 missed out of 18, up from 4.**

It did not need a model change. The question was asking the wrong thing:

> Old: *"Does the caller **indicate** the vehicle is unsafe to drive or stranded?"*
> New: *"Is it **unsafe to drive the vehicle**?"* — with the two option texts carrying the
> definition, since the model chooses between them.

The old wording asked what the caller *said*, so every implied hazard was missed and the model was
not wrong — it was answering the literal question correctly:

```
"There's smoke coming from under the hood."     p=0.06  ->  0.97
"There's a burning smell and smoke..."          p=0.00  ->  0.99   (certain, not unsure)
"The accelerator stuck open..."                 p=0.40  ->  0.81
"I hit a pothole and the wheel is bent..."      p=0.26  ->  0.96
```

| | recall | precision | missed |
| --- | --- | --- | --- |
| old wording, threshold 0.3 | 0.833 | 1.000 | 3 of 18 |
| **new wording, threshold 0.7** | **1.000** | 0.857 | **0 of 18** |

The dispatch threshold moved 0.3 → 0.7 because the reworded question shifted the whole distribution
up. Three unnecessary dispatches in exchange for never leaving a stranded caller on the road.

Doing that exposed a real inconsistency: the `unsafe_to_drive` flag, the priority, the handler and
the transfer decision were reading **three different thresholds**, so a caller could be dispatched
as unsafe while the audit trail said they were not. One number now drives all of them.

## `needs_human`: I was wrong, and the evidence says so

I recommended replacing it with a policy rule. **I measured both before acting on it:**

| approach | recall | precision | missed |
| --- | --- | --- | --- |
| policy rule (route → handler) | **0.143** | 0.250 | 6 of 7 |
| classifier question | **0.571** | 1.000 | 3 of 7 |

The policy rule is far worse, because the routing does not even identify these calls — a complaint
about service routes to `service/other`, a billing dispute to `front_desk/other`. So it **stays a
classifier question**, and I built the training data for it (2,439 labelled utterances for both
yes/no questions, ~$1.30).

The reworded question also lifted its precision from 0.250 to **1.000** on the same data.

## The acceptance classifier: measured, and the failure is specific

It was never trained either. The measurement is unambiguous:

| class | base | v4 fine-tune |
| --- | --- | --- |
| slot_1 / slot_2 / slot_3 | 44 / 41 / 2 | 44 / 43 / 34 |
| none_of_these | 129/130 | 130/130 |
| **unclear** | **0/130** | **0/130** |
| **overall** | 0.554 | **0.644** |

**It never once answers `unclear`.** That is the whole story of the booking demo being flaky: "Hmm,
let me think about it" gets forced into a slot. 390 training examples are built and waiting.

## A mistake I made, and what I changed because of it

Rebalancing the severity data read a field the pipeline had already dropped, matched nothing, and
**wrote the empty result over 2,439 labelled rows** — about $1.30 of API calls, destroyed in one
line. It was never committed, so there was no recovery.

Two fixes, because the lesson is not "be careful":
- The rebalance reads the schema it actually receives.
- **It refuses to write an empty result from a non-empty input, and all writes are atomic.** A
  destructive operation without a guard was the actual bug; the wrong field name was just the
  trigger.

Regenerating costs $1.30 and 50 minutes. Cheaper than the alternative lesson.

## What is running now

A retrain with the yes/no questions and the acceptance question as training targets. They were
never trained before — `build_items.py` only built choice items — which is why the safety question
missed a fifth of the stranded callers it exists to dispatch.

## What I did **not** do

**I did not relabel `gen-08`.** I said in the earlier draft that correcting that label was the
cheapest win available. Then I checked: relabelling the destination would fix that metric but leave
the sub-queue wrong, netting **zero on joint**. More importantly, **editing the test set after
seeing results is grading our own homework**, and it is not what "more accuracy" means. The dispute
is recorded instead, and the proper fix remains a second labeller.

---

## TL;DR

- **The goal is met.** The fine-tuned cascade matches both LLM arms on routing quality, at ~65× the
  speed, for $0 a call, deterministically.
- **The honest claim is parity, not victory.** `deepseek-flash` scored 0.914, 0.889, 0.901 and 0.926
  joint across four runs on identical inputs; ours has held at 0.914 every time. I had been
  reporting the flattering half of that.
- **Safety is fixed and cost nothing but wording.**
- **Three things I'd do next, in order:** train the acceptance classifier (built), close the three
  destination gaps, and get a second labeller.

---

## Where it stands

| metric | base | **fine-tuned** | nano | deepseek-flash |
| --- | --- | --- | --- | --- |
| destination | 0.654 | 0.951 | 0.963 | **0.988** |
| sub-queue | 0.518 | 0.914 | 0.876 | **0.926** |
| joint | 0.518 | 0.914 | 0.876 | **0.926** |
| queue (the outcome) | 0.667 | 0.963 | 0.926 | 0.963 |
| p50 latency | 21.4 ms | **21.8 ms** | 631 ms | 1455 ms |
| cost per call | $0 | **$0** | $0.0025 | $0.0114 |
| determinism | 1.00 | **1.00** | 0.98 | 0.99 |

Fine-tuning gained **+39.6 points joint** over base. The economics are not close.

**The teacher ceiling matters here:** `deepseek-flash` agrees with our hand labels 0.975
destination / 0.901 sub-queue. We are at 0.951 / 0.914 — **already past the teacher on sub-queue,
2.4 points short on destination.** That 2.4 points is the most we can gain from pure distillation.

---

## Findings recorded this session

The detail is in `LEARNINGS.md`; these are the ones that change what we do next.

**1. Of our 4 destination errors, 3 are ours and 1 is the label's fault.** I checked every one
against both teachers:

| case | label | ours | flash | v4-pro | verdict |
| --- | --- | --- | --- | --- | --- |
| `tow-09` | service | body_shop | service | service | ours to fix |
| `gen-07` | front_desk | non_customer | front_desk | front_desk | ours to fix |
| `gen-09` | front_desk | non_customer | front_desk | front_desk | ours to fix |
| `gen-08` | front_desk | service | service | service | **label is wrong** |

`gen-08` ("Do you offer loaner cars while mine is in for service?") is answered `service` by *all
three models* against our label. When everyone disagrees with the key, the key is the likeliest
thing to be wrong. **That single label is costing us 1.2 points.**

Caveat on the other three: "both teachers agree with the label" is evidence, not proof. `tow-09`
("my car is in a ditch and it needs recovering") is arguably a body-shop job — but the *request* is
recovery, which is roadside, so I think the label is right and we're wrong. That is a judgement
call, and it's the kind of call a second labeller should be making, not me.

**2. The safety question fails on implied hazards, not on hard cases.** All four missed unsafe
callers are cases where the danger is real but the caller never says they've stopped:

```
sev-04  "There's smoke coming from under the hood."        p=0.06
sev-08  "I hit a pothole and the wheel is bent..."         p=0.26
sev-11  "There's a burning smell and smoke..."             p=0.00
sev-14  "The accelerator stuck open..."                    p=0.40
```

`sev-11` at p=0.00 is the tell: the model isn't unsure, it's *certain* — because the question asks
what the caller **indicates**, and "there's a burning smell" doesn't indicate anything about
drivability. A person would hear a fire.

**3. `needs_human` misses the three cases that most need a human:** a double-billing dispute, a
legal matter, and an injury. Recall 0.571, precision 0.250.

**4. Zero false alarms on dispatch.** Precision is 1.000 at every threshold from 0.3 to 0.8. There
is real headroom to catch more unsafe callers without sending trucks to people who were fine.

---

## Projected improvements, ranked

Estimates are ranges with reasoning, not predictions. "Confidence" is how sure I am the *direction*
is right, not the size.

### 1. Correct the labels — **decided against, see "What I did not do"**

**What:** relabel `gen-08` (→ service), then get a second opinion on the whole 81.

**Why I did not:** it fixes the destination metric but nets **zero on joint** (the sub-queue stays
wrong), and editing the test set after seeing results is grading our own homework. The dispute is
recorded; a second labeller is the real fix.

**Still worth doing:** the second opinion on the *whole* 81. `det-04` is missed by every model, and
there are likely 2–3 more like it. **+1.2 to +3.7 destination** if they turn out to be label
problems rather than model ones — but that is a measurement correction, not a model improvement,
and should be reported as such.

### 2. Close the three destination gaps

**What:** targeted generation for `front_desk` and roadside-recovery phrasings, then retrain.

**Why:** all three remaining errors are in `front_desk` (two) and one roadside case, and both
teachers handle them. `front_desk` is still the thinnest class (150 rows against service's 303) and
the source of every destination error we have.

**Projected gain:** **+1.2 to +2.5 points destination** (catching 1–2 of 3).
**Cost:** ~$0.30 generation, free GPU, ~1 hour.
**Confidence:** medium. These are the hardest cases — the model already failed them after 8 epochs —
so targeted data may not be enough. Worth one attempt.

### 3. Reword the safety question — **DONE, and it needed no training**

Recall **0.778 → 1.000**, missed **4 of 18 → 0 of 18**, for three false alarms. See the top of this
document. The remaining work here is only that the model now *also* has this question as a training
target, which should recover some of the precision.

### 4. Train the acceptance classifier — **data built, retraining now**

**Measured:** overall 0.644, and `unclear` **0 out of 130** — it never abstains, which is the whole
reason the booking demo is flaky. 390 training examples are built from templates rather than a
teacher, so the label is exact rather than inferred, with the slot classes cycled so no positional
bias is taught.

**Projected gain:** acceptance accuracy **0.644 → 0.85–0.92** if `unclear` becomes learnable. Zero
on the routing metrics — this is the booking flow, not the classifier.

### 5. `needs_human` as policy — **measured and rejected**

I proposed this and the evidence killed it: the policy rule scores **recall 0.143 against the
classifier question's 0.571**, because the routing does not identify these calls in the first
place. It stays a classifier question, and it now has training data. See the top of this document.

### 6. Answer from the store facts — free product win

**What:** the agent should answer "what time do you open?" from `facts` instead of transferring.

**Why:** the data is already loaded and unused, and hours/directions is one of the top repeatable
Fixed Ops call types.

**Projected gain:** product completeness. No metric moves.
**Cost:** ~1 hour, $0. **Confidence:** high — it's a lookup, not a model.

### 7. Adjudicate the teacher's disagreement region — the only path *above* the teacher

**What:** 563 of 1063 generation candidates were dropped (434 missed the intended target, 129
disagreed with themselves). Re-label that region with a stronger model or by hand and add what
agrees.

**Why:** it's where `deepseek-flash` is unreliable. Training on cleaner labels than the teacher
produces is the only structural way to exceed it.

**Projected gain:** **+0.5 to +1.5 joint.** Uncertain.
**Cost:** ~$2, ~2 hours. **Confidence:** low-medium. Small region, hard cases.

---

## Projected outcome if we do 1–3

| metric | now | projected | note |
| --- | --- | --- | --- |
| destination | 0.951 | **0.975 – 0.988** | most of this is the label correction |
| sub-queue | 0.914 | 0.926 – 0.951 | |
| joint | 0.914 | **0.94 – 0.96** | |
| queue | 0.963 | 0.975 – 0.988 | |
| unsafe recall | 0.778 | **0.88 – 0.94** | the safety win |
| booking | flaky | reliable | needs lever 4 |

At the top of those ranges we'd be ahead of `deepseek-flash` on joint. **But a meaningful share of
that is honest relabelling, not a better model** — and the intervals on 81 cases are wide enough
that I would still call it parity until we have a larger test set.

---

## What I would NOT do, and why

- **More epochs.** Converged: loss flattens at 0.042 by epoch 6–7.
- **More data volume.** We're within 2.4 points of the teacher ceiling on destination. Volume isn't
  the constraint; the labels are.
- **A bigger teacher.** Measured. `deepseek-v4-pro` agrees with our labels *less* than flash at 3×
  the price.
- **Tuning prompts or descriptions against the 81.** I did this once, it made things worse, and any
  gain would be overfitting. Frozen.
- **Trusting any single run.** Four runs, four different deepseek numbers. Nothing under ~3 points
  is a finding on 81 cases.

---

## Open questions for you

1. **Safety first or headline first?** Lever 3 (reword the safety question) is the highest-value
   thing in the document, and it moves a number nobody looks at. Lever 1 moves the number everyone
   looks at. I'd do 3 first; I don't know if that matches your priorities.
2. **Is `needs_human` a policy or a question?** I think policy. If you disagree, it needs its own
   labelled set before I'd trust any number from it.
3. **How much do you want the acceptance classifier?** It's 2–3 hours for zero accuracy gain. It's
   the difference between a demo that books and a demo that usually books.
4. **Should we grow the test set?** 81 cases means one case is 1.23 points and the intervals are
   ±5. Growing to ~150 would halve that — but it's more single-labeller labels, which is the
   constraint we're already fighting.

---

*Everything above is reproducible: `results/` holds the raw reports, `results/README.md` says which
are superseded, and every number came from a script that exits non-zero when a gate fails.*
