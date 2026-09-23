# Morning review

Where the project stands, what I found this session, and what I'd do next — with honest estimates
rather than confident-sounding ones.

Read `LEARNINGS.md` for the narrative. This is the decision document.

---

## TL;DR

- **The goal is met.** The fine-tuned cascade matches both LLM arms on routing quality, at ~65× the
  speed, for $0 a call, deterministically.
- **The honest claim is parity, not victory.** `deepseek-flash` scored 0.914, 0.889, 0.901 and 0.926
  joint across four runs on identical inputs; ours has held at 0.914 every time. I had been
  reporting the flattering half of that.
- **Three things I'd do next, in order:** correct the labels (cheapest, most certain), close the
  three destination gaps (targeted data), and reword the safety question (biggest safety win).
- **One thing I'd stop doing:** trying to fix `needs_human` as a classifier question. I think it's
  the wrong kind of question, and we already have evidence for that.

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

### 1. Correct the labels — cheapest, most certain

**What:** relabel `gen-08` (→ service). Then get a second opinion on the whole 81 — a model pass is
enough to *flag* candidates, and I adjudicate.

**Why:** measured directly above. `gen-08` is a 1.2-point error against every model we have. There
are likely 2–3 more like it; `det-04` was already known to be missed by all three.

**Projected gain:** **+1.2 to +3.7 points destination**, and roughly the same on joint.
**Cost:** ~1 hour, ~$0.05.
**Confidence:** high for `gen-08`; medium for how many others exist.

**Important caveat:** this improves the *measurement*, not the model. If we do this and the number
goes up, that is us grading our own homework more honestly — not the model getting better. I'd want
that stated in the README rather than quietly banked.

### 2. Close the three destination gaps

**What:** targeted generation for `front_desk` and roadside-recovery phrasings, then retrain.

**Why:** all three remaining errors are in `front_desk` (two) and one roadside case, and both
teachers handle them. `front_desk` is still the thinnest class (150 rows against service's 303) and
the source of every destination error we have.

**Projected gain:** **+1.2 to +2.5 points destination** (catching 1–2 of 3).
**Cost:** ~$0.30 generation, free GPU, ~1 hour.
**Confidence:** medium. These are the hardest cases — the model already failed them after 8 epochs —
so targeted data may not be enough. Worth one attempt.

### 3. Reword the safety question — biggest safety win

**What:** change `is_safe_to_drive` from "does the caller indicate the vehicle is unsafe" to
something that asks about **hazard** rather than **statement**. Then regenerate that question's
training data.

**Why:** finding #2. The model isn't failing to understand the cases, it's answering a literal
question correctly. Precision is 1.000, so we have room to be more aggressive.

**Projected gain:** **recall 0.778 → 0.88–0.94**, i.e. 1–3 more stranded callers caught.
**Cost:** ~$0.20 generation, free GPU, ~1 hour.
**Confidence:** medium-high on direction; the failure mode is clearly identified and uniform.
**This is the one I'd do first if safety matters more than the headline number.**

### 4. Train the acceptance classifier — biggest *product* gap

**What:** it's currently the base checkpoint answering a question it has never seen. Needs a new
generation mode: offered times → which one the caller took.

**Why:** the booking flow is the MVP, and it's flaky. The same scripted call books at p=0.68 on one
run and clarifies at p=0.53 on another. The confidence floor stops it filing a booking nobody agreed
to — which is the right behaviour — but "asks again about half the time" is not a product.

**Projected gain:** **zero on routing metrics.** This is completeness, not accuracy.
**Cost:** ~$0.50–1.00 generation, free GPU, ~2–3 hours (new generation mode).
**Confidence:** high that the recipe works — it's the same fine-tuning that took routing from 0.518
to 0.914.

### 5. Replace `needs_human` with policy — I think the question is wrong

**What:** stop asking a classifier. Derive it: complaints and feedback → human, billing or legal
language → human, repeated contact → human.

**Why:** recall 0.571 / precision 0.250 is worse than useless. And we already have evidence that
control-flow questions classify badly — the earlier `next_action` question answered at *confidence
0.03* and picked the same option regardless of input. "Does this need a person" is a policy
judgement dressed as a content question.

**Projected gain:** recall ~0.57 → ~0.85 **if** the sub-queue is right, since it would inherit that
accuracy.
**Cost:** ~1 hour, $0.
**Confidence:** medium. This is the one I'd most like a second opinion on.

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
