# Morning review

Where the project stands, what I did overnight, and what I'd do next — with honest estimates
rather than confident-sounding ones.

Read `LEARNINGS.md` for the narrative. This is the decision document.

---

## The one-line version

**We now have our best checkpoint, and it ties `deepseek-flash` — and the reason the previous retrain
looked like a regression was two separate confounds, both of them mine.**

v6 (8 epochs, all five questions) is the best thing we have built: joint **0.926** against
`deepseek-flash`'s 0.926, with a better queue outcome (0.975 vs 0.963). The v5 "regression" was
**the epoch count, not the new training tasks**. And the call-level regression that followed it was
**not the training at all** — it was the safety rewording, which I measured by re-running the *same*
v4 checkpoint under the current policy.

Along the way three of our measurements turned out to be reading their own training data, including
a 1.000 accuracy on the booking question that was pure memorisation.

The one headline that survived every audit: **`needs_human` recall went from 0.571 to 0.857 on
genuinely held-out data.**

---

## Where it stands

| metric | base | v4 (8ep, 2 tasks) | v5 (4ep, 5 tasks) | **v6 (8ep, 5 tasks)** | nano | deepseek-flash |
| --- | --- | --- | --- | --- | --- | --- |
| destination | 0.654 | 0.951 | 0.938 | **0.963** | 0.951 | 0.988 |
| sub-queue | 0.518 | 0.914 | 0.877 | **0.926** | 0.876 | 0.926 |
| joint | 0.518 | 0.914 | 0.877 | **0.926** | 0.876 | 0.926 |
| queue (the outcome) | 0.667 | 0.963 | 0.938 | **0.975** | 0.926 | 0.963 |
| call-level queue (27) | 0.593 | 0.852\* | 0.889 | 0.852 | 0.963 | 0.963 |
| p50 latency | 21.4 ms | 21.8 ms | 21.8 ms | 22.9 ms | 747 ms | 1432 ms |
| cost per call | $0 | $0 | $0 | **$0** | $0.0025 | $0.0114 |

\* `eval_v4.json` reports 0.963, but it was measured **before** the safety rewording. Re-measured
under the current policy, v4 scores **0.852** — identical to v6. That is the evidence that the
call-level drop is policy, not training.

**The confound is resolved.** At 8 epochs, adding the four training tasks is strictly better than
v4: joint 0.914 → 0.926, queue 0.963 → 0.975. And the call level is *unchanged* by the training
(0.852 either way), which means training did not cost us anything at the product level.

The safety questions, on 45 genuinely held-out cases:

| question | v4 encoder (untrained) | v5 | **v6** |
| --- | --- | --- | --- |
| `is_safe_to_drive` recall | 1.000 | 1.000 | 1.000 (0 of 18 missed) |
| `is_safe_to_drive` precision | **0.857** | 0.667 | 0.692 |
| `needs_human` recall | 0.571 | 0.857 | **0.857** (1 of 7) |
| `needs_human` precision | **1.000** | 0.207 | 0.240 |

---

## The thing that actually cost us, and it was not the model

`is_safe_to_drive` does not just flag a call — **it overwrites the call's final queue**:

```python
# dealership.py
if roadside in flags or unsafe >= unsafe_threshold():
    queue = PROFILE.policy.get("roadside_queue", "Roadside / Towing")
```

So a false "unsafe" does not add a false alarm to a report. It **throws away a correct routing
decision and sends the call to Roadside / Towing.** Every one of the call-level failures is that
same line:

```
call-collision     expected Body Shop        got Roadside / Towing   ("rear-ended me yesterday, I need body work")
call-glass         expected Body Shop        got Roadside / Towing   ("a stone cracked my windscreen, I need it replaced")
call-recall        expected Warranty Desk    got Roadside / Towing   ("I got a recall notice, I need to get it done")
call-vague         expected Service          got Roadside / Towing
```

None of those callers is stranded. All four are booking non-urgent work. **The dispatch override is
the bug**, and it converts a safety false-positive into a routing failure.

The ledger on the rewording, all measured:

| | recall | false alarms | call-level |
| --- | --- | --- | --- |
| old wording, threshold 0.3 | 0.833 (3 of 18 missed) | **0** | 0.963 |
| new wording, threshold 0.7 | **1.000 (0 missed)** | 3 | 0.852 |

**We caught four more stranded callers and paid three misrouted calls for it.** Whether that is a
good trade depends on your priorities — I think catching a stranded caller is worth more than a
routing statistic, and I would still make it. But nobody measured the cost at the time, and it cost
11 points on the metric we quote.

And note the direction of the threshold: it went **up** (0.3 → 0.7) and the false alarms went **up**
too (0 → 3), because the rewording raised the whole distribution. That is the saturation finding
showing up at the policy level: **the threshold is not a control we actually have.**

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

### 1. Stop the safety flag from overwriting the routing — **the biggest measured win available**

**What:** `dealership.py` sets `queue = "Roadside / Towing"` whenever `unsafe` clears the threshold.
Make dispatch a **parallel action** — a flag and a truck — instead of a replacement for the routing
decision. The call still routes to Body Shop; roadside rolls alongside it.

**Why:** this single line is responsible for **every** call-level failure we have. The destination
decisions are now 0.963 accurate and we are throwing them away. It is also the honest reading of the
data: "where does this call belong" and "does a truck roll" are two different questions and they are
collapsed into one string.

**Projected gain:** call-level **0.852 → 0.93–0.96**, recovering the pre-rewording number without
giving back a single stranded caller.
**Cost:** ~1 hour, $0, no GPU. **Confidence:** high — I have named the line and the failing calls.

### 2. Lower the severity positive rate toward the real base rate

**What:** re-cap `severity_train.jsonl` below 40%; the natural rate for `needs_human` is far lower.

**Why:** the trained model over-fires (precision 0.692 and 0.240), the probabilities are saturated so
no threshold helps, and a shifted prior is the classic cause. The comment in `generate_severity.py`
already predicted this and capped at 40% — **40% was still too high.**
**Projected gain:** `is_safe_to_drive` precision **0.692 → 0.80-0.86**. **Cost:** `--rebalance-only`,
no API calls. **Confidence:** medium-high on direction.

### 3. Re-measure acceptance on the held-out split

**What:** retrain (v13) and score against `acceptance_dev.jsonl`.

**Why:** the current number is void — it was memorisation. The base model on the dev split is a
legitimate zero-shot reading worth having regardless.
**Projected gain:** unknown, and that is the point — **we have never measured this.**
**Cost:** free GPU, one run. **Confidence:** high it lands between 0.554 and 1.000; I will not guess.

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

1. **Is the rewording trade the right one?** It catches **4 more stranded callers** (0.833 → 1.000
   recall) and costs **3 misrouted calls** out of 27, because a false "unsafe" overwrites the routing.
   I would keep the safety behaviour and fix the override — I think sending a truck to someone who
   was fine is a smaller harm than leaving someone at the roadside, and the routing cost is a bug
   rather than a necessary price. But it is your call, and it is the one place where "more accurate"
   and "safer" genuinely pull apart.

2. **How much is the booking demo worth?** The acceptance classifier is trained and *unmeasurable*
   until we retrain on the held-out split. That is one GPU run away, but it is another run.

3. **Do we grow the test set?** 81 cases means one case is 1.23 points. Growing to ~150 would halve
   the interval — but it is more single-labeller labels, which is the constraint we are already
   fighting.

---

*Everything above is reproducible: `results/` holds the raw reports, `results/README.md` says which
are superseded, and every number came from a script that exits non-zero when a gate fails.*
