# The plan

Where the project stands, what the measurements actually say, and the order I would do the rest in —
with honest estimates rather than confident-sounding ones.

Read `LEARNINGS.md` for the narrative. This is the decision document.

**The short version:** routing is done and ties the frontier model. **The safety surface is the weak
half**, and it was being flattered by a measurement — the dispatch fires on the wrong calls about
19 times in 20 at the rate a real switchboard sees. Fixing that is the next move, and the fix for the
*routing symptom* must not be mistaken for the fix.

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

## The thing that actually cost us

`is_safe_to_drive` does not just flag a call — **it overwrites the call's final queue**:

```python
# dealership.py
if roadside in flags or unsafe >= unsafe_threshold():
    queue = PROFILE.policy.get("roadside_queue", "Roadside / Towing")
```

For a genuinely stranded caller that is exactly right: it is how someone stuck on the highway gets a
tow instead of a booking. Two of our call cases depend on it — `call-no-start` ("my car won't start
at all") and `call-flat-tire` ("I'm stuck on the highway") reach Roadside **only** through this line.

For a false positive it throws a correct routing decision away. Every call-level failure is that:

```
call-collision     expected Body Shop        got Roadside / Towing   ("rear-ended me yesterday, I need body work")
call-glass         expected Body Shop        got Roadside / Towing   ("a stone cracked my windscreen, I need it replaced")
call-recall        expected Warranty Desk    got Roadside / Towing   ("I got a recall notice, I need to get it done")
call-vague         expected Service          got Roadside / Towing
```

None of those callers is stranded. **So the fault is not the override — it is that a classifier with
0.704 specificity is gating a high-consequence action.** I recommended separating the two and had to
withdraw it (see Move 1): the separation would have broken the two stranded calls above to fix these
four, and removed the truck-sending mechanism entirely.

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

## The plan

Estimates are ranges with reasoning. "Confidence" is how sure I am the *direction* is right.

**What "the best version" means here:** every call reaches the right queue, a truck rolls only when
someone is genuinely stranded, the booking flow works, and the numbers are defensible. Routing is
essentially done. **The safety surface is the weak half** — and one measurement was flattering it.

Ordered by (impact × confidence) ÷ cost.

### Move 1 — Fix the register of the training data

**This replaces a move I had to withdraw, and the withdrawal is worth reading.**

I recommended uncoupling dispatch from routing, and you approved it. Checking it against the ground
truth first killed it: **three call cases legitimately expect `Roadside / Towing`**, and for two of
them that queue comes *only* from the unsafe override —

```
queue_for('service','mechanical_diagnostic') = 'Service Department'   but call-no-start expects Roadside
queue_for('service','tires')                 = 'Tire Bay'             but call-flat-tire expects Roadside
```

— and the callers are "my car won't start at all" and "I'm stuck on the highway". Uncoupling would
have routed them away from the tow they need, broken two correct calls to fix four wrong ones, and
removed the mechanism that sends a truck. **It was the `front_desk` / `non_customer` mistake again: a
metric that improves because we stopped counting the fault.** Retracted.

**What the check actually found is bigger.** The training data is **27 words** a line; the eval sets
are **11**; hand-written caller turns are **5**. We trained on a register nobody uses — 77% of the
routing eval is ≤12 words, against 9% of what we trained on. And the negative class contains
essentially no *short, bare statement of a fault that isn't a hazard*, which is exactly what the
false positives are:

```
"The air conditioning isn't blowing cold air any more."    unsafe, p=0.999
"The driver's seat won't slide forward any more."          unsafe, p=1.000
```

**What:** generate training data in the caller's register — short, terse, first-person — and in
particular **non-hazard fault reports with no reassurance clause** ("the radio stopped working",
"the seat won't slide", against "the brakes failed", "smoke from under the hood"). The distinction
the model must learn is *which system, and whether it affects control, braking or visibility.*

**Why it is first:** every trained question inherits the mismatch, not just severity. It is the only
finding that explains the false positives, the saturation (a register shortcut is cleanly separable)
and why thresholds and priors did nothing.

**Cost:** ~$0.30-1.00 of teacher calls, then one free GPU run. **Confidence:** medium-high — the
mismatch is measured, the causal link is a testable hypothesis, and the test is cheap.

### Move 2 — Give the dispatch a control it can actually use

**2a. Report precision at a deployment base rate**, next to every safety number, forever. Our
severity set is 40% positive because it was built to measure recall; its precision describes a world
that does not exist. This is free and immediate.

**2b. Re-check the dispatch bar.** It is not a control today — the sweep is flat from 0.3 to 0.6. The
hope is that better data de-saturates the probabilities and makes the bar mean something again. If it
does not, accept that confidence cannot gate this decision and design accordingly rather than
reporting an operating point as though it were a choice.

### Move 3 — v13: one GPU run that consolidates everything

The first run that can measure what we actually ship:
- the **register-corrected** training data,
- the **fixed calibration** (`train_ddp.py` no longer ships the inherited map that overrode its own fit),
- the **acceptance split**, so the booking number is held out for the first time.

**Gates:** routing must hold at ≥ 0.926 joint; `is_safe_to_drive` recall must stay at 1.000 on the 18
hazards; the false positives should fall. If recall drops, the terse negatives were too aggressive.
**Cost:** one free 2×T4 run, ~50 min. **Confidence:** high that it measures; the numbers are unknown,
which is the point.

### Move 4 — The ceiling: a second labeller, and a bigger test set

`det-04` is missed by every model; `gen-08` is answered against our label by all three. Where every
model disagrees with the key, the key is the likeliest thing to be wrong. **+1.2 to +3.7
destination** as a *measurement correction*, not a model improvement — and growing 81 cases to ~150
is the only way a 2-point delta stops being noise.

### Move 5 — Product completeness

Answer "what time do you open?" from `facts` instead of transferring (loaded, unused, top repeatable
Fixed Ops volume). And decide what to do about the dead `other` fallback: the fine-tuned model never
abstains (0.0%), so there is no "I'm not sure" left anywhere in the system.

---

## What I would NOT do

- **More epochs, blindly.** v4 converged at 0.042 and v6 at 0.036, and v12 settled the epoch
  question: 8 epochs with all five tasks is the best checkpoint we have. It is not a knob to keep
  turning.
- **Tune a threshold to fix the safety precision.** The sweep is flat from 0.3 to 0.6 for a reason:
  the model is confidently wrong. A threshold on a saturated distribution is a no-op — and the
  threshold went *up* while the false alarms went *up*, which is what that looks like in practice.
- **A bigger teacher.** Measured: `deepseek-v4-pro` agrees with our labels *less* than flash at 3×
  the cost.
- **Trust any single run.** One case on 81 is 1.23 points; the LLM arms moved 2.5 points between
  identical runs.
- **Uncouple dispatch from routing.** Withdrawn after checking it against the ground truth: two
  genuinely stranded callers reach `Roadside / Towing` only through that override, so the change
  would have broken correct behaviour while improving the metric. The `front_desk` / `non_customer`
  mistake again — and the reason I now check a structural fix against the three cases it affects
  before recommending it.

---

## Open questions for you

1. **Confirm the register fix is the priority.** The evidence is that our training data is 27 words
   a line where callers speak 5-11, and that the negative class contains no short non-hazard fault
   statements — which is precisely what the false positives are. It is a hypothesis with a clean
   test, and it is a bigger project than the one I proposed before it. If you would rather I take a
   smaller, surer step first (the base-rate reporting, or the held-out booking measurement), say so.

2. **The trade from the rewording still stands unanswered.** It catches **4 more stranded callers**
   and misroutes **3 calls**, because a false "unsafe" replaces the queue. I would keep the
   sensitivity and improve the precision — leaving someone at the roadside is a worse failure than
   sending a truck to someone who was fine. But it is the one place where "more accurate" and
   "safer" genuinely pull apart.

3. **Is the booking flow worth one more GPU run?** Move 3 is ~50 minutes and free, and it is the
   first honest measurement of whether the switchboard can book. If the demo matters, it does.

4. **Do we grow the test set?** 81 cases means one case is 1.23 points and a 2-point "win" is noise.
   Growing to ~150 would halve the interval — but it is more single-labeller labels, which is the
   constraint we are already fighting.

---

*Everything above is reproducible: `results/` holds the raw reports, `results/README.md` says which
are superseded, and every number came from a script that exits non-zero when a gate fails.*
