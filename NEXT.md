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

**An audit found that every fine-tune from v3 onward was scored partly on data it had trained on, and
my own guard could not have caught it because it checked the wrong files.**

v6 is still the best model we have built — clean, that is destination **0.951** and joint **0.914** —
and the v5 "regression" was still the epoch count rather than the four new training tasks. But the
leak correction costs the headline: against `deepseek-flash`'s 0.988 / 0.926, clean v6 is **1.2 points
behind on joint** rather than level with it. On 81 cases that is inside the noise, but the point
estimate no longer favours us, and "we tie deepseek" is no longer a claim I can make.

The audit also cut down two of my own explanations:

- The **training register** is a *hypothesis* about the safety false positives, not a demonstrated
  cause — stratification by length does not support it (3/7 false alarms at ≤9 words, 3/11 at 10-12,
  2/9 at ≥13).
- The **precision of 0.064** was a modelled quantity quoted as an observation, at an assumed base
  rate, from 8 false positives out of 27. The honest form is a range with the assumption stated.

The headline that survived every audit untouched: **`needs_human` recall went from 0.571 to 0.857 on
genuinely held-out data.** `sev-04` is the only leaked severity case and it is not a `needs_human`
positive, so that number is clean.

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

**Every fine-tune in that table is scored partly on data it trained on.** `gen-01` appears verbatim
inside a training row in the packaged snapshot of v3, v3-e8, v4 *and* v6, and all four answer it
correctly — the shape of memorisation. So the case is unusable, and the correction is to **exclude
it**, not to count it as wrong. That distinction changes the answer.

For our own arms the arithmetic is safe, because they are deterministic: dropping a case they got
right removes one from both the numerator and the denominator. On the 80 uncontaminated cases:

| | reported joint (81) | **excluding `gen-01` (80)** |
| --- | --- | --- |
| v3 | 0.864 | 0.863 |
| v4 | 0.914 | **0.913** |
| v5 | 0.876 | 0.875 |
| **v6** | 0.926 | **0.925** (74/80) |

**v6 is still ahead of v4, and that is the comparison that matters for the training change.** The
gap is ~1.2 points, same as before — because both arms lost the same leaked case.

### Against `deepseek-flash` the honest answer is "indistinguishable", not "behind"

Excluding `gen-01` from every arm and re-running gives:

| 80 cases, one run | dest | sub | joint | queue |
| --- | --- | --- | --- | --- |
| **v6** | 0.963 | 0.925 | **0.925** | **0.975** |
| `deepseek-flash` | **0.975** | 0.887 | 0.887 | 0.938 |
| `gpt-5.4-nano` | 0.963 | 0.875 | 0.875 | 0.925 |

**That is not a win, because it is not the same measurement.** `deepseek-flash` is non-deterministic
and answered differently in this run — it missed `gen-08` in the 81-case run and `gen-08` *and*
`det-04` here. Its joint has scored 0.889 to 0.926 across four runs on identical inputs; ours has not
moved once. So its 80-case number is a fresh sample, not its 81-case number minus one case, and the
two rows above cannot be subtracted from each other.

The defensible statement: **the two are indistinguishable at 81 cases, with the point estimate
swinging ±3.8 points run to run.** My earlier "1.2 points behind" was an artefact of counting the
leaked case as wrong — a what-if penalty, not a measurement — and the tie I then reached for is not
evidence either. `eval.py --exclude-cases` now makes the exclusion a reproducible measurement applied
to every arm, rather than arithmetic done by hand in a comment.

`sev-04` is likewise inside v6's severity snapshot six times. **Excluding it** — rather than assuming
the model would have missed it — leaves **17 of 17 uncontaminated hazards caught, recall 1.000**: the
leaked case cannot be measured at all, and "17 of 18" would have been a hypothetical penalty. It is
not a `needs_human` positive, so that question's numbers are unaffected.

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

### Move 1 — Separate the two questions the labels currently conflate

**This move has been through three positions and the audit settled it. The history is the point.**

I first proposed uncoupling dispatch from routing, and you approved it. Then I checked it against the
ground truth and withdrew it, because `call-no-start` and `call-flat-tire` reach `Roadside / Towing`
*only* through that override, so the change looked like it would break two stranded callers.

**The audit showed both positions were arguing about a label that cannot settle the question.**
`Roadside / Towing` in those cases is not evidence that the *owning* queue must be Roadside — it is
how the current labels encode a *combined* outcome. The ground truth stores "who owns this call" and
"does this caller need help now" in one field, so neither my fix nor my retraction was entailed by it.
Meanwhile the taxonomy's own stated principle says roadside is a flag, not a place the call goes —
which contradicts the labels.

Also worth reviewing: `call-vague` says *"the steering feels off"*, so calling its dispatch a false
positive is a judgement, not a fact.

**What:** define the two outputs separately — the **owning queue** (which department handles it) and
the **dispatch decision** (does a truck roll) — then relabel the call set against that, with
`call-vague` reviewed on its merits. *Then* decide whether the policy should change.
**Cost:** the relabelling is 27 cases by hand; no GPU, no API. **Confidence:** high that this is the
prerequisite, and I should not have formed a view on the policy before it.

### Move 2 — Test the register hypothesis, bounded

**What the check found.** The training data is **27 words** a line, the eval sets are **11**, and
hand-written caller turns are **5** — 77% of the routing eval is ≤12 words against 9% of what we
trained on. The negative class contains essentially no *short, bare statement of a fault that isn't a
hazard*, which is what the false positives look like:

```
"The air conditioning isn't blowing cold air any more."    unsafe, p=0.999
"The driver's seat won't slide forward any more."          unsafe, p=1.000
```

**But it is a hypothesis, not a cause, and the audit is right to say so.** Stratified by length, the
false alarms are **3/7 at ≤9 words, 3/11 at 10-12, 2/9 at ≥13** — length alone does not separate them,
and the model correctly rejects some short faults while falsely flagging a longer complaint. The
training data may still be why it learned the wrong thing, but I have not demonstrated it.

**What:** the 1,199 terse non-hazard faults are generated (~$0.33 so far) — a sensible experiment.
**The gate, fixed in advance:** `is_safe_to_drive` recall must stay at 17 of 18 on the *clean* hazard
set, and the false-alarm count must fall. If recall drops, the negatives were too aggressive and I
subsample them; if false alarms do not fall, the register was not the cause and this move is
abandoned rather than reworked.
**Confidence:** unknown, which is the honest word for it. It is worth one run *because* the test is
cheap and the alternative is guessing.

### Move 3 — Report precision honestly, and give the dispatch a control

**3a. Report precision as a range at a stated rate.** Our severity set is 40% positive because it was
built to measure recall, so its precision describes a world that does not exist. The corrected
version, done properly: the false-alarm rate is **8/27, Wilson 95% [0.159, 0.485]**, which at an
*assumed* 2% prevalence gives precision **0.040 – 0.114**. The prevalence itself has not been
measured. And note the code sets a flag and a queue — it does not send a truck; "most trucks roll for
nothing" was a model, stated as an observation.

**3b. Re-check the dispatch bar.** It is not a control today — the sweep is flat from 0.3 to 0.6. If
it stays flat once the probabilities de-saturate, accept that confidence cannot gate this decision
and design accordingly rather than reporting an operating point as though it were a choice.

### Move 4 — v13: one GPU run, with provenance

The first run that can measure what we actually ship:
- the **register experiment** from Move 2,
- the **fixed calibration** (`train_ddp.py` no longer ships the inherited map — v6 still carries it),
- the **acceptance split**, so the booking number is held out for the first time.

**Provenance is now automatic.** `kaggle_run.py watch` audits the packaged snapshot against the eval
sets at download and writes a manifest with a sha256 per file; a test fails if a leaky checkpoint has
no manifest. v13's numbers should therefore be quotable without an asterisk — which no checkpoint so
far has been.

**Gate:** leak audit clean, routing ≥ clean-v6 (0.951 destination), hazard recall ≥ 17 of 18.

### Move 5 — The ceiling: a second labeller, and a bigger test set

`det-04` is missed by every model; `gen-08` is answered against our label by all three. Where every
model disagrees with the key, the key is the likeliest thing to be wrong. **+1.2 to +3.7
destination** as a *measurement correction*, not a model improvement — and growing 81 cases to ~150
is the only way a 2-point delta stops being noise.

### Move 6 — Product completeness

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
- **Change the dispatch policy before the labels can answer the question.** Both my original fix and
  my retraction argued from a field that stores the owning queue and the dispatch outcome together.
  Relabel them as two outputs first (Move 1); until then there is no evidence either way.

---

## Open questions for you

1. **Confirm Move 1 is the prerequisite.** The audit's point lands: the call labels store the owning
   queue and the dispatch outcome in one field, so neither my fix nor my retraction was entailed by
   them. Relabelling 27 cases as two separate outputs is hand work with no API cost, and it settles
   a question I have now got wrong in both directions. I would do it before anything else.

2. **The register fix is an experiment, not the plan.** Stratifying by length does not support my
   causal claim (3/7 false alarms at ≤9 words, 3/11 at 10-12, 2/9 at ≥13). The data is generated and
   cheap to try, with a gate fixed in advance, but if you would rather not spend the GPU run on a
   hypothesis, the relabelling and the base-rate reporting are both free and certain.

3. **The trade from the rewording is still unanswered.** It catches **4 more stranded callers** and
   misroutes **3 calls**, because a false "unsafe" replaces the queue. I would keep the sensitivity
   and improve the precision. It is the one place where "more accurate" and "safer" pull apart.

4. **Is the booking flow worth one more GPU run?** Move 4 is ~50 minutes and free, and it is the
   first honest measurement of whether the switchboard can book.

5. **Do we grow the test set?** 81 cases means one case is 1.23 points, and the audit has just shown
   how much one case is worth. Growing to ~150 would halve the interval — but it is more
   single-labeller labels, which is the constraint we are already fighting.

---

*Everything above is reproducible: `results/` holds the raw reports, `results/README.md` says which
are superseded, and every number came from a script that exits non-zero when a gate fails.*
