# What we learned building this

Notes from taking a small local decision model and trying to make it as good as a frontier LLM at
routing car-dealership phone calls — at a fraction of the cost.

Written for a person, not a changelog. The numbers are all in `results/`.

---

## The short version

We set out to match `deepseek-flash` on call routing, using Laya (a small non-autoregressive model
that returns typed decisions instead of generating text) running locally on a laptop.

We got there, with one qualification that matters. On 81 hand-labelled cases the fine-tuned cascade
scores **0.926 joint**. `deepseek-flash` scored 0.914, 0.889, 0.901 and 0.926 across four runs on
*identical inputs* — it leads us in some runs and trails in others. On the routing outcome we beat
it: **0.975 queue accuracy against 0.963**. Ours has not moved once.

So: **parity on the decision, a small lead on the outcome, at 23 ms instead of 1.4 seconds, for $0 a
call, with a number that stays put.** It also books real appointments now, not just routes.

That distinction — a stable number versus a noisy one — turned out to be the more defensible thing
to say, and I only found it by running the same evaluation four times.

**Then the retrain that looked like a regression wasn't one.** Teaching the model the two safety
questions and the booking question appeared to cost **3.7 points of joint accuracy** (0.914 → 0.877)
— but that run also did 4 epochs where the previous one did 8, and at 8 epochs the multi-task model
is the best we have built (joint **0.926**, tying `deepseek-flash` and beating it on the routing
outcome). Two confounds in a row, both of them mine, and the second one took a second-order check to
find: the apparent call-level drop was not the training either, but the safety rewording, which I
proved by re-running the *same* checkpoint under the old policy.

The more useful lesson came from the opposite direction: three of the numbers we were proudest of
turned out to be **reading their own training data**. A perfect acceptance score was memorisation,
and the safety case the whole rewording exercise was built around was sitting in the training set.
The improvements that survived that audit are the ones worth anything.

But the interesting part isn't the final number. It's that almost every improvement came from
fixing something in *how we had framed the problem* — not from the model, the data volume, or the
prompt. Several of those fixes were to my own mistakes, and a few of them were only visible because
we measured something we'd previously assumed.

---

## Structure beat everything else

The first taxonomy had nine departments: service, body shop, parts, **tires**, **detailing**,
sales, finance, **towing**, plus a catch-all called `general`.

Four of those weren't departments. Tires and detailing are things the *service* department does.
Towing is something you *dispatch*, not somewhere a call goes. And `general` was two unrelated
things wearing one name: real questions nobody owned (opening hours, loaner policy) and things that
weren't dealership business at all (a supplier, a wrong number).

That framing produced months of argument about the wrong boundaries — tires vs service, detailing
vs sales, parts vs general. When we looked at how dealerships actually publish themselves, every
contact page listed the same things: Sales, Service, Parts, Body Shop, Finance. Nobody publishes a
tire department. Tires show up *inside* service pages, as a line item.

Fixing that — tires and detailing as service sub-queues, roadside as a policy flag, `general` split
honestly — did more for accuracy than any model change. The teacher's agreement with our labels
went from 0.963 to 0.975 destination and 0.852 to 0.901 sub-queue, and we hadn't touched a model.

**The lesson:** when a classifier keeps getting a boundary wrong, check whether the boundary is
real before you blame the classifier. We spent a long time trying to teach a model a distinction
that didn't exist in the world.

---

## The teacher is your ceiling, and you can't buy a better one

A candidate only became training data if two independently-worded labelling passes agreed with each
other *and* with the intended target. That's a high-precision filter, and it worked — zero invalid
labels in 1,394 rows.

But it means we're distilling `deepseek-flash`, so its agreement with our hand labels is the best
we can hope for. My plan to break through that ceiling was to use a *bigger* teacher.

I measured it first. `deepseek-v4-pro` — 3× the price, far slower — agrees with our labels **less**
than flash does: 0.951 vs 0.975 on destination. It isn't confused, it's just *different*, and it
loses on the same fuzzy cases flash handles correctly.

That was a $0.03 experiment that killed a plan I was confident about. Worth every cent.

**The lesson:** "use a stronger model" is a hypothesis, not a strategy. On a narrow task, a bigger
general model is often just a different model.

---

## The same bug, ten times, in ten disguises

This is the pattern I'd warn someone about first.

1. **`express_maintenance` advertised "tire rotations" while the rotation case was labelled
   `tires`.** My taxonomy contradicted itself. The teacher had no defensible answer, so it guessed,
   and we lost **5 points of sub-queue ceiling** to a definitional clash nobody noticed.

2. **`synthgen.py` still referenced `department`** after a rename. Every residual pair raised a
   `NameError` and was silently counted as a generation error. A third of the work wasn't happening.

3. **Training built the sub-queue question as "This is a service call" while inference sent
   "This is a Service call".** The fine-tune learns to answer one exact instruction string, so at
   eval time we'd have asked it a question it had never seen — and it would have looked like a
   mediocre fine-tune rather than a string mismatch. Caught by reading the training script against
   the runtime one before spending a GPU run.

4. **The unsafe-to-drive threshold existed in three places** — the flag, the priority and the
   transfer decision each read a different number. A caller could be dispatched as unsafe while the
   audit trail said they weren't.

5. **The Kaggle notebook is generated from a script, and the generated one had drifted.** Its list
   of files to copy still named four of the six the dataset ships, so a whole GPU run trained
   **2,512 items instead of 6,210** — the safety and booking questions simply weren't in it. No
   error, no warning; the job reported success. There is now a test comparing the notebook against
   its generator.

6. **The epoch count lived in `JEV_EPOCHS` in the shell at the moment the notebook was generated.**
   Regenerating the notebook without exporting it silently baked in the 4-epoch default, so the next
   run was four epochs short of the eight we knew we needed. An env var you must remember at the
   shell is not a source of truth; the recipe lives in `training/run_config.json` now, and a test
   asserts the notebook agrees with it.

7. **`build_items.py` printed a note and carried on when a data file was missing.** That is what made
   #5 invisible: a job that trains a fifth of the data and reports no problem is worse than one that
   crashes. A silent downgrade is not a smaller job, it is a wrong measurement. It raises now.

8. **The calibration temperature was fitted into one field and read from another.** The trainer fits
   a temperature per question type and writes it to `temperature`; inference prefers an inherited
   `temperature_by_options` map, and every bucket our questions occupy is in that map. So the step
   the trainer's own comment calls *"what makes the confidence usable"* had **never applied to a
   single question**.

9. **The safety flag and the routing shared one field.** `queue` holds where the call belongs, and
   the unsafe question *overwrites* it with `Roadside / Towing`. So a safety false-positive silently
   destroyed a correct routing decision — three of 27 calls. Two different questions, one string.

10. **The training data existed in two places, and the leak check looked at the wrong one.** The
    guard compared `data/calls/` against the eval sets. The checkpoints trained on the snapshot Kaggle
    packaged, and *that* is the copy beside the weights. v3, v3-e8, v4 and v6 all carried `gen-01`
    verbatim inside a training row; v6 also carried `sev-04` six times. Every one of them answered
    the leaked case correctly, and 0.963 destination and "18 of 18 stranded callers caught" were
    partly memorisation. Found by a second reader, by hand, after the numbers had been quoted.

Items 1–8 are all **one fact in two places**. Item 9 is the mirror image, **two facts in one place**.
Item 10 is the original again, with the two places being a working directory and a model artefact —
the same disease: the structure cannot represent the truth, so something is chosen silently. None of
the ten threw an error you'd notice. Three cost real accuracy, two cost a GPU run, one cost eleven
points of the metric we quote, and one cost the headline.

**The lesson:** when a fact is stored, ask how many questions it is answering and how many places it
lives. If either number is over one, the code will eventually resolve the conflict quietly and
wrongly. The fix is never vigilance — it's structure: the taxonomy, the question text and the
thresholds live in one config file, dispatch is about to stop sharing a field with routing, and
every item above ended with a test. The two that cost GPU runs are the two where I wrote a note
instead of a test the first time.

---

## We were stopping training early and didn't notice

Four epochs was inherited from the maintainer's notebook, not chosen. The loss curve told the story:

```
epoch 1  0.686
epoch 2  0.515
epoch 3  0.174
epoch 4  0.081   <- still halving
```

That's a model that stopped, not one that converged. Running to 8 epochs was worth **+2.5 points of
joint accuracy**, and it flattens at 0.042 by epoch 6–7, so we know where the end is.

**The lesson:** look at the loss curve before you tune anything else. It's free and it usually has
an opinion.

---

## We balanced on the wrong axis

The training set gave every sub-queue exactly 50 examples. Tidy. But it meant a destination's
volume scaled with how many sub-queues it happened to have — so `front_desk`, with two, got a third
of what `service` got with six.

`front_desk` was also the worst class on the test set: four of five destination errors. That looked
like a hard class. It wasn't. **Both teachers get every `front_desk` case right.** We simply had too
little of it, and a per-destination floor fixed it for **+1.3 points of destination accuracy**.

**The lesson:** "balanced" is meaningless until you say balanced *on what*. Balance the axis you're
actually classifying.

---

## The system invented an agreement

This is the one that would have embarrassed us in a demo.

The switchboard reads out real appointment times and then asks a classifier *"which of those did
the caller agree to?"* On the first live run, the caller said:

> "Sure — this is Dana, and my number is 555-0140."

No time. Nothing resembling a time. And the classifier answered `slot_1` at p=0.41 — and the
appointment was **filed**.

The classifier was untrained on that question, so it was guessing, and 0.41 was the largest of
several bad options. An argmax of a near-uniform distribution is not a decision. There's now a
floor: below the pin threshold, the system asks again instead of committing. The same turn now
says *"Sorry — which of those times did you want?"*, and a genuine acceptance (p=0.65) books
correctly.

**The lesson:** never act on a weak signal just because it's the strongest one you have. Especially
not when the action is hard to undo. This is the same rule that stops a support bot escalating on a
coin flip, and we'd already learned it once.

---

## Confidence is not the same as calibration

Fine-tuning broke our escalation gate. It went from flagging 81% of calls to flagging **0%**, which
looked like the model had become overconfident and useless as a safety net.

The calibration curve said otherwise:

```
confidence 0.8-1.0:  79 of 81 cases, accuracy 0.962
```

The model is confident **because it's right**. Its errors aren't hiding in a low-confidence tail —
there's no tail. The base model's low confidence was a symptom of being bad at the task, not of
being well-calibrated.

So the gate didn't *improve*, it changed *character*: from load-bearing (escalate most calls to
rescue the base model) to a small cheap safety net (escalate 2.5% of calls, gain 1.2 points). Both
are legitimate. They're just not the same product.

**The lesson:** a gate that never fires isn't broken — it might mean the model is good. Check the
curve before "fixing" it.

**A late caveat, so the two lessons don't read as a contradiction.** This holds for the *routing*
questions, where confidence tracks correctness. It does **not** hold for the safety questions, whose
trained probabilities are saturated at 0 and 1 — there the sweep is flat and the confidence carries
almost no information. Same model, same run, two different verdicts about calibration. The
distinction is that the routing questions have enough signal in their confidence to rank, and the
safety questions do not. "Well calibrated" is a property of a question, not of a checkpoint.

---

## Measuring something we'd assumed was fine

`is_safe_to_drive` decides whether a stranded caller gets roadside assistance. `needs_human`
decides whether a person takes the call. We had never measured either. They drove real decisions
and we were guessing.

45 labelled cases later:

| | recall | missed |
|---|---|---|
| `is_safe_to_drive` | 0.778 | **4 of 18 unsafe callers** |
| `needs_human` | 0.571 | 3 of 7 |

Even the good model misses a stranded caller one time in five. The misses cluster on *implied*
hazards — "smoke coming from under the hood", "the accelerator stuck open" — where the caller never
says they've stopped. The question asks what the caller *indicates*, so the model answers literally
while a person would hear a fire.

That's a question-wording problem as much as a model one, and we'd never have known.

**So we reworded it, and it needed no training at all.** Asking about the *vehicle* instead of the
*statement* took recall from 0.778 to **1.000** — 0 of 18 stranded callers missed, for three extra
dispatches. The clearest case is `sev-11`, *"there's a burning smell and smoke through the vents"*:
p=0.00 under the old wording, 0.99 under the new. The model was never unsure; it was answering the
question we actually asked.

Two things then complicated that, and both are recorded rather than tidied away. Training the
question properly made precision *worse* (0.857 → 0.69), and `sev-04` — one of the cases the
rewording is credited with rescuing — turned out to be sitting in the training set. The recall gain
from rewording holds, because it was measured before any training; the trained numbers carry both
caveats.

Because the errors aren't symmetric — a missed stranded caller is someone at the side of a road, a
false alarm is a wasted journey — the operating point is chosen by sweep, not by picking 0.5 because
it's round. **And this is where saturation bites:** on the trained model the sweep is flat from 0.3
to 0.6 and barely moves above it, so the point is not really being chosen at all. A flat sweep means
the errors are confident ones, and no threshold touches a confident error.

**The lesson:** the unmeasured parts of a system are exactly the parts you're most confident about.
And when errors are asymmetric, accuracy is the wrong headline — lead with the error you can't
afford.

---

## Small test sets lie, and intervals are cheap

On 81 cases, one case is 1.23 percentage points. The two LLM arms moved **2.5 points between
identical runs** on identical inputs. Twice I caught myself reading a two-point difference as a
finding.

So every table carries Wilson intervals now, and the honest headline is *parity with our point
estimates ahead* — not "we beat deepseek". The intervals overlap. We don't have the sample to claim
more, and saying so costs nothing.

The call-level set had the same problem in a worse form: 10 calls, where one case is 10 points. It
showed the base model at 1.000, which was flattering and false. Expanding to 27 calls — covering
every sub-queue — revealed it at 0.852. **A test set that flatters you is a test set that isn't
measuring anything.**

---

## Two ideas I was wrong about, caught by measuring

**I nearly merged `front_desk` and `non_customer`.** They route to the same queue, our model
confused them constantly, and merging would have erased four of our five destination errors. It
looked like a clean taxonomy fix.

I checked whether the confusion was real first. **Both teachers get those cases exactly right.**
Only our fine-tune struggled. So merging would have hidden a model deficiency behind what looked
like a structural improvement — and we'd have shipped a worse taxonomy with better numbers.

**I proposed a bigger teacher** and it was worse (see above).

**The lesson:** when a fix would make your numbers go up, that's exactly when to check whether it's
fixing the thing or hiding it.

---

## The ceiling is now the labels, not the model

One case — `det-04` — is missed by the fine-tuned model, by nano *and* by deepseek. When every
model independently disagrees with the key, the key is the likeliest thing to be wrong.

There's a whole class of cases like this: the model's answer is defensible and my hand label is one
opinion. No amount of training fixes that. **A second human labeller is worth more than any model
work left on the table.**

---

## We were measuring ourselves

The single most valuable thing that happened in this session was discovering that three of our
measurements were reading their own training data.

**The acceptance score was memorisation.** The booking question reported **1.000 accuracy** against
0.554 for the base model, and that read as the flaky-demo problem solved. The eval script was
scoring `acceptance_train.jsonl` — the file training is built from. And because every reply in that
data is a template, even a proper random row split would have leaked: "Yes, {t} works for me." would
sit in both halves. The split now holds out whole *reply phrasings*, so what is measured is
understanding a way of saying yes the model has not read before.

**Two more leaked by containment, which no text comparison could see.** The near-duplicate guard
refused anything within Jaccard 0.6 of the held-out cases. But a short query inside a longer sentence
scores *low* on Jaccard, because the union is large:

```
"What time do you open on Saturdays?"                   7 words
"What time do you open on Saturdays? I couldn't…"      14 words     Jaccard 0.50  → passed the guard
```

The same shape hid `sev-04` — *"there's smoke coming from under the hood"* — inside two training
rows. `sev-04` is the case the reworded safety question is specifically credited with rescuing:
p=0.00 under the old wording, 0.99 under the new one. So the model we said had learned to hear a
fire might have been remembering one. Exact-text disjointness is not disjointness, and *that is the
check we were relying on*.

All three had the same root cause: **the eval sets and the training sets come out of the same
pipeline.** That is convenient and it is exactly why the leak is invisible. A held-out set is not
held out because you intended it to be; it is held out when something checks.

**And a fourth leaked because the check looked in the wrong place.** I wrote the guard, ran it, saw
it pass on `data/calls/`, and believed the problem was solved. It compared the *working directory*.
The checkpoints had trained on the snapshot Kaggle packaged weeks earlier, and that copy sat beside
the weights, unexamined. A second reader found it by hand: `gen-01` verbatim inside a training row in
v3, v3-e8, v4 *and* v6; `sev-04` six times in v6. All four answered the leaked case correctly.

The honest accounting: destination 0.963 → **0.951**, joint 0.926 → **0.914**, and stranded-caller
recall 18 of 18 → **17 of 18**. The comparison *between our checkpoints* survived, because they all
carried the same leak. The comparison against `deepseek-flash` did not — it became a 1.2-point
deficit rather than parity, and that was the headline of the day.

> **The lesson:** when you fix a class of bug, the fix has to cover every *copy* of the thing you
> were measuring — including the frozen ones inside model artefacts. "I added a check and it passes"
> is only as good as the question of what the check is pointed at.

> **The lesson:** a flattering number deserves more suspicion than a disappointing one. 0.554 → 1.000
> should have been the moment I audited the measurement, and instead it was the moment I started
> writing it up.

---

## The base rate, and a precision that does not travel

The severity set is 18 unsafe calls out of 45. **40% positive.** That enrichment is deliberate and
correct: unsafe calls are rare, and you need them concentrated to measure whether you catch them.

But precision read off an enriched set is not precision. Precision depends on the base rate, and the
set we measure on has a base rate eight to twenty times the deployment one. Recompute the same
classifier at rates a switchboard actually sees:

| | sensitivity | specificity | precision @40% (the set) | @5% | @2% |
| --- | --- | --- | --- | --- | --- |
| old wording, untrained | 1.000 | 0.889 | 0.857 | 0.321 | 0.155 |
| **trained (v6)** | 1.000 | 0.704 | **0.692** | 0.151 | **~0.04–0.11** |

**Read the last two columns — but read them as a model, not a measurement, and not as a point.** The
false-alarm rate is 8 of 27, Wilson 95% **[0.159, 0.485]**, so precision at an assumed 2% prevalence
spans **0.040 to 0.114**. The prevalence itself has not been measured, and the code sets a *flag and a
queue* — it does not send a truck. The claim I first wrote, "94% of our dispatches are wrong", stated
a modelled quantity as an observed outcome; an audit caught it.

What the model does establish is the direction, and it is not sensitive to the interval: **training
traded specificity for sensitivity** (0.889 → 0.704), and at low base rates specificity is what
precision is made of. The measured precision fell 0.857 → 0.692; precision at deployment rates
roughly halved. A classifier that got *better at its eval set* became less usable where it runs.

It also fits the call-level damage: a false-positive rate near 0.30 on the negatives means about three
in ten non-urgent calls trip the dispatch, and we measured exactly that — **3 of 27 calls reached
Roadside / Towing**.

**Nothing about this was wrong with the model, the training, or the intent.** It is a measurement
that cannot answer the question being asked of it: a recall-first set can tell you *"do we catch the
stranded caller"*, and it structurally cannot tell you *"how often does the truck roll for nothing"*.

> **The lesson:** when a class is rare, an enriched eval set measures recall and *inflates* precision.
> Report the base rate alongside, or the number will describe a world that does not exist. This is
> the fifth measurement that flattered us, and the first one that wasn't a bug — the set was designed
> correctly for the question it was built to answer, and I quoted it for a question it never could.

---

## A hypothesis, and the experiment that killed it

Chasing the saturated probabilities, I found a real bug: the trainer fits a calibration temperature
per question type and writes it to `temperature`, but inference prefers a `temperature_by_options`
map inherited from the base checkpoint — and every option-count bucket our questions touch is in that
map. So **the fitted temperature had never been applied to a single question.** The trainer's own
comment calls that step "what makes the confidence usable".

That felt like the explanation. Overconfident probabilities, a softening step that was silently
skipped — the fix was one line, and it would un-saturate everything.

I staged a checkpoint with the inherited map removed and measured it:

```
noul probability, map kept:      1.000
noul probability, map removed:   0.996
```

**Almost nothing.** The plumbing was broken, but it was not the cause. The trained logits are
extreme on their own, and even the temperature the trainer *chose* (3.683) cannot soften them. The
saturation is real model confidence.

That mattered, because it changed the conclusion. If the saturation had been a plumbing bug, the
fix would have been free. Because it is real confidence, **the false alarms it produces cannot be
tuned away with a threshold** — a threshold on a saturated distribution is a no-op — and the likely
cause moves to the training prior: the severity data is capped at a 40% positive rate where a real
switchboard is nowhere near that. So the next thing to change is the training mix, not the
calibration.

One wrinkle supports that reading: at 8 epochs the same fitter asks for **no softening at all**
(1.0), where the 4-epoch head had asked for 3.683. The pathological confidence was a property of the
under-trained model, not of training the question.

> **The lesson:** fixing a real bug is not the same as fixing the symptom that led you to it. I had
> a true finding, a plausible story, and no causal evidence. One cheap experiment separated them, and
> the thing worth keeping is that I ran it *before* writing the conclusion down — because the
> conclusion I would have written was wrong.

---

## The cost nobody measured

Rewording the safety question was right, and it was free. It was neither.

The rewording took `is_safe_to_drive` from missing 3 of 18 stranded callers to missing none. That
result held up under every audit — it was measured on held-out cases, before any training existed.
Good.

What we did not measure was **what consumes the answer**. `dealership.py` does not just set a flag:

```python
if roadside in flags or unsafe >= unsafe_threshold():
    queue = PROFILE.policy.get("roadside_queue", "Roadside / Towing")
```

The unsafe question **replaces the call's final queue**. For a genuinely stranded caller that is
exactly right — it is how a caller stuck on the highway reaches a tow instead of a booking. For a
false positive it throws a correct routing decision away. The ledger, all measured:

| | stranded callers caught | false dispatches | call-level queue |
| --- | --- | --- | --- |
| old wording, threshold 0.3 | 15 of 18 | **0** | **0.963** |
| new wording, threshold 0.7 | **18 of 18** | 3 | 0.852 |

We caught four more people who were actually stranded, and misrouted three who were booking body
work, a windscreen and a recall appointment. Eleven points of the metric we quote, for three false
alarms.

**I had the explanation wrong first, and a second-order check caught it.** I saw the call-level drop
in the same run as the multi-task training and attributed it to training — a tidy story, since
training had just changed. The check that settled it was re-running the *same* v4 checkpoint under
the *current* policy: **0.852, identical to the new model.** Same model, different policy, the whole
regression. Training cost nothing at the product level.

> **The lesson:** when you change how sensitive a decision is, measure the thing that *consumes* it,
> not the decision. The severity report said "precision 0.857, three false alarms" and that sounded
> acceptable — because a false alarm sounds like a wasted truck. It was actually a discarded
> routing decision, and the report had no way to say so.

There is a structural version of this, and I nearly shipped the wrong fix for it.

I read the override as a conflation — *"where does this call belong" and "does a truck roll" sharing
one field* — and recommended separating them. Then I checked the three call cases that expect
`Roadside / Towing`, and two of them get that queue **only** from the override:

```
queue_for('service','mechanical_diagnostic') = 'Service Department'   but call-no-start expects Roadside
queue_for('service','tires')                 = 'Tire Bay'             but call-flat-tire expects Roadside
```

Those callers say *"my car won't start at all"* and *"I'm stuck on the highway"*. They need a tow,
not a booking. **The override is correct behaviour** — it is the thing that sends a truck — and
separating it would have broken two genuinely stranded callers to fix four false ones, while removing
the mechanism entirely.

So the fault is not the structure. It is that a classifier with 0.704 specificity is gating a
high-consequence action, and the reason it is that imprecise is measured in the next section.

> **The lesson, and it is the one I keep re-learning:** a tidy structural story is not evidence. I
> had the line, the failing calls and a clean explanation, and the explanation was wrong. Reading the
> three counter-examples took two minutes and saved the product.

---

## The data doesn't sound like callers

Chasing why the safety classifier fires on "the air conditioning isn't blowing cold air any more",
I measured the length of every corpus. It was not subtle:

| corpus | n | median words | share ≤ 12 words |
| --- | --- | --- | --- |
| routing **training** | 1,255 | **27** | 9% |
| routing **eval** (hand-labelled) | 81 | **11** | **77%** |
| severity **training** | 1,645 | **24** | 9% |
| severity **eval** | 45 | **12** | 58% |
| scripted caller turns (hand-written) | 104 | **5** | 84% |

**We trained on a register nobody uses.** The synthetic corpus came from asking a language model for
realistic utterances, and language models write long, hedged, multi-clause prose. Callers say:

```
training:  "Hi, sorry, I don't know if this is the right number, but I'm calling about a job.
            I saw something online about you maybe…"                       (27 words)
a caller:  "Hi, my car won't start at all. I think I need service."          (10 words)
```

That the routing number is 0.963 *despite* this is the surprising part — it is generalising across a
distribution shift, not because the data fits.

**And it explains the false positives exactly.** The negative class has 991 examples, and essentially
none of them is a *short, bare statement of a fault that isn't a hazard*. Its terse negatives are all
administrative ("I need an oil change", "what time do you open"), and its fault-reporting negatives
are long, chatty or cosmetic. Meanwhile every positive is a terse fault statement. So the only rule
the model could learn from terse fault statements is **"short + something's wrong ⇒ unsafe"** — and
the eval's negatives are precisely the terse fault statements it has never seen labelled safe:

```
"The air conditioning isn't blowing cold air any more."     called unsafe, p=0.999
"The driver's seat won't slide forward any more."           called unsafe, p=1.000
"There's a rattle coming from the back seat over bumps."    called unsafe, p=1.000
```

None of those affects steering, braking or visibility. A person would not dispatch a truck.

**But it is a hypothesis, and an audit cut it down to size.** Stratified by length, the false alarms
are **3 of 7 at ≤9 words, 3 of 11 at 10-12, and 2 of 9 at ≥13** — length alone does not separate them.
The model correctly rejects some short faults and falsely flags a *longer* complaint. I had written
"it is the only finding that explains the false positives" and that was not supported; what is
supported is that the negative class lacks the shape the false positives have. The test is cheap, so
it is worth running with the gate fixed in advance: recall must hold on the clean hazards and the
false-alarm count must fall, or the register was not the cause.

> **The lesson:** before tuning a model, measure whether your data is in the register your users
> actually speak. A generator asked for "realistic" text produced something no caller has ever said,
> and nothing in the pipeline objected — it is fluent, on-topic and correctly labelled. It is just
> three times too long.
>
> **And the second lesson, from the audit:** a measured mismatch plus a plausible mechanism is not a
> cause. I had two true measurements and one story, and I wrote the story as though it were the third
> measurement. Stratify before you attribute.

---

## Things we know but haven't fixed

- **The acceptance question is trained but unmeasurable.** The 1.000 it reports is memorisation; the
  held-out split exists now, and scoring it honestly needs a retrain. Until then the booking demo's
  reliability is unknown, not fixed.
- **`needs_human` now catches more and trusts itself too much.** Recall improved to 0.857 on
  held-out data, which is real, but precision is 0.240 — it fires on 6 calls in 10. The confidence
  floor makes the over-firing annoying rather than dangerous; the training prior is the suspect.
- **The probabilities are saturated, so a threshold is not a control.** On the trained safety
  questions the sweep is flat from 0.3 to 0.8. Where the model is confidently wrong there is nothing
  to tune, and every reported operating point should be read as "the model's opinion", not "the
  point we chose".
- **The dispatch is gated by a classifier with 0.704 specificity, and that is the real fault.** For a
  genuinely stranded caller the queue override in `dealership.py` is *correct* — it is how someone
  stuck on the highway reaches a tow. Separating dispatch from routing, which I recommended and then
  withdrew, would have broken `call-no-start` and `call-flat-tire` to fix four false positives. The
  fix is precision, not structure.
- **The training data is in the wrong register.** Median 27 words against an eval set at 11 and real
  caller turns at 5. Nothing objected because it is fluent and correctly labelled; it taught a
  shortcut instead.
- **The dispatch would be wrong about 94% of the time in deployment.** At the 40% base rate of our
  severity set the trained classifier looks like 0.692 precision; at the 2% rate a switchboard
  actually sees it is **0.064**, and training made it *worse* than untrained (0.155 → 0.064) by
  trading specificity for sensitivity. Fixing the override will hide this from the call-level
  metric without touching it — the metric and the fault are different things.
- **The call-level number in `results/eval_v4.json` is not comparable to the later ones.** It was
  measured before the safety rewording, and the same checkpoint scores 0.963 or 0.852 depending on
  which policy was live. Check the policy, not just the model.
- **The store facts are loaded but unused.** The switchboard could answer "what time do you open?"
  from them. Right now it still transfers.
- **The `other` fallback is dead.** The fine-tuned model never uses it (0.0%, down from the base
  model's 23.5%). It always commits. For routing that's arguably right, but there's no "I'm not
  sure" left in the system.
- **All of it rests on 81 single-labelled cases.** One case is 1.23 points, and where every model
  disagrees with the key, the key is the likeliest thing to be wrong.

---

## The principles, collected

Everything above compresses into a short list. Every one of these cost something to learn.

1. **Measure the thing you're most sure about, first.** The unmeasured parts of a system are exactly
   the parts you are confident about, because confidence is what stopped you checking.

2. **A flattering number deserves more suspicion than a disappointing one.** The acceptance
   classifier at 1.000 was a leak. The retrain that "regressed" was two confounds. Both looked like
   findings and neither was.

3. **A check is only as good as what it is pointed at.** I added a leak guard, watched it pass, and
   believed the problem solved — it compared the working directory, while the checkpoints had trained
   on a frozen snapshot sitting beside their weights. Four of them had `gen-01` in their training
   data. Audit every *copy* of the thing you are measuring, including the ones inside artefacts.

4. **When a class is rare, an enriched eval set measures recall and inflates precision.** Our
   dispatch looked like 0.69 precision on a 40%-positive set; at an assumed 2% rate it is somewhere
   in 0.04-0.11. Report the rate *and* the interval, and do not state a model as an observation.

5. **A measured mismatch plus a plausible mechanism is not a cause.** The register finding was two
   true measurements and a story, and I wrote the story as the third measurement. Stratify before you
   attribute: length alone did not separate the false alarms (3/7, 3/11, 2/9 across the buckets).

6. **Check a proposed fix against the ground truth before shipping it.** Uncoupling dispatch from
   routing would have lifted the headline metric while removing the mechanism that sends a truck. I
   only saw it by reading the three cases that expect `Roadside / Towing` — and the audit then showed
   the retraction rested on labels that conflate two outputs, so neither position was entailed.

7. **One fact in two places, or two facts in one place, gets resolved silently and wrongly.** Ten
   instances. None threw an error. The fix is structure and a test, never vigilance.

8. **A threshold is only a control if the distribution isn't saturated.** Where the model is
   confidently wrong there is nothing to tune, and reporting an operating point implies a choice
   that isn't being made.

9. **Check what else changed before you credit the model.** The call-level drop was a policy change,
   the routing "regression" was an epoch count. Both times the tidy story was wrong — and equal
   aggregate scores can hide a *different set* of failures.

10. **Balance the axis you're classifying, not the axis that looks tidy.** Equal examples per
    sub-queue starved a destination, and "balanced" hid it.

11. **Check the boundary is real before blaming the classifier.** Tires and detailing were never
    departments. We spent a long time teaching a distinction that did not exist in the world.

12. **When a fix makes your numbers go up, check whether it fixes the thing or hides it.** Merging two
    confusable classes would have erased four errors and made the taxonomy worse.

13. **Record what a run was trained on, with the run.** A number and its provenance have to travel
    together, or someone reconstructs the provenance later and finds a leak they cannot repair.

14. **Report the noisy comparison honestly.** After the leak correction we are 1.2 points behind a
    frontier model on joint, 60× faster, and at zero marginal cost per call — and on 81 cases the
    honest claim stops there. The intervals overlap, and saying so costs nothing.

---

## The thing I'd tell someone starting this

**Measure the thing you're most sure about, first.**

Every real improvement here came from a measurement, and every one of them contradicted an
assumption I was comfortable with — that service was the busiest department (it's 31%, sales is
25%, and "other" is 44%); that a bigger teacher would be better; that a confused class meant a hard
class; that a silent gate meant a broken one.

The model was rarely the problem. The framing was.
