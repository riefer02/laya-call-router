# What we learned building this

Notes from taking a small local decision model and trying to make it as good as a frontier LLM at
routing car-dealership phone calls — at a fraction of the cost.

Written for a person, not a changelog. The numbers are all in `results/`.

---

## The short version

We set out to match `deepseek-flash` on call routing, using Laya (a small non-autoregressive model
that returns typed decisions instead of generating text) running locally on a laptop.

We got there, with one qualification that matters. On 81 hand-labelled cases the fine-tuned cascade
scores **0.914 joint**. `deepseek-flash` scored 0.914, 0.889, 0.901 and 0.926 across four runs on
*identical inputs* — it leads us in some runs and trails in others. Ours has not moved once.

So: **parity on quality, at 21 ms instead of 1.5 seconds, for $0 a call, with a number that stays
put.** It also books real appointments now, not just routes.

That distinction — a stable number versus a noisy one — turned out to be the more defensible thing
to say, and I only found it by running the same evaluation four times.

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
labels in 1,395 rows.

But it means we're distilling `deepseek-flash`, so its agreement with our hand labels is the best
we can hope for. My plan to break through that ceiling was to use a *bigger* teacher.

I measured it first. `deepseek-v4-pro` — 3× the price, far slower — agrees with our labels **less**
than flash does: 0.951 vs 0.975 on destination. It isn't confused, it's just *different*, and it
loses on the same fuzzy cases flash handles correctly.

That was a $0.03 experiment that killed a plan I was confident about. Worth every cent.

**The lesson:** "use a stronger model" is a hypothesis, not a strategy. On a narrow task, a bigger
general model is often just a different model.

---

## The same bug, four times, in four disguises

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

Every one of these was silent. None threw an error you'd notice. Three of them cost real accuracy.

**The lesson:** one fact living in two places will eventually disagree, and it will do it quietly.
The fix isn't vigilance, it's structure — the taxonomy, the question text and the thresholds all
live in one config file now, and there are tests either side of them.

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

The same measurement paid for itself immediately. Because the errors aren't symmetric — a missed
stranded caller is someone at the side of a road, a false alarm is a wasted journey — we swept the
threshold instead of picking 0.5 because it's round. Precision stayed at 1.000 from 0.3 all the way
to 0.8, while recall fell. **Lowering dispatch to 0.3 catches one more stranded caller and sends no
extra trucks.**

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

## Things we know but haven't fixed

- **The acceptance classifier is untrained.** It's the base model answering a question it's never
  seen. The confidence floor makes that safe, not correct.
- **`needs_human` isn't trustworthy** — 43% of complaints and escalations are missed.
- **The store facts are loaded but unused.** The switchboard could answer "what time do you open?"
  from them. Right now it still transfers.
- **The `other` fallback is dead.** The fine-tuned model never uses it (0.0%, down from the base
  model's 23.5%). It always commits. For routing that's arguably right, but there's no "I'm not
  sure" left in the system.

---

## The thing I'd tell someone starting this

**Measure the thing you're most sure about, first.**

Every real improvement here came from a measurement, and every one of them contradicted an
assumption I was comfortable with — that service was the busiest department (it's 31%, sales is
25%, and "other" is 44%); that a bigger teacher would be better; that a confused class meant a hard
class; that a silent gate meant a broken one.

The model was rarely the problem. The framing was.
