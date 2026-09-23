# jev-classifier — elevator pitch

## The pitch

**What it is.** A call-routing switchboard for a car dealership that runs entirely on a laptop. A
caller speaks; the system decides which department owns the call, whether anyone is stranded, whether
a person should take it, and books a real appointment — in about 20 milliseconds, for nothing per
call.

**The new technology.** It is built on **Laya**, a *non-autoregressive* decision model. That is the
interesting part: it does not generate text. You hand it a question and a set of options **defined at
request time**, and it returns a typed, calibrated decision — `destination = service`,
`is_safe_to_drive = false`, each with a probability attached. `tokens generated` stays at zero on
every call, and that one property produces everything else:

- no sampling → **deterministic**. Same input, same answer, every time.
- no generation → **~20 ms instead of ~1.4 seconds**.
- no API → **free and private**, on the machine it already runs on.
- options supplied per request → a store can add a Fleet queue by editing a config file,
  **with no retraining**.

**What we built.** A working switchboard with a visual debugger: rows are conversation turns, columns
are decisions, and clicking any box shows the question, every option with its probability, which
checkpoint answered, and why it was routed there. The taxonomy, the question wording and the policy
thresholds all live in one config file as *data*, because training and inference must ask the
identical question — a copy drifted once and the model was asked something it had never seen.

**What we did.** Fine-tuned Laya on ~6,500 question-and-answer items built from a frontier teacher,
behind a high-precision filter: a candidate became training data only when two independently-worded
labelling passes agreed with each other *and* with the intended target. Then measured it against
`gpt-5.4-nano` and `deepseek-flash` on 81 hand-labelled routing cases, 27 scripted calls and 45
safety cases.

**The result.** Parity on routing — joint **0.876–0.926** against `deepseek-flash`'s 0.889–0.926 — at
**23 ms instead of 1,430 ms, for $0 a call**. Our number has never moved; theirs swings ±3.8 points
between *identical* runs. The honest claim is **parity with a stable number, not victory**.

**And what we actually learned was about measurement, not models.** Three of our evaluations were
reading their own training data. Every fine-tune had been scored partly on cases it trained on. Our
safety precision was flattered by an eval set that is 40% positive when reality is nearer 2%. Seven
silent failures now have tests — and that is the part I would trust most.

---

## Cost breakdown

| | |
| --- | --- |
| Teacher calls to build the routing data | ~$0.91 |
| Teacher calls to build the safety data | ~$2.57 |
| The register-experiment top-up | ~$0.79 |
| Teacher validation + LLM eval arms (all runs) | ~$0.10 |
| **Fine-tuning** | **$0** — Kaggle's free 2×T4 tier, ~40 min per run |
| **Inference** | **$0 per call**, forever, on local hardware |
| **Total for the whole project** | **~$4.40** |

**The comparison that matters.** `deepseek-flash` costs **$0.0113 per call**. The entire project —
data, training and every evaluation — cost about as much as **400 calls** to the frontier model. The
next 400,000 cost nothing.

Two honest footnotes: **$1.28 of that total is re-work** after a destructive bug of mine deleted a
generated dataset, and re-generating it was cheaper than the lesson. And the teacher cost is
one-time: it buys a checkpoint that then runs for free, so the cost per call *falls* with volume
rather than rising.

---

## If someone asks about the architecture

**Is it one model call per decision box?** No. The backend computes the whole call eagerly —
`events = list(session.advance())` — and returns an event list; play, pause, step and scrub are pure
client-side animation. But the *compute* is far smaller than the node count: there is one
`router.predict(...)` per turn, taking a **dict** of questions, so they are batched into a single
forward pass.

A real six-turn booking call:

```
51 decision boxes drawn in the UI
17 classifier questions asked
 6 forward passes (one per turn)
191 ms of compute, 0 tokens generated, $0.00
```

Most boxes in the graph are **policy** and **regex** nodes, which cost nothing — the UI colours them
differently and badges them `policy` / `regex` so it never pretends a model decided something a rule
decided. The question count also falls per turn, because settled facts are skipped: turn 1 asks all
seven, and from turn 2 on only what is still open plus one change-detector.

**Why batch rather than call per decision?** Because the transcript is the expensive part, not the
questions. Seven questions against one transcript costs one pass; seven passes would cost seven times
the tokens for the same answer.

---

## What not to claim

- **Not "we beat deepseek."** Joint is a tie inside the noise, and the point estimate swings ±3.8
  points across runs of *their* model. Say: parity, 60× faster, for nothing, with a stable number.
- **Not "the safety classifier is fixed."** It catches every stranded caller and its false alarms fell
  from 8 to 2, but at an assumed 2% hazard rate most dispatch flags would still be wrong.
- **Not "1.000 booking accuracy"** without saying it is measured on held-out *reply phrasings*, not
  held-out conversations.
- **Not "the threshold is tuned."** The sweep is flat from 0.3 to 0.8 — the probabilities are
  saturated, so the threshold is not a control we actually have.
