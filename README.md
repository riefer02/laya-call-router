# jev-classifier

A local **call-routing debugger** for a car-dealership switchboard, built on
[Laya](https://github.com/NandhaKishorM/laya) — a non-autoregressive _System 1_ decision model
that returns **typed, calibrated decisions** instead of generated text — running on Apple Silicon
via [`laya-mlx`](https://pypi.org/project/laya-mlx/).

A scripted caller talks to a classifier-driven switchboard. Every decision the switchboard makes
is a Laya question, the graph shows the call flowing left to right, and clicking any box reveals
exactly how that decision was made.

**The model never generates text.** `tokens generated` stays at `0` in the HUD on every call —
that is the point of the technology, and the reason the economics work.

![the debugger](docs/overview.png)

## Run it

Backend (Apple Silicon, Python 3.12 — `uv` fetches it):

```bash
uv sync
uv run uvicorn jev_classifier.api:app --port 8765
```

Frontend, one of:

```bash
cd web && npm install && npm run dev      # dev server on :5173, proxies /api to :8765
cd web && npm install && npm run build    # or build; FastAPI then serves it at :8765
```

Open <http://127.0.0.1:8765> (build) or <http://localhost:5173> (dev).

## The visual debugger

```
┌ controls ────────────────────────────────────────────────────────────────────────────┐
│ scenario ▾   ▶ Run call   ⏸ Play  ⏭ Step  ⏩ Skip  ↺ Reset   speed ─●──  ⤢ Fit  ☑ follow │
├──────────────┬──────────────────────────────────────────────────────┬───────────────┤
│ CONVERSATION │              D E C I S I O N   G R A P H             │   INSPECTOR   │
│  (collapsible)                                                       │  (drill-down) │
│  turn 1      │  turn 1  ☎ ─▶ [Department] ─▶ [Vehicle] ─▶ … ─▶ 🎧    │  question     │
│  ☎ caller    │  turn 2  ☎ ─▶ [Department] ─▶ [Vehicle] ─▶ … ─▶ 🎧    │  every option │
│  🎧 agent    │  turn 3  ☎ ─▶ …                                       │  + probability│
│              │  turn 4  ☎ ─▶ … ─▶ [Next step] ─▶ 🎧 ─▶ ◆ ROUTE       │  confidence   │
└──────────────┴──────────────────────────────────────────────────────┴───────────────┘
```

- **Rows are turns, columns are decisions.** Positions are computed from `(col, turn)` and never
  recomputed, so nothing jumps around while the call streams in — which matters when you are
  screen-recording.
- **Colour says who or what**: caller (blue) · model decision (violet) · rule / regex (slate) ·
  switchboard (green) · route out (amber). A node's badge is its primitive (`choice` / `noul`)
  or, for the deterministic nodes, `policy` / `regex` — the UI never pretends a model decided
  something a rule decided.
- **Click any box** for the question, every option with its probability, the entropy confidence
  _and_ the top probability, which checkpoint answered and why it was routed there, latency,
  how many questions shared the forward pass, and raw JSON.
- **`follow` keeps the camera on the active decision** so you can watch the call move through the
  pipeline; turn it off (or press `F`) to frame the whole call.
- **Playback is client-side** — play / pause / step / speed / scrub all work without touching the
  backend, because a call is executed eagerly, recorded, and returned as an event list. That also
  means **every run is replayable**: `replay recording…` re-plays a past call deterministically
  for a clean recording. Space toggles play, `→` steps.

## The call

Seven destinations, following how dealerships actually organise — **Fixed Operations**
(`service · parts · body_shop`), **Variable Operations** (`sales · finance`), plus `front_desk` and
`non_customer`.

**Tires and detailing are Service sub-queues, not departments.** Roadside assistance is a *policy
flag* on an unsafe-to-drive call, not a place the call goes. And `general` never was a department:
it was absorbing two unrelated things, so it is split honestly between `front_desk` (a real
question no department owns) and `non_customer` (a supplier, a job applicant, a wrong number).

The taxonomy is **data, not code** (`config/store_profile.json`). A store with no body shop deletes
a line; a store with its own tyre centre promotes `tires` to a destination. Both are covered by
tests, and neither needs a code change. The question text lives there too, because the *same*
string must be used when training and at inference — a copy drifted once and the model was asked a
question it had never been trained on.

Each turn runs two batched forward passes:

1. **destination + slots** — vehicle, location, when, unsafe-to-drive, needs-a-human
2. **sub-queue** (branched on the destination) — then policy picks the next step

Then, once the slots are known, the switchboard offers **real appointment times** drawn from the
store's own opening hours and service durations, classifies which one the caller accepted, and
files an appointment with their name on it. "I'm booking you into service for next week" was never
an appointment.

Slots are `choice` questions over fixed enums, not free-text extraction, because Laya classifies
rather than parses. Three things are deliberately **not** classifier questions, and are rendered as
distinct node kinds: the exact appointment time, and the caller's name and number. A callback
number is the one field where a plausible-looking invention does real damage.

**Where the classifier decides vs. where policy decides.** The classifier does _understanding_:
department, intent, slot values, urgency, escalation. A deterministic policy does _control flow_:
which slot to ask for next, when the booking is complete, which queue it lands in. That split is
not an accident — see the measurements below.

## What measuring the model changed

Laya's base checkpoints are weak zero-shot and very sensitive to wording, so every question here
was chosen from a measurement, not intuition. `scripts/probe_slots.py` and `scripts/probe_questions.py`
are the evidence. Three findings shaped the design:

1. **"Explicitly mention… otherwise `not_stated`"** — the first slot questions _invented_ facts:
   asked "What kind of vehicle?" about a sentence that never mentioned a vehicle, the model
   answered `sedan`. Rewording to "What kind of vehicle did the caller explicitly mention? If no
   vehicle type was mentioned, choose `not_stated`" took slot accuracy from **19/24 to 23/24**.
2. **Do not let the agent read out the option list.** When the switchboard asked "today, tomorrow,
   later this week, or next week?", the slot classifier started answering `today` even when the
   caller had said nothing about timing — the agent's own question was leaking into the transcript
   the model classifies. The spoken prompts no longer enumerate; the options are still shown as
   chips in the UI.
3. **Control flow is not a good classifier question.** A `next_action` question ("what should the
   switchboard do next?") answered at **confidence 0.03** and picked `ask_detail` almost
   regardless of input, while an earlier unconstrained version happily chose `confirm_booking`
   with the location still unknown. It was removed and replaced with policy. The classifier stayed
   where it is strong.

A fourth finding, from the earlier support-triage build, still applies: a **hard stop must not
fire on an argmax of a near-uniform distribution** — reject only on a confident signal.

## Why it doesn't re-decide what it already knows

The first implementation ran all seven questions on **every** turn, re-reading the whole growing
transcript each time. Turn 4 of a four-turn call re-asked what turn 1 had already established:

| collision scenario    | turn 1 | turn 2 | turn 3 | turn 4 | total             |
| --------------------- | ------ | ------ | ------ | ------ | ----------------- |
| questions — before    | 7      | 7      | 7      | 7      | **28**            |
| questions — now       | 7      | 4      | 3      | 2      | **16** (−43%)     |
| input tokens — before | 720    | 895    | 1070   | 1259   | 3944              |
| input tokens — now    | 720    | 479    | 406    | 318    | **1923** (−51%)   |
| compute — before      | 60 ms  | 64 ms  | 71 ms  | 78 ms  | **272 ms**        |
| compute — now         | 59 ms  | 30 ms  | 22 ms  | 23 ms  | **134 ms (−51%)** |

Three mechanisms:

- **Settled facts are skipped.** Each turn keeps a session of what is already known. A fact is
  _pinned_ when it is a concrete value answered decisively enough to rely on; pinned facts are not
  re-evaluated, and appear in the graph as dimmed _"already known — settled turn N"_ nodes.
- **`not_stated` is never pinned.** Resolving "the caller hasn't said" is the whole point of a
  later turn, so those stay open. This is what keeps self-correction working: in the collision
  call the `when` answer is wrong for two turns (`today`, inferred from "yesterday") and then
  settles correctly to `next_week` on turn 4 — because it was never pinned.
- **One change-detector buys the right to skip several.** From turn 2 on, a single `noul`
  question — _"does the caller's latest message change or add to anything said earlier?"_ — runs
  alongside the unresolved questions. If it fires, the pinned facts are re-evaluated in a second
  pass; if not, they are skipped.

**Pinning uses top probability, not entropy confidence.** They are different questions and want
different numbers: `confidence` (entropy, 0.75) asks _"should a human look at this?"_, while
pinning (top probability, 0.6) asks _"can we stop re-deciding this?"_. Judging pinning by entropy
confidence never settled `intent`, because a 7-option question with a clear winner (p = 0.72)
scores only 0.42.

`scripts/bench_call.py` reproduces the table; `--out results/*.json` keeps a baseline to diff against.

## Second opinions, not second guesses

When a classification question is not decisive (top probability below 0.75 for `department` or
`intent`), the switchboard asks it **again in different words** — the paraphrase lives in
`dealership.department_question_paraphrase` — and compares the two answers.

- **Agreement** settles the question: two independently-worded phrasings landing on the same
  answer is evidence, and the fact is pinned (so later turns skip it).
- **Disagreement** does _not_ overturn the first answer. It keeps it and flags the call for an
  LLM or a human.

In the vague-complaint scenario: **2 second opinions, 0 LLM calls**, and the call routes cleanly
to the Service Department.

Verification costs one extra question and largely pays for itself, because agreeing phrasings let
`department` and `intent` settle a turn earlier:

| collision        | questions | tokens | compute    |
| ---------------- | --------- | ------ | ---------- |
| before           | 28        | 3944   | 272 ms     |
| incremental only | 16        | 1923   | 134 ms     |
| + verification   | 17        | 2020   | **130 ms** |

**The rejected design, and why.** The obvious tier-2 is to _narrow_: take the top three options
from a low-confidence pass and re-ask with only those. Measured, it does not improve the decision —
it re-rolls it and inflates confidence:

| utterance                                              | tier 1       | narrowed to               | tier 2               |
| ------------------------------------------------------ | ------------ | ------------------------- | -------------------- |
| "I have a problem with my car and need to bring it in" | service 0.37 | service/body_shop/general | **body_shop 0.73** ✗ |

A wrong answer made to look decisive is worse than no escalation at all, because everything
downstream now trusts it. `scripts/probe_slots.py` keeps the evidence. Confidence is only useful
if it tracks correctness.

## Speaking before thinking

Every turn emits a fixed acknowledgement — _"Let me take a look at that for you."_ — **before any
forward pass runs**. It appears in the graph as its own node and in the conversation rail as an
extra switchboard bubble, so you can see the agent speak immediately rather than after the
cascade. It is a template, not generation; the point is that a voice channel needs _something_
within a few hundred milliseconds, and 30–60 ms of classification is not the only latency that
matters.

## What the four-arm evaluation found

`scripts/eval.py` runs 81 hand-labelled routing cases and 27 scripted calls through four arms: our
cascade (base and fine-tuned), a cheap structured-output model (`gpt-5.4-nano`), and
`deepseek-flash`.

### Decision level — 81 cases

| metric | base | **fine-tuned** | gpt-5.4-nano | deepseek-flash |
| --- | --- | --- | --- | --- |
| destination | 0.654 | 0.951 | 0.951 | **0.975** |
| sub-queue | 0.518 | **0.914** | 0.852 | 0.901 |
| joint | 0.518 | **0.914** | 0.852 | 0.901 |
| **queue (the outcome)** | 0.667 | **0.963** | 0.926 | 0.951 |
| ±95% (queue) | ±0.101 | ±0.045 | ±0.059 | ±0.050 |
| p50 latency | 21.4 ms | **21.8 ms** | 631 ms | 1455 ms |
| cost per case | **$0** | **$0** | $0.0025 | $0.0114 |
| determinism (3 repeats) | **1.00** | **1.00** | 0.99 | 1.00 |

**The fine-tuned cascade is above both LLM arms on joint and on queue accuracy, at ~65× the speed
and for nothing per call.**

The caveat that belongs next to that sentence: on 81 cases one case is 1.23 points and the Wilson
intervals overlap (ours ±0.063, theirs ±0.066 on joint). So this is **parity with our point
estimates ahead**, not proven superiority — and the LLM arms themselves move ±2.5 points between
identical runs, which is why every table here carries intervals.

`destination` is the one metric still behind. Two things are worth knowing about it. The errors are
concentrated in `front_desk`, and **queue accuracy is higher than destination accuracy because two
labels can route to the same place** — `front_desk` and `non_customer` both go to Front Desk, so a
label miss there is not a routing miss. Both numbers are reported; neither replaces the other.

### Confidence calibration — why the gate behaves as it does

| confidence band | base: n / accuracy | fine-tuned: n / accuracy |
| --- | --- | --- |
| 0.0-0.2 | 8 / 0.500 | — |
| 0.2-0.4 | 26 / 0.385 | — |
| 0.4-0.6 | 24 / 0.750 | — |
| 0.6-0.8 | 13 / 0.846 | 2 / 0.500 |
| 0.8-1.0 | 10 / 1.000 | **79 / 0.962** |

The fine-tune puts **79 of 81 cases in the top band at 0.962 accuracy**. Its confidence is high
*because it is accurate*, not because it is overconfident — which is why no threshold "rescues" it:
the errors are not hiding in a low-confidence tail. The base model's low confidence was a symptom
of weakness, not of miscalibration.

That is why the escalation gate changed *character* rather than simply improving. The base model
flags 81.5% of calls — uselessly expensive. The fine-tune flags 2.5% and catches 1 of its 4 errors.
On the frontier that is worth **0.951 → 0.963 for ~$0.000003/call**: a small, cheap safety net
rather than the load-bearing tier it is for the base model.

### Call level — 27 calls, final queue

| metric | cascade-full | base | **fine-tuned** | hybrid | nano | deepseek |
| --- | --- | --- | --- | --- | --- | --- |
| queue accuracy | 0.704 | 0.852 | **0.963** | 0.926 | 0.963 | 0.963 |
| questions asked | 645 | 439 | **397** | 380 | 0 | 0 |
| p50 latency | 233 ms | 237 ms | **203 ms** | 143 ms | 634 ms | 1574 ms |
| cost total | $0 | $0 | **$0** | $0 | $0.00087 | $0.00456 |

A three-way tie on quality. Note the base model scored **1.000 on the original 10-call set and
0.852 on 27** — the small set was too easy once a conversation ran a few turns, which is exactly
why it was expanded to cover all 26 sub-queues.

`results/eval_v4.json` holds the raw per-case predictions, confusions and calibration.

## Training data

The fine-tune is capped by its labels, so the dataset has explicit acceptance criteria and does
not ship unless it passes (`scripts/generate_training.py` exits non-zero otherwise):

| criterion | bar | result |
| --- | --- | --- |
| teacher agreement on the held-out 81 | ≥ 0.92 destination, ≥ 0.85 sub-queue | **0.975 / 0.901** |
| every *specific* sub-queue | ≥ 25 | **exactly 50** |
| every destination | ≥ max(25 × its sub-queues, 150) | **min 150** |
| labels in vocabulary | 100% | **100%** (0 invalid) |
| near-duplicates of the held-out 81 | 0 | **0** |
| intra-corpus duplicates | 0 | **0** |
| two-phrasing agreement | 100% of kept rows | **100%** |

**1,395 examples** (1,256 train / 139 dev) for **$0.91** in teacher calls. Teacher is
`deepseek-flash`; a candidate only becomes a training example when **two independently-worded
labelling passes agree with each other and with the intended target**.

### Balancing on the wrong axis starved a class

The dataset is balanced per sub-queue, so every one of the 26 specific sub-queues has exactly 50
examples. That turned out to be the wrong axis for the *destination* question: a destination's
volume scaled with how many sub-queues it happened to have, so `front_desk` — two sub-queues
against service's six — got a third of the volume and was the smallest class at 90 rows.

It was also the worst class on the test set: **four of the five destination errors were
`front_desk`**. That is not a guess about the errors — both teachers get every `front_desk` case
right while the fine-tune got 4 of 8 wrong, so the boundary is learnable and we simply had too
little of it. `--min-per-destination` (default 150) now raises the per-sub-queue target for any
destination that would otherwise fall short. It moved destination accuracy 0.938 → 0.951.

### A residual class cannot be generated into existence

`other` is what a branch falls back to when nothing specific fits. Asking the teacher for "an
example of other" produces utterances that clearly belong to a *specific* sub-queue, so the
independent labelling pass relabels them and the mismatch filter drops them — **34 of 36 in one
measured run**. That is the filter working: you cannot manufacture positives for the class defined
by *not* matching the others.

So the criteria never gate on residual counts, and `other` is learned as the branch's softmax
fallback rather than as a category. The fine-tuned model now answers `other` **0.0%** of the time,
down from the base model's 23.5% — it commits rather than abstains, which for routing is the right
behaviour but does mean there is no "I'm not sure" left.

### What the labels cost, and what they cannot reach

A bigger teacher was measured and rejected: **`deepseek-v4-pro` agrees with the hand labels *less*
than flash does** (0.951 / 0.864 against 0.975 / 0.901) at 3× the price. A larger model is not a
better teacher for this task, and it breaks the same fuzzy cases flash gets right
(`detailing` ↔ `paint`, `inventory` ↔ `used_vehicle`). That is worth knowing before anyone
proposes "just use a better model".

One case — `det-04` — is missed by the fine-tune, nano *and* deepseek. When every model disagrees
with the key, the key is the likeliest thing to be wrong. **The single-labeller ceiling is the real
remaining constraint on this number, not model capacity.**

## Fine-tuning: the step that actually closed the gap

The evaluation above is the *before*. Laya's base checkpoints are weak zero-shot — their own
documentation says so, and 0.654 destination accuracy is what that looks like. Fine-tuning on the
synthetic set (RLCD, official trainer, 2×T4, ~15 min for 8 epochs) is the *after*:

| metric | base cascade | **fine-tuned** | gpt-5.4-nano | deepseek-flash |
| --- | --- | --- | --- | --- |
| destination accuracy | 0.654 | **0.951** | 0.951 | 0.975 |
| sub-queue accuracy | 0.518 | **0.914** | 0.852 | 0.901 |
| joint accuracy | 0.518 | **0.914** | 0.852 | 0.901 |
| call-level queue accuracy | 0.852 | **0.963** | 0.963 | 0.963 |
| p50 latency | 21.4 ms | **21.8 ms** | 631 ms | 1455 ms |
| cost per case | **$0** | **$0** | $0.0025 | $0.0114 |
| determinism (3 repeats) | **1.00** | **1.00** | 0.99 | 1.00 |

**+39.6 points of joint accuracy, at the same ~21 ms, for $0, deterministically** — and now above
both LLM arms rather than below them.

Four changes produced that, and the order matters:

1. **Correcting the taxonomy** — tires and detailing as service sub-queues, `general` split
   honestly. The teacher then agreed with the hand labels at 0.975 / 0.901.
2. **Training for 8 epochs instead of 4.** The loss was still halving every epoch
   (0.686 → 0.515 → 0.174 → 0.081); the model had simply stopped early. Worth +2.5 points of joint,
   and it converges at 0.042 by epoch 6-7, so more would not help.
3. **The per-destination floor** — `front_desk` had been starved by sub-queue balancing. Worth
   +1.3 points of destination.
4. **Fixing a question-text drift.** Training built the sub-queue question as *"This is a service
   call"* while inference sent the label verbatim, *"This is a Service call"*. The fine-tune learns
   to answer one exact instruction, so the model would have been asked a question it had never been
   trained on — and it would have looked like a mediocre fine-tune rather than a string mismatch.
   Caught before the GPU run; the templates now live in the store profile and both sides read them.

### What the earlier fine-tunes got wrong, and why

The first attempt regressed at call level (1.000 → 0.900): the "my neighbour's dog" call started
routing to Service Department. The cause was precise — the synthetic set was *all* plausible
car/dealership calls, so it contained no negatives, and the fine-tune lost the base model's habit
of answering `general` for things that are not the dealership's business. An off-topic generation
pass fixed it, which is why `non_customer` exists as a destination rather than being folded into
`front_desk`.

Those models are kept in `results/`, **labelled as superseded**. They were trained on the old
nine-department taxonomy, so their numbers measure a partly-wrong task and are not comparable to
the current ones. The checkpoint at `models/kaggle-out-v2/` is invalid for the current taxonomy and
is named as such rather than deleted.

## Does the fine-tune still work on questions it was never trained on?

This matters more than the domain numbers. Laya's defining property is that **the option space is
defined at request time**, so a new schema needs no retraining — which is what makes a per-store
configurable taxonomy viable. We then fine-tuned the encoder hard on 32 fixed sub-queues for eight
epochs, and had never checked what that cost. `scripts/generality_test.py` measures it on real
human text from outside our domain:

| suite | base | fine-tuned |
| --- | --- | --- |
| **Banking77** — 77 real customer-service intents | 0.415 | **0.390** |
| **CLINC150** — 150 intents incl. out-of-domain (151 options) | 0.095 | **0.210** |
| CLINC150 out-of-domain recall | 0.025 | 0.037 |

**Verdict: generality held.** A 2.5-point drop on a completely different domain with 77
runtime-defined options — inside the noise, and the intervals overlap. Per-store configuration is
viable; a store adding a Fleet queue does not require retraining. The base model scoring **0.415
against Laya's published 0.425** also validates the harness — we are measuring the same thing they
measured.

Two further findings worth keeping:

- **The option token budget is a real, large lever.** Banking77 scores **0.415 at
  `head_max_len=256` and 0.470 at 512** — +5.5 points just from giving 77 options room to stay
  distinct. The top confusions are semantically adjacent pairs (`top_up_limits` ↔
  `top_up_reverted`, `unable_to_verify_identity` ↔ `verify_my_identity`), i.e. sensible confusions
  caused by labels running out of tokens, not random noise. Above ~20 options, raise the budget
  before blaming the model.
- **The model cannot detect out-of-domain.** Asked as a `noul` question — *"is this outside the
  domain?"* — the mean P(ood) was **0.166 on genuinely out-of-domain text and 0.160 on in-domain
  text**. It is a constant, not a signal. This is a **design constraint, not a tuning problem**:
  "is this even a dealership call?" must be a *`choice` option the model can select*, never a
  confidence threshold we read off a boolean. The fine-tuned model does the former.
- Counter-intuitively, fine-tuning **improved** 151-option performance (0.095 → 0.260). Practising
  discriminating 43 classes appears to have helped general many-option behaviour rather than
  hurting it.

## Is it safe? The severity questions

`is_safe_to_drive` dispatches roadside assistance and sets priority to HIGH. `needs_human` decides
whether the automated flow runs at all. Neither had ever been measured — the entire safety surface
of this system was assumed. `scripts/eval_severity.py` runs 45 labelled cases (18 unsafe, 7
needs-human).

**The errors are not symmetric, so accuracy would be the wrong headline.** Missing a stranded
caller leaves someone at the side of a road; a false alarm sends a truck to someone who was fine.
The report leads with recall and names the missed cases:

| question | recall | precision | missed |
| --- | --- | --- | --- |
| `is_safe_to_drive` (fine-tuned) | **0.778** | 1.000 | **4 of 18** |
| `needs_human` (fine-tuned) | 0.571 | 0.250 | 3 of 7 |
| `is_safe_to_drive` (base) | 0.722 | 0.929 | 5 of 18 |
| `needs_human` (base) | **0.143** | 0.071 | 6 of 7 |

**Even the fine-tuned model fails to dispatch 4 of 18 stranded or unsafe callers.** The misses
cluster on *implied* hazards — "smoke coming from under the hood", "the accelerator stuck open" —
where the caller never says they are stopped. The question asks what the caller *indicates*, so the
model answers literally while a person would hear a fire risk. That is a question-wording problem
as much as a model one, and it is written down rather than smoothed over.

`needs_human` is much weaker: complaints and escalations are missed 43% of the time with heavy
false positives. It is not yet trustworthy.

**One thing the threshold sweep made obvious and free.** On the fine-tuned model precision is
1.000 at every threshold from 0.3 to 0.8, while recall falls as the bar rises:

```
thr 0.3   recall 0.833   precision 1.0   missed 3   false alarms 0
thr 0.5   recall 0.778   precision 1.0   missed 4   false alarms 0
thr 0.8   recall 0.722   precision 1.0   missed 5   false alarms 0
```

Lowering dispatch from 0.5 to 0.3 catches one more stranded caller and sends no extra trucks. The
policy now uses **0.3**. Doing that exposed a real inconsistency — the `unsafe_to_drive` flag, the
priority, the handler and the transfer decision were reading three different thresholds, so a
caller could be dispatched as unsafe while the audit trail said they were not. One number now
drives all of them, with an invariant test either side of it.

## Scheduling: the call ends in an appointment

The switchboard used to *say* "I'm booking you into service for next week" and file nothing — and
"next week" is not an appointment. It now offers **real times** drawn from the store's own opening
hours and a per-service duration table, classifies which one the caller accepted, and files a
booking with their name on it:

> *"You're all set: new vehicle at westside — Tuesday 22 September at 8am for Dana."*

`minutes` in the profile is how long the **booked slot** occupies, not how long the repair takes —
a collision repair is days but its appointment is a drop-off. That distinction is load-bearing: a
1440-minute "slot" can never fit an 8am-6pm window and silently offered nothing at all until an
invariant test caught it.

Two deliberate choices:

- **Deciding whether a caller accepted a time is a `choice` question**, not generated prose. It is
  a classification, and it is exactly the kind of thing a generative model would answer with
  invented text.
- **An indecisive answer clarifies rather than commits.** On the first live run the caller said
  *"this is Dana, and my number is 555-0140"* — no time at all — and the untrained acceptance
  classifier answered `slot_1` at p=0.41, and the appointment was **filed**. It had invented an
  agreement. There is now a floor at the pin threshold: the same turn asks *"Sorry — which of those
  times did you want?"*, and a genuine acceptance (p=0.65) books correctly. An argmax of a
  near-uniform distribution is not a decision — the same rule that stops a hard stop firing on a
  weak signal elsewhere in this system.

The acceptance classifier is still **untrained** — it is the base checkpoint answering a question it
has never seen, which is why the floor matters. Training it is the next piece of work.

## Measurements (Apple M5 Pro, 64 GB)

Whole triage schema, batched in one forward pass:

| checkpoint                         | 1 question p50 | 6 questions p50 | throughput |
| ---------------------------------- | -------------- | --------------- | ---------- |
| `english` (ModernBERT-large, 421M) | 9.0 ms         | 27.7 ms         | 219 q/s    |
| `multilingual` (mmBERT-base, 322M) | 4.6 ms         | 11.1 ms         | 520 q/s    |
| `typed-decisions` (421M)           | 9.2 ms         | 27.8 ms         | 218 q/s    |

A call now ends in a filed appointment: the fine-tuned cascade runs **~200 ms of compute** and
**397 questions across 27 calls** (≈15 per call), generates **0 tokens**, makes **0 LLM calls** and
costs **$0.00**.

Quality, 18 hand-labelled tickets (`data/tickets/labelled.jsonl`, from the support-domain work):

| checkpoint        | department acc | churn acc | refund acc | noul ECE ↓ |
| ----------------- | -------------- | --------- | ---------- | ---------- |
| `english`         | 0.83           | 1.00      | 0.94       | 0.073      |
| `multilingual`    | 0.61           | 0.83      | 0.94       | 0.132      |
| `typed-decisions` | 0.83           | 0.94      | 0.89       | 0.170      |
| _chance_          | _0.28_         | _0.17_    | _0.28_     | —          |

`typed-decisions` does not win: it is fine-tuned on four synthetic workflows matched by _exact
question-id sets_, so it transfers nothing to a custom schema.

## Layout

```
config/
  store_profile.json  the taxonomy, question text, facts and schedule — as data, not code
src/jev_classifier/
  store_profile.py  loads/validates the profile; builds the typed questions
  dealership.py     slot enums, response templates, routing policy, contact extraction
  schedule.py       availability and bookings (mock DMS, DMS-shaped interface)
  call.py           the turn driver -> node/edge event stream
  scenarios.py      scripted caller calls
  labels.py         normalise/validate model output against the profile vocabulary
  teacher.py        the labelling prompt and schema
  synthgen.py       generation prompts
  runs.py           record / replay (results/runs/*.jsonl)
  api.py            FastAPI: /api/scenarios /api/call /api/runs
  pipeline.py       the original generic support cascade (still reachable via CLI)
web/                Vite + React + React Flow front end
  src/graph/        canvas, deterministic layout, node components
  src/inspector/    drill-down panel
  src/conversation/ collapsible rail
  src/controls/     run / step / speed / replay
scripts/
  try_call.py           run a call and print the trace headless
  eval.py               the four-arm evaluation (cascade / LLM arms / hybrid frontier)
  eval_severity.py      the safety questions: recall of stranded callers
  validate_teacher.py   the teacher gate — run before spending on generation
  generate_training.py  build the training set, with acceptance criteria
  relabel_groundtruth.py  migrate the ground truth onto a new taxonomy, with a drift report
  generality_test.py    does fine-tuning still handle unseen option spaces?
  probe_slots.py        slot-wording measurements
  bench_call.py         per-turn cost, with a saved baseline
training/
  build_items.py     labelled utterance -> (sequence, target) pairs
  train_ddp.py       the vendored RLCD trainer (parameterised, defaults unchanged)
  kaggle_run.py      submit / watch the free 2xT4 fine-tune
data/calls/         labelled ground truth: routing, calls, severity
results/            the raw reports, tracked, with a README saying which are superseded
tests/              policy, schedule, labelling and statistics tests (no model required)
```

CLI, no browser:

```bash
uv run python scripts/try_call.py --scenario buy_car     # a full call trace, ending in a booking
uv run python scripts/eval.py --skip-llm                 # the cascade arms only
uv run pytest -q
```

## Not built yet

**The acceptance classifier is untrained.** It is the base checkpoint answering a question it has
never seen, which is why it once filed an appointment for a caller who agreed to nothing. The
confidence floor makes that safe rather than correct; training it needs multi-turn examples
(offered times → which one was taken), which is a generation mode we do not have yet.

**The other open items, in the order I would do them:**

- **`needs_human` is not trustworthy** (recall 0.571, precision 0.250). Complaints and escalations
  are missed 43% of the time.
- **`is_safe_to_drive` misses 4 of 18.** The misses are *implied* hazards where the caller never
  says they are stopped — the question asks what the caller indicates, so the model answers
  literally. Rewording it to ask about hazard rather than statement is the fix to try first.
- **A second human labeller.** One case (`det-04`) is missed by every model; where all three
  disagree with the key, the key is the likeliest thing to be wrong. This is the ceiling on the
  destination number and no amount of model work moves it.
- **The store facts are unused.** `facts` (hours, address, directions, loaner policy) is loaded and
  rendered, but the switchboard does not yet *answer* from it — it still transfers a factual
  question. Measured motivation: hours and directions is one of the top repeatable Fixed Ops call
  types, so this is real call volume.
- **Real speech-to-text and voice, and dropped-call recovery.** Both deliberately left out. The
  record/replay event log is already a serialisable run, so a real caller drops in without touching
  the cascade or the UI.

The training and evaluation loop is complete and reproducible: `validate_teacher.py` →
`generate_training.py` → `kaggle_run.py submit --watch` → `eval.py` → `generality_test.py`. Every
gate is a command that exits non-zero when it fails, rather than a judgement call.

## Security

This is a **local development tool** with no authentication. Run it on loopback only
(`--host 127.0.0.1`, the default in the docs); binding it to `0.0.0.0` exposes the model and,
once the LLM/STT arms exist, a billable API key. The app prints a warning if it detects a
non-loopback bind.

**Secrets**

- Keys live in `.env` at the repo root, are git-ignored, and should be `chmod 600`.
- `.githooks/pre-commit` refuses to commit an environment file or anything matching a credential
  pattern. Enable it once per clone:

  ```bash
  git config core.hooksPath .githooks
  ```

- The browser never sees a key. When the LLM and speech-to-text arms land, the frontend calls our
  API and only the server talks to the provider.
- `results/runs/*.jsonl` contain support conversations. They are git-ignored; treat them as
  personal data in any real deployment.

**Input handling**

- `GET /api/runs/{run_id}` validates the id against a strict charset _and_ checks the resolved
  path stays inside `results/runs/` — a URL segment is never interpolated straight into a
  filesystem path.

## Licence

Project code is yours. Laya and its weights are Apache-2.0 by Convai Innovations; `laya-mlx` is an
independent Apache-2.0 port.
