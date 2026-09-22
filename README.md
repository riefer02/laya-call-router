# jev-classifier

A local **call-routing debugger** for a car-dealership switchboard, built on
[Laya](https://github.com/NandhaKishorM/laya) — a non-autoregressive *System 1* decision model
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
  *and* the top probability, which checkpoint answered and why it was routed there, latency,
  how many questions shared the forward pass, and raw JSON.
- **`follow` keeps the camera on the active decision** so you can watch the call move through the
  pipeline; turn it off (or press `F`) to frame the whole call.
- **Playback is client-side** — play / pause / step / speed / scrub all work without touching the
  backend, because a call is executed eagerly, recorded, and returned as an event list. That also
  means **every run is replayable**: `replay recording…` re-plays a past call deterministically
  for a clean recording. Space toggles play, `→` steps.

## The call

Eight departments: `service · body_shop · parts · tires · detailing · sales · finance · towing`
(plus `general`). Each turn the switchboard re-reads the whole conversation and runs two batched
forward passes:

1. **department + slots** — vehicle, location, when, unsafe-to-drive, needs-a-human
2. **intent** (branched on the department) — then policy picks the next step

Slots are `choice` questions over fixed enums, not free-text extraction, because Laya classifies
rather than parses. The one exception is the exact appointment time, which is a deterministic
regex rendered as a different node kind.

**Where the classifier decides vs. where policy decides.** The classifier does *understanding*:
department, intent, slot values, urgency, escalation. A deterministic policy does *control flow*:
which slot to ask for next, when the booking is complete, which queue it lands in. That split is
not an accident — see the measurements below.

## What measuring the model changed

Laya's base checkpoints are weak zero-shot and very sensitive to wording, so every question here
was chosen from a measurement, not intuition. `scripts/probe_slots.py` and `scripts/probe_questions.py`
are the evidence. Three findings shaped the design:

1. **"Explicitly mention… otherwise `not_stated`"** — the first slot questions *invented* facts:
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

| collision scenario | turn 1 | turn 2 | turn 3 | turn 4 | total |
|---|---|---|---|---|---|
| questions — before | 7 | 7 | 7 | 7 | **28** |
| questions — now | 7 | 4 | 3 | 2 | **16** (−43%) |
| input tokens — before | 720 | 895 | 1070 | 1259 | 3944 |
| input tokens — now | 720 | 479 | 406 | 318 | **1923** (−51%) |
| compute — before | 60 ms | 64 ms | 71 ms | 78 ms | **272 ms** |
| compute — now | 59 ms | 30 ms | 22 ms | 23 ms | **134 ms (−51%)** |

Three mechanisms:

- **Settled facts are skipped.** Each turn keeps a session of what is already known. A fact is
  *pinned* when it is a concrete value answered decisively enough to rely on; pinned facts are not
  re-evaluated, and appear in the graph as dimmed *"already known — settled turn N"* nodes.
- **`not_stated` is never pinned.** Resolving "the caller hasn't said" is the whole point of a
  later turn, so those stay open. This is what keeps self-correction working: in the collision
  call the `when` answer is wrong for two turns (`today`, inferred from "yesterday") and then
  settles correctly to `next_week` on turn 4 — because it was never pinned.
- **One change-detector buys the right to skip several.** From turn 2 on, a single `noul`
  question — *"does the caller's latest message change or add to anything said earlier?"* — runs
  alongside the unresolved questions. If it fires, the pinned facts are re-evaluated in a second
  pass; if not, they are skipped.

**Pinning uses top probability, not entropy confidence.** They are different questions and want
different numbers: `confidence` (entropy, 0.75) asks *"should a human look at this?"*, while
pinning (top probability, 0.6) asks *"can we stop re-deciding this?"*. Judging pinning by entropy
confidence never settled `intent`, because a 7-option question with a clear winner (p = 0.72)
scores only 0.42.

`scripts/bench_call.py` reproduces the table; `--out results/*.json` keeps a baseline to diff against.

## Second opinions, not second guesses

When a classification question is not decisive (top probability below 0.75 for `department` or
`intent`), the switchboard asks it **again in different words** — the paraphrase lives in
`dealership.department_question_paraphrase` — and compares the two answers.

- **Agreement** settles the question: two independently-worded phrasings landing on the same
  answer is evidence, and the fact is pinned (so later turns skip it).
- **Disagreement** does *not* overturn the first answer. It keeps it and flags the call for an
  LLM or a human.

In the vague-complaint scenario: **2 second opinions, 0 LLM calls**, and the call routes cleanly
to the Service Department.

Verification costs one extra question and largely pays for itself, because agreeing phrasings let
`department` and `intent` settle a turn earlier:

| collision | questions | tokens | compute |
|---|---|---|---|
| before | 28 | 3944 | 272 ms |
| incremental only | 16 | 1923 | 134 ms |
| + verification | 17 | 2020 | **130 ms** |

**The rejected design, and why.** The obvious tier-2 is to *narrow*: take the top three options
from a low-confidence pass and re-ask with only those. Measured, it does not improve the decision —
it re-rolls it and inflates confidence:

| utterance | tier 1 | narrowed to | tier 2 |
|---|---|---|---|
| "I have a problem with my car and need to bring it in" | service 0.37 | service/body_shop/general | **body_shop 0.73** ✗ |

A wrong answer made to look decisive is worse than no escalation at all, because everything
downstream now trusts it. `scripts/probe_slots.py` keeps the evidence. Confidence is only useful
if it tracks correctness.

## Speaking before thinking

Every turn emits a fixed acknowledgement — *"Let me take a look at that for you."* — **before any
forward pass runs**. It appears in the graph as its own node and in the conversation rail as an
extra switchboard bubble, so you can see the agent speak immediately rather than after the
cascade. It is a template, not generation; the point is that a voice channel needs *something*
within a few hundred milliseconds, and 30–60 ms of classification is not the only latency that
matters.

## Measurements (Apple M5 Pro, 64 GB)

Whole triage schema, batched in one forward pass:

| checkpoint | 1 question p50 | 6 questions p50 | throughput |
|---|---|---|---|
| `english` (ModernBERT-large, 421M) | 9.0 ms | 27.7 ms | 219 q/s |
| `multilingual` (mmBERT-base, 322M) | 4.6 ms | 11.1 ms | 520 q/s |
| `typed-decisions` (421M) | 9.2 ms | 27.8 ms | 218 q/s |

A full 4-turn call costs **~130 ms of compute**, **0 generated tokens**, **$0.00**, asks
**17 questions instead of 28**, and needs **0 LLM calls** (see the efficiency and second-opinion
sections above).

Quality, 18 hand-labelled tickets (`data/tickets/labelled.jsonl`, from the support-domain work):

| checkpoint | department acc | churn acc | refund acc | noul ECE ↓ |
|---|---|---|---|---|
| `english` | 0.83 | 1.00 | 0.94 | 0.073 |
| `multilingual` | 0.61 | 0.83 | 0.94 | 0.132 |
| `typed-decisions` | 0.83 | 0.94 | 0.89 | 0.170 |
| *chance* | *0.28* | *0.17* | *0.28* | — |

`typed-decisions` does not win: it is fine-tuned on four synthetic workflows matched by *exact
question-id sets*, so it transfers nothing to a custom schema.

## Layout

```
src/jev_classifier/
  dealership.py   departments, intents, slot enums, response templates, routing policy
  call.py         the turn driver -> node/edge event stream
  scenarios.py    scripted caller calls
  runs.py         record / replay (results/runs/*.jsonl)
  api.py          FastAPI: /api/scenarios /api/call /api/runs
  pipeline.py     the original generic support cascade (still reachable via CLI)
web/              Vite + React + React Flow front end
  src/graph/      canvas, deterministic layout, node components
  src/inspector/  drill-down panel
  src/conversation/ collapsible rail
  src/controls/   run / step / speed / replay
scripts/
  try_call.py        run a call and print the trace headless
  probe_slots.py     slot-wording measurements
  probe_questions.py phrasing measurements (support domain)
  bench_latency.py   latency / throughput
  bench_quality.py   accuracy / calibration
tests/            routing + policy tests (no model required)
```

CLI, no browser:

```bash
uv run python scripts/try_call.py --scenario collision   # a full call trace
uv run jev-classify --persona outage                     # the generic support cascade
uv run pytest -q
```

## Not built yet

Real speech-to-text and voice, and dropped-call recovery. Both were deliberately left out. The
record/replay event log is already a serialisable run, which is the foundation recovery would
need — the seam is a plain list of caller strings and a stream of events, so a real caller drops
in without touching the cascade or the UI.

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

- `GET /api/runs/{run_id}` validates the id against a strict charset *and* checks the resolved
  path stays inside `results/runs/` — a URL segment is never interpolated straight into a
  filesystem path.

## Licence

Project code is yours. Laya and its weights are Apache-2.0 by Convai Innovations; `laya-mlx` is an
independent Apache-2.0 port.
