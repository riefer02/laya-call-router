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

## Measurements (Apple M5 Pro, 64 GB)

Whole triage schema, batched in one forward pass:

| checkpoint | 1 question p50 | 6 questions p50 | throughput |
|---|---|---|---|
| `english` (ModernBERT-large, 421M) | 9.0 ms | 27.7 ms | 219 q/s |
| `multilingual` (mmBERT-base, 322M) | 4.6 ms | 11.1 ms | 520 q/s |
| `typed-decisions` (421M) | 9.2 ms | 27.8 ms | 218 q/s |

A full 4-turn call costs **~250 ms of compute**, **0 generated tokens**, **$0.00**.

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
