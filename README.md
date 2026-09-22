# jev-classifier

A local, staged **support-triage cascade** built on [Laya](https://github.com/NandhaKishorM/laya) —
a non-autoregressive *System 1* decision model that returns **typed, calibrated decisions**
instead of generated text — running on Apple Silicon through [`laya-mlx`](https://pypi.org/project/laya-mlx/).

A support call is walked through a chain of classifiers (gate → language → department →
sub-intent → severity → routing). Every stage shows the question asked, the probability of each
option, the pick, and the branch taken. A low-confidence stage **asks the customer a follow-up**
instead of guessing, and the answer is appended to the transcript so the next forward pass is
better informed.

**The model never generates text.** `output_tokens` is `0` on every call — the UI displays it as
a live counter, because that is the whole point of the technology.

```
gate ──▶ language ──▶ department ──▶ sub-intent ──▶ severity ──▶ routing
choice    Router      choice         choice         score+noul    policy
 │                        │              │
 │                    <0.75 conf?    <0.75 conf?
 │                        └──── templated follow-up question ────┘
 └── spam / marketing ──▶ rejected, not routed
```

## Quick start

Requires an Apple Silicon Mac and Python 3.12 (`uv` will fetch it).

```bash
uv sync
uv run uvicorn jev_classifier.api:app --port 8765
# open http://127.0.0.1:8765
```

Pick a persona, or type your own support message. The right pane lights up stage by stage.

Headless, without a browser:

```bash
uv run jev-classify --persona outage
uv run jev-classify --ticket "I was charged twice, refund me or I cancel"
uv run jev-classify --list-personas
```

First run downloads the checkpoints (~0.8 GB English + ~0.65 GB multilingual; `typed-decisions`
is ~0.8 GB more and only fetched if you ask for it).

## What a run looks like

```
▶ Gate  [choice+noul]            english · 25.3 ms
    is_support_request: genuine_support  (conf 0.35, p 0.73)
      → genuine_support   ▓▓▓▓▓▓▓░░░ 0.73
        other             ▓▓░░░░░░░░ 0.21
        spam_or_marketing ▓░░░░░░░░░ 0.05
▶ Language  [router]             english · 0.09 ms
▶ Department  [choice]           english · 13.5 ms
    department: billing  (conf 0.95, p 0.99)
▶ Sub-intent  [choice]           english · 12.0 ms
    sub_intent: duplicate_charge  (conf 0.80, p 0.93)
▶ Severity  [score+noul]         english · 20.2 ms
    is_blocking: P(true)=0.82   frustration: 1.32/3
    churn_risk: P(true)=0.72    refund_requested: P(true)=0.80
▶ Routing
  queue=Refund Desk  priority=HIGH  handler=human
  flags=['churn_risk', 'refund_requested']

═ done: 71 ms compute, 6 stages, 0 tokens generated
```

## Design notes

- **Clarifying questions are templated, not generated.** Laya cannot generate text, so a
  low-confidence stage emits a fixed prompt from `schemas.CLARIFY_PROMPTS` with chips drawn from
  its option set. The customer's reply is appended to the transcript and re-sent as the model's
  state. That is what makes the cascade multi-step without a chatbot.
- **Confidence is entropy-based.** Laya's `confidence` is `1 − H(p)/log k`, not the top
  probability, so it depends on how many options a question has. A decisive 5-way choice tops out
  around 0.9; the default follow-up threshold is therefore **0.75**, not the 0.85 used in the
  upstream README. The UI shows *both* the entropy confidence and the top probability.
- **Routing policy is deterministic** (`routing.py`). Everything upstream is a calibrated
  probability; the policy that turns it into a queue is ordinary, readable business logic.
- **Stage 6 costs nothing.** The language stage is a script/language check (microseconds), and
  routing is pure Python — only four stages run a forward pass.

## Measurements on this machine (Apple M5 Pro, 64 GB)

### Latency / throughput — whole triage schema, one forward pass

| checkpoint | 1 question p50 | 6 questions p50 | throughput |
|---|---|---|---|
| `english` (ModernBERT-large, 421M) | **9.0 ms** | 27.7 ms | 219 q/s |
| `multilingual` (mmBERT-base, 322M) | **4.6 ms** | 11.1 ms | 520 q/s |
| `typed-decisions` (421M) | 9.2 ms | 27.8 ms | 218 q/s |

For reference, the upstream project's published T4 figures are 39.5 ms and 32.8 ms for one
question — this machine is ~4–7× faster. Peak MLX memory with all three checkpoints resident:
**2.9 GiB**.

### Quality — 18 hand-labelled tickets (`data/tickets/labelled.jsonl`)

| checkpoint | department acc | churn acc | refund acc | noul ECE ↓ |
|---|---|---|---|---|
| `english` | 0.83 | **1.00** | 0.94 | **0.073** |
| `multilingual` | 0.61 | 0.83 | 0.94 | 0.132 |
| `typed-decisions` | 0.83 | 0.94 | 0.89 | 0.170 |
| *chance baseline* | *0.28* | *0.17* | *0.28* | — |

Reproduce with `uv run python scripts/bench_latency.py` and `scripts/bench_quality.py`.

**Honest reading of the numbers**

- These are 18 tickets, so treat them as a smoke signal, not a benchmark. But the ordering is
  informative: **the plain `english` checkpoint is the best of the three here**.
- `typed-decisions` did *not* win, which matches the upstream documentation once you read it
  closely: it is fine-tuned on four specific synthetic workflows matched by *exact question-id
  sets* (`invoice_processing`, `security_incidents`, `customer_service`,
  `agent_trace_observability`). This project uses its own question ids, so that checkpoint gets
  none of its fine-tuning benefit and scores like the base model. It would matter only if you
  adopted those schemas verbatim.
- The follow-up threshold interacts with the metric: at the upstream 0.85 a clean 5-way
  department decision (`technical`, conf 0.77) spuriously triggers a follow-up. See
  `pipeline.DEFAULT_THRESHOLD`.
- `score` questions (`frustration`) are the weakest primitive — confidence 0.1–0.3 — matching the
  documented SST-5 weakness. The `noul` and `choice` primitives carry the cascade.
- Calibration measured better than the docs' warning implies (ECE 0.073 for English), but that is
  on easy, balanced examples. The docs' warning about over-confidence is worth taking seriously
  before gating real actions on these probabilities.

## Fine-tuning was the finding we did not pursue

The base checkpoints are weak zero-shot and highly sensitive to phrasing. `scripts/probe_questions.py`
is the evidence: asking *"Is this a genuine support request?"* as a `noul` scored **0.13** on an
obvious billing email, while a `choice` framing scored it correctly at 0.73; a churn question
phrased *"may leave for a competitor"* missed a message that literally said *"we'll have to
cancel"* (0.11) where *"threaten to cancel their plan or stop being a customer"* caught it (0.58).

Every schema in `schemas.py` was chosen from those measurements. The upstream documentation is
blunt that the real gains come from fine-tuning on your own domain (RLCD). That is the natural
next step and is not part of this build.

## The gate is threshold-based, not argmax

The gate was originally "reject unless the choice question picks `genuine_support`". The web demo
falsified that: the multilingual checkpoint scored a genuine Chinese billing ticket
**spam 0.37 / other 0.33 / genuine 0.30** — an entropy confidence of 0.00 — so the argmax rule
rejected a valid ticket. Real spam scores 0.94–0.96, so the gate now rejects only when
`p(spam) ≥ 0.6` (`schemas.GATE_REJECT_THRESHOLD`). A high-confidence requirement on a hard stop
is the general lesson: never let a near-uniform distribution trigger an irreversible action.

That leaves a thinner margin than is comfortable on sales enquiries, which legitimately resemble
marketing (a pricing request scored `p(spam)=0.51`). This is exactly the kind of boundary a
fine-tuned checkpoint would sharpen.

## Layout

```
src/jev_classifier/
  agent.py      shared Router (preloaded checkpoints)
  schemas.py    the 6 stages' typed questions — tuned by measurement
  pipeline.py   resumable cascade state machine, confidence gating, multi-turn
  routing.py    deterministic stage-6 routing policy
  api.py        FastAPI + SSE stream of stage events
  cli.py        headless runner
  personas.py   scripted example support calls
web/            single-page UI (no build step)
scripts/
  smoke_test.py      primitives + Router
  probe_questions.py phrasing experiments
  bench_latency.py   latency / throughput
  bench_quality.py   accuracy / calibration on the labelled set
data/tickets/   sample + labelled tickets
tests/          routing logic tests (no model required)
```

## Licence

Project code is yours. Laya and its weights are Apache-2.0, by Convai Innovations; `laya-mlx` is
an independent Apache-2.0 port. See `NOTICE`-style attribution in those projects.
