# jev-classifier — demo pack

A local call-router for a car-dealership switchboard. It runs on a laptop, decides in ~20 ms, costs
nothing per call, and every decision it makes is inspectable.

The model is [Laya](https://github.com/NandhaKishorM/laya) — a small **non-autoregressive** model that
returns *typed decisions* instead of generating text. `tokens generated` stays at `0` on every call.
That is the point of the technology and the reason the economics work.

![the debugger](docs/overview.png)

---

## How a call flows

```mermaid
flowchart LR
    A["caller speaks"] --> B["acknowledgement<br/>a template, no model"]
    B --> C{"destination · vehicle<br/>location · when<br/>unsafe-to-drive?<br/>needs-a-human?"}
    C -->|"one batched forward pass"| D{"sub-queue<br/>branched on the destination"}
    D --> E["policy: what next?"]
    E -->|"a slot is missing"| F["ask the caller"]
    F --> A
    E -->|"ready to book"| G["offer real times<br/>from the store's hours"]
    G --> H{"which one did<br/>they accept?"}
    H -->|"confident, and it matches<br/>a time that was offered"| I["file the appointment"]
    H -->|"weak, or names a time<br/>that was not offered"| G
    C -->|"unsafe to drive"| J["dispatch roadside assistance"]
```

Two things to notice, because they are the design:

- **The classifier does *understanding*** (department, intent, slot values, urgency, escalation).
  **A deterministic policy does *control flow*** (which slot to ask for, when the booking is done,
  which queue it lands in). Measured: a `next_action` question answered at confidence 0.03 and picked
  `ask_detail` regardless of input. Control flow is not a good classifier question.
- **The un-gettable-by-classifier facts are rules**: the exact appointment time, the caller's name,
  the callback number, and whether the reply names a time that was offered. A model should not be
  asked to parse what a regex cannot get wrong.

---

## What you see in the debugger

Rows are turns. Columns are decisions, left to right. Nothing moves while a call streams in, so it
screen-records cleanly.

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

- **Colour says who decided**: caller (blue) · model (violet) · rule or regex (slate) · switchboard
  (green) · route out (amber). A node's badge is its primitive — `choice`, `noul`, `policy`, `regex`.
  **The UI never pretends a model decided something a rule decided.**
- **Click any box** for the question, every option with its probability, entropy confidence *and* top
  probability, which checkpoint answered, why it was routed there, latency, and raw JSON.
- **`follow` keeps the camera on the active decision**; turn it off to frame the whole call.
- **Playback is client-side**, so play / pause / step / scrub never touch the backend. Every run is
  replayable from the recording.

---

## How it was built and measured

```mermaid
flowchart TB
    subgraph build["building the training data"]
        A["store profile<br/>taxonomy · question text · policy"] --> B["generate candidate<br/>utterances"]
        B --> C{"two independently-worded<br/>labelling passes agree<br/>with each other and the target?"}
        C -->|"yes"| D["kept as a training example"]
        C -->|"no"| E["dropped"]
        D --> F["acceptance criteria:<br/>per-class floors · no duplicates ·<br/>nothing near the held-out 81"]
    end
    subgraph train["training — free, on Kaggle 2xT4"]
        F --> G["packaged with a manifest<br/>sha256 per file"]
        G --> H["RLCD fine-tune<br/>8 epochs, five questions"]
        H --> I["snapshot audited against the eval sets<br/>before anyone quotes a number"]
    end
    subgraph measure["measuring — four arms, same cases"]
        I --> J["our cascade<br/>base + fine-tuned"]
        I --> K["gpt-5.4-nano"]
        I --> L["deepseek-flash"]
        J --> M["81 routing cases<br/>27 scripted calls<br/>45 safety cases"]
        K --> M
        L --> M
    end
```

Every gate in that pipeline is a command that **exits non-zero when it fails**, rather than a
judgement call. The teacher is `deepseek-flash`, and a candidate only becomes training data when two
independently-worded passes agree — a high-precision filter that yielded zero invalid labels in
1,394 rows, and which is also why the teacher's agreement with our hand labels (0.975 destination /
0.901 sub-queue) is the ceiling anything distilled from it can reach.

---

Measured on 81 hand-labelled routing cases and 27 scripted calls, against two frontier arms.

| | base model | **the fine-tune we demo** | gpt-5.4-nano | deepseek-flash |
| --- | --- | --- | --- | --- |
| destination | 0.654 | 0.914 | 0.951 | **0.975** |
| joint (dest + sub-queue) | 0.518 | 0.876 | 0.864 | **0.889** |
| **call outcome** (27 calls) | 0.778 | **0.926** | 0.963 | 0.963 |
| latency, per call | 22 ms | **23 ms** | 750 ms | 1,430 ms |
| cost per call | **$0** | **$0** | $0.0025 | $0.0113 |
| determinism (3 repeats) | **1.00** | **1.00** | 0.98 | 0.99 |

**The honest claim is parity, not victory.** `deepseek-flash` has scored joint 0.889–0.926 across four
runs on *identical inputs*; ours has never moved. On 81 cases one case is 1.2 points, so the intervals
overlap and the point estimate is not a finding. What is defensible: **we match a frontier model's
routing at 1/60th the latency, for nothing, with a number that stays put.**

Three checkpoints exist and the differences matter for what you show:

| | routing joint | call outcome | booking | provenance |
| --- | --- | --- | --- | --- |
| v6 (two questions trained) | **0.926** | 0.852 | not measured | 7 contaminated rows |
| **v7 — what `models/active` points at** | 0.876 | **0.926** | **1.000** | clean |
| v14 (the clean experiment) | 0.889 | 0.889 | 1.000 | clean |

v7 is loaded because **the call outcome is the product metric**, and v7 is the only one that gets the
body-shop bookings right — v6 sends *"rear-ended me, I need body work"* and *"a stone cracked my
windscreen"* both to Roadside / Towing, which a viewer would spot immediately.

---

## The safety questions

`is_safe_to_drive` dispatches roadside assistance. `needs_human` decides whether a person takes the
call. Neither had ever been measured before this session.

At the shipped threshold, on 45 labelled cases — **17 of 17 stranded callers caught, none missed**:

| | recall | false alarms | precision on the set | @5% base rate | @2% base rate |
| --- | --- | --- | --- | --- | --- |
| untrained | 1.000 | 3 | 0.857 | 0.321 | 0.155 |
| trained on the long data | 1.000 | 8 | 0.692 | 0.151 | 0.064 |
| **trained on the register-corrected data** | 1.000 | **2** | **0.900** | **0.415** | **0.216** |

**Training made this worse, and then fixing the training data made it better than not training at
all.** That is the clearest single result of the session: the same 45 cases, the same model, the same
question — 8 false alarms down to 2, with recall untouched.

**But read the last column, because it flatters us.** The set is **40% positive** — deliberately,
because unsafe calls are rare and you need them concentrated to measure recall at all. Precision
depends on the base rate, and 40% is eight to twenty times what a switchboard sees. At an assumed 2%
hazard rate, precision is 0.216 — so most dispatch flags would still be wrong, just far fewer of them.
The base rate itself is assumed, not measured, and the code sets a *flag and a queue* rather than
sending a truck. `eval_severity.py` prints this table for every arm and a test pins the round trip.

---

## What we found

The model barely changed this session. **What changed is that the measurements behind it are true.**

**Seven silent failures now have tests.** The recurring shape is one fact in two places — or two facts
in one place — resolved quietly and wrongly. None of them threw an error:

1. The taxonomy contradicted itself (a sub-queue advertised tire rotations while the rotation case was
   labelled elsewhere) — 5 points of ceiling lost to a definitional clash.
2. A renamed field left a stale reference that silently swallowed a third of the generation work.
3. Training and inference asked the sub-queue question with different capitalisation — the fine-tune
   would have been asked a question it had never seen.
4. One safety threshold existed in three places, so a caller could be dispatched as unsafe while the
   audit trail said they were not.
5. The Kaggle notebook is *generated*; the generated copy had drifted and copied four of six data
   files, so a whole GPU run trained 2,512 items instead of 6,210 with no error.
6. The epoch count lived in a shell variable, and regenerating the notebook silently reset the run
   from eight epochs to four.
7. `build_items.py` printed a note and carried on when a data file was missing — which is what made
   #5 invisible.

**Three measurements were reading their own training data.** The acceptance classifier reported
**1.000 accuracy** — it was scoring the file training is built from. Two more leaked by *containment*:
a short eval utterance sitting inside a longer training row, which no exact-text check can see.

**And every fine-tune from v3 onward was scored partly on data it trained on.** `gen-01` sat verbatim
inside a training row in four checkpoints, and all four answered it correctly. My guard checked
`data/calls/`; the checkpoints trained on the frozen snapshot beside their weights. **A second reader
found that by hand, after the numbers had been published.** `kaggle_run.py watch` now audits the
snapshot at download and records a sha256 per file, so a checkpoint carries its own provenance.

**The training data was in the wrong register.** 27-word chatbot prose against 11-word eval cases and
5-word caller turns. The negative class contained no *short, bare statement of a fault that isn't a
hazard*, so the only rule available to the model was **"short + something's wrong ⇒ unsafe"** — which
is exactly the false-positive pattern:

```
"The air conditioning isn't blowing cold air any more."    called unsafe, p=0.999
"The driver's seat won't slide forward any more."          called unsafe, p=1.000
```

Generating terse examples in **both** classes cut dispatch false alarms from **10 to 3** with recall
held at 1.000 — confirmed by a clean additive run. The same addition cost 3.7 points of routing and
`needs_human` recall, because all five questions share one encoder, so data added for one question is
not local to it.

**And the app was serving the base model.** `get_router()` never named a checkpoint, so every call was
the stock 0.654 model while the docs said 0.963. Calls ran, the graph drew, the probabilities were
plausible. They were the wrong model's.

---

## What isn't done

- **The dispatch still fires too often for a real switchboard.** Precision at an assumed 2% hazard
  rate is 0.216 — far better than the 0.064 it was, and still meaning most flags would be wrong. The
  queue override itself is *correct*: it is how someone stuck on the highway reaches a tow instead of
  a booking. So the fix is more precision, not structure.
- **`needs_human` recall** fell to 0.714 when the terse hazard rows were added: they are all escalation
  positives and they dilute a class otherwise made of complaints and billing disputes.
- **A second human labeller.** `det-04` is missed by every model including both frontier arms; `gen-08`
  is answered against our label by all three. Where every model disagrees with the key, the key is the
  likeliest thing to be wrong. This is the ceiling on the destination number.
- **The store facts are loaded but unused** — the switchboard transfers "what time do you open?"
  instead of answering it.
- **The `other` fallback is dead.** The fine-tuned model never abstains, so there is no "I'm not sure"
  left anywhere in the system.

---

## Running the demo

```bash
uv sync
uv run uvicorn jev_classifier.api:app --port 8765      # backend
cd web && npm install && npm run dev                   # frontend on :5173
```

Open <http://localhost:5173>.

**Check the startup line before you present:**

```
laya: serving fine-tuned checkpoint models/active        <- correct
laya: serving the BASE checkpoint. Routing quality ...   <- STOP, wrong model
```

`models/active` is a symlink to the checkpoint to serve. To switch it:

```bash
ln -sfn "$(pwd)/models/kaggle-out-v7/laya-dealership-routing" models/active
```

---

## The demo script

**1. Route a call.** Run `call-collision` — *"Someone rear-ended me in a parking lot yesterday. I need
body work."* → **Body Shop**. Click the destination node to show the options and probabilities, and
point out the badge: `choice`, a model decision.

**2. Show the same decision working the other way.** Run `call-no-start` — *"my car won't start at
all"* → **Roadside / Towing**, dispatched, priority HIGH. Same question, opposite answer. The two
bookers and the three stranded callers in the table below are one decision working in both directions.

**3. Book an appointment.** Run `buy_car`. Watch it refuse three times, then book:

```
turn 3  "maybe Thursday"             -> asks again   (Thursday is not offered)
turn 4  "this is Dana, 555-0140"     -> asks again   (no time mentioned at all)
turn 5  "Tuesday at 8 works for me"  -> asks again   (Tuesday is not offered)
turn 6  "the first one please"       -> books
```

Turns 4 and 5 are the two failure modes this used to have — inventing an agreement from no time, and
from the wrong day. **This is the best moment in the demo**: a system declining to act on something it
cannot verify, three times, and then filing a real appointment with a name on it.

**4. Show the evidence.** The `Evidence` tab reads the measurements live from `results/*.json`.

### Calls that demo well

| scenario | what it shows |
| --- | --- |
| `call-collision` | booking body work → **Body Shop** |
| `call-glass` | cracked windscreen → **Body Shop** |
| `call-detailing` | full detail → **Detailing** |
| `call-no-start` | won't start → **Roadside / Towing**, dispatched |
| `call-flat-tire` | stuck on the highway → **Roadside / Towing**, dispatched |
| `call-roadside` | dead on the shoulder, needs a tow → **Roadside / Towing** |
| `buy_car` | the booking sequence above |

### If something goes wrong

- **Calls route badly** → the app is on the base checkpoint. Check the startup line.
- **A call misroutes to Roadside / Towing** → known: `call-vague` and `call-out-of-scope` both do. It
  is the safety flag overwriting the queue, and it is the next thing to fix.
- **The booking asks again instead of booking** → the offered-time veto doing its job. If the caller
  names a day that was not offered it will always ask again; that is the fix, not a fault.

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
