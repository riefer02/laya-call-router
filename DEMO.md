# Demo notes

Everything below is measured; the caveats are the honest reading, not hedging. Say them — the
findings are the strongest part of this project, and a number quoted without its caveat is the one
thing that could embarrass you live.

## Before you start

```bash
uv sync
uv run uvicorn jev_classifier.api:app --port 8765      # backend
cd web && npm install && npm run dev                   # frontend on :5173
```

**Check the checkpoint is the fine-tune.** The app prints one of:

```
laya: serving fine-tuned checkpoint models/active        <- what you want
laya: serving the BASE checkpoint. Routing quality ...   <- the app is serving the wrong model
```

`models/active` is a symlink to the checkpoint to serve. Switch it with:

```bash
ln -sfn "$(pwd)/models/kaggle-out-v7/laya-dealership-routing" models/active
```

If it says BASE, the visualizer will show 0.654 destination / 0.518 sub-queue — the model this
project exists to replace — and nothing else about the app will look wrong.

## The calls that demo well

| scenario | what it shows |
| --- | --- |
| `call-collision` | "rear-ended me yesterday, I need body work" → **Body Shop**. Was going to Roadside. |
| `call-glass` | "a stone cracked my windscreen" → **Body Shop**. Was going to Roadside. |
| `call-detailing` | "full detail, inside and out" → **Detailing** |
| `call-no-start` | "my car won't start at all" → **Roadside / Towing**, dispatched |
| `call-flat-tire` | "I'm stuck on the highway" → **Roadside / Towing**, dispatched |
| `call-roadside` | "dead on the shoulder, I need a tow" → **Roadside / Towing** |

The two bookers routing to Body Shop and the three stranded callers dispatching to Roadside are the
same decision working in both directions — that is the point worth making.

**A booking call** ends in a filed appointment: the switchboard offers real times from the store's
hours, classifies which one was accepted, and files it with the caller's name.

## Numbers to quote

| metric | value | note |
| --- | --- | --- |
| routing, joint | **0.925** | 74 of 80 clean cases; `deepseek-flash` is indistinguishable at this size |
| routing, queue (the outcome) | **0.975** | |
| latency | **~23 ms** | against 1,400 ms for `deepseek-flash` |
| cost per call | **$0** | vs $0.011 |
| determinism | **1.00** | the LLM arms move 2.5 points between identical runs; ours has never moved |
| stranded callers caught | **17 of 17** | uncontaminated cases, zero missed |
| booking question, held out | **1.000** | including `unclear` 33/33 — it used to never abstain |

## What not to claim

- **Not "we beat deepseek."** Joint is a tie inside the noise, and the point estimate has swung
  ±3.8 points across runs of *their* model. The defensible claim is *parity, at 1/60th the latency,
  for nothing, with a number that does not move.*
- **Not "the safety classifier is fixed."** `is_safe_to_drive` catches every stranded caller, but at
  a realistic 2% hazard rate most dispatch flags would still be wrong (precision ~0.15, not 0.86 —
  the eval set is deliberately 40% positive). The queue override is right; the classifier's
  specificity is the weak part.
- **Not "1.000 booking accuracy"** without saying it is measured on held-out *reply phrasings*, not
  held-out conversations.
- **Not "the threshold is tuned."** The sweep is flat from 0.3 to 0.8 — the probabilities are
  saturated, so the threshold is not a control we actually have.

## If something goes wrong

- **Calls route badly** → the app is on the base checkpoint. Check the startup line.
- **A call misroutes to Roadside / Towing** → known: `call-vague` and `call-out-of-scope` both do.
  It is the safety flag overwriting the queue, and it is the next thing to fix.
- **The booking asks again instead of booking** → the confidence floor doing its job; add the turn
  where the caller names a time.

## The three findings worth saying out loud

1. **Seven silent failures now have tests** — including three measurements that were reading their
   own training data, and a checkpoint that trained on cases it was scored against.
2. **The safety question's precision is flattered by its own eval set** — 0.69 on a 40%-positive set,
   ~0.15 at the rate a switchboard sees. Nobody had computed that.
3. **The training data was in the wrong register** — 27-word chatbot prose against 5-word callers —
   which taught the safety classifier to treat any short fault report as a hazard.
