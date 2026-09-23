# Audit brief

**Historical brief.** This records questions posed before the v7 and v14 audits. Some claims and
counts below are superseded. Use [README.md](README.md) for the current demo,
[results/README.md](results/README.md) for measured reports, and [NEXT.md](NEXT.md) for current
work.

A second agent is auditing this project's approach and conclusions. This file exists so the audit is
adversarial and specific rather than a re-read. It is a briefing, not a status report — `NEXT.md` is
the plan and `LEARNINGS.md` is the narrative.

**Please try to falsify the claims below.** The author has already been wrong three times in one
session — on the cause of the probability saturation, on the fix for the dispatch problem, and on the
cause of a call-level regression — so the prior should be that some of what follows is wrong too.
A summary of agreement is worth less than the strongest counter-evidence you can find.

## What the project is

A local car-dealership call-router built on Laya (a small non-autoregressive model returning typed
decisions), fine-tuned via the official RLCD trainer on Kaggle 2×T4. It is evaluated against
`gpt-5.4-nano` and `deepseek-flash` on 81 hand-labelled routing cases and 27 scripted calls.

## Where the truth is

| what | where |
| --- | --- |
| taxonomy, question text, policy thresholds | `config/store_profile.json` |
| routing policy (the `queue` decision) | `src/jev_classifier/dealership.py`, `route()` |
| evaluation harness, Wilson CIs, base-rate maths | `src/jev_classifier/evalharness.py` |
| raw measurements | `results/*.json`, indexed by `results/README.md` |
| ground truth | `data/calls/*.jsonl` |

## Claims to falsify, in the order I would expect to be wrong

### 1. "The training data's register causes the safety false positives" — LIKELIEST TO BE WRONG

**Claim.** Training utterances have a median 27 words; the eval sets are 11-12 and hand-written
caller turns are 5. The negative class contains essentially no *short, bare statement of a fault
that isn't a hazard*, while every positive is a terse fault statement. Therefore the model learned
"short + something's wrong ⇒ unsafe", which is the observed false-positive pattern.

**Why to distrust it.** It is a correlation between two measured distributions plus a plausible
mechanism. No *interventional* evidence yet. Specific things to check:

- Is output actually predicted by length? Stratify the 45 severity cases (and ideally the training
  rows) by word count and see whether P(unsafe) jumps at short lengths. If a short *hazard* and a
  short *non-hazard* are separated correctly, length is not the driver and the claim is weak.
- Are the 8 false positives separable by something other than length — the presence of a "fault
  verb" without a "reassurance" clause? If so, the mechanism is semantic, not register, and the fix
  (terse data) may not address it.
- Only 8 false positives and 27 negatives: the whole story rests on ~8 items. What is the interval?

### 2. "Uncoupling dispatch from routing would break correct behaviour" — my most recent decision

**Claim.** `dealership.py` overwrites `queue` with `Roadside / Towing` when `is_safe_to_drive` fires.
I proposed separating them, then withdrew it because `call-no-start` and `call-flat-tire` expect
`Roadside / Towing` and reach it only via that override, so the change would break two correct calls.

**Why to distrust it.** The retraction is only as good as the ground truth it defers to. Check:
- Is `Roadside / Towing` actually the right *queue* for those two cases, or does the ground truth
  encode the very conflation being questioned? If the ground truth is wrong, the retraction is wrong
  and the original fix may have been right.
- Note the taxonomy's own stated principle (`LEARNINGS.md`, "Structure beat everything else") says
  roadside is a *flag, not a place the call goes*. That contradicts the ground truth. Which is right?

### 3. "Precision is 0.064 at a 2% base rate"

**Claim.** Specificity 0.704 on 27 negatives; Bayes-corrected precision at a 2% base rate is 0.064.

**Why to distrust it.** Specificity is estimated from 8 false positives out of 27 — the interval is
wide (roughly ±0.15), and the 2% deployment rate is assumed, not measured. Check whether the
conclusion ("most trucks roll for nothing") survives plausible variation in both, and whether 2% is
defensible for a dealership switchboard at all.

### 4. "The eval sets are genuinely held out"

**Claim.** Tests assert no eval case is a substring of a training row, and the acceptance dev split
shares no reply phrasing.

**Why to distrust it.** I found three leaks by luck. Try to find a fourth: paraphrase-level
near-duplicates that survive exact and containment checks, and label leakage in
`data/calls/severity.jsonl` versus `severity_train.jsonl` at the *semantic* level, not the string
level.

### 5. "The routing numbers are comparable across runs"

**Claim.** v6 measures joint 0.926; `eval_v4.json` is labelled non-comparable because it predates the
safety rewording.

**Why to distrust it.** Check whether the training data used by each kernel run is the same as what is
now on disk. The severity file has since been trimmed for held-out echoes (1736 → 1645 rows) and is
being extended again right now. If any two runs being compared used different data, the comparison
needs a caveat that is not currently written.

## Things that are cheap to check and would be embarrassing if wrong

- Does `scripts/eval_acceptance.py::assert_held_out` actually fire when pointed at the training file?
  Run it with `--data data/calls/acceptance_train.jsonl` and confirm it refuses.
- Does `training/train_ddp.py` still reproduce the v6 checkpoint's behaviour, given the
  `temperature_by_options` change? v6 was trained *before* that fix.
- Does the `minor_fault` prompt actually produce the register it claims after labelling, or does the
  teacher's two-pass filter admit longer, hedged examples?

## Environment

- No GPU needed for any of the above; the model runs locally via `laya-mlx` on Apple Silicon.
- `uv run pytest -q` (111 tests) and `uv run python scripts/eval_severity.py` are the entry points.
- API keys exist in `.env` for the LLM arms. **Do not spend on generation or labelling without
  asking** — the teacher calls are the only real cost in this project.
- `data/calls/severity_train.jsonl` is being regenerated by a background job as this is written.
  Read it before or after, not during.
- Do not commit secrets; `.githooks/pre-commit` refuses credential patterns.

## What would change the plan

The plan's Move 1 is "generate training data in the caller's register". It should be abandoned or
reordered if claim 1 fails, or if claim 2's ground truth turns out to be the thing that is wrong.
`NEXT.md` says which move each finding would change.
