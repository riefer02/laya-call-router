"""Build training data for the two yes/no questions that drive dispatch and escalation.

They were never trained. `is_safe_to_drive` and `needs_human` were answered by whichever head the
base checkpoint shipped, with an encoder fine-tuned on a different task — and it showed:
`is_safe_to_drive` missed 4 of 18 stranded callers.

Two sources, deliberately:

* **The existing synthetic utterances**, which carry a realistic base rate. Most dealership calls
  are routine, and a model that never saw that would flag everything.
* **Generated hazards and complaints**, because the realistic rate is heavily one-sided and a model
  trained only on it learns to answer "safe" and be right most of the time. That is exactly the
  failure a safety-first policy cannot tolerate.

A candidate is kept only when **two independently-worded labelling passes agree**, the same rule the
routing labels use.

    uv run python scripts/generate_severity.py --limit 600
    uv run python scripts/generate_severity.py --resume
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_classifier import llm, synthgen, teacher  # noqa: E402

KEYS = ("is_safe_to_drive", "needs_human")
OUT = ROOT / "data" / "calls" / "severity_train.jsonl"

# Prompts for the positive class. Asking for "an unsafe call" produces things that are obviously
# unsafe and unlike real traffic; asking for the *shape* - a caller mentioning a symptom without
# saying whether they stopped - produces the cases that actually broke us.
HAZARD_PROMPT = (
    "Write {n} different things a caller might say to a car dealership that describe a vehicle "
    "that should NOT be driven. Include hazards the caller does not explicitly name as unsafe: "
    "smoke or a burning smell, a brake or steering failure, a stuck accelerator, overheating, "
    "a wheel or tyre that has failed, damage from an impact, or a vehicle that cannot be moved. "
    "Some should say they are stopped; some should just describe the symptom. "
    "Each must be one self-contained sentence. Write them {style}. Return JSON only."
)
COMPLAINT_PROMPT = (
    "Write {n} different things a caller might say to a car dealership that need a PERSON rather "
    "than an automated booking: a complaint about a visit or a repair, a billing dispute, a legal "
    "matter, an injury, a request to speak to a manager, or someone calling back after being "
    "ignored. Each must be one self-contained sentence. Write them {style}. Return JSON only."
)
ROUTINE_PROMPT = (
    "Write {n} different routine things a caller might say to a car dealership: booking a service, "
    "asking about a part, a quote, a test drive, opening hours, a cosmetic matter, or a minor fault "
    "the car still drives fine with. Each must be one self-contained sentence. "
    "Write them {style}. Return JSON only."
)
# The negative class this project was missing, and the reason the safety question fires on
# "the air conditioning isn't blowing cold any more".
#
# The old routine prompt asked for "a minor fault the car still drives fine with", so every
# fault-reporting negative arrived carrying its own reassurance. The model learned to look for the
# reassuring clause rather than judge the fault: it called any bare fault statement unsafe. And the
# register was wrong too - training lines ran to 27 words where a caller speaks five.
#
# So this asks for the deployment shape: short, bare, first-person, stating the fault and stopping.
MINOR_FAULT_PROMPT = (
    "Write {n} short things a caller might say to a car dealership about a fault that does NOT make "
    "the car unsafe to drive: air conditioning, heating, a radio or screen, a window, a seat, trim, "
    "paint, a door or boot release, a non-critical warning light, a cosmetic dent or scratch, a "
    "noise that is annoying rather than dangerous, or a slow leak that does not affect control. "
    "CRITICAL: each must be ONE SHORT SENTENCE of at most 14 words, stated plainly, the way a real "
    "caller speaks - name the fault and stop. Do NOT add that the car is safe, fine, drivable or "
    "still running. Do NOT reassure, explain or apologise. Do NOT ask whether they have the right "
    "number or the right place, and do not add any framing around the fault. "
    "Write them {style}. Return JSON only."
)

# The register is the point of this category. Anything long, hedged or self-reassuring teaches the
# shortcut rather than the distinction, so it is dropped rather than labelled.
MAX_MINOR_FAULT_WORDS = 18


def generate_utterances(kind: str, n: int, *, provider: str, model: str) -> List[str]:
    prompt = {
        "hazard": HAZARD_PROMPT,
        "complaint": COMPLAINT_PROMPT,
        "routine": ROUTINE_PROMPT,
        "minor_fault": MINOR_FAULT_PROMPT,
    }[kind]
    style = random.choice(synthgen.STYLES)
    schema = {
        "type": "object",
        "properties": {"utterances": {"type": "array", "items": {"type": "string"}}},
        "required": ["utterances"],
        "additionalProperties": False,
    }
    try:
        call = llm.chat_json(
            "You write realistic caller utterances for a car dealership phone line.",
            prompt.format(n=n, style=style),
            schema,
            provider=provider,
            model=model,
        )
    except Exception:  # noqa: BLE001 - a failed batch is not worth aborting the run
        return []
    data = call.get("data") if isinstance(call.get("data"), dict) else {}
    out = data.get("utterances") or []
    texts = [u.strip() for u in out if isinstance(u, str) and len(u.strip()) > 12]
    if kind == "minor_fault":
        texts = [t for t in texts if len(t.split()) <= MAX_MINOR_FAULT_WORDS]
    return texts


def label_two_passes(text: str, key: str, provider: str, model: str) -> Dict[str, Any]:
    """Agreement between two independently-worded passes, or nothing."""
    a = teacher.label_noul(text, key, provider=provider, model=model, variant=1)
    b = teacher.label_noul(text, key, provider=provider, model=model, variant=2)
    cost = (teacher.cost_of(a) or 0.0) + (teacher.cost_of(b) or 0.0)
    agreed = a["valid"] and b["valid"] and a["value"] == b["value"]
    return {
        "key": key,
        "value": a["value"] if agreed else None,
        "agreed": agreed,
        "raw": [a["raw"], b["raw"]],
        "cost": cost,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap the realistic utterances used (0 = all)")
    ap.add_argument("--hazards", type=int, default=60, help="batches of hazard utterances to generate")
    ap.add_argument("--complaints", type=int, default=40)
    ap.add_argument("--routine", type=int, default=30)
    ap.add_argument(
        "--minor-faults",
        type=int,
        default=0,
        help="batches of SHORT non-hazard fault reports (the deployment register)",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="")
    ap.add_argument(
        "--skip-realistic",
        action="store_true",
        help="do not re-use the routing corpus as severity utterances (for targeted top-up runs)",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument(
        "--rebalance-only",
        action="store_true",
        help="re-apply the positive-rate cap to what is already on disk, with no API calls",
    )
    args = ap.parse_args()

    if args.rebalance_only:
        if not OUT.is_file():
            raise SystemExit(f"nothing to rebalance: {OUT} does not exist")
        rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
        print(f"rebalancing {len(rows)} rows already on disk (no API calls)\n")
        final = rebalance(rows)
        write_rows(OUT, final)
        print(f"\nwrote {OUT} ({len(final)} utterances)")
        for key in KEYS:
            c = Counter(r[key] for r in final if key in r)
            tot = c.get(True, 0) + c.get(False, 0)
            rate = c.get(True, 0) / tot if tot else 0
            print(f"  {key:18s} true={c.get(True,0):4d} false={c.get(False,0):4d}  ({rate:.0%} positive)")
        return 0

    provider = args.provider
    model = args.model or llm.PROVIDERS[provider].default_model
    if not llm.available(provider):
        print(f"no API key for {provider!r}")
        return 2

    # ---- sources
    realistic = []
    src = ROOT / "data" / "calls" / "synthetic.jsonl"
    if src.is_file() and not args.skip_realistic:
        realistic = [json.loads(l)["text"] for l in src.read_text().splitlines() if l.strip()]
        random.Random(7).shuffle(realistic)
        if args.limit:
            realistic = realistic[: args.limit]
    print(f"realistic utterances: {len(realistic)}")

    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and OUT.is_file():
        for line in OUT.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["text"]] = row
        print(f"resuming with {len(existing)} labelled utterances already on disk")

    # ---- generate the positive classes (and some routine, to keep the negatives varied)
    generated: List[str] = []
    jobs = (
        [("hazard", args.hazards), ("complaint", args.complaints), ("routine", args.routine)]
        + ([("minor_fault", args.minor_faults)] if args.minor_faults else [])
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = []
        for kind, batches in jobs:
            for _ in range(batches):
                futs.append(pool.submit(generate_utterances, kind, args.batch,
                                        provider=provider, model=model))
        for fut in as_completed(futs):
            generated.extend(fut.result())
    generated = [g for g in dict.fromkeys(generated) if g not in existing]
    print(f"generated utterances: {len(generated)}")

    todo = [t for t in (realistic + generated) if t not in existing]
    print(f"to label: {len(todo)} utterances x {len(KEYS)} questions x 2 passes "
          f"= {len(todo) * len(KEYS) * 2} calls\n")

    rows: List[Dict[str, Any]] = list(existing.values())
    cost = 0.0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(label_two_passes, t, k, provider, model): (t, k)
            for t in todo
            for k in KEYS
        }
        done = 0
        for fut in as_completed(futs):
            text, key = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {"key": key, "value": None, "agreed": False, "raw": [str(exc)], "cost": 0.0}
            cost += res["cost"]
            row = next((r for r in rows if r["text"] == text), None)
            if row is None:
                row = {"text": text, "src": "generated" if text in generated else "synthetic"}
                rows.append(row)
            row[key] = res["value"]
            row[f"{key}_agreed"] = res["agreed"]
            done += 1
            if done % 400 == 0:
                kept = sum(1 for r in rows if r.get(f"{KEYS[0]}_agreed"))
                print(f"  {done}/{len(futs)} calls · ${cost:.4f} · {kept} usable", flush=True)

    # ---- balance
    # The generated hazards and complaints are deliberately over-represented, because the natural
    # rate is one-sided and a model trained only on it learns to answer "safe" and be right most of
    # the time. But over-representing has its own failure: at a 60% positive rate the model learns a
    # prior nothing like a real switchboard, and a threshold cannot undo a shifted prior. So the
    # positive rate is capped at ~40%, which keeps recall learnable without distorting the base rate
    # beyond what the operating point can absorb.
    final = rebalance(rows)
    write_rows(OUT, final)
    print(f"\nwrote {OUT} ({len(final)} utterances, ${cost:.4f})")
    for key in KEYS:
        c = Counter(r[key] for r in final if key in r)
        tot = c.get(True, 0) + c.get(False, 0)
        rate = c.get(True, 0) / tot if tot else 0
        print(f"  {key:18s} true={c.get(True,0):4d} false={c.get(False,0):4d}  ({rate:.0%} positive)")
    return 0


def rebalance(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge to one row per utterance, then cap each question's positive rate.

    Split out so it can be re-run without touching the API: `--rebalance-only`. Getting the rate
    wrong is a judgement call, and a judgement call should be cheap to revisit.

    **The cap has to come after the merge.** Capping each question's source rows independently and
    then merging undoes it: a row kept for one question carries the other question's label too, so
    the union drifts back towards whichever class was over-generated. Measured - `needs_human` was
    capped to 40% and came out at 57% once merged.
    """

    def labelled(row, key) -> bool:
        # Presence of the key IS the agreement. The pipeline tracks a `{key}_agreed` flag while
        # labelling, but it is dropped when rows are merged to one-per-utterance - so checking for
        # it here matched nothing and emptied this file once. The data was recoverable only by
        # paying for it again; the guard below exists so that cannot happen twice.
        return key in row and row[key] is not None

    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not any(labelled(row, k) for k in KEYS):
            continue
        m = merged.setdefault(row["text"], {"text": row["text"], "src": row.get("src", "")})
        for key in KEYS:
            if labelled(row, key):
                m[key] = row[key]
    final = list(merged.values())

    # Drop training rows that contain a held-out eval utterance. `sev-04` ("there's smoke coming
    # from under the hood.") appears verbatim inside two training rows, so the model credited with
    # catching it had already read it. An exact-text check does not see that - the eval row is a
    # *substring* of the longer training row - so the eval looked clean while it was not. Only
    # utterances long enough not to match by accident are compared, and either direction counts.
    eval_path = ROOT / "data" / "calls" / "severity.jsonl"
    if eval_path.is_file():
        held_out = [
            json.loads(line)["text"]
            for line in eval_path.read_text().splitlines()
            if line.strip()
        ]
        if held_out:
            before = len(final)
            final = [r for r in final if not synthgen.echoes_heldout(r["text"], held_out)]
            if before != len(final):
                print(
                    f"dropped {before - len(final)} training rows that contain a held-out "
                    "eval utterance verbatim"
                )

    rng = random.Random(3)
    keep = set(range(len(final)))
    for key in KEYS:
        idx = [i for i in keep if key in final[i]]
        pos = [i for i in idx if final[i][key] is True]
        neg = [i for i in idx if final[i][key] is False]
        # at most 2 positives per 3 negatives, i.e. a 40% ceiling
        allowed = max(len(neg), 1) * 2 // 3
        if len(pos) <= allowed:
            rate = len(pos) / max(1, len(pos) + len(neg))
            print(f"{key}: {len(pos)} true / {len(neg)} false kept ({rate:.0%} positive, no cap)")
            continue
        # Which rows to drop matters, because dropping a row removes it from BOTH questions.
        # Measured: capping `needs_human` cost 354 of 759 `is_safe_to_drive` positives - it thinned
        # the safety class by half as collateral. So rows that carry a positive label for another
        # question are dropped last. The two questions are trained as separate items, so preferring
        # to keep one does not distort the other's conditional distribution.
        def other_positive(i: int) -> bool:
            return any(final[i].get(k) is True for k in KEYS if k != key)

        rng.shuffle(pos)
        # sort DESCENDING by "carries another positive label", so the tail - the part that gets
        # dropped - is the rows whose removal costs the other question nothing
        pos.sort(key=other_positive, reverse=True)
        keep -= set(pos[allowed:])
        rate = allowed / max(1, allowed + len(neg))
        print(
            f"{key}: {allowed} true / {len(neg)} false kept "
            f"({rate:.0%} positive, capped from {len(pos)})"
        )

    out = [final[i] for i in sorted(keep) if any(k in final[i] for k in KEYS)]
    if not out:
        raise SystemExit(
            f"refusing to write an empty file from {len(rows)} input rows. "
            "Zero out of many is a bug, not a result."
        )
    return out


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write atomically, so a crash cannot leave a truncated file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
