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


def generate_utterances(kind: str, n: int, *, provider: str, model: str) -> List[str]:
    prompt = {"hazard": HAZARD_PROMPT, "complaint": COMPLAINT_PROMPT, "routine": ROUTINE_PROMPT}[kind]
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
    return [u.strip() for u in out if isinstance(u, str) and len(u.strip()) > 12]


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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="")
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
    if src.is_file():
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
    """Cap each question's positive rate, and merge to one row per utterance.

    Split out so it can be re-run without touching the API: `--rebalance-only`. Getting the rate
    wrong is a judgement call, and a judgement call should be cheap to revisit.
    """

    def usable(r, key):
        # Presence of the key IS the agreement. The pipeline tracks a `{key}_agreed` flag while
        # labelling, but it is dropped when rows are merged to one-per-utterance — so checking for
        # it here matched nothing and emptied this file once. The data was recoverable only by
        # paying for it again; the guard below exists so that cannot happen twice.
        return key in r and r[key] is not None

    rng = random.Random(3)
    out: List[Dict[str, Any]] = []
    for key in KEYS:
        pos = [r for r in rows if usable(r, key) and r[key] is True]
        neg = [r for r in rows if usable(r, key) and r[key] is False]
        rng.shuffle(pos)
        rng.shuffle(neg)
        keep_pos = pos[: max(len(neg), 1) * 2 // 3]
        keep_neg = neg[: max(len(keep_pos) * 2, 200)]
        out.extend(keep_pos)
        out.extend(keep_neg)
        rate = len(keep_pos) / max(1, len(keep_pos) + len(keep_neg))
        print(
            f"{key}: {len(keep_pos)} true / {len(keep_neg)} false kept "
            f"({rate:.0%} positive, from {len(pos)}/{len(neg)} available)"
        )

    merged: Dict[str, Dict[str, Any]] = {}
    for r in out:
        m = merged.setdefault(r["text"], {"text": r["text"], "src": r.get("src", "")})
        for key in KEYS:
            if usable(r, key):
                m[key] = r[key]
    final = [m for m in merged.values() if any(k in m for k in KEYS)]

    if not final:
        raise SystemExit(
            f"refusing to write an empty file from {len(rows)} input rows. "
            "Zero out of many is a bug, not a result."
        )
    return final


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write atomically, so a crash cannot leave a truncated file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
