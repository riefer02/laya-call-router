"""H1 — the external generality test.

Does the fine-tune still answer questions it was never trained on? Laya's core claim is that the
option space is defined at request time, so a new schema needs no retraining. We fine-tuned hard on
43 fixed intents; this measures what that cost.

    uv run python scripts/generality_test.py --limit 400
    uv run python scripts/generality_test.py --finetuned models/kaggle-out-v2/laya-dealership-routing

Runs locally. No API keys, no cost.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import laya_mlx as laya

from jev_classifier import generality as G

ROOT = Path(__file__).resolve().parents[1]


def load_agent(path: str, head_max_len: int, max_len: int):
    agent = laya.load(path)
    G.configure(agent, head_max_len=head_max_len, max_len=max_len)
    return agent


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t0) * 1000


def run_suite(agent, texts, gold, options, label: str, report: int) -> dict:
    preds, ms = timed(G.ask_choice, agent, texts, options, batch_report=report)
    acc = G.accuracy(preds, gold)
    print(f"    {label:22s} accuracy {acc:.3f}   ({ms/1000:.0f}s, {ms/max(1,len(texts)):.0f} ms/item)")
    return {
        "arm": label,
        "n": len(texts),
        "accuracy": acc,
        "latency_ms_per_item": round(ms / max(1, len(texts)), 1),
        "confusions": G.top_confusions(preds, gold, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="cases per suite (0 = all)")
    ap.add_argument("--finetuned", default="models/kaggle-out-v2/laya-dealership-routing")
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--raised", action="store_true",
                    help="also run Banking77 with a 512-token option budget, to measure the "
                         "documented >20-option weakness")
    ap.add_argument("--out", default="results/generality.json")
    args = ap.parse_args()

    print("=== external generality suites (real human text, not dealership) ===\n")
    b_texts, b_gold, b_labels = G.load_banking77(limit=args.limit)
    c_texts, c_gold, c_ood, c_ood_gold, c_options = G.load_clinc150(limit=args.limit)
    print(f"Banking77 : {len(b_texts)} queries, {len(b_labels)} runtime-defined options")
    print(f"CLINC150  : {len(c_texts)} in-domain + {len(c_ood)} out-of-domain, {len(c_options)} options")
    print(f"token budget pinned to max_len={args.max_len} head_max_len={args.head_max_len}\n")

    report: dict = {
        "limit": args.limit or None,
        "token_budget": {"max_len": args.max_len, "head_max_len": args.head_max_len},
        "suites": {},
    }

    ft_path = ROOT / args.finetuned
    arms = [("base", "convaiinnovations/laya")]
    if ft_path.is_dir():
        arms.append(("fine-tuned", str(ft_path)))
    else:
        print(f"!! fine-tuned checkpoint not found at {ft_path} — running base only\n")

    for name, path in arms:
        print(f"[{name}] loading ...", flush=True)
        agent = load_agent(path, args.head_max_len, args.max_len)
        entry: dict = {}

        print("  Banking77 (77 real customer-service intents)")
        entry["banking77"] = run_suite(agent, b_texts, b_gold, b_labels, name, max(100, len(b_texts) // 4))

        print("  CLINC150 in-domain (151 options incl. out_of_domain)")
        entry["clinc150_indomain"] = run_suite(agent, c_texts, c_gold, c_options, name, max(100, len(c_texts) // 4))

        print("  CLINC150 out-of-domain detection (P(ood) via noul)")
        probs = G.ask_noul(
            agent,
            c_ood,
            "Is this request outside the domain of the assistant, i.e. unrelated to what it "
            "handles, rather than a genuine in-domain request?",
        )
        in_probs = G.ask_noul(
            agent,
            c_texts,
            "Is this request outside the domain of the assistant, i.e. unrelated to what it "
            "handles, rather than a genuine in-domain request?",
        )
        tp = sum(1 for p in probs if p >= 0.5)
        fp = sum(1 for p in in_probs if p >= 0.5)
        entry["clinc150_ood"] = {
            "ood_recall": round(tp / max(1, len(probs)), 4),
            "in_domain_false_positive_rate": round(fp / max(1, len(in_probs)), 4),
            "mean_p_ood_on_ood": round(statistics.fmean(probs), 4) if probs else None,
            "mean_p_ood_on_indomain": round(statistics.fmean(in_probs), 4) if in_probs else None,
        }
        print(
            f"    {name:22s} OOD recall {entry['clinc150_ood']['ood_recall']:.3f}"
            f"  false-positive {entry['clinc150_ood']['in_domain_false_positive_rate']:.3f}"
        )

        report["suites"][name] = entry

        if args.raised and name == "base":
            print("  Banking77 with a raised 512-token option budget")
            agent512 = load_agent(path, 512, 1024)
            entry["banking77_head512"] = run_suite(
                agent512, b_texts, b_gold, b_labels, f"{name}/head512", max(100, len(b_texts) // 4)
            )

    # ---- verdict -------------------------------------------------------------
    print("\n" + "=" * 68)
    rows = []
    for suite, label in (("banking77", "Banking77 acc"), ("clinc150_indomain", "CLINC150 in-domain acc")):
        row = [label]
        for name, _ in arms:
            s = report["suites"].get(name, {}).get(suite)
            row.append(f"{s['accuracy']:.3f}" if s else "-")
        rows.append(row)
    rows.append(
        ["CLINC150 OOD recall"]
        + [
            f"{report['suites'][n]['clinc150_ood']['ood_recall']:.3f}"
            if n in report["suites"]
            else "-"
            for n, _ in arms
        ]
    )
    from jev_classifier.evalharness import render

    print(render("EXTERNAL GENERALITY (real human text)", rows, ["metric", *[n for n, _ in arms]]))
    print("\nreference: Laya publishes 0.425 on Banking77 at a 256-token option budget.")

    base = report["suites"].get("base", {}).get("banking77", {}).get("accuracy")
    ft = report["suites"].get("fine-tuned", {}).get("banking77", {}).get("accuracy")
    if base is not None and ft is not None:
        change = ft - base  # signed: positive means fine-tuning gained
        drop = base - ft  # unsigned amount lost, which is what the verdict thresholds mean
        if drop <= 0.03:
            verdict = "GENERALITY HELD — fine-tuning did not cost meaningful option-space flexibility."
        elif drop <= 0.12:
            verdict = "PARTIAL LOSS — fine-tuning cost some generality; per-store config is still plausible but check it."
        else:
            verdict = "GENERALITY LOST — the fine-tune no longer handles new option spaces; per-store config means per-store retraining."
        # Report the signed change. Printing the drop as "delta" reads as a gain when generality
        # was lost, which is the one direction this must never be wrong about.
        print(
            f"\n{verdict}\n  Banking77 base {base:.3f} -> fine-tuned {ft:.3f}  "
            f"(change {change:+.3f}, {drop:+.3f} lost)"
        )
        report["verdict"] = {
            "basis": "banking77",
            "base": base,
            "finetuned": ft,
            "change": round(change, 4),
            "drop": round(drop, 4),
            "text": verdict,
        }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
