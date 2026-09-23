"""Multi-arm evaluation: our cascade against cheap generative models, on labelled ground truth.

    uv run python scripts/eval.py
    uv run python scripts/eval.py --llm-arms openai:gpt-5.4-nano,deepseek:deepseek-flash
    uv run python scripts/eval.py --skip-llm           # cascade arms only, no key needed
    uv run python scripts/eval.py --limit 20 --determinism 1

Writes results/eval.json. Keys are read from .env / the environment and never printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev_classifier import evalharness as H
from jev_classifier import llm
from jev_classifier.agent import get_router

import laya_mlx as laya

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS = "openai:gpt-5.4-nano,deepseek:deepseek-flash"


def short(ref: str) -> str:
    return ref.split(":")[-1]


def usable(refs: list[str]) -> list[str]:
    out = []
    for ref in refs:
        provider, _ = llm.parse_model(ref)
        if llm.available(provider):
            out.append(ref)
        else:
            print(f"  skipping {ref}: no {llm.PROVIDERS[provider].key_env}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap the routing cases (0 = all)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--determinism", type=int, default=3, help="repeats for the agreement check")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--llm-arms", default=DEFAULT_ARMS)
    ap.add_argument("--hybrid-arm", default="", help="which arm the hybrid escalates to")
    ap.add_argument(
        "--finetuned",
        default="",
        help="path to a locally fine-tuned Laya checkpoint; adds a 'cascade-ft' arm",
    )
    ap.add_argument(
        "--exclude-cases",
        default="",
        help=(
            "comma-separated routing ids to drop from EVERY arm. For cases a checkpoint was "
            "contaminated by (see scripts/audit_snapshots.py): excluding is a measurement, whereas "
            "counting the case as wrong is a what-if penalty. Applied symmetrically, or not at all."
        ),
    )
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    routing = H.load_routing()
    calls = H.load_calls()
    if args.limit:
        routing = routing[: args.limit]

    excluded = [c.strip() for c in args.exclude_cases.split(",") if c.strip()]
    if excluded:
        known = {c.id for c in routing}
        unknown = [c for c in excluded if c not in known]
        if unknown:
            raise SystemExit(f"--exclude-cases names ids that are not routing cases: {unknown}")
        routing = [c for c in routing if c.id not in excluded]
        print(f"excluding {len(excluded)} case(s) from every arm: {', '.join(excluded)}")

    refs = [] if args.skip_llm else usable([r.strip() for r in args.llm_arms.split(",") if r.strip()])
    hybrid_ref = args.hybrid_arm or (refs[0] if refs else "")

    print(f"ground truth: {len(routing)} routing cases, {len(calls)} calls")
    print(f"llm arms: {', '.join(refs) if refs else 'none'}")

    router = get_router()
    print("preloading checkpoints ...", flush=True)
    router.preload(["english", "multilingual"])

    ft_router = None
    if args.finetuned:
        path = Path(args.finetuned)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            print(f"  --finetuned path does not exist: {path}")
            return
        print(f"loading fine-tuned checkpoint from {path} ...", flush=True)
        # Swap only the English checkpoint; the router keeps routing non-English to multilingual.
        ft_router = laya.Router(models={"english": (str(path), None)}, max_loaded=2)
        ft_router.preload(["english", "multilingual"])

    report: dict = {"llm_arms": refs, "finetuned": args.finetuned or None, "routing": {}, "calls": {}}

    # ---- decision level ------------------------------------------------------
    # The cascade is measured for determinism too. It is deterministic by construction - no
    # sampling anywhere - but "deterministic" is a claim we make in the README, and an unmeasured
    # claim is exactly the kind this project keeps catching itself making.
    print("\nrunning cascade arm ...", flush=True)
    laya_runs = [
        H.run_laya_routing(routing, router) for _ in range(max(1, args.determinism))
    ]
    report["routing"]["laya"] = H.score_routing(routing, laya_runs[0])
    report["routing"]["laya"]["determinism"] = H.agreement(laya_runs)

    ft_routing = None
    if ft_router is not None:
        print("running fine-tuned cascade arm ...", flush=True)
        ft_runs = [
            H.run_laya_routing(routing, ft_router) for _ in range(max(1, args.determinism))
        ]
        ft_routing = ft_runs[0]
        report["routing"]["cascade-ft"] = H.score_routing(routing, ft_routing)
        report["routing"]["cascade-ft"]["determinism"] = H.agreement(ft_runs)
        ft_router_arm = "cascade-ft"
    else:
        ft_router_arm = None

    llm_routing: dict[str, list] = {}
    for ref in refs:
        print(f"running {ref} ({args.concurrency} workers, {args.determinism} repeats) ...", flush=True)
        runs = [
            H.run_llm_routing(routing, model_ref=ref, concurrency=args.concurrency)
            for _ in range(max(1, args.determinism))
        ]
        llm_routing[ref] = runs[0]
        score = H.score_routing(routing, runs[0], model_ref=ref)
        score["determinism"] = H.agreement(runs)
        report["routing"][short(ref)] = score

    arms = ["laya"] + (["cascade-ft"] if ft_routing is not None else []) + [short(r) for r in refs]
    rows = []
    for label, key in [
        ("destination accuracy", "destination_accuracy"),
        ("   ±95%", "_dest_ci"),
        ("sub-queue accuracy", "subqueue_accuracy"),
        ("   ±95%", "_sub_ci"),
        ("joint accuracy", "joint_accuracy"),
        ("queue accuracy", "queue_accuracy"),
        ("   ±95%", "_queue_ci"),
        ("invalid labels", "invalid_labels"),
        ("other rate (sub-q)", "_other_rate"),
        ("misses -> other", "_other_miss"),
        ("p50 latency (ms)", "_p50"),
        ("p95 latency (ms)", "_p95"),
        # `cost_usd` is the arm's TOTAL across every case, not a per-case figure. Labelled "cost per
        # case" it overstated the LLM arms by 81x, on screen, in the evidence table.
        ("cost (all cases)", "cost_usd"),
        ("determinism", "determinism"),
    ]:
        row = [label]
        for arm in arms:
            s = report["routing"][arm]
            other = s.get("other") or {}
            if key == "cost_usd":
                row.append(H.money(s.get(key)) if s.get("cost_usd") else "$0")
            elif key == "_p50":
                row.append(f"{s['latency_ms']['p50']}")
            elif key == "_p95":
                row.append(f"{s['latency_ms']['p95']}")
            elif key == "_dest_ci":
                row.append(f"±{(s.get('destination_ci95') or {}).get('half_width', 0):.3f}")
            elif key == "_sub_ci":
                row.append(f"±{(s.get('subqueue_ci95') or {}).get('half_width', 0):.3f}")
            elif key == "_queue_ci":
                row.append(f"±{(s.get('queue_ci95') or {}).get('half_width', 0):.3f}")
            elif key == "_other_rate":
                rate = other.get("rate_of_subqueue_predictions")
                row.append(f"{rate:.3f}" if rate is not None else "n/a")
            elif key == "_other_miss":
                share = other.get("share_of_misses")
                row.append(f"{share:.0%}" if share is not None else "n/a")
            elif key == "determinism":
                row.append(f"{s.get('determinism', 1.0):.2f}")
            elif key in ("destination_accuracy", "subqueue_accuracy", "joint_accuracy", "queue_accuracy"):
                row.append(f"{s[key]:.3f}")
            else:
                row.append(str(s.get(key, "")))
        rows.append(row)
    print()
    print(H.render(f"DECISION LEVEL  ({len(routing)} cases: destination + sub-queue)", rows, ["metric", *arms]))
    print("\n±95% is the Wilson half-width. On 81 cases one case is 1.23 points, so two arms")
    print("whose intervals overlap are not distinguishable at this sample size.")

    for arm_name in [a for a in arms if a in report["routing"]]:
        gate = report["routing"][arm_name].get("gate") or {}
        if not gate:
            continue
        print(
            f"\nGATE QUALITY ({arm_name}) — escalate when destination confidence < {gate.get('threshold')}\n"
            f"  errors {gate.get('errors')} · flagged {gate.get('flagged')} ({gate.get('flag_rate'):.1%})"
            f" · caught {gate.get('errors_caught')} (recall {gate.get('recall')})\n"
            f"  accuracy when confident {gate.get('accuracy_when_confident')}"
            f" · when flagged {gate.get('accuracy_when_flagged')}"
        )

    # ---- is confidence worth anything? --------------------------------------
    # The escalation gate is only as good as this curve. A flat curve means no threshold works.
    for arm_name in [a for a in arms if a in report["routing"]]:
        cal = report["routing"][arm_name].get("calibration") or {}
        if not cal:
            continue
        print(f"\nCONFIDENCE CALIBRATION ({arm_name}) — accuracy within each confidence band")
        for band, row in cal.items():
            print(f"  conf {band}   n={row['n']:>3}  accuracy {row['accuracy']:.3f}")

    # ---- where the errors actually go ---------------------------------------
    # Reading an accuracy number tells you how often we are wrong; it does not tell you whether the
    # errors are adjacent (tires -> roadside, a defensible near-miss) or random. The distinction
    # decides whether the remaining gap is a model problem or a label-boundary problem.
    primary = "cascade-ft" if ft_routing is not None else "laya"
    pscore = report["routing"].get(primary) or {}
    for key, title, width in (
        ("destination_confusion", "DESTINATION", 12),
        ("subqueue_confusion", "SUB-QUEUE", 24),
        ("queue_confusion", "QUEUE", 24),
    ):
        conf = pscore.get(key) or {}
        lines = []
        for expected, gots in conf.items():
            wrong = [(g, k) for g, k in gots.items() if g != expected]
            if not wrong:
                continue
            total = sum(gots.values())
            parts = ", ".join(f"{g} x{k}" for g, k in wrong)
            lines.append(f"  {expected:{width}s} {parts}   ({sum(k for _, k in wrong)}/{total} wrong)")
        if lines:
            print(f"\n{title} CONFUSION ({primary}) — expected -> got, only classes with errors")
            print("\n".join(lines))

    # ---- hybrid frontier, per llm arm ---------------------------------------
    # Escalation is simulated on whichever cascade we would actually ship.
    base_for_hybrid = ft_routing if ft_routing is not None else laya_routing
    base_label = "cascade-ft" if ft_routing is not None else "laya"
    for ref in refs:
        frontier = [
            H.simulate_hybrid(routing, base_for_hybrid, llm_routing[ref], t, model_ref=ref)
            for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
        ]
        report.setdefault("routing", {})[f"hybrid_frontier:{short(ref)}"] = frontier
        frows = [
            [
                f"{f['threshold']:.2f}",
                f"{f['accuracy']:.3f}",
                f"{f['flag_rate']:.1%}",
                f"{f['recall']:.2f}" if f["recall"] is not None else "n/a",
                f"${f['cost_per_case']:.6f}",
            ]
            for f in frontier
        ]
        print()
        print(
            H.render(
                f"HYBRID FRONTIER → {short(ref)}  (escalating from {base_label})",
                frows,
                ["threshold", "accuracy", "% to llm", "error recall", "cost/case"],
            )
        )

    # ---- call level ----------------------------------------------------------
    print("\nrunning call arms ...", flush=True)
    report["calls"]["laya-full"] = H.score_calls(
        calls, [H.run_laya_call(c, router, incremental=False, verify=False) for c in calls]
    )
    report["calls"]["laya"] = H.score_calls(calls, [H.run_laya_call(c, router) for c in calls])
    if ft_router is not None:
        report["calls"]["cascade-ft"] = H.score_calls(
            calls, [H.run_laya_call(c, ft_router) for c in calls]
        )
    if hybrid_ref:
        report["calls"][f"hybrid→{short(hybrid_ref)}"] = H.score_calls(
            calls,
            [H.run_hybrid_call(c, router, model_ref=hybrid_ref) for c in calls],
            model_ref=hybrid_ref,
        )
    for ref in refs:
        try:
            report["calls"][short(ref)] = H.score_calls(
                calls, [H.run_llm_call(c, model_ref=ref) for c in calls], model_ref=ref
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  call arm {ref} failed: {exc}")

    carms = [a for a in report["calls"]]
    crows = []
    for label, key in [
        ("queue accuracy", "queue_accuracy"),
        ("llm calls made", "llm_calls"),
        ("would-escalate (cascade)", "would_escalate"),
        ("questions asked", "questions"),
        ("p50 latency (ms)", "_p50"),
        ("cost total", "cost_usd"),
    ]:
        row = [label]
        for arm in carms:
            s = report["calls"][arm]
            if key == "cost_usd":
                row.append(H.money(s.get("cost_usd")) if s.get("cost_usd") else "$0")
            elif key == "_p50":
                row.append(f"{s['latency_ms']['p50']}")
            elif key == "queue_accuracy":
                row.append(f"{s[key]:.3f}")
            else:
                row.append(str(s.get(key, "")))
        crows.append(row)
    print()
    print(H.render(f"CALL LEVEL  ({len(calls)} calls: final queue)", crows, ["metric", *carms]))

    for arm in ["laya"] + [short(r) for r in refs]:
        misses = report["routing"][arm]["misses"]
        if misses:
            print(f"\n{arm} destination misses ({len(misses)}):")
            for m in misses[:10]:
                print(f"  {m['id']:9s} expected {str(m['expected']):11s} got {str(m.get('got'))}")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
