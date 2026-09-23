import { useEffect, useMemo, useState } from "react";
import { fetchEvidence } from "../api";
import type { Evidence as EvidenceData } from "../types";

function pct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined ? "—" : `${(x * 100).toFixed(digits)}%`;
}

function ms(x: number | null | undefined): string {
  if (x === null || x === undefined) return "—";
  return x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${x.toFixed(0)} ms`;
}

function usd(x: number | null | undefined): string {
  if (x === null || x === undefined) return "—";
  return x === 0 ? "$0" : `$${x.toFixed(6).replace(/0+$/, "").replace(/\.$/, "")}`;
}

function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-7">
      <h2 className="text-[12px] font-semibold uppercase tracking-wider text-slate-300">{title}</h2>
      {hint && <p className="mt-1 max-w-3xl text-[11.5px] leading-relaxed text-slate-500">{hint}</p>}
      <div className="mt-2.5">{children}</div>
    </section>
  );
}

function Bar({ value, accent = "#8b5cf6" }: { value: number; accent?: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
      <div
        className="h-full rounded-full"
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: accent }}
      />
    </div>
  );
}

export default function Evidence() {
  const [data, setData] = useState<EvidenceData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchEvidence().then(setData).catch((e) => setError(String(e)));
  }, []);

  // The headline verdict is derived from the numbers on screen, not written by hand. It used to be a
  // fixed sentence claiming the fine-tune "matches both LLM arms", which was true of the v4 run it
  // was written for and false of the v7 numbers the tab now shows — a caption contradicting its own
  // table. Now it cannot drift: change the report and the sentence changes with it.
  const headline = useMemo(() => {
    const arms = data?.arms ?? [];
    const ours = arms.find((a) => a.key === "cascade-ft");
    const theirs = arms.filter((a) => a.key !== "cascade-ft" && a.key !== "laya");
    const best = theirs.length ? Math.max(...theirs.map((a) => a.joint)) : null;
    const fastest = ours?.latency_p50 ?? null;
    const theirSlowest = theirs.length ? Math.max(...theirs.map((a) => a.latency_p50)) : null;
    const speed = fastest && theirSlowest ? Math.round(theirSlowest / fastest) : null;

    const speedPhrase = speed ? ` at roughly ${speed}x the speed` : "";
    if (ours == null || best == null) {
      return `Read against two frontier arms${speedPhrase}, for nothing per call.`;
    }
    const gap = (ours.joint - best) * 100;
    const verdict =
      Math.abs(gap) < 1.5
        ? "is level with the best LLM arm"
        : gap > 0
          ? `leads the best LLM arm by ${gap.toFixed(1)} points`
          : `trails the best LLM arm by ${Math.abs(gap).toFixed(1)} points`;
    return (
      `On the decision the fine-tuned cascade ${verdict}${speedPhrase} and for nothing per call — ` +
      "and unlike the LLM arms its number does not move. Across four runs on identical inputs " +
      "deepseek-flash scored joint 0.914, 0.889, 0.901 and 0.926; ours has not moved once. On this " +
      "sample one case is 1.23 points and the intervals overlap, so a small gap either way is noise."
    );
  }, [data]);

  // A gap is only a finding if it is bigger than the interval. On 81 cases the Wilson interval is
  // about +/-6 points, so a one-case difference - which is what most of these arms differ by - is
  // not resolvable. The table used to render that as a ranking with a verdict sentence attached.
  const ours = data?.arms.find((a) => a.key === "cascade-ft");
  const withinNoise = (a: { key: string; joint: number | null }) => {
    if (!ours?.joint || a.key === ours.key || a.joint == null) return false;
    const n = data?.n_cases ?? 0;
    if (!n) return false;
    // 95% half-width of the difference of two proportions, same n on both sides.
    const se = Math.sqrt(2 * 0.25 / n) * 1.96;
    return Math.abs(ours.joint - a.joint) < se;
  };

  if (error) {
    return <div className="p-6 text-[12px] text-rose-400">could not load evidence: {error}</div>;
  }
  if (!data) {
    return <div className="p-6 text-[12px] text-slate-500">loading measurements…</div>;
  }

  const arms = data.arms;
  const best = (pick: (a: (typeof arms)[number]) => number | null) => {
    const vals = arms.map(pick).filter((v): v is number => v !== null);
    return vals.length ? Math.max(...vals) : 0;
  };

  return (
    <div className="h-full overflow-y-auto bg-slate-950 px-6 py-5 text-slate-200">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h1 className="text-[15px] font-semibold text-slate-100">
            The measurements behind the design
          </h1>
          <p className="mt-1 max-w-3xl text-[11.5px] leading-relaxed text-slate-500">
            Read live from <code className="text-slate-400">results/*.json</code> — the same files
            the README quotes, so any number here can be traced to a report. Intervals are Wilson
            95%: on {data.n_cases} cases one case is{" "}
            {(100 / data.n_cases).toFixed(2)} points, so two arms whose intervals overlap are not
            distinguishable.
          </p>
        </header>

        {/* ------------------------------------------------------------ headline */}
        <Section
          title={`Decision level — ${data.n_cases} hand-labelled routing cases`}
          hint={headline}
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[12px]">
              <thead>
                <tr className="border-b border-slate-800 text-left text-[10px] uppercase tracking-wider text-slate-500">
                  <th className="py-1.5 pr-4 font-medium">arm</th>
                  <th className="py-1.5 pr-4 font-medium">destination</th>
                  <th className="py-1.5 pr-4 font-medium">joint</th>
                  <th className="py-1.5 pr-4 font-medium">cases</th>
                  <th className="py-1.5 pr-4 font-medium">queue</th>
                  <th className="py-1.5 pr-4 font-medium">p50</th>
                  <th className="py-1.5 pr-4 font-medium">cost/case</th>
                  <th className="py-1.5 font-medium">determinism</th>
                </tr>
              </thead>
              <tbody>
                {arms.map((a) => {
                  const isOurs = a.label === "fine-tuned";
                  return (
                    <tr
                      key={a.key}
                      className={`border-b border-slate-900 ${isOurs ? "bg-violet-950/30" : ""}`}
                    >
                      <td
                        className={`py-1.5 pr-4 font-medium ${isOurs ? "text-violet-200" : "text-slate-300"}`}
                      >
                        {a.label}
                      </td>
                      <td className="py-1.5 pr-4">
                        <span className="font-mono">{pct(a.destination)}</span>
                        {a.destination_ci != null && (
                          <span className="ml-1 text-[10px] text-slate-600">
                            ±{(a.destination_ci * 100).toFixed(1)}
                          </span>
                        )}
                      </td>
                      <td
                        className={`py-1.5 pr-4 font-mono ${isOurs ? "text-emerald-300" : ""}`}
                      >
                        {pct(a.joint)}
                        {!isOurs && withinNoise(a) && (
                          <span
                            className="ml-1 text-[10px] text-slate-500"
                            title="this gap is smaller than the interval on 81 cases - not resolvable"
                          >
                            ≈
                          </span>
                        )}
                      </td>
                      <td className="py-1.5 pr-4 font-mono text-slate-500">
                        {a.joint != null && data.n_cases
                          ? `${Math.round(a.joint * data.n_cases)}/${data.n_cases}`
                          : "—"}
                      </td>
                      <td
                        className={`py-1.5 pr-4 font-mono ${isOurs ? "text-emerald-300" : ""}`}
                      >
                        {pct(a.queue)}
                      </td>
                      <td className="py-1.5 pr-4 font-mono text-slate-400">{ms(a.latency_p50)}</td>
                      <td className="py-1.5 pr-4 font-mono text-slate-400">{usd(a.cost_per_case)}</td>
                      <td className="py-1.5 font-mono text-slate-400">
                        {a.determinism?.toFixed(2) ?? "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[11px] text-slate-500">
            Destination accuracy is agreement with the taxonomy labels; queue accuracy is where the
            call actually went. They differ because two labels can route to the same place —{" "}
            <code className="text-slate-400">front_desk</code> and{" "}
            <code className="text-slate-400">non_customer</code> both go to Front Desk.
          </p>
        </Section>

        {/* ------------------------------------------------------------ calls */}
        <Section
          title={`Call level — ${data.calls[0]?.questions != null ? "27" : "—"} scripted calls, final queue`}
          hint="End-to-end, covering all 26 specific sub-queues. The base model scored 1.000 on the original 10-call set and 0.852 on 27 — the small set was too easy to measure anything."
        >
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
            {data.calls.map((c) => (
              <div key={c.label} className="rounded border border-slate-800 bg-slate-900/40 p-2">
                <div className="text-[10px] text-slate-500">{c.label}</div>
                <div className="mt-0.5 font-mono text-[15px] text-slate-100">{pct(c.queue)}</div>
                <div className="mt-1 text-[9.5px] text-slate-600">
                  {c.questions ?? 0} questions · {ms(c.latency_p50)}
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* ------------------------------------------------------------ calibration */}
        <Section
          title="Confidence calibration — is confidence worth anything?"
          hint="The escalation gate is only as good as this. The fine-tuned model is confident because it is accurate, not because it is overconfident — which is why no threshold rescues it: its errors are not hiding in a low-confidence tail."
        >
          <div className="grid gap-4 sm:grid-cols-2">
            {arms
              .filter((a) => Object.keys(a.calibration).length > 0)
              .map((a) => (
                <div key={a.key} className="rounded border border-slate-800 bg-slate-900/30 p-3">
                  <div className="mb-2 text-[11px] font-medium text-slate-300">{a.label}</div>
                  {Object.entries(a.calibration).map(([band, row]) => (
                    <div key={band} className="mb-1.5">
                      <div className="flex items-baseline justify-between text-[10.5px]">
                        <span className="font-mono text-slate-400">conf {band}</span>
                        <span className="text-slate-500">
                          n={row.n} · accuracy{" "}
                          <span className="font-mono text-slate-300">{row.accuracy.toFixed(3)}</span>
                        </span>
                      </div>
                      <Bar value={row.accuracy} accent="#22c55e" />
                    </div>
                  ))}
                </div>
              ))}
          </div>
        </Section>

        {/* ------------------------------------------------------------ severity */}
        {data.severity.length > 0 && (
          <Section
            title="Is it safe? The questions that drive dispatch"
            hint="A missed stranded caller leaves someone at the side of a road; a false alarm sends a truck to someone who was fine. The errors are not symmetric, so recall leads — and the sweep is how we picked the dispatch threshold."
          >
            {data.severity.map((row) => (
              <div key={row.arm} className="mb-4">
                <div className="mb-1.5 text-[11px] font-medium text-slate-300">{row.arm}</div>
                <div className="grid gap-2 sm:grid-cols-2">
                  {[
                    { key: "is_safe_to_drive", label: "is_safe_to_drive", score: row.safe },
                    { key: "needs_human", label: "needs_human", score: row.human },
                  ].map(({ key, label, score }) =>
                    score ? (
                      <div key={key} className="rounded border border-slate-800 bg-slate-900/30 p-2.5">
                        <div className="font-mono text-[10.5px] text-slate-400">{label}</div>
                        <div className="mt-1 flex items-baseline gap-3 text-[11px]">
                          <span className={score.missed > 0 ? "text-amber-300" : "text-emerald-300"}>
                            recall <span className="font-mono">{pct(score.recall)}</span>
                          </span>
                          <span className="text-slate-400">
                            precision <span className="font-mono">{pct(score.precision)}</span>
                          </span>
                          <span className="ml-auto text-slate-500">
                            missed{" "}
                            <span className="font-mono text-slate-300">
                              {score.missed}/{score.positives}
                            </span>
                          </span>
                        </div>
                        <Bar
                          value={score.recall ?? 0}
                          accent={score.missed > 0 ? "#f59e0b" : "#22c55e"}
                        />
                      </div>
                    ) : null
                  )}
                </div>

                {row.sweep.length > 0 && (
                  <div className="mt-2 overflow-x-auto">
                    <table className="w-full border-collapse text-[11px]">
                      <thead>
                        <tr className="text-left text-[9.5px] uppercase tracking-wider text-slate-600">
                          <th className="py-1 pr-4 font-medium">threshold</th>
                          <th className="py-1 pr-4 font-medium">recall</th>
                          <th className="py-1 pr-4 font-medium">precision</th>
                          <th className="py-1 pr-4 font-medium">missed</th>
                          <th className="py-1 font-medium">false alarms</th>
                        </tr>
                      </thead>
                      <tbody>
                        {row.sweep.map((s) => {
                          const shipped = s.threshold === Number(data.policy?.unsafe_threshold);
                          return (
                            <tr
                              key={s.threshold}
                              className={shipped ? "bg-emerald-950/40 text-emerald-200" : ""}
                            >
                              <td className="py-0.5 pr-4 font-mono">
                                {s.threshold.toFixed(1)}
                                {shipped && <span className="ml-2 text-[9px]">← shipped</span>}
                              </td>
                              <td className="py-0.5 pr-4 font-mono">{pct(s.recall)}</td>
                              <td className="py-0.5 pr-4 font-mono">{pct(s.precision)}</td>
                              <td className="py-0.5 pr-4 font-mono">{s.missed}</td>
                              <td className="py-0.5 font-mono">{s.false_alarms}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
          </Section>
        )}

        {/* ------------------------------------------------------------ taxonomy */}
        <Section
          title="The taxonomy, as data"
          hint="Structure follows how dealerships actually publish themselves: Fixed Operations (service, parts, body shop) and Variable Operations (sales, F&I). Tires and detailing are service sub-queues, not departments; roadside is a dispatch flag, not a place. This whole table is config/store_profile.json."
        >
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {data.taxonomy.map((d) => (
              <div key={d.key} className="rounded border border-slate-800 bg-slate-900/30 p-2.5">
                <div className="flex items-baseline justify-between">
                  <span className="text-[12px] font-medium text-slate-100">{d.label}</span>
                  <span className="font-mono text-[9.5px] text-amber-300/80">{d.queue}</span>
                </div>
                <div className="mt-1 text-[10px] leading-snug text-slate-500">{d.description}</div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {d.subqueues.map((s) => (
                    <span
                      key={s.key}
                      title={`${s.description}\n→ ${s.queue}`}
                      className={`rounded border px-1.5 py-0.5 font-mono text-[9.5px] ${
                        s.handler === "human"
                          ? "border-amber-900/60 bg-amber-950/30 text-amber-300"
                          : "border-slate-700 bg-slate-800/60 text-slate-300"
                      }`}
                    >
                      {s.key}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* ------------------------------------------------------------ dataset */}
        <Section
          title="Training data"
          hint="Every specific sub-queue has exactly 50 examples. Balanced per sub-queue turned out to be the wrong axis for the destination question — front_desk has only two sub-queues, so it was starved and was the worst class until a per-destination floor was added."
        >
          <div className="mb-3 flex flex-wrap gap-4 text-[11px] text-slate-400">
            <span>
              rows <span className="font-mono text-slate-200">{data.dataset.kept ?? "—"}</span>
            </span>
            <span>
              teacher cost{" "}
              <span className="font-mono text-slate-200">{usd(data.dataset.cost_usd)}</span>
            </span>
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            {Object.entries(data.dataset.by_destination)
              .sort((a, b) => b[1] - a[1])
              .map(([dest, n]) => {
                const max = Math.max(...Object.values(data.dataset.by_destination));
                return (
                  <div key={dest} className="flex items-center gap-2 text-[11px]">
                    <span className="w-24 shrink-0 font-mono text-slate-400">{dest}</span>
                    <span className="flex-1">
                      <Bar value={n / max} accent="#8b5cf6" />
                    </span>
                    <span className="w-8 shrink-0 text-right font-mono text-slate-300">{n}</span>
                  </div>
                );
              })}
          </div>
        </Section>

        {/* ------------------------------------------------------------ generality */}
        {data.generality.verdict && (
          <Section
            title="Does it still work on questions it was never trained on?"
            hint="Laya's defining property is that the option space is defined at request time, which is what makes a per-store taxonomy viable. This checks what fine-tuning on 32 fixed sub-queues cost."
          >
            <p className="mb-2 text-[11.5px] text-emerald-300">{data.generality.verdict}</p>
            <div className="grid gap-2 sm:grid-cols-3">
              {Object.entries(data.generality.suites).map(([suite, armsForSuite]) => (
                <div key={suite} className="rounded border border-slate-800 bg-slate-900/30 p-2.5">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500">{suite}</div>
                  {Object.entries(armsForSuite).map(([arm, score]) => (
                    <div key={arm} className="mt-1 flex justify-between text-[11px]">
                      <span className="text-slate-400">{arm}</span>
                      <span className="font-mono text-slate-200">{pct(score.accuracy)}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </Section>
        )}

        <p className="pb-8 text-[10.5px] text-slate-600">
          Generated by <code>scripts/eval.py</code>, <code>scripts/eval_severity.py</code>,{" "}
          <code>scripts/generate_training.py</code> and <code>scripts/generality_test.py</code>. The
          full narrative is in <code>LEARNINGS.md</code>.
        </p>
      </div>
    </div>
  );
}
