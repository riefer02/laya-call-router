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

  // Keep the summary tied to the report being shown. This is a description of the sample,
  // not a claim that a small difference establishes a model ranking.
  const headline = useMemo(() => {
    const arms = data?.arms ?? [];
    const ours = arms.find((a) => a.key === "cascade-ft");
    const hosted = arms.filter((a) => a.key !== "cascade-ft" && a.key !== "laya" && a.joint != null);
    const best = hosted.reduce<(typeof hosted)[number] | null>(
      (winner, arm) => !winner || (arm.joint ?? 0) > (winner.joint ?? 0) ? arm : winner, null
    );
    if (!ours?.joint || !best?.joint || !data?.n_cases) return "Compare the models on the same labelled calls below.";
    const localCorrect = Math.round(ours.joint * data.n_cases);
    const hostedCorrect = Math.round(best.joint * data.n_cases);
    const speed = ours.latency_p50 && best.latency_p50
      ? ` The local routing pass took ${ms(ours.latency_p50)}; ${best.label} took ${ms(best.latency_p50)} per case.`
      : "";
    return `The fine-tuned model got ${localCorrect}/${data.n_cases} complete routing decisions right; ${best.label} got ${hostedCorrect}/${data.n_cases}.${speed} This small set cannot reliably rank scores this close.`;
  }, [data]);

  if (error) {
    return <div className="p-6 text-[12px] text-rose-400">Could not load the results: {error}</div>;
  }
  if (!data) {
    return <div className="p-6 text-[12px] text-slate-500">Loading results…</div>;
  }

  const arms = data.arms;

  return (
    <div className="h-full overflow-y-auto bg-slate-950 px-6 py-5 text-slate-200">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h1 className="text-[15px] font-semibold text-slate-100">
            What the tests found
          </h1>
          <p className="mt-1 max-w-3xl text-[11.5px] leading-relaxed text-slate-500">
            These figures come from the versioned reports in <code className="text-slate-400">results/</code>.
            The routing test has {data.n_cases} labelled cases. One case changes the score by{" "}
            {(100 / data.n_cases).toFixed(2)} percentage points, so read small gaps with care.
          </p>
          <p className={`mt-2 max-w-3xl text-[11px] ${data.sources.matches_served_model ? "text-slate-500" : "text-amber-300"}`}>
            {data.sources.matches_served_model
              ? `Report for the loaded checkpoint: ${data.sources.routing} and ${data.sources.severity}.`
              : `Reference reports: ${data.sources.routing} and ${data.sources.severity}. The loaded model is ${data.sources.checkpoint}; these results may not match calls you run here.`}
          </p>
        </header>

        {/* ------------------------------------------------------------ headline */}
        <Section
          title={`Routing decisions — ${data.n_cases} labelled cases`}
          hint={headline}
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[12px]">
              <thead>
                <tr className="border-b border-slate-800 text-left text-[10px] uppercase tracking-wider text-slate-500">
                  <th className="py-1.5 pr-4 font-medium">model</th>
                  <th className="py-1.5 pr-4 font-medium">destination</th>
                  <th className="py-1.5 pr-4 font-medium" title="Both destination and specific queue are correct">both right</th>
                  <th className="py-1.5 pr-4 font-medium">cases</th>
                  <th className="py-1.5 pr-4 font-medium">queue</th>
                  <th className="py-1.5 pr-4 font-medium" title="Median time for one routing case">median time</th>
                  <th className="py-1.5 pr-4 font-medium">cost/case</th>
                  <th className="py-1.5 font-medium" title="Agreement across three identical runs">repeatability</th>
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
            “Destination” means the right department. “Both right” also requires the right type
            of request within that department. “Queue” is the team selected for the call. Two labels
            can lead to the same team: a store question and a non-customer call both go to Front
            Desk.
          </p>
        </Section>

        {/* ------------------------------------------------------------ calls */}
        <Section
          title={`Whole calls — ${data.calls[0]?.n ?? "—"} scripted examples`}
          hint="Each score checks the final team across a full scripted call. These calls cover all 26 specific request types. The earlier 10-call set was too small to expose several failures."
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
          title="When is the model confident?"
          hint="A useful confidence score would separate correct answers from wrong ones. This model gives high confidence to several wrong routes, so confidence alone does not catch them."
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
                        <span className="font-mono text-slate-400">confidence {band}</span>
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
            title="Safety and human handoff"
            hint="A missed stranded caller may need help; a false alarm routes a safe call to Roadside in this prototype. No truck is sent. Recall is the share of real positives caught; precision is the share of flags that were right."
          >
            {data.severity.map((row) => (
              <div key={row.arm} className="mb-4">
                <div className="mb-1.5 text-[11px] font-medium text-slate-300">{row.arm}</div>
                <div className="grid gap-2 sm:grid-cols-2">
                  {[
                    { key: "is_safe_to_drive", label: "Unsafe to drive", score: row.safe },
                    { key: "needs_human", label: "Needs a person", score: row.human },
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
                                {shipped && <span className="ml-2 text-[9px]">← used in demo</span>}
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
          title="Teams and request types"
          hint="Each card is a department. The tags show the requests it handles. A store can edit this structure in config/store_profile.json, then test its own examples before using it."
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
                      {s.label}
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
          hint="Examples were checked for valid labels, duplicates and overlap with the test set. We also set a minimum number of examples for each department so smaller departments were represented."
        >
          <div className="mb-3 flex flex-wrap gap-4 text-[11px] text-slate-400">
            <span>
              examples kept <span className="font-mono text-slate-200">{data.dataset.kept ?? "—"}</span>
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
            title="Can it handle new answer choices?"
            hint="The answer choices are supplied with each question. This check asks whether fine-tuning for this store hurt the model's ability to answer different questions."
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
          Reproduce these reports with <code>scripts/eval.py</code>, <code>scripts/eval_severity.py</code>,{" "}
          <code>scripts/generate_training.py</code> and <code>scripts/generality_test.py</code>. The
          full narrative is in <code>LEARNINGS.md</code>.
        </p>
      </div>
    </div>
  );
}
