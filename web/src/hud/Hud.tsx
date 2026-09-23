import { useEffect, useState } from "react";
import { fetchHealth } from "../api";
import { useCounters, useRun } from "../store/run";
import { endingLabel } from "../copy";
import type { View } from "../App";

function Stat({ label, value, accent, hint }: { label: string; value: string; accent?: string; hint?: string }) {
  return (
    <div className="flex flex-col leading-tight" title={hint}>
      <span className="font-mono text-[13px] font-semibold" style={{ color: accent ?? "#e2e8f0" }}>
        {value}
      </span>
      <span className="text-[9px] uppercase tracking-wider text-slate-500">{label}</span>
    </div>
  );
}

function Tabs({ view, onView }: { view: View; onView: (v: View) => void }) {
  const items: Array<{ id: View; label: string; hint: string }> = [
    { id: "call", label: "Call", hint: "Watch a call from start to finish" },
    { id: "evidence", label: "Evidence", hint: "See the measured results (E)" },
  ];
  return (
    <div className="flex items-center gap-0.5 rounded border border-slate-800 bg-slate-900/60 p-0.5">
      {items.map((it) => (
        <button
          key={it.id}
          title={it.hint}
          onClick={() => onView(it.id)}
          className={`rounded px-2.5 py-1 text-[11px] transition-colors ${
            view === it.id
              ? "bg-slate-700/70 text-slate-100"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

export default function Hud({ view, onView }: { view: View; onView: (v: View) => void }) {
  const [model, setModel] = useState<"base" | "fine-tuned" | null>(null);
  useEffect(() => {
    fetchHealth().then((health) => setModel(health.model)).catch(() => {});
  }, []);
  const { decisions, skipped, turns } = useCounters();
  const summary = useRun((s) => s.summary);
  const routing = useRun((s) => s.routing);
  const label = useRun((s) => s.scenarioLabel);

  return (
    <header className="flex items-center gap-5 border-b border-slate-800 bg-slate-950 px-4 py-2">
      <div className="flex items-baseline gap-2">
        <span className="text-violet-400">◈</span>
        <span className="text-[13px] font-semibold text-slate-100">jev-classifier</span>
        <span className="text-[10px] text-slate-500">See how each call is handled</span>
      </div>

      <Tabs view={view} onView={onView} />

      {model === "base" && (
        <span className="rounded border border-amber-800 bg-amber-950/50 px-2 py-1 text-[10px] text-amber-300" title="The public repo does not include the fine-tuned weights. See README for setup.">
          Base model loaded · demo scores will differ
        </span>
      )}

      {label && view === "call" && (
        <span className="rounded border border-slate-800 bg-slate-900 px-2 py-0.5 text-[10px] text-slate-300">
          {label}
        </span>
      )}

      {view === "call" && (
        <div className="ml-auto flex items-center gap-5">
          <Stat label="model questions" value={String(decisions)} accent="#a78bfa" hint="Questions sent to the local classifier" />
          <Stat label="answers reused" value={String(skipped)} accent="#22c55e" hint="Known answers that were not recomputed" />
          <Stat
            label="second checks"
            value={summary ? String(summary.escalations ?? 0) : "—"}
            accent="#38bdf8"
          />
          <Stat
            label="flagged for review"
            value={summary ? String(summary.llm_escalations ?? 0) : "—"}
            accent={summary && (summary.llm_escalations ?? 0) > 0 ? "#f59e0b" : "#22c55e"}
            hint="Disagreements are flagged; this demo does not call another model"
          />
          <Stat label="turns" value={String(turns)} />
          <Stat label="model time" value={summary ? `${summary.compute_ms.toFixed(0)} ms` : "—"} hint="Classifier time added across all turns; it excludes playback" />
          <Stat
            label="generated tokens"
            value={String(summary?.tokens_generated ?? 0)}
            accent="#22c55e"
            hint="The classifier picks labels and never writes reply text"
          />
          {routing && (
            <div className="flex flex-col leading-tight">
              <span className="text-[13px] font-semibold text-amber-300">
                {endingLabel(summary?.completion)}
              </span>
              <span className="text-[9px] uppercase tracking-wider text-slate-500">
                {summary?.completion === "awaiting_caller"
                  ? `Likely team: ${routing.queue}`
                  : `${routing.queue} · ${routing.priority.toLowerCase()} priority`}
              </span>
            </div>
          )}
        </div>
      )}
    </header>
  );
}
