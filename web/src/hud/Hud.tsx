import { useCounters, useRun } from "../store/run";
import type { View } from "../App";

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex flex-col leading-tight">
      <span className="font-mono text-[13px] font-semibold" style={{ color: accent ?? "#e2e8f0" }}>
        {value}
      </span>
      <span className="text-[9px] uppercase tracking-wider text-slate-500">{label}</span>
    </div>
  );
}

function Tabs({ view, onView }: { view: View; onView: (v: View) => void }) {
  const items: Array<{ id: View; label: string; hint: string }> = [
    { id: "call", label: "Call", hint: "watch a call route and book" },
    { id: "evidence", label: "Evidence", hint: "the measurements behind the design (E)" },
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
  const { decisions, skipped, turns } = useCounters();
  const summary = useRun((s) => s.summary);
  const routing = useRun((s) => s.routing);
  const label = useRun((s) => s.scenarioLabel);

  return (
    <header className="flex items-center gap-5 border-b border-slate-800 bg-slate-950 px-4 py-2">
      <div className="flex items-baseline gap-2">
        <span className="text-violet-400">◈</span>
        <span className="text-[13px] font-semibold text-slate-100">jev-classifier</span>
        <span className="text-[10px] text-slate-500">call routing debugger</span>
      </div>

      <Tabs view={view} onView={onView} />

      {label && view === "call" && (
        <span className="rounded border border-slate-800 bg-slate-900 px-2 py-0.5 text-[10px] text-slate-300">
          {label}
        </span>
      )}

      {view === "call" && (
        <div className="ml-auto flex items-center gap-5">
          <Stat label="questions asked" value={String(decisions)} accent="#a78bfa" />
          <Stat label="skipped (already known)" value={String(skipped)} accent="#22c55e" />
          <Stat
            label="2nd opinions"
            value={summary ? String(summary.escalations ?? 0) : "—"}
            accent="#38bdf8"
          />
          <Stat
            label="would call an LLM"
            value={summary ? String(summary.llm_escalations ?? 0) : "—"}
            accent={summary && (summary.llm_escalations ?? 0) > 0 ? "#f59e0b" : "#22c55e"}
          />
          <Stat label="turns" value={String(turns)} />
          <Stat label="compute" value={summary ? `${summary.compute_ms.toFixed(0)} ms` : "—"} />
          <Stat
            label="tokens generated"
            value={String(summary?.tokens_generated ?? 0)}
            accent="#22c55e"
          />
          {routing && (
            <div className="flex flex-col leading-tight">
              <span className="text-[13px] font-semibold text-amber-300">{routing.queue}</span>
              <span className="text-[9px] uppercase tracking-wider text-slate-500">
                {routing.priority} · {routing.handler}
              </span>
            </div>
          )}
        </div>
      )}
    </header>
  );
}
