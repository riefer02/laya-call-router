import { useEffect, useState } from "react";
import { useReactFlow } from "@xyflow/react";
import { fetchRun, fetchRuns, fetchScenarios, runScenario } from "../api";
import { useRun } from "../store/run";
import type { RunListItem, Scenario } from "../types";

function Btn({
  children,
  onClick,
  disabled,
  active,
  title,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  active?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors disabled:opacity-35 ${
        active
          ? "border-violet-500/70 bg-violet-500/20 text-violet-200"
          : "border-slate-700 bg-slate-800/70 text-slate-300 hover:border-slate-600 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}

export default function Controls() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const playing = useRun((s) => s.playing);
  const applied = useRun((s) => s.applied);
  const total = useRun((s) => s.events.length);
  const speed = useRun((s) => s.speed);
  const load = useRun((s) => s.load);
  const step = useRun((s) => s.step);
  const applyAll = useRun((s) => s.applyAll);
  const toggle = useRun((s) => s.toggle);
  const setSpeed = useRun((s) => s.setSpeed);
  const follow = useRun((s) => s.follow);
  const setFollow = useRun((s) => s.setFollow);
  const reset = useRun((s) => s.reset);
  const { fitView } = useReactFlow();

  useEffect(() => {
    fetchScenarios()
      .then((s) => {
        setScenarios(s);
        if (s.length) setScenarioId(s[0].id);
      })
      .catch(() => setError("backend not reachable — start uvicorn on :8765"));
    fetchRuns().then(setRuns).catch(() => {});
  }, []);

  async function run() {
    if (!scenarioId) return;
    setBusy(true);
    setError(null);
    try {
      const payload = await runScenario(scenarioId);
      load(payload.events, { label: scenarioId });
      fetchRuns().then(setRuns).catch(() => {});
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function replay(id: string) {
    setBusy(true);
    try {
      const payload = await fetchRun(id);
      load(payload.events, { runId: id, label: id });
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const done = total > 0 && applied >= total;

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 bg-slate-950/80 px-3 py-2">
      <select
        value={scenarioId}
        onChange={(e) => setScenarioId(e.target.value)}
        className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-[11px] text-slate-200 outline-none focus:border-violet-500"
      >
        {scenarios.map((s) => (
          <option key={s.id} value={s.id}>
            {s.label} — {s.blurb}
          </option>
        ))}
      </select>

      <Btn onClick={run} disabled={busy || !scenarioId} active>
        {busy ? "running…" : "▶ Run call"}
      </Btn>

      <div className="mx-1 h-5 w-px bg-slate-800" />

      <Btn onClick={toggle} disabled={total === 0 || done} title="play / pause">
        {playing ? "⏸ Pause" : "▶ Play"}
      </Btn>
      <Btn onClick={step} disabled={total === 0 || done} title="reveal the next event">
        ⏭ Step
      </Btn>
      <Btn onClick={applyAll} disabled={total === 0 || done} title="reveal everything">
        ⏩ Skip
      </Btn>
      <Btn onClick={reset} disabled={total === 0}>
        ↺ Reset
      </Btn>

      <label className="ml-1 flex items-center gap-2 text-[10px] text-slate-400">
        speed
        <input
          type="range"
          min={1}
          max={20}
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          className="w-24 accent-violet-500"
        />
        <span className="w-6 font-mono">{speed}×</span>
      </label>

      <Btn
        onClick={() => fitView({ padding: 0.12, minZoom: 0.22, maxZoom: 1, duration: 400 })}
        title="frame the whole call (F)"
      >
        ⤢ Fit
      </Btn>
      <label className="flex items-center gap-1.5 text-[10px] text-slate-400" title="camera follows the active decision">
        <input
          type="checkbox"
          checked={follow}
          onChange={(e) => setFollow(e.target.checked)}
          className="accent-violet-500"
        />
        follow
      </label>

      <div className="ml-auto flex items-center gap-2">
        <span className="font-mono text-[10px] text-slate-500">
          {applied}/{total || 0} events
        </span>
        <select
          onChange={(e) => e.target.value && replay(e.target.value)}
          value=""
          className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-[11px] text-slate-300 outline-none"
        >
          <option value="">↺ replay recording…</option>
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.scenario?.label ?? r.id} · {new Date(r.modified * 1000).toLocaleTimeString()}
            </option>
          ))}
        </select>
      </div>

      {error && <span className="text-[11px] text-rose-400">{error}</span>}
    </div>
  );
}
