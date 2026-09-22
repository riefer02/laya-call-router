import { useEffect, useState } from "react";
import { ReactFlowProvider, useReactFlow } from "@xyflow/react";
import Controls from "./controls/Controls";
import Rail from "./conversation/Rail";
import GraphCanvas from "./graph/GraphCanvas";
import Hud from "./hud/Hud";
import Inspector from "./inspector/Inspector";
import { useRun } from "./store/run";

function Shell() {
  const [railCollapsed, setRailCollapsed] = useState(false);
  const { fitView } = useReactFlow();

  const playing = useRun((s) => s.playing);
  const speed = useRun((s) => s.speed);
  const applied = useRun((s) => s.applied);
  const total = useRun((s) => s.events.length);
  const step = useRun((s) => s.step);

  // Playback clock: reveal one event at a time at the chosen speed. Keeping the reveal in the
  // client means play/pause/step/scrub all work without touching the backend.
  useEffect(() => {
    if (!playing || applied >= total) return;
    const id = window.setTimeout(() => step(), Math.max(30, 1000 / speed));
    return () => window.clearTimeout(id);
  }, [playing, speed, applied, total, step]);

  // Keyboard: space toggles playback, → steps, F frames the whole call.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA"))
        return;
      if (e.code === "Space") {
        e.preventDefault();
        useRun.getState().toggle();
      } else if (e.key === "ArrowRight") {
        useRun.getState().step();
      } else if (e.key === "f" || e.key === "F") {
        fitView({ padding: 0.12, minZoom: 0.22, maxZoom: 1, duration: 400 });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fitView]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-950 text-slate-100">
      <Hud />
      <Controls />
      <div className="flex min-h-0 flex-1">
        <Rail collapsed={railCollapsed} onToggle={() => setRailCollapsed((v) => !v)} />
        <div className="min-w-0 flex-1">
          <GraphCanvas />
        </div>
        <Inspector />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <Shell />
    </ReactFlowProvider>
  );
}
