import type { NodeProps } from "@xyflow/react";

export default function LaneNode({ data }: NodeProps) {
  const { turn, speaker } = data as { turn: number; speaker?: string };
  return (
    <div className="relative h-full w-full rounded-xl border border-slate-800 bg-slate-900/40">
      <div className="absolute -top-px left-3 flex items-center gap-2 rounded-b-md border border-t-0 border-slate-800 bg-slate-900 px-2 py-0.5">
        <span className="font-mono text-[10px] uppercase tracking-widest text-slate-400">
          turn {turn}
        </span>
        {speaker && <span className="text-[9px] text-slate-500">{speaker}</span>}
      </div>
    </div>
  );
}
