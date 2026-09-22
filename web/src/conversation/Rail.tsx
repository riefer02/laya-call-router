import { useMemo } from "react";
import { useRun } from "../store/run";
import type { AppliedNode } from "../store/run";

export default function Rail({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const nodes = useRun((s) => s.nodes);
  const select = useRun((s) => s.select);
  const selected = useRun((s) => s.selected);

  const turns = useMemo(() => {
    const byTurn = new Map<number, AppliedNode[]>();
    for (const n of Object.values(nodes)) {
      if (n.kind !== "utterance") continue;
      if (!byTurn.has(n.turn)) byTurn.set(n.turn, []);
      byTurn.get(n.turn)!.push(n);
    }
    return [...byTurn.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([turn, list]) => ({
        turn,
        items: list.sort((a, b) => a.col - b.col),
      }));
  }, [nodes]);

  if (collapsed) {
    return (
      <div className="flex w-10 shrink-0 flex-col items-center gap-3 border-r border-slate-800 bg-slate-950/60 py-3">
        <button
          onClick={onToggle}
          title="show conversation"
          className="rounded border border-slate-700 px-1.5 py-1 text-[11px] text-slate-400 hover:text-white"
        >
          ›
        </button>
        <div className="flex flex-col items-center gap-1">
          {turns.map((t) => (
            <span
              key={t.turn}
              className="font-mono text-[9px] text-slate-600 [writing-mode:vertical-rl]"
            >
              turn {t.turn}
            </span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <aside className="flex w-80 shrink-0 flex-col border-r border-slate-800 bg-slate-950/60">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2">
        <span className="text-[10px] uppercase tracking-widest text-slate-500">
          Support conversation
        </span>
        <button
          onClick={onToggle}
          title="collapse"
          className="rounded border border-slate-700 px-1.5 py-0.5 text-[11px] text-slate-400 hover:text-white"
        >
          ‹
        </button>
      </div>
      <div className="scroll-thin flex-1 overflow-y-auto p-3">
        {turns.length === 0 && (
          <p className="mt-6 text-center text-[11px] text-slate-600">
            Run a call to see the conversation.
          </p>
        )}
        {turns.map((t) => (
          <div key={t.turn} className="mb-4">
            <div className="mb-1.5 font-mono text-[9px] uppercase tracking-widest text-slate-600">
              turn {t.turn}
            </div>
            <div className="space-y-2">
              {t.items.map((n) => {
                const isCaller = n.key === "caller";
                return (
                  <button
                    key={n.id}
                    onClick={() => select(n.id)}
                    className={`block w-full rounded-lg border px-2.5 py-1.5 text-left text-[11px] leading-snug transition-colors ${
                      selected === n.id ? "border-violet-500/70" : "border-slate-800"
                    } ${
                      isCaller
                        ? "bg-blue-500/10 text-slate-200"
                        : "bg-emerald-500/10 text-slate-200"
                    }`}
                  >
                    <span
                      className={`mb-0.5 block text-[8.5px] uppercase tracking-wider ${
                        isCaller ? "text-blue-400" : "text-emerald-400"
                      }`}
                    >
                      {isCaller ? "☎ caller" : "🎧 switchboard"}
                    </span>
                    {String(n.result?.value ?? "…")}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
