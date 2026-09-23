import { useEffect, useMemo, useState } from "react";
import { useRun } from "../store/run";
import type { AppliedNode } from "../store/run";
import { runTurns } from "../api";

export default function Rail({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const nodes = useRun((s) => s.nodes);
  const select = useRun((s) => s.select);
  const selected = useRun((s) => s.selected);
  const load = useRun((s) => s.load);
  const scenarioLabel = useRun((s) => s.scenarioLabel);

  // Live mode: the turns you type are kept here and the whole list is re-sent on each submit, so the
  // call is recomputed from the top and the graph stays consistent with the scripted path. Nothing
  // session-shaped lives on the server, which is why a live call and a recorded one look the same.
  const [draft, setDraft] = useState("");
  const [typed, setTyped] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  // A newly loaded scenario or replay is a different call. Keep the typed caller's hidden
  // turn list from leaking into the next call they try.
  useEffect(() => {
    if (scenarioLabel !== "Typed call") setTyped([]);
  }, [scenarioLabel]);

  async function send() {
    const text = draft.trim();
    if (!text || busy) return;
    const next = [...typed, text];
    setBusy(true);
    try {
      const payload = await runTurns(next, next[0].slice(0, 28));
      setTyped(next);
      setDraft("");
      load(payload.events, { label: "Typed call" });
    } finally {
      setBusy(false);
    }
  }

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
          Conversation
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
        {turns.length === 0 && typed.length === 0 && (
          <p className="mt-6 text-center text-[11px] text-slate-600">
            Run a call, or type a caller turn below.
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
      {/* Each submit replays the typed turns from the start. This is text input for the demo,
          not a voice connection or a persistent server session. */}
      <div className="border-t border-slate-800 p-2">
        <div className="flex items-center gap-1.5">
          <input
            id="live-turn"
            name="live-turn"
            aria-label="type a caller turn"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void send();
            }}
            placeholder={busy ? "Checking this turn…" : "Type what the caller says, then press Enter"}
            className="min-w-0 flex-1 rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-[11px] text-slate-200 placeholder:text-slate-600 outline-none focus:border-violet-500/60"
          />
          <button
            type="button"
            onClick={() => void send()}
            disabled={busy || !draft.trim()}
            title="route this turn"
            className="rounded-md border border-slate-700 px-2 py-1.5 text-[11px] text-slate-300 hover:text-white disabled:opacity-40"
          >
            ⏎
          </button>
        </div>
        <div className="mt-1 flex items-center justify-between px-0.5">
          <span className="font-mono text-[9px] text-slate-600">
            {typed.length ? `${typed.length} turn${typed.length === 1 ? "" : "s"} entered · full call reruns each time` : "Try your own caller lines here · text only"}
          </span>
          {typed.length > 0 && (
            <button
              type="button"
              onClick={() => {
                setTyped([]);
                setDraft("");
              }}
              className="text-[9px] text-slate-500 hover:text-slate-300"
            >
              ↺ clear
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}
