import { useState } from "react";
import { useRun } from "../store/run";
import { confColor, styleFor } from "../theme";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2 py-1">
      <span className="w-24 shrink-0 text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </span>
      <span className="min-w-0 flex-1 text-[11px] text-slate-200">{children}</span>
    </div>
  );
}

export default function Inspector() {
  const selected = useRun((s) => s.selected);
  const node = useRun((s) => (s.selected ? s.nodes[s.selected] : undefined));
  const select = useRun((s) => s.select);
  const [raw, setRaw] = useState(false);

  if (!selected || !node) {
    return (
      <aside className="flex w-96 shrink-0 flex-col border-l border-slate-800 bg-slate-950/60">
        <div className="border-b border-slate-800 px-3 py-2 text-[10px] uppercase tracking-widest text-slate-500">
          Inspector
        </div>
        <p className="m-auto px-6 text-center text-[11px] text-slate-600">
          Click any box in the graph to see exactly how that decision was made.
        </p>
      </aside>
    );
  }

  const style = styleFor(node.kind, node.key);
  const r = node.result;
  const summary = r?.summary ?? {};
  const ranked = summary.probabilities
    ? Object.entries(summary.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <aside className="flex w-96 shrink-0 flex-col border-l border-slate-800 bg-slate-950/60">
      <div className="flex items-center gap-2 border-b border-slate-800 px-3 py-2">
        <span
          className="rounded px-1.5 py-0.5 text-[9px] font-mono uppercase"
          style={{ background: style.tint, color: style.accent }}
        >
          {node.kind}
        </span>
        <span className="text-[12px] font-semibold text-slate-100">{node.title}</span>
        <span className="font-mono text-[9px] text-slate-500">
          t{node.turn} · {node.id}
        </span>
        <button
          onClick={() => select(null)}
          className="ml-auto text-[11px] text-slate-500 hover:text-white"
        >
          ✕
        </button>
      </div>

      <div className="scroll-thin flex-1 overflow-y-auto px-3 py-3">
        {r?.question && (
          <p className="mb-3 rounded border border-slate-800 bg-slate-900/60 p-2 text-[11px] italic text-slate-300">
            “{r.question}”
          </p>
        )}

        {node.kind === "utterance" && (
          <p className="rounded border border-slate-800 bg-slate-900/60 p-2.5 text-[11.5px] leading-relaxed text-slate-200">
            {String(r?.value ?? "")}
          </p>
        )}

        {node.kind === "terminal" && r?.routing && (
          <div className="space-y-1">
            <div className="text-[16px] font-semibold text-amber-300">{r.routing.queue}</div>
            <Row label="priority">{r.routing.priority}</Row>
            <Row label="handler">{r.routing.handler}</Row>
            <Row label="flags">
              {r.routing.flags.length ? (
                <span className="flex flex-wrap gap-1">
                  {r.routing.flags.map((f) => (
                    <span
                      key={f}
                      className="rounded-full border border-amber-500/40 px-2 py-px font-mono text-[9px] text-amber-300"
                    >
                      {f}
                    </span>
                  ))}
                </span>
              ) : (
                "none"
              )}
            </Row>
            <Row label="why">
              <ul className="list-disc pl-4 text-slate-400">
                {r.routing.reasons.map((x) => (
                  <li key={x}>{x}</li>
                ))}
              </ul>
            </Row>
          </div>
        )}

        {node.kind === "policy" && (
          <>
            <Row label="decision">
              <span className="font-semibold text-slate-100">{String(r?.value)}</span>
            </Row>
            {r?.reason && <Row label="reason">{r.reason}</Row>}
            {r?.inputs && (
              <Row label="inputs">
                <code className="font-mono text-[10px] text-slate-400">
                  {JSON.stringify(r.inputs)}
                </code>
              </Row>
            )}
            {r?.missing && r.missing.length > 0 && (
              <Row label="missing">
                <span className="font-mono text-[10px] text-amber-300">
                  {r.missing.join(", ")}
                </span>
              </Row>
            )}
          </>
        )}

        {node.kind === "extract" && (
          <Row label="extracted">
            <span className="font-mono text-slate-100">{String(r?.value)}</span>
          </Row>
        )}

        {node.kind === "decision" && (
          <>
            {summary.primitive === "choice" && (
              <>
                <Row label="chosen">
                  <span className="font-semibold text-slate-100">{summary.choice}</span>
                </Row>
                <div className="mt-2 space-y-1.5">
                  {ranked.map(([label, p]) => (
                    <div key={label} className="flex items-center gap-2">
                      <span
                        className={`w-28 shrink-0 truncate text-[10px] ${
                          label === summary.choice ? "text-slate-100" : "text-slate-500"
                        }`}
                      >
                        {label}
                      </span>
                      <span className="h-2 flex-1 overflow-hidden rounded-sm bg-slate-800">
                        <span
                          className="block h-full rounded-sm transition-[width] duration-300"
                          style={{
                            width: `${Math.max(1, Math.round(p * 100))}%`,
                            background: label === summary.choice ? style.accent : "#475569",
                          }}
                        />
                      </span>
                      <span className="w-9 shrink-0 text-right font-mono text-[9.5px] text-slate-400">
                        {p.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
            {summary.primitive === "noul" && (
              <>
                <Row label="P(true)">
                  <span className="font-mono text-slate-100">{summary.noul?.toFixed(3)}</span>
                </Row>
                <Row label="P(false)">
                  <span className="font-mono text-slate-400">
                    {(1 - (summary.noul ?? 0)).toFixed(3)}
                  </span>
                </Row>
              </>
            )}
          </>
        )}

        {(() => {
          const v = (r as unknown as { verification?: any })?.verification;
          if (!v) return null;
          return (
            <div className="mt-3 rounded border border-slate-800 bg-slate-900/50 p-2">
              <div className="mb-1 font-mono text-[9px] uppercase tracking-wider text-slate-500">
                tier 2 · verification
              </div>
              <Row label="primary">
                {v.primary.choice}{" "}
                <span className="text-slate-500">p {v.primary.top_probability?.toFixed(2)}</span>
              </Row>
              <Row label="2nd phrasing">
                {v.paraphrase.choice}{" "}
                <span className="text-slate-500">p {v.paraphrase.top_probability?.toFixed(2)}</span>
              </Row>
              <Row label="outcome">
                {v.agrees ? (
                  <span className="text-emerald-400">agrees — settled with confidence</span>
                ) : (
                  <span className="text-rose-400">
                    differs — kept the first answer, flagged for LLM/human
                  </span>
                )}
              </Row>
            </div>
          );
        })()}

        {r && (
          <div className="mt-4 border-t border-slate-800 pt-2">
            {summary.confidence !== undefined && summary.confidence !== null && (
              <Row label="confidence">
                <span style={{ color: confColor(summary.confidence) }}>
                  {summary.confidence.toFixed(3)}
                </span>{" "}
                <span className="text-[9px] text-slate-600">
                  (entropy-normalised, not top-p)
                </span>
              </Row>
            )}
            {summary.top_probability !== undefined && (
              <Row label="top prob">
                <span className="font-mono">{summary.top_probability.toFixed(3)}</span>
              </Row>
            )}
            <Row label="checkpoint">{r.model ?? "— (rule, no model)"}</Row>
            {r.routing_reason && <Row label="routed by">{r.routing_reason}</Row>}
            <Row label="latency">
              <span className="font-mono">{r.latency_ms.toFixed(1)} ms</span>
              {r.batch_size > 1 && (
                <span className="text-slate-500">
                  {" "}
                  · batched with {r.batch_size - 1} other questions in one forward pass
                </span>
              )}
            </Row>
            <Row label="status">{r.status}</Row>
            {r.note && <Row label="note">{r.note}</Row>}
            <Row label="tokens out">
              <span className="font-mono text-emerald-400">0</span>
            </Row>
          </div>
        )}

        <button
          onClick={() => setRaw((v) => !v)}
          className="mt-3 text-[10px] text-slate-500 underline hover:text-slate-300"
        >
          {raw ? "hide" : "show"} raw JSON
        </button>
        {raw && (
          <pre className="scroll-thin mt-2 max-h-72 overflow-auto rounded border border-slate-800 bg-black/50 p-2 font-mono text-[9.5px] leading-relaxed text-slate-400">
            {JSON.stringify(r ?? node, null, 2)}
          </pre>
        )}
      </div>
    </aside>
  );
}
