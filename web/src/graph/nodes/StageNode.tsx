import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { AppliedNode } from "../../store/run";
import { confColor, styleFor, STATUS_ACCENT } from "../../theme";

function Bar({ p, accent }: { p: number; accent: string }) {
  return (
    <div className="h-1.5 w-full rounded-sm bg-slate-700/50 overflow-hidden">
      <div
        className="h-full rounded-sm transition-[width] duration-300"
        style={{ width: `${Math.max(2, Math.round(p * 100))}%`, background: accent }}
      />
    </div>
  );
}

export default function StageNode({ data, selected }: NodeProps) {
  const node = data as unknown as AppliedNode;
  const result = node.result;
  const style = styleFor(node.kind, node.key);
  const statusAccent = STATUS_ACCENT[node.status] ?? STATUS_ACCENT.ok;
  const summary = result?.summary ?? {};

  const isUtterance = node.kind === "utterance";
  const isTerminal = node.kind === "terminal";
  const isSkipped = node.status === "skipped";

  // top probabilities for choice questions
  let ranked: [string, number][] = [];
  if (summary.probabilities) {
    ranked = Object.entries(summary.probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, isUtterance ? 0 : 3);
  }

  return (
    <div
      className={`node-pop rounded-lg border bg-slate-900/95 shadow-lg backdrop-blur-sm transition-shadow ${
        isSkipped ? "border-dashed opacity-55" : ""
      }`}
      style={{
        width: 190,
        borderColor: selected ? style.accent : isSkipped ? "#334155" : "#1e293b",
        boxShadow: selected ? `0 0 0 2px ${style.accent}55` : undefined,
        borderLeft: `3px solid ${isSkipped ? "#334155" : result ? statusAccent : style.accent}`,
      }}
    >
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />

      <div className="flex items-center gap-1.5 px-2.5 pt-2">
        <span
          className="text-[11px] font-semibold truncate"
          style={{ color: isSkipped ? "#64748b" : style.accent }}
        >
          {node.title}
        </span>
        <span
          className="ml-auto shrink-0 rounded px-1 py-px text-[8.5px] font-mono uppercase"
          style={{
            background: isSkipped ? "rgba(51,65,85,0.5)" : style.tint,
            color: isSkipped ? "#64748b" : style.accent,
          }}
        >
          {isSkipped ? "skipped" : isUtterance ? style.label : node.primitive || style.label}
        </span>
      </div>

      <div className="px-2.5 pb-2 pt-1">
        {isSkipped ? (
          <div>
            <div className="text-[10.5px] text-slate-500">already known</div>
            <div className="mt-0.5 text-[10.5px] font-medium text-slate-400">
              {summary.primitive === "choice"
                ? String(summary.choice)
                : summary.primitive === "noul"
                  ? `P(true) ${(summary.noul ?? 0).toFixed(2)}`
                  : "—"}
            </div>
            <div className="mt-1 font-mono text-[8.5px] text-slate-600">
              settled turn {String((result as unknown as { from_turn?: number })?.from_turn ?? "?")}
            </div>
          </div>
        ) : isUtterance ? (
          <p className="line-clamp-3 text-[10.5px] leading-snug text-slate-200">
            {String(result?.value ?? "…")}
          </p>
        ) : isTerminal ? (
          <div>
            <div className="text-[12px] font-semibold text-amber-300 truncate">
              {String(result?.value ?? "…")}
            </div>
            {result?.routing && (
              <div className="mt-1 flex gap-2 text-[9px] text-slate-400">
                <span>{result.routing.priority}</span>
                <span>·</span>
                <span>{result.routing.handler}</span>
              </div>
            )}
          </div>
        ) : summary.primitive === "choice" ? (
          <div>
            <div className="flex items-baseline gap-1.5">
              <span className="truncate text-[11.5px] font-semibold text-slate-100">
                {summary.choice}
              </span>
              <span
                className="ml-auto shrink-0 font-mono text-[9px]"
                style={{ color: confColor(summary.confidence) }}
              >
                {summary.confidence?.toFixed(2)}
              </span>
            </div>
            <div className="mt-1 space-y-1">
              {ranked.map(([label, p]) => (
                <div key={label} className="flex items-center gap-1.5">
                  <span className="w-[86px] shrink-0 truncate text-[8.5px] text-slate-400">
                    {label}
                  </span>
                  <Bar p={p} accent={label === summary.choice ? "#8b5cf6" : "#475569"} />
                  <span className="w-6 shrink-0 text-right font-mono text-[8.5px] text-slate-500">
                    {p.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : summary.primitive === "noul" ? (
          <div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-[11.5px] font-semibold text-slate-100">
                P(true) {(summary.noul ?? 0).toFixed(2)}
              </span>
              <span
                className="ml-auto shrink-0 font-mono text-[9px]"
                style={{ color: confColor(summary.confidence) }}
              >
                {summary.confidence?.toFixed(2)}
              </span>
            </div>
            <div className="mt-1.5">
              <Bar p={summary.noul ?? 0} accent={(summary.noul ?? 0) >= 0.5 ? "#8b5cf6" : "#475569"} />
            </div>
          </div>
        ) : (
          <div className="text-[11px] text-slate-200 line-clamp-2">
            {typeof result?.value === "string"
              ? result.value
              : JSON.stringify(result?.value ?? "…")}
          </div>
        )}

        <div className="mt-1.5 flex items-center gap-1.5 font-mono text-[8.5px] text-slate-500">
          {result?.model && <span className="truncate">{result.model}</span>}
          {!result?.model && result && <span>rule</span>}
          {result && result.latency_ms > 0 && (
            <>
              <span>·</span>
              <span>{result.latency_ms.toFixed(0)}ms</span>
              {result.batch_size > 1 && <span className="text-slate-600">b{result.batch_size}</span>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
