import type { NodeKind } from "./types";

export interface KindStyle {
  accent: string;
  label: string;
  /** solid background for the node header strip */
  tint: string;
}

/** Colour is semantic: who is speaking, or whether a model or a rule made the decision. */
export const KIND_STYLE: Record<string, KindStyle> = {
  caller: { accent: "#3b82f6", label: "caller", tint: "rgba(59,130,246,0.14)" },
  agent: { accent: "#10b981", label: "switchboard", tint: "rgba(16,185,129,0.14)" },
  decision: { accent: "#8b5cf6", label: "model", tint: "rgba(139,92,246,0.14)" },
  extract: { accent: "#64748b", label: "regex", tint: "rgba(100,116,139,0.14)" },
  policy: { accent: "#64748b", label: "policy", tint: "rgba(100,116,139,0.14)" },
  terminal: { accent: "#f59e0b", label: "route", tint: "rgba(245,158,11,0.16)" },
};

export function styleFor(kind: NodeKind, key: string): KindStyle {
  if (kind === "utterance") return key === "caller" ? KIND_STYLE.caller : KIND_STYLE.agent;
  return KIND_STYLE[kind] ?? KIND_STYLE.decision;
}

export const STATUS_ACCENT: Record<string, string> = {
  running: "#334155",
  ok: "#22c55e",
  low_confidence: "#f59e0b",
  warn: "#f59e0b",
  rejected: "#f43f5e",
  skipped: "#334155",
};

export function confColor(c: number | undefined | null): string {
  if (c === undefined || c === null) return "#64748b";
  if (c >= 0.75) return "#22c55e";
  if (c >= 0.45) return "#f59e0b";
  return "#f43f5e";
}
