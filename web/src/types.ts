export type NodeKind = "utterance" | "decision" | "extract" | "policy" | "terminal";

export interface GraphNode {
  type: "node";
  id: string;
  turn: number;
  col: number;
  key: string;
  title: string;
  kind: NodeKind;
  stage: string;
  primitive: string;
  status: string;
}

export interface NodeSummary {
  primitive?: "choice" | "noul" | "score";
  choice?: string;
  probabilities?: Record<string, number>;
  top_probability?: number;
  confidence?: number;
  noul?: number;
}

export interface NodeResult {
  type: "node_result";
  id: string;
  turn: number;
  status: string;
  summary: NodeSummary;
  question: string;
  options: string[];
  model: string | null;
  routing_reason: string;
  latency_ms: number;
  batch_size: number;
  note: string;
  value?: unknown;
  missing?: string[];
  reason?: string;
  inputs?: Record<string, unknown>;
  template_id?: string;
  driven_by?: string[];
  routing?: RouteOutcome;
}

export interface RouteOutcome {
  queue: string;
  priority: string;
  handler: string;
  flags: string[];
  department?: string;
  intent?: string;
  reasons: string[];
}

export interface GraphEdge {
  type: "edge";
  id: string;
  source: string;
  target: string;
  label: string;
  kind: string;
}

export interface CallStart {
  type: "call_start";
  call_id: string;
  scenario: { id: string; label: string };
  columns: Record<string, string>;
}

export interface TurnStart {
  type: "turn_start";
  turn: number;
  speaker: string;
}

export interface CallEnd {
  type: "call_end";
  call_id: string;
  turns: number;
  decisions: number;
  compute_ms: number;
  total_ms: number;
  input_tokens: number;
  output_tokens: number;
  tokens_generated: number;
  cost_usd: number;
  routing: RouteOutcome | null;
}

export type RunEvent = CallStart | TurnStart | GraphNode | NodeResult | GraphEdge | CallEnd;

export interface Scenario {
  id: string;
  label: string;
  blurb: string;
  turns: string[];
}

export interface RunListItem {
  id: string;
  scenario: { id?: string; label?: string };
  modified: number;
  bytes: number;
}
