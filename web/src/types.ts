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
  completion?: "booked" | "dispatched" | "transferred" | "awaiting_caller";
  booking?: Booking | null;
  contact?: Contact | null;
}

export interface RouteOutcome {
  queue: string;
  priority: string;
  handler: string;
  flags: string[];
  destination?: string;
  subqueue?: string;
  reasons: string[];
}

/** An appointment that was actually filed, with a real time and an id. */
export interface Booking {
  id: string;
  location: string;
  destination: string;
  subqueue: string | null;
  queue: string;
  slot_day: string;
  slot_time: string;
  duration_min: number;
  caller_name: string;
  callback_number: string;
  vehicle: string;
}

export interface Contact {
  caller_name?: string | null;
  callback_number?: string | null;
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
  completion?: "booked" | "dispatched" | "transferred" | "awaiting_caller";
  booking?: Booking | null;
  contact?: Contact | null;
  offered?: string[];
  escalations?: number;
  llm_escalations?: number;
  turn_stats?: Array<{ turn: number; questions: number; input_tokens: number; compute_ms: number }>;
}

export type RunEvent = CallStart | TurnStart | GraphNode | NodeResult | GraphEdge | CallEnd;

export interface Scenario {
  id: string;
  label: string;
  blurb: string;
  turns: string[];
  expect?: { queue: string; completion: string };
  known_issue?: string;
}

export interface RunListItem {
  id: string;
  scenario: { id?: string; label?: string };
  modified: number;
  bytes: number;
}

// --------------------------------------------------------------------------- evidence
export interface ArmScore {
  key: string;
  label: string;
  destination: number | null;
  destination_ci: number | null;
  subqueue: number | null;
  joint: number | null;
  queue: number | null;
  latency_p50: number | null;
  cost_per_case: number | null;
  determinism: number | null;
  calibration: Record<string, { n: number; accuracy: number }>;
  other_rate: number | null;
  gate: Record<string, unknown>;
}

export interface SubQueue {
  key: string;
  label: string;
  description: string;
  queue: string;
  handler: string;
}

export interface TaxonomyNode {
  key: string;
  label: string;
  description: string;
  queue: string;
  subqueues: SubQueue[];
}

export interface SeverityScore {
  recall: number | null;
  precision: number | null;
  missed: number;
  positives: number;
  false_alarms: number;
  missed_ids?: string[];
}

export interface SeverityRow {
  arm: string;
  safe: SeverityScore | null;
  human: SeverityScore | null;
  sweep: Array<SeverityScore & { threshold: number }>;
}

export interface Evidence {
  taxonomy: TaxonomyNode[];
  n_cases: number;
  arms: ArmScore[];
  calls: Array<{
    label: string;
    queue: number | null;
    questions: number | null;
    latency_p50: number | null;
    cost: number | null;
  }>;
  calibration: Record<string, { n: number; accuracy: number }>;
  severity: SeverityRow[];
  dataset: {
    kept: number | null;
    by_destination: Record<string, number>;
    cost_usd: number | null;
    criteria: Record<string, unknown>;
  };
  generality: {
    suites: Record<string, Record<string, { accuracy?: number }>>;
    verdict: string | null;
  };
  policy: Record<string, unknown>;
  facts: Record<string, string>;
}
