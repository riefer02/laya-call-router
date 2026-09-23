import { create } from "zustand";
import { useMemo } from "react";
import type {
  CallEnd,
  GraphEdge,
  GraphNode,
  NodeResult,
  RunEvent,
  RouteOutcome,
} from "../types";

export interface AppliedNode extends GraphNode {
  result?: NodeResult;
}

interface RunState {
  events: RunEvent[];
  applied: number;
  playing: boolean;
  speed: number; // events per second
  runId: string | null;
  scenarioLabel: string;
  nodes: Record<string, AppliedNode>;
  edges: Record<string, GraphEdge>;
  turns: number[];
  routing: RouteOutcome | null;
  summary: CallEnd | null;
  selected: string | null;
  activeId: string | null;
  follow: boolean;

  load: (events: RunEvent[], opts?: { runId?: string; label?: string }) => void;
  step: () => void;
  applyAll: () => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  setSpeed: (n: number) => void;
  setFollow: (v: boolean) => void;
  select: (id: string | null) => void;
  reset: () => void;
}

type GraphSlice = Pick<
  RunState,
  | "nodes"
  | "edges"
  | "turns"
  | "routing"
  | "summary"
  | "scenarioLabel"
  | "playing"
  | "activeId"
>;

function applyOne(slice: GraphSlice, ev: RunEvent): Partial<GraphSlice> {
  switch (ev.type) {
    case "call_start":
      return { scenarioLabel: ev.scenario?.label ?? "call" };
    case "turn_start":
      return slice.turns.includes(ev.turn) ? {} : { turns: [...slice.turns, ev.turn] };
    case "node": {
      const node = ev as GraphNode;
      return {
        nodes: { ...slice.nodes, [node.id]: { ...node } },
        turns: slice.turns.includes(node.turn) ? slice.turns : [...slice.turns, node.turn],
        activeId: node.id,
      };
    }
    case "node_result": {
      const res = ev as NodeResult;
      const existing = slice.nodes[res.id];
      const patch: Partial<GraphSlice> = { activeId: res.id };
      if (existing) {
        patch.nodes = { ...slice.nodes, [res.id]: { ...existing, result: res, status: res.status } };
      }
      if (res.routing) patch.routing = res.routing;
      return patch;
    }
    case "edge":
      return { edges: { ...slice.edges, [ev.id]: ev as GraphEdge } };
    case "call_end":
      return {
        summary: ev as CallEnd,
        playing: false,
        routing: (ev as CallEnd).routing ?? slice.routing,
      };
    default:
      return {};
  }
}

function mergeInto(target: GraphSlice, patch: Partial<GraphSlice>): void {
  if (patch.nodes) target.nodes = patch.nodes;
  if (patch.edges) target.edges = patch.edges;
  if (patch.turns) target.turns = patch.turns;
  if (patch.routing !== undefined) target.routing = patch.routing;
  if (patch.summary !== undefined) target.summary = patch.summary;
  if (patch.scenarioLabel !== undefined) target.scenarioLabel = patch.scenarioLabel;
  if (patch.playing !== undefined) target.playing = patch.playing;
  if (patch.activeId !== undefined) target.activeId = patch.activeId;
}

const initial = {
  events: [] as RunEvent[],
  applied: 0,
  playing: false,
  speed: 3,
  runId: null as string | null,
  scenarioLabel: "",
  nodes: {} as Record<string, AppliedNode>,
  edges: {} as Record<string, GraphEdge>,
  turns: [] as number[],
  routing: null as RouteOutcome | null,
  summary: null as CallEnd | null,
  selected: null as string | null,
  activeId: null as string | null,
  follow: true,
};

export const useRun = create<RunState>((set, get) => ({
  ...initial,

  load: (events, opts) =>
    set((current) => ({
      ...initial,
      speed: current.speed,
      follow: current.follow,
      events,
      runId: opts?.runId ?? null,
      scenarioLabel:
        opts?.label ??
        (events.find((e) => e.type === "call_start") as { scenario?: { label?: string } } | undefined)
          ?.scenario?.label ??
        "",
      playing: true,
    })),

  step: () => {
    const s = get();
    if (s.applied >= s.events.length) {
      set({ playing: false });
      return;
    }
    const patch = applyOne(s, s.events[s.applied]);
    set({ ...patch, applied: s.applied + 1 } as Partial<RunState>);
  },

  applyAll: () => {
    const s = get();
    const target: GraphSlice = {
      nodes: s.nodes,
      edges: s.edges,
      turns: s.turns,
      routing: s.routing,
      summary: s.summary,
      scenarioLabel: s.scenarioLabel,
      playing: s.playing,
      activeId: s.activeId,
    };
    for (let i = s.applied; i < s.events.length; i++) {
      mergeInto(target, applyOne(target, s.events[i]));
    }
    set({ ...target, playing: false, applied: s.events.length });
  },

  play: () => {
    if (get().applied >= get().events.length) return;
    set({ playing: true });
  },
  pause: () => set({ playing: false }),
  toggle: () => set({ playing: !get().playing }),
  setSpeed: (n) => set({ speed: n }),
  setFollow: (v) => set({ follow: v }),
  select: (id) => set({ selected: id }),
  reset: () => set((current) => ({ ...initial, speed: current.speed, follow: current.follow })),
}));

/** Counts for the HUD, derived from what has been revealed so far. */
export function useCounters() {
  const nodes = useRun((s) => s.nodes);
  const turnCount = useRun((s) => s.turns.length);
  const applied = useRun((s) => s.applied);
  const total = useRun((s) => s.events.length);
  return useMemo(() => {
    const list = Object.values(nodes);
    const decisions = list.filter((n) => n.kind === "decision");
    return {
      nodes: list.length,
      decisions: decisions.filter((n) => n.status !== "skipped").length,
      skipped: decisions.filter((n) => n.status === "skipped").length,
      turns: turnCount,
      applied,
      total,
    };
  }, [nodes, turnCount, applied, total]);
}
