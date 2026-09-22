import { useCallback, useEffect, useMemo } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  useReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import { useRun } from "../store/run";
import { laneRect, nodeCenter, nodePosition } from "./layout";
import StageNode from "./nodes/StageNode";
import LaneNode from "./nodes/LaneNode";

const nodeTypes = { stage: StageNode, lane: LaneNode };

const EDGE_COLOR: Record<string, string> = {
  flow: "#334155",
  branch: "#8b5cf6",
  policy: "#475569",
  extract: "#64748b",
  terminal: "#f59e0b",
};

function Legend() {
  const items = [
    ["#3b82f6", "caller"],
    ["#8b5cf6", "model decision"],
    ["#64748b", "rule / regex"],
    ["#10b981", "switchboard"],
    ["#f59e0b", "route out"],
  ];
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-1.5 backdrop-blur">
      {items.map(([c, l]) => (
        <span key={l} className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <span className="h-2 w-2 rounded-sm" style={{ background: c }} />
          {l}
        </span>
      ))}
    </div>
  );
}

function Canvas() {
  const nodesMap = useRun((s) => s.nodes);
  const edgesMap = useRun((s) => s.edges);
  const turns = useRun((s) => s.turns);
  const selected = useRun((s) => s.selected);
  const select = useRun((s) => s.select);
  const runId = useRun((s) => s.runId);
  const summary = useRun((s) => s.summary);
  const activeId = useRun((s) => s.activeId);
  const follow = useRun((s) => s.follow);
  const { fitView, setCenter } = useReactFlow();

  // Frame the whole call when a run loads, and again once it completes (the graph grows tall as
  // turns accumulate). A zoom floor keeps the default view readable rather than microscopic.
  useEffect(() => {
    const id = window.setTimeout(() => fitView({ padding: 0.12, minZoom: 0.3, maxZoom: 1 }), 70);
    return () => window.clearTimeout(id);
  }, [runId, summary, fitView]);

  // Camera follows the active node so you can watch the decision move through the pipeline.
  useEffect(() => {
    if (!follow || !activeId) return;
    const node = nodesMap[activeId];
    if (!node) return;
    const c = nodeCenter(node.col, node.turn);
    const id = window.setTimeout(
      () => setCenter(c.x, c.y, { zoom: 0.72, duration: 420 }),
      40
    );
    return () => window.clearTimeout(id);
  }, [activeId, follow, nodesMap, setCenter]);

  const rfNodes = useMemo<Node[]>(() => {
    const lanes: Node[] = turns.map((t) => {
      const r = laneRect(t);
      return {
        id: `lane-${t}`,
        type: "lane",
        position: { x: r.x, y: r.y },
        data: { turn: t },
        draggable: false,
        selectable: false,
        focusable: false,
        zIndex: -1,
        style: { width: r.width, height: r.height, pointerEvents: "none" },
      };
    });
    const stage: Node[] = Object.values(nodesMap).map((n) => ({
      id: n.id,
      type: "stage",
      position: nodePosition(n.col, n.turn),
      data: n as unknown as Record<string, unknown>,
      draggable: false,
      selected: n.id === selected,
    }));
    return [...lanes, ...stage];
  }, [nodesMap, turns, selected]);

  const rfEdges = useMemo<Edge[]>(
    () =>
      Object.values(edgesMap).map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        type: "smoothstep",
        label: e.label || undefined,
        animated: e.kind === "branch" || e.kind === "terminal",
        style: { stroke: EDGE_COLOR[e.kind] ?? EDGE_COLOR.flow },
        labelStyle: { fill: "#94a3b8", fontSize: 9, fontFamily: "ui-monospace" },
        labelBgStyle: { fill: "#0b1120" },
        labelBgPadding: [3, 2] as [number, number],
        labelBgBorderRadius: 3,
      })),
    [edgesMap]
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (node.type === "stage") select(node.id);
    },
    [select]
  );

  return (
    <ReactFlow
      nodes={rfNodes}
      edges={rfEdges}
      nodeTypes={nodeTypes}
      onNodeClick={onNodeClick}
      onPaneClick={() => select(null)}
      onNodesChange={() => {}}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable
      minZoom={0.15}
      maxZoom={1.6}
      fitView
      fitViewOptions={{ padding: 0.14, minZoom: 0.55, maxZoom: 1 }}
      proOptions={{ hideAttribution: false }}
      key={runId ?? "empty"}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#162033" />
      <Controls
        showInteractive={false}
        className="!border !border-slate-800 !bg-slate-900/90 [&_button]:!border-slate-800 [&_button]:!bg-slate-900 [&_button]:!fill-slate-300 [&_button:hover]:!bg-slate-800"
      />
      <MiniMap
        pannable
        zoomable
        style={{ background: "#0b1120", border: "1px solid #1e293b" }}
        maskColor="rgba(2,6,23,0.6)"
        nodeColor={(n) => (n.type === "lane" ? "#0f172a" : "#334155")}
      />
      <Panel position="top-left">
        <Legend />
      </Panel>
    </ReactFlow>
  );
}

export default function GraphCanvas() {
  return <Canvas />;
}
