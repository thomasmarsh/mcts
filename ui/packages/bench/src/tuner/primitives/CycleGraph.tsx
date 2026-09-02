// CycleGraph — an inline-SVG directed graph of candidate-vs-candidate
// matchups laid out on a ring in objective-rank order. Directed edges are
// arrows (A → B meaning "A beats B"); undirected edges are plain lines;
// nodes in a material cycle are highlighted. Pure layout: the caller
// derives nodes and edges (see `diagnostic-model.ts`).

import { For, Show, type Component } from "solid-js";

export interface CycleGraphNode {
  key: string;
  label: string;
  badge?: string;
  highlight?: boolean;
  onClick?: () => void;
}

export interface CycleGraphEdge {
  from: string;
  to: string;
  label?: string;
  undirected?: boolean;
}

const SIZE = 220;
const R = 80;
const CENTER = SIZE / 2;

export const CycleGraph: Component<{
  nodes: CycleGraphNode[];
  edges: CycleGraphEdge[];
  testid?: string;
}> = (props) => {
  const pos = (): Map<string, { x: number; y: number }> => {
    const m = new Map<string, { x: number; y: number }>();
    const nodes = props.nodes;
    nodes.forEach((node, i) => {
      const a = (i / Math.max(1, nodes.length)) * Math.PI * 2 - Math.PI / 2;
      m.set(node.key, { x: CENTER + R * Math.cos(a), y: CENTER + R * Math.sin(a) });
    });
    return m;
  };

  return (
    <div class="tuner-cyclegraph" data-testid={props.testid ?? "cycle-graph"}>
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} class="tuner-cyclegraph-svg" role="img">
        <defs>
          <marker
            id="tuner-cyclegraph-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
          </marker>
        </defs>
        <For each={props.edges}>
          {(edge) => {
            const p = pos();
            const a = p.get(edge.from);
            const b = p.get(edge.to);
            if (!a || !b) return null;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const len = Math.hypot(dx, dy) || 1;
            const ux = dx / len;
            const uy = dy / len;
            const x1 = a.x + ux * 16;
            const y1 = a.y + uy * 16;
            const x2 = b.x - ux * 16;
            const y2 = b.y - uy * 16;
            return (
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                class="tuner-cyclegraph-edge"
                stroke="currentColor"
                marker-end={edge.undirected ? undefined : "url(#tuner-cyclegraph-arrow)"}
              >
                <Show when={edge.label}>
                  <title>{edge.label}</title>
                </Show>
              </line>
            );
          }}
        </For>
        <For each={props.nodes}>
          {(node) => {
            const p = pos().get(node.key);
            if (!p) return null;
            return (
              <g
                class="tuner-cyclegraph-node"
                classList={{
                  "tuner-cyclegraph-node-cycle": node.highlight,
                  "tuner-tr-click": !!node.onClick,
                }}
                onClick={() => node.onClick?.()}
              >
                <circle cx={p.x} cy={p.y} r="14" />
                <text x={p.x} y={p.y + 3} text-anchor="middle" class="tuner-cyclegraph-badge">
                  {node.badge ?? ""}
                </text>
                <text x={p.x} y={p.y + 28} text-anchor="middle" class="tuner-cyclegraph-label">
                  {node.label}
                </text>
              </g>
            );
          }}
        </For>
      </svg>
    </div>
  );
};
