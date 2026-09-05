// TimelineLanes — a lightweight inline-SVG lane chart: one row per run-loop
// stage, each drawn from the stage's first span to its last on a shared
// [0, wall] axis, with an inner fill for the time actually recorded in that
// stage (its duty within its own span). Not a flame graph — Perfetto is one
// download away for that. Pure layout; the caller derives the lanes.

import { For, Show, type Component } from "solid-js";
import type { TimelineLane } from "../models/timeline-model.js";

const VW = 100;
const ROW = 12;
const GAP = 4;

function fmtMs(ms: number): string {
  if (ms >= 3_600_000) return `${(ms / 3_600_000).toFixed(1)}h`;
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(1)}m`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export const TimelineLanes: Component<{
  lanes: TimelineLane[];
  wallMs: number;
  testid?: string;
}> = (props) => {
  const scale = (ms: number): number =>
    props.wallMs > 0 ? Math.max(0, Math.min(VW, (ms / props.wallMs) * VW)) : 0;
  const vh = (): number => Math.max(ROW, props.lanes.length * (ROW + GAP));

  return (
    <Show
      when={props.lanes.length > 0}
      fallback={<p class="tuner-fleet-empty">No timing recorded for this run.</p>}
    >
      <div class="tuner-timeline" data-testid={props.testid ?? "timeline-lanes"}>
        <svg
          class="tuner-timeline-svg"
          viewBox={`0 0 ${VW} ${vh()}`}
          preserveAspectRatio="none"
          role="img"
        >
          <For each={props.lanes}>
            {(lane, i) => (
              <g transform={`translate(0 ${i() * (ROW + GAP)})`}>
                <rect
                  class="tuner-timeline-extent"
                  x={scale(lane.offsetMs)}
                  y={0}
                  width={Math.max(0.5, scale(lane.extentMs))}
                  height={ROW}
                />
                <rect
                  class="tuner-timeline-active"
                  x={scale(lane.offsetMs)}
                  y={0}
                  width={Math.max(0.5, scale(Math.min(lane.activeMs, lane.extentMs)))}
                  height={ROW}
                />
              </g>
            )}
          </For>
        </svg>
        <ul class="tuner-timeline-legend">
          <For each={props.lanes}>
            {(lane) => (
              <li>
                <span class="tuner-timeline-legend-name">{lane.name}</span>
                <span class="tuner-timeline-legend-val">
                  {fmtMs(lane.activeMs)} · {lane.spanCount} spans
                </span>
              </li>
            )}
          </For>
        </ul>
      </div>
    </Show>
  );
};
