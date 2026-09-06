// StepLine — a tiny inline-SVG step chart: one horizontal tread per point,
// risers between them, a dot at each point. Theme-aware via `currentColor`.
// Pure layout: the caller supplies points already on data coordinates plus
// the y-domain.

import { For, Show, type Component } from "solid-js";

export interface StepPoint {
  x: number;
  y: number;
  label?: string;
}

export interface StepLineProps {
  points: StepPoint[];
  /** y-axis domain; x spans the points' own [min, max]. */
  domain: [number, number];
  height?: number;
  format?: (n: number) => string;
}

const VW = 100;

export const StepLine: Component<StepLineProps> = (props) => {
  const vh = (): number => props.height ?? 40;
  const xs = (): number[] => props.points.map((p) => p.x);
  const xMin = (): number => Math.min(...xs());
  const xMax = (): number => Math.max(...xs());

  const sx = (x: number): number => {
    const lo = xMin();
    const hi = xMax();
    return hi === lo ? VW / 2 : ((x - lo) / (hi - lo)) * VW;
  };
  const sy = (y: number): number => {
    const [lo, hi] = props.domain;
    const t = hi === lo ? 0.5 : (y - lo) / (hi - lo);
    return vh() - t * vh();
  };

  const path = (): string => {
    const pts = [...props.points].sort((a, b) => a.x - b.x);
    if (pts.length === 0) return "";
    let d = `M ${sx(pts[0]!.x)} ${sy(pts[0]!.y)}`;
    for (let i = 1; i < pts.length; i++) {
      d += ` H ${sx(pts[i]!.x)} V ${sy(pts[i]!.y)}`;
    }
    return d;
  };

  const fmt = (n: number): string => (props.format ? props.format(n) : n.toFixed(3));

  return (
    <div class="tuner-stepline" data-testid="step-line">
      <Show
        when={props.points.length > 0}
        fallback={<p class="tuner-fleet-empty">No steps.</p>}
      >
        <svg
          viewBox={`0 0 ${VW} ${vh()}`}
          preserveAspectRatio="none"
          class="tuner-stepline-svg"
          role="img"
        >
          <path d={path()} fill="none" stroke="currentColor" stroke-width="1.2" />
          <For each={props.points}>
            {(p) => <circle cx={sx(p.x)} cy={sy(p.y)} r="1.6" fill="currentColor" />}
          </For>
        </svg>
        <div class="tuner-stepline-axis">
          <For each={props.points}>
            {(p) => (
              <span class="tuner-stepline-tick">
                {p.label ?? p.x}
                <span class="tuner-stepline-val">{fmt(p.y)}</span>
              </span>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
};
