// IntervalBar — a single mean ± interval drawn on a shared domain. Pure
// layout: the caller supplies already-derived numbers and the domain (see
// `verdict-model.ts`), this only positions them. Used by `<Forest>`, the
// validation table, and the ship verdict.

import { Show, type Component } from "solid-js";

export interface IntervalBarProps {
  mean: number;
  lower: number;
  upper: number;
  domain: [number, number];
  /** Optional guide line (e.g. 0 = "no better than the opponent"). */
  reference?: number;
  format?: (n: number) => string;
}

const pct = (x: number, [lo, hi]: [number, number]): number => {
  if (hi <= lo) return 0;
  return Math.min(100, Math.max(0, ((x - lo) / (hi - lo)) * 100));
};

export const IntervalBar: Component<IntervalBarProps> = (props) => {
  const fmt = (n: number): string => (props.format ? props.format(n) : n.toFixed(3));
  const left = (): number => pct(props.lower, props.domain);
  const width = (): number => Math.max(0.5, pct(props.upper, props.domain) - left());
  return (
    <div class="tuner-interval" data-testid="interval-bar">
      <div class="tuner-interval-track">
        <Show when={props.reference != null}>
          <span
            class="tuner-interval-ref"
            style={{ left: `${pct(props.reference!, props.domain)}%` }}
          />
        </Show>
        <span class="tuner-interval-range" style={{ left: `${left()}%`, width: `${width()}%` }} />
        <span class="tuner-interval-mean" style={{ left: `${pct(props.mean, props.domain)}%` }} />
      </div>
      <span class="tuner-interval-label">
        {fmt(props.mean)} [{fmt(props.lower)}, {fmt(props.upper)}]
      </span>
    </div>
  );
};
