// Forest — a stack of labelled `<IntervalBar>` rows on one shared domain,
// the classic racing / candidate-performance plot. Pure layout; the caller
// derives the rows and (optionally) the domain.

import { For, type Component } from "solid-js";
import { IntervalBar } from "./IntervalBar.js";

export interface ForestRow {
  key: string;
  label: string;
  mean: number;
  lower: number;
  upper: number;
  highlight?: boolean;
  onClick?: () => void;
}

export interface ForestProps {
  rows: ForestRow[];
  /** Shared x-domain; when omitted, spans the rows' own extent. */
  domain?: [number, number];
  reference?: number;
  format?: (n: number) => string;
}

function spanOf(rows: ForestRow[]): [number, number] {
  if (rows.length === 0) return [-1, 1];
  const lo = Math.min(...rows.map((r) => r.lower));
  const hi = Math.max(...rows.map((r) => r.upper));
  const pad = (hi - lo) * 0.05 || 0.05;
  return [lo - pad, hi + pad];
}

export const Forest: Component<ForestProps> = (props) => {
  const domain = (): [number, number] => props.domain ?? spanOf(props.rows);
  return (
    <div class="tuner-forest" data-testid="forest">
      <For each={props.rows}>
        {(row) => (
          <div
            class="tuner-forest-row"
            classList={{ "tuner-forest-row-highlight": row.highlight }}
            onClick={() => row.onClick?.()}
          >
            <span class="tuner-forest-label">{row.label}</span>
            <IntervalBar
              mean={row.mean}
              lower={row.lower}
              upper={row.upper}
              domain={domain()}
              reference={props.reference}
              format={props.format}
            />
          </div>
        )}
      </For>
    </div>
  );
};
