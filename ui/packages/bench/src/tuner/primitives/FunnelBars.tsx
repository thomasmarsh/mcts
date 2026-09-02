// FunnelBars — one horizontal bar per stage on a shared scale, with a
// filled portion (e.g. accepted) inside the total (e.g. attempted) and an
// optional note. Pure layout: the caller derives every number.

import { For, Show, type Component } from "solid-js";

export interface FunnelRow {
  label: string;
  /** Bar length on the shared scale. */
  total: number;
  /** Highlighted sub-portion, `0..total`. */
  filled: number;
  note?: string;
}

export const FunnelBars: Component<{ rows: FunnelRow[]; testid?: string }> = (props) => {
  const max = (): number => Math.max(1, ...props.rows.map((r) => r.total));
  return (
    <div class="tuner-funnel" data-testid={props.testid ?? "funnel-bars"}>
      <For each={props.rows}>
        {(row) => (
          <div class="tuner-funnel-row">
            <span class="tuner-funnel-label">{row.label}</span>
            <div class="tuner-funnel-track">
              <span
                class="tuner-funnel-total"
                style={{ width: `${(row.total / max()) * 100}%` }}
              />
              <span
                class="tuner-funnel-filled"
                style={{ width: `${(Math.min(row.filled, row.total) / max()) * 100}%` }}
              />
            </div>
            <span class="tuner-funnel-count">
              {row.filled}/{row.total}
              <Show when={row.note}>
                <span class="tuner-funnel-note"> · {row.note}</span>
              </Show>
            </span>
          </div>
        )}
      </For>
    </div>
  );
};
