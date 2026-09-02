// KpiRow — a row of small "label: value" stat tiles. Pure layout; the
// caller supplies already-formatted strings.

import { For, type Component } from "solid-js";

export interface KpiItem {
  label: string;
  value: string;
  hint?: string;
}

export const KpiRow: Component<{ items: KpiItem[]; testid?: string }> = (props) => (
  <div class="tuner-kpi-tiles" data-testid={props.testid ?? "kpi-row"}>
    <For each={props.items}>
      {(item) => (
        <div class="tuner-kpi-tile" title={item.hint}>
          <span class="tuner-kpi-tile-value">{item.value}</span>
          <span class="tuner-kpi-tile-label">{item.label}</span>
        </div>
      )}
    </For>
  </div>
);
