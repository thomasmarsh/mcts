// Treemap — one stacked proportional bar per group, each split into child
// segments sized by value. A flat "treemap" that stays legible in a small
// viewport and in both themes. Pure layout: the caller derives the groups
// and the child values.

import { For, Show, type Component } from "solid-js";

export interface TreemapChild {
  label: string;
  value: number;
}

export interface TreemapGroup {
  key: string;
  label: string;
  /** Optional explicit denominator; defaults to the sum of children. */
  total?: number;
  children: TreemapChild[];
}

/** Stable segment tint index by conventional disposition name. */
function segmentClass(label: string): string {
  const l = label.toLowerCase();
  if (l.includes("complete")) return "tuner-treemap-seg-done";
  if (l.includes("fail")) return "tuner-treemap-seg-fail";
  if (l.includes("censor")) return "tuner-treemap-seg-censor";
  if (l.includes("overrun")) return "tuner-treemap-seg-overrun";
  if (l.includes("unspent")) return "tuner-treemap-seg-unspent";
  return "tuner-treemap-seg-other";
}

export const Treemap: Component<{
  groups: TreemapGroup[];
  format?: (n: number) => string;
  testid?: string;
}> = (props) => {
  const fmt = (n: number): string => (props.format ? props.format(n) : String(n));
  return (
    <div class="tuner-treemap" data-testid={props.testid ?? "treemap"}>
      <For each={props.groups}>
        {(group) => {
          const total = (): number =>
            group.total ?? (group.children.reduce((a, c) => a + c.value, 0) || 1);
          return (
            <div class="tuner-treemap-group">
              <div class="tuner-treemap-head">
                <span class="tuner-treemap-label">{group.label}</span>
                <span class="tuner-treemap-total">{fmt(total())}</span>
              </div>
              <div class="tuner-treemap-track">
                <For each={group.children}>
                  {(child) => (
                    <span
                      class={`tuner-treemap-seg ${segmentClass(child.label)}`}
                      style={{ width: `${(child.value / total()) * 100}%` }}
                      title={`${child.label}: ${fmt(child.value)}`}
                    />
                  )}
                </For>
              </div>
              <div class="tuner-treemap-legend">
                <For each={group.children}>
                  {(child) => (
                    <span class="tuner-treemap-key">
                      <span class={`tuner-treemap-swatch ${segmentClass(child.label)}`} />
                      {child.label} {fmt(child.value)}
                    </span>
                  )}
                </For>
              </div>
            </div>
          );
        }}
      </For>
      <Show when={props.groups.length === 0}>
        <p class="tuner-fleet-empty">No compute recorded.</p>
      </Show>
    </div>
  );
};
