// RaceStrip — a candidate × prefix grid of shadow dispositions: one row per
// candidate, one column per common prefix, each cell tinted by the
// disposition recorded there. Pure layout; `<CohortRace>` derivation lives
// in `race-model.ts`.

import { For, Show, type Component } from "solid-js";

export interface RaceStripColumn {
  label: string;
  title?: string;
}

export interface RaceStripRow {
  key: string;
  label: string;
  note?: string;
  highlight?: boolean;
  cells: (string | null)[];
  onClick?: () => void;
}

/** Coarse disposition → CSS modifier. Unknown dispositions fall through to
 * a neutral cell so a new backend value still renders. */
export function dispositionClass(disposition: string | null): string {
  if (!disposition) return "tuner-race-cell-empty";
  const d = disposition.toLowerCase();
  if (d.includes("promot")) return "tuner-race-cell-promote";
  if (d.includes("eliminat") || d.includes("prune")) return "tuner-race-cell-eliminate";
  if (d.includes("protect")) return "tuner-race-cell-protect";
  if (d.includes("continue") || d.includes("audit")) return "tuner-race-cell-continue";
  return "tuner-race-cell-other";
}

export const RaceStrip: Component<{
  columns: RaceStripColumn[];
  rows: RaceStripRow[];
  testid?: string;
}> = (props) => (
  <div class="tuner-race-wrap">
    <table class="tuner-race" data-testid={props.testid ?? "race-strip"}>
      <thead>
        <tr>
          <th />
          <For each={props.columns}>{(c) => <th title={c.title}>{c.label}</th>}</For>
        </tr>
      </thead>
      <tbody>
        <For each={props.rows}>
          {(row) => (
            <tr
              classList={{
                "tuner-tr-click": !!row.onClick,
                "tuner-race-row-highlight": row.highlight,
              }}
              onClick={() => row.onClick?.()}
            >
              <th scope="row" class="tuner-race-rowhead">
                {row.label}
                <Show when={row.note}>
                  <span class="tuner-race-rownote"> {row.note}</span>
                </Show>
              </th>
              <For each={row.cells}>
                {(cell) => (
                  <td class={dispositionClass(cell)} title={cell ?? "no look"}>
                    <span class="tuner-race-dot" />
                  </td>
                )}
              </For>
            </tr>
          )}
        </For>
      </tbody>
    </table>
  </div>
);
