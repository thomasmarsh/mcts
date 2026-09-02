// Heatmap — a labelled grid of cells tinted by an intensity in [0, 1].
// Pure layout: the caller derives every cell's label and intensity and
// decides which cells to flag. Used by the opponent-response matrix and the
// shadow-calibration bins.

import { For, Show, type Component } from "solid-js";

export interface HeatmapCell {
  label: string;
  title?: string;
  /** 0..1 — drives the cell background opacity. */
  intensity: number;
  /** Draw an attention outline (material interaction, reversal). */
  flag?: boolean;
  onClick?: () => void;
}

export interface HeatmapColumn {
  key: string;
  label: string;
  title?: string;
}

export interface HeatmapRow {
  key: string;
  label: string;
  note?: string;
  cells: HeatmapCell[];
  onClick?: () => void;
}

export const Heatmap: Component<{
  columns: HeatmapColumn[];
  rows: HeatmapRow[];
  testid?: string;
}> = (props) => (
  <div class="tuner-heatmap-wrap">
    <table class="tuner-heatmap" data-testid={props.testid ?? "heatmap"}>
      <thead>
        <tr>
          <th />
          <For each={props.columns}>{(c) => <th title={c.title}>{c.label}</th>}</For>
        </tr>
      </thead>
      <tbody>
        <For each={props.rows}>
          {(row) => (
            <tr classList={{ "tuner-tr-click": !!row.onClick }} onClick={() => row.onClick?.()}>
              <th scope="row" class="tuner-heatmap-rowhead">
                {row.label}
                <Show when={row.note}>
                  <span class="tuner-heatmap-rownote"> {row.note}</span>
                </Show>
              </th>
              <For each={row.cells}>
                {(cell) => (
                  <td
                    class="tuner-heatmap-cell"
                    classList={{
                      "tuner-heatmap-cell-flag": cell.flag,
                      "tuner-tr-click": !!cell.onClick,
                    }}
                    title={cell.title ?? cell.label}
                    onClick={(e) => {
                      if (cell.onClick) {
                        e.stopPropagation();
                        cell.onClick();
                      }
                    }}
                  >
                    <span
                      class="tuner-heatmap-fill"
                      style={{ opacity: String(Math.min(1, Math.max(0, cell.intensity))) }}
                    />
                    <span class="tuner-heatmap-label">{cell.label}</span>
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
