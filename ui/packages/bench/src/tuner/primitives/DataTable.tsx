// DataTable — a thin generic table. Columns declare a header and a cell
// renderer; the caller does all derivation. Kept deliberately small (no
// built-in sort/paging yet — added when a view needs it).

import { For, Show, type JSX } from "solid-js";

export interface DataColumn<T> {
  key: string;
  header: string;
  render: (row: T) => JSX.Element;
  align?: "left" | "right";
}

export interface DataTableProps<T> {
  columns: DataColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  empty?: string;
  testid?: string;
}

export function DataTable<T>(props: DataTableProps<T>): JSX.Element {
  return (
    <div class="tuner-table-wrap">
      <Show
        when={props.rows.length > 0}
        fallback={<p class="tuner-fleet-empty">{props.empty ?? "No rows."}</p>}
      >
        <table class="tuner-table" data-testid={props.testid ?? "data-table"}>
          <thead>
            <tr>
              <For each={props.columns}>
                {(col) => (
                  <th classList={{ "tuner-td-right": col.align === "right" }}>{col.header}</th>
                )}
              </For>
            </tr>
          </thead>
          <tbody>
            <For each={props.rows}>
              {(row) => (
                <tr
                  classList={{ "tuner-tr-click": !!props.onRowClick }}
                  onClick={() => props.onRowClick?.(row)}
                >
                  <For each={props.columns}>
                    {(col) => (
                      <td classList={{ "tuner-td-right": col.align === "right" }}>
                        {col.render(row)}
                      </td>
                    )}
                  </For>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </Show>
    </div>
  );
}
