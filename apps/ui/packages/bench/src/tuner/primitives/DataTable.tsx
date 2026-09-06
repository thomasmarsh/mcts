// DataTable — a thin generic table. Columns declare a header and a cell
// renderer; the caller does all derivation. Kept deliberately small: no
// built-in sort, and paging is a hand-rolled client-side window (`pageSize`)
// rather than a virtualization library, for tables whose row count can grow
// large enough that rendering every row as a `<tr>` is itself the cost.

import { createEffect, createMemo, createSignal, For, Show, type JSX } from "solid-js";

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
  /** When set, renders only this many rows at a time (with Prev/Next
   * controls) instead of every row in `rows`. Purely a client-side window
   * over whatever `rows` already holds -- it doesn't fetch anything. */
  pageSize?: number;
}

export function DataTable<T>(props: DataTableProps<T>): JSX.Element {
  const [page, setPage] = createSignal(0);
  const pageCount = createMemo(() =>
    props.pageSize ? Math.max(1, Math.ceil(props.rows.length / props.pageSize)) : 1,
  );
  // Clamp back onto a valid page if `rows` shrinks (e.g. a filter change)
  // out from under the current page index.
  createEffect(() => {
    if (page() >= pageCount()) setPage(0);
  });
  const windowRows = createMemo(() => {
    if (!props.pageSize) return props.rows;
    const start = page() * props.pageSize;
    return props.rows.slice(start, start + props.pageSize);
  });
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
            <For each={windowRows()}>
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
        <Show when={props.pageSize && pageCount() > 1}>
          <div class="tuner-table-pager" data-testid={`${props.testid ?? "data-table"}-pager`}>
            <button disabled={page() === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>
              ← Prev
            </button>
            <span>
              Page {page() + 1} of {pageCount()}
            </span>
            <button
              disabled={page() >= pageCount() - 1}
              onClick={() => setPage((p) => Math.min(pageCount() - 1, p + 1))}
            >
              Next →
            </button>
          </div>
        </Show>
      </Show>
    </div>
  );
}
