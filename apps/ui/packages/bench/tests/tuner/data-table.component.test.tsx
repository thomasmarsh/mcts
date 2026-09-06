// data-table.component.test.tsx — unit test for `DataTable`'s windowed
// rendering mode (Task 14c). No store, no env: `DataTable` is a plain
// presentational component, rendered directly with `@solidjs/testing-library`.

import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { DataTable } from "../../src/tuner/primitives/DataTable.js";

afterEach(cleanup);

interface Row {
  id: string;
  n: number;
}

const rows: Row[] = Array.from({ length: 2000 }, (_, i) => ({ id: `r${i}`, n: i }));

const columns = [{ key: "n", header: "N", render: (r: Row) => String(r.n) }];

describe("DataTable — windowed mode", () => {
  it("renders every row when no pageSize is given", () => {
    render(() => (
      <DataTable testid="t" rows={rows} rowKey={(r) => r.id} columns={columns} />
    ));
    expect(screen.getAllByRole("row")).toHaveLength(rows.length + 1); // +1 header row
  });

  it("renders only one page's worth of rows when pageSize is set", () => {
    render(() => (
      <DataTable testid="t" rows={rows} rowKey={(r) => r.id} columns={columns} pageSize={50} />
    ));
    expect(screen.getAllByRole("row")).toHaveLength(51); // 50 data rows + header
    expect(screen.getByText("Page 1 of 40")).toBeInTheDocument();
  });

  it("Next/Prev walk the window forward and back", () => {
    render(() => (
      <DataTable testid="t" rows={rows} rowKey={(r) => r.id} columns={columns} pageSize={50} />
    ));
    expect(screen.getByText("0")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Next →"));
    expect(screen.getByText("Page 2 of 40")).toBeInTheDocument();
    expect(screen.getByText("50")).toBeInTheDocument();
    expect(screen.queryByText("0")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("← Prev"));
    expect(screen.getByText("Page 1 of 40")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("has no pager when rows fit in one page", () => {
    render(() => (
      <DataTable testid="t" rows={rows.slice(0, 10)} rowKey={(r) => r.id} columns={columns} pageSize={50} />
    ));
    expect(screen.queryByTestId("t-pager")).not.toBeInTheDocument();
  });
});
