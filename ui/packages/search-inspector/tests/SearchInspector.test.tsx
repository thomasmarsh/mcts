import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import "@testing-library/jest-dom/vitest";
import type { SearchReport } from "@mcts/game";
import { SearchInspector, type SearchInspectorPoint } from "../src/index.js";

afterEach(() => cleanup());

type Move = { move: string };

function report(overrides: Partial<SearchReport<Move>> = {}): SearchReport<Move> {
  return {
    status: "available",
    schema_version: 1,
    reason: null,
    elapsed_seconds: 0,
    iteration_limit: 0,
    time_limit_seconds: 0,
    completed_iterations: 0,
    termination: "iterations",
    selected_action: { move: "a" },
    actions: [
      { action: { move: "a" }, visits: 0, share: 0, mean_value: 0, is_proven: false },
      { action: { move: "b" }, visits: 7, share: 1, mean_value: 0.5, is_proven: true },
    ],
    principal_variation: [{ move: "a" }, { move: "b" }],
    root_visits: 0,
    tree_nodes: 0,
    mean_depth: 0,
    max_depth: 0,
    graph_mode: "tree",
    tt_reads: 0,
    tt_writes: 0,
    tt_hits: 0,
    tt_hit_ratio: 0,
    iterations_per_second: 0,
    warnings: [],
    ...overrides,
  } as SearchReport<Move>;
}

describe("SearchInspector", () => {
  it("distinguishes legacy, unavailable, and partial reports", () => {
    const { unmount } = render(() => <SearchInspector report={null} before={{ turn: 1 }} />);
    expect(screen.getByRole("status")).toHaveTextContent("legacy result");
    unmount();

    render(() => <SearchInspector report={report({ status: "unavailable", reason: "strategy_unsupported" })} before={{ turn: 1 }} />);
    expect(screen.getByRole("status")).toHaveTextContent("evidence unavailable. This strategy does not expose final-search evidence.");
    cleanup();

    render(() => <SearchInspector report={report({ status: "partial", reason: null, warnings: ["actions_truncated", "structural_diagnostics_omitted"] })} before={{ turn: 1 }} />);
    expect(screen.getByRole("status")).toHaveTextContent("evidence is partial");
    expect(screen.getByText("The action list was truncated before every root action could be retained.")).toBeInTheDocument();
    expect(screen.getByText("Tree and graph diagnostics were not retained for this search.")).toBeInTheDocument();
  });

  it("keeps zero counters visible and marks the selected action without calling outcome proof a win", () => {
    render(() => <SearchInspector report={report()} before={{ turn: 1 }} />);

    const summary = screen.getByRole("heading", { name: "Search summary" }).parentElement!;
    expect(summary).toHaveTextContent("Completed iterations0");
    expect(summary).toHaveTextContent("TT reads0");
    expect(summary).toHaveTextContent("TT hit ratio0.0%");
    expect(screen.getAllByText("Selected")).toHaveLength(2);
    expect(screen.getByRole("columnheader", { name: "Outcome proven" })).toBeInTheDocument();
    expect(screen.queryByText(/proven win/i)).toBeNull();
  });

  it("formats root actions and only the first PV action with the known pre-move state", () => {
    render(() => <SearchInspector report={report()} before={{ turn: 3 }} formatMove={(move, before) => `${move.move} at ${before.turn}`} />);

    expect(screen.getByText("a at 3")).toBeInTheDocument();
    expect(screen.getByText('→ {"move":"b"}', { exact: false })).toBeInTheDocument();
    cleanup();

    render(() => <SearchInspector report={report()} before={{ turn: 3 }} />);
    expect(screen.getAllByText('{"move":"a"}').length).toBeGreaterThan(0);
  });

  it("plots the selected metric with gaps and provides exact per-ply values", () => {
    const points: SearchInspectorPoint<Move>[] = [
      { ply: 1, player: "Black", move: { move: "a" }, report: report({ completed_iterations: 10, elapsed_seconds: 2, tt_hit_ratio: 0.25 }) },
      { ply: 2, player: "White", move: { move: "b" }, report: null },
      { ply: 3, player: "Black", move: { move: "c" }, report: report({ status: "unavailable", completed_iterations: 999, tt_hit_ratio: 1 }) },
      { ply: 4, player: "White", move: { move: "d" }, report: report({ completed_iterations: 0, elapsed_seconds: 0, tt_hit_ratio: 0 }) },
    ];
    render(() => <SearchInspector report={report()} before={{ turn: 1 }} points={points} />);

    const metric = screen.getByLabelText("Metric") as HTMLSelectElement;
    metric.focus();
    expect(metric).toHaveFocus();
    expect(metric.options).toHaveLength(6);
    expect(screen.getByRole("img", { name: "Search metric trend: Iterations" })).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "Exact per-ply values for Iterations" });
    expect(table).toHaveTextContent("10");
    expect(table).toHaveTextContent("Unavailable");
    expect(table).toHaveTextContent("0");

    fireEvent.keyDown(metric, { key: "ArrowDown" });
    fireEvent.change(metric, { target: { value: "ttHitRatio" } });
    expect(metric).toHaveFocus();
    expect(screen.getByRole("img", { name: "Search metric trend: TT hit ratio" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Exact per-ply values for TT hit ratio" })).toHaveTextContent("25.0%");
  });
});
