// tests/BenchApp.test.tsx — Component-level regression tests for the bench UI
// (RunList, RunDetailPanel). The version-4 tuner UI has its own tests under
// packages/bench/tests/tuner/ and tests/TunerApp.test.tsx.
//
// Uses `@solidjs/testing-library` + a real `createStore` (not `TestStore`)
// against a mocked `BenchEnv` — the same pattern as `GameShell.test.tsx`.
// The mocked env returns deterministic data from the fake-bench fixture,
// so no real server, DuckDB, or browser is involved.

import { afterEach, describe, expect, it, vi } from "vitest";
import type { Component } from "solid-js";
import { Effect } from "@mcts/core";
import { render, screen, fireEvent, cleanup } from "@solidjs/testing-library";
import { createStore } from "@mcts/core";
import {
  benchReducer,
  initialBenchState,
  type BenchState,
  type BenchAction,
  RunList,
  RunDetailPanel,
} from "@mcts/bench";
import { createMockBenchEnv, FAKE_RUN_ID, FAKE_tuner_RUN_ID } from "./fixtures/fake-bench.js";
import type { BenchEnv } from "@mcts/bench";

/** Create a seeded test store with a mocked bench env. */
function createTestStore(envOverrides?: Partial<BenchEnv>) {
  const env = createMockBenchEnv(envOverrides);
  const store = createStore<BenchState, BenchAction>(initialBenchState(), benchReducer, env);
  // Pre-fetch tuner kinds and runs so the UI shows data immediately.
  store.dispatch({ tag: "tunerKinds", action: { tag: "request" } });
  store.dispatch({ tag: "runs", action: { tag: "request" } });
  return { store, env };
}

afterEach(() => {
  cleanup();
});

describe("RunList", () => {
  it("renders the run list from store state", async () => {
    const { store } = createTestStore();
    render(() => <RunList store={store} />);

    // Should show "Runs" heading
    expect(screen.getByText("Runs")).toBeInTheDocument();

    // Should show the fake runs (both status badges and filter options contain these strings;
    // use getAllByText to confirm they appear at least once)
    expect(screen.getAllByText("Completed").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Running").length).toBeGreaterThanOrEqual(1);
  });

  it("opens a run on click", async () => {
    const { store } = createTestStore();
    render(() => (
      <>
        <RunList store={store} />
        <RunDetailPanel store={store} />
      </>
    ));

    // Click on a run row (the first visible one)
    const rows = screen.getAllByText("10 matches");
    fireEvent.click(rows[0]!);

    // The detail panel should now be visible
    await vi.waitFor(() => {
      expect(screen.getByText("Run Detail")).toBeInTheDocument();
    });
  });
});

describe("RunDetailPanel", () => {
  it("shows run metadata when a run is open", async () => {
    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);

    // Nothing visible initially (no open run)
    expect(screen.queryByText("Run Detail")).not.toBeInTheDocument();

    // Open a run via dispatch
    store.dispatch({ tag: "openRun", runId: FAKE_RUN_ID });

    await vi.waitFor(() => {
      expect(screen.getByText("Run Detail")).toBeInTheDocument();
    });

    // Should show run metadata
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("druid")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument(); // match_count
  });

  it("opens the read-only spectator and deletes a completed run through the env", async () => {
    const deleteRun = vi.fn(() => Effect.send(undefined));
    const { store } = createTestStore({ deleteRun });
    const Spectator: Component<{ runId: string; game: string; kind: string; live: boolean }> = (
      props,
    ) => (
      <div data-testid="spectator">
        {props.runId}:{props.game}:{props.kind}:{String(props.live)}
      </div>
    );
    render(() => <RunDetailPanel store={store} Spectator={Spectator} />);
    store.dispatch({ tag: "openRun", runId: FAKE_RUN_ID });

    await vi.waitFor(() => expect(screen.getByText("Browse games")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Browse games"));
    expect(screen.getByTestId("spectator")).toHaveTextContent(
      `${FAKE_RUN_ID}:druid:round_robin:false`,
    );

    fireEvent.click(screen.getByText("Delete"));
    fireEvent.click(screen.getByText("Confirm delete"));
    await vi.waitFor(() => expect(deleteRun).toHaveBeenCalledWith(FAKE_RUN_ID));
    expect(screen.queryByText("Run Detail")).not.toBeInTheDocument();
  });

  it("shows a historical tuner round-robin run in the plain run detail", async () => {
    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);
    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });

    await vi.waitFor(() =>
      expect(screen.getByRole("heading", { name: "Run Detail" })).toBeInTheDocument(),
    );
    expect(screen.getByText("traffic-lights")).toBeInTheDocument();
  });
});
