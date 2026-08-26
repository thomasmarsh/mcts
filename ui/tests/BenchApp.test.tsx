// tests/BenchApp.test.tsx — Component-level regression tests for the bench UI
// (LaunchForm, RunList, RunDetailPanel).
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
  LaunchForm,
  RunList,
  RunDetailPanel,
} from "@mcts/bench";
import { createMockBenchEnv, FAKE_RUN_ID, FAKE_tuner_RUN_ID } from "./fixtures/fake-bench.js";
import type { BenchEnv, RunSummary } from "@mcts/bench";

/** Create a seeded test store with a mocked bench env. */
function createTestStore(envOverrides?: Partial<BenchEnv>) {
  const env = createMockBenchEnv(envOverrides);
  const store = createStore<BenchState, BenchAction>(initialBenchState(), benchReducer, env);
  // Pre-fetch kinds and runs so the UI shows data immediately.
  store.dispatch({ tag: "kinds", action: { tag: "request" } });
  store.dispatch({ tag: "runs", action: { tag: "request" } });
  return { store, env };
}

afterEach(() => {
  cleanup();
});

describe("LaunchForm", () => {
  it("renders the form and selects a kind", async () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    // Should show "Launch New Run" heading
    expect(screen.getByText("Launch New Run")).toBeInTheDocument();

    // Should show the kind selector with "Round Robin" option
    const kindSelect = screen.getByLabelText("Run Kind") as HTMLSelectElement;
    expect(kindSelect).toBeInTheDocument();
    expect(kindSelect.value).toBe("");

    // Select "Round Robin"
    fireEvent.change(kindSelect, { target: { value: "round_robin" } });
    expect(kindSelect.value).toBe("round_robin");

    // Should now show the game selector with "druid"
    const gameSelect = screen.getByLabelText("Game") as HTMLSelectElement;
    expect(gameSelect).toBeInTheDocument();
    expect(gameSelect.value).toBe("druid");
  });

  it("shows strategy checkboxes when a game is selected", async () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    // Select kind (game auto-selects)
    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "round_robin" } });

    // Strategies should be visible
    expect(screen.getByText("Strong")).toBeInTheDocument();
    expect(screen.getByText("Master")).toBeInTheDocument();
    expect(screen.getByText("1s UCB1")).toBeInTheDocument();

    // Launch button should be disabled (no strategies selected yet)
    const launchBtn = screen.getByText("Launch") as HTMLButtonElement;
    expect(launchBtn.disabled).toBe(true);

    // Select two strategies
    fireEvent.click(screen.getByText("Strong"));
    fireEvent.click(screen.getByText("Master"));

    // Launch button should now be enabled
    expect(launchBtn.disabled).toBe(false);
  });
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

  it("keeps a starting modern tuner attempt available without calling it legacy", async () => {
    const modernAttempt: RunSummary = {
      run_id: "tuner-modern-attempt",
      kind: "tuner",
      game: "nim",
      project_id: null,
      experiment_id: null,
      label: null,
      git_sha: "abc1234",
      git_dirty: false,
      host: "testhost",
      pid: 42,
      started_at: "2026-01-01T00:00:00Z",
      ended_at: null,
      status: "running",
      match_count: 0,
      trial_count: 0,
      tuning_session_id: "session-modern",
    };
    const { store } = createTestStore({ listRuns: () => Effect.send([modernAttempt]) });
    render(() => <RunList store={store} />);

    await vi.waitFor(() => expect(store.getState()().runs.status).toBe("done"));
    expect(screen.getByText("Tuning attempt")).toBeInTheDocument();
    expect(screen.queryByText("Legacy tuner run")).not.toBeInTheDocument();
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

  it("keeps a modern tuner attempt focused on its logical session", async () => {
    const { store } = createTestStore();
    const Spectator: Component<{ runId: string; game: string; kind: string; live: boolean }> = (
      props,
    ) => (
      <div data-testid="spectator">
        {props.runId}:{props.kind}
      </div>
    );
    render(() => <RunDetailPanel store={store} Spectator={Spectator} />);
    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });

    await vi.waitFor(() =>
      expect(screen.getByRole("heading", { name: "Tuning attempt" })).toBeInTheDocument(),
    );
    expect(screen.queryByText("3 / 50 (6%) complete")).not.toBeInTheDocument();
    expect(screen.queryByText("Browse games")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open tuning session" }));
    expect(store.getState()().tuningNavigation.selection.sessionId).toBe("session-traffic-lights");
  });
});
