import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { createStore } from "@mcts/core";
import {
  benchReducer,
  initialBenchState,
  type BenchAction,
  type BenchState,
  LaunchForm,
  RunDetailPanel,
} from "@mcts/bench";
import { createMockBenchEnv, FAKE_tuner_RUN_ID } from "./fixtures/fake-bench.js";

afterEach(cleanup);

describe("tuner launch and physical-run diagnostics", () => {
  it("renders resolved tuner launch fields without retired baseline controls", () => {
    const store = createStore<BenchState, BenchAction>(
      initialBenchState(),
      benchReducer,
      createMockBenchEnv(),
    );
    store.dispatch({ tag: "tunerKinds", action: { tag: "request" } });
    render(() => <LaunchForm store={store} />);
    expect(screen.getByLabelText("Target trials")).toBeInTheDocument();
    expect(screen.getByLabelText(/Minimum pairs/)).toBeInTheDocument();
    expect(screen.queryByText(/Starting baseline panel/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Max rungs/)).not.toBeInTheDocument();
  });

  it("keeps a modern physical tuner run compact and links it to its session", async () => {
    const store = createStore<BenchState, BenchAction>(
      initialBenchState(),
      benchReducer,
      createMockBenchEnv(),
    );
    render(() => <RunDetailPanel store={store} />);
    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });
    expect(await screen.findByRole("heading", { name: "Tuning attempt" })).toBeInTheDocument();
    expect(screen.getByText(/Continue and analyze this work/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open tuning session" }));
    expect(store.getState()().tuningNavigation.selection.sessionId).toBe("session-traffic-lights");
    expect(screen.queryByRole("button", { name: "Resume" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
    expect(screen.queryByText("Run Detail")).not.toBeInTheDocument();
    expect(screen.queryByText("Log Tail")).not.toBeInTheDocument();
    expect(screen.getByText("Attempt diagnostics")).toBeInTheDocument();
    expect(screen.queryByText("Best score (mu − 3σ)")).not.toBeInTheDocument();
    expect(screen.queryByText("Copy as baseline config")).not.toBeInTheDocument();
    expect(document.querySelector("#tuner-cost-chart")).toBeNull();
    expect(document.querySelector("#tuner-trials-table")).toBeNull();
  });
});
