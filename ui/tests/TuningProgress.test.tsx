import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { createStore, type Store } from "@mcts/core";
import {
  benchReducer,
  initialBenchState,
  type BenchAction,
  type BenchEnv,
  type BenchState,
  type TuningAnalysisOverview,
  type TuningTrialDetail,
} from "@mcts/bench";
import { TuningProgressView } from "../packages/bench/src/tuning/TuningProgressView.js";
import { createMockBenchEnv } from "./fixtures/fake-bench.js";

const counts = {
  total: 7,
  queued: 1,
  running: 1,
  terminal: 5,
  completed: 2,
  failed: 1,
  pruned: 1,
  cancelled: 1,
};
const point = (
  trial_id: string,
  trial_number: number,
  trial_status: string,
  bracket_id: string | null,
  resource: number,
  score: number,
  mu: number,
  sigma: number,
  reason = "max_pairs",
) => ({
  trial_id,
  trial_number,
  trial_status,
  bracket_id,
  resource,
  score,
  rating: { mu, sigma },
  outcome:
    trial_status === "pruned" ? "prune" : trial_status === "complete" ? "complete" : "continue",
  reason,
  pruning_exempt: false,
  rung_resource: bracket_id === null ? null : resource,
});

const overview: TuningAnalysisOverview = {
  schema_version: 1,
  policy: null,
  objective: { metric: "score", direction: "maximize", complete_trials_only: true },
  cursor: { session_sequence: 2 },
  coverage: {
    trials: counts,
    reports: 9,
    pairs: { total: 12, running: 1, complete: 9, failed: 2, unmatched_pool_revisions: 0 },
    points: { total: 9, returned: 7, sampled: true },
  },
  bracket_resources: [
    { bracket_id: null, resource: 3, rung_resource: null, reports: 1, trials: 1 },
    { bracket_id: "alpha", resource: 1, rung_resource: 1, reports: 1, trials: 1 },
    { bracket_id: "alpha", resource: 2, rung_resource: 2, reports: 1, trials: 1 },
    { bracket_id: "alpha", resource: 4, rung_resource: 4, reports: 1, trials: 1 },
    { bracket_id: "beta", resource: 2, rung_resource: 2, reports: 1, trials: 1 },
    { bracket_id: "beta", resource: 4, rung_resource: 4, reports: 1, trials: 1 },
    { bracket_id: "gamma", resource: 2, rung_resource: 2, reports: 1, trials: 1 },
    { bracket_id: "gamma", resource: 4, rung_resource: 4, reports: 1, trials: 1 },
  ],
  decision_groups: [],
  points: [
    point("complete", 1, "complete", "alpha", 2, 8, 21, 1),
    point("running", 2, "running", "alpha", 4, 5, 18, 3, "in_progress"),
    point("pruned", 3, "pruned", "beta", 2, 3, 12, 4, "hyperband_prune"),
    point("failed", 4, "failed", null, 3, -2, 8, 5, "worker_failed"),
    point("queued", 5, "queued", "alpha", 1, 0, 15, 6, "pending"),
    point("cancelled", 6, "cancelled", "beta", 4, 1, 13, 5, "operator_cancelled"),
    point("tie", 7, "complete", "alpha", 4, 8, 22, 1),
  ],
  best: { score: 8, trial_ids: ["complete", "tie"] },
  pool_revisions: [],
};

const selectedDetail: TuningTrialDetail = {
  schema_version: 1,
  cursor: { session_sequence: 2 },
  trial: {
    trial_id: "outside-sample",
    trial_number: 8,
    attempt_id: "attempt",
    state: "complete",
    config: {},
    score: 7,
    rating: { mu: 20, sigma: 2 },
    reason: "max_pairs",
    failure: null,
    pairs: [],
    reports: [
      {
        completed_pairs: 2,
        rating: { mu: 16, sigma: 3 },
        score: 2,
        score_formula_version: 1,
        conservative_k: 3,
        decision: {
          outcome: "continue",
          reason: "startup_exempt",
          pruning_exempt: true,
          bracket_id: "gamma",
          rung_resource: 2,
        },
        reported_at: "",
      },
      {
        completed_pairs: 4,
        rating: { mu: 20, sigma: 2 },
        score: 7,
        score_formula_version: 1,
        conservative_k: 3,
        decision: {
          outcome: "complete",
          reason: "max_pairs",
          pruning_exempt: false,
          bracket_id: "gamma",
          rung_resource: 4,
        },
        reported_at: "",
      },
    ],
  },
};

function setup(snapshot: TuningAnalysisOverview | null = overview): Store<BenchState, BenchAction> {
  const env: BenchEnv = createMockBenchEnv();
  const initial = initialBenchState();
  initial.tuningNavigation.selection = {
    sessionId: "session",
    attemptId: null,
    trialId: "outside-sample",
    pairId: null,
    gameId: null,
  };
  initial.tuningNavigation.overview = {
    status: snapshot ? "done" : "error",
    snapshot,
    error: snapshot ? null : "offline",
    generation: 1,
    sessionId: "session",
  };
  initial.tuningNavigation.trialDetails["outside-sample"] = {
    status: "done",
    snapshot: selectedDetail,
    error: null,
    generation: 1,
    sessionId: "session",
    trialId: "outside-sample",
  };
  return createStore<BenchState, BenchAction>(initial, benchReducer, env);
}

afterEach(cleanup);

describe("tuning Progress view", () => {
  it("renders ordered bracket facets, terminal markers, rung guides, tables, and sampled disclosure", () => {
    render(() => <TuningProgressView store={setup()} />);
    expect(screen.getByText(/Showing 7 sampled reports of 9 observed/)).toBeInTheDocument();
    expect(
      screen.getByText(/Exact full-population counts: 7 trials and 9 reports/),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("img")).toHaveLength(4);
    expect(screen.getAllByRole("table")).toHaveLength(4);
    expect(screen.getAllByTestId("progress-mark")).toHaveLength(9);
    expect(
      screen.getAllByRole("heading", { level: 5 }).map((heading) => heading.textContent),
    ).toEqual(["Unassigned", "alpha", "beta", "gamma"]);
    expect(screen.getAllByRole("cell", { name: /best complete/ })).toHaveLength(2);
    expect(screen.getByTestId("progress-selected-path")).toBeInTheDocument();
    expect(screen.getAllByText("completed pairs")).toHaveLength(4);
  });

  it("selects a stable trial with mouse and keyboard without changing the active tab", () => {
    const store = setup();
    render(() => <TuningProgressView store={store} />);
    const mark = screen
      .getAllByTestId("progress-mark")
      .find((element) => element.getAttribute("data-trial-id") === "complete")!;
    fireEvent.keyDown(mark, { key: "Enter" });
    expect(store.getState()().tuningNavigation.selection.trialId).toBe("complete");
    expect(store.getState()().tuningNavigation.tab).toBe("progress");
    fireEvent.click(screen.getByRole("button", { name: "Select trial 7" }));
    expect(store.getState()().tuningNavigation.selection.trialId).toBe("tie");
  });

  it("filters facets and keeps metric and scale controls stable through a progress refresh", async () => {
    const store = setup();
    render(() => <TuningProgressView store={store} />);
    fireEvent.change(screen.getByLabelText("Progress bracket"), { target: { value: "beta" } });
    fireEvent.change(screen.getByLabelText("Progress state"), { target: { value: "pruned" } });
    fireEvent.change(screen.getByLabelText("Progress metric"), { target: { value: "sigma" } });
    fireEvent.change(screen.getByLabelText("Progress Y scale"), { target: { value: "local" } });
    fireEvent.input(screen.getByLabelText("Progress family"), { target: { value: "rave" } });
    expect(store.getState()().tuningNavigation).toMatchObject({
      progressMetric: "sigma",
      progressScale: "local",
      filters: { bracket: "beta", state: "pruned", family: "rave" },
    });
    await vi.waitFor(() => expect(screen.getAllByRole("img")).toHaveLength(1));
    expect(screen.getAllByTestId("progress-mark")).toHaveLength(1);
    store.dispatch({
      tag: "tuningNavigation",
      action: { tag: "overviewLoaded", sessionId: "session", generation: 1, overview },
    });
    expect(store.getState()().tuningNavigation).toMatchObject({
      progressMetric: "sigma",
      progressScale: "local",
      filters: { bracket: "beta", state: "pruned", family: "rave" },
    });
    expect(screen.getByText(/Y scale: local to each bracket/)).toBeInTheDocument();
  });

  it("explains legacy, empty, and error progress states without falling back to the old chart", () => {
    const legacy = {
      ...overview,
      coverage: {
        ...overview.coverage,
        reports: 0,
        points: { total: 0, returned: 0, sampled: false },
      },
      points: [],
    };
    const { unmount } = render(() => <TuningProgressView store={setup(legacy)} />);
    expect(screen.getByText(/Progress evidence was not retained/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open Trials" })).toBeInTheDocument();
    unmount();
    render(() => <TuningProgressView store={setup(null)} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Could not load progress: offline");
  });
});
