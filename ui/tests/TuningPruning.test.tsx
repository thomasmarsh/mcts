import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { createStore, type Store } from "@mcts/core";
import { benchReducer, initialBenchState, type BenchAction, type BenchState, type TuningAnalysisOverview, type TuningTrialDetail } from "@mcts/bench";
import { TuningPruningView } from "../packages/bench/src/tuning/TuningPruningView.js";
import { createMockBenchEnv } from "./fixtures/fake-bench.js";

const counts = { total: 12, queued: 0, running: 1, terminal: 11, completed: 4, failed: 2, pruned: 3, cancelled: 2 };
const point = (trial_id: string, trial_number: number, trial_status: string, resource: number, score: number, mu: number, sigma: number, outcome: string, reason: string, bracket_id: string | null, rung_resource: number | null) => ({
  trial_id, trial_number, trial_status, resource, score, rating: { mu, sigma }, outcome, reason, pruning_exempt: reason === "startup_exempt", bracket_id, rung_resource,
});
const overview: TuningAnalysisOverview = {
  schema_version: 1,
  policy: { resource: { min_pairs: 2, max_pairs: 8 }, rating: { model: "tm", score: "mu_minus_k_sigma", sigma_stop: null, conservative_k: 3 }, sampler: { kind: "tpe", seed: 4, deterministic: true, startup_trials: 2 }, pruning: { enabled: true, kind: "hyperband", reduction_factor: 3, startup_trials: 2 } },
  objective: { metric: "score", direction: "maximize", complete_trials_only: true },
  cursor: { session_sequence: 3 },
  coverage: { trials: counts, reports: 14, pairs: { total: 20, running: 2, complete: 16, failed: 2, unmatched_pool_revisions: 0 }, points: { total: 14, returned: 4, sampled: true } },
  bracket_resources: [
    { bracket_id: null, resource: 2, rung_resource: null, reports: 2, trials: 2 },
    { bracket_id: "bracket-2", resource: 2, rung_resource: 2, reports: 5, trials: 5 },
    { bracket_id: "bracket-10", resource: 10, rung_resource: 10, reports: 7, trials: 7 },
  ],
  decision_groups: [
    { outcome: "continue", reason: "below_min_pairs", pruning_exempt: false, reports: 1 },
    { outcome: "continue", reason: "startup_exempt", pruning_exempt: true, reports: 2 },
    { outcome: "continue", reason: "pruning_disabled", pruning_exempt: false, reports: 1 },
    { outcome: "continue", reason: "hyperband_keep", pruning_exempt: false, reports: 3 },
    { outcome: "prune", reason: "hyperband_prune", pruning_exempt: false, reports: 2 },
    { outcome: "complete", reason: "confidence", pruning_exempt: false, reports: 1 },
    { outcome: "complete", reason: "max_pairs", pruning_exempt: false, reports: 4 },
  ],
  points: [
    point("trial-1", 1, "pruned", 2, 8, 18, 3, "continue", "startup_exempt", "bracket-2", 2),
    point("trial-1", 1, "pruned", 10, 6, 19, 3, "prune", "hyperband_prune", "bracket-10", 10),
    point("trial-2", 2, "complete", 2, 9, 20, 2, "continue", "hyperband_keep", "bracket-2", 2),
    point("trial-2", 2, "complete", 10, 15, 24, 1, "complete", "max_pairs", "bracket-10", 10),
  ],
  best: null,
  pool_revisions: [],
};
const selectedDetail: TuningTrialDetail = {
  schema_version: 1,
  cursor: { session_sequence: 3 },
  trial: {
    trial_id: "outside-sample", trial_number: 9, attempt_id: "attempt", state: "complete", config: {}, score: 17, rating: { mu: 26, sigma: 1 }, reason: "confidence", failure: null, pairs: [],
    reports: [
      { completed_pairs: 2, rating: { mu: 21, sigma: 2 }, score: 12, score_formula_version: 1, conservative_k: 3, decision: { outcome: "continue", reason: "hyperband_keep", pruning_exempt: false, bracket_id: "bracket-2", rung_resource: 2 }, reported_at: "" },
      { completed_pairs: 10, rating: { mu: 26, sigma: 1 }, score: 17, score_formula_version: 1, conservative_k: 3, decision: { outcome: "complete", reason: "confidence", pruning_exempt: false, bracket_id: "bracket-10", rung_resource: 10 }, reported_at: "" },
    ],
  },
};

function setup(snapshot: TuningAnalysisOverview | null = overview): Store<BenchState, BenchAction> {
  const initial = initialBenchState();
  initial.tuningNavigation.tab = "pruning";
  initial.tuningNavigation.selection = { sessionId: "session", attemptId: null, trialId: "outside-sample", pairId: null, gameId: null };
  initial.tuningNavigation.overview = { status: snapshot ? "done" : "error", snapshot, error: snapshot ? null : "offline", generation: 1, sessionId: "session" };
  initial.tuningNavigation.trialDetails["outside-sample"] = { status: "done", snapshot: selectedDetail, error: null, generation: 1, sessionId: "session", trialId: "outside-sample" };
  return createStore(initial, benchReducer, createMockBenchEnv());
}

afterEach(cleanup);

describe("tuning Pruning view", () => {
  it("renders exact disjoint decision totals, numeric rungs, separate terminal totals, and sampled trajectories", () => {
    render(() => <TuningPruningView store={setup()} />);
    expect(screen.getByText(/Pruning cutoff \/ threshold:/)).toHaveTextContent("Not recorded");
    expect(screen.getByText(/Failure 2 and cancellation 2 remain separate/)).toBeInTheDocument();
    expect(screen.getByText(/Showing 4 sampled reports of 14 observed/)).toBeInTheDocument();
    expect(screen.getAllByRole("img")).toHaveLength(3);
    expect(screen.getAllByRole("table")).toHaveLength(3);
    expect(screen.getAllByTestId("pruning-trajectory")).toHaveLength(9);
    expect(screen.getByRole("cell", { name: "bracket-2" })).toBeInTheDocument();
    expect(screen.getAllByText("10").length).toBeGreaterThan(0);
    expect(screen.getByRole("cell", { name: "complete / confidence" })).toBeInTheDocument();
    expect(screen.getAllByText(/startup allowance exempted/)).toHaveLength(2);
  });

  it("filters the exact Trials read from a decision segment and selects trajectories with a keyboard", () => {
    const store = setup();
    render(() => <TuningPruningView store={store} />);
    fireEvent.click(screen.getByRole("button", { name: /Show 2 trials with pruned reason/ }));
    expect(store.getState()().tuningNavigation).toMatchObject({ tab: "trials", filters: { reason: "hyperband_prune", state: null } });
    store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "pruning" } });
    const trajectory = screen.getAllByTestId("pruning-trajectory").find((element) => element.getAttribute("data-trial-id") === "trial-1")!;
    fireEvent.keyDown(trajectory, { key: "Enter" });
    expect(store.getState()().tuningNavigation.selection.trialId).toBe("trial-1");
    expect(store.getState()().tuningNavigation.tab).toBe("pruning");
  });

  it("keeps selection and filters through refresh, and explains disabled, empty, and error evidence", () => {
    const store = setup();
    const first = render(() => <TuningPruningView store={store} />);
    store.dispatch({ tag: "tuningNavigation", action: { tag: "setTrialFilters", filters: { bracket: "bracket-10", reason: "max_pairs" } } });
    store.dispatch({ tag: "tuningNavigation", action: { tag: "overviewLoaded", sessionId: "session", generation: 1, overview } });
    expect(store.getState()().tuningNavigation).toMatchObject({ tab: "pruning", selection: { trialId: "outside-sample" }, filters: { bracket: "bracket-10", reason: "max_pairs" } });
    first.unmount();
    const disabled = { ...overview, policy: { ...overview.policy!, pruning: { ...overview.policy!.pruning, enabled: false } } };
    const second = render(() => <TuningPruningView store={setup(disabled)} />);
    expect(screen.getByText("Pruning was disabled by the recorded policy; continued reports are shown without claiming a pruning decision.")).toBeInTheDocument();
    second.unmount();
    const empty = { ...overview, coverage: { ...overview.coverage, reports: 0, points: { total: 0, returned: 0, sampled: false } }, points: [] };
    const third = render(() => <TuningPruningView store={setup(empty)} />);
    expect(screen.getByText(/Pruning evidence was not retained/)).toBeInTheDocument();
    third.unmount();
    render(() => <TuningPruningView store={setup(null)} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Could not load pruning evidence: offline");
  });
});
