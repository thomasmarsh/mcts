import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { createStore, type Store } from "@mcts/core";
import {
  benchReducer,
  initialBenchState,
  type BenchAction,
  type BenchState,
  type TuningAnalysisOverview,
  type TuningTrialDetail,
} from "@mcts/bench";
import { TuningLadderView } from "../packages/bench/src/tuning/TuningLadderView.js";
import { createMockBenchEnv } from "./fixtures/fake-bench.js";

const revisions = [
  {
    pool_snapshot_fingerprint: "pool-3",
    display_ordinal: 3,
    observed_at: "2026-08-23T12:03:00Z",
    pair_count: 3,
    anchors: [
      {
        anchor_ordinal: 1,
        anchor_id: "anchor-a",
        config: { family: "rave", mcgs: true },
        rating: { mu: 25, sigma: 1 },
        provenance: "candidate",
        insertion_reason: "promotion",
        source_trial_id: "trial-old",
      },
      {
        anchor_ordinal: 2,
        anchor_id: "anchor-b",
        config: { family: "ucb" },
        rating: { mu: 16, sigma: 3 },
        provenance: "baseline",
        insertion_reason: "seed",
        source_trial_id: null,
      },
    ],
  },
  {
    pool_snapshot_fingerprint: "pool-1",
    display_ordinal: 1,
    observed_at: "2026-08-23T12:01:00Z",
    pair_count: 1,
    anchors: [
      {
        anchor_ordinal: 1,
        anchor_id: "anchor-a",
        config: { family: "rave", mcgs: true },
        rating: { mu: 20, sigma: 2 },
        provenance: "candidate",
        insertion_reason: "promotion",
        source_trial_id: "trial-seed",
      },
    ],
  },
];
const overview: TuningAnalysisOverview = {
  schema_version: 1,
  policy: null,
  objective: { metric: "score", direction: "maximize", complete_trials_only: true },
  cursor: { session_sequence: 4 },
  coverage: {
    trials: {
      total: 3,
      queued: 0,
      running: 0,
      terminal: 3,
      completed: 3,
      failed: 0,
      pruned: 0,
      cancelled: 0,
    },
    reports: 4,
    pairs: { total: 3, running: 0, complete: 3, failed: 0, unmatched_pool_revisions: 1 },
    points: { total: 4, returned: 4, sampled: false },
  },
  bracket_resources: [],
  decision_groups: [],
  points: [],
  best: null,
  pool_revisions: revisions,
};
const detail: TuningTrialDetail = {
  schema_version: 1,
  cursor: { session_sequence: 4 },
  trial: {
    trial_id: "trial-selected",
    trial_number: 8,
    attempt_id: "attempt",
    state: "complete",
    config: { family: "rave" },
    score: 18,
    rating: { mu: 26, sigma: 1 },
    reason: "max_pairs",
    failure: null,
    reports: [
      {
        completed_pairs: 2,
        rating: { mu: 21, sigma: 2 },
        score: 15,
        score_formula_version: 1,
        conservative_k: 3,
        decision: {
          outcome: "continue",
          reason: "hyperband_keep",
          pruning_exempt: false,
          bracket_id: null,
          rung_resource: null,
        },
        reported_at: "",
      },
      {
        completed_pairs: 4,
        rating: { mu: 26, sigma: 1 },
        score: 18,
        score_formula_version: 1,
        conservative_k: 3,
        decision: {
          outcome: "complete",
          reason: "max_pairs",
          pruning_exempt: false,
          bracket_id: null,
          rung_resource: null,
        },
        reported_at: "",
      },
    ],
    pairs: [
      {
        pair_id: "pair-1",
        pair_index: 0,
        state: "complete",
        seed: 1,
        round: 1,
        opponent: {
          anchor_id: "anchor-a",
          config: { family: "rave" },
          mu: 25,
          sigma: 1,
          label: "A",
          provenance: "candidate",
        },
        pool_snapshot_fingerprint: "pool-3",
        pool_revision: revisions[0]!,
        rating_before: { mu: 21, sigma: 2 },
        rating_after: { mu: 23, sigma: 1.5 },
        score: 1,
        failure: null,
        games: [],
      },
      {
        pair_id: "pair-2",
        pair_index: 1,
        state: "complete",
        seed: 2,
        round: 1,
        opponent: {
          anchor_id: "legacy-anchor",
          config: { family: "ucb" },
          mu: 18,
          sigma: 2,
          label: null,
          provenance: null,
        },
        pool_snapshot_fingerprint: "gone",
        pool_revision: null,
        rating_before: { mu: 23, sigma: 1.5 },
        rating_after: { mu: 26, sigma: 1 },
        score: 1,
        failure: null,
        games: [],
      },
    ],
  },
};

function setup(snapshot: TuningAnalysisOverview | null = overview): Store<BenchState, BenchAction> {
  const initial = initialBenchState();
  initial.tuningNavigation.tab = "ladder";
  initial.tuningNavigation.selection = {
    sessionId: "session",
    attemptId: null,
    trialId: "trial-selected",
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
  initial.tuningNavigation.detail = {
    status: "done",
    snapshot: {
      fingerprint: "session-fingerprint",
      summary: { status: "idle" },
    } as BenchState["tuningNavigation"]["detail"]["snapshot"],
    error: null,
    generation: 1,
    sessionId: "session",
  };
  initial.tuningNavigation.trialDetails["trial-selected"] = {
    status: "done",
    snapshot: detail,
    error: null,
    generation: 1,
    sessionId: "session",
    trialId: "trial-selected",
  };
  return createStore(initial, benchReducer, createMockBenchEnv());
}

afterEach(cleanup);

describe("tuning Ladder view", () => {
  it("renders ordered immutable snapshots, intervals, history, exact joins, and session-local labeling", () => {
    render(() => <TuningLadderView store={setup()} />);
    expect(screen.getByText(/Ratings are session-local/)).toHaveTextContent("session-fingerprint");
    expect(screen.getAllByRole("img")).toHaveLength(1);
    expect(screen.getAllByTestId("ladder-anchor")).toHaveLength(3);
    expect(screen.getAllByRole("table")).toHaveLength(2);
    expect(screen.getByText(/2 stored revisions/)).toBeInTheDocument();
    expect(screen.getAllByText(/snapshots 1, 3/)).toHaveLength(2);
    expect(screen.getByText("3 · pool-3")).toBeInTheDocument();
    expect(screen.getByText(/immutable pool revision was not retained/)).toBeInTheDocument();
  });

  it("selects immutable anchors with mouse and keyboard, and retains reducer-owned selectors on refresh", async () => {
    const store = setup();
    render(() => <TuningLadderView store={store} />);
    fireEvent.click(screen.getByRole("button", { name: "Select anchor anchor-a revision 3" }));
    expect(store.getState()().tuningNavigation.ladderAnchorKey).toBe("pool-3:anchor-a");
    await vi.waitFor(() =>
      expect(screen.getByRole("button", { name: "Copy opponent preset" })).toBeInTheDocument(),
    );
    fireEvent.change(screen.getByLabelText("Ladder revision"), { target: { value: "3" } });
    expect(store.getState()().tuningNavigation.ladderRevision).toBe(3);
    await vi.waitFor(() => expect(screen.getAllByTestId("ladder-anchor")).toHaveLength(2));
    const anchor = screen
      .getAllByTestId("ladder-anchor")
      .find((element) => element.getAttribute("data-anchor-key") === "pool-3:anchor-a")!;
    fireEvent.keyDown(anchor, { key: "Enter" });
    expect(store.getState()().tuningNavigation.ladderAnchorKey).toBe("pool-3:anchor-a");
    store.dispatch({
      tag: "tuningNavigation",
      action: { tag: "overviewLoaded", sessionId: "session", generation: 1, overview },
    });
    expect(store.getState()().tuningNavigation).toMatchObject({
      tab: "ladder",
      ladderRevision: 3,
      ladderAnchorKey: "pool-3:anchor-a",
      selection: { trialId: "trial-selected" },
    });
  });

  it("explains missing immutable revisions and load failures without substituting a current pool", () => {
    const empty = {
      ...overview,
      pool_revisions: [],
      coverage: {
        ...overview.coverage,
        pairs: { ...overview.coverage.pairs, unmatched_pool_revisions: 2 },
      },
    };
    const first = render(() => <TuningLadderView store={setup(empty)} />);
    expect(screen.getByText(/current or newest pool is not substituted/)).toBeInTheDocument();
    first.unmount();
    render(() => <TuningLadderView store={setup(null)} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Could not load ladder evidence: offline");
  });
});
