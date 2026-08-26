import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { createStore, Effect } from "@mcts/core";
import {
  benchReducer,
  initialBenchState,
  type BenchAction,
  type BenchEnv,
  type BenchSpectatorProps,
  type BenchState,
  type TuningTrialDetail,
  type TuningTrialPage,
} from "@mcts/bench";
import { TuningSessionWorkbench } from "../packages/bench/src/tuning/TuningSessionWorkbench.js";

const counts = {
  total: 1,
  queued: 0,
  running: 0,
  terminal: 1,
  completed: 1,
  failed: 0,
  pruned: 0,
  cancelled: 0,
};
const capabilities = {
  has_lifecycle: true,
  has_pairs: true,
  has_renderer_trace: true,
  has_search_reports: true,
  has_trial_reports: true,
};
const session = {
  session_id: "session-a",
  game: "nim",
  label: "Observable tuning",
  status: "completed",
  target_trial_count: 1,
  counts,
  created_at: "2026-08-23T12:00:00Z",
  last_activity_at: "2026-08-23T12:10:00Z",
  attempts: [
    {
      attempt_id: "attempt-a",
      bench_run_id: "physical-a",
      status: "completed",
      started_at: "2026-08-23T12:00:00Z",
      ended_at: "2026-08-23T12:10:00Z",
      failure: null,
    },
  ],
  capabilities,
};
const overview = {
  schema_version: 1 as const,
  policy: null,
  objective: { metric: "score", direction: "max", complete_trials_only: true },
  cursor: { session_sequence: 1 },
  coverage: {
    trials: counts,
    reports: 1,
    pairs: { total: 1, running: 0, complete: 1, failed: 0, unmatched_pool_revisions: 0 },
    points: { total: 1, returned: 1, sampled: false },
  },
  bracket_resources: [],
  decision_groups: [],
  points: [
    {
      trial_id: "trial-a",
      trial_number: 1,
      trial_status: "complete",
      resource: 2,
      rating: { mu: 25, sigma: 1 },
      score: 22,
      outcome: "complete",
      reason: "max_pairs",
      pruning_exempt: false,
      bracket_id: null,
      rung_resource: null,
    },
  ],
  best: { score: 22, trial_ids: ["trial-a"] },
  pool_revisions: [],
};
const page: TuningTrialPage = {
  schema_version: 1,
  trials: [
    {
      trial_id: "trial-a",
      trial_number: 1,
      attempt_id: "attempt-a",
      state: "complete",
      reason: "max_pairs",
      rating: { mu: 25, sigma: 1 },
      score: 22,
      family: "ucb1",
      config_summary: null,
      bracket_id: null,
      resource: 2,
      pair_count: 1,
      wins: 2,
      losses: 0,
      draws: 0,
      elapsed_ms: 10,
      search_iterations_total: 20,
      search_move_time_ms: 3,
      has_detail: true,
    },
  ],
  total_count: 1,
  limit: 50,
  next_cursor: null,
  cursor: { session_sequence: 1 },
};
const trialDetail: TuningTrialDetail = {
  schema_version: 1,
  cursor: { session_sequence: 1 },
  trial: {
    trial_id: "trial-a",
    trial_number: 1,
    attempt_id: "attempt-a",
    state: "complete",
    config: { family: "ucb1" },
    score: 22,
    rating: { mu: 25, sigma: 1 },
    reason: "max_pairs",
    failure: null,
    reports: [],
    pairs: [
      {
        pair_id: "pair-a",
        pair_index: 0,
        state: "complete",
        seed: 7,
        round: 1,
        opponent: {
          anchor_id: "anchor",
          config: {},
          mu: 24,
          sigma: 1,
          label: "Anchor",
          provenance: "pool",
        },
        pool_snapshot_fingerprint: "pool",
        pool_revision: null,
        rating_before: { mu: 24, sigma: 2 },
        rating_after: { mu: 25, sigma: 1 },
        score: 22,
        failure: null,
        games: [
          {
            game_id: "game-a",
            candidate_side: "first",
            outcome: "candidate_win",
            seed: 7,
            round: 1,
            plies: 12,
            elapsed_ms: 30,
            candidate: { iterations_total: 4, iterations_first_half: 2, move_time_ms: 3 },
            baseline: { iterations_total: 4, iterations_first_half: 2, move_time_ms: 3 },
            replay: {
              run_id: "physical-a",
              game_seq: 41,
              has_renderer_trace: true,
              has_search_reports: true,
            },
          },
        ],
      },
    ],
  },
};

function setup() {
  let oldSessionCalls = 0;
  const pageQueries: unknown[] = [];
  const details: string[] = [];
  const env = {
    listTuningSessions: () => Effect.send({ schema_version: 1 as const, sessions: [session] }),
    getTuningSession: () => {
      oldSessionCalls += 1;
      return Effect.send(null as never);
    },
    getTuningAnalysisOverview: () => Effect.send(overview),
    getTuningTrialPage: (_sessionId: string, query: unknown) => {
      pageQueries.push(query);
      return Effect.send(page);
    },
    getTuningTrialDetail: (_sessionId: string, trialId: string) => {
      details.push(trialId);
      return Effect.send(trialDetail);
    },
  } as unknown as BenchEnv;
  const store = createStore<BenchState, BenchAction>(initialBenchState(), benchReducer, env);
  store.dispatch({ tag: "tuningNavigation", action: { tag: "listRequest" } });
  store.dispatch({
    tag: "tuningNavigation",
    action: { tag: "selectSession", sessionId: session.session_id },
  });
  return { store, details, pageQueries, oldSessionCalls: () => oldSessionCalls };
}

afterEach(cleanup);

describe("tuning session cutover", () => {
  it("keeps selection through all tabs and loads game evidence one trial at a time", async () => {
    const { store, details, pageQueries, oldSessionCalls } = setup();
    const seen: BenchSpectatorProps[] = [];
    const Spectator = (props: BenchSpectatorProps) => {
      seen.push(props);
      return <div data-testid="spectator" />;
    };
    render(() => <TuningSessionWorkbench store={store} Spectator={Spectator} />);
    await screen.findByText("Observable tuning");
    for (const name of ["Pruning", "Ladder", "Trials", "Game"] as const) {
      fireEvent.click(screen.getByRole("tab", { name }));
      await vi.waitFor(() =>
        expect(screen.getByRole("tab", { name })).toHaveAttribute("aria-selected", "true"),
      );
    }
    await screen.findByText("Trial #1 · complete · 1 pairs · 22.000");
    expect(oldSessionCalls()).toBe(0);
    expect(pageQueries).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Trial #1 · complete · 1 pairs · 22.000" }));
    await screen.findByRole("button", { name: /Pair 1 · complete · 1 games/ });
    expect(details).toEqual(["trial-a"]);
    fireEvent.click(screen.getByRole("button", { name: /Pair 1 · complete · 1 games/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Game 1 · candidate first/ }));
    expect(await screen.findByTestId("spectator")).toBeInTheDocument();
    expect(seen.at(-1)).toMatchObject({ runId: "physical-a", game: "nim", initialGameSeq: 41 });
    expect(store.getState()().tuningNavigation.selection).toMatchObject({
      sessionId: "session-a",
      trialId: "trial-a",
      pairId: "pair-a",
      gameId: "game-a",
    });
  });

  it("rejects a stale overview refresh without disturbing the selected game", () => {
    const { store } = setup();
    store.dispatch({ tag: "tuningNavigation", action: { tag: "selectTrial", trialId: "trial-a" } });
    store.dispatch({
      tag: "tuningNavigation",
      action: {
        tag: "overviewLoaded",
        sessionId: "session-a",
        generation: 0,
        overview: { ...overview, cursor: { session_sequence: 99 } },
      },
    });
    expect(store.getState()().tuningNavigation.selection.trialId).toBe("trial-a");
    expect(store.getState()().tuningNavigation.overview.snapshot?.cursor.session_sequence).toBe(1);
  });

  it("uses the legacy session response only when lifecycle evidence is unavailable", () => {
    let calls = 0;
    const legacy = {
      ...session,
      session_id: "legacy",
      capabilities: { ...capabilities, has_lifecycle: false },
    };
    const env = {
      listTuningSessions: () => Effect.send({ schema_version: 1 as const, sessions: [legacy] }),
      getTuningSession: () => {
        calls += 1;
        return Effect.send({
          schema_version: 1 as const,
          policy: null,
          summary: legacy,
          attempts: [],
          trials: [],
          manifest: {},
          fingerprint: null,
          capabilities: legacy.capabilities,
          cursor: { session_sequence: 1 },
        });
      },
      getTuningAnalysisOverview: () =>
        Effect.send({
          ...overview,
          coverage: {
            ...overview.coverage,
            reports: 0,
            points: { total: 0, returned: 0, sampled: false },
          },
        }),
    } as unknown as BenchEnv;
    const store = createStore<BenchState, BenchAction>(initialBenchState(), benchReducer, env);
    store.dispatch({ tag: "tuningNavigation", action: { tag: "listRequest" } });
    store.dispatch({
      tag: "tuningNavigation",
      action: { tag: "selectSession", sessionId: "legacy" },
    });
    expect(calls).toBe(1);
  });
});
