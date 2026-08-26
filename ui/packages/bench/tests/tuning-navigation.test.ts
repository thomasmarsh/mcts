import { describe, expect, it } from "vitest";
import { Effect } from "@mcts/core";
import { createTestStore } from "../../../tests/test-store.js";
import {
  TUNING_DETAIL_REFRESH_MS,
  initialTuningNavigationState,
  tuningNavigationReducer,
  type TuningNavigationAction,
  type TuningNavigationState,
} from "../src/tuning-navigation.js";
import type { BenchEnv } from "../src/reducer.js";
import type {
  TuningAnalysisOverview,
  TuningSessionsResponse,
  TuningTrialDetail,
  TuningTrialPage,
} from "../src/types.js";

const control = {
  version: 0,
  continuation: {
    target_trial_count: 2,
    consumed_trial_count: 1,
    remaining_trial_count: 1,
    active_attempt_id: "attempt-1",
    launch_reservation: null,
    stop_attempt_id: null,
    recovery_required: false,
  },
  allowed_commands: [
    { command: "stop" as const, allowed: true, denial_reason: null },
    { command: "resume" as const, allowed: false, denial_reason: "active_attempt" },
    { command: "add_budget" as const, allowed: true, denial_reason: null },
  ],
};

const sessions: TuningSessionsResponse = {
  schema_version: 1,
  sessions: [
    {
      session_id: "session-1",
      game: "nim",
      label: null,
      status: "active",
      target_trial_count: 2,
      counts: {
        total: 1,
        queued: 0,
        running: 1,
        terminal: 0,
        completed: 0,
        failed: 0,
        pruned: 0,
        cancelled: 0,
      },
      created_at: "2026-08-23T12:00:00Z",
      last_activity_at: "2026-08-23T12:01:00Z",
      attempts: [],
      capabilities: {
        has_lifecycle: true,
        has_pairs: true,
        has_renderer_trace: true,
        has_search_reports: false,
        has_trial_reports: true,
      },
      control,
    },
  ],
};
const overview = (sequence = 1): TuningAnalysisOverview => ({
  schema_version: 1,
  policy: null,
  objective: { metric: "score", direction: "maximize", complete_trials_only: true },
  cursor: { session_sequence: sequence },
  coverage: {
    trials: sessions.sessions[0]!.counts,
    reports: 0,
    pairs: { total: 0, running: 0, complete: 0, failed: 0, unmatched_pool_revisions: 0 },
    points: { total: 0, returned: 0, sampled: false },
  },
  bracket_resources: [],
  decision_groups: [],
  points: [],
  best: null,
  pool_revisions: [],
  control,
});
const page = (cursor: number, next_cursor: string | null = null): TuningTrialPage => ({
  schema_version: 1,
  trials: [],
  total_count: 0,
  limit: 50,
  next_cursor,
  cursor: { session_sequence: cursor },
});
const detail = (
  trialId = "trial-1",
  pairs: TuningTrialDetail["trial"]["pairs"] = [],
): TuningTrialDetail => ({
  schema_version: 1,
  trial: {
    trial_id: trialId,
    trial_number: 1,
    attempt_id: "attempt-1",
    state: "complete",
    config: {},
    score: 1,
    rating: { mu: 2, sigma: 1 },
    reason: "max_pairs",
    failure: null,
    reports: [],
    pairs,
  },
  cursor: { session_sequence: 1 },
});
function env(overrides: Partial<BenchEnv> = {}): BenchEnv {
  return {
    listTuningSessions: () => Effect.none(),
    getTuningSession: () => Effect.none(),
    getTuningAnalysisOverview: () => Effect.none(),
    getTuningTrialPage: () => Effect.none(),
    getTuningTrialDetail: () => Effect.none(),
    ...overrides,
  } as unknown as BenchEnv;
}
const reducer = (
  state: TuningNavigationState,
  action: TuningNavigationAction,
  environment: BenchEnv,
) => tuningNavigationReducer(state, action, environment);

describe("tuningNavigationReducer analysis state", () => {
  it("defaults to Progress, and capable sessions open it without changing evidence selection", () => {
    expect(initialTuningNavigationState()).toMatchObject({
      tab: "progress",
      progressMetric: "score",
      progressScale: "shared",
      filters: { state: null, bracket: null, reason: null, family: null, q: null },
      sort: { sort: "trial", direction: "desc" },
      trialPageLimit: 50,
    });
    const initial = initialTuningNavigationState();
    initial.list.snapshot = sessions;
    const ts = createTestStore(reducer, env(), initial);
    ts.send({ tag: "selectSession", sessionId: "session-1" }, (state) => {
      state.tab = "progress";
      state.selection = {
        sessionId: "session-1",
        attemptId: null,
        trialId: null,
        pairId: null,
        gameId: null,
      };
      // Capable sessions begin with compact analysis evidence; the retired
      // whole-session detail read remains idle until legacy fallback needs it.
      state.detail.status = "idle";
      state.detail.sessionId = null;
      state.detail.generation = 0;
      state.overview.status = "loading";
      state.overview.sessionId = "session-1";
      state.overview.generation = 2;
      state.trialPage.generation = 1;
    });
  });

  it("drops stale overview, page, and keyed-detail responses", () => {
    const initial = initialTuningNavigationState();
    initial.selection.sessionId = "session-1";
    initial.tab = "trials";
    const ts = createTestStore(reducer, env(), initial);
    ts.send({ tag: "overviewRequest", sessionId: "session-1" }, (s) => {
      s.overview.status = "loading";
      s.overview.sessionId = "session-1";
      s.overview.generation = 1;
    });
    ts.send({ tag: "overviewRequest", sessionId: "session-1" }, (s) => {
      s.overview.generation = 2;
    });
    ts.send({
      tag: "overviewLoaded",
      sessionId: "session-1",
      generation: 1,
      overview: overview(1),
    });
    ts.send({ tag: "trialPageRequest", sessionId: "session-1" }, (s) => {
      s.trialPage.status = "loading";
      s.trialPage.sessionId = "session-1";
      s.trialPage.generation = 1;
      s.trialPage.queryKey = JSON.stringify({
        state: null,
        bracket: null,
        reason: null,
        family: null,
        q: null,
        sort: "trial",
        direction: "desc",
        limit: 50,
        cursor: null,
      });
    });
    ts.send({ tag: "trialPageRequest", sessionId: "session-1" }, (s) => {
      s.trialPage.generation = 2;
    });
    ts.send({
      tag: "trialPageLoaded",
      sessionId: "session-1",
      generation: 1,
      queryKey: ts.getState().trialPage.queryKey!,
      page: page(1),
    });
    ts.send({ tag: "selectTrial", trialId: "trial-1" }, (s) => {
      s.selection.trialId = "trial-1";
    });
    ts.send({ tag: "trialDetailRequest", sessionId: "session-1", trialId: "trial-1" }, (s) => {
      s.trialDetails["trial-1"] = {
        status: "loading",
        snapshot: null,
        error: null,
        generation: 1,
        sessionId: "session-1",
        trialId: "trial-1",
      };
    });
    ts.send({
      tag: "trialDetailLoaded",
      sessionId: "session-1",
      trialId: "trial-1",
      generation: 0,
      detail: detail(),
    });
  });

  it("invalidates page identity on rapid session, filter, and tab changes without changing selection", () => {
    const initial = initialTuningNavigationState();
    initial.selection = {
      sessionId: "session-1",
      attemptId: "a",
      trialId: "trial-1",
      pairId: null,
      gameId: null,
    };
    initial.tab = "trials";
    initial.trialPage = {
      ...initial.trialPage,
      status: "done",
      snapshot: page(1),
      sessionId: "session-1",
      queryKey: "old",
      generation: 4,
    };
    const ts = createTestStore(reducer, env(), initial);
    ts.send({ tag: "setTrialFilters", filters: { state: "complete", q: "rave" } }, (s) => {
      s.filters.state = "complete";
      s.filters.q = "rave";
      s.trialPage = {
        ...initialTuningNavigationState().trialPage,
        generation: 5,
        sessionId: "session-1",
      };
      s.trialPage.status = "loading";
      s.trialPage.generation = 6;
      s.trialPage.queryKey = JSON.stringify({
        state: "complete",
        bracket: null,
        reason: null,
        family: null,
        q: "rave",
        sort: "trial",
        direction: "desc",
        limit: 50,
        cursor: null,
      });
    });
    ts.send({ tag: "setAnalysisTab", tab: "game" }, (s) => {
      s.tab = "game";
    });
    expect(ts.getState().selection).toEqual({
      sessionId: "session-1",
      attemptId: "a",
      trialId: "trial-1",
      pairId: null,
      gameId: null,
    });
  });

  it("keeps ladder revision and immutable anchor selection user-owned while lazily loading its selected trial", () => {
    let details = 0;
    const initial = initialTuningNavigationState();
    initial.selection = {
      sessionId: "session-1",
      attemptId: null,
      trialId: "trial-1",
      pairId: null,
      gameId: null,
    };
    const ts = createTestStore(
      reducer,
      env({
        getTuningTrialDetail: () => {
          details += 1;
          return Effect.none();
        },
      }),
      initial,
    );
    ts.send({ tag: "setAnalysisTab", tab: "ladder" }, (s) => {
      s.tab = "ladder";
      s.trialDetails["trial-1"] = {
        status: "loading",
        snapshot: null,
        error: null,
        generation: 1,
        sessionId: "session-1",
        trialId: "trial-1",
      };
    });
    ts.send({ tag: "setLadderRevision", revision: 3 }, (s) => {
      s.ladderRevision = 3;
    });
    ts.send({ tag: "selectLadderAnchor", anchorKey: "pool-3:anchor-a" }, (s) => {
      s.ladderAnchorKey = "pool-3:anchor-a";
    });
    expect(details).toBe(1);
    expect(ts.getState()).toMatchObject({
      tab: "ladder",
      ladderRevision: 3,
      ladderAnchorKey: "pool-3:anchor-a",
      selection: { trialId: "trial-1" },
    });
  });

  it("invalidates in-flight analysis when a newer session wins a rapid switch", () => {
    const initial = initialTuningNavigationState();
    initial.list.snapshot = {
      ...sessions,
      sessions: [...sessions.sessions, { ...sessions.sessions[0]!, session_id: "session-2" }],
    };
    const ts = createTestStore(reducer, env(), initial);
    ts.send({ tag: "selectSession", sessionId: "session-1" });
    ts.send({ tag: "selectSession", sessionId: "session-2" }, (s) => {
      s.selection = {
        sessionId: "session-2",
        attemptId: null,
        trialId: null,
        pairId: null,
        gameId: null,
      };
      s.tab = "progress";
      s.detail.sessionId = null;
      s.detail.generation = 0;
      s.overview.sessionId = "session-2";
      s.overview.generation = 4;
      s.trialPage.generation = 2;
    });
    ts.send({
      tag: "overviewLoaded",
      sessionId: "session-1",
      generation: 2,
      overview: overview(9),
    });
  });

  it("refreshes the visible page and selected detail only after an overview cursor advances", () => {
    let pages = 0;
    let details = 0;
    const initial = initialTuningNavigationState();
    initial.tab = "trials";
    initial.selection = {
      sessionId: "session-1",
      attemptId: null,
      trialId: "trial-1",
      pairId: null,
      gameId: null,
    };
    initial.list.snapshot = sessions;
    initial.overview = {
      status: "loading",
      snapshot: overview(1),
      error: null,
      generation: 1,
      sessionId: "session-1",
    };
    initial.trialPage = {
      ...initial.trialPage,
      status: "done",
      snapshot: page(1),
      sessionId: "session-1",
      queryKey: JSON.stringify({
        state: null,
        bracket: null,
        reason: null,
        family: null,
        q: null,
        sort: "trial",
        direction: "desc",
        limit: 50,
        cursor: null,
      }),
      generation: 1,
    };
    const ts = createTestStore(
      reducer,
      env({
        getTuningTrialPage: () => {
          pages += 1;
          return Effect.none();
        },
        getTuningTrialDetail: () => {
          details += 1;
          return Effect.none();
        },
      }),
      initial,
    );
    ts.send(
      { tag: "overviewLoaded", sessionId: "session-1", generation: 1, overview: overview(2) },
      (s) => {
        s.overview.status = "done";
        s.overview.snapshot = overview(2);
        s.trialPage.status = "loading";
        s.trialPage.generation = 2;
        s.trialPage.queryKey = JSON.stringify({
          state: null,
          bracket: null,
          reason: null,
          family: null,
          q: null,
          sort: "trial",
          direction: "desc",
          limit: 50,
          cursor: null,
        });
        s.trialDetails["trial-1"] = {
          status: "loading",
          snapshot: null,
          error: null,
          generation: 1,
          sessionId: "session-1",
          trialId: "trial-1",
        };
      },
    );
    expect({ pages, details }).toEqual({ pages: 1, details: 1 });
    ts.advance(TUNING_DETAIL_REFRESH_MS);
    ts.receive({ tag: "overviewRefreshTick", sessionId: "session-1", generation: 1 }, (s) => {
      s.overview.status = "loading";
      s.overview.generation = 2;
    });
  });

  it("does not poll an already terminal session", () => {
    const initial = initialTuningNavigationState();
    initial.selection.sessionId = "session-1";
    initial.list.snapshot = {
      ...sessions,
      sessions: [{ ...sessions.sessions[0]!, status: "completed" }],
    };
    initial.tab = "game";
    initial.overview = {
      status: "loading",
      snapshot: null,
      error: null,
      generation: 1,
      sessionId: "session-1",
    };
    const ts = createTestStore(reducer, env(), initial);
    ts.send(
      { tag: "overviewLoaded", sessionId: "session-1", generation: 1, overview: overview(1) },
      (s) => {
        s.overview.status = "done";
        s.overview.snapshot = overview(1);
      },
    );
    expect(ts.scheduler.pendingCount).toBe(0);
  });

  it("retains successful snapshots while newer overview, page, and detail requests fail", () => {
    const initial = initialTuningNavigationState();
    initial.selection.sessionId = "session-1";
    initial.overview = {
      status: "done",
      snapshot: overview(1),
      error: null,
      generation: 0,
      sessionId: "session-1",
    };
    initial.trialPage = {
      ...initial.trialPage,
      status: "done",
      snapshot: page(1),
      sessionId: "session-1",
      queryKey: JSON.stringify({
        state: null,
        bracket: null,
        reason: null,
        family: null,
        q: null,
        sort: "trial",
        direction: "desc",
        limit: 50,
        cursor: null,
      }),
    };
    initial.trialDetails["trial-1"] = {
      status: "done",
      snapshot: detail(),
      error: null,
      generation: 0,
      sessionId: "session-1",
      trialId: "trial-1",
    };
    const ts = createTestStore(reducer, env(), initial);
    ts.send({ tag: "overviewRequest", sessionId: "session-1" }, (s) => {
      s.overview.status = "loading";
      s.overview.generation = 1;
    });
    ts.send(
      { tag: "overviewFailed", sessionId: "session-1", generation: 1, error: "offline" },
      (s) => {
        s.overview.status = "error";
        s.overview.error = "offline";
      },
    );
    ts.send({ tag: "trialPageRequest", sessionId: "session-1" }, (s) => {
      s.trialPage.status = "loading";
      s.trialPage.generation = 1;
    });
    ts.send(
      {
        tag: "trialPageFailed",
        sessionId: "session-1",
        generation: 1,
        queryKey: sQuery(ts),
        error: "offline",
      },
      (s) => {
        s.trialPage.status = "error";
        s.trialPage.error = "offline";
      },
    );
    ts.send({ tag: "trialDetailRequest", sessionId: "session-1", trialId: "trial-1" });
  });

  it("keeps expansion and selection stable, pages forward and back, and reconciles missing child detail to its trial", () => {
    const initial = initialTuningNavigationState();
    initial.selection = {
      sessionId: "session-1",
      attemptId: "attempt-1",
      trialId: "trial-1",
      pairId: "pair-old",
      gameId: "game-old",
    };
    initial.tab = "trials";
    initial.trialPage = {
      ...initial.trialPage,
      status: "done",
      snapshot: page(1, "next"),
      sessionId: "session-1",
      queryKey: "old",
      generation: 1,
    };
    const seen: (string | null | undefined)[] = [];
    const ts = createTestStore(
      reducer,
      env({
        getTuningTrialPage: (_id, query) => {
          seen.push(query?.cursor);
          return Effect.none();
        },
      }),
      initial,
    );
    ts.send({ tag: "toggleExpanded", id: "trial:trial-1" }, (s) => {
      s.expandedIds = ["trial:trial-1"];
    });
    ts.send({ tag: "nextTrialPage" }, (s) => {
      s.trialPage.previousCursors = [null];
      s.trialPage.cursor = "next";
      s.trialPage.status = "loading";
      s.trialPage.snapshot = null;
      s.trialPage.generation = 2;
      s.trialPage.queryKey = JSON.stringify({
        state: null,
        bracket: null,
        reason: null,
        family: null,
        q: null,
        sort: "trial",
        direction: "desc",
        limit: 50,
        cursor: "next",
      });
    });
    ts.send({ tag: "previousTrialPage" }, (s) => {
      s.trialPage.previousCursors = [];
      s.trialPage.cursor = null;
      s.trialPage.status = "loading";
      s.trialPage.generation = 3;
      s.trialPage.queryKey = JSON.stringify({
        state: null,
        bracket: null,
        reason: null,
        family: null,
        q: null,
        sort: "trial",
        direction: "desc",
        limit: 50,
        cursor: null,
      });
    });
    expect(seen).toEqual(["next", null]);
    ts.send({ tag: "trialDetailRequest", sessionId: "session-1", trialId: "trial-1" }, (s) => {
      s.trialDetails["trial-1"] = {
        status: "loading",
        snapshot: null,
        error: null,
        generation: 1,
        sessionId: "session-1",
        trialId: "trial-1",
      };
    });
    ts.send(
      {
        tag: "trialDetailLoaded",
        sessionId: "session-1",
        trialId: "trial-1",
        generation: 1,
        detail: detail(),
      },
      (s) => {
        s.trialDetails["trial-1"]!.status = "done";
        s.trialDetails["trial-1"]!.snapshot = detail();
        s.selection.pairId = null;
        s.selection.gameId = null;
        s.unavailable = "pair unavailable";
      },
    );
  });
});

function sQuery(ts: { getState(): TuningNavigationState }): string {
  return ts.getState().trialPage.queryKey!;
}
