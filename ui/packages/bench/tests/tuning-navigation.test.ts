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
import type { TuningSessionDetail, TuningSessionsResponse } from "../src/types.js";

const sessions: TuningSessionsResponse = {
  schema_version: 1,
  sessions: [{
    session_id: "session-1", game: "nim", label: null, status: "idle", target_trial_count: 2,
    counts: { total: 1, queued: 0, running: 0, terminal: 1, completed: 1, failed: 0, pruned: 0, cancelled: 0 },
    created_at: "2026-08-23 12:00:00", last_activity_at: "2026-08-23 12:01:00",
    attempts: [{ attempt_id: "attempt-1", bench_run_id: "run-1", status: "completed", started_at: "2026-08-23 12:00:00", ended_at: "2026-08-23 12:01:00", failure: null }],
    capabilities: { has_lifecycle: true, has_pairs: true, has_renderer_trace: true, has_search_reports: false },
  }],
};

function detail(status = "idle", includeGame = true): TuningSessionDetail {
  return {
    schema_version: 1,
    policy: null,
    summary: { session_id: "session-1", status, target_trial_count: 2, counts: sessions.sessions[0]!.counts },
    attempts: sessions.sessions[0]!.attempts,
    trials: [{
      trial_id: "trial-1", trial_number: 1, attempt_id: "attempt-1", status: "complete", config: { family: "ucb1" }, score: 1, mu: 2, sigma: 0.5, stop_reason: null, failure: null,
      pairs: [{
        pair_id: "pair-1", pair_index: 0, status: "complete", seed: 7, round: 1,
        opponent: { anchor_id: "anchor-1", config: {}, mu: 1, sigma: 0.5, label: null, provenance: null },
        pool_snapshot_fingerprint: "pool", rating_before: { mu: 1, sigma: 1 }, rating_after: { mu: 2, sigma: 0.5 }, score: 1, failure: null,
        games: includeGame ? [{ game_id: "game-1", candidate_side: "first", outcome: "candidate_win", seed: 7, round: 1, trace_game_seq: 8, plies: 10, elapsed_ms: 2, candidate: { iterations_total: 3, iterations_first_half: 2, move_time_ms: 1 }, baseline: { iterations_total: 3, iterations_first_half: 2, move_time_ms: 1 } }] : [],
      }],
      reports: [],
    }],
    manifest: {}, fingerprint: null,
    capabilities: sessions.sessions[0]!.capabilities,
    cursor: { session_sequence: 1 },
  };
}

function without(entity: "attempt" | "trial" | "pair"): TuningSessionDetail {
  const snapshot = detail();
  if (entity === "attempt") return { ...snapshot, attempts: [] };
  if (entity === "trial") return { ...snapshot, trials: [] };
  return { ...snapshot, trials: snapshot.trials.map((trial) => ({ ...trial, pairs: [] })) };
}

function env(list = sessions, ...snapshots: TuningSessionDetail[]): BenchEnv {
  let index = 0;
  return {
    listTuningSessions: () => Effect.send(list),
    getTuningSession: () => Effect.send(snapshots[Math.min(index++, snapshots.length - 1)] ?? detail()),
  } as unknown as BenchEnv;
}

const reducer = (state: TuningNavigationState, action: TuningNavigationAction, environment: BenchEnv) =>
  tuningNavigationReducer(state, action, environment);

describe("tuningNavigationReducer", () => {
  it("ignores a stale list response without replacing the newer snapshot", () => {
    const ts = createTestStore(reducer, env(), initialTuningNavigationState());
    ts.send({ tag: "listRequest" }, (state) => { state.list.status = "loading"; state.list.generation = 1; });
    ts.send({ tag: "listRequest" }, (state) => { state.list.generation = 2; });
    ts.receive({ tag: "listLoaded", generation: 1, response: sessions });
    ts.receive({ tag: "listLoaded", generation: 2, response: sessions }, (state) => {
      state.list.status = "done"; state.list.snapshot = sessions;
    });
  });

  it("preserves selection and expansion when a list refresh adds a sibling", () => {
    const sibling: TuningSessionsResponse = {
      schema_version: 1,
      sessions: [...sessions.sessions, { ...sessions.sessions[0]!, session_id: "session-2" }],
    };
    const initial = initialTuningNavigationState();
    initial.list = { ...initial.list, status: "done", snapshot: sessions };
    initial.selection = { sessionId: "session-1", attemptId: "attempt-1", trialId: "trial-1", pairId: "pair-1", gameId: "game-1" };
    initial.expandedIds = ["session-1", "attempt-1"];
    const ts = createTestStore(reducer, env(sibling), initial);

    ts.send({ tag: "listRequest" }, (state) => { state.list.status = "loading"; state.list.generation = 1; });
    ts.receive({ tag: "listLoaded", generation: 1, response: sibling }, (state) => {
      state.list.status = "done"; state.list.snapshot = sibling;
    });
  });

  it("falls back only to the pair when a selected game disappears", () => {
    const ts = createTestStore(reducer, env(sessions, detail(), detail("idle", false)), initialTuningNavigationState());
    ts.send({ tag: "selectSession", sessionId: "session-1" }, (state) => { state.selection.sessionId = "session-1"; state.detail.status = "loading"; state.detail.sessionId = "session-1"; state.detail.generation = 1; });
    ts.receive({ tag: "detailLoaded", generation: 1, sessionId: "session-1", detail: detail() }, (state) => { state.detail.status = "done"; state.detail.snapshot = detail(); });
    ts.send({ tag: "selectAttempt", attemptId: "attempt-1" }, (state) => { state.selection.attemptId = "attempt-1"; });
    ts.send({ tag: "selectTrial", trialId: "trial-1" }, (state) => { state.selection.trialId = "trial-1"; });
    ts.send({ tag: "selectPair", pairId: "pair-1" }, (state) => { state.selection.pairId = "pair-1"; });
    ts.send({ tag: "selectGame", gameId: "game-1" }, (state) => { state.selection.gameId = "game-1"; });
    ts.send({ tag: "detailRequest", sessionId: "session-1" }, (state) => { state.detail.status = "loading"; state.detail.generation = 2; });
    ts.receive({ tag: "detailLoaded", generation: 2, sessionId: "session-1", detail: detail("idle", false) }, (state) => {
      state.detail.status = "done"; state.detail.snapshot = detail("idle", false);
      state.selection.gameId = null; state.unavailable = "game unavailable";
    });
  });

  it.each([
    { entity: "attempt" as const, selection: { sessionId: "session-1", attemptId: null, trialId: null, pairId: null, gameId: null } },
    { entity: "trial" as const, selection: { sessionId: "session-1", attemptId: "attempt-1", trialId: null, pairId: null, gameId: null } },
    { entity: "pair" as const, selection: { sessionId: "session-1", attemptId: "attempt-1", trialId: "trial-1", pairId: null, gameId: null } },
  ])("falls back to the nearest ancestor when an $entity disappears", ({ entity, selection }) => {
    const ts = createTestStore(reducer, env(sessions, detail(), without(entity)), initialTuningNavigationState());
    ts.send({ tag: "selectSession", sessionId: "session-1" }, (state) => {
      state.selection.sessionId = "session-1"; state.detail.status = "loading"; state.detail.sessionId = "session-1"; state.detail.generation = 1;
    });
    ts.receive({ tag: "detailLoaded", generation: 1, sessionId: "session-1", detail: detail() }, (state) => {
      state.detail.status = "done"; state.detail.snapshot = detail();
    });
    ts.send({ tag: "selectGame", gameId: "game-1" }, (state) => {
      state.selection = { sessionId: "session-1", attemptId: "attempt-1", trialId: "trial-1", pairId: "pair-1", gameId: "game-1" };
    });
    ts.send({ tag: "detailRequest", sessionId: "session-1" }, (state) => { state.detail.status = "loading"; state.detail.generation = 2; });
    ts.receive({ tag: "detailLoaded", generation: 2, sessionId: "session-1", detail: without(entity) }, (state) => {
      state.detail.status = "done"; state.detail.snapshot = without(entity); state.selection = selection; state.unavailable = `${entity} unavailable`;
    });
  });

  it("refreshes only a selected active detail snapshot", () => {
    const ts = createTestStore(reducer, env(sessions, detail("active")), initialTuningNavigationState());
    ts.send({ tag: "selectSession", sessionId: "session-1" }, (state) => { state.selection.sessionId = "session-1"; state.detail.status = "loading"; state.detail.sessionId = "session-1"; state.detail.generation = 1; });
    ts.receive({ tag: "detailLoaded", generation: 1, sessionId: "session-1", detail: detail("active") }, (state) => { state.detail.status = "done"; state.detail.snapshot = detail("active"); });
    ts.advance(TUNING_DETAIL_REFRESH_MS);
    ts.receive({ tag: "detailRefreshTick", sessionId: "session-1", generation: 1 }, (state) => {
      state.detail.status = "loading"; state.detail.generation = 2;
    });
    ts.send({ tag: "clearSession" }, (state) => {
      state.selection = { sessionId: null, attemptId: null, trialId: null, pairId: null, gameId: null };
      state.detail = { status: "idle", snapshot: null, error: null, generation: 3, sessionId: null };
    });
    ts.receive({ tag: "detailLoaded", generation: 2, sessionId: "session-1", detail: detail("active") });
  });

  it("drops an active timer after a newer idle snapshot replaces it", () => {
    const ts = createTestStore(reducer, env(sessions, detail("active"), detail("idle")), initialTuningNavigationState());
    ts.send({ tag: "selectSession", sessionId: "session-1" }, (state) => {
      state.selection.sessionId = "session-1"; state.detail.status = "loading"; state.detail.sessionId = "session-1"; state.detail.generation = 1;
    });
    ts.receive({ tag: "detailLoaded", generation: 1, sessionId: "session-1", detail: detail("active") }, (state) => {
      state.detail.status = "done"; state.detail.snapshot = detail("active");
    });
    ts.send({ tag: "detailRequest", sessionId: "session-1" }, (state) => { state.detail.status = "loading"; state.detail.generation = 2; });
    ts.receive({ tag: "detailLoaded", generation: 2, sessionId: "session-1", detail: detail("idle") }, (state) => {
      state.detail.status = "done"; state.detail.snapshot = detail("idle");
    });
    ts.advance(TUNING_DETAIL_REFRESH_MS);
    ts.receive({ tag: "detailRefreshTick", sessionId: "session-1", generation: 1 });
  });

  it("does not schedule a timer for an idle detail snapshot", () => {
    const ts = createTestStore(reducer, env(sessions, detail("idle")), initialTuningNavigationState());
    ts.send({ tag: "selectSession", sessionId: "session-1" }, (state) => {
      state.selection.sessionId = "session-1"; state.detail.status = "loading"; state.detail.sessionId = "session-1"; state.detail.generation = 1;
    });
    ts.receive({ tag: "detailLoaded", generation: 1, sessionId: "session-1", detail: detail("idle") }, (state) => {
      state.detail.status = "done"; state.detail.snapshot = detail("idle");
    });
    expect(ts.scheduler.pendingCount).toBe(0);
  });

  it("ignores stale detail failures while preserving the latest request error", () => {
    const noResponse = {
      listTuningSessions: () => Effect.none(),
      getTuningSession: () => Effect.none(),
    } as unknown as BenchEnv;
    const ts = createTestStore(reducer, noResponse, initialTuningNavigationState());

    ts.send({ tag: "selectSession", sessionId: "session-1" }, (state) => {
      state.selection.sessionId = "session-1"; state.detail.status = "loading"; state.detail.sessionId = "session-1"; state.detail.generation = 1;
    });
    ts.send({ tag: "detailRequest", sessionId: "session-1" }, (state) => { state.detail.generation = 2; });
    ts.send({ tag: "detailFailed", generation: 1, sessionId: "session-1", error: "old" });
    ts.send({ tag: "detailFailed", generation: 2, sessionId: "session-1", error: "new" }, (state) => {
      state.detail.status = "error"; state.detail.error = "new";
    });
  });
});
