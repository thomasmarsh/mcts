// tuner-reducer.test.ts — pure-reducer tests via the shared TestStore
// harness against a mocked `TunerEnv` (no live server, per AGENTS.md). The
// journal and log-tail poll loops schedule with `Effect.delay` and run on
// TestStore's manual scheduler; every test winds them down so no sleep is
// left pending.

import { describe, expect, it } from "vitest";
import { Effect } from "@mcts/core";
import { createTestStore } from "../../../../tests/test-store.js";
import {
  initialTunerState,
  tunerReducer,
  LOG_TAIL_MS,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { JOURNAL_POLL_MS, PROJECTION_REFRESH_MS } from "../../src/tuner/tuner-poll.js";
import type { RemoteData } from "../../src/tuner/remote-data.js";
import { mockTunerEnv, runView } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";

const store = (env: TunerEnv) =>
  createTestStore<TunerState, TunerAction, TunerEnv>(tunerReducer, env, initialTunerState());

/** `{status:"ok", value, fetchedAt: <any number>}` — the reducer stamps
 * `Date.now()`, which `toEqual` matches with an asymmetric matcher. */
function ok<T>(value: T): RemoteData<T> {
  return { status: "ok", value, fetchedAt: expect.any(Number) as unknown as number };
}

const loadingAll = (s: TunerState): void => {
  s.tunableGames = { status: "loading" };
  s.objectives = { status: "loading" };
  s.profiles = { status: "loading" };
  s.runs = { status: "loading" };
  s.projectionRuns = { status: "loading" };
  s.journalGeneration = 1;
};

const drainInit = (ts: ReturnType<typeof store>): void => {
  ts.receive({ tag: "tunableGamesLoaded", tunableGames: [] }, (s) => {
    s.tunableGames = ok([]);
  });
  ts.receive({ tag: "objectivesLoaded", objectives: [] }, (s) => {
    s.objectives = ok([]);
  });
  ts.receive({ tag: "profilesLoaded", profiles: [] }, (s) => {
    s.profiles = ok([]);
  });
};

describe("tunerReducer", () => {
  it("init loads every resource and stops polling when nothing is live", () => {
    const ts = store(
      mockTunerEnv({ listRuns: () => Effect.send([runView({ status: "exited" })]) }),
    );

    ts.send({ tag: "init" }, loadingAll);
    drainInit(ts);
    ts.receive({ tag: "runsLoaded", runs: [runView({ status: "exited" })] }, (s) => {
      s.runs = ok([runView({ status: "exited" })]);
    });
    ts.receive({ tag: "projectionLoaded", runs: [] }, (s) => {
      s.projectionRuns = ok([]);
    });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });
  });

  it("keeps polling the journal while a run is live, then stops and refreshes", () => {
    let call = 0;
    const ts = store(
      mockTunerEnv({
        listRuns: () => {
          call += 1;
          return Effect.send(
            call < 2 ? [runView({ status: "live" })] : [runView({ status: "exited" })],
          );
        },
      }),
    );

    ts.send({ tag: "init" }, loadingAll);
    drainInit(ts);
    ts.receive({ tag: "runsLoaded", runs: [runView({ status: "live" })] }, (s) => {
      s.runs = ok([runView({ status: "live" })]);
    });
    ts.receive({ tag: "projectionLoaded", runs: [] }, (s) => {
      s.projectionRuns = ok([]);
    });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });

    ts.advance(JOURNAL_POLL_MS);
    ts.receive({ tag: "journalTick", generation: 1 });
    ts.receive({ tag: "runsLoaded", runs: [runView({ status: "exited" })] }, (s) => {
      s.runs = ok([runView({ status: "exited" })]);
    });
    ts.receive({ tag: "refreshProjection" }, (s) => {
      s.refreshing = true;
    });
    ts.receive({ tag: "refreshDone" }, (s) => {
      s.refreshing = false;
      s.lastProjectionRefreshAt = expect.any(Number) as unknown as number;
    });
    ts.receive({ tag: "projectionLoaded", runs: [] });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });
  });

  it("does not run its own projection-refresh loop while the evidence stream is healthy", () => {
    let refreshes = 0;
    const s0 = initialTunerState();
    s0.openRunId = "r1";
    s0.runs = { status: "ok", value: [runView({ run_id: "r1", status: "live" })], fetchedAt: 0 };
    s0.evidenceStreamActive = true;
    s0.evidenceStreamOk = true;
    s0.evidenceGeneration = 1;
    const detail = {
      run_id: "r1",
      terminal_status: null,
      report_available: false,
      ingest_error: null,
      manifest: null,
      report: null,
      compute: [],
    };
    const ts = createTestStore<TunerState, TunerAction, TunerEnv>(
      tunerReducer,
      mockTunerEnv({
        listRuns: () => Effect.send([runView({ run_id: "r1", status: "exited" })]),
        refreshProjection: () => {
          refreshes += 1;
          return Effect.send({ projected: 1, skipped: 0, ingest_errors: 0, pruned: 0 });
        },
      }),
      s0,
    );

    // A journal poll shows the run still live: the headless follower owns the
    // projection, so no client auto-refresh loop starts (it only schedules the
    // cheap journal poll).
    ts.send({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "live" })] }, (s) => {
      s.runs = ok([runView({ run_id: "r1", status: "live" })]);
    });

    // The stream's `projection-updated` frame pulls fresh rows directly — a
    // silent science reload, and crucially *no* `refreshProjection` POST.
    ts.send({ tag: "projectionUpdatedPush", generation: 1 }, (s) => {
      s.lastProjectionRefreshAt = expect.any(Number) as unknown as number;
      s.resourceGeneration = 1;
      s.pairsGeneration = 1;
    });
    ts.receive({ tag: "projectionLoaded", runs: [] }, (s) => {
      s.projectionRuns = ok([]);
    });
    ts.receive({ tag: "detailLoaded", generation: 1, detail }, (s) => {
      s.projectionDetail = ok(detail);
    });
    ts.receive(
      { tag: "validationLoaded", generation: 1, validation: { rows: [], unresolved_ties: null } },
      (s) => {
        s.validation = ok({ rows: [], unresolved_ties: null });
      },
    );
    ts.receive({ tag: "candidatesLoaded", generation: 1, candidates: [] }, (s) => {
      s.candidates = ok([]);
    });
    ts.receive({ tag: "pairsLoaded", generation: 1, pairs: [] }, (s) => {
      s.pairs = ok([]);
    });
    ts.receive({ tag: "reportLoaded", generation: 1, report: {} }, (s) => {
      s.report = ok({});
    });
    ts.receive({ tag: "proposalsLoaded", generation: 1, proposals: [] }, (s) => {
      s.proposals = ok([]);
    });
    ts.receive({ tag: "observationsLoaded", generation: 1, observations: [] }, (s) => {
      s.observations = ok([]);
    });
    ts.receive({ tag: "shadowDecisionsLoaded", generation: 1, rows: [] }, (s) => {
      s.shadowDecisions = ok([]);
    });
    ts.receive({ tag: "activeEliminationsLoaded", generation: 1, rows: [] }, (s) => {
      s.activeEliminations = ok([]);
    });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });

    // No client refresh POST happened while the stream was healthy.
    expect(refreshes).toBe(0);

    // Wind the journal poll down: the next poll sees the run exited, which
    // deactivates the loop (and fires the one terminal-transition refresh).
    ts.advance(JOURNAL_POLL_MS);
    ts.receive({ tag: "journalTick", generation: 0 });
    ts.receive({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "exited" })] }, (s) => {
      s.runs = ok([runView({ run_id: "r1", status: "exited" })]);
      s.evidenceStreamActive = false;
      s.evidenceGeneration = 2;
    });
    ts.receive({ tag: "refreshProjection" }, (s) => {
      s.refreshing = true;
    });
    ts.receive({ tag: "refreshDone" }, (s) => {
      s.refreshing = false;
      s.lastProjectionRefreshAt = expect.any(Number) as unknown as number;
      s.resourceGeneration = 2;
      s.pairsGeneration = 2;
      s.projectionDetail = { status: "loading", previous: detail };
      s.validation = { status: "loading", previous: { rows: [], unresolved_ties: null } };
      s.candidates = { status: "loading", previous: [] };
      s.pairs = { status: "loading", previous: [] };
      s.proposals = { status: "loading", previous: [] };
      s.observations = { status: "loading", previous: [] };
      s.shadowDecisions = { status: "loading", previous: [] };
      s.activeEliminations = { status: "loading", previous: [] };
      s.report = { status: "loading", previous: {} };
    });
    ts.receive({ tag: "projectionLoaded", runs: [] });
    ts.receive({ tag: "detailLoaded", generation: 2, detail }, (s) => {
      s.projectionDetail = ok(detail);
    });
    ts.receive(
      { tag: "validationLoaded", generation: 2, validation: { rows: [], unresolved_ties: null } },
      (s) => {
        s.validation = ok({ rows: [], unresolved_ties: null });
      },
    );
    ts.receive({ tag: "candidatesLoaded", generation: 2, candidates: [] }, (s) => {
      s.candidates = ok([]);
    });
    ts.receive({ tag: "pairsLoaded", generation: 2, pairs: [] }, (s) => {
      s.pairs = ok([]);
    });
    ts.receive({ tag: "reportLoaded", generation: 2, report: {} }, (s) => {
      s.report = ok({});
    });
    ts.receive({ tag: "proposalsLoaded", generation: 2, proposals: [] }, (s) => {
      s.proposals = ok([]);
    });
    ts.receive({ tag: "observationsLoaded", generation: 2, observations: [] }, (s) => {
      s.observations = ok([]);
    });
    ts.receive({ tag: "shadowDecisionsLoaded", generation: 2, rows: [] }, (s) => {
      s.shadowDecisions = ok([]);
    });
    ts.receive({ tag: "activeEliminationsLoaded", generation: 2, rows: [] }, (s) => {
      s.activeEliminations = ok([]);
    });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });
    expect(refreshes).toBe(1);
  });

  it("falls back to the client refresh loop once the evidence stream fails", () => {
    let refreshes = 0;
    const s0 = initialTunerState();
    s0.openRunId = "r1";
    s0.runs = { status: "ok", value: [runView({ run_id: "r1", status: "live" })], fetchedAt: 0 };
    s0.evidenceStreamActive = true;
    s0.evidenceStreamOk = true;
    s0.evidenceGeneration = 3;
    const ts = createTestStore<TunerState, TunerAction, TunerEnv>(
      tunerReducer,
      mockTunerEnv({
        refreshProjection: () => {
          refreshes += 1;
          return Effect.send({ projected: 1, skipped: 0, ingest_errors: 0, pruned: 0 });
        },
      }),
      s0,
    );

    ts.send({ tag: "evidenceStreamFailed", generation: 3, error: "connection lost" }, (s) => {
      s.evidenceStreamOk = false;
      s.projectionRefreshActive = true;
      s.projectionRefreshGeneration = 1;
    });
    // The degraded loop's first tick fires a real `refreshProjection` POST.
    ts.receive({ tag: "projectionRefreshTick", generation: 1 });
    ts.receive({ tag: "autoRefreshProjection" }, (s) => {
      s.autoRefreshing = true;
    });
    expect(refreshes).toBe(1);

    ts.receive({ tag: "autoRefreshDone" }, (s) => {
      s.autoRefreshing = false;
      s.lastProjectionRefreshAt = expect.any(Number) as unknown as number;
      s.resourceGeneration = 1;
      s.pairsGeneration = 1;
    });
    ts.receive({ tag: "projectionLoaded", runs: [] }, (s) => {
      s.projectionRuns = ok([]);
    });
    const detail = {
      run_id: "r1",
      terminal_status: null,
      report_available: false,
      ingest_error: null,
      manifest: null,
      report: null,
      compute: [],
    };
    ts.receive({ tag: "detailLoaded", generation: 1, detail }, (s) => {
      s.projectionDetail = ok(detail);
    });
    ts.receive(
      { tag: "validationLoaded", generation: 1, validation: { rows: [], unresolved_ties: null } },
      (s) => {
        s.validation = ok({ rows: [], unresolved_ties: null });
      },
    );
    ts.receive({ tag: "candidatesLoaded", generation: 1, candidates: [] }, (s) => {
      s.candidates = ok([]);
    });
    ts.receive({ tag: "pairsLoaded", generation: 1, pairs: [] }, (s) => {
      s.pairs = ok([]);
    });
    ts.receive({ tag: "reportLoaded", generation: 1, report: {} }, (s) => {
      s.report = ok({});
    });
    ts.receive({ tag: "proposalsLoaded", generation: 1, proposals: [] }, (s) => {
      s.proposals = ok([]);
    });
    ts.receive({ tag: "observationsLoaded", generation: 1, observations: [] }, (s) => {
      s.observations = ok([]);
    });
    ts.receive({ tag: "shadowDecisionsLoaded", generation: 1, rows: [] }, (s) => {
      s.shadowDecisions = ok([]);
    });
    ts.receive({ tag: "activeEliminationsLoaded", generation: 1, rows: [] }, (s) => {
      s.activeEliminations = ok([]);
    });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });

    // Closing the run winds the fallback loop and evidence poll down.
    ts.send({ tag: "closeRun" }, (s) => {
      s.openRunId = null;
      s.projectionRefreshActive = false;
      s.projectionRefreshGeneration = 2;
      s.evidenceStreamActive = false;
      s.evidenceGeneration = 4;
      s.resourceGeneration = 2;
      s.pairsGeneration = 2;
      s.projectionDetail = { status: "idle" };
      s.validation = { status: "idle" };
      s.candidates = { status: "idle" };
      s.pairs = { status: "idle" };
      s.proposals = { status: "idle" };
      s.observations = { status: "idle" };
      s.shadowDecisions = { status: "idle" };
      s.activeEliminations = { status: "idle" };
      s.report = { status: "idle" };
    });

    // Both stale ticks fire inert (their generations are now behind).
    ts.advance(PROJECTION_REFRESH_MS);
    ts.receive({ tag: "evidencePollTick", generation: 3 });
    ts.receive({ tag: "projectionRefreshTick", generation: 1 });
    expect(refreshes).toBe(1);
  });

  it("optimistically inserts and opens a launched run, then tails its log", () => {
    const launched = runView({ run_id: "fresh", status: "live" });
    const detail = {
      run_id: "fresh",
      terminal_status: null,
      report_available: false,
      ingest_error: null,
      manifest: null,
      report: null,
      compute: [],
    };
    const ts = store(
      mockTunerEnv({
        launchRun: () => Effect.send(launched),
        getRunLog: () =>
          Effect.send({ lines: ["cohort 0 starting"], next_offset: 17, err_lines: [] }),
        listRuns: () => Effect.send([runView({ run_id: "fresh", status: "exited" })]),
        getProjectionRun: () => Effect.send(detail),
      }),
    );

    ts.send({ tag: "launch", request: launchRequest() }, (s) => {
      s.launch = { status: "pending", error: null, lastRunId: null };
    });
    ts.receive({ tag: "launchOk", run: launched }, (s) => {
      s.launch = { status: "done", error: null, lastRunId: "fresh" };
      s.runs = ok([launched]);
      s.openRunId = "fresh";
      s.logGeneration = 1;
      s.journalGeneration = 1;
      s.resourceGeneration = 1;
      s.pairsGeneration = 1;
      s.evidenceStreamActive = true;
      s.evidenceGeneration = 1;
      s.projectionDetail = { status: "loading" };
      s.validation = { status: "loading" };
      s.candidates = { status: "loading" };
      s.pairs = { status: "loading" };
      s.proposals = { status: "loading" };
      s.observations = { status: "loading" };
      s.shadowDecisions = { status: "loading" };
      s.activeEliminations = { status: "loading" };
      s.report = { status: "loading" };
      s.log = { lines: [], errLines: [], offset: 0, error: null, active: true };
    });

    ts.receive({ tag: "logTick", generation: 1 });
    ts.receive({ tag: "journalTick", generation: 1 });
    ts.receive({ tag: "detailLoaded", generation: 1, detail }, (s) => {
      s.projectionDetail = ok(detail);
    });
    ts.receive(
      { tag: "validationLoaded", generation: 1, validation: { rows: [], unresolved_ties: null } },
      (s) => {
        s.validation = ok({ rows: [], unresolved_ties: null });
      },
    );
    ts.receive({ tag: "candidatesLoaded", generation: 1, candidates: [] }, (s) => {
      s.candidates = ok([]);
    });
    ts.receive({ tag: "pairsLoaded", generation: 1, pairs: [] }, (s) => {
      s.pairs = ok([]);
    });
    ts.receive({ tag: "reportLoaded", generation: 1, report: {} }, (s) => {
      s.report = ok({});
    });
    ts.receive({ tag: "proposalsLoaded", generation: 1, proposals: [] }, (s) => {
      s.proposals = ok([]);
    });
    ts.receive({ tag: "observationsLoaded", generation: 1, observations: [] }, (s) => {
      s.observations = ok([]);
    });
    ts.receive({ tag: "shadowDecisionsLoaded", generation: 1, rows: [] }, (s) => {
      s.shadowDecisions = ok([]);
    });
    ts.receive({ tag: "activeEliminationsLoaded", generation: 1, rows: [] }, (s) => {
      s.activeEliminations = ok([]);
    });

    ts.receive(
      {
        tag: "logLoaded",
        generation: 1,
        lines: ["cohort 0 starting"],
        errLines: [],
        nextOffset: 17,
      },
      (s) => {
        s.log = {
          lines: ["cohort 0 starting"],
          errLines: [],
          offset: 17,
          error: null,
          active: true,
        };
      },
    );
    ts.receive(
      { tag: "runsLoaded", runs: [runView({ run_id: "fresh", status: "exited" })] },
      (s) => {
        s.runs = ok([runView({ run_id: "fresh", status: "exited" })]);
        s.evidenceStreamActive = false;
        s.evidenceGeneration = 2;
      },
    );
    ts.receive({ tag: "refreshProjection" }, (s) => {
      s.refreshing = true;
    });
    ts.receive({ tag: "refreshDone" }, (s) => {
      s.refreshing = false;
      s.lastProjectionRefreshAt = expect.any(Number) as unknown as number;
      s.resourceGeneration = 2;
      s.pairsGeneration = 2;
      s.projectionDetail = { status: "loading", previous: detail };
      s.validation = { status: "loading", previous: { rows: [], unresolved_ties: null } };
      s.candidates = { status: "loading", previous: [] };
      s.pairs = { status: "loading", previous: [] };
      s.proposals = { status: "loading", previous: [] };
      s.observations = { status: "loading", previous: [] };
      s.shadowDecisions = { status: "loading", previous: [] };
      s.activeEliminations = { status: "loading", previous: [] };
      s.report = { status: "loading", previous: {} };
    });
    ts.receive({ tag: "projectionLoaded", runs: [] });
    ts.receive({ tag: "detailLoaded", generation: 2, detail }, (s) => {
      s.projectionDetail = ok(detail);
    });
    ts.receive(
      { tag: "validationLoaded", generation: 2, validation: { rows: [], unresolved_ties: null } },
      (s) => {
        s.validation = ok({ rows: [], unresolved_ties: null });
      },
    );
    ts.receive({ tag: "candidatesLoaded", generation: 2, candidates: [] }, (s) => {
      s.candidates = ok([]);
    });
    ts.receive({ tag: "pairsLoaded", generation: 2, pairs: [] }, (s) => {
      s.pairs = ok([]);
    });
    ts.receive({ tag: "reportLoaded", generation: 2, report: {} }, (s) => {
      s.report = ok({});
    });
    ts.receive({ tag: "proposalsLoaded", generation: 2, proposals: [] }, (s) => {
      s.proposals = ok([]);
    });
    ts.receive({ tag: "observationsLoaded", generation: 2, observations: [] }, (s) => {
      s.observations = ok([]);
    });
    ts.receive({ tag: "shadowDecisionsLoaded", generation: 2, rows: [] }, (s) => {
      s.shadowDecisions = ok([]);
    });
    ts.receive({ tag: "activeEliminationsLoaded", generation: 2, rows: [] }, (s) => {
      s.activeEliminations = ok([]);
    });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });
    expect(LOG_TAIL_MS).toBe(3000);
  });

  it("surfaces a fast launch failure and repulls the journal so the fleet shows it", async () => {
    const dead = runView({
      run_id: "fresh",
      status: "failed",
      terminal_outcome: "exited",
      error_detail: "tuner failed: objective file does not exist",
    });
    const ts = store(
      mockTunerEnv({
        launchRun: () =>
          Effect.fromPromise(() =>
            Promise.reject(
              new Error(
                "failed to launch tuner run: tuner run 'fresh' died during startup (exit status 3)",
              ),
            ),
          ),
        listRuns: () => Effect.send([dead]),
      }),
    );

    ts.send({ tag: "launch", request: launchRequest() }, (s) => {
      s.launch = { status: "pending", error: null, lastRunId: null };
    });
    await ts.drain();
    ts.receive(
      {
        tag: "launchFailed",
        error: "Error: failed to launch tuner run: tuner run 'fresh' died during startup (exit status 3)",
      },
      (s) => {
        s.launch = {
          status: "error",
          error:
            "Error: failed to launch tuner run: tuner run 'fresh' died during startup (exit status 3)",
          lastRunId: null,
        };
      },
    );
    ts.receive({ tag: "runsLoaded", runs: [dead] }, (s) => {
      s.runs = ok([dead]);
    });
    ts.receive({ tag: "refreshProjection" }, (s) => {
      s.refreshing = true;
    });
    ts.receive({ tag: "refreshDone" }, (s) => {
      s.refreshing = false;
      s.lastProjectionRefreshAt = expect.any(Number) as unknown as number;
    });
    ts.receive({ tag: "projectionLoaded", runs: [] });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });
  });

  it("blocks the launch when preflight reports the config is invalid", () => {
    const ts = store(
      mockTunerEnv({
        preflightRun: () =>
          Effect.send({
            ok: false,
            errors: ["validation pairs cannot exceed production validation pairs"],
          }),
      }),
    );

    ts.send({ tag: "preflight", request: launchRequest() }, (s) => {
      s.preflightGeneration = 1;
      s.preflight = { status: "checking", errors: [], error: null };
    });
    ts.receive(
      {
        tag: "preflightChecked",
        generation: 1,
        result: {
          ok: false,
          errors: ["validation pairs cannot exceed production validation pairs"],
        },
      },
      (s) => {
        s.preflight = {
          status: "invalid",
          errors: ["validation pairs cannot exceed production validation pairs"],
          error: null,
        };
      },
    );
  });

  it("drops a stale preflight response", () => {
    const ts = store(mockTunerEnv());
    ts.send({ tag: "preflight", request: launchRequest() }, (s) => {
      s.preflightGeneration = 1;
      s.preflight = { status: "checking", errors: [], error: null };
    });
    // A second edit supersedes the first before its response lands.
    ts.send({ tag: "preflight", request: launchRequest() }, (s) => {
      s.preflightGeneration = 2;
    });
    // The generation-1 response lands first and is ignored.
    ts.receive({ tag: "preflightChecked", generation: 1, result: { ok: true, errors: [] } });
    expect(ts.getState().preflight.status).toBe("checking");
    // The current generation-2 response takes effect.
    ts.receive({ tag: "preflightChecked", generation: 2, result: { ok: true, errors: [] } }, (s) => {
      s.preflight = { status: "ok", errors: [], error: null };
    });
  });
});

describe("tunerReducer objectives", () => {
  const file = {
    key: "nim-v1",
    objective_id: "nim-v1",
    game_kind: "nim",
    opponent_count: 2,
    updated_at: null,
    is_seed: false,
  };

  it("opens an objective and loads its detail", () => {
    const detail = { key: "nim-v1", content: { schema_version: 1 }, updated_at: null, is_seed: false };
    const ts = store(mockTunerEnv({ getObjective: () => Effect.send(detail) }));
    ts.send({ tag: "openObjective", key: "nim-v1" }, (s) => {
      s.openObjectiveKey = "nim-v1";
      s.objectiveDetail = { status: "loading" };
    });
    ts.receive({ tag: "objectiveDetailLoaded", key: "nim-v1", detail }, (s) => {
      s.openObjectiveKey = "nim-v1";
      s.objectiveDetail = ok(detail);
    });
  });

  it("saves an objective and re-lists the corpus", () => {
    const detail = { key: "nim-v1", content: { schema_version: 1 }, updated_at: null, is_seed: false };
    const ts = store(
      mockTunerEnv({
        putObjective: () => Effect.send(detail),
        listObjectives: () => Effect.send([file]),
      }),
    );
    ts.send({ tag: "saveObjective", key: "nim-v1", content: { schema_version: 1 } }, (s) => {
      s.objectiveSave = { status: "pending", error: null };
    });
    ts.receive({ tag: "saveObjectiveOk", detail }, (s) => {
      s.objectiveSave = { status: "done", error: null };
      s.openObjectiveKey = "nim-v1";
      s.objectiveDetail = ok(detail);
    });
    ts.receive({ tag: "objectivesLoaded", objectives: [file] }, (s) => {
      s.objectives = ok([file]);
    });
  });

  it("deletes an objective and re-lists the corpus", () => {
    const ts = store(
      mockTunerEnv({
        deleteObjective: () => Effect.send(undefined),
        listObjectives: () => Effect.send([]),
      }),
    );
    ts.send({ tag: "deleteObjective", key: "nim-v1" }, (s) => {
      s.objectiveMutating = "nim-v1";
    });
    ts.receive({ tag: "deleteObjectiveOk" }, (s) => {
      s.objectiveMutating = null;
    });
    ts.receive({ tag: "objectivesLoaded", objectives: [] }, (s) => {
      s.objectives = ok([]);
    });
  });

  it("runs a server-side validation dry run", () => {
    const result = { ok: false, errors: ["exactly one default opponent is required"] };
    const ts = store(mockTunerEnv({ validateObjective: () => Effect.send(result) }));
    ts.send({ tag: "validateObjective", key: "nim-v1", content: { schema_version: 1 } }, (s) => {
      s.objectiveValidation = { status: "loading" };
    });
    ts.receive({ tag: "validateObjectiveOk", result }, (s) => {
      s.objectiveValidation = ok(result);
    });
  });
});

describe("tunerReducer profiles", () => {
  const file = {
    key: "nim-sweep",
    profile_id: "nim-sweep",
    game_kind: "nim",
    objective_key: "nim-v1",
    constraint_count: 2,
    updated_at: null,
    is_seed: false,
  };
  const detail = { key: "nim-sweep", content: { profile_id: "nim-sweep" }, updated_at: null, is_seed: false };

  it("opens a profile and loads its detail", () => {
    const ts = store(mockTunerEnv({ getProfile: () => Effect.send(detail) }));
    ts.send({ tag: "openProfile", key: "nim-sweep" }, (s) => {
      s.openProfileKey = "nim-sweep";
      s.profileDetail = { status: "loading" };
    });
    ts.receive({ tag: "profileDetailLoaded", key: "nim-sweep", detail }, (s) => {
      s.openProfileKey = "nim-sweep";
      s.profileDetail = ok(detail);
    });
  });

  it("saves a profile and re-lists the corpus", () => {
    const ts = store(
      mockTunerEnv({
        putProfile: () => Effect.send(detail),
        listProfiles: () => Effect.send([file]),
      }),
    );
    ts.send({ tag: "saveProfile", key: "nim-sweep", content: { profile_id: "nim-sweep" } }, (s) => {
      s.profileSave = { status: "pending", error: null };
    });
    ts.receive({ tag: "saveProfileOk", detail }, (s) => {
      s.profileSave = { status: "done", error: null };
      s.openProfileKey = "nim-sweep";
      s.profileDetail = ok(detail);
    });
    ts.receive({ tag: "profilesLoaded", profiles: [file] }, (s) => {
      s.profiles = ok([file]);
    });
  });

  it("deletes a profile and re-lists the corpus", () => {
    const ts = store(
      mockTunerEnv({
        deleteProfile: () => Effect.send(undefined),
        listProfiles: () => Effect.send([]),
      }),
    );
    ts.send({ tag: "deleteProfile", key: "nim-sweep" }, (s) => {
      s.profileMutating = "nim-sweep";
    });
    ts.receive({ tag: "deleteProfileOk" }, (s) => {
      s.profileMutating = null;
    });
    ts.receive({ tag: "profilesLoaded", profiles: [] }, (s) => {
      s.profiles = ok([]);
    });
  });

  it("runs a server-side launch preflight dry run", () => {
    const result = { ok: false, errors: ["objective file does not exist"] };
    const ts = store(mockTunerEnv({ validateProfile: () => Effect.send(result) }));
    ts.send({ tag: "validateProfile", key: "nim-sweep", content: { profile_id: "nim-sweep" } }, (s) => {
      s.profileValidation = { status: "loading" };
    });
    ts.receive({ tag: "validateProfileOk", result }, (s) => {
      s.profileValidation = ok(result);
    });
  });
});

describe("tunerReducer scienceStale", () => {
  const env = mockTunerEnv();

  it("is set by a scientific evidence event and cleared by a completed refresh", () => {
    const draft = initialTunerState();

    // A non-scientific event leaves it alone.
    tunerReducer(
      draft,
      { tag: "evidenceEvents", generation: 0, events: [{ sequence: 1, type: "pair_started", payload: {} }] },
      env,
    );
    expect(draft.scienceStale).toBe(false);

    // A pair completion means the projection is behind.
    tunerReducer(
      draft,
      {
        tag: "evidenceEvents",
        generation: 0,
        events: [{ sequence: 2, type: "pair_completed", payload: {} }],
      },
      env,
    );
    expect(draft.scienceStale).toBe(true);

    // Any completed refresh brings the science level again.
    tunerReducer(draft, { tag: "autoRefreshDone" }, env);
    expect(draft.scienceStale).toBe(false);
  });
});

function launchRequest() {
  return {
    game_kind: "nim",
    objective_key: "nim-v1",
    run_id: "fresh",
    task_seed: 1,
    tuning_pair_budget: 24,
    validation_pair_budget: 24,
    production_validation_pairs: 6,
  };
}
