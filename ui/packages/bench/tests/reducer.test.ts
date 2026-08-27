// tests/reducer.test.ts — Pure-reducer tests for benchReducer via the
// shared TestStore harness (ui/tests/test-store.ts) against a mocked
// BenchEnv — no live server, per AGENTS.md.
//
// The log-tail poll loop schedules its next tick via `Effect.delay`, which
// TestStore runs on its manual TestScheduler — ticks only become receivable
// when the test advances virtual time (`ts.advance(ms)`), never from a real
// or mocked timer. Every tailing test ends with the loop wound down
// (terminal run, close, or give-up) so no sleep is left pending; TestStore's
// afterEach fails the test otherwise.

import { describe, expect, it } from "vitest";
import { Effect } from "@mcts/core";
import { createTestStore } from "../../../tests/test-store.js";
import {
  benchReducer,
  tailDelayMs,
  TAIL_BACKOFF_MAX_MS,
  TAIL_BACKOFF_START_MS,
  TAIL_MAX_FAILURES,
  type BenchEnv,
} from "../src/reducer.js";
import { initialBenchState } from "../src/state.js";
import type {
  LaunchResponse,
  RunDetail,
  RunFilters,
  RunSummary,
  TunerGameInfo,
} from "../src/types.js";

// ── Fixtures ────────────────────────────────────────────────────────────────

const summary: RunSummary = {
  run_id: "tuner-druid-20260101T000000-abc1234",
  kind: "tuner",
  game: "druid",
  project_id: null,
  experiment_id: null,
  label: null,
  git_sha: "abc1234",
  git_dirty: false,
  host: "testhost",
  pid: 1234,
  started_at: "2026-01-01T00:00:00Z",
  ended_at: null,
  status: "running",
  match_count: 4,
  trial_count: 0,
};

function makeDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  return {
    run_id: summary.run_id,
    kind: "tuner",
    game: "druid",
    project_id: null,
    experiment_id: null,
    experiment_spec: null,
    label: null,
    config: null,
    git_sha: "abc1234",
    git_dirty: false,
    host: "testhost",
    pid: 1234,
    started_at: "2026-01-01T00:00:00Z",
    ended_at: null,
    status: "running",
    log_path: "/bench-runs/x/log.jsonl",
    exit_code: null,
    match_count: 4,
    trial_count: 0,
    incumbent: null,
    ...overrides,
  };
}

const runningDetail = makeDetail();
const terminalDetail = makeDetail({
  status: "completed",
  ended_at: "2026-01-01T01:00:00Z",
  exit_code: 0,
});
function loadingTuningSessions(state: ReturnType<typeof initialBenchState>): void {
  state.tuningNavigation.list.status = "loading";
  state.tuningNavigation.list.generation += 1;
}

const mockEnv: BenchEnv = {
  listRuns: () => Effect.none(),
  getRun: () => Effect.none(),
  getRunLog: () => Effect.none(),
  getRunStdout: () => Effect.none(),
  launchRun: () => Effect.none(),
  stopRun: () => Effect.none(),
  getTunerKinds: () => Effect.none(),
  listTuningSessions: () => Effect.none(),
  getTuningSession: () => Effect.none(),
  getTuningAnalysisOverview: () => Effect.none(),
  getTuningTrialPage: () => Effect.none(),
  getTuningTrialDetail: () => Effect.none(),
  stopTuningSession: () => Effect.none(),
  resumeTuningSession: () => Effect.none(),
  addTuningSessionBudget: () => Effect.none(),
  // Unlike the others, every tailTick's Promise.all includes a trials fetch
  // unconditionally (see reducer.ts) -- Effect.none() here would never
  // resolve and hang every tailing test, so the default must actually send.
  getRunTrials: () => Effect.send([]),
  getRunGames: () => Effect.send([]),
  getRunGameMoves: () => Effect.none(),
  deleteRun: () => Effect.none(),
  // Same reasoning as getRunTrials above: every tick fetches the chain too.
  // An empty chain means the per-rung trials fetch loop is a no-op, which
  // also keeps every existing `trialsCalls` assertion below unaffected by
  // this addition.
};

// ── Runs list ───────────────────────────────────────────────────────────────

describe("benchReducer / runs", () => {
  it("request -> submitted('done') populates the list", () => {
    const env: BenchEnv = { ...mockEnv, listRuns: () => Effect.send([summary]) };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "runs", action: { tag: "request" } }, (s) => {
      s.runs.status = "pending";
      loadingTuningSessions(s);
    });
    ts.receive(
      {
        tag: "runs",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: [summary] } },
        },
      },
      (s) => {
        s.runs.status = "done";
        s.runs.result = [summary];
      },
    );
  });

  it("setRunFilters stores the filters and refetches with them", () => {
    const seen: RunFilters[] = [];
    const env: BenchEnv = {
      ...mockEnv,
      listRuns: (filters) => {
        seen.push({ ...filters });
        return Effect.send([summary]);
      },
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "setRunFilters", status: "running", game: "druid" }, (s) => {
      s.runFilters = { status: "running", game: "druid" };
      s.runs.status = "pending";
      loadingTuningSessions(s);
    });
    ts.receive(
      {
        tag: "runs",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: [summary] } },
        },
      },
      (s) => {
        s.runs.status = "done";
        s.runs.result = [summary];
      },
    );
    expect(seen).toEqual([{ status: "running", game: "druid" }]);
  });
});

// ── Log tail ────────────────────────────────────────────────────────────────

describe("benchReducer / log tail", () => {
  it("tailDelayMs doubles per idle attempt and caps at the max", () => {
    expect(tailDelayMs(0)).toBe(TAIL_BACKOFF_START_MS);
    expect(tailDelayMs(1)).toBe(TAIL_BACKOFF_START_MS * 2);
    expect(tailDelayMs(2)).toBe(TAIL_BACKOFF_START_MS * 4);
    expect(tailDelayMs(100)).toBe(TAIL_BACKOFF_MAX_MS);
  });

  it("openRun starts the tail loop; new lines append and reset the backoff; a terminal status stops it", async () => {
    let logCalls = 0;
    let runCalls = 0;
    const env: BenchEnv = {
      ...mockEnv,
      getRunLog: () =>
        Effect.send(
          ++logCalls === 1
            ? { lines: ["l1", "l2"], next_offset: 42 }
            : { lines: [], next_offset: 42 },
        ),
      getRun: () => Effect.send(++runCalls === 1 ? runningDetail : terminalDetail),
      listRuns: () => Effect.send([summary]),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "openRun", runId: summary.run_id }, (s) => {
      s.openGeneration = 1;
      s.openRun = {
        runId: summary.run_id,
        detail: null,
        tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
        trials: [],
        games: [],
      };
    });

    // First tick: lines arrive -> backoff stays at 0, next tick at START.
    ts.receive({ tag: "tailTick", generation: 1 });
    await ts.drain();
    ts.receive(
      {
        tag: "tailed",
        generation: 1,
        lines: ["l1", "l2"],
        nextOffset: 42,
        detail: runningDetail,
        trials: [],
      },
      (s) => {
        s.openRun!.detail = runningDetail;
        s.openRun!.tail.lines = ["l1", "l2"];
        s.openRun!.tail.offset = 42;
      },
    );

    // Second tick: run is now terminal -> tail stops, and the runs list is
    // refreshed in the same reduction (this run's status/counts changed).
    ts.advance(tailDelayMs(0));
    ts.receive({ tag: "tailTick", generation: 1 });
    await ts.drain();
    ts.receive(
      {
        tag: "tailed",
        generation: 1,
        lines: [],
        nextOffset: 42,
        detail: terminalDetail,
        trials: [],
      },
      (s) => {
        s.openRun!.detail = terminalDetail;
        s.openRun!.tail.active = false;
        s.runs.status = "pending";
        loadingTuningSessions(s);
      },
    );
    ts.receive(
      {
        tag: "runs",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: [summary] } },
        },
      },
      (s) => {
        s.runs.status = "done";
        s.runs.result = [summary];
      },
    );
    expect(logCalls).toBe(2);
    expect(runCalls).toBe(2);
  });

  it("backs off on consecutive empty polls, by exactly tailDelayMs(idleAttempts)", async () => {
    let runCalls = 0;
    const env: BenchEnv = {
      ...mockEnv,
      getRunLog: () => Effect.send({ lines: [], next_offset: 0 }),
      getRun: () => Effect.send(++runCalls < 3 ? runningDetail : terminalDetail),
      listRuns: () => Effect.send([]),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "openRun", runId: summary.run_id }, (s) => {
      s.openGeneration = 1;
      s.openRun = {
        runId: summary.run_id,
        detail: null,
        tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
        trials: [],
        games: [],
      };
    });
    ts.receive({ tag: "tailTick", generation: 1 });
    await ts.drain();
    ts.receive(
      { tag: "tailed", generation: 1, lines: [], nextOffset: 0, detail: runningDetail, trials: [] },
      (s) => {
        s.openRun!.detail = runningDetail;
        s.openRun!.tail.idleAttempts = 1;
      },
    );

    // One ms short of the backoff: nothing has fired. At exactly the
    // backoff: the tick is delivered.
    ts.advance(tailDelayMs(1) - 1);
    ts.assertDrained();
    ts.advance(1);
    ts.receive({ tag: "tailTick", generation: 1 });
    await ts.drain();
    ts.receive(
      { tag: "tailed", generation: 1, lines: [], nextOffset: 0, detail: runningDetail, trials: [] },
      (s) => {
        s.openRun!.tail.idleAttempts = 2;
      },
    );

    // Third poll observes the terminal status and stops the loop.
    ts.advance(tailDelayMs(2));
    ts.receive({ tag: "tailTick", generation: 1 });
    await ts.drain();
    ts.receive(
      {
        tag: "tailed",
        generation: 1,
        lines: [],
        nextOffset: 0,
        detail: terminalDetail,
        trials: [],
      },
      (s) => {
        s.openRun!.detail = terminalDetail;
        s.openRun!.tail.active = false;
        // Terminal winds the loop down, including the backoff counter.
        s.openRun!.tail.idleAttempts = 0;
        s.runs.status = "pending";
        loadingTuningSessions(s);
      },
    );
    ts.receive(
      {
        tag: "runs",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: [] } },
        },
      },
      (s) => {
        s.runs.status = "done";
        s.runs.result = [];
      },
    );
  });

  it("closeRun drops an in-flight tick's result", async () => {
    const env: BenchEnv = {
      ...mockEnv,
      getRunLog: () => Effect.send({ lines: ["x"], next_offset: 7 }),
      getRun: () => Effect.send(runningDetail),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "openRun", runId: summary.run_id }, (s) => {
      s.openGeneration = 1;
      s.openRun = {
        runId: summary.run_id,
        detail: null,
        tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
        trials: [],
        games: [],
      };
    });
    ts.receive({ tag: "tailTick", generation: 1 });
    // Close while the tick's fetch is still in flight, then let it land.
    ts.send({ tag: "closeRun" }, (s) => {
      s.openRun = null;
    });
    await ts.drain();
    // The tailed arrives with a valid generation but no open run — dropped,
    // no state change, no further tick scheduled.
    ts.receive(
      {
        tag: "tailed",
        generation: 1,
        lines: ["x"],
        nextOffset: 7,
        detail: runningDetail,
        trials: [],
      },
      () => {},
    );
  });

  it("drops a stale generation's tailed after a different run is opened", async () => {
    const env: BenchEnv = {
      ...mockEnv,
      getRunLog: () => Effect.send({ lines: ["x"], next_offset: 7 }),
      // Terminal from the start, so the surviving generation's tailed stops
      // the loop instead of scheduling a timer this test would have to reap.
      getRun: () => Effect.send(terminalDetail),
      listRuns: () => Effect.send([]),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "openRun", runId: "run-a" }, (s) => {
      s.openGeneration = 1;
      s.openRun = {
        runId: "run-a",
        detail: null,
        tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
        trials: [],
        games: [],
      };
    });
    ts.receive({ tag: "tailTick", generation: 1 });
    // Open a different run while run-a's tick is still in flight.
    ts.send({ tag: "openRun", runId: "run-b" }, (s) => {
      s.openGeneration = 2;
      s.openRun = {
        runId: "run-b",
        detail: null,
        tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
        trials: [],
        games: [],
      };
    });
    ts.receive({ tag: "tailTick", generation: 2 });
    await ts.drain();
    // Run-a's tailed lands first (its fetch started first) and is dropped.
    ts.receive(
      {
        tag: "tailed",
        generation: 1,
        lines: ["x"],
        nextOffset: 7,
        detail: terminalDetail,
        trials: [],
      },
      () => {},
    );
    // Run-b's tailed applies normally.
    ts.receive(
      {
        tag: "tailed",
        generation: 2,
        lines: ["x"],
        nextOffset: 7,
        detail: terminalDetail,
        trials: [],
      },
      (s) => {
        s.openRun!.detail = terminalDetail;
        s.openRun!.tail.lines = ["x"];
        s.openRun!.tail.offset = 7;
        s.openRun!.tail.active = false;
        s.runs.status = "pending";
        loadingTuningSessions(s);
      },
    );
    ts.receive(
      {
        tag: "runs",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: [] } },
        },
      },
      (s) => {
        s.runs.status = "done";
        s.runs.result = [];
      },
    );
  });

  it("retries failed ticks with backoff and gives up after TAIL_MAX_FAILURES", async () => {
    const env: BenchEnv = {
      ...mockEnv,
      getRunLog: () => Effect.fromPromise(() => Promise.reject(new Error("boom"))),
      getRun: () => Effect.fromPromise(() => Promise.reject(new Error("boom"))),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "openRun", runId: summary.run_id }, (s) => {
      s.openGeneration = 1;
      s.openRun = {
        runId: summary.run_id,
        detail: null,
        tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
        trials: [],
        games: [],
      };
    });

    for (let f = 1; f <= TAIL_MAX_FAILURES; f++) {
      ts.receive({ tag: "tailTick", generation: 1 });
      await ts.drain();
      ts.receive({ tag: "tailFailed", generation: 1, error: "Error: boom" }, (s) => {
        s.openRun!.tail.error = "Error: boom";
        s.openRun!.tail.failures = f;
        if (f < TAIL_MAX_FAILURES) {
          s.openRun!.tail.idleAttempts = f;
        } else {
          s.openRun!.tail.active = false;
        }
      });
      if (f < TAIL_MAX_FAILURES) {
        ts.advance(tailDelayMs(f));
      }
    }
    // Gave up: no further tick is scheduled, and the error stays visible.
    expect(ts.getState().openRun?.tail.active).toBe(false);
    expect(ts.getState().openRun?.tail.error).toBe("Error: boom");
  });
});

// ── tuner kinds ─────────────────────────────────────────────────────────────

describe("benchReducer / tunerKinds", () => {
  const tlKind: TunerGameInfo = {
    game: "traffic-lights",
    tuner: {
      id: "rave",
      baselines: ["strong"],
      eval_rounds: 20,
      parameters: [{ name: "c", type: "float", bounds: [0, 3], default: 1.4 }],
      conditions: [],
      game_config: {},
    },
  };

  it("request -> submitted('done') populates the tuner metadata", () => {
    const env: BenchEnv = { ...mockEnv, getTunerKinds: () => Effect.send([tlKind]) };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "tunerKinds", action: { tag: "request" } }, (s) => {
      s.tunerKinds.status = "pending";
    });
    ts.receive(
      {
        tag: "tunerKinds",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: [tlKind] } },
        },
      },
      (s) => {
        s.tunerKinds.status = "done";
        s.tunerKinds.result = [tlKind];
      },
    );
  });
});

// ── Launch / stop ───────────────────────────────────────────────────────────

describe("benchReducer / launch", () => {
  it("request -> submitted('done') stores the response and refreshes the runs list", () => {
    const launchResponse: LaunchResponse = {
      run_id: "new-run",
      pid: 4321,
      log_path: "/x/log.jsonl",
    };
    const seen: { kind: string; game: string; config?: unknown }[] = [];
    const env: BenchEnv = {
      ...mockEnv,
      launchRun: (kind, game, config) => {
        seen.push({ kind, game, config });
        return Effect.send(launchResponse);
      },
      listRuns: () => Effect.send([summary]),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send(
      {
        tag: "launch",
        action: { tag: "request", kind: "tuner", game: "druid", config: { overrides: {} } },
      },
      (s) => {
        s.launch.status = "pending";
      },
    );
    ts.receive(
      {
        tag: "launch",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: launchResponse } },
        },
      },
      (s) => {
        s.launch.status = "done";
        s.launch.result = launchResponse;
        // The completed launch refreshes the runs list in the same
        // reduction, so the new run appears without a manual reload.
        s.runs.status = "pending";
        loadingTuningSessions(s);
      },
    );
    ts.receive(
      {
        tag: "runs",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: [summary] } },
        },
      },
      (s) => {
        s.runs.status = "done";
        s.runs.result = [summary];
      },
    );
    expect(seen).toEqual([{ kind: "tuner", game: "druid", config: { overrides: {} } }]);
  });
});

describe("benchReducer / stopRun", () => {
  it("stopRun -> stopFinished refreshes the runs list", () => {
    const env: BenchEnv = {
      ...mockEnv,
      stopRun: () => Effect.send({ run_id: summary.run_id, message: "stop signal sent" }),
      listRuns: () => Effect.send([summary]),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "stopRun", runId: summary.run_id });
    ts.receive({ tag: "stopFinished", runId: summary.run_id }, (s) => {
      s.runs.status = "pending";
      loadingTuningSessions(s);
    });
    ts.receive(
      {
        tag: "runs",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: [summary] } },
        },
      },
      (s) => {
        s.runs.status = "done";
        s.runs.result = [summary];
      },
    );
  });

  it("a rejected stop lands in stopError", async () => {
    const env: BenchEnv = {
      ...mockEnv,
      stopRun: () => Effect.fromPromise(() => Promise.reject(new Error("nope"))),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "stopRun", runId: summary.run_id });
    await ts.drain();
    ts.receive({ tag: "stopFailed", runId: summary.run_id, error: "Error: nope" }, (s) => {
      s.stopError = "Error: nope";
    });
  });
});
