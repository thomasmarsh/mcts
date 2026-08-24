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
  emptyExperimentSpec,
  type BenchEnv,
} from "../src/reducer.js";
import { initialBenchState } from "../src/state.js";
import type { Experiment, ExperimentCell, LaunchResponse, LeaderboardEntry, Project, RunDetail, RunFilters, RunSummary, TunerGameInfo } from "../src/types.js";
import { deriveSeed, expandExperimentSpec } from "../src/experiment-grid.js";

// ── Fixtures ────────────────────────────────────────────────────────────────

const summary: RunSummary = {
  run_id: "rr-druid-20260101T000000-abc1234",
  kind: "round_robin",
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
    kind: "round_robin",
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
const terminalDetail = makeDetail({ status: "completed", ended_at: "2026-01-01T01:00:00Z", exit_code: 0 });
function loadingTuningSessions(state: ReturnType<typeof initialBenchState>): void {
  state.tuningNavigation.list.status = "loading";
  state.tuningNavigation.list.generation += 1;
}

describe("experiment grid expansion", () => {
  it("matches the deterministic 2-game by 2-budget by 3-variant acceptance shape", () => {
    const plan = expandExperimentSpec({
      version: 1,
      games: [{ game: "game-a", game_config: null }, { game: "game-b", game_config: null }],
      baseline: { id: "base", label: "Base", config: {} },
      variants: [{ id: "v1", label: "V1", config: {} }, { id: "v2", label: "V2", config: {} }, { id: "v3", label: "V3", config: {} }],
      budgets: [{ kind: "iterations", value: 10 }, { kind: "time_per_move_ms", value: 20 }],
      rounds_per_cell: 2, base_seed: 42, max_parallel_cells: 2,
    });
    expect(plan.cells).toHaveLength(12);
    expect(plan.total_planned_games).toBe(48);
    expect(plan.cells.slice(0, 3).map((cell) => [cell.cell_id, cell.game, cell.budget.kind, cell.variant_id])).toEqual([
      ["cell-000001", "game-a", "iterations", "v1"], ["cell-000002", "game-a", "iterations", "v2"], ["cell-000003", "game-a", "iterations", "v3"],
    ]);
    expect(plan.cells[0]?.cell_seed).toBe(7294331206661666);
    expect(plan.cells[0]?.round_seeds).toEqual([8360105604253074, 5482876856761435]);
    expect(deriveSeed(42, 1)).toBe(6529064058449557);
  });
});

const mockEnv: BenchEnv = {
  listProjects: () => Effect.none(),
  createProject: () => Effect.none(),
  getProject: () => Effect.none(),
  updateProject: () => Effect.none(),
  listExperiments: () => Effect.none(),
  createExperiment: () => Effect.none(),
  getExperiment: () => Effect.none(),
  updateExperiment: () => Effect.none(),
  launchExperiment: () => Effect.none(),
  getRunCells: () => Effect.send([]),
  listRuns: () => Effect.none(),
  getRun: () => Effect.none(),
  getRunLog: () => Effect.none(),
  getRunStdout: () => Effect.none(),
  downloadFile: () => Effect.none(),
  getLeaderboard: () => Effect.none(),
  fetchCommitTrends: () => Effect.none(),
  launchRun: () => Effect.none(),
  stopRun: () => Effect.none(),
  getBenchKinds: () => Effect.none(),
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
      { tag: "runs", action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [summary] } } } },
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
      { tag: "runs", action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [summary] } } } },
      (s) => {
        s.runs.status = "done";
        s.runs.result = [summary];
      },
    );
    expect(seen).toEqual([{ status: "running", game: "druid" }]);
  });
});

describe("benchReducer / persisted experiments", () => {
  const project: Project = {
    project_id: "project-1", name: "Nim study", description: "small", archived: false,
    created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
  };
  const experiment: Experiment = {
    experiment_id: "experiment-1", project_id: project.project_id, name: "Baseline", description: "one cell",
    spec: emptyExperimentSpec(), created_at: project.created_at, updated_at: project.updated_at,
  };
  const cell: ExperimentCell = {
    cell_id: "cell-1", cell_seed: 7294331206661666, game: "nim", game_config: null, variant_id: "variant", variant_label: "Variant",
    candidate_config: {}, baseline_id: "baseline", baseline_label: "Baseline", baseline_config: {},
    budget: { kind: "iterations", value: 1 }, rounds: 1, planned_games: 2, completed_games: 1,
    status: "running", started_at: project.created_at, ended_at: null, error: null,
    wins: 1, losses: 0, draws: 0, win_rate: 1, ci_lower: 0.2, ci_upper: 1,
  };

  it("keeps project, experiment, launch, run, and cell transitions in reducer state", async () => {
    const env: BenchEnv = {
      ...mockEnv,
      createProject: () => Effect.send(project),
      listProjects: () => Effect.none(),
      listExperiments: () => Effect.none(),
      createExperiment: () => Effect.send(experiment),
      launchExperiment: () => Effect.send({ run_id: "experiment-run", pid: 7, log_path: "/tmp/log.jsonl" }),
      getRunCells: () => Effect.send([cell]),
      getRunLog: () => Effect.send({ lines: [], next_offset: 0 }),
      getRun: () => Effect.send({ ...terminalDetail, run_id: "experiment-run", kind: "experiment", game: "nim", experiment_id: experiment.experiment_id, project_id: project.project_id, experiment_spec: experiment.spec }),
      listRuns: () => Effect.none(),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "projectDraft", name: project.name, description: project.description });
    ts.send({ tag: "createProject" });
    await ts.drain();
    ts.receive({ tag: "projectCreated", project });
    ts.send({ tag: "newExperiment" }, (s) => {
      s.experimentDraft = { name: "", description: "", spec: emptyExperimentSpec() };
    });
    ts.send({ tag: "experimentDraft", draft: { name: experiment.name, description: experiment.description, spec: experiment.spec } });
    ts.send({ tag: "saveExperiment" });
    await ts.drain();
    ts.receive({ tag: "experimentSaved", experiment }, (s) => {
      s.selectedExperimentId = experiment.experiment_id;
      s.selectedExperiment = experiment;
      s.experimentDraft = { name: experiment.name, description: experiment.description, spec: experiment.spec };
      s.experimentSavedDraft = { name: experiment.name, description: experiment.description, spec: experiment.spec };
      s.experimentSaveStatus = "idle";
      s.experimentFieldErrors = {};
    });
    ts.send({ tag: "launchExperiment" });
    await ts.drain();
    ts.receive({ tag: "experimentLaunched", response: { run_id: "experiment-run", pid: 7, log_path: "/tmp/log.jsonl" } }, (s) => {
      s.activeTab = "runs";
      s.experimentLaunchStatus = "idle";
    });
    ts.receive({ tag: "openRun", runId: "experiment-run" });
    ts.receive({ tag: "tailTick", generation: 1 });
    await ts.drain();
    ts.receive({ tag: "tailed", generation: 1, lines: [], nextOffset: 0, detail: { ...terminalDetail, run_id: "experiment-run", kind: "experiment", game: "nim", experiment_id: experiment.experiment_id, project_id: project.project_id, experiment_spec: experiment.spec }, trials: [], cells: [cell] });
    ts.send({ tag: "openCell", cellId: "cell-1" }, (s) => { s.selectedCellId = "cell-1"; });
    expect(ts.getState().selectedCellId).toBe("cell-1");
    expect(ts.getState().openRun?.cells[0]?.completed_games).toBe(1);
  });

  it("surfaces one-cell validation errors before calling the API", () => {
    let called = false;
    const env: BenchEnv = { ...mockEnv, createExperiment: () => { called = true; return Effect.send(experiment); } };
    const draft = initialBenchState();
    draft.selectedProjectId = project.project_id;
    draft.experimentDraft = { name: "bad", description: "", spec: { ...emptyExperimentSpec(), variants: [] } };
    benchReducer(draft, { tag: "saveExperiment" }, env);
    expect(called).toBe(false);
    expect(draft.experimentFieldErrors["spec.variants"]).toContain("variants");
  });

  it("launches only the last successfully saved draft and rejects edits until they are saved", () => {
    let createCalls = 0;
    let launchCalls = 0;
    const saved = { ...experiment, name: "Saved definition" };
    const env: BenchEnv = {
      ...mockEnv,
      createExperiment: () => { createCalls++; return Effect.none(); },
      launchExperiment: () => { launchCalls++; return Effect.none(); },
    };
    const state = initialBenchState();
    state.selectedProjectId = project.project_id;
    state.experimentDraft = { name: saved.name, description: saved.description, spec: saved.spec };

    benchReducer(state, { tag: "saveExperiment" }, env);
    expect(state.experimentSaveStatus).toBe("saving");
    benchReducer(state, { tag: "saveExperiment" }, env);
    expect(createCalls).toBe(1);

    benchReducer(state, { tag: "experimentSaved", experiment: saved }, env);
    expect(state.experimentSaveStatus).toBe("idle");
    expect(state.experimentSavedDraft).toEqual(state.experimentDraft);
    benchReducer(state, { tag: "launchExperiment" }, env);
    expect(state.experimentLaunchStatus).toBe("launching");
    expect(launchCalls).toBe(1);

    state.experimentLaunchStatus = "idle";
    benchReducer(state, { tag: "experimentDraft", draft: { ...state.experimentDraft!, description: "changed" } }, env);
    benchReducer(state, { tag: "launchExperiment" }, env);
    expect(state.experimentRunError).toContain("Save the current experiment");
    expect(launchCalls).toBe(1);
  });

  it("changes a game and installs the metadata-provided default configuration", () => {
    const state = initialBenchState();
    state.experimentDraft = { name: "Game change", description: "", spec: emptyExperimentSpec("nim") };
    benchReducer(state, { tag: "experimentGameChanged", game: "druid", gameConfig: { size: 7 } }, mockEnv);
    expect(state.experimentDraft?.spec.games[0]).toEqual({ game: "druid", game_config: { size: 7 } });
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
        Effect.send(++logCalls === 1 ? { lines: ["l1", "l2"], next_offset: 42 } : { lines: [], next_offset: 42 }),
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
        cells: [],
        games: [],
      };
    });

    // First tick: lines arrive -> backoff stays at 0, next tick at START.
    ts.receive({ tag: "tailTick", generation: 1 });
    await ts.drain();
    ts.receive(
      { tag: "tailed", generation: 1, lines: ["l1", "l2"], nextOffset: 42, detail: runningDetail, trials: [] },
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
      { tag: "tailed", generation: 1, lines: [], nextOffset: 42, detail: terminalDetail, trials: [] },
      (s) => {
        s.openRun!.detail = terminalDetail;
        s.openRun!.tail.active = false;
        s.runs.status = "pending";
        loadingTuningSessions(s);
      },
    );
    ts.receive(
      { tag: "runs", action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [summary] } } } },
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
        cells: [],
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
      { tag: "tailed", generation: 1, lines: [], nextOffset: 0, detail: terminalDetail, trials: [] },
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
      { tag: "runs", action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [] } } } },
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
        cells: [],
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
    ts.receive({ tag: "tailed", generation: 1, lines: ["x"], nextOffset: 7, detail: runningDetail, trials: [] }, () => {});
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
        cells: [],
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
        cells: [],
        games: [],
      };
    });
    ts.receive({ tag: "tailTick", generation: 2 });
    await ts.drain();
    // Run-a's tailed lands first (its fetch started first) and is dropped.
    ts.receive({ tag: "tailed", generation: 1, lines: ["x"], nextOffset: 7, detail: terminalDetail, trials: [] }, () => {});
    // Run-b's tailed applies normally.
    ts.receive(
      { tag: "tailed", generation: 2, lines: ["x"], nextOffset: 7, detail: terminalDetail, trials: [] },
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
      { tag: "runs", action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [] } } } },
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
        cells: [],
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
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [tlKind] } } },
      },
      (s) => {
        s.tunerKinds.status = "done";
        s.tunerKinds.result = [tlKind];
      },
    );
  });
});

// ── Leaderboard ─────────────────────────────────────────────────────────────

describe("benchReducer / leaderboard", () => {
  const entry: LeaderboardEntry = {
    strategy: "strong",
    total: 3,
    wins: 2,
    losses: 0,
    draws: 1,
    win_rate: 2.5 / 3,
    ci_lower: 0.3,
    ci_upper: 0.99,
  };

  it("request -> submitted('done') populates the entries", () => {
    const env: BenchEnv = { ...mockEnv, getLeaderboard: () => Effect.send([entry]) };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "leaderboard", action: { tag: "request" } }, (s) => {
      s.leaderboard.status = "pending";
    });
    ts.receive(
      { tag: "leaderboard", action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [entry] } } } },
      (s) => {
        s.leaderboard.status = "done";
        s.leaderboard.result = [entry];
      },
    );
  });

  it("setLeaderboardFilters stores the filters and refetches with them", () => {
    const seen: { game: string | null; gitSha: string | null; since: string | null }[] = [];
    const env: BenchEnv = {
      ...mockEnv,
      getLeaderboard: (filters) => {
        seen.push({ ...filters });
        return Effect.send([entry]);
      },
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "setLeaderboardFilters", game: "druid", gitSha: "abc1234", since: null }, (s) => {
      s.leaderboardFilters = { game: "druid", gitSha: "abc1234", since: null };
      s.leaderboard.status = "pending";
    });
    ts.receive(
      { tag: "leaderboard", action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [entry] } } } },
      (s) => {
        s.leaderboard.status = "done";
        s.leaderboard.result = [entry];
      },
    );
    expect(seen).toEqual([{ game: "druid", gitSha: "abc1234", since: null }]);
  });

  it("fetchCommitTrends populates commitTrends on success", async () => {
    const trendData = {
      abc1234: [{ strategy: "strong", total: 2, wins: 1, losses: 0, draws: 1, win_rate: 0.75, ci_lower: 0.3, ci_upper: 0.99 }],
      def5678: [{ strategy: "strong", total: 5, wins: 3, losses: 1, draws: 1, win_rate: 0.7, ci_lower: 0.4, ci_upper: 0.92 }],
    };
    const env: BenchEnv = {
      ...mockEnv,
      fetchCommitTrends: () => Effect.send(trendData),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "fetchCommitTrends", game: "druid" }, (s) => {
      s.commitTrends = { data: {}, shas: [], status: "loading", error: null };
    });
    await ts.drain();
    ts.receive(
      { tag: "commitTrendsLoaded", data: trendData, shas: ["def5678", "abc1234"] },
      (s) => {
        s.commitTrends = { data: trendData, shas: ["def5678", "abc1234"], status: "done", error: null };
      },
    );
  });

  it("fetchCommitTrends stores error on failure", async () => {
    const env: BenchEnv = {
      ...mockEnv,
      fetchCommitTrends: () => Effect.fromPromise(() => Promise.reject(new Error("boom"))),
    };
    const ts = createTestStore(benchReducer, env, initialBenchState());

    ts.send({ tag: "fetchCommitTrends", game: "druid" }, (s) => {
      s.commitTrends = { data: {}, shas: [], status: "loading", error: null };
    });
    await ts.drain();
    ts.receive(
      { tag: "commitTrendsFailed", error: "Error: boom" },
      (s) => {
        s.commitTrends = { data: {}, shas: [], status: "error", error: "Error: boom" };
      },
    );
  });
});

// ── Launch / stop ───────────────────────────────────────────────────────────

describe("benchReducer / launch", () => {
  it("request -> submitted('done') stores the response and refreshes the runs list", () => {
    const launchResponse: LaunchResponse = { run_id: "new-run", pid: 4321, log_path: "/x/log.jsonl" };
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
      { tag: "launch", action: { tag: "request", kind: "round_robin", game: "druid", config: { rounds: 2 } } },
      (s) => {
        s.launch.status = "pending";
      },
    );
    ts.receive(
      {
        tag: "launch",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: launchResponse } } },
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
      { tag: "runs", action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [summary] } } } },
      (s) => {
        s.runs.status = "done";
        s.runs.result = [summary];
      },
    );
    expect(seen).toEqual([{ kind: "round_robin", game: "druid", config: { rounds: 2 } }]);
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
      { tag: "runs", action: { tag: "job", action: { tag: "submitted", result: { status: "done", result: [summary] } } } },
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

describe("benchReducer / experiment exports", () => {
  function exportState() {
    const state = initialBenchState();
    const spec = emptyExperimentSpec("nim");
    state.openRun = {
      runId: "export-run",
      detail: makeDetail({ kind: "experiment", experiment_spec: spec, run_id: "export-run", status: "completed" }),
      tail: { lines: [], offset: 0, active: false, error: null, idleAttempts: 0, failures: 0 },
      trials: [], games: [],
      cells: [{ cell_id: "export-cell", cell_seed: 1, game: "nim", game_config: null, variant_id: "variant", variant_label: "Variant", candidate_config: {}, baseline_id: "baseline", baseline_label: "Baseline", baseline_config: {}, budget: spec.budgets[0]!, rounds: 1, planned_games: 2, completed_games: 0, status: "pending", started_at: null, ended_at: null, error: null, wins: 0, losses: 0, draws: 0, win_rate: 0.5, ci_lower: 0, ci_upper: 1 }],
    };
    return state;
  }

  it("downloads a deterministic snapshot once while pending", async () => {
    const downloads: Array<[string, string, string]> = [];
    const env: BenchEnv = { ...mockEnv, downloadFile: (filename, mimeType, contents) => { downloads.push([filename, mimeType, contents]); return Effect.send(undefined); } };
    const ts = createTestStore(benchReducer, env, exportState());
    ts.send({ tag: "exportExperimentRun", format: "json" }, (s) => { s.experimentExportStatus = "pending"; s.experimentExportError = null; });
    ts.send({ tag: "exportExperimentRun", format: "json" });
    expect(downloads).toHaveLength(1);
    expect(downloads[0]![0]).toBe("experiment-export-run.json");
    expect(downloads[0]![1]).toBe("application/json");
    ts.receive({ tag: "experimentExportFinished" }, (s) => { s.experimentExportStatus = "idle"; });
  });

  it("clears pending state on download failure without losing the open run", async () => {
    const env: BenchEnv = { ...mockEnv, downloadFile: () => Effect.fromPromise(() => Promise.reject(new Error("download failed"))) };
    const ts = createTestStore(benchReducer, env, exportState());
    ts.send({ tag: "exportExperimentRun", format: "csv" }, (s) => { s.experimentExportStatus = "pending"; s.experimentExportError = null; });
    await ts.drain();
    ts.receive({ tag: "experimentExportFailed", error: "Error: download failed" }, (s) => {
      s.experimentExportStatus = "idle";
      s.experimentExportError = "Error: download failed";
    });
    expect(ts.getState().openRun?.runId).toBe("export-run");
  });
});
