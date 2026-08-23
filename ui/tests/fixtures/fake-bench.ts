// tests/fixtures/fake-bench.ts — Mock bench API responses for component tests.
// Mirrors tests/fixtures/fake-game.tsx's pattern: lightweight, no real
// server or browser dependencies.

import { Effect } from "@mcts/core";
import type { BenchEnv } from "@mcts/bench";
import type {
  BenchKindInfo,
  ChainRung,
  RunDetail,
  RunLogResponse,
  RunSummary,
  LaunchResponse,
  StopResponse,
  LeaderboardEntry,
  TunerGameInfo,
  TrialRow,
  GameTraceSummary,
  GameMove,
} from "@mcts/bench";

export const FAKE_RUN_ID = "rr-druid-20260101T000000-abc1234";

export const fakeRunSummaries: RunSummary[] = [
  {
    run_id: FAKE_RUN_ID,
    kind: "round_robin",
    project_id: null,
    experiment_id: null,
    game: "druid",
    label: null,
    git_sha: "abc1234def5678",
    git_dirty: false,
    host: "testhost",
    pid: null,
    started_at: "2026-01-01T00:00:00Z",
    ended_at: "2026-01-01T01:00:00Z",
    status: "completed",
    match_count: 10,
    trial_count: 0,
  },
  {
    run_id: "rr-druid-20260201T000000-def5678",
    kind: "round_robin",
    project_id: null,
    experiment_id: null,
    game: "druid",
    label: "test run",
    git_sha: "def5678abc1234",
    git_dirty: false,
    host: "testhost",
    pid: 12345,
    started_at: "2026-02-01T00:00:00Z",
    ended_at: null,
    status: "running",
    match_count: 5,
    trial_count: 2,
  },
];

export const fakeRunDetail: RunDetail = {
  run_id: FAKE_RUN_ID,
  kind: "round_robin",
  project_id: null,
  experiment_id: null,
  experiment_spec: null,
  game: "druid",
  label: null,
  config: { strategies: ["strong", "master"], rounds: 1 },
  git_sha: "abc1234def5678",
  git_dirty: false,
  host: "testhost",
  pid: null,
  started_at: "2026-01-01T00:00:00Z",
  ended_at: "2026-01-01T01:00:00Z",
  status: "completed",
  log_path: "/tmp/nope/log.jsonl",
  exit_code: 0,
  match_count: 10,
  trial_count: 0,
  incumbent: null,
};

export const fakeRunningDetail: RunDetail = {
  ...fakeRunDetail,
  run_id: "rr-druid-20260201T000000-def5678",
  status: "running",
  ended_at: null,
  match_count: 5,
  trial_count: 2,
};

export const FAKE_tuner_RUN_ID = "tuner-traffic-lights-20260301T000000-abc1234";

export const fakeTunerRunDetail: RunDetail = {
  ...fakeRunDetail,
  run_id: FAKE_tuner_RUN_ID,
  kind: "tuner",
  game: "traffic-lights",
  config: {
    overrides: ["optimizer.n_trials=50", "target.baselines=[\"flat_mc\"]"],
    baseline_settings: { flat_mc: { family: "flat_mc", q_init: "Infinity" } },
  },
  trial_count: 3,
  incumbent: { config: { family: "rave", c: 0.7 }, cost: 0.2 },
};

// Mirrors `mcts_tune::strategy_tuner_info`'s real shape: `family` is a
// top-level categorical gating other parameters via two levels of
// `TunerCondition`s (family -> schedule -> rave, family -> rave_ucb -> c),
// not a single fixed family's flat schema. Trimmed to a handful of
// families/params rather than the full ~14-family catalog -- enough to
// exercise multi-level conditions and a non-RAVE best trial (see
// fakeTrialRows below).
export const fakeTunerKinds: TunerGameInfo[] = [
  {
    game: "traffic-lights",
    tuner: {
      id: "strategy",
      baselines: ["strong"],
      eval_rounds: 20,
      parameters: [
        { name: "family", type: "categorical", choices: ["ucb1", "ucb1_tuned", "rave"], default: "rave" },
        { name: "final_action", type: "categorical", choices: ["max_avg", "secure_child", "robust_child"], default: "robust_child" },
        { name: "epsilon", type: "float", bounds: [0, 1], default: 0.1 },
        { name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 },
        { name: "rave", type: "int", bounds: [0, 2000], default: 700 },
        { name: "schedule", type: "categorical", choices: ["hand_selected", "min_mse", "threshold"], default: "threshold" },
        { name: "rave_ucb", type: "categorical", choices: ["none", "ucb1", "tuned"], default: "tuned" },
      ],
      conditions: [
        { if: { family: ["ucb1", "ucb1_tuned", "rave"] }, then: ["final_action"] },
        { if: { family: ["ucb1_tuned", "rave"] }, then: ["epsilon"] },
        { if: { family: "rave" }, then: ["schedule", "rave_ucb"] },
        { if: { schedule: "threshold" }, then: ["rave"] },
        { if: { rave_ucb: ["ucb1", "tuned"] }, then: ["c"] },
      ],
      // No game-setup config -- traffic-lights' board is fixed at compile
      // time, so the "Game config" field must stay hidden for this game.
      game_config: {},
    },
  },
  // A minimal second entry standing in for Druid: a genuinely non-empty
  // `game_config` is what makes the "Game config" field render at all.
  {
    game: "druid",
    tuner: {
      id: "strategy",
      baselines: ["strong", "master"],
      eval_rounds: 20,
      parameters: [
        { name: "family", type: "categorical", choices: ["ucb1", "rave"], default: "rave" },
      ],
      conditions: [],
      game_config: { size: { w: 5, h: 5 } },
    },
  },
];

// Best trial (#2, cost 0.3) is deliberately `family: "ucb1_tuned"`, unlike
// the run's `flat_mc` baseline -- proves the run-detail baseline comparison
// and the trial table's Family column work across different configs.
export const fakeTrialRows: TrialRow[] = [
  {
    trial_id: 1,
    ts: "2026-03-01T00:00:01Z",
    config: { family: "rave", final_action: "robust_child", epsilon: 0.1, schedule: "threshold", rave: 700, rave_ucb: "tuned", c: 1.4 },
    seed: 0,
    cost: 0.55,
    extra: null,
  },
  {
    trial_id: 2,
    ts: "2026-03-01T00:01:00Z",
    config: { family: "ucb1_tuned", final_action: "max_avg", epsilon: 0.2 },
    seed: 0,
    cost: 0.3,
    extra: null,
  },
  {
    trial_id: 3,
    ts: "2026-03-01T00:02:00Z",
    config: { family: "ucb1", final_action: "robust_child" },
    seed: 0,
    cost: 0.4,
    extra: null,
  },
];

export const fakeGameTraces: GameTraceSummary[] = [
  {
    game_seq: 10,
    ply_count: 3,
    started_at: "2026-01-01T00:00:00Z",
    ended_at: "2026-01-01T00:00:02Z",
    strategy_a: "strong",
    strategy_b: "master",
    outcome: "win",
    winner: "Black",
  },
];

export const fakeGameMoves: GameMove[] = [
  { ply: 0, ts: "2026-01-01T00:00:00Z", state: { player: "Black", board: [] }, mv: null, player: null },
  { ply: 1, ts: "2026-01-01T00:00:01Z", state: { player: "White", board: [] }, mv: ["Black", 0], player: "Black" },
  { ply: 2, ts: "2026-01-01T00:00:02Z", state: { player: "Black", board: [] }, mv: ["White", 1], player: "White" },
];

// A variant trial set where trial #2's exact config is re-evaluated twice
// more (trials #4/#5, different seeds/costs) -- exercises the confidence
// band: those three rows must pool into one group with a tighter CI than
// any single evaluation, rather than rendering as three independent points.
export const fakeTrialRowsWithRepeats: TrialRow[] = [
  ...fakeTrialRows,
  {
    trial_id: 4,
    ts: "2026-03-01T00:03:00Z",
    config: { family: "ucb1_tuned", final_action: "max_avg", epsilon: 0.2 },
    seed: 1,
    cost: 0.25,
    extra: null,
  },
  {
    trial_id: 5,
    ts: "2026-03-01T00:04:00Z",
    config: { family: "ucb1_tuned", final_action: "max_avg", epsilon: 0.2 },
    seed: 2,
    cost: 0.35,
    extra: null,
  },
];

// Same `config` evaluated against two different baseline instances (tuner's
// `Scenario(instances=...)`, e.g. druid's "strong"/"master") -- unlike
// fakeTrialRowsWithRepeats' same-instance re-evaluation, these must render
// as two *separate* confidence-band groups, not pool together, since a
// config's win rate against one baseline says nothing about its win rate
// against another.
export const fakeTrialRowsMultiInstance: TrialRow[] = [
  {
    trial_id: 1,
    ts: "2026-03-01T00:00:01Z",
    config: { family: "rave", final_action: "robust_child", epsilon: 0.1, schedule: "threshold", rave: 700, rave_ucb: "tuned", c: 1.4 },
    seed: 0,
    cost: 0.1,
    extra: { instance: "strong" },
  },
  {
    trial_id: 2,
    ts: "2026-03-01T00:01:00Z",
    config: { family: "rave", final_action: "robust_child", epsilon: 0.1, schedule: "threshold", rave: 700, rave_ucb: "tuned", c: 1.4 },
    seed: 0,
    cost: 0.6,
    extra: { instance: "master" },
  },
];

export const fakeKinds: BenchKindInfo[] = [
  {
    kind: "round_robin",
    label: "Round Robin",
    description: "Every strategy plays every other strategy.",
    games: [
      {
        game: "druid",
        strategies: [
          { id: "strong", label: "Strong", description: "3s per move" },
          { id: "master", label: "Master", description: "8s per move" },
          { id: "1s-ucb1", label: "1s UCB1", description: "Plain UCB1" },
        ],
      },
    ],
  },
  {
    kind: "tuner",
    label: "Tuner Tuning",
    description: "Runs a tuner hyperparameter-optimization sweep.",
    // Mirrors the real server: tuner's per-game info comes from
    // GET /api/bench/tuner/kinds (fakeTunerKinds), not this list.
    games: [],
  },
];

export function createMockBenchEnv(overrides?: Partial<BenchEnv>): BenchEnv {
  const base: BenchEnv = {
    listProjects: () => Effect.send([]),
    createProject: () => Effect.send({ project_id: "project-1", name: "Test", description: "", archived: false, created_at: "", updated_at: "" }),
    getProject: () => Effect.send({ project_id: "project-1", name: "Test", description: "", archived: false, created_at: "", updated_at: "" }),
    updateProject: () => Effect.send({ project_id: "project-1", name: "Test", description: "", archived: false, created_at: "", updated_at: "" }),
    listExperiments: () => Effect.send([]),
    createExperiment: (_projectId, body) => Effect.send({ experiment_id: "experiment-1", project_id: "project-1", name: body.name, description: body.description, spec: body.spec, created_at: "", updated_at: "" }),
    getExperiment: () => Effect.send({ experiment_id: "experiment-1", project_id: "project-1", name: "Experiment", description: "", spec: { version: 1, games: [{ game: "nim", game_config: null }], baseline: { id: "base", label: "Base", config: {} }, variants: [{ id: "variant", label: "Variant", config: {} }], budgets: [{ kind: "iterations", value: 25 }], rounds_per_cell: 1, base_seed: 42, max_parallel_cells: 1 }, created_at: "", updated_at: "" }),
    updateExperiment: (_id, body) => Effect.send({ experiment_id: "experiment-1", project_id: "project-1", name: body.name, description: body.description, spec: body.spec, created_at: "", updated_at: "" }),
    launchExperiment: () => Effect.send({ run_id: FAKE_RUN_ID, pid: 1, log_path: "" }),
    getRunCells: () => Effect.send([]),
    listRuns: () => Effect.send(fakeRunSummaries),
    getRun: (runId: string) =>
      Effect.send(
        runId === FAKE_RUN_ID ? fakeRunDetail : runId === FAKE_tuner_RUN_ID ? fakeTunerRunDetail : fakeRunningDetail,
      ),
    getRunLog: (_runId: string, _since: number): Effect<RunLogResponse> =>
      Effect.send({ lines: ['{"type":"match_result","seq":1}'], next_offset: 42 }),
    getLeaderboard: (): Effect<LeaderboardEntry[]> => Effect.send([]),
    launchRun: (_kind: string, _game: string, _config?: unknown): Effect<LaunchResponse> =>
      Effect.send({ run_id: "new-run-123", pid: 99999, log_path: "/tmp/new/log.jsonl" }),
    stopRun: (_runId: string): Effect<StopResponse> =>
      Effect.send({ run_id: "stopped-run", message: "stopped" }),
    resumeRun: (_runId: string, _nTrials: number, _nWorkers?: number): Effect<LaunchResponse> =>
      Effect.send({ run_id: "resumed-run-123", pid: 99998, log_path: "/tmp/resumed/log.jsonl" }),
    advanceBaseline: (_runId: string, _nTrials?: number, _nWorkers?: number): Effect<LaunchResponse> =>
      Effect.send({ run_id: "advanced-run-123", pid: 99997, log_path: "/tmp/advanced/log.jsonl" }),
    getBenchKinds: () => Effect.send(fakeKinds),
    getTunerKinds: () => Effect.send(fakeTunerKinds),
    listTuningSessions: () => Effect.send({ schema_version: 1, sessions: [] }),
    getTuningSession: () => Effect.none(),
    getRunTrials: (_runId: string, _limit?: number): Effect<TrialRow[]> => Effect.send(fakeTrialRows),
    // A single-rung chain containing just the requested run -- the common
    // case (a plain tuner run, never baseline-advanced). Tests exercising
    // an actual multi-rung chain override this directly.
    getRunChain: (runId: string): Effect<ChainRung[]> =>
      Effect.send([
        {
          run_id: runId,
          label: null,
          status: "completed",
          started_at: "2026-03-01T00:00:00Z",
          ended_at: "2026-03-01T01:00:00Z",
          trial_count: fakeTrialRows.length,
          incumbent: null,
        },
      ]),
    getRunGames: (): Effect<GameTraceSummary[]> => Effect.send(fakeGameTraces),
    getRunGameMoves: (): Effect<GameMove[]> => Effect.send(fakeGameMoves),
    deleteRun: (): Effect<void> => Effect.send(undefined),
    downloadFile: (): Effect<void> => Effect.send(undefined),
  };
  return { ...base, ...overrides };
}
