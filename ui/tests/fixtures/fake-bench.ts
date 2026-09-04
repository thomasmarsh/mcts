// tests/fixtures/fake-bench.ts — Mock bench API responses for component tests.
// Mirrors tests/fixtures/fake-game.tsx's pattern: lightweight, no real
// server or browser dependencies.

import { Effect } from "@mcts/core";
import type { BenchEnv } from "@mcts/bench";
import type {
  RunDetail,
  RunLogResponse,
  RunSummary,
  LaunchResponse,
  StopResponse,
  TunableGame,
  TrialRow,
  GameTraceSummary,
  GameMove,
} from "@mcts/bench";

// This run predates tuner-only launches (the web UI could still launch
// round_robin runs when it was recorded) -- kept as "round_robin" so the
// run-detail/spectator paths are exercised against a historical run kind
// that may still exist on disk, not just the tuner kind the launch form
// produces today.
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

export const fakePhysicalTunerRun: RunDetail = {
  ...fakeRunDetail,
  run_id: FAKE_tuner_RUN_ID,
  kind: "tuner",
  game: "traffic-lights",
  config: {
    overrides: ["optimizer.n_trials=50", 'target.baselines=["bandit"]'],
    baseline_settings: { bandit: { algorithm: "bandit", q_init: "Infinity" } },
  },
  trial_count: 3,
  incumbent: { config: { select: "rave", c: 0.7 }, cost: 0.2 },
};

// Mirrors `mcts_tune::strategy_tuner_info`'s real shape: `algorithm` is the
// always-active root categorical and `select` a policy-axis categorical,
// with other parameters gated via two levels of `TunerCondition`s
// (select -> schedule -> rave, select -> rave_ucb -> c), not a single fixed
// variant's flat schema. Trimmed to a handful of variants/params rather than
// the full catalog -- enough to exercise multi-level conditions and a
// non-RAVE best trial (see fakeTrialRows below).
export const fakeTunableGames: TunableGame[] = [
  {
    game: "traffic-lights",
    tuner: {
      id: "strategy",
      baselines: ["strong"],
      eval_rounds: 20,
      parameters: [
        {
          name: "algorithm",
          type: "categorical",
          choices: ["mcts", "random", "bandit", "negamax"],
          default: "mcts",
        },
        {
          name: "select",
          type: "categorical",
          choices: ["ucb1", "ucb1_tuned", "rave"],
          default: "rave",
        },
        {
          name: "final_action",
          type: "categorical",
          choices: ["max_avg", "secure_child", "robust_child"],
          default: "robust_child",
        },
        { name: "epsilon", type: "float", bounds: [0, 1], default: 0.1 },
        { name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 },
        { name: "rave", type: "int", bounds: [0, 2000], default: 700 },
        {
          name: "schedule",
          type: "categorical",
          choices: ["hand_selected", "min_mse", "threshold"],
          default: "threshold",
        },
        {
          name: "rave_ucb",
          type: "categorical",
          choices: ["none", "ucb1", "tuned"],
          default: "tuned",
        },
      ],
      conditions: [
        { if: { algorithm: "mcts" }, then: ["select"] },
        { if: { select: ["ucb1", "ucb1_tuned", "rave"] }, then: ["final_action"] },
        { if: { select: ["ucb1_tuned", "rave"] }, then: ["epsilon"] },
        { if: { select: "rave" }, then: ["schedule", "rave_ucb"] },
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
        { name: "algorithm", type: "categorical", choices: ["mcts"], default: "mcts" },
        { name: "select", type: "categorical", choices: ["ucb1", "rave"], default: "rave" },
      ],
      conditions: [],
      game_config: { size: { w: 5, h: 5 } },
    },
  },
];

// Best trial (#2, cost 0.3) is deliberately `select: "ucb1_tuned"`, unlike
// the run's `bandit` baseline -- proves the run-detail baseline comparison
// and the trial table work across different configs.
export const fakeTrialRows: TrialRow[] = [
  {
    trial_id: 1,
    ts: "2026-03-01T00:00:01Z",
    config: {
      select: "rave",
      final_action: "robust_child",
      epsilon: 0.1,
      schedule: "threshold",
      rave: 700,
      rave_ucb: "tuned",
      c: 1.4,
    },
    seed: 0,
    cost: 0.55,
    extra: null,
  },
  {
    trial_id: 2,
    ts: "2026-03-01T00:01:00Z",
    config: { select: "ucb1_tuned", final_action: "max_avg", epsilon: 0.2 },
    seed: 0,
    cost: 0.3,
    extra: null,
  },
  {
    trial_id: 3,
    ts: "2026-03-01T00:02:00Z",
    config: { select: "ucb1", final_action: "robust_child" },
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
  {
    ply: 0,
    ts: "2026-01-01T00:00:00Z",
    state: { player: "Black", board: [] },
    mv: null,
    player: null,
  },
  {
    ply: 1,
    ts: "2026-01-01T00:00:01Z",
    state: { player: "White", board: [] },
    mv: ["Black", 0],
    player: "Black",
  },
  {
    ply: 2,
    ts: "2026-01-01T00:00:02Z",
    state: { player: "Black", board: [] },
    mv: ["White", 1],
    player: "White",
  },
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
    config: { select: "ucb1_tuned", final_action: "max_avg", epsilon: 0.2 },
    seed: 1,
    cost: 0.25,
    extra: null,
  },
  {
    trial_id: 5,
    ts: "2026-03-01T00:04:00Z",
    config: { select: "ucb1_tuned", final_action: "max_avg", epsilon: 0.2 },
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
    config: {
      select: "rave",
      final_action: "robust_child",
      epsilon: 0.1,
      schedule: "threshold",
      rave: 700,
      rave_ucb: "tuned",
      c: 1.4,
    },
    seed: 0,
    cost: 0.1,
    extra: { instance: "strong" },
  },
  {
    trial_id: 2,
    ts: "2026-03-01T00:01:00Z",
    config: {
      select: "rave",
      final_action: "robust_child",
      epsilon: 0.1,
      schedule: "threshold",
      rave: 700,
      rave_ucb: "tuned",
      c: 1.4,
    },
    seed: 0,
    cost: 0.6,
    extra: { instance: "master" },
  },
];

export function createMockBenchEnv(overrides?: Partial<BenchEnv>): BenchEnv {
  const base: BenchEnv = {
    listRuns: () => Effect.send(fakeRunSummaries),
    getRun: (runId: string) =>
      Effect.send(
        runId === FAKE_RUN_ID
          ? fakeRunDetail
          : runId === FAKE_tuner_RUN_ID
            ? fakePhysicalTunerRun
            : fakeRunningDetail,
      ),
    getRunLog: (_runId: string, _since: number): Effect<RunLogResponse> =>
      Effect.send({ lines: ['{"type":"match_result","seq":1}'], next_offset: 42 }),
    launchRun: (_kind: string, _game: string, _config?: unknown): Effect<LaunchResponse> =>
      Effect.send({ run_id: "new-run-123", pid: 99999, log_path: "/tmp/new/log.jsonl" }),
    stopRun: (_runId: string): Effect<StopResponse> =>
      Effect.send({ run_id: "stopped-run", message: "stopped" }),
    getTunableGames: () => Effect.send(fakeTunableGames),
    getRunTrials: (_runId: string, _limit?: number): Effect<TrialRow[]> =>
      Effect.send(fakeTrialRows),
    getRunGames: (): Effect<GameTraceSummary[]> => Effect.send(fakeGameTraces),
    getRunGameMoves: (): Effect<GameMove[]> => Effect.send(fakeGameMoves),
    deleteRun: (): Effect<void> => Effect.send(undefined),
  };
  return { ...base, ...overrides };
}
