// tests/fixtures/fake-bench.ts — Mock bench API responses for component tests.
// Mirrors tests/fixtures/fake-game.tsx's pattern: lightweight, no real
// server or browser dependencies.

import { Effect } from "@mcts/core";
import type { BenchEnv } from "@mcts/bench";
import type {
  BenchKindInfo,
  RunDetail,
  RunLogResponse,
  RunSummary,
  LaunchResponse,
  StopResponse,
  LeaderboardEntry,
  Smac3GameInfo,
  TrialRow,
} from "@mcts/bench";

export const FAKE_RUN_ID = "rr-druid-20260101T000000-abc1234";

export const fakeRunSummaries: RunSummary[] = [
  {
    run_id: FAKE_RUN_ID,
    kind: "round_robin",
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
};

export const fakeRunningDetail: RunDetail = {
  ...fakeRunDetail,
  run_id: "rr-druid-20260201T000000-def5678",
  status: "running",
  ended_at: null,
  match_count: 5,
  trial_count: 2,
};

export const FAKE_SMAC3_RUN_ID = "smac3-traffic-lights-20260301T000000-abc1234";

export const fakeSmac3RunDetail: RunDetail = {
  ...fakeRunDetail,
  run_id: FAKE_SMAC3_RUN_ID,
  kind: "smac3",
  game: "traffic-lights",
  config: { overrides: ["optimizer.n_trials=50"] },
  trial_count: 3,
};

// Mirrors `mcts_tune::strategy_tuner_info`'s real shape: `family` is a
// top-level categorical gating other parameters via two levels of
// `TunerCondition`s (family -> schedule -> rave, family -> rave_ucb -> c),
// not a single fixed family's flat schema. Trimmed to a handful of
// families/params rather than the full ~14-family catalog -- enough to
// exercise multi-level conditions and a non-RAVE best trial (see
// fakeTrialRows below).
export const fakeSmac3Kinds: Smac3GameInfo[] = [
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
    },
  },
];

// Best trial (#2, cost 0.3) is deliberately `family: "ucb1_tuned"`, not the
// search space's default `family: "rave"` -- proves the run-detail
// best-vs-default diff table (and the trial table's Family column) work
// across two different families' configs, not just two RAVE configs.
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

// Same `config` evaluated against two different baseline instances (SMAC3's
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
    kind: "smac3",
    label: "SMAC3 Tuning",
    description: "Runs a SMAC3 hyperparameter-optimization sweep.",
    // Mirrors the real server: smac3's per-game info comes from
    // GET /api/bench/smac3/kinds (fakeSmac3Kinds), not this list.
    games: [],
  },
];

export function createMockBenchEnv(overrides?: Partial<BenchEnv>): BenchEnv {
  const base: BenchEnv = {
    listRuns: () => Effect.send(fakeRunSummaries),
    getRun: (runId: string) =>
      Effect.send(
        runId === FAKE_RUN_ID ? fakeRunDetail : runId === FAKE_SMAC3_RUN_ID ? fakeSmac3RunDetail : fakeRunningDetail,
      ),
    getRunLog: (_runId: string, _since: number): Effect<RunLogResponse> =>
      Effect.send({ lines: ['{"type":"match_result","seq":1}'], next_offset: 42 }),
    getLeaderboard: (): Effect<LeaderboardEntry[]> => Effect.send([]),
    launchRun: (_kind: string, _game: string, _config?: unknown): Effect<LaunchResponse> =>
      Effect.send({ run_id: "new-run-123", pid: 99999, log_path: "/tmp/new/log.jsonl" }),
    stopRun: (_runId: string): Effect<StopResponse> =>
      Effect.send({ run_id: "stopped-run", message: "stopped" }),
    getBenchKinds: () => Effect.send(fakeKinds),
    getSmac3Kinds: () => Effect.send(fakeSmac3Kinds),
    getRunTrials: (_runId: string, _limit?: number): Effect<TrialRow[]> => Effect.send(fakeTrialRows),
  };
  return { ...base, ...overrides };
}