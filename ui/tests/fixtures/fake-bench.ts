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

export const fakeSmac3Kinds: Smac3GameInfo[] = [
  {
    game: "traffic-lights",
    tuner: {
      id: "rave",
      baseline: "strong",
      eval_rounds: 20,
      parameters: [
        { name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 },
        { name: "epsilon", type: "float", bounds: [0, 1], default: 0.1 },
        { name: "rave", type: "int", bounds: [0, 2000], default: 700 },
        { name: "schedule", type: "categorical", choices: ["hand_selected", "min_mse", "threshold"], default: "threshold" },
        { name: "final_action", type: "constant", value: "robust_child" },
      ],
      conditions: [
        { if: { schedule: "threshold" }, then: ["rave"] },
        { if: { rave_ucb: ["ucb1", "tuned"] }, then: ["c"] },
      ],
    },
  },
];

export const fakeTrialRows: TrialRow[] = [
  { trial_id: 1, ts: "2026-03-01T00:00:01Z", config: { c: 1.4, epsilon: 0.1, rave: 700, schedule: "threshold" }, seed: 0, cost: 0.55, extra: null },
  {
    trial_id: 2,
    ts: "2026-03-01T00:01:00Z",
    config: { c: 0.9, epsilon: 0.2, rave: 500, schedule: "threshold" },
    seed: 0,
    cost: 0.3,
    extra: null,
  },
  { trial_id: 3, ts: "2026-03-01T00:02:00Z", config: { c: 2.1, epsilon: 0.05, rave: 900, schedule: "threshold" }, seed: 0, cost: 0.4, extra: null },
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