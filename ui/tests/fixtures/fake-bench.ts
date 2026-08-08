// tests/fixtures/fake-bench.ts — Mock bench API responses for component tests.
// Mirrors tests/fixtures/fake-game.tsx's pattern: lightweight, no real
// server or browser dependencies.

import { Effect } from "@mcts/core";
import type { BenchEnv } from "@mcts/bench";
import type { BenchKindInfo, RunDetail, RunLogResponse, RunSummary, LaunchResponse, StopResponse, LeaderboardEntry } from "@mcts/bench";

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
];

export function createMockBenchEnv(overrides?: Partial<BenchEnv>): BenchEnv {
  const base: BenchEnv = {
    listRuns: () => Effect.send(fakeRunSummaries),
    getRun: (runId: string) => Effect.send(runId === FAKE_RUN_ID ? fakeRunDetail : fakeRunningDetail),
    getRunLog: (_runId: string, _since: number): Effect<RunLogResponse> =>
      Effect.send({ lines: ['{"type":"match_result","seq":1}'], next_offset: 42 }),
    getLeaderboard: (): Effect<LeaderboardEntry[]> => Effect.send([]),
    launchRun: (_kind: string, _game: string, _config?: unknown): Effect<LaunchResponse> =>
      Effect.send({ run_id: "new-run-123", pid: 99999, log_path: "/tmp/new/log.jsonl" }),
    stopRun: (_runId: string): Effect<StopResponse> =>
      Effect.send({ run_id: "stopped-run", message: "stopped" }),
    getBenchKinds: () => Effect.send(fakeKinds),
  };
  return { ...base, ...overrides };
}