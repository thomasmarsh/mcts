// state.ts — Bench feature state: one flat tree of slices, mirroring
// @mcts/game's state.ts convention. The bench UI is independent of the
// game store — it gets its own `createStore(benchReducer, benchEnv)` — so
// nothing here references game types.

import {
  initialJobPollState,
  type JobPollState,
} from "@mcts/core";
import { initialTuningNavigationState, type TuningNavigationState } from "./tuning-navigation.js";
import type {
  BenchKindInfo,
  CommitTrendData,
  LeaderboardEntry,
  LeaderboardFilters,
  LaunchResponse,
  RunDetail,
  RunFilters,
  RunSummary,
  TunerGameInfo,
  TrialRow,
  Project,
  Experiment,
  ExperimentCell,
  ExperimentSpecV1,
  GameTraceSummary,
} from "./types.js";

/** Live tail of one open run's `log.jsonl`, fed by the reducer's
 * self-scheduling poll loop (see reducer.ts). */
export interface LogTailState {
  /** Raw JSONL lines, in file order, oldest first. */
  lines: string[];
  /** Byte-offset cursor into the run's log file — passed as `since` on the
   * next tick, straight from the server's `next_offset`. */
  offset: number;
  /** False once the run went terminal (log complete) or the tail gave up
   * after too many consecutive failures — no further ticks are scheduled. */
  active: boolean;
  /** Last tick failure's message; cleared by the next successful tick. */
  error: string | null;
  /** Consecutive ticks that returned no new lines — drives the backoff
   * (`tailDelayMs`). Reset to 0 whenever lines arrive. */
  idleAttempts: number;
  /** Consecutive failed ticks — the tail gives up at TAIL_MAX_FAILURES. */
  failures: number;
}

/** The run currently open in the detail/log panel. Only one run is open at
 * a time; opening another replaces this wholesale. */
export interface OpenRunState {
  runId: string;
  /** Null until the first tick resolves — the detail row rides along on
   * every tail tick (see reducer.ts), so there's no separate detail fetch
   * to wait on, and the status/match counts stay live for free. */
  detail: RunDetail | null;
  tail: LogTailState;
  /** Recorded physical-run trials, kept with the run diagnostics. */
  trials: TrialRow[];
  cells: ExperimentCell[];
  games: GameTraceSummary[];
}

/** Win-rate-over-commits trend data: one leaderboard snapshot per git SHA. */
export interface CommitTrendsState {
  data: CommitTrendData;
  /** Sorted SHAs, newest first. */
  shas: string[];
  status: "idle" | "loading" | "done" | "error";
  error: string | null;
}

export interface BenchState {
  activeTab: "projects" | "runs" | "leaderboard";
  projects: JobPollState<Project[]>;
  selectedProjectId: string | null;
  selectedExperimentId: string | null;
  selectedCellId: string | null;
  selectedProject: Project | null;
  selectedExperiment: Experiment | null;
  experiments: JobPollState<Experiment[]>;
  cells: JobPollState<ExperimentCell[]>;
  projectDraft: { name: string; description: string };
  experimentDraft: { name: string; description: string; spec: ExperimentSpecV1 } | null;
  /** The last server-confirmed experiment value. A draft is launchable only
   * when it is equal to this snapshot, so editing after a save cannot launch
   * an older or merely selected definition by accident. */
  experimentSavedDraft: { name: string; description: string; spec: ExperimentSpecV1 } | null;
  experimentSaveStatus: "idle" | "saving";
  experimentLaunchStatus: "idle" | "launching";
  experimentFieldErrors: Record<string, string>;
  projectError: string | null;
  experimentError: string | null;
  experimentRunError: string | null;
  runs: JobPollState<RunSummary[]>;
  runFilters: RunFilters;
  openRun: OpenRunState | null;
  /** Bumped by every `openRun` dispatch and stamped onto the tail actions
   * that open spawns. A tick/tailed arriving after a close or after a
   * different run was opened carries a stale generation and is dropped, so
   * an in-flight poll from a previous view can never append lines to the
   * newly opened run. */
  openGeneration: number;
  leaderboard: JobPollState<LeaderboardEntry[]>;
  leaderboardFilters: LeaderboardFilters;
  commitTrends: CommitTrendsState;
  launch: JobPollState<LaunchResponse>;
  /** Last failed stop attempt's message; cleared by the next `stopRun`. */
  stopError: string | null;
  /** Last failed run deletion's message; cleared by the next delete. */
  deleteError: string | null;
  /** True when the launch form should be shown in the main pane instead of the run detail panel. */
  showLaunchForm: boolean;
  /** Available run kinds loaded on mount — populates the launch form. */
  kinds: JobPollState<BenchKindInfo[]>;
  /** Per-game tuner metadata for every tuner-tunable game, loaded on mount
   * — populates the tuner launch fields' game picker and the run-detail
   * baseline parameter comparison. */
  tunerKinds: JobPollState<TunerGameInfo[]>;
  /** Logical tuning-session data and user-owned hierarchy navigation. */
  tuningNavigation: TuningNavigationState;
  experimentExportStatus: "idle" | "pending";
  experimentExportError: string | null;
}

export function initialBenchState(): BenchState {
  return {
    activeTab: "projects",
    projects: initialJobPollState<Project[]>(),
    selectedProjectId: null,
    selectedExperimentId: null,
    selectedCellId: null,
    selectedProject: null,
    selectedExperiment: null,
    experiments: initialJobPollState<Experiment[]>(),
    cells: initialJobPollState<ExperimentCell[]>(),
    projectDraft: { name: "", description: "" },
    experimentDraft: null,
    experimentSavedDraft: null,
    experimentSaveStatus: "idle",
    experimentLaunchStatus: "idle",
    experimentFieldErrors: {},
    projectError: null,
    experimentError: null,
    experimentRunError: null,
    runs: initialJobPollState<RunSummary[]>(),
    runFilters: { status: null, game: null },
    openRun: null,
    openGeneration: 0,
    leaderboard: initialJobPollState<LeaderboardEntry[]>(),
    leaderboardFilters: { game: null, gitSha: null, since: null },
    commitTrends: { data: {}, shas: [], status: "idle", error: null },
    launch: initialJobPollState<LaunchResponse>(),
    stopError: null,
    deleteError: null,
    showLaunchForm: false,
    kinds: initialJobPollState<BenchKindInfo[]>(),
    tunerKinds: initialJobPollState<TunerGameInfo[]>(),
    tuningNavigation: initialTuningNavigationState(),
    experimentExportStatus: "idle",
    experimentExportError: null,
  };
}
