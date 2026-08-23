// reducer.ts — Bench reducer: run list (with filters), one open run's
// detail + live log tail, leaderboard (with filters), launch/stop.
//
// The one-shot fetches (runs list, leaderboard, launch) go through
// @mcts/core's jobPollReduce with a `submitJob` that resolves directly to
// `{status: "done"}` — the same "blocking request dressed as a job" wiring
// @mcts/game uses for aiMove/analyze, so pending/done/error transitions
// stay uniform across the app.
//
// The log tail is the one genuinely long-lived piece: a self-scheduling
// poll loop built from `Effect.delay`, the same backoff shape as
// core/job-poll.ts. Each tick fetches the run's new log lines *and* its
// detail row together, so the detail panel's status and match/trial counts
// stay live without a second poller. The loop stops when the detail row
// reports a terminal status (a finished run's log file is complete — see
// `isTerminalStatus` in types.ts) or after TAIL_MAX_FAILURES consecutive
// failures, and every action the loop dispatches carries the
// `openGeneration` it was scheduled under so a close/reopen invalidates
// whatever is still in flight.

import {
  Effect,
  jobPollReduce,
  type JobPollAction,
  type JobPollEnv,
  type JobSubmitResult,
} from "@mcts/core";
import type { BenchState, ChainedTrial } from "./state.js";
import { tuningNavigationReducer, type TuningNavigationAction } from "./tuning-navigation.js";
import {
  isTerminalStatus,
  type BenchKindInfo,
  type ChainRung,
  type CommitTrendData,
  type LaunchResponse,
  type LeaderboardEntry,
  type LeaderboardFilters,
  type RunDetail,
  type RunFilters,
  type RunLogResponse,
  type RunSummary,
  type TunerGameInfo,
  type StopResponse,
  type TrialRow,
  type GameTraceSummary,
  type GameMove,
  type Project,
  type Experiment,
  type ExperimentCell,
  type ExperimentSpecV1,
  type TuningSessionDetail,
  type TuningSessionsResponse,
} from "./types.js";
import { expandExperimentSpec } from "./experiment-grid.js";
import { serializeExperimentRunCsv, serializeExperimentRunJson, sanitizeExportRunId } from "./experiment-export.js";

/** Every network operation the bench reducer may perform, lifted to
 * `Effect` — hard rule (enforced by ui/eslint.config.js's fetch ban): no
 * reducer or component calls `fetch`/`BenchApiClient` directly, only
 * `env.xxx()`. */
export interface BenchEnv {
  listProjects(): Effect<Project[]>;
  createProject(name: string, description: string): Effect<Project>;
  getProject(projectId: string): Effect<Project>;
  updateProject(projectId: string, body: { name?: string; description?: string; archived?: boolean }): Effect<Project>;
  listExperiments(projectId: string): Effect<Experiment[]>;
  createExperiment(projectId: string, body: { name: string; description: string; spec: ExperimentSpecV1 }): Effect<Experiment>;
  getExperiment(experimentId: string): Effect<Experiment>;
  updateExperiment(experimentId: string, body: { name: string; description: string; spec: ExperimentSpecV1 }): Effect<Experiment>;
  launchExperiment(experimentId: string): Effect<LaunchResponse>;
  getRunCells(runId: string): Effect<ExperimentCell[]>;
  listRuns(filters: RunFilters): Effect<RunSummary[]>;
  getRun(runId: string): Effect<RunDetail>;
  getRunLog(runId: string, since: number): Effect<RunLogResponse>;
  /** Fetch the full raw content of the run's stdout.log file (stderr
   * output redirected by the launcher). */
  getRunStdout(runId: string): Effect<string>;
  getLeaderboard(filters: LeaderboardFilters): Effect<LeaderboardEntry[]>;
  /** Fetch one leaderboard snapshot per distinct git SHA that has runs for
   * the given game, building a map from SHA -> entries. */
  fetchCommitTrends(game: string | null): Effect<CommitTrendData>;
  launchRun(kind: string, game: string, config?: unknown): Effect<LaunchResponse>;
  stopRun(runId: string): Effect<StopResponse>;
  /** Relaunch a finished/stopped tuner run with a bigger trial budget,
   * seeded from its saved state. */
  resumeRun(runId: string, nTrials: number, nWorkers?: number): Effect<LaunchResponse>;
  /** Promote this run's current incumbent to a new baseline instance and
   * relaunch as the next rung in its ladder chain. Stops the run first if
   * it's still running. */
  advanceBaseline(runId: string, nTrials?: number, nWorkers?: number): Effect<LaunchResponse>;
  getBenchKinds(): Effect<BenchKindInfo[]>;
  /** Per-game tuner metadata for every tuner-tunable game. */
  getTunerKinds(): Effect<TunerGameInfo[]>;
  listTuningSessions(): Effect<TuningSessionsResponse>;
  getTuningSession(sessionId: string): Effect<TuningSessionDetail>;
  /** Trial rows for one run, oldest first. */
  getRunTrials(runId: string, limit: number): Effect<TrialRow[]>;
  /** Every rung of the ladder chain `runId` belongs to, oldest first. */
  getRunChain(runId: string): Effect<ChainRung[]>;
  getRunGames(runId: string, limit?: number, cellId?: string | null): Effect<GameTraceSummary[]>;
  getRunGameMoves(runId: string, gameSeq: number): Effect<GameMove[]>;
  deleteRun(runId: string): Effect<void>;
  downloadFile(filename: string, mimeType: string, contents: string): Effect<void>;
}

export const TAIL_BACKOFF_START_MS = 1000;
export const TAIL_BACKOFF_MAX_MS = 10_000;
export const TAIL_MAX_FAILURES = 5;

/** Delay before the next tick after `idleAttempts` consecutive empty (or
 * failed) polls — doubles per idle attempt up to the max, so a run that
 * just produced output is polled again quickly while a quiet run costs
 * almost nothing. */
export function tailDelayMs(idleAttempts: number): number {
  return Math.min(TAIL_BACKOFF_START_MS * 2 ** idleAttempts, TAIL_BACKOFF_MAX_MS);
}

export type RunsAction =
  | { tag: "request" }
  | { tag: "job"; action: JobPollAction<RunSummary[]> };

export type LeaderboardAction =
  | { tag: "request" }
  | { tag: "job"; action: JobPollAction<LeaderboardEntry[]> };

export type LaunchAction =
  | { tag: "request"; kind: string; game: string; config?: unknown }
  | { tag: "job"; action: JobPollAction<LaunchResponse> };

export type BenchAction =
  | { tag: "runs"; action: RunsAction }
  | { tag: "tuningNavigation"; action: TuningNavigationAction }
  /** Replace the run-list filters and refetch with them. */
  | { tag: "setRunFilters"; status: string | null; game: string | null; project_id?: string | null; experiment_id?: string | null }
  | { tag: "openRun"; runId: string }
  | { tag: "closeRun" }
  /** Internal, dispatched by the tail loop itself. */
  | { tag: "tailTick"; generation: number }
  | {
      tag: "tailed";
      generation: number;
      lines: string[];
      nextOffset: number;
      detail: RunDetail;
      /** Every tick's trial rows (see `tailTick` below for why this isn't
       * gated on run kind). Empty for every non-`"tuner"` run. */
      trials: TrialRow[];
      /** This run's ladder chain and every rung's trials, concatenated in
       * chain order — see `tailTick`. */
      chain: ChainRung[];
      chainedTrials: ChainedTrial[];
      cells?: ExperimentCell[];
      games?: GameTraceSummary[];
    }
  | { tag: "tailFailed"; generation: number; error: string }
  | { tag: "leaderboard"; action: LeaderboardAction }
  /** Replace the leaderboard filters and refetch with them. */
  | { tag: "setLeaderboardFilters"; game: string | null; gitSha: string | null; since: string | null }
  /** Fetch win-rate data for every commit that has runs. */
  | { tag: "fetchCommitTrends"; game: string | null }
  | { tag: "commitTrendsLoaded"; data: CommitTrendData; shas: string[] }
  | { tag: "commitTrendsFailed"; error: string }
  | { tag: "launch"; action: LaunchAction }
  | { tag: "stopRun"; runId: string }
  | { tag: "stopFinished"; runId: string }
  | { tag: "stopFailed"; runId: string; error: string }
  | { tag: "resumeRun"; runId: string; nTrials: number; nWorkers?: number }
  | { tag: "resumeFinished"; runId: string }
  | { tag: "resumeFailed"; runId: string; error: string }
  | { tag: "advanceBaseline"; runId: string; nTrials?: number; nWorkers?: number }
  | { tag: "advanceBaselineFinished"; runId: string; newRunId: string }
  | { tag: "advanceBaselineFailed"; runId: string; error: string }
  | { tag: "deleteRun"; runId: string }
  | { tag: "deleteFinished"; runId: string }
  | { tag: "deleteFailed"; runId: string; error: string }
  /** Load all available bench kinds/games/strategies for the launch form. */
  | { tag: "kinds"; action: KindsAction }
  /** Load per-game tuner tuner metadata for the launch form + run detail. */
  | { tag: "tunerKinds"; action: TunerKindsAction }
  | { tag: "setTab"; tab: "projects" | "runs" | "leaderboard" }
  | { tag: "setShowLaunchForm"; show: boolean }
  | { tag: "projectsRequest" }
  | { tag: "projectsLoaded"; projects: Project[] }
  | { tag: "projectsFailed"; error: string }
  | { tag: "openProject"; projectId: string }
  | { tag: "experimentsLoaded"; experiments: Experiment[] }
  | { tag: "experimentsFailed"; error: string }
  | { tag: "projectDraft"; name: string; description: string }
  | { tag: "createProject" }
  | { tag: "projectCreated"; project: Project }
  | { tag: "newExperiment" }
  | { tag: "openExperiment"; experimentId: string }
  | { tag: "experimentLoaded"; experiment: Experiment }
  | { tag: "experimentDraft"; draft: { name: string; description: string; spec: ExperimentSpecV1 } }
  | { tag: "experimentGameChanged"; game: string; gameConfig: unknown }
  | { tag: "experimentGameAdded"; game?: string; gameConfig?: unknown }
  | { tag: "experimentGameRemoved"; index: number }
  | { tag: "experimentGameEdited"; index: number; game: string; gameConfig: unknown }
  | { tag: "experimentVariantAdded" }
  | { tag: "experimentVariantRemoved"; index: number }
  | { tag: "experimentVariantEdited"; index: number; field: "id" | "label" | "config"; value: string | Record<string, unknown> }
  | { tag: "experimentBudgetAdded" }
  | { tag: "experimentBudgetRemoved"; index: number }
  | { tag: "experimentBudgetEdited"; index: number; field: "kind" | "value"; value: string | number }
  | { tag: "saveExperiment" }
  | { tag: "experimentSaved"; experiment: Experiment }
  | { tag: "experimentFailed"; error: string }
  | { tag: "launchExperiment" }
  | { tag: "experimentLaunched"; response: LaunchResponse }
  | { tag: "experimentRunFailed"; error: string }
  | { tag: "openCell"; cellId: string }
  | { tag: "cellGamesLoaded"; cellId: string; games: GameTraceSummary[] }
  | { tag: "exportExperimentRun"; format: "json" | "csv" }
  | { tag: "experimentExportFinished" }
  | { tag: "experimentExportFailed"; error: string };

export type KindsAction =
  | { tag: "request" }
  | { tag: "job"; action: JobPollAction<BenchKindInfo[]> };

export type TunerKindsAction =
  | { tag: "request" }
  | { tag: "job"; action: JobPollAction<TunerGameInfo[]> };

/** Runs an `Effect` for its single value, as a `Promise` — lets the tick
 * branch combine `getRunLog` + `getRun` with `Promise.all` while still
 * routing every network call through `env`, never `fetch` directly (the
 * hard rule only forbids the latter). Same helper @mcts/game's reducer
 * uses for its `position` fetch. */
function toPromise<T>(effect: Effect<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    effect.execute((v) => resolve(v)).catch(reject);
  });
}

/** `jobPollReduce` only ever calls `submitJob`/`pollJob` for the `"start"`/
 * `"tick"` tags. Every `submitJob` this reducer builds resolves directly to
 * `{status: "done", ...}`, and `"start"` actions only ever originate from
 * the "request" branches (which build their own real `jobEnv`), so the
 * forwarded-"job" branches below never reach either. This stub satisfies
 * `JobPollEnv`'s shape for those unreachable paths and throws loudly if
 * that assumption is ever wrong. Same pattern as @mcts/game's reducer. */
function unreachableJobEnv<T>(reason: string): JobPollEnv<T> {
  return {
    submitJob: () => {
      throw new Error(reason);
    },
    pollJob: () => {
      throw new Error(reason);
    },
  };
}

/** Kick off a runs-list fetch with the state's current filters. Returns
 * null (no-op) if a fetch is already in flight — jobPollReduce's "start"
 * is idempotent that way. */
function startRunsFetch(draft: BenchState, env: BenchEnv): Effect<BenchAction> | null {
  const filters: RunFilters = { ...draft.runFilters };
  const jobEnv: JobPollEnv<RunSummary[]> = {
    submitJob: () =>
      env.listRuns(filters).map((result): JobSubmitResult<RunSummary[]> => ({ status: "done", result })),
    pollJob: () => {
      throw new Error("unreachable: the runs list resolves synchronously (see submitJob above)");
    },
  };
  const eff = jobPollReduce(draft.runs, { tag: "start" }, jobEnv);
  return eff ? eff.map((a): BenchAction => ({ tag: "runs", action: { tag: "job", action: a } })) : null;
}

/** Keep the logical-session navigator current whenever physical run rows refresh. */
function refreshRunViews(draft: BenchState, env: BenchEnv): Effect<BenchAction> | null {
  const runs = startRunsFetch(draft, env);
  const sessions = tuningNavigationReducer(draft.tuningNavigation, { tag: "listRequest" }, env)
    ?.map((action): BenchAction => ({ tag: "tuningNavigation", action }));
  if (runs && sessions) return Effect.merge(runs, sessions);
  return runs ?? sessions ?? null;
}

function startLeaderboardFetch(draft: BenchState, env: BenchEnv): Effect<BenchAction> | null {
  const filters: LeaderboardFilters = { ...draft.leaderboardFilters };
  const jobEnv: JobPollEnv<LeaderboardEntry[]> = {
    submitJob: () =>
      env.getLeaderboard(filters).map((result): JobSubmitResult<LeaderboardEntry[]> => ({ status: "done", result })),
    pollJob: () => {
      throw new Error("unreachable: the leaderboard resolves synchronously (see submitJob above)");
    },
  };
  const eff = jobPollReduce(draft.leaderboard, { tag: "start" }, jobEnv);
  return eff ? eff.map((a): BenchAction => ({ tag: "leaderboard", action: { tag: "job", action: a } })) : null;
}

function requestProjects(env: BenchEnv): Effect<BenchAction> {
  return env.listProjects().map((projects): BenchAction => ({ tag: "projectsLoaded", projects })).catch((error): BenchAction => ({ tag: "projectsFailed", error: String(error) }));
}

export function emptyExperimentSpec(game = "nim"): ExperimentSpecV1 {
  return {
    version: 1,
    games: [{ game, game_config: null }],
    baseline: { id: "baseline", label: "Baseline", config: {} },
    variants: [{ id: "variant", label: "Variant", config: {} }],
    budgets: [{ kind: "iterations", value: 25 }],
    rounds_per_cell: 1,
    base_seed: 42,
    max_parallel_cells: 1,
  };
}

function validateExperimentSpec(spec: ExperimentSpecV1): string | null {
  const errors: string[] = [];
  if (spec.games.length === 0) errors.push("spec.games: must contain at least one game");
  if (spec.variants.length === 0) errors.push("spec.variants: must contain at least one variant");
  if (spec.budgets.length === 0) errors.push("spec.budgets: must contain at least one budget");
  if (!Number.isFinite(spec.rounds_per_cell) || spec.rounds_per_cell <= 0) errors.push("spec.rounds_per_cell: must be positive");
  if (!Number.isSafeInteger(spec.max_parallel_cells) || spec.max_parallel_cells <= 0) errors.push("spec.max_parallel_cells: must be positive");
  if (!Number.isSafeInteger(spec.base_seed) || spec.base_seed < 0) errors.push("spec.base_seed: must be a non-negative safe integer");
  spec.games.forEach((game, index) => { if (!game.game.trim()) errors.push(`spec.games[${index}].game: must not be empty`); });
  if (!spec.baseline.id.trim()) errors.push("spec.baseline.id: must not be empty");
  if (!spec.baseline.label.trim()) errors.push("spec.baseline.label: must not be empty");
  const variantIds = new Set<string>();
  const variantLabels = new Set<string>();
  spec.variants.forEach((variant, index) => {
    if (!variant.id.trim()) errors.push(`spec.variants[${index}].id: must not be empty`);
    if (!variant.label.trim()) errors.push(`spec.variants[${index}].label: must not be empty`);
    if (variant.id === spec.baseline.id) errors.push(`spec.variants[${index}].id: must differ from baseline.id`);
    if (variant.label === spec.baseline.label) errors.push(`spec.variants[${index}].label: must differ from baseline.label`);
    if (variantIds.has(variant.id)) errors.push(`spec.variants[${index}].id: duplicate variant id`);
    if (variantLabels.has(variant.label)) errors.push(`spec.variants[${index}].label: duplicate variant label`);
    variantIds.add(variant.id); variantLabels.add(variant.label);
  });
  if (!spec.baseline.config || typeof spec.baseline.config !== "object" || Array.isArray(spec.baseline.config)) errors.push("spec.baseline.config: must be a JSON object");
  const budgets = new Set<string>();
  spec.budgets.forEach((budget, index) => { if (!Number.isSafeInteger(budget.value) || budget.value <= 0) errors.push(`spec.budgets[${index}].value: must be a positive safe integer`); const key = `${budget.kind}:${budget.value}`; if (budgets.has(key)) errors.push(`spec.budgets[${index}]: duplicate budget`); budgets.add(key); });
  try { expandExperimentSpec(spec); } catch (error) { errors.push(`spec: ${String(error)}`); }
  return errors.length > 0 ? errors.join("; ") : null;
}

type ExperimentDraft = { name: string; description: string; spec: ExperimentSpecV1 };

function cloneExperimentDraft(value: ExperimentDraft): ExperimentDraft {
  return JSON.parse(JSON.stringify(value)) as ExperimentDraft;
}

function sameExperimentDraft(left: ExperimentDraft | null, right: ExperimentDraft | null): boolean {
  return left !== null && right !== null && JSON.stringify(left) === JSON.stringify(right);
}

function validationErrors(message: string): { fields: Record<string, string>; form: string | null } {
  const fields: Record<string, string> = {};
  const parts = message.split(/;\s*/);
  const pathPattern = /^(spec\.[^:]+|name|description):\s*(.*)$/;
  const formParts: string[] = [];
  for (const part of parts) {
    const match = pathPattern.exec(part);
    if (!match) {
      formParts.push(part);
      continue;
    }
    const path = match[1]!;
    const messageText = match[2]!;
    const knownPath = /^(name|description|spec\.(games|variants|budgets)(\[\d+\])?(\.(game|game_config|id|label|config|kind|value))?|spec\.(baseline\.(id|label|config)|rounds_per_cell|base_seed|max_parallel_cells))$/.test(path);
    if (!knownPath) {
      formParts.push(`${path}: ${messageText}`);
      continue;
    }
    const friendlyPath = path
      .replace("spec.games[0].game", "game")
      .replace("spec.games[0].game_config", "game configuration")
      .replace("spec.baseline.label", "baseline label")
      .replace("spec.baseline.config", "baseline configuration")
      .replace("spec.variants[0].label", "variant label")
      .replace("spec.variants[0].config", "variant configuration")
      .replace("spec.variants", "strategy variants")
      .replace("spec.games", "games")
      .replace("spec.budgets", "budgets")
      .replace("spec.max_parallel_cells", "parallel cells")
      .replace("spec.budgets[0].value", "budget value")
      .replace("spec.rounds_per_cell", "paired rounds")
      .replace("spec.base_seed", "base seed")
      .replace(/^spec\./, "");
    fields[path] = `${friendlyPath}: ${messageText}`;
  }
  return { fields, form: formParts.length > 0 ? formParts.join("; ") : null };
}

export function benchReducer(
  draft: BenchState,
  action: BenchAction,
  env: BenchEnv,
): Effect<BenchAction> | null {
  if (action.tag === "setTab") { draft.activeTab = action.tab; return null; }
  if (action.tag === "tuningNavigation") {
    const effect = tuningNavigationReducer(draft.tuningNavigation, action.action, env);
    return effect?.map((next): BenchAction => ({ tag: "tuningNavigation", action: next })) ?? null;
  }
  if (action.tag === "projectsRequest") {
    draft.projects = { ...draft.projects, status: "pending", result: null, error: null };
    return requestProjects(env);
  }
  if (action.tag === "projectsLoaded") { draft.projects = { ...draft.projects, status: "done", result: action.projects, error: null }; draft.projectError = null; return null; }
  if (action.tag === "projectsFailed") { draft.projects = { ...draft.projects, status: "error", result: null, error: action.error }; draft.projectError = action.error; return null; }
  if (action.tag === "projectDraft") { draft.projectDraft = { name: action.name, description: action.description }; return null; }
  if (action.tag === "createProject") {
    const name = draft.projectDraft.name.trim();
    if (!name) { draft.projectError = "name: must not be empty"; return null; }
    draft.projectError = null;
    return env.createProject(name, draft.projectDraft.description).map((project): BenchAction => ({ tag: "projectCreated", project })).catch((error): BenchAction => ({ tag: "projectsFailed", error: String(error) }));
  }
  if (action.tag === "projectCreated") {
    draft.selectedProjectId = action.project.project_id; draft.selectedProject = action.project; draft.projectDraft = { name: "", description: "" };
    draft.selectedExperimentId = null; draft.selectedExperiment = null; draft.experimentDraft = null; draft.experimentSavedDraft = null;
    draft.experimentSaveStatus = "idle"; draft.experimentLaunchStatus = "idle"; draft.experimentFieldErrors = {};
    return Effect.merge(requestProjects(env), env.listExperiments(action.project.project_id).map((experiments): BenchAction => ({ tag: "experimentsLoaded", experiments })).catch((error): BenchAction => ({ tag: "experimentsFailed", error: String(error) })));
  }
  if (action.tag === "openProject") {
    draft.activeTab = "projects"; draft.selectedProjectId = action.projectId; draft.selectedExperimentId = null; draft.experimentDraft = null;
    draft.runFilters = { status: null, game: null, project_id: action.projectId, experiment_id: null };
    const project = draft.projects.result?.find((value) => value.project_id === action.projectId) ?? null; draft.selectedProject = project;
    const runsEffect = refreshRunViews(draft, env);
    const effects = Effect.merge(env.getProject(action.projectId).map((value): BenchAction => ({ tag: "projectCreated", project: value })).catch((error): BenchAction => ({ tag: "projectsFailed", error: String(error) })), env.listExperiments(action.projectId).map((experiments): BenchAction => ({ tag: "experimentsLoaded", experiments })).catch((error): BenchAction => ({ tag: "experimentsFailed", error: String(error) })));
    return runsEffect ? Effect.merge(effects, runsEffect) : effects;
  }
  if (action.tag === "experimentsLoaded") { draft.experiments = { ...draft.experiments, status: "done", result: action.experiments, error: null }; return null; }
  if (action.tag === "experimentsFailed") { draft.experiments = { ...draft.experiments, status: "error", result: null, error: action.error }; draft.experimentError = action.error; return null; }
  if (action.tag === "newExperiment") {
    const first = draft.tunerKinds.result?.[0];
    const spec = emptyExperimentSpec(first?.game ?? "nim");
    if (first) spec.games[0]!.game_config = first.tuner.game_config;
    draft.experimentDraft = { name: "", description: "", spec }; draft.experimentSavedDraft = null;
    draft.experimentSaveStatus = "idle"; draft.experimentLaunchStatus = "idle";
    draft.experimentFieldErrors = {}; draft.selectedExperimentId = null; draft.experimentError = null; return null;
  }
  if (action.tag === "openExperiment") {
    draft.selectedExperimentId = action.experimentId; draft.selectedExperiment = null; draft.experimentDraft = null;
    draft.experimentSavedDraft = null; draft.experimentSaveStatus = "idle"; draft.experimentLaunchStatus = "idle";
    draft.experimentFieldErrors = {}; draft.experimentError = null;
    draft.runFilters = { status: null, game: null, project_id: null, experiment_id: action.experimentId };
    const experimentEffect = env.getExperiment(action.experimentId).map((experiment): BenchAction => ({ tag: "experimentLoaded", experiment })).catch((error): BenchAction => ({ tag: "experimentFailed", error: String(error) }));
    const runsEffect = refreshRunViews(draft, env);
    return runsEffect ? Effect.merge(experimentEffect, runsEffect) : experimentEffect;
  }
  if (action.tag === "experimentLoaded") {
    draft.selectedExperiment = action.experiment;
    draft.experimentDraft = { name: action.experiment.name, description: action.experiment.description, spec: action.experiment.spec };
    draft.experimentSavedDraft = cloneExperimentDraft(draft.experimentDraft);
    draft.experimentSaveStatus = "idle"; draft.experimentLaunchStatus = "idle";
    draft.experimentFieldErrors = {}; draft.experimentError = null; return null;
  }
  if (action.tag === "experimentDraft") { draft.experimentDraft = action.draft; draft.experimentFieldErrors = {}; draft.experimentError = null; return null; }
  if (action.tag === "experimentGameChanged") {
    if (!draft.experimentDraft) return null;
    const spec = JSON.parse(JSON.stringify(draft.experimentDraft.spec)) as ExperimentSpecV1;
    if (!spec.games[0]) spec.games[0] = { game: action.game, game_config: action.gameConfig };
    else { spec.games[0].game = action.game; spec.games[0].game_config = action.gameConfig; }
    draft.experimentDraft = { ...draft.experimentDraft, spec };
    draft.experimentFieldErrors = {};
    draft.experimentError = null;
    return null;
  }
  if (action.tag === "experimentGameAdded" || action.tag === "experimentGameRemoved" || action.tag === "experimentGameEdited" || action.tag === "experimentVariantAdded" || action.tag === "experimentVariantRemoved" || action.tag === "experimentVariantEdited" || action.tag === "experimentBudgetAdded" || action.tag === "experimentBudgetRemoved" || action.tag === "experimentBudgetEdited") {
    if (!draft.experimentDraft) return null;
    const spec = JSON.parse(JSON.stringify(draft.experimentDraft.spec)) as ExperimentSpecV1;
    if (action.tag === "experimentGameAdded") spec.games.push({ game: action.game ?? spec.games.at(-1)?.game ?? "nim", game_config: action.gameConfig ?? spec.games.at(-1)?.game_config ?? null });
    if (action.tag === "experimentGameRemoved" && spec.games.length > 1) spec.games.splice(action.index, 1);
    if (action.tag === "experimentGameEdited" && spec.games[action.index]) { spec.games[action.index]!.game = action.game; spec.games[action.index]!.game_config = action.gameConfig; }
    if (action.tag === "experimentVariantAdded") { let n = spec.variants.length + 1; while (spec.variants.some((variant) => variant.id === `variant-${n}` || variant.label === `Variant ${n}`)) n += 1; spec.variants.push({ id: `variant-${n}`, label: `Variant ${n}`, config: {} }); }
    if (action.tag === "experimentVariantRemoved" && spec.variants.length > 1) spec.variants.splice(action.index, 1);
    if (action.tag === "experimentVariantEdited" && spec.variants[action.index]) {
      if (action.field === "config") spec.variants[action.index]!.config = action.value as Record<string, unknown>;
      else spec.variants[action.index]![action.field] = action.value as string;
    }
    if (action.tag === "experimentBudgetAdded") spec.budgets.push({ kind: "iterations", value: 25 });
    if (action.tag === "experimentBudgetRemoved" && spec.budgets.length > 1) spec.budgets.splice(action.index, 1);
    if (action.tag === "experimentBudgetEdited" && spec.budgets[action.index]) {
      if (action.field === "kind") spec.budgets[action.index] = { kind: action.value as "iterations" | "time_per_move_ms", value: spec.budgets[action.index]!.value };
      else spec.budgets[action.index]!.value = action.value as number;
    }
    draft.experimentDraft = { ...draft.experimentDraft, spec }; draft.experimentFieldErrors = {}; draft.experimentError = null; return null;
  }
  if (action.tag === "saveExperiment") {
    const draftValue = draft.experimentDraft;
    if (draft.experimentSaveStatus === "saving") return null;
    if (!draftValue || !draftValue.name.trim()) { draft.experimentError = "Enter an experiment name."; draft.experimentFieldErrors = { name: "Experiment name is required." }; return null; }
    const specError = validateExperimentSpec(draftValue.spec);
    if (specError) {
      const parsed = validationErrors(specError);
      draft.experimentError = parsed.form ?? "Review the highlighted experiment settings.";
      draft.experimentFieldErrors = parsed.fields;
      return null;
    }
    const method = draft.selectedExperimentId ? env.updateExperiment(draft.selectedExperimentId, draftValue) : (draft.selectedProjectId ? env.createExperiment(draft.selectedProjectId, draftValue) : null);
    if (!method) { draft.experimentError = "select a project first"; return null; }
    draft.experimentError = null; draft.experimentFieldErrors = {}; draft.experimentSaveStatus = "saving";
    return method.map((experiment): BenchAction => ({ tag: "experimentSaved", experiment })).catch((error): BenchAction => ({ tag: "experimentFailed", error: String(error) }));
  }
  if (action.tag === "experimentSaved") {
    draft.selectedExperiment = action.experiment; draft.selectedExperimentId = action.experiment.experiment_id;
    draft.experimentDraft = { name: action.experiment.name, description: action.experiment.description, spec: action.experiment.spec };
    draft.experimentSavedDraft = cloneExperimentDraft(draft.experimentDraft);
    draft.experimentSaveStatus = "idle"; draft.experimentError = null; draft.experimentFieldErrors = {};
    return draft.selectedProjectId ? env.listExperiments(draft.selectedProjectId).map((experiments): BenchAction => ({ tag: "experimentsLoaded", experiments })) : null;
  }
  if (action.tag === "launchExperiment") {
    if (draft.experimentLaunchStatus === "launching") return null;
    if (!draft.selectedExperimentId || !draft.experimentDraft || !sameExperimentDraft(draft.experimentDraft, draft.experimentSavedDraft)) { draft.experimentRunError = "Save the current experiment before launching."; return null; }
    const specError = validateExperimentSpec(draft.experimentDraft.spec);
    if (specError) { draft.experimentRunError = "Review the experiment settings before launching."; return null; }
    draft.experimentRunError = null; draft.experimentLaunchStatus = "launching";
    return env.launchExperiment(draft.selectedExperimentId).map((response): BenchAction => ({ tag: "experimentLaunched", response })).catch((error): BenchAction => ({ tag: "experimentRunFailed", error: String(error) }));
  }
  if (action.tag === "experimentLaunched") { draft.experimentLaunchStatus = "idle"; draft.activeTab = "runs"; draft.experimentRunError = null; return Effect.send({ tag: "openRun", runId: action.response.run_id }); }
  if (action.tag === "experimentRunFailed") { draft.experimentLaunchStatus = "idle"; draft.experimentRunError = action.error; return null; }
  if (action.tag === "experimentFailed") {
    draft.experimentSaveStatus = "idle";
    const parsed = validationErrors(action.error);
    draft.experimentError = parsed.form ?? "The experiment could not be saved.";
    draft.experimentFieldErrors = parsed.fields;
    return null;
  }
  if (action.tag === "openCell") {
    draft.selectedCellId = action.cellId;
    const runId = draft.openRun?.runId;
    return runId ? env.getRunGames(runId, 5000, action.cellId).map((games): BenchAction => ({ tag: "cellGamesLoaded", cellId: action.cellId, games })) : null;
  }
  if (action.tag === "cellGamesLoaded") {
    if (draft.selectedCellId === action.cellId && draft.openRun) draft.openRun.games = action.games;
    return null;
  }
  if (action.tag === "exportExperimentRun") {
    if (draft.experimentExportStatus === "pending") return null;
    const detail = draft.openRun?.detail;
    const cells = draft.openRun?.cells;
    if (!detail || !detail.experiment_spec || !cells) {
      draft.experimentExportError = "The run snapshot and cells are not loaded yet.";
      return null;
    }
    draft.experimentExportStatus = "pending";
    draft.experimentExportError = null;
    try {
      const isJson = action.format === "json";
      const contents = isJson ? serializeExperimentRunJson(detail, cells) : serializeExperimentRunCsv(detail, cells);
      const filename = `experiment-${sanitizeExportRunId(detail.run_id)}.${action.format}`;
      const mimeType = isJson ? "application/json" : "text/csv;charset=utf-8";
      return env.downloadFile(filename, mimeType, contents)
        .map((): BenchAction => ({ tag: "experimentExportFinished" }))
        .catch((error): BenchAction => ({ tag: "experimentExportFailed", error: String(error) }));
    } catch (error) {
      draft.experimentExportStatus = "idle";
      draft.experimentExportError = String(error);
      return null;
    }
  }
  if (action.tag === "experimentExportFinished") {
    draft.experimentExportStatus = "idle";
    return null;
  }
  if (action.tag === "experimentExportFailed") {
    draft.experimentExportStatus = "idle";
    draft.experimentExportError = action.error;
    return null;
  }
  if (action.tag === "runs") {
    const ra = action.action;
    if (ra.tag === "request") return refreshRunViews(draft, env);
    const eff = jobPollReduce(
      draft.runs,
      ra.action,
      unreachableJobEnv("unreachable: a forwarded runs/job action never re-submits or polls"),
    );
    return eff ? eff.map((a): BenchAction => ({ tag: "runs", action: { tag: "job", action: a } })) : null;
  }

  if (action.tag === "setRunFilters") {
    draft.runFilters = {
      status: action.status,
      game: action.game,
      ...(action.project_id !== undefined ? { project_id: action.project_id } : {}),
      ...(action.experiment_id !== undefined ? { experiment_id: action.experiment_id } : {}),
    };
    return refreshRunViews(draft, env);
  }

  if (action.tag === "openRun") {
    if (draft.tuningNavigation.selection.sessionId) {
      tuningNavigationReducer(draft.tuningNavigation, { tag: "clearSession" }, env);
    }
    draft.showLaunchForm = false;
    draft.openGeneration += 1;
    draft.selectedCellId = null;
    draft.experimentExportStatus = "idle";
    draft.experimentExportError = null;
    draft.openRun = {
      runId: action.runId,
      detail: null,
      tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
      trials: [],
      chain: [],
      chainedTrials: [],
      cells: [],
      games: [],
    };
    // The first tick doubles as the detail fetch — no separate request.
    return Effect.send<BenchAction>({ tag: "tailTick", generation: draft.openGeneration });
  }

  if (action.tag === "closeRun") {
    draft.openRun = null;
    // In-flight ticks/taileds from the closed run are dropped by their
    // generation guard when they land — nothing to cancel here.
    return null;
  }

  if (action.tag === "tailTick") {
    const open = draft.openRun;
    if (!open || draft.openGeneration !== action.generation || !open.tail.active) return null;
    const { runId } = open;
    const since = open.tail.offset;
    const { generation } = action;
    // Trials have no incremental cursor (unlike the log), so this refetches
    // the full list every tick -- fine at tuner's trial-count scale. Fetched
    // unconditionally rather than gated on `detail.kind === "tuner"":
    // opening an *already-completed* run goes terminal on this very first
    // tick (before any prior tick could have told us the kind), which would
    // otherwise mean its trials are never fetched at all. The cost for
    // every other run kind is one query returning an empty row set.
    return Effect.fromPromise(async () => {
      const [log, detail, trials, chain, cells, games] = await Promise.all([
        toPromise(env.getRunLog(runId, since)),
        toPromise(env.getRun(runId)),
        toPromise(env.getRunTrials(runId, 5000)),
        toPromise(env.getRunChain(runId)),
        toPromise(env.getRunCells(runId)),
        toPromise(env.getRunGames(runId, 5000, draft.selectedCellId)),
      ]);
      // Refetched per rung every tick, same "just refetch the whole thing"
      // tradeoff `trials` above already makes rather than an incremental
      // cursor -- fine at tuner's trial-count *and* chain-length scale.
      // This duplicates fetching `runId`'s own trials a second time when
      // it's already in the chain (rather than reusing `trials` above) --
      // deliberately, to keep this uniform across every rung instead of
      // special-casing the currently-open one.
      const rungTrials = await Promise.all(
        chain.map((rung) => toPromise(env.getRunTrials(rung.run_id, 5000))),
      );
      const chainedTrials: ChainedTrial[] = rungTrials.flatMap((list, rungIndex) =>
        list.map((trial) => ({ rungIndex, trial })),
      );
      return { log, detail, trials, chain, chainedTrials, cells, games };
    })
      .map((r): BenchAction => ({
        tag: "tailed",
        generation,
        lines: r.log.lines,
        nextOffset: r.log.next_offset,
        detail: r.detail,
        trials: r.trials,
        chain: r.chain,
        chainedTrials: r.chainedTrials,
        ...(r.cells.length > 0 ? { cells: r.cells } : {}),
        ...(r.games.length > 0 ? { games: r.games } : {}),
      }))
      .catch((e): BenchAction => ({ tag: "tailFailed", generation, error: String(e) }));
  }

  if (action.tag === "tailed") {
    const open = draft.openRun;
    if (!open || draft.openGeneration !== action.generation) return null; // stale poll from a closed/replaced run
    open.tail.lines.push(...action.lines);
    open.tail.offset = action.nextOffset;
    open.tail.error = null;
    open.tail.failures = 0;
    open.detail = action.detail;
    open.trials = action.trials;
    open.chain = action.chain;
    open.chainedTrials = action.chainedTrials;
    open.cells = action.cells ?? open.cells;
    open.games = action.games ?? open.games;
    if (isTerminalStatus(action.detail.status)) {
      // The run's log file is complete once the process is done — one last
      // append (this tick's lines) and the loop stops. The runs list just
      // changed too (this run's status/counts), so refresh it in the same
      // reduction rather than waiting for the next manual poll.
      open.tail.active = false;
      open.tail.idleAttempts = 0;
      return refreshRunViews(draft, env);
    }
    open.tail.idleAttempts = action.lines.length > 0 ? 0 : open.tail.idleAttempts + 1;
    return Effect.delay(tailDelayMs(open.tail.idleAttempts), {
      tag: "tailTick",
      generation: action.generation,
    });
  }

  if (action.tag === "tailFailed") {
    const open = draft.openRun;
    if (!open || draft.openGeneration !== action.generation) return null;
    open.tail.error = action.error;
    open.tail.failures += 1;
    if (open.tail.failures >= TAIL_MAX_FAILURES) {
      open.tail.active = false;
      return null;
    }
    // Back off like an idle poll; transient failures (server restarting
    // mid-run, say) shouldn't kill the tail.
    open.tail.idleAttempts += 1;
    return Effect.delay(tailDelayMs(open.tail.idleAttempts), {
      tag: "tailTick",
      generation: action.generation,
    });
  }

  if (action.tag === "leaderboard") {
    const la = action.action;
    if (la.tag === "request") return startLeaderboardFetch(draft, env);
    const eff = jobPollReduce(
      draft.leaderboard,
      la.action,
      unreachableJobEnv("unreachable: a forwarded leaderboard/job action never re-submits or polls"),
    );
    return eff ? eff.map((a): BenchAction => ({ tag: "leaderboard", action: { tag: "job", action: a } })) : null;
  }

  if (action.tag === "setLeaderboardFilters") {
    draft.leaderboardFilters = { game: action.game, gitSha: action.gitSha, since: action.since };
    return startLeaderboardFetch(draft, env);
  }

  if (action.tag === "fetchCommitTrends") {
    draft.commitTrends = { data: {}, shas: [], status: "loading", error: null };
    return env
      .fetchCommitTrends(action.game)
      .map((data): BenchAction => ({
        tag: "commitTrendsLoaded",
        data,
        shas: Object.keys(data).sort().reverse(),
      }))
      .catch((e): BenchAction => ({
        tag: "commitTrendsFailed",
        error: String(e),
      }));
  }

  if (action.tag === "commitTrendsLoaded") {
    draft.commitTrends = { data: action.data, shas: action.shas, status: "done", error: null };
    return null;
  }

  if (action.tag === "commitTrendsFailed") {
    draft.commitTrends = { data: {}, shas: [], status: "error", error: action.error };
    return null;
  }

  if (action.tag === "launch") {
    const la = action.action;
    if (la.tag === "request") {
      const { kind, game, config } = la;
      const jobEnv: JobPollEnv<LaunchResponse> = {
        submitJob: () =>
          env.launchRun(kind, game, config).map((result): JobSubmitResult<LaunchResponse> => ({ status: "done", result })),
        pollJob: () => {
          throw new Error("unreachable: launch resolves synchronously (see submitJob above)");
        },
      };
      const eff = jobPollReduce(draft.launch, { tag: "start" }, jobEnv);
      return eff ? eff.map((a): BenchAction => ({ tag: "launch", action: { tag: "job", action: a } })) : null;
    }
    const eff = jobPollReduce(
      draft.launch,
      la.action,
      unreachableJobEnv("unreachable: a forwarded launch/job action never re-submits or polls"),
    );
    const launchEff = eff ? eff.map((a): BenchAction => ({ tag: "launch", action: { tag: "job", action: a } })) : null;
    // A completed launch means the runs table just gained a row — refresh
    // the list so the new run shows up without a manual reload.
    const refreshEff = draft.launch.status === "done" ? refreshRunViews(draft, env) : null;
    if (launchEff && refreshEff) return Effect.merge(launchEff, refreshEff);
    return launchEff ?? refreshEff;
  }

  if (action.tag === "stopRun") {
    draft.stopError = null;
    const { runId } = action;
    return env
      .stopRun(runId)
      .map((): BenchAction => ({ tag: "stopFinished", runId }))
      .catch((e): BenchAction => ({ tag: "stopFailed", runId, error: String(e) }));
  }

  if (action.tag === "stopFinished") {
    // The stop route marks the run stopped synchronously, so the list is
    // stale until refetched. If the stopped run is the open one, the next
    // tail tick observes the terminal status and winds the loop down on
    // its own — nothing extra to do for that case here.
    return refreshRunViews(draft, env);
  }

  if (action.tag === "stopFailed") {
    draft.stopError = action.error;
    return null;
  }

  if (action.tag === "resumeRun") {
    draft.resumeError = null;
    const { runId, nTrials, nWorkers } = action;
    return env
      .resumeRun(runId, nTrials, nWorkers)
      .map((): BenchAction => ({ tag: "resumeFinished", runId }))
      .catch((e): BenchAction => ({ tag: "resumeFailed", runId, error: String(e) }));
  }

  if (action.tag === "resumeFinished") {
    // The resumed run is a brand-new row (its own run_id) -- refresh the
    // list so it shows up without a manual reload, same as a fresh launch.
    return refreshRunViews(draft, env);
  }

  if (action.tag === "resumeFailed") {
    draft.resumeError = action.error;
    return null;
  }

  if (action.tag === "advanceBaseline") {
    draft.advanceBaselineError = null;
    const { runId, nTrials, nWorkers } = action;
    return env
      .advanceBaseline(runId, nTrials, nWorkers)
      .map((r): BenchAction => ({ tag: "advanceBaselineFinished", runId, newRunId: r.run_id }))
      .catch((e): BenchAction => ({ tag: "advanceBaselineFailed", runId, error: String(e) }));
  }

  if (action.tag === "advanceBaselineFinished") {
    // The new physical run changes the navigator's available evidence, but
    // does not change the operator's current physical selection.
    return refreshRunViews(draft, env);
  }

  if (action.tag === "advanceBaselineFailed") {
    draft.advanceBaselineError = action.error;
    return null;
  }

  if (action.tag === "deleteRun") {
    draft.deleteError = null;
    return env
      .deleteRun(action.runId)
      .map((): BenchAction => ({ tag: "deleteFinished", runId: action.runId }))
      .catch((e): BenchAction => ({ tag: "deleteFailed", runId: action.runId, error: String(e) }));
  }

  if (action.tag === "deleteFinished") {
    if (draft.openRun?.runId === action.runId) {
      draft.openRun = null;
      draft.openGeneration += 1;
    }
    return refreshRunViews(draft, env);
  }

  if (action.tag === "deleteFailed") {
    draft.deleteError = action.error;
    return null;
  }

  if (action.tag === "setShowLaunchForm") {
    draft.showLaunchForm = action.show;
    return null;
  }

  if (action.tag === "kinds") {
    const ka = action.action;
    if (ka.tag === "request") {
      const jobEnv: JobPollEnv<BenchKindInfo[]> = {
        submitJob: () =>
          env.getBenchKinds().map(
            (result): JobSubmitResult<BenchKindInfo[]> => ({
              status: "done",
              result,
            }),
          ),
        pollJob: () => {
          throw new Error(
            "unreachable: kinds resolves synchronously (see submitJob above)",
          );
        },
      };
      const eff = jobPollReduce(draft.kinds, { tag: "start" }, jobEnv);
      return eff
        ? eff.map(
            (a): BenchAction => ({ tag: "kinds", action: { tag: "job", action: a } }),
          )
        : null;
    }
    const eff = jobPollReduce(
      draft.kinds,
      ka.action,
      unreachableJobEnv(
        "unreachable: a forwarded kinds/job action never re-submits or polls",
      ),
    );
    return eff
      ? eff.map(
          (a): BenchAction => ({
            tag: "kinds",
            action: { tag: "job", action: a },
          }),
        )
      : null;
  }

  if (action.tag === "tunerKinds") {
    const ka = action.action;
    if (ka.tag === "request") {
      const jobEnv: JobPollEnv<TunerGameInfo[]> = {
        submitJob: () =>
          env.getTunerKinds().map(
            (result): JobSubmitResult<TunerGameInfo[]> => ({
              status: "done",
              result,
            }),
          ),
        pollJob: () => {
          throw new Error(
            "unreachable: tunerKinds resolves synchronously (see submitJob above)",
          );
        },
      };
      const eff = jobPollReduce(draft.tunerKinds, { tag: "start" }, jobEnv);
      return eff
        ? eff.map(
            (a): BenchAction => ({ tag: "tunerKinds", action: { tag: "job", action: a } }),
          )
        : null;
    }
    const eff = jobPollReduce(
      draft.tunerKinds,
      ka.action,
      unreachableJobEnv(
        "unreachable: a forwarded tunerKinds/job action never re-submits or polls",
      ),
    );
    return eff
      ? eff.map(
          (a): BenchAction => ({
            tag: "tunerKinds",
            action: { tag: "job", action: a },
          }),
        )
      : null;
  }

  return null;
}
