// reducer.ts — Bench reducer: run list (with filters), one open run's
// detail + live log tail, launch/stop.
//
// The one-shot fetches (runs list, launch) go through @mcts/core's
// jobPollReduce with a `submitJob` that resolves directly to
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
import type { BenchState } from "./state.js";
import {
  isTerminalStatus,
  type LaunchResponse,
  type RunDetail,
  type RunFilters,
  type RunLogResponse,
  type RunSummary,
  type TunerGameInfo,
  type StopResponse,
  type TrialRow,
  type GameTraceSummary,
  type GameMove,
} from "./types.js";

/** Every network operation the bench reducer may perform, lifted to
 * `Effect` — hard rule (enforced by ui/eslint.config.js's fetch ban): no
 * reducer or component calls `fetch`/`BenchApiClient` directly, only
 * `env.xxx()`. */
export interface BenchEnv {
  listRuns(filters: RunFilters): Effect<RunSummary[]>;
  getRun(runId: string): Effect<RunDetail>;
  getRunLog(runId: string, since: number): Effect<RunLogResponse>;
  /** Fetch the full raw content of the run's stdout.log file (stderr
   * output redirected by the launcher). */
  getRunStdout(runId: string): Effect<string>;
  launchRun(kind: string, game: string, config?: unknown): Effect<LaunchResponse>;
  stopRun(runId: string): Effect<StopResponse>;
  /** Per-game tuner metadata for every tuner-tunable game. */
  getTunerKinds(): Effect<TunerGameInfo[]>;
  /** Trial rows for one run, oldest first. */
  getRunTrials(runId: string, limit: number): Effect<TrialRow[]>;
  getRunGames(runId: string, limit?: number, cellId?: string | null): Effect<GameTraceSummary[]>;
  getRunGameMoves(runId: string, gameSeq: number): Effect<GameMove[]>;
  deleteRun(runId: string): Effect<void>;
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

export type RunsAction = { tag: "request" } | { tag: "job"; action: JobPollAction<RunSummary[]> };

export type LaunchAction =
  | { tag: "request"; kind: string; game: string; config?: unknown }
  | { tag: "job"; action: JobPollAction<LaunchResponse> };

export type BenchAction =
  | { tag: "runs"; action: RunsAction }
  /** Replace the run-list filters and refetch with them. */
  | {
      tag: "setRunFilters";
      status: string | null;
      game: string | null;
      project_id?: string | null;
      experiment_id?: string | null;
    }
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
      /** Recorded physical-run trials, retained as diagnostics. */
      trials: TrialRow[];
      games?: GameTraceSummary[];
    }
  | { tag: "tailFailed"; generation: number; error: string }
  | { tag: "launch"; action: LaunchAction }
  | { tag: "stopRun"; runId: string }
  | { tag: "stopFinished"; runId: string }
  | { tag: "stopFailed"; runId: string; error: string }
  | { tag: "deleteRun"; runId: string }
  | { tag: "deleteFinished"; runId: string }
  | { tag: "deleteFailed"; runId: string; error: string }
  /** Load per-game tuner tuner metadata for the launch form + run detail. */
  | { tag: "tunerKinds"; action: TunerKindsAction }
  | { tag: "setShowLaunchForm"; show: boolean };

export type TunerKindsAction =
  { tag: "request" } | { tag: "job"; action: JobPollAction<TunerGameInfo[]> };

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
      env
        .listRuns(filters)
        .map((result): JobSubmitResult<RunSummary[]> => ({ status: "done", result })),
    pollJob: () => {
      throw new Error("unreachable: the runs list resolves synchronously (see submitJob above)");
    },
  };
  const eff = jobPollReduce(draft.runs, { tag: "start" }, jobEnv);
  return eff
    ? eff.map((a): BenchAction => ({ tag: "runs", action: { tag: "job", action: a } }))
    : null;
}

/** Refetch the physical run rows (the round-robin run list). */
function refreshRunViews(draft: BenchState, env: BenchEnv): Effect<BenchAction> | null {
  return startRunsFetch(draft, env);
}

export function benchReducer(
  draft: BenchState,
  action: BenchAction,
  env: BenchEnv,
): Effect<BenchAction> | null {
  if (action.tag === "runs") {
    const ra = action.action;
    if (ra.tag === "request") return refreshRunViews(draft, env);
    const eff = jobPollReduce(
      draft.runs,
      ra.action,
      unreachableJobEnv("unreachable: a forwarded runs/job action never re-submits or polls"),
    );
    return eff
      ? eff.map((a): BenchAction => ({ tag: "runs", action: { tag: "job", action: a } }))
      : null;
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
    draft.showLaunchForm = false;
    draft.openGeneration += 1;
    draft.openRun = {
      runId: action.runId,
      detail: null,
      tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
      trials: [],
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
    return Effect.fromPromise(async () => {
      const [log, detail, games] = await Promise.all([
        toPromise(env.getRunLog(runId, since)),
        toPromise(env.getRun(runId)),
        toPromise(env.getRunGames(runId, 5000, null)),
      ]);
      return { log, detail, games };
    })
      .map((r): BenchAction => ({
        tag: "tailed",
        generation,
        lines: r.log.lines,
        nextOffset: r.log.next_offset,
        detail: r.detail,
        trials: [],
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

  if (action.tag === "launch") {
    const la = action.action;
    if (la.tag === "request") {
      const { kind, game, config } = la;
      const jobEnv: JobPollEnv<LaunchResponse> = {
        submitJob: () =>
          env
            .launchRun(kind, game, config)
            .map((result): JobSubmitResult<LaunchResponse> => ({ status: "done", result })),
        pollJob: () => {
          throw new Error("unreachable: launch resolves synchronously (see submitJob above)");
        },
      };
      const eff = jobPollReduce(draft.launch, { tag: "start" }, jobEnv);
      return eff
        ? eff.map((a): BenchAction => ({ tag: "launch", action: { tag: "job", action: a } }))
        : null;
    }
    const eff = jobPollReduce(
      draft.launch,
      la.action,
      unreachableJobEnv("unreachable: a forwarded launch/job action never re-submits or polls"),
    );
    const launchEff = eff
      ? eff.map((a): BenchAction => ({ tag: "launch", action: { tag: "job", action: a } }))
      : null;
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

  if (action.tag === "tunerKinds") {
    const ka = action.action;
    if (ka.tag === "request") {
      const jobEnv: JobPollEnv<TunerGameInfo[]> = {
        submitJob: () =>
          env.getTunerKinds().map((result): JobSubmitResult<TunerGameInfo[]> => ({
            status: "done",
            result,
          })),
        pollJob: () => {
          throw new Error("unreachable: tunerKinds resolves synchronously (see submitJob above)");
        },
      };
      const eff = jobPollReduce(draft.tunerKinds, { tag: "start" }, jobEnv);
      return eff
        ? eff.map((a): BenchAction => ({ tag: "tunerKinds", action: { tag: "job", action: a } }))
        : null;
    }
    const eff = jobPollReduce(
      draft.tunerKinds,
      ka.action,
      unreachableJobEnv("unreachable: a forwarded tunerKinds/job action never re-submits or polls"),
    );
    return eff
      ? eff.map((a): BenchAction => ({
          tag: "tunerKinds",
          action: { tag: "job", action: a },
        }))
      : null;
  }

  return null;
}
