// tuner-reducer.ts — the version-4 tuner UI's reducer. One `RemoteData<T>`
// slot per server resource; components dispatch and read, they never fetch
// (every network call is an `Effect` on the injected `TunerEnv`, per
// AGENTS.md "mock the environment").
//
// Two self-scheduling poll loops, both built from `Effect.delay` and sized
// by the pure cadence functions in `tuner-poll.ts`:
//   - the fleet journal (`listRuns`): polls every `JOURNAL_POLL_MS` while
//     any run reports `status: "live"`, and stops once every run has exited.
//   - the open run's launch-log tail (`getRunLog`): polls while that run is
//     still live in the journal, so the overview shows what the detached
//     process is doing before the projection catches up.
// Each loop carries the generation it was scheduled under; opening a
// different run or re-initialising invalidates whatever is still in flight.

import { Effect } from "@mcts/core";
import {
  idle,
  toErr,
  toLoading,
  toOk,
  peek,
  type RemoteData,
} from "./remote-data.js";
import { JOURNAL_POLL_MS, journalPollDelayMs } from "./tuner-poll.js";
import type { TunerEnv } from "./tuner-env.js";
import type {
  ProjectionRunListItem,
  TunerLaunchRequest,
  TunerObjectiveFile,
  TunerRunView,
} from "./tuner-types.js";
import type { TunerGameInfo } from "../types.js";

/** Fixed cadence for the open run's launch-log tail. */
export const LOG_TAIL_MS = 3_000;

export interface TunerLaunchState {
  status: "idle" | "pending" | "done" | "error";
  error: string | null;
  /** run id of the last run this session launched — used to highlight it in
   * the fleet and to open its overview. */
  lastRunId: string | null;
}

export interface TunerLogTailState {
  lines: string[];
  errLines: string[];
  offset: number;
  error: string | null;
  /** false once the open run has exited (its `launch.out` is complete) — no
   * further ticks are scheduled. */
  active: boolean;
}

export interface TunerState {
  /** Operational journal — liveness only, no science. */
  runs: RemoteData<TunerRunView[]>;
  /** Completed / failed runs from the projection (science list). */
  projectionRuns: RemoteData<ProjectionRunListItem[]>;
  kinds: RemoteData<TunerGameInfo[]>;
  objectives: RemoteData<TunerObjectiveFile[]>;
  launch: TunerLaunchState;
  /** null → fleet dashboard; a run id → that run's overview. */
  openRunId: string | null;
  log: TunerLogTailState;
  stopError: string | null;
  /** true while a manual `projection/refresh` POST is in flight. */
  refreshing: boolean;
  refreshError: string | null;
  lastProjectionRefreshAt: number | null;
  journalGeneration: number;
  logGeneration: number;
}

export function initialTunerState(): TunerState {
  return {
    runs: idle(),
    projectionRuns: idle(),
    kinds: idle(),
    objectives: idle(),
    launch: { status: "idle", error: null, lastRunId: null },
    openRunId: null,
    log: { lines: [], errLines: [], offset: 0, error: null, active: false },
    stopError: null,
    refreshing: false,
    refreshError: null,
    lastProjectionRefreshAt: null,
    journalGeneration: 0,
    logGeneration: 0,
  };
}

export type TunerAction =
  | { tag: "init" }
  | { tag: "kindsLoaded"; kinds: TunerGameInfo[] }
  | { tag: "kindsFailed"; error: string }
  | { tag: "objectivesLoaded"; objectives: TunerObjectiveFile[] }
  | { tag: "objectivesFailed"; error: string }
  | { tag: "journalTick"; generation: number }
  | { tag: "runsLoaded"; runs: TunerRunView[] }
  | { tag: "runsFailed"; error: string }
  | { tag: "projectionLoaded"; runs: ProjectionRunListItem[] }
  | { tag: "projectionFailed"; error: string }
  | { tag: "refreshProjection" }
  | { tag: "refreshDone" }
  | { tag: "refreshFailed"; error: string }
  | { tag: "launch"; request: TunerLaunchRequest }
  | { tag: "launchOk"; run: TunerRunView }
  | { tag: "launchFailed"; error: string }
  | { tag: "openRun"; runId: string }
  | { tag: "closeRun" }
  | { tag: "logTick"; generation: number }
  | {
      tag: "logLoaded";
      generation: number;
      lines: string[];
      errLines: string[];
      nextOffset: number;
    }
  | { tag: "logFailed"; generation: number; error: string }
  | { tag: "stopRun"; runId: string }
  | { tag: "stopOk" }
  | { tag: "stopFailed"; error: string };

const liveCount = (runs: TunerRunView[] | undefined): number =>
  (runs ?? []).filter((r) => r.status === "live").length;

const isOpenRunLive = (draft: TunerState): boolean => {
  const runs = peek(draft.runs) ?? [];
  return runs.some((r) => r.run_id === draft.openRunId && r.status === "live");
};

/** Fetch the journal once, tagged with the loop generation. */
function fetchJournal(env: TunerEnv): Effect<TunerAction> {
  return env
    .listRuns()
    .map((runs): TunerAction => ({ tag: "runsLoaded", runs }))
    .catch((e): TunerAction => ({ tag: "runsFailed", error: String(e) }));
}

function fetchProjection(env: TunerEnv): Effect<TunerAction> {
  return env
    .listProjectionRuns()
    .map((runs): TunerAction => ({ tag: "projectionLoaded", runs }))
    .catch((e): TunerAction => ({ tag: "projectionFailed", error: String(e) }));
}

function fetchLog(env: TunerEnv, runId: string, since: number, generation: number): Effect<TunerAction> {
  return env
    .getRunLog(runId, since)
    .map((log): TunerAction => ({
      tag: "logLoaded",
      generation,
      lines: log.lines,
      errLines: log.err_lines,
      nextOffset: log.next_offset,
    }))
    .catch((e): TunerAction => ({ tag: "logFailed", generation, error: String(e) }));
}

export function tunerReducer(
  draft: TunerState,
  action: TunerAction,
  env: TunerEnv,
): Effect<TunerAction> | null {
  switch (action.tag) {
    case "init": {
      draft.kinds = toLoading(draft.kinds);
      draft.objectives = toLoading(draft.objectives);
      draft.runs = toLoading(draft.runs);
      draft.projectionRuns = toLoading(draft.projectionRuns);
      draft.journalGeneration += 1;
      return Effect.merge(
        env
          .listKinds()
          .map((kinds): TunerAction => ({ tag: "kindsLoaded", kinds }))
          .catch((e): TunerAction => ({ tag: "kindsFailed", error: String(e) })),
        env
          .listObjectives()
          .map((objectives): TunerAction => ({ tag: "objectivesLoaded", objectives }))
          .catch((e): TunerAction => ({ tag: "objectivesFailed", error: String(e) })),
        fetchJournal(env),
        fetchProjection(env),
      );
    }

    case "kindsLoaded":
      draft.kinds = toOk(action.kinds, Date.now());
      return null;
    case "kindsFailed":
      draft.kinds = toErr(action.error, draft.kinds);
      return null;
    case "objectivesLoaded":
      draft.objectives = toOk(action.objectives, Date.now());
      return null;
    case "objectivesFailed":
      draft.objectives = toErr(action.error, draft.objectives);
      return null;

    case "journalTick": {
      if (action.generation !== draft.journalGeneration) return null;
      return fetchJournal(env);
    }

    case "runsLoaded": {
      const before = liveCount(peek(draft.runs));
      draft.runs = toOk(action.runs, Date.now());
      const after = liveCount(action.runs);
      const delay = journalPollDelayMs(after);
      const tick =
        delay === null
          ? null
          : Effect.delay<TunerAction>(delay, {
              tag: "journalTick",
              generation: draft.journalGeneration,
            });
      // A run just went terminal — pull a fresh projection so the completed
      // list gains its row without waiting for a manual refresh.
      const refresh = after < before ? Effect.send<TunerAction>({ tag: "refreshProjection" }) : null;
      if (tick && refresh) return Effect.merge(tick, refresh);
      return tick ?? refresh;
    }
    case "runsFailed": {
      draft.runs = toErr(action.error, draft.runs);
      // Keep trying — a transient server blip shouldn't kill the fleet view.
      return Effect.delay<TunerAction>(JOURNAL_POLL_MS, {
        tag: "journalTick",
        generation: draft.journalGeneration,
      });
    }

    case "projectionLoaded":
      draft.projectionRuns = toOk(action.runs, Date.now());
      return null;
    case "projectionFailed":
      draft.projectionRuns = toErr(action.error, draft.projectionRuns);
      return null;

    case "refreshProjection": {
      if (draft.refreshing) return null;
      draft.refreshing = true;
      draft.refreshError = null;
      return env
        .refreshProjection()
        .map((): TunerAction => ({ tag: "refreshDone" }))
        .catch((e): TunerAction => ({ tag: "refreshFailed", error: String(e) }));
    }
    case "refreshDone":
      draft.refreshing = false;
      draft.lastProjectionRefreshAt = Date.now();
      return fetchProjection(env);
    case "refreshFailed":
      draft.refreshing = false;
      draft.refreshError = action.error;
      return null;

    case "launch": {
      if (draft.launch.status === "pending") return null;
      draft.launch = { status: "pending", error: null, lastRunId: null };
      return env
        .launchRun(action.request)
        .map((run): TunerAction => ({ tag: "launchOk", run }))
        .catch((e): TunerAction => ({ tag: "launchFailed", error: String(e) }));
    }
    case "launchOk": {
      draft.launch = { status: "done", error: null, lastRunId: action.run.run_id };
      // Optimistic insert so the fleet shows the run as `live` immediately.
      const current = peek(draft.runs) ?? [];
      if (!current.some((r) => r.run_id === action.run.run_id)) {
        draft.runs = toOk([action.run, ...current], Date.now());
      }
      // Open its overview and (re)start both loops.
      draft.openRunId = action.run.run_id;
      draft.logGeneration += 1;
      draft.journalGeneration += 1;
      draft.log = { lines: [], errLines: [], offset: 0, error: null, active: true };
      return Effect.merge(
        Effect.send<TunerAction>({ tag: "logTick", generation: draft.logGeneration }),
        Effect.send<TunerAction>({ tag: "journalTick", generation: draft.journalGeneration }),
      );
    }
    case "launchFailed":
      draft.launch = { status: "error", error: action.error, lastRunId: null };
      return null;

    case "openRun": {
      draft.openRunId = action.runId;
      draft.stopError = null;
      draft.logGeneration += 1;
      draft.log = { lines: [], errLines: [], offset: 0, error: null, active: true };
      return Effect.send<TunerAction>({ tag: "logTick", generation: draft.logGeneration });
    }
    case "closeRun":
      draft.openRunId = null;
      draft.log = { lines: [], errLines: [], offset: 0, error: null, active: false };
      return null;

    case "logTick": {
      if (action.generation !== draft.logGeneration || !draft.openRunId || !draft.log.active) {
        return null;
      }
      return fetchLog(env, draft.openRunId, draft.log.offset, action.generation);
    }
    case "logLoaded": {
      if (action.generation !== draft.logGeneration) return null;
      draft.log.lines.push(...action.lines);
      draft.log.errLines = action.errLines;
      draft.log.offset = action.nextOffset;
      draft.log.error = null;
      if (!isOpenRunLive(draft)) {
        draft.log.active = false;
        return null;
      }
      return Effect.delay<TunerAction>(LOG_TAIL_MS, {
        tag: "logTick",
        generation: action.generation,
      });
    }
    case "logFailed": {
      if (action.generation !== draft.logGeneration) return null;
      draft.log.error = action.error;
      if (!isOpenRunLive(draft)) {
        draft.log.active = false;
        return null;
      }
      return Effect.delay<TunerAction>(LOG_TAIL_MS, {
        tag: "logTick",
        generation: action.generation,
      });
    }

    case "stopRun": {
      draft.stopError = null;
      return env
        .stopRun(action.runId)
        .map((): TunerAction => ({ tag: "stopOk" }))
        .catch((e): TunerAction => ({ tag: "stopFailed", error: String(e) }));
    }
    case "stopOk":
      return fetchJournal(env);
    case "stopFailed":
      draft.stopError = action.error;
      return null;
  }
}
