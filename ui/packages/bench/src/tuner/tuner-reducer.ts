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
import { idle, toErr, toLoading, toOk, peek, type RemoteData } from "./remote-data.js";
import { JOURNAL_POLL_MS, journalPollDelayMs } from "./tuner-poll.js";
import type { TunerEnv } from "./tuner-env.js";
import type {
  ProjectionCandidate,
  ProjectionGameRow,
  ProjectionPairRow,
  ProjectionRunDetail,
  ProjectionRunListItem,
  ProjectionValidation,
  ObjectiveValidationResult,
  TunerLaunchRequest,
  TunerObjectiveDetail,
  TunerObjectiveFile,
  TunerRunView,
} from "./tuner-types.js";
import type { JsonValue, TunerGameInfo } from "../types.js";

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
  /** The objective the editor has open (`null` in create mode), keyed so a
   * stale detail response for a previously-open objective is ignored. */
  openObjectiveKey: string | null;
  objectiveDetail: RemoteData<TunerObjectiveDetail>;
  objectiveSave: { status: "idle" | "pending" | "done" | "error"; error: string | null };
  objectiveValidation: RemoteData<ObjectiveValidationResult>;
  /** Key of the objective currently being deleted, or null. */
  objectiveMutating: string | null;
  objectiveMutateError: string | null;
  launch: TunerLaunchState;
  /** null → fleet dashboard; a run id → that run's overview. */
  openRunId: string | null;
  /** Per-run projection resources for the open run's overview / drawer.
   * Reloaded (under a fresh `resourceGeneration`) whenever the open run
   * changes or a `projection/refresh` completes. */
  projectionDetail: RemoteData<ProjectionRunDetail>;
  validation: RemoteData<ProjectionValidation>;
  candidates: RemoteData<ProjectionCandidate[]>;
  /** Evidence view: the open run's pair rows (server-capped; filtered
   * client-side by the pairs table). */
  pairs: RemoteData<ProjectionPairRow[]>;
  report: RemoteData<JsonValue>;
  /** `?candidate=<cid>` — the candidate drawer's subject, or null. */
  openCandidateId: string | null;
  /** The pair whose inspector is open in the evidence view, or null. */
  openPairId: string | null;
  /** Seat-swapped game summaries for `openPairId`. */
  pairGames: RemoteData<ProjectionGameRow[]>;
  pairGamesGeneration: number;
  resourceGeneration: number;
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
    openObjectiveKey: null,
    objectiveDetail: idle(),
    objectiveSave: { status: "idle", error: null },
    objectiveValidation: idle(),
    objectiveMutating: null,
    objectiveMutateError: null,
    launch: { status: "idle", error: null, lastRunId: null },
    openRunId: null,
    projectionDetail: idle(),
    validation: idle(),
    candidates: idle(),
    pairs: idle(),
    report: idle(),
    openCandidateId: null,
    openPairId: null,
    pairGames: idle(),
    pairGamesGeneration: 0,
    resourceGeneration: 0,
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
  | { tag: "openObjective"; key: string | null }
  | { tag: "closeObjective" }
  | { tag: "objectiveDetailLoaded"; key: string; detail: TunerObjectiveDetail }
  | { tag: "objectiveDetailFailed"; key: string; error: string }
  | { tag: "saveObjective"; key: string; content: JsonValue }
  | { tag: "saveObjectiveOk"; detail: TunerObjectiveDetail }
  | { tag: "saveObjectiveFailed"; error: string }
  | { tag: "deleteObjective"; key: string }
  | { tag: "deleteObjectiveOk" }
  | { tag: "deleteObjectiveFailed"; error: string }
  | { tag: "validateObjective"; key: string; content: JsonValue }
  | { tag: "validateObjectiveOk"; result: ObjectiveValidationResult }
  | { tag: "validateObjectiveFailed"; error: string }
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
  | { tag: "loadRunResources"; runId: string }
  | { tag: "detailLoaded"; generation: number; detail: ProjectionRunDetail }
  | { tag: "detailFailed"; generation: number; error: string }
  | { tag: "validationLoaded"; generation: number; validation: ProjectionValidation }
  | { tag: "validationFailed"; generation: number; error: string }
  | { tag: "candidatesLoaded"; generation: number; candidates: ProjectionCandidate[] }
  | { tag: "candidatesFailed"; generation: number; error: string }
  | { tag: "pairsLoaded"; generation: number; pairs: ProjectionPairRow[] }
  | { tag: "pairsFailed"; generation: number; error: string }
  | { tag: "selectPair"; pairId: string | null }
  | { tag: "pairGamesLoaded"; generation: number; games: ProjectionGameRow[] }
  | { tag: "pairGamesFailed"; generation: number; error: string }
  | { tag: "reportLoaded"; generation: number; report: JsonValue }
  | { tag: "reportFailed"; generation: number; error: string }
  | { tag: "openCandidate"; candidateId: string }
  | { tag: "closeCandidate" }
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

function fetchObjectives(env: TunerEnv): Effect<TunerAction> {
  return env
    .listObjectives()
    .map((objectives): TunerAction => ({ tag: "objectivesLoaded", objectives }))
    .catch((e): TunerAction => ({ tag: "objectivesFailed", error: String(e) }));
}

function fetchProjection(env: TunerEnv): Effect<TunerAction> {
  return env
    .listProjectionRuns()
    .map((runs): TunerAction => ({ tag: "projectionLoaded", runs }))
    .catch((e): TunerAction => ({ tag: "projectionFailed", error: String(e) }));
}

function fetchLog(
  env: TunerEnv,
  runId: string,
  since: number,
  generation: number,
): Effect<TunerAction> {
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

/** Load every per-run projection resource the overview / drawer needs,
 * tagged with the current `resourceGeneration` so a stale response for a
 * previously-open run is ignored. */
function fetchRunResources(env: TunerEnv, runId: string, generation: number): Effect<TunerAction> {
  return Effect.merge(
    env
      .getProjectionRun(runId)
      .map((detail): TunerAction => ({ tag: "detailLoaded", generation, detail }))
      .catch((e): TunerAction => ({ tag: "detailFailed", generation, error: String(e) })),
    env
      .getProjectionValidation(runId)
      .map((validation): TunerAction => ({ tag: "validationLoaded", generation, validation }))
      .catch((e): TunerAction => ({ tag: "validationFailed", generation, error: String(e) })),
    env
      .getProjectionCandidates(runId)
      .map((candidates): TunerAction => ({ tag: "candidatesLoaded", generation, candidates }))
      .catch((e): TunerAction => ({ tag: "candidatesFailed", generation, error: String(e) })),
    env
      .getProjectionPairs(runId)
      .map((pairs): TunerAction => ({ tag: "pairsLoaded", generation, pairs }))
      .catch((e): TunerAction => ({ tag: "pairsFailed", generation, error: String(e) })),
    env
      .getProjectionReport(runId)
      .map((report): TunerAction => ({ tag: "reportLoaded", generation, report }))
      .catch((e): TunerAction => ({ tag: "reportFailed", generation, error: String(e) })),
  );
}

function startResourceLoad(draft: TunerState, env: TunerEnv, runId: string): Effect<TunerAction> {
  draft.resourceGeneration += 1;
  draft.projectionDetail = toLoading(draft.projectionDetail);
  draft.validation = toLoading(draft.validation);
  draft.candidates = toLoading(draft.candidates);
  draft.pairs = toLoading(draft.pairs);
  draft.report = toLoading(draft.report);
  return fetchRunResources(env, runId, draft.resourceGeneration);
}

function clearResources(draft: TunerState): void {
  draft.resourceGeneration += 1;
  draft.projectionDetail = idle();
  draft.validation = idle();
  draft.candidates = idle();
  draft.pairs = idle();
  draft.report = idle();
  draft.openCandidateId = null;
  draft.openPairId = null;
  draft.pairGames = idle();
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
        fetchObjectives(env),
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

    case "openObjective": {
      draft.openObjectiveKey = action.key;
      draft.objectiveSave = { status: "idle", error: null };
      draft.objectiveValidation = idle();
      draft.objectiveMutateError = null;
      if (action.key === null) {
        draft.objectiveDetail = idle();
        return null;
      }
      draft.objectiveDetail = toLoading(draft.objectiveDetail);
      const key = action.key;
      return env
        .getObjective(key)
        .map((detail): TunerAction => ({ tag: "objectiveDetailLoaded", key, detail }))
        .catch((e): TunerAction => ({ tag: "objectiveDetailFailed", key, error: String(e) }));
    }
    case "closeObjective":
      draft.openObjectiveKey = null;
      draft.objectiveDetail = idle();
      draft.objectiveSave = { status: "idle", error: null };
      draft.objectiveValidation = idle();
      draft.objectiveMutateError = null;
      return null;
    case "objectiveDetailLoaded":
      if (action.key !== draft.openObjectiveKey) return null;
      draft.objectiveDetail = toOk(action.detail, Date.now());
      return null;
    case "objectiveDetailFailed":
      if (action.key !== draft.openObjectiveKey) return null;
      draft.objectiveDetail = toErr(action.error, draft.objectiveDetail);
      return null;

    case "saveObjective": {
      if (draft.objectiveSave.status === "pending") return null;
      draft.objectiveSave = { status: "pending", error: null };
      const { key, content } = action;
      return env
        .putObjective(key, content)
        .map((detail): TunerAction => ({ tag: "saveObjectiveOk", detail }))
        .catch((e): TunerAction => ({ tag: "saveObjectiveFailed", error: String(e) }));
    }
    case "saveObjectiveOk":
      draft.objectiveSave = { status: "done", error: null };
      draft.openObjectiveKey = action.detail.key;
      draft.objectiveDetail = toOk(action.detail, Date.now());
      // Re-list so the manager and launch form reflect the change immediately.
      return fetchObjectives(env);
    case "saveObjectiveFailed":
      draft.objectiveSave = { status: "error", error: action.error };
      return null;

    case "deleteObjective": {
      if (draft.objectiveMutating) return null;
      draft.objectiveMutating = action.key;
      draft.objectiveMutateError = null;
      return env
        .deleteObjective(action.key)
        .map((): TunerAction => ({ tag: "deleteObjectiveOk" }))
        .catch((e): TunerAction => ({ tag: "deleteObjectiveFailed", error: String(e) }));
    }
    case "deleteObjectiveOk":
      draft.objectiveMutating = null;
      return fetchObjectives(env);
    case "deleteObjectiveFailed":
      draft.objectiveMutating = null;
      draft.objectiveMutateError = action.error;
      return null;

    case "validateObjective": {
      draft.objectiveValidation = toLoading(draft.objectiveValidation);
      const { key, content } = action;
      return env
        .validateObjective(key, content)
        .map((result): TunerAction => ({ tag: "validateObjectiveOk", result }))
        .catch((e): TunerAction => ({ tag: "validateObjectiveFailed", error: String(e) }));
    }
    case "validateObjectiveOk":
      draft.objectiveValidation = toOk(action.result, Date.now());
      return null;
    case "validateObjectiveFailed":
      draft.objectiveValidation = toErr(action.error, draft.objectiveValidation);
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
      const refresh =
        after < before ? Effect.send<TunerAction>({ tag: "refreshProjection" }) : null;
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
    case "refreshDone": {
      draft.refreshing = false;
      draft.lastProjectionRefreshAt = Date.now();
      const list = fetchProjection(env);
      return draft.openRunId
        ? Effect.merge(list, startResourceLoad(draft, env, draft.openRunId))
        : list;
    }
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
        startResourceLoad(draft, env, action.run.run_id),
      );
    }
    case "launchFailed":
      draft.launch = { status: "error", error: action.error, lastRunId: null };
      return null;

    case "openRun": {
      const changed = draft.openRunId !== action.runId;
      draft.openRunId = action.runId;
      draft.stopError = null;
      draft.logGeneration += 1;
      draft.log = { lines: [], errLines: [], offset: 0, error: null, active: true };
      const logTick = Effect.send<TunerAction>({
        tag: "logTick",
        generation: draft.logGeneration,
      });
      if (!changed) return logTick;
      draft.openCandidateId = null;
      draft.openPairId = null;
      draft.pairGames = idle();
      return Effect.merge(logTick, startResourceLoad(draft, env, action.runId));
    }
    case "closeRun":
      draft.openRunId = null;
      draft.log = { lines: [], errLines: [], offset: 0, error: null, active: false };
      clearResources(draft);
      return null;

    case "loadRunResources": {
      if (action.runId !== draft.openRunId) return null;
      return startResourceLoad(draft, env, action.runId);
    }
    case "detailLoaded":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.projectionDetail = toOk(action.detail, Date.now());
      return null;
    case "detailFailed":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.projectionDetail = toErr(action.error, draft.projectionDetail);
      return null;
    case "validationLoaded":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.validation = toOk(action.validation, Date.now());
      return null;
    case "validationFailed":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.validation = toErr(action.error, draft.validation);
      return null;
    case "candidatesLoaded":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.candidates = toOk(action.candidates, Date.now());
      return null;
    case "candidatesFailed":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.candidates = toErr(action.error, draft.candidates);
      return null;
    case "pairsLoaded":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.pairs = toOk(action.pairs, Date.now());
      return null;
    case "pairsFailed":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.pairs = toErr(action.error, draft.pairs);
      return null;
    case "selectPair": {
      draft.openPairId = action.pairId;
      draft.pairGamesGeneration += 1;
      if (!action.pairId || !draft.openRunId) {
        draft.pairGames = idle();
        return null;
      }
      draft.pairGames = toLoading(draft.pairGames);
      const generation = draft.pairGamesGeneration;
      const runId = draft.openRunId;
      return env
        .getProjectionPairGames(runId, action.pairId)
        .map((games): TunerAction => ({ tag: "pairGamesLoaded", generation, games }))
        .catch((e): TunerAction => ({ tag: "pairGamesFailed", generation, error: String(e) }));
    }
    case "pairGamesLoaded":
      if (action.generation !== draft.pairGamesGeneration) return null;
      draft.pairGames = toOk(action.games, Date.now());
      return null;
    case "pairGamesFailed":
      if (action.generation !== draft.pairGamesGeneration) return null;
      draft.pairGames = toErr(action.error, draft.pairGames);
      return null;
    case "reportLoaded":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.report = toOk(action.report, Date.now());
      return null;
    case "reportFailed":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.report = toErr(action.error, draft.report);
      return null;

    case "openCandidate":
      draft.openCandidateId = action.candidateId;
      return null;
    case "closeCandidate":
      draft.openCandidateId = null;
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
