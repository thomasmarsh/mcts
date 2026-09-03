// tuner-reducer.ts — the version-4 tuner UI's reducer. One `RemoteData<T>`
// slot per server resource; components dispatch and read, they never fetch
// (every network call is an `Effect` on the injected `TunerEnv`, per
// AGENTS.md "mock the environment").
//
// Three self-scheduling poll loops, all built from `Effect.delay` and sized
// by the pure cadence functions in `tuner-poll.ts`:
//   - the fleet journal (`listRuns`): polls every `JOURNAL_POLL_MS` while
//     any run reports `status: "live"`, and stops once every run has exited.
//   - the open run's launch-log tail (`getRunLog`): polls while that run is
//     still live in the journal, so the overview shows what the detached
//     process is doing before the projection catches up.
//   - the open run's projection auto-refresh: while the open run is live,
//     re-runs the projector every `PROJECTION_REFRESH_MS` and silently
//     reloads the per-run science, so the overview / science / evidence
//     views fill in and keep updating without a manual "Refresh science".
// Each loop carries the generation it was scheduled under; opening a
// different run or re-initialising invalidates whatever is still in flight.

import { Effect } from "@mcts/core";
import { idle, toErr, toLoading, toOk, peek, type RemoteData } from "./remote-data.js";
import {
  JOURNAL_POLL_MS,
  journalPollDelayMs,
  evidencePollDelayMs,
  projectionRefreshDelayMs,
} from "./tuner-poll.js";
import type { TunerEnv } from "./tuner-env.js";
import type {
  EvidenceEnvelope,
  EvidenceTailResponse,
  ProjectionActiveElimination,
  ProjectionCandidate,
  ProjectionGameRow,
  ProjectionObservation,
  ProjectionPairRow,
  ProjectionMeta,
  ProjectionProposal,
  ProjectionRunDetail,
  ProjectionRunListItem,
  ProjectionShadowDecision,
  ProjectionValidation,
  ObjectiveValidationResult,
  LaunchPreflightResult,
  TunerLaunchRequest,
  TunerObjectiveDetail,
  TunerObjectiveFile,
  TunerRunView,
} from "./tuner-types.js";
import type { JsonValue, TunerGameInfo } from "../types.js";

/** Fixed cadence for the open run's launch-log tail. */
export const LOG_TAIL_MS = 3_000;

/** Bounded ring buffer for the open run's live evidence envelopes. */
export const EVIDENCE_RING_MAX = 400;

export interface TunerLaunchState {
  status: "idle" | "pending" | "done" | "error";
  error: string | null;
  /** run id of the last run this session launched — used to highlight it in
   * the fleet and to open its overview. */
  lastRunId: string | null;
}

/** Result of the dry-run the launch form runs against the current form
 * values, so a launch is never attempted for a knowable reason. */
export interface TunerPreflightState {
  status: "idle" | "checking" | "ok" | "invalid" | "error";
  /** Concrete reasons the launch would fail (`status: "invalid"`). */
  errors: string[];
  /** The preflight request itself failed to run (`status: "error"`). */
  error: string | null;
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
  preflight: TunerPreflightState;
  /** Bumped on every preflight request so a stale response is dropped. */
  preflightGeneration: number;
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
  /** Live science row tables — populated on every projection refresh (partial
   * or complete), so the science charts fill in from these before
   * `report.json` exists. Re-fetched on every projection refresh alongside
   * `candidates` / `pairs`. */
  proposals: RemoteData<ProjectionProposal[]>;
  observations: RemoteData<ProjectionObservation[]>;
  shadowDecisions: RemoteData<ProjectionShadowDecision[]>;
  activeEliminations: RemoteData<ProjectionActiveElimination[]>;
  report: RemoteData<JsonValue>;
  /** Set when the evidence stream sees a scientific event since the last
   * projection refresh; shortens the next auto-refresh cycle. Cleared by any
   * completed refresh. */
  scienceStale: boolean;
  /** `?candidate=<cid>` — the candidate drawer's subject, or null. */
  openCandidateId: string | null;
  /** The pair whose inspector is open in the evidence view, or null. */
  openPairId: string | null;
  /** Seat-swapped game summaries for `openPairId`. */
  pairGames: RemoteData<ProjectionGameRow[]>;
  pairGamesGeneration: number;
  resourceGeneration: number;
  /** The open run's live evidence journal: a bounded ring of the most recent
   * envelopes and the highest sequence applied. Populated by the SSE stream
   * (or its degraded poll fallback) while the run is live. */
  evidence: { seq: number; ring: EvidenceEnvelope[] };
  /** true while the evidence stream (or its poll fallback) is following the
   * open live run. */
  evidenceStreamActive: boolean;
  /** false once the SSE push has failed and the reducer has fallen back to
   * polling `getRunEvidence`. */
  evidenceStreamOk: boolean;
  /** Bumped whenever the evidence follower starts or stops, so a stale
   * `evidenceEvents` / poll tick is dropped. */
  evidenceGeneration: number;
  log: TunerLogTailState;
  stopError: string | null;
  /** true while a manual `projection/refresh` POST is in flight. */
  refreshing: boolean;
  refreshError: string | null;
  lastProjectionRefreshAt: number | null;
  /** `projection_meta.last_pass_at` from the server — the headless follower's
   * last pass. Used for the fleet freshness indicator on a cold open, before
   * this tab has ever driven a refresh of its own. */
  projectionLastPassAt: string | null;
  /** true while an automatic (loop-driven) `projection/refresh` POST is in
   * flight — kept separate from `refreshing` so the manual button's spinner
   * doesn't flicker every cadence. */
  autoRefreshing: boolean;
  /** true while the projection auto-refresh loop is scheduled for the open
   * live run. */
  projectionRefreshActive: boolean;
  /** Bumped whenever the auto-refresh loop starts or stops, so a stale
   * `projectionRefreshTick` is dropped. */
  projectionRefreshGeneration: number;
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
    preflight: { status: "idle", errors: [], error: null },
    preflightGeneration: 0,
    openRunId: null,
    projectionDetail: idle(),
    validation: idle(),
    candidates: idle(),
    pairs: idle(),
    proposals: idle(),
    observations: idle(),
    shadowDecisions: idle(),
    activeEliminations: idle(),
    report: idle(),
    scienceStale: false,
    openCandidateId: null,
    openPairId: null,
    pairGames: idle(),
    pairGamesGeneration: 0,
    resourceGeneration: 0,
    evidence: { seq: 0, ring: [] },
    evidenceStreamActive: false,
    evidenceStreamOk: true,
    evidenceGeneration: 0,
    log: { lines: [], errLines: [], offset: 0, error: null, active: false },
    stopError: null,
    refreshing: false,
    refreshError: null,
    lastProjectionRefreshAt: null,
    projectionLastPassAt: null,
    autoRefreshing: false,
    projectionRefreshActive: false,
    projectionRefreshGeneration: 0,
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
  | { tag: "projectionMetaLoaded"; meta: ProjectionMeta }
  | { tag: "projectionUpdatedPush"; generation: number }
  | { tag: "refreshProjection" }
  | { tag: "refreshDone" }
  | { tag: "refreshFailed"; error: string }
  | { tag: "projectionRefreshTick"; generation: number }
  | { tag: "autoRefreshProjection" }
  | { tag: "autoRefreshDone" }
  | { tag: "launch"; request: TunerLaunchRequest }
  | { tag: "launchOk"; run: TunerRunView }
  | { tag: "launchFailed"; error: string }
  | { tag: "preflight"; request: TunerLaunchRequest }
  | { tag: "preflightChecked"; generation: number; result: LaunchPreflightResult }
  | { tag: "preflightErrored"; generation: number; error: string }
  | { tag: "resetPreflight" }
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
  | { tag: "proposalsLoaded"; generation: number; proposals: ProjectionProposal[] }
  | { tag: "proposalsFailed"; generation: number; error: string }
  | { tag: "observationsLoaded"; generation: number; observations: ProjectionObservation[] }
  | { tag: "observationsFailed"; generation: number; error: string }
  | { tag: "shadowDecisionsLoaded"; generation: number; rows: ProjectionShadowDecision[] }
  | { tag: "shadowDecisionsFailed"; generation: number; error: string }
  | { tag: "activeEliminationsLoaded"; generation: number; rows: ProjectionActiveElimination[] }
  | { tag: "activeEliminationsFailed"; generation: number; error: string }
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
  | { tag: "evidenceEvents"; generation: number; events: EvidenceEnvelope[]; nextSeq?: number }
  | { tag: "evidenceStreamEnded"; generation: number }
  | { tag: "evidenceStreamFailed"; generation: number; error: string }
  | { tag: "evidencePollTick"; generation: number }
  | { tag: "evidencePolled"; generation: number; response: EvidenceTailResponse }
  | { tag: "stopRun"; runId: string }
  | { tag: "stopOk" }
  | { tag: "stopFailed"; error: string };

const liveCount = (runs: TunerRunView[] | undefined): number =>
  (runs ?? []).filter((r) => r.status === "live").length;

const isOpenRunLive = (draft: TunerState): boolean => {
  const runs = peek(draft.runs) ?? [];
  return runs.some((r) => r.run_id === draft.openRunId && r.status === "live");
};

/** Start or stop the client-side projection auto-refresh loop. This loop is
 * the **degraded-mode fallback** only: while the evidence SSE stream is
 * healthy the server's headless follower keeps the projection fresh and the
 * `projection-updated` frame tells this tab when to re-fetch, so the client
 * never POSTs a refresh of its own. The loop runs only when the open run is
 * live *and* its stream has failed. Returns the first `projectionRefreshTick`
 * when the loop needs starting, else `null` (bumping the generation so any
 * loop still in flight winds itself down on its next tick). */
function syncAutoRefresh(draft: TunerState): Effect<TunerAction> | null {
  const live = isOpenRunLive(draft) && !draft.evidenceStreamOk;
  if (live && !draft.projectionRefreshActive) {
    draft.projectionRefreshActive = true;
    draft.projectionRefreshGeneration += 1;
    return Effect.send<TunerAction>({
      tag: "projectionRefreshTick",
      generation: draft.projectionRefreshGeneration,
    });
  }
  if (!live && draft.projectionRefreshActive) {
    draft.projectionRefreshActive = false;
    draft.projectionRefreshGeneration += 1;
  }
  return null;
}

/** Evidence event types that change what the projection would materialise —
 * a new one since the last refresh means the science charts are behind. */
const SCIENTIFIC_EVIDENCE_TYPES: ReadonlySet<string> = new Set([
  "pair_completed",
  "observation_completed",
  "shadow_race_decided",
  "cohort_completed",
  "allocation_decided",
  "proposal_created",
  "proposal_accepted",
  "proposal_rejected",
  "finalists_selected",
  "run_completed",
]);

const hasScientificEvent = (events: EvidenceEnvelope[]): boolean =>
  events.some((e) => SCIENTIFIC_EVIDENCE_TYPES.has(e.type));

/** Append envelopes to the bounded ring and advance the applied sequence. */
function applyEvidence(
  draft: TunerState,
  events: EvidenceEnvelope[],
  nextSeq: number,
): void {
  const ring = [...draft.evidence.ring, ...events];
  const seq = events.reduce((max, e) => Math.max(max, e.sequence), draft.evidence.seq);
  draft.evidence = {
    seq: Math.max(seq, nextSeq),
    ring: ring.length > EVIDENCE_RING_MAX ? ring.slice(-EVIDENCE_RING_MAX) : ring,
  };
}

/** The long-lived SSE effect for the open run, tagged with `generation` so a
 * message that arrives after the follower was torn down is ignored. */
function evidenceStreamEffect(
  env: TunerEnv,
  runId: string,
  sinceSeq: number,
  generation: number,
): Effect<TunerAction> {
  return env.openEvidenceStream(runId, sinceSeq).map((message): TunerAction => {
    switch (message.kind) {
      case "events":
        return { tag: "evidenceEvents", generation, events: message.events };
      case "projectionUpdated":
        return { tag: "projectionUpdatedPush", generation };
      case "ended":
        return { tag: "evidenceStreamEnded", generation };
      case "error":
        return { tag: "evidenceStreamFailed", generation, error: message.error };
    }
  });
}

/** Start or stop the open run's evidence follower to match its liveness,
 * mirroring `syncAutoRefresh`. Returns the stream effect when it needs
 * starting, else `null` (bumping the generation so a follower still in
 * flight is disowned). */
function syncEvidenceStream(draft: TunerState, env: TunerEnv): Effect<TunerAction> | null {
  const live = isOpenRunLive(draft);
  if (live && !draft.evidenceStreamActive && draft.openRunId) {
    draft.evidenceStreamActive = true;
    draft.evidenceStreamOk = true;
    draft.evidenceGeneration += 1;
    return evidenceStreamEffect(
      env,
      draft.openRunId,
      draft.evidence.seq,
      draft.evidenceGeneration,
    );
  }
  if (!live && draft.evidenceStreamActive) {
    draft.evidenceStreamActive = false;
    draft.evidenceGeneration += 1;
  }
  return null;
}

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

/** Projection-wide freshness: the headless follower's last pass, so a cold
 * open on an unattended run shows a real age instead of "not yet refreshed".
 * Fired alongside the initial load and after each `projectionLoaded`. */
function fetchProjectionMeta(env: TunerEnv): Effect<TunerAction> {
  return env
    .getProjectionMeta()
    .map((meta): TunerAction => ({ tag: "projectionMetaLoaded", meta }))
    .catch((): TunerAction => ({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } }));
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
    env
      .getProjectionProposals(runId)
      .map((proposals): TunerAction => ({ tag: "proposalsLoaded", generation, proposals }))
      .catch((e): TunerAction => ({ tag: "proposalsFailed", generation, error: String(e) })),
    env
      .getProjectionObservations(runId)
      .map((observations): TunerAction => ({ tag: "observationsLoaded", generation, observations }))
      .catch((e): TunerAction => ({ tag: "observationsFailed", generation, error: String(e) })),
    env
      .getProjectionShadowDecisions(runId)
      .map((rows): TunerAction => ({ tag: "shadowDecisionsLoaded", generation, rows }))
      .catch((e): TunerAction => ({ tag: "shadowDecisionsFailed", generation, error: String(e) })),
    env
      .getProjectionActiveEliminations(runId)
      .map((rows): TunerAction => ({ tag: "activeEliminationsLoaded", generation, rows }))
      .catch(
        (e): TunerAction => ({ tag: "activeEliminationsFailed", generation, error: String(e) }),
      ),
  );
}

function startResourceLoad(draft: TunerState, env: TunerEnv, runId: string): Effect<TunerAction> {
  draft.resourceGeneration += 1;
  draft.projectionDetail = toLoading(draft.projectionDetail);
  draft.validation = toLoading(draft.validation);
  draft.candidates = toLoading(draft.candidates);
  draft.pairs = toLoading(draft.pairs);
  draft.proposals = toLoading(draft.proposals);
  draft.observations = toLoading(draft.observations);
  draft.shadowDecisions = toLoading(draft.shadowDecisions);
  draft.activeEliminations = toLoading(draft.activeEliminations);
  draft.report = toLoading(draft.report);
  return fetchRunResources(env, runId, draft.resourceGeneration);
}

function clearResources(draft: TunerState): void {
  draft.resourceGeneration += 1;
  draft.projectionDetail = idle();
  draft.validation = idle();
  draft.candidates = idle();
  draft.pairs = idle();
  draft.proposals = idle();
  draft.observations = idle();
  draft.shadowDecisions = idle();
  draft.activeEliminations = idle();
  draft.report = idle();
  draft.scienceStale = false;
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
      const effects: Effect<TunerAction>[] = [];
      if (delay !== null) {
        effects.push(
          Effect.delay<TunerAction>(delay, {
            tag: "journalTick",
            generation: draft.journalGeneration,
          }),
        );
      }
      // A run just went terminal — pull a fresh projection so the completed
      // list gains its row without waiting for a manual refresh.
      if (after < before) {
        effects.push(Effect.send<TunerAction>({ tag: "refreshProjection" }));
      }
      // The open run's liveness may have changed with this poll — start or
      // stop its projection auto-refresh loop and evidence follower to match.
      const auto = syncAutoRefresh(draft);
      if (auto) effects.push(auto);
      const evidence = syncEvidenceStream(draft, env);
      if (evidence) effects.push(evidence);
      if (effects.length === 0) return null;
      return effects.length === 1 ? effects[0]! : Effect.merge(...effects);
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
      return fetchProjectionMeta(env);
    case "projectionFailed":
      draft.projectionRuns = toErr(action.error, draft.projectionRuns);
      return null;
    case "projectionMetaLoaded":
      draft.projectionLastPassAt = action.meta.last_pass_at;
      return null;

    case "projectionUpdatedPush": {
      // The follower committed a pass covering this run's newest evidence.
      // Pull the fresh rows straight down -- no `projection/refresh` POST
      // while the stream is healthy.
      if (action.generation !== draft.evidenceGeneration) return null;
      draft.scienceStale = false;
      draft.lastProjectionRefreshAt = Date.now();
      const list = fetchProjection(env);
      if (!draft.openRunId) return list;
      // Silent reload: refetch under a bumped generation without flipping the
      // per-run slots to `loading`, so the science on screen doesn't flash.
      draft.resourceGeneration += 1;
      return Effect.merge(
        list,
        fetchRunResources(env, draft.openRunId, draft.resourceGeneration),
      );
    }

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
      draft.scienceStale = false;
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

    case "projectionRefreshTick": {
      if (
        action.generation !== draft.projectionRefreshGeneration ||
        !draft.projectionRefreshActive
      ) {
        return null;
      }
      // The stream recovered: the follower + `projection-updated` frame have
      // this covered again, so wind the fallback loop down.
      if (!isOpenRunLive(draft) || draft.evidenceStreamOk) {
        draft.projectionRefreshActive = false;
        return null;
      }
      const delay =
        projectionRefreshDelayMs("live", draft.scienceStale, draft.evidenceStreamOk) ?? 6_000;
      return Effect.merge(
        Effect.send<TunerAction>({ tag: "autoRefreshProjection" }),
        Effect.delay<TunerAction>(delay, {
          tag: "projectionRefreshTick",
          generation: action.generation,
        }),
      );
    }
    case "autoRefreshProjection": {
      // A manual refresh in flight already covers this cadence.
      if (draft.refreshing || draft.autoRefreshing) return null;
      draft.autoRefreshing = true;
      return env
        .refreshProjection()
        .map((): TunerAction => ({ tag: "autoRefreshDone" }))
        // A periodic refresh that fails is non-critical: the launch-log tail
        // still shows progress and the next tick tries again. Don't raise
        // `refreshError` — that banner is for the manual button.
        .catch((): TunerAction => ({ tag: "autoRefreshDone" }));
    }
    case "autoRefreshDone": {
      draft.autoRefreshing = false;
      draft.scienceStale = false;
      draft.lastProjectionRefreshAt = Date.now();
      const list = fetchProjection(env);
      if (!draft.openRunId) return list;
      // Silent reload: bump the generation and refetch without flipping the
      // per-run slots to `loading`, so the science already on screen stays
      // put (no dim flash) while the fresher data loads underneath.
      draft.resourceGeneration += 1;
      return Effect.merge(
        list,
        fetchRunResources(env, draft.openRunId, draft.resourceGeneration),
      );
    }

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
      draft.evidence = { seq: 0, ring: [] };
      const effects = [
        Effect.send<TunerAction>({ tag: "logTick", generation: draft.logGeneration }),
        Effect.send<TunerAction>({ tag: "journalTick", generation: draft.journalGeneration }),
        startResourceLoad(draft, env, action.run.run_id),
      ];
      const evidence = syncEvidenceStream(draft, env);
      if (evidence) effects.push(evidence);
      return Effect.merge(...effects);
    }
    case "launchFailed":
      draft.launch = { status: "error", error: action.error, lastRunId: null };
      // A launch that failed fast still leaves a journalled, dead run — pull
      // the journal (and reproject) so the fleet shows it as "failed to
      // start" with its diagnostics rather than nothing at all.
      return Effect.merge(fetchJournal(env), Effect.send<TunerAction>({ tag: "refreshProjection" }));

    case "preflight": {
      draft.preflightGeneration += 1;
      draft.preflight = { status: "checking", errors: [], error: null };
      const generation = draft.preflightGeneration;
      return env
        .preflightRun(action.request)
        .map((result): TunerAction => ({ tag: "preflightChecked", generation, result }))
        .catch((e): TunerAction => ({ tag: "preflightErrored", generation, error: String(e) }));
    }
    case "preflightChecked":
      if (action.generation !== draft.preflightGeneration) return null;
      draft.preflight = action.result.ok
        ? { status: "ok", errors: [], error: null }
        : { status: "invalid", errors: action.result.errors, error: null };
      return null;
    case "preflightErrored":
      if (action.generation !== draft.preflightGeneration) return null;
      // The check itself failed to run — don't block the launch on it (the
      // server preflights again as a backstop), just surface the problem.
      draft.preflight = { status: "error", errors: [], error: action.error };
      return null;
    case "resetPreflight":
      draft.preflightGeneration += 1;
      draft.preflight = { status: "idle", errors: [], error: null };
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
      draft.evidence = { seq: 0, ring: [] };
      const effects = [logTick, startResourceLoad(draft, env, action.runId)];
      // If the newly opened run is already live, start its projection
      // auto-refresh loop and evidence follower now rather than waiting for
      // the next journal poll.
      const auto = syncAutoRefresh(draft);
      if (auto) effects.push(auto);
      const evidence = syncEvidenceStream(draft, env);
      if (evidence) effects.push(evidence);
      return Effect.merge(...effects);
    }
    case "closeRun":
      draft.openRunId = null;
      draft.log = { lines: [], errLines: [], offset: 0, error: null, active: false };
      clearResources(draft);
      draft.evidence = { seq: 0, ring: [] };
      // No open run — wind the auto-refresh loop and evidence follower down.
      syncAutoRefresh(draft);
      syncEvidenceStream(draft, env);
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
    case "proposalsLoaded":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.proposals = toOk(action.proposals, Date.now());
      return null;
    case "proposalsFailed":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.proposals = toErr(action.error, draft.proposals);
      return null;
    case "observationsLoaded":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.observations = toOk(action.observations, Date.now());
      return null;
    case "observationsFailed":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.observations = toErr(action.error, draft.observations);
      return null;
    case "shadowDecisionsLoaded":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.shadowDecisions = toOk(action.rows, Date.now());
      return null;
    case "shadowDecisionsFailed":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.shadowDecisions = toErr(action.error, draft.shadowDecisions);
      return null;
    case "activeEliminationsLoaded":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.activeEliminations = toOk(action.rows, Date.now());
      return null;
    case "activeEliminationsFailed":
      if (action.generation !== draft.resourceGeneration) return null;
      draft.activeEliminations = toErr(action.error, draft.activeEliminations);
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

    case "evidenceEvents": {
      if (action.generation !== draft.evidenceGeneration) return null;
      applyEvidence(draft, action.events, action.nextSeq ?? draft.evidence.seq);
      if (hasScientificEvent(action.events)) draft.scienceStale = true;
      return null;
    }
    case "evidenceStreamEnded": {
      if (action.generation !== draft.evidenceGeneration) return null;
      draft.evidenceStreamActive = false;
      return null;
    }
    case "evidenceStreamFailed": {
      if (action.generation !== draft.evidenceGeneration) return null;
      draft.evidenceStreamOk = false;
      // The push channel is gone; fall back to polling the tail *and* to the
      // client-driven projection refresh loop while the run is still live.
      if (!isOpenRunLive(draft)) {
        draft.evidenceStreamActive = false;
        return null;
      }
      const effects: Effect<TunerAction>[] = [];
      const auto = syncAutoRefresh(draft);
      if (auto) effects.push(auto);
      const delay = evidencePollDelayMs(false, "live");
      if (delay !== null) {
        effects.push(
          Effect.delay<TunerAction>(delay, {
            tag: "evidencePollTick",
            generation: action.generation,
          }),
        );
      }
      return effects.length === 0
        ? null
        : effects.length === 1
          ? effects[0]!
          : Effect.merge(...effects);
    }
    case "evidencePollTick": {
      if (action.generation !== draft.evidenceGeneration || !draft.evidenceStreamActive) {
        return null;
      }
      if (!isOpenRunLive(draft) || !draft.openRunId) {
        draft.evidenceStreamActive = false;
        return null;
      }
      const runId = draft.openRunId;
      const sinceSeq = draft.evidence.seq;
      const generation = action.generation;
      return env
        .getRunEvidence(runId, sinceSeq)
        .map((response): TunerAction => ({ tag: "evidencePolled", generation, response }))
        .catch(
          (): TunerAction => ({
            tag: "evidencePolled",
            generation,
            response: { events: [], next_seq: sinceSeq, run_status: "unknown" },
          }),
        );
    }
    case "evidencePolled": {
      if (action.generation !== draft.evidenceGeneration) return null;
      applyEvidence(draft, action.response.events, action.response.next_seq);
      if (hasScientificEvent(action.response.events)) draft.scienceStale = true;
      if (!isOpenRunLive(draft)) {
        draft.evidenceStreamActive = false;
        return null;
      }
      const delay = evidencePollDelayMs(false, "live");
      return delay === null
        ? null
        : Effect.delay<TunerAction>(delay, {
            tag: "evidencePollTick",
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
