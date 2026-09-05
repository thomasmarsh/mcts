// tuner-env.ts — lifts every `TunerApiClient` method to an `Effect`, giving
// the `TunerEnv` the tuner reducer receives. Fully mockable: a test builds a
// `TunerEnv` whose methods return `Effect.send(...)` and never touches the
// network (AGENTS.md "mock the environment").

import { Effect } from "@mcts/core";
import type { TunerApiClient } from "./tuner-api-client.js";
import type {
  EvidenceStreamMessage,
  EvidenceTailResponse,
  ProjectionActiveElimination,
  ProjectionCandidate,
  ProjectionCohort,
  ProjectionObservation,
  ProjectionProposal,
  ProjectionShadowDecision,
  ProjectionGameRow,
  ProjectionPairQuery,
  ProjectionPairRow,
  ProjectionRunPageQuery,
  ProjectionMeta,
  ProjectionRefreshResult,
  ProjectionRunDetail,
  ProjectionRunListItem,
  ProjectionValidation,
  ObjectiveValidationResult,
  TunerBudgetExtension,
  TunerLaunchRequest,
  LaunchPreflightResult,
  RunPlan,
  TunerObjectiveDetail,
  TunerObjectiveFile,
  TunerProfileDetail,
  TunerProfileFile,
  TunerRunLog,
  TunerRunView,
} from "./tuner-types.js";
import type { JsonValue, TunableGame } from "../types.js";

export interface TunerEnv {
  listTunableGames(): Effect<TunableGame[]>;
  listObjectives(): Effect<TunerObjectiveFile[]>;
  getObjective(key: string): Effect<TunerObjectiveDetail>;
  putObjective(key: string, content: JsonValue): Effect<TunerObjectiveDetail>;
  deleteObjective(key: string): Effect<void>;
  validateObjective(key: string, content: JsonValue): Effect<ObjectiveValidationResult>;
  listProfiles(): Effect<TunerProfileFile[]>;
  getProfile(key: string): Effect<TunerProfileDetail>;
  putProfile(key: string, content: JsonValue): Effect<TunerProfileDetail>;
  deleteProfile(key: string): Effect<void>;
  validateProfile(key: string, content: JsonValue): Effect<LaunchPreflightResult>;
  listRuns(): Effect<TunerRunView[]>;
  getRun(runId: string): Effect<TunerRunView>;
  launchRun(body: TunerLaunchRequest): Effect<TunerRunView>;
  preflightRun(body: TunerLaunchRequest): Effect<LaunchPreflightResult>;
  planRun(body: TunerLaunchRequest): Effect<RunPlan>;
  stopRun(runId: string): Effect<TunerRunView>;
  extendRun(runId: string, body: TunerBudgetExtension): Effect<TunerRunView>;
  resumeRun(runId: string): Effect<TunerRunView>;
  deleteRun(runId: string): Effect<void>;
  getRunLog(runId: string, since: number): Effect<TunerRunLog>;
  getRunEvidence(runId: string, sinceSeq: number): Effect<EvidenceTailResponse>;
  /** A long-lived effect: pushes `{kind:"events"}` messages as the run's
   * evidence stream appends, then exactly one terminal `{kind:"ended"}` or
   * `{kind:"error"}` and resolves. Opening a new stream closes the previous
   * one (the UI only ever follows one run at a time). */
  openEvidenceStream(runId: string, sinceSeq: number): Effect<EvidenceStreamMessage>;
  /** Close whatever evidence stream is currently open, without opening a
   * replacement. `openEvidenceStream` already closes the previous stream
   * when a *new* one starts (switching between two live runs is
   * self-cleaning), but nothing previously closed the connection when the
   * reducer simply stops following one -- the run went non-live, or its
   * overview was navigated away from -- leaving it open, still receiving
   * and parsing every event, for as long as the tab lives. The reducer
   * calls this the moment it decides a stream should stop. */
  closeEvidenceStream(): void;
  refreshProjection(): Effect<ProjectionRefreshResult>;
  getProjectionMeta(): Effect<ProjectionMeta>;
  listProjectionRuns(): Effect<ProjectionRunListItem[]>;
  getProjectionRun(runId: string): Effect<ProjectionRunDetail>;
  getProjectionCohorts(runId: string): Effect<ProjectionCohort[]>;
  getProjectionCandidates(runId: string): Effect<ProjectionCandidate[]>;
  getProjectionCandidate(runId: string, candidateId: string): Effect<ProjectionCandidate>;
  getProjectionPairs(runId: string, query?: ProjectionPairQuery): Effect<ProjectionPairRow[]>;
  getProjectionPairGames(runId: string, pairId: string): Effect<ProjectionGameRow[]>;
  getProjectionValidation(runId: string): Effect<ProjectionValidation>;
  getProjectionReport(runId: string): Effect<JsonValue>;
  getProjectionProposals(
    runId: string,
    query?: ProjectionRunPageQuery,
  ): Effect<ProjectionProposal[]>;
  getProjectionObservations(
    runId: string,
    query?: ProjectionRunPageQuery,
  ): Effect<ProjectionObservation[]>;
  getProjectionShadowDecisions(
    runId: string,
    query?: ProjectionRunPageQuery,
  ): Effect<ProjectionShadowDecision[]>;
  getProjectionActiveEliminations(
    runId: string,
    query?: ProjectionRunPageQuery,
  ): Effect<ProjectionActiveElimination[]>;
}

export function createTunerEnv(api: TunerApiClient): TunerEnv {
  const lift = <T>(thunk: () => Promise<T>): Effect<T> => Effect.fromPromise(thunk);
  // Only one evidence stream is ever open: opening the next closes this.
  let activeStream: { close(): void } | null = null;
  return {
    listTunableGames: () => lift(() => api.listTunableGames()),
    listObjectives: () => lift(() => api.listObjectives()),
    getObjective: (key) => lift(() => api.getObjective(key)),
    putObjective: (key, content) => lift(() => api.putObjective(key, content)),
    deleteObjective: (key) => lift(() => api.deleteObjective(key)),
    validateObjective: (key, content) => lift(() => api.validateObjective(key, content)),
    listProfiles: () => lift(() => api.listProfiles()),
    getProfile: (key) => lift(() => api.getProfile(key)),
    putProfile: (key, content) => lift(() => api.putProfile(key, content)),
    deleteProfile: (key) => lift(() => api.deleteProfile(key)),
    validateProfile: (key, content) => lift(() => api.validateProfile(key, content)),
    listRuns: () => lift(() => api.listRuns()),
    getRun: (runId) => lift(() => api.getRun(runId)),
    launchRun: (body) => lift(() => api.launchRun(body)),
    preflightRun: (body) => lift(() => api.preflightRun(body)),
    planRun: (body) => lift(() => api.planRun(body)),
    stopRun: (runId) => lift(() => api.stopRun(runId)),
    extendRun: (runId, body) => lift(() => api.extendRun(runId, body)),
    resumeRun: (runId) => lift(() => api.resumeRun(runId)),
    deleteRun: (runId) => lift(() => api.deleteRun(runId)),
    getRunLog: (runId, since) => lift(() => api.getRunLog(runId, since)),
    getRunEvidence: (runId, sinceSeq) => lift(() => api.getRunEvidence(runId, sinceSeq)),
    openEvidenceStream: (runId, sinceSeq) =>
      Effect.stream<EvidenceStreamMessage>((send, done) => {
        activeStream?.close();
        activeStream = api.openEvidenceStream(runId, sinceSeq, {
          onEvents: (events) => send({ kind: "events", events }),
          onProjectionUpdated: () => send({ kind: "projectionUpdated" }),
          onEnd: () => {
            activeStream = null;
            send({ kind: "ended" });
            done();
          },
          onError: (error) => {
            activeStream = null;
            send({ kind: "error", error });
            done();
          },
        });
      }),
    closeEvidenceStream: () => {
      activeStream?.close();
      activeStream = null;
    },
    refreshProjection: () => lift(() => api.refreshProjection()),
    getProjectionMeta: () => lift(() => api.getProjectionMeta()),
    listProjectionRuns: () => lift(() => api.listProjectionRuns()),
    getProjectionRun: (runId) => lift(() => api.getProjectionRun(runId)),
    getProjectionCohorts: (runId) => lift(() => api.getProjectionCohorts(runId)),
    getProjectionCandidates: (runId) => lift(() => api.getProjectionCandidates(runId)),
    getProjectionCandidate: (runId, candidateId) =>
      lift(() => api.getProjectionCandidate(runId, candidateId)),
    getProjectionPairs: (runId, query) => lift(() => api.getProjectionPairs(runId, query)),
    getProjectionPairGames: (runId, pairId) =>
      lift(() => api.getProjectionPairGames(runId, pairId)),
    getProjectionValidation: (runId) => lift(() => api.getProjectionValidation(runId)),
    getProjectionReport: (runId) => lift(() => api.getProjectionReport(runId)),
    getProjectionProposals: (runId, query) =>
      lift(() => api.getProjectionProposals(runId, query)),
    getProjectionObservations: (runId, query) =>
      lift(() => api.getProjectionObservations(runId, query)),
    getProjectionShadowDecisions: (runId, query) =>
      lift(() => api.getProjectionShadowDecisions(runId, query)),
    getProjectionActiveEliminations: (runId, query) =>
      lift(() => api.getProjectionActiveEliminations(runId, query)),
  };
}
