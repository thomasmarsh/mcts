// tuner-env.ts — lifts every `TunerApiClient` method to an `Effect`, giving
// the `TunerEnv` the tuner reducer receives. Fully mockable: a test builds a
// `TunerEnv` whose methods return `Effect.send(...)` and never touches the
// network (AGENTS.md "mock the environment").

import { Effect } from "@mcts/core";
import type { TunerApiClient } from "./tuner-api-client.js";
import type {
  ProjectionCandidate,
  ProjectionCohort,
  ProjectionPairQuery,
  ProjectionPairRow,
  ProjectionRefreshResult,
  ProjectionRunDetail,
  ProjectionRunListItem,
  ProjectionValidation,
  TunerBudgetExtension,
  TunerLaunchRequest,
  TunerObjectiveFile,
  TunerRunLog,
  TunerRunView,
} from "./tuner-types.js";
import type { JsonValue, TunerGameInfo } from "../types.js";

export interface TunerEnv {
  listKinds(): Effect<TunerGameInfo[]>;
  listObjectives(): Effect<TunerObjectiveFile[]>;
  listRuns(): Effect<TunerRunView[]>;
  getRun(runId: string): Effect<TunerRunView>;
  launchRun(body: TunerLaunchRequest): Effect<TunerRunView>;
  stopRun(runId: string): Effect<TunerRunView>;
  extendRun(runId: string, body: TunerBudgetExtension): Effect<TunerRunView>;
  getRunLog(runId: string, since: number): Effect<TunerRunLog>;
  refreshProjection(): Effect<ProjectionRefreshResult>;
  listProjectionRuns(): Effect<ProjectionRunListItem[]>;
  getProjectionRun(runId: string): Effect<ProjectionRunDetail>;
  getProjectionCohorts(runId: string): Effect<ProjectionCohort[]>;
  getProjectionCandidates(runId: string): Effect<ProjectionCandidate[]>;
  getProjectionCandidate(runId: string, candidateId: string): Effect<ProjectionCandidate>;
  getProjectionPairs(runId: string, query?: ProjectionPairQuery): Effect<ProjectionPairRow[]>;
  getProjectionValidation(runId: string): Effect<ProjectionValidation>;
  getProjectionReport(runId: string): Effect<JsonValue>;
}

export function createTunerEnv(api: TunerApiClient): TunerEnv {
  const lift = <T>(thunk: () => Promise<T>): Effect<T> => Effect.fromPromise(thunk);
  return {
    listKinds: () => lift(() => api.listKinds()),
    listObjectives: () => lift(() => api.listObjectives()),
    listRuns: () => lift(() => api.listRuns()),
    getRun: (runId) => lift(() => api.getRun(runId)),
    launchRun: (body) => lift(() => api.launchRun(body)),
    stopRun: (runId) => lift(() => api.stopRun(runId)),
    extendRun: (runId, body) => lift(() => api.extendRun(runId, body)),
    getRunLog: (runId, since) => lift(() => api.getRunLog(runId, since)),
    refreshProjection: () => lift(() => api.refreshProjection()),
    listProjectionRuns: () => lift(() => api.listProjectionRuns()),
    getProjectionRun: (runId) => lift(() => api.getProjectionRun(runId)),
    getProjectionCohorts: (runId) => lift(() => api.getProjectionCohorts(runId)),
    getProjectionCandidates: (runId) => lift(() => api.getProjectionCandidates(runId)),
    getProjectionCandidate: (runId, candidateId) =>
      lift(() => api.getProjectionCandidate(runId, candidateId)),
    getProjectionPairs: (runId, query) => lift(() => api.getProjectionPairs(runId, query)),
    getProjectionValidation: (runId) => lift(() => api.getProjectionValidation(runId)),
    getProjectionReport: (runId) => lift(() => api.getProjectionReport(runId)),
  };
}
