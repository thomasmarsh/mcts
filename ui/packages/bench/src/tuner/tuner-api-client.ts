// tuner-api-client.ts — the only file in the tuner UI allowed to call
// `fetch` (eslint.config.js's fetch ban whitelists `**/tuner-api-client.ts`
// alongside `api-client.ts`). One method per version-4 tuner route; typed
// request/response from `tuner-types.ts`. `tuner-env.ts` lifts each method
// to an `Effect`; the reducer and components only ever touch the env.

import type {
  EvidenceEnvelope,
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
  ProjectionRefreshResult,
  ProjectionRunDetail,
  ProjectionMeta,
  ProjectionRunListItem,
  LaunchPreflightResult,
  RunPlan,
  ProjectionValidation,
  ObjectiveValidationResult,
  TunerBudgetExtension,
  TunerLaunchRequest,
  TunerObjectiveDetail,
  TunerObjectiveFile,
  TunerRunLog,
  TunerRunView,
} from "./tuner-types.js";

/** Callbacks the evidence SSE subscription drives; `onEnd` / `onError` fire
 * at most once and are terminal. */
export interface EvidenceStreamHandlers {
  onEvents(events: EvidenceEnvelope[]): void;
  /** The server's headless follower reprojected this run's newest evidence.
   * The UI should re-fetch its projection slices — it must not POST a
   * refresh of its own while the stream is healthy. */
  onProjectionUpdated(): void;
  onEnd(): void;
  onError(message: string): void;
}

export interface EvidenceSubscription {
  close(): void;
}
import type { JsonValue, TunerGameInfo } from "../types.js";

export interface TunerApiClient {
  // Launch metadata.
  listKinds(): Promise<TunerGameInfo[]>;
  listObjectives(): Promise<TunerObjectiveFile[]>;
  getObjective(key: string): Promise<TunerObjectiveDetail>;
  putObjective(key: string, content: JsonValue): Promise<TunerObjectiveDetail>;
  deleteObjective(key: string): Promise<void>;
  validateObjective(key: string, content: JsonValue): Promise<ObjectiveValidationResult>;
  // Operational journal.
  listRuns(): Promise<TunerRunView[]>;
  getRun(runId: string): Promise<TunerRunView>;
  launchRun(body: TunerLaunchRequest): Promise<TunerRunView>;
  preflightRun(body: TunerLaunchRequest): Promise<LaunchPreflightResult>;
  planRun(body: TunerLaunchRequest): Promise<RunPlan>;
  stopRun(runId: string): Promise<TunerRunView>;
  extendRun(runId: string, body: TunerBudgetExtension): Promise<TunerRunView>;
  /** Permanently remove a terminal run. `409` if the run is still live. */
  deleteRun(runId: string): Promise<void>;
  getRunLog(runId: string, since?: number): Promise<TunerRunLog>;
  getRunEvidence(runId: string, sinceSeq: number): Promise<EvidenceTailResponse>;
  /** Subscribe to the run's evidence SSE stream from `sinceSeq`. Returns a
   * handle whose `close()` tears down the underlying `EventSource`. */
  openEvidenceStream(
    runId: string,
    sinceSeq: number,
    handlers: EvidenceStreamHandlers,
  ): EvidenceSubscription;
  // Projection.
  refreshProjection(): Promise<ProjectionRefreshResult>;
  getProjectionMeta(): Promise<ProjectionMeta>;
  listProjectionRuns(): Promise<ProjectionRunListItem[]>;
  getProjectionRun(runId: string): Promise<ProjectionRunDetail>;
  getProjectionCohorts(runId: string): Promise<ProjectionCohort[]>;
  getProjectionCandidates(runId: string): Promise<ProjectionCandidate[]>;
  getProjectionCandidate(runId: string, candidateId: string): Promise<ProjectionCandidate>;
  getProjectionPairs(runId: string, query?: ProjectionPairQuery): Promise<ProjectionPairRow[]>;
  getProjectionPairGames(runId: string, pairId: string): Promise<ProjectionGameRow[]>;
  getProjectionValidation(runId: string): Promise<ProjectionValidation>;
  getProjectionReport(runId: string): Promise<JsonValue>;
  // Live science row tables — populated on every projection refresh, partial
  // or complete, so they carry the run's science before `report.json` exists.
  getProjectionProposals(runId: string): Promise<ProjectionProposal[]>;
  getProjectionObservations(runId: string): Promise<ProjectionObservation[]>;
  getProjectionShadowDecisions(runId: string): Promise<ProjectionShadowDecision[]>;
  getProjectionActiveEliminations(runId: string): Promise<ProjectionActiveElimination[]>;
}

class TunerApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "TunerApiError";
  }
}

async function errorMessage(r: Response): Promise<string> {
  const text = await r.text().catch(() => "");
  if (text) {
    try {
      const body: unknown = JSON.parse(text);
      if (
        body &&
        typeof body === "object" &&
        typeof (body as { error?: unknown }).error === "string"
      ) {
        return (body as { error: string }).error;
      }
    } catch {
      // Not JSON -- fall through to raw text.
    }
  }
  return text || `API ${r.status}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new TunerApiError(await errorMessage(r), r.status);
  return r.json() as Promise<T>;
}

async function sendJson<T>(
  url: string,
  method: "POST" | "PUT",
  body?: unknown,
): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new TunerApiError(await errorMessage(r), r.status);
  return r.json() as Promise<T>;
}

async function sendNoContent(url: string, method: "DELETE"): Promise<void> {
  const r = await fetch(url, { method });
  if (!r.ok) throw new TunerApiError(await errorMessage(r), r.status);
}

function queryString(params: object): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (typeof v === "string" || typeof v === "number") q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

/** `baseUrl` defaults to `""` (relative URLs), same convention as
 * `createBenchApiClient`. */
export function createTunerApiClient(baseUrl = ""): TunerApiClient {
  const url = (path: string): string => baseUrl + path;
  const runPath = (runId: string): string =>
    `/api/bench/tuner/runs/${encodeURIComponent(runId)}`;
  const projPath = (runId: string): string =>
    `/api/bench/tuner/projection/runs/${encodeURIComponent(runId)}`;
  const objectivePath = (key: string): string =>
    `/api/bench/tuner/objectives/${encodeURIComponent(key)}`;
  return {
    listKinds: () => fetchJson(url("/api/bench/tuner/kinds")),
    listObjectives: () => fetchJson(url("/api/bench/tuner/objectives")),
    getObjective: (key) => fetchJson(url(objectivePath(key))),
    putObjective: (key, content) => sendJson(url(objectivePath(key)), "PUT", content),
    deleteObjective: (key) => sendNoContent(url(objectivePath(key)), "DELETE"),
    validateObjective: (key, content) =>
      sendJson(url(`${objectivePath(key)}/validate`), "POST", content),
    listRuns: () => fetchJson(url("/api/bench/tuner/runs")),
    getRun: (runId) => fetchJson(url(runPath(runId))),
    launchRun: (body) => sendJson(url("/api/bench/tuner/runs"), "POST", body),
    preflightRun: (body) => sendJson(url("/api/bench/tuner/runs/preflight"), "POST", body),
    planRun: (body) => sendJson(url("/api/bench/tuner/runs/plan"), "POST", body),
    stopRun: (runId) => sendJson(url(`${runPath(runId)}/stop`), "POST"),
    extendRun: (runId, body) => sendJson(url(`${runPath(runId)}/extend`), "POST", body),
    deleteRun: (runId) => sendNoContent(url(runPath(runId)), "DELETE"),
    getRunLog: (runId, since) =>
      fetchJson(url(`${runPath(runId)}/log${queryString({ since })}`)),
    getRunEvidence: (runId, sinceSeq) =>
      fetchJson(url(`${runPath(runId)}/evidence${queryString({ since_seq: sinceSeq })}`)),
    openEvidenceStream: (runId, sinceSeq, handlers) => {
      const source = new EventSource(
        url(`${runPath(runId)}/evidence/stream${queryString({ since_seq: sinceSeq })}`),
      );
      let closed = false;
      const close = (): void => {
        if (!closed) {
          closed = true;
          source.close();
        }
      };
      source.onmessage = (event: MessageEvent<string>) => {
        try {
          const envelope = JSON.parse(event.data) as EvidenceEnvelope;
          handlers.onEvents([envelope]);
        } catch (error: unknown) {
          handlers.onError(`invalid evidence event: ${String(error)}`);
        }
      };
      source.addEventListener("projection-updated", () => {
        handlers.onProjectionUpdated();
      });
      // The server names its final frame `event: end`.
      source.addEventListener("end", () => {
        close();
        handlers.onEnd();
      });
      source.onerror = () => {
        // EventSource reconnects on a transient drop; a hard failure (run
        // gone, 4xx) leaves it CLOSED — only then surface the error.
        if (source.readyState === EventSource.CLOSED) {
          close();
          handlers.onError("evidence stream connection lost");
        }
      };
      return { close };
    },
    refreshProjection: () =>
      sendJson(url("/api/bench/tuner/projection/refresh"), "POST"),
    getProjectionMeta: () => fetchJson(url("/api/bench/tuner/projection/meta")),
    listProjectionRuns: () => fetchJson(url("/api/bench/tuner/projection/runs")),
    getProjectionRun: (runId) => fetchJson(url(projPath(runId))),
    getProjectionCohorts: (runId) => fetchJson(url(`${projPath(runId)}/cohorts`)),
    getProjectionCandidates: (runId) => fetchJson(url(`${projPath(runId)}/candidates`)),
    getProjectionCandidate: (runId, candidateId) =>
      fetchJson(url(`${projPath(runId)}/candidates/${encodeURIComponent(candidateId)}`)),
    getProjectionPairs: (runId, query = {}) =>
      fetchJson(url(`${projPath(runId)}/pairs${queryString(query)}`)),
    getProjectionPairGames: (runId, pairId) =>
      fetchJson(url(`${projPath(runId)}/pairs/${encodeURIComponent(pairId)}/games`)),
    getProjectionValidation: (runId) => fetchJson(url(`${projPath(runId)}/validation`)),
    getProjectionReport: (runId) => fetchJson(url(`${projPath(runId)}/report`)),
    getProjectionProposals: (runId) => fetchJson(url(`${projPath(runId)}/proposals`)),
    getProjectionObservations: (runId) => fetchJson(url(`${projPath(runId)}/observations`)),
    getProjectionShadowDecisions: (runId) =>
      fetchJson(url(`${projPath(runId)}/shadow-decisions`)),
    getProjectionActiveEliminations: (runId) =>
      fetchJson(url(`${projPath(runId)}/active-eliminations`)),
  };
}
