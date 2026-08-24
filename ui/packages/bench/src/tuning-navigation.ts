// Tuning-session navigation and analysis workspace state.

import { Effect } from "@mcts/core";
import type { BenchEnv } from "./reducer.js";
import type {
  TuningAnalysisOverview,
  TuningSessionCommandKind,
  TuningSessionCommandResponse,
  TuningSessionControl,
  TuningSessionDetail,
  TuningSessionsResponse,
  TuningTrialDetail,
  TuningTrialPage,
  TuningTrialPageQuery,
} from "./types.js";

export const TUNING_DETAIL_REFRESH_MS = 5_000;
export const DEFAULT_TRIAL_PAGE_LIMIT = 50;
export const MAX_TRIAL_PAGE_LIMIT = 200;

export interface TuningLoadState<T> {
  status: "idle" | "loading" | "done" | "error";
  snapshot: T | null;
  error: string | null;
  generation: number;
}
export interface TuningSelection {
  sessionId: string | null; attemptId: string | null; trialId: string | null; pairId: string | null; gameId: string | null;
}
export interface TuningTrialFilters {
  state: string | null; bracket: string | null; reason: string | null; family: string | null; q: string | null;
}
export interface TuningTrialSort {
  sort: NonNullable<TuningTrialPageQuery["sort"]>;
  direction: NonNullable<TuningTrialPageQuery["direction"]>;
}
export type TuningProgressMetric = "score" | "mu" | "sigma";
export type TuningProgressScale = "shared" | "local";
export interface TuningTrialPageState extends TuningLoadState<TuningTrialPage> {
  sessionId: string | null; queryKey: string | null; cursor: string | null; previousCursors: (string | null)[];
}
export interface TuningTrialDetailState extends TuningLoadState<TuningTrialDetail> {
  sessionId: string; trialId: string;
}
export interface TuningSessionCommandState {
  kind: TuningSessionCommandKind;
  commandId: string;
  expectedVersion: number;
  delta?: number;
  start?: boolean;
  nWorkers?: number;
  status: "pending" | "succeeded" | "failed";
  error: string | null;
  retriable: boolean;
  response: TuningSessionCommandResponse | null;
}
export interface TuningNavigationState {
  list: TuningLoadState<TuningSessionsResponse>;
  /** Retained for the existing game-evidence workbench until it moves to the lazy analysis routes. */
  detail: TuningLoadState<TuningSessionDetail> & { sessionId: string | null };
  overview: TuningLoadState<TuningAnalysisOverview> & { sessionId: string | null };
  trialPage: TuningTrialPageState;
  trialDetails: Record<string, TuningTrialDetailState>;
  tab: "progress" | "pruning" | "ladder" | "trials" | "game";
  progressMetric: TuningProgressMetric;
  progressScale: TuningProgressScale;
  filters: TuningTrialFilters;
  sort: TuningTrialSort;
  trialPageLimit: number;
  /** Immutable-pool selectors are user-owned and never follow refreshes. */
  ladderRevision: number | null;
  ladderAnchorKey: string | null;
  selection: TuningSelection;
  expandedIds: string[];
  unavailable: string | null;
  /** The last command for each session, retained so a transport retry replays its id. */
  commands: Record<string, TuningSessionCommandState>;
}

export type TuningNavigationAction =
  | { tag: "listRequest" }
  | { tag: "listLoaded"; generation: number; response: TuningSessionsResponse }
  | { tag: "listFailed"; generation: number; error: string }
  | { tag: "detailRequest"; sessionId: string }
  | { tag: "detailLoaded"; generation: number; sessionId: string; detail: TuningSessionDetail }
  | { tag: "detailFailed"; generation: number; sessionId: string; error: string }
  | { tag: "overviewRequest"; sessionId: string }
  | { tag: "overviewRefreshTick"; sessionId: string; generation: number }
  | { tag: "overviewLoaded"; generation: number; sessionId: string; overview: TuningAnalysisOverview }
  | { tag: "overviewFailed"; generation: number; sessionId: string; error: string }
  | { tag: "trialPageRequest"; sessionId: string }
  | { tag: "trialPageLoaded"; generation: number; sessionId: string; queryKey: string; page: TuningTrialPage }
  | { tag: "trialPageFailed"; generation: number; sessionId: string; queryKey: string; error: string }
  | { tag: "trialDetailRequest"; sessionId: string; trialId: string }
  | { tag: "trialDetailLoaded"; generation: number; sessionId: string; trialId: string; detail: TuningTrialDetail }
  | { tag: "trialDetailFailed"; generation: number; sessionId: string; trialId: string; error: string }
  | { tag: "selectSession"; sessionId: string }
  | { tag: "clearSession" }
  | { tag: "setAnalysisTab"; tab: "progress" | "pruning" | "ladder" | "trials" | "game" }
  | { tag: "setProgressMetric"; metric: TuningProgressMetric }
  | { tag: "setProgressScale"; scale: TuningProgressScale }
  | { tag: "setTrialFilters"; filters: Partial<TuningTrialFilters> }
  | { tag: "setTrialSort"; sort: TuningTrialSort }
  | { tag: "setTrialPageLimit"; limit: number }
  | { tag: "setLadderRevision"; revision: number | null }
  | { tag: "selectLadderAnchor"; anchorKey: string | null }
  | { tag: "nextTrialPage" }
  | { tag: "previousTrialPage" }
  | { tag: "selectAttempt"; attemptId: string }
  | { tag: "selectTrial"; trialId: string }
  | { tag: "selectPair"; pairId: string }
  | { tag: "selectGame"; gameId: string }
  | { tag: "toggleExpanded"; id: string }
  | { tag: "sessionCommandSubmit"; sessionId: string; kind: TuningSessionCommandKind; commandId: string; expectedVersion: number; delta?: number; start?: boolean; nWorkers?: number }
  | { tag: "sessionCommandRetry"; sessionId: string }
  | { tag: "sessionCommandSucceeded"; sessionId: string; commandId: string; response: TuningSessionCommandResponse }
  | { tag: "sessionCommandFailed"; sessionId: string; commandId: string; error: string; retriable: boolean }
  | { tag: "openCommandAttempt"; sessionId: string; attemptId: string };

function loadState<T>(): TuningLoadState<T> { return { status: "idle", snapshot: null, error: null, generation: 0 }; }
function defaultFilters(): TuningTrialFilters { return { state: null, bracket: null, reason: null, family: null, q: null }; }
function pageState(): TuningTrialPageState {
  return { ...loadState<TuningTrialPage>(), sessionId: null, queryKey: null, cursor: null, previousCursors: [] };
}
export function initialTuningNavigationState(): TuningNavigationState {
  return {
    list: loadState<TuningSessionsResponse>(), detail: { ...loadState<TuningSessionDetail>(), sessionId: null },
    overview: { ...loadState<TuningAnalysisOverview>(), sessionId: null }, trialPage: pageState(), trialDetails: {},
    tab: "progress", progressMetric: "score", progressScale: "shared",
    filters: defaultFilters(), sort: { sort: "trial", direction: "desc" }, trialPageLimit: DEFAULT_TRIAL_PAGE_LIMIT,
    ladderRevision: null, ladderAnchorKey: null,
    selection: { sessionId: null, attemptId: null, trialId: null, pairId: null, gameId: null }, expandedIds: [], unavailable: null, commands: {},
  };
}
function selectionForSession(sessionId: string | null): TuningSelection {
  return { sessionId, attemptId: null, trialId: null, pairId: null, gameId: null };
}
function sessionCanAnalyze(state: TuningNavigationState, sessionId: string): boolean {
  return state.list.snapshot?.sessions.find((session) => session.session_id === sessionId)?.capabilities.has_lifecycle ?? false;
}
function sessionIsActive(state: TuningNavigationState, sessionId: string): boolean {
  const status = state.list.snapshot?.sessions.find((session) => session.session_id === sessionId)?.status;
  return status === "active";
}
function pageQuery(state: TuningNavigationState): TuningTrialPageQuery {
  const { filters, sort, trialPage } = state;
  return { ...filters, sort: sort.sort, direction: sort.direction, limit: state.trialPageLimit, cursor: trialPage.cursor };
}
function clearDetail(state: TuningNavigationState): void {
  state.detail = { ...loadState<TuningSessionDetail>(), generation: state.detail.generation + 1, sessionId: null };
}
function clearAnalysis(state: TuningNavigationState): void {
  state.overview = { ...loadState<TuningAnalysisOverview>(), generation: state.overview.generation + 1, sessionId: null };
  state.trialPage = { ...pageState(), generation: state.trialPage.generation + 1 };
  state.trialDetails = {};
}
function requestList(state: TuningNavigationState, env: BenchEnv): Effect<TuningNavigationAction> {
  const generation = state.list.generation + 1;
  state.list = { ...state.list, status: "loading", error: null, generation };
  return env.listTuningSessions().map((response): TuningNavigationAction => ({ tag: "listLoaded", generation, response }))
    .catch((error): TuningNavigationAction => ({ tag: "listFailed", generation, error: String(error) }));
}
function requestDetail(state: TuningNavigationState, sessionId: string, env: BenchEnv): Effect<TuningNavigationAction> {
  const generation = state.detail.generation + 1;
  const snapshot = state.detail.sessionId === sessionId ? state.detail.snapshot : null;
  state.detail = { status: "loading", snapshot, error: null, generation, sessionId };
  return env.getTuningSession(sessionId).map((detail): TuningNavigationAction => ({ tag: "detailLoaded", generation, sessionId, detail }))
    .catch((error): TuningNavigationAction => ({ tag: "detailFailed", generation, sessionId, error: String(error) }));
}
function requestOverview(state: TuningNavigationState, sessionId: string, env: BenchEnv): Effect<TuningNavigationAction> {
  const generation = state.overview.generation + 1;
  const snapshot = state.overview.sessionId === sessionId ? state.overview.snapshot : null;
  state.overview = { status: "loading", snapshot, error: null, generation, sessionId };
  return env.getTuningAnalysisOverview(sessionId).map((overview): TuningNavigationAction => ({ tag: "overviewLoaded", generation, sessionId, overview }))
    .catch((error): TuningNavigationAction => ({ tag: "overviewFailed", generation, sessionId, error: String(error) }));
}
function requestTrialPage(state: TuningNavigationState, sessionId: string, env: BenchEnv): Effect<TuningNavigationAction> {
  const query = pageQuery(state);
  const queryKey = JSON.stringify(query);
  const generation = state.trialPage.generation + 1;
  const snapshot = state.trialPage.sessionId === sessionId && state.trialPage.queryKey === queryKey ? state.trialPage.snapshot : null;
  state.trialPage = { ...state.trialPage, status: "loading", snapshot, error: null, generation, sessionId, queryKey };
  return env.getTuningTrialPage(sessionId, query).map((page): TuningNavigationAction => ({ tag: "trialPageLoaded", generation, sessionId, queryKey, page }))
    .catch((error): TuningNavigationAction => ({ tag: "trialPageFailed", generation, sessionId, queryKey, error: String(error) }));
}
function requestTrialDetail(state: TuningNavigationState, sessionId: string, trialId: string, env: BenchEnv): Effect<TuningNavigationAction> {
  const existing = state.trialDetails[trialId];
  if (existing?.sessionId === sessionId && (existing.status === "loading" || (existing.status === "done" && existing.snapshot !== null))) return Effect.none();
  const generation = (existing?.generation ?? 0) + 1;
  const snapshot = existing?.sessionId === sessionId ? existing.snapshot : null;
  state.trialDetails[trialId] = { status: "loading", snapshot, error: null, generation, sessionId, trialId };
  return env.getTuningTrialDetail(sessionId, trialId).map((detail): TuningNavigationAction => ({ tag: "trialDetailLoaded", generation, sessionId, trialId, detail }))
    .catch((error): TuningNavigationAction => ({ tag: "trialDetailFailed", generation, sessionId, trialId, error: String(error) }));
}
function merge(...effects: (Effect<TuningNavigationAction> | null)[]): Effect<TuningNavigationAction> | null {
  const present = effects.filter((effect): effect is Effect<TuningNavigationAction> => effect !== null);
  return present.length === 0 ? null : present.reduce((left, right) => Effect.merge(left, right));
}
function selectAttempt(state: TuningNavigationState, attemptId: string): void {
  state.selection = { ...state.selection, attemptId, trialId: null, pairId: null, gameId: null }; state.unavailable = null;
}
function selectTrial(state: TuningNavigationState, trialId: string): void {
  const legacyTrial = state.detail.snapshot?.trials.find((row) => row.trial_id === trialId);
  const pageTrial = state.trialPage.snapshot?.trials.find((row) => row.trial_id === trialId);
  const loadedTrial = state.trialDetails[trialId]?.snapshot?.trial;
  state.selection = { ...state.selection, attemptId: legacyTrial?.attempt_id ?? pageTrial?.attempt_id ?? loadedTrial?.attempt_id ?? state.selection.attemptId, trialId, pairId: null, gameId: null }; state.unavailable = null;
}
function selectPair(state: TuningNavigationState, pairId: string): void {
  const legacyTrial = state.detail.snapshot?.trials.find((row) => row.pairs.some((pair) => pair.pair_id === pairId));
  const loadedTrial = Object.values(state.trialDetails).map((value) => value.snapshot?.trial).find((trial) => trial?.pairs.some((pair) => pair.pair_id === pairId));
  state.selection = { ...state.selection, attemptId: legacyTrial?.attempt_id ?? loadedTrial?.attempt_id ?? state.selection.attemptId, trialId: legacyTrial?.trial_id ?? loadedTrial?.trial_id ?? state.selection.trialId, pairId, gameId: null }; state.unavailable = null;
}
function selectGame(state: TuningNavigationState, gameId: string): void {
  const legacyTrial = state.detail.snapshot?.trials.find((row) => row.pairs.some((pair) => pair.games.some((game) => game.game_id === gameId)));
  const loadedTrial = Object.values(state.trialDetails).map((value) => value.snapshot?.trial).find((trial) => trial?.pairs.some((pair) => pair.games.some((game) => game.game_id === gameId)));
  const trial = legacyTrial ?? loadedTrial;
  const pair = trial?.pairs.find((row) => row.games.some((game) => game.game_id === gameId));
  state.selection = { sessionId: state.selection.sessionId, attemptId: trial?.attempt_id ?? state.selection.attemptId, trialId: trial?.trial_id ?? state.selection.trialId, pairId: pair?.pair_id ?? state.selection.pairId, gameId }; state.unavailable = null;
}
function unavailable(state: TuningNavigationState, entity: "attempt" | "trial" | "pair" | "game"): void {
  const selected = state.selection;
  if (entity === "attempt") state.selection = selectionForSession(selected.sessionId!);
  if (entity === "trial") state.selection = { ...selected, trialId: null, pairId: null, gameId: null };
  if (entity === "pair") state.selection = { ...selected, pairId: null, gameId: null };
  if (entity === "game") state.selection = { ...selected, gameId: null };
  state.unavailable = `${entity} unavailable`;
}
function reconcileSelection(state: TuningNavigationState, detail: TuningSessionDetail): void {
  const selected = state.selection;
  const attempt = detail.attempts.find((row) => row.attempt_id === selected.attemptId);
  if (selected.attemptId && !attempt) return unavailable(state, "attempt");
  const trial = detail.trials.find((row) => row.trial_id === selected.trialId && row.attempt_id === selected.attemptId);
  if (selected.trialId && !trial) return unavailable(state, "trial");
  const pair = trial?.pairs.find((row) => row.pair_id === selected.pairId);
  if (selected.pairId && !pair) return unavailable(state, "pair");
  if (selected.gameId && !pair?.games.some((game) => game.game_id === selected.gameId)) unavailable(state, "game");
  else state.unavailable = null;
}
function reconcileTrialDetail(state: TuningNavigationState, detail: TuningTrialDetail): void {
  const selected = state.selection;
  if (selected.trialId !== detail.trial.trial_id) return;
  const pair = detail.trial.pairs.find((row) => row.pair_id === selected.pairId);
  if (selected.pairId && !pair) return unavailable(state, "pair");
  if (selected.gameId && !pair?.games.some((game) => game.game_id === selected.gameId)) unavailable(state, "game");
}
function overviewRefresh(state: TuningNavigationState, sessionId: string): Effect<TuningNavigationAction> | null {
  return state.selection.sessionId === sessionId && sessionIsActive(state, sessionId)
    ? Effect.delay(TUNING_DETAIL_REFRESH_MS, { tag: "overviewRefreshTick", sessionId, generation: state.overview.generation }) : null;
}
function resetTrialPage(state: TuningNavigationState): void {
  state.trialPage = { ...pageState(), generation: state.trialPage.generation + 1, sessionId: state.selection.sessionId };
}

function commandEffect(command: TuningSessionCommandState, sessionId: string, env: BenchEnv): Effect<TuningNavigationAction> {
  const body = { command_id: command.commandId, expected_version: command.expectedVersion };
  const effect = command.kind === "stop"
    ? env.stopTuningSession(sessionId, body)
    : command.kind === "resume"
      ? env.resumeTuningSession(sessionId, body)
      : env.addTuningSessionBudget(sessionId, {
        ...body,
        delta: command.delta!,
        start: command.start!,
        n_workers: command.nWorkers,
      });
  return effect
    .map((response): TuningNavigationAction => ({ tag: "sessionCommandSucceeded", sessionId, commandId: command.commandId, response }))
    .catch((error): TuningNavigationAction => {
      const status = typeof error === "object" && error !== null && typeof (error as { status?: unknown }).status === "number"
        ? (error as { status: number }).status : null;
      return { tag: "sessionCommandFailed", sessionId, commandId: command.commandId, error: String(error), retriable: status === null || status >= 500 };
    });
}

function applyControl(state: TuningNavigationState, sessionId: string, control: TuningSessionControl): void {
  const sessions = state.list.snapshot?.sessions;
  const listed = sessions?.find((session) => session.session_id === sessionId);
  if (listed) {
    listed.control = control;
    listed.target_trial_count = control.continuation.target_trial_count;
  }
  if (state.detail.sessionId === sessionId && state.detail.snapshot) {
    state.detail.snapshot.control = control;
    state.detail.snapshot.summary.target_trial_count = control.continuation.target_trial_count;
  }
  if (state.overview.sessionId === sessionId && state.overview.snapshot) state.overview.snapshot.control = control;
}

function refreshAfterCommand(state: TuningNavigationState, sessionId: string, env: BenchEnv): Effect<TuningNavigationAction> | null {
  return merge(
    requestList(state, env),
    state.selection.sessionId === sessionId ? requestOverview(state, sessionId, env) : null,
    state.selection.sessionId === sessionId && state.detail.sessionId === sessionId ? requestDetail(state, sessionId, env) : null,
  );
}

export function tuningNavigationReducer(state: TuningNavigationState, action: TuningNavigationAction, env: BenchEnv): Effect<TuningNavigationAction> | null {
  if (action.tag === "sessionCommandSubmit") {
    if (state.commands[action.sessionId]?.status === "pending") return null;
    if ((action.kind === "add_budget" && (!Number.isSafeInteger(action.delta) || action.delta! <= 0 || typeof action.start !== "boolean" || (!action.start && action.nWorkers !== undefined))) || !action.commandId) return null;
    const command: TuningSessionCommandState = {
      kind: action.kind, commandId: action.commandId, expectedVersion: action.expectedVersion,
      delta: action.delta, start: action.start, nWorkers: action.nWorkers,
      status: "pending", error: null, retriable: false, response: null,
    };
    state.commands[action.sessionId] = command;
    return commandEffect(command, action.sessionId, env);
  }
  if (action.tag === "sessionCommandRetry") {
    const command = state.commands[action.sessionId];
    if (!command || command.status !== "failed" || !command.retriable) return null;
    command.status = "pending"; command.error = null; command.retriable = false;
    return commandEffect(command, action.sessionId, env);
  }
  if (action.tag === "sessionCommandSucceeded") {
    const command = state.commands[action.sessionId];
    if (!command || command.commandId !== action.commandId || command.status !== "pending") return null;
    command.response = action.response;
    command.status = action.response.launch_error ? "failed" : "succeeded";
    command.error = action.response.launch_error ?? null;
    command.retriable = false;
    applyControl(state, action.sessionId, action.response.control);
    return refreshAfterCommand(state, action.sessionId, env);
  }
  if (action.tag === "sessionCommandFailed") {
    const command = state.commands[action.sessionId];
    if (!command || command.commandId !== action.commandId || command.status !== "pending") return null;
    command.status = "failed"; command.error = action.error; command.retriable = action.retriable;
    return null;
  }
  if (action.tag === "openCommandAttempt") {
    if (state.selection.sessionId === action.sessionId) selectAttempt(state, action.attemptId);
    return null;
  }
  if (action.tag === "listRequest") return requestList(state, env);
  if (action.tag === "listLoaded" || action.tag === "listFailed") {
    if (action.generation !== state.list.generation) return null;
    state.list = action.tag === "listLoaded" ? { ...state.list, status: "done", snapshot: action.response, error: null } : { ...state.list, status: "error", error: action.error };
    return null;
  }
  if (action.tag === "detailRequest") return state.selection.sessionId === action.sessionId ? requestDetail(state, action.sessionId, env) : null;
  if (action.tag === "detailLoaded" || action.tag === "detailFailed") {
    if (action.generation !== state.detail.generation || action.sessionId !== state.selection.sessionId) return null;
    if (action.tag === "detailLoaded") { state.detail = { ...state.detail, status: "done", snapshot: action.detail, error: null }; reconcileSelection(state, action.detail); }
    else state.detail = { ...state.detail, status: "error", error: action.error };
    return null;
  }
  if (action.tag === "overviewRequest") return state.selection.sessionId === action.sessionId ? requestOverview(state, action.sessionId, env) : null;
  if (action.tag === "overviewRefreshTick") {
    return action.generation === state.overview.generation && action.sessionId === state.selection.sessionId && sessionIsActive(state, action.sessionId)
      ? requestOverview(state, action.sessionId, env) : null;
  }
  if (action.tag === "overviewLoaded" || action.tag === "overviewFailed") {
    if (action.generation !== state.overview.generation || action.sessionId !== state.selection.sessionId) return null;
    if (action.tag === "overviewFailed") { state.overview = { ...state.overview, status: "error", error: action.error }; return null; }
    const priorCursor = state.overview.snapshot?.cursor.session_sequence;
    state.overview = { ...state.overview, status: "done", snapshot: action.overview, error: null };
    const advanced = priorCursor !== undefined && action.overview.cursor.session_sequence > priorCursor;
    return merge(
      state.tab === "trials" && (state.trialPage.snapshot === null || advanced) ? requestTrialPage(state, action.sessionId, env) : null,
      advanced && state.selection.trialId ? requestTrialDetail(state, action.sessionId, state.selection.trialId, env) : null,
      overviewRefresh(state, action.sessionId),
    );
  }
  if (action.tag === "trialPageRequest") return state.selection.sessionId === action.sessionId ? requestTrialPage(state, action.sessionId, env) : null;
  if (action.tag === "trialPageLoaded" || action.tag === "trialPageFailed") {
    if (action.generation !== state.trialPage.generation || action.sessionId !== state.selection.sessionId || action.queryKey !== state.trialPage.queryKey) return null;
    state.trialPage = action.tag === "trialPageLoaded" ? { ...state.trialPage, status: "done", snapshot: action.page, error: null } : { ...state.trialPage, status: "error", error: action.error };
    return null;
  }
  if (action.tag === "trialDetailRequest") return state.selection.sessionId === action.sessionId ? requestTrialDetail(state, action.sessionId, action.trialId, env) : null;
  if (action.tag === "trialDetailLoaded" || action.tag === "trialDetailFailed") {
    const current = state.trialDetails[action.trialId];
    if (!current || current.generation !== action.generation || current.sessionId !== action.sessionId) return null;
    if (action.tag === "trialDetailLoaded") { state.trialDetails[action.trialId] = { ...current, status: "done", snapshot: action.detail, error: null }; reconcileTrialDetail(state, action.detail); }
    else state.trialDetails[action.trialId] = { ...current, status: "error", error: action.error };
    return null;
  }
  if (action.tag === "selectSession") {
    state.selection = selectionForSession(action.sessionId);
    state.tab = sessionCanAnalyze(state, action.sessionId) ? "progress" : "game";
    state.ladderRevision = null; state.ladderAnchorKey = null;
    state.unavailable = null; clearAnalysis(state);
    return merge(
      sessionCanAnalyze(state, action.sessionId) ? null : requestDetail(state, action.sessionId, env),
      requestOverview(state, action.sessionId, env),
    );
  }
  if (action.tag === "clearSession") {
    state.selection = selectionForSession(null); state.ladderRevision = null; state.ladderAnchorKey = null; clearDetail(state); clearAnalysis(state); state.unavailable = null; return null;
  }
  if (action.tag === "setAnalysisTab") {
    state.tab = action.tab;
    if (action.tab === "trials" && state.selection.sessionId && state.trialPage.snapshot === null) return requestTrialPage(state, state.selection.sessionId, env);
    if (action.tab === "game" && state.selection.sessionId && sessionCanAnalyze(state, state.selection.sessionId)) {
      return merge(
        state.trialPage.snapshot === null ? requestTrialPage(state, state.selection.sessionId, env) : null,
        state.selection.trialId ? requestTrialDetail(state, state.selection.sessionId, state.selection.trialId, env) : null,
      );
    }
    if (action.tab === "ladder" && state.selection.sessionId && state.selection.trialId) return requestTrialDetail(state, state.selection.sessionId, state.selection.trialId, env);
    return null;
  }
  if (action.tag === "setProgressMetric") { state.progressMetric = action.metric; return null; }
  if (action.tag === "setProgressScale") { state.progressScale = action.scale; return null; }
  if (action.tag === "setLadderRevision") { state.ladderRevision = action.revision; return null; }
  if (action.tag === "selectLadderAnchor") { state.ladderAnchorKey = action.anchorKey; return null; }
  if (action.tag === "setTrialFilters" || action.tag === "setTrialSort" || action.tag === "setTrialPageLimit") {
    if (action.tag === "setTrialFilters") state.filters = { ...state.filters, ...action.filters };
    else if (action.tag === "setTrialSort") state.sort = action.sort;
    else state.trialPageLimit = Math.max(1, Math.min(MAX_TRIAL_PAGE_LIMIT, Math.floor(action.limit) || DEFAULT_TRIAL_PAGE_LIMIT));
    resetTrialPage(state);
    return state.selection.sessionId && state.tab === "trials" ? requestTrialPage(state, state.selection.sessionId, env) : null;
  }
  if (action.tag === "nextTrialPage") {
    const next = state.trialPage.snapshot?.next_cursor;
    if (!next || !state.selection.sessionId) return null;
    state.trialPage.previousCursors = [...state.trialPage.previousCursors, state.trialPage.cursor]; state.trialPage.cursor = next;
    return requestTrialPage(state, state.selection.sessionId, env);
  }
  if (action.tag === "previousTrialPage") {
    if (!state.selection.sessionId || state.trialPage.previousCursors.length === 0) return null;
    state.trialPage.cursor = state.trialPage.previousCursors.at(-1)!; state.trialPage.previousCursors = state.trialPage.previousCursors.slice(0, -1);
    return requestTrialPage(state, state.selection.sessionId, env);
  }
  if (action.tag === "selectAttempt") selectAttempt(state, action.attemptId);
  if (action.tag === "selectTrial") {
    selectTrial(state, action.trialId);
    return (state.tab === "ladder" || state.tab === "game") && state.selection.sessionId && sessionCanAnalyze(state, state.selection.sessionId)
      ? requestTrialDetail(state, state.selection.sessionId, action.trialId, env) : null;
  }
  if (action.tag === "selectPair") selectPair(state, action.pairId);
  if (action.tag === "selectGame") selectGame(state, action.gameId);
  if (action.tag === "toggleExpanded") state.expandedIds = state.expandedIds.includes(action.id) ? state.expandedIds.filter((id) => id !== action.id) : [...state.expandedIds, action.id];
  return null;
}
