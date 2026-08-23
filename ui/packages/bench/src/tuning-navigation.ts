// Tuning-session navigation state stays separate from physical bench-run state.

import { Effect } from "@mcts/core";
import type { BenchEnv } from "./reducer.js";
import type { TuningSessionDetail, TuningSessionsResponse } from "./types.js";

export const TUNING_DETAIL_REFRESH_MS = 5_000;

export interface TuningLoadState<T> {
  status: "idle" | "loading" | "done" | "error";
  snapshot: T | null;
  error: string | null;
  generation: number;
}

export interface TuningSelection {
  sessionId: string | null;
  attemptId: string | null;
  trialId: string | null;
  pairId: string | null;
  gameId: string | null;
}

export interface TuningNavigationState {
  list: TuningLoadState<TuningSessionsResponse>;
  detail: TuningLoadState<TuningSessionDetail> & { sessionId: string | null };
  selection: TuningSelection;
  expandedIds: string[];
  unavailable: string | null;
}

export type TuningNavigationAction =
  | { tag: "listRequest" }
  | { tag: "listLoaded"; generation: number; response: TuningSessionsResponse }
  | { tag: "listFailed"; generation: number; error: string }
  | { tag: "detailRequest"; sessionId: string }
  | { tag: "detailRefreshTick"; sessionId: string; generation: number }
  | { tag: "detailLoaded"; generation: number; sessionId: string; detail: TuningSessionDetail }
  | { tag: "detailFailed"; generation: number; sessionId: string; error: string }
  | { tag: "selectSession"; sessionId: string }
  | { tag: "clearSession" }
  | { tag: "selectAttempt"; attemptId: string }
  | { tag: "selectTrial"; trialId: string }
  | { tag: "selectPair"; pairId: string }
  | { tag: "selectGame"; gameId: string }
  | { tag: "toggleExpanded"; id: string };

function loadState<T>(): TuningLoadState<T> {
  return { status: "idle", snapshot: null, error: null, generation: 0 };
}

export function initialTuningNavigationState(): TuningNavigationState {
  return {
    list: loadState<TuningSessionsResponse>(),
    detail: { ...loadState<TuningSessionDetail>(), sessionId: null },
    selection: { sessionId: null, attemptId: null, trialId: null, pairId: null, gameId: null },
    expandedIds: [],
    unavailable: null,
  };
}

function selectionForSession(sessionId: string | null): TuningSelection {
  return { sessionId, attemptId: null, trialId: null, pairId: null, gameId: null };
}

function clearDetail(state: TuningNavigationState): void {
  state.detail.generation += 1;
  state.detail = { ...loadState<TuningSessionDetail>(), generation: state.detail.generation, sessionId: null };
}

function requestList(state: TuningNavigationState, env: BenchEnv): Effect<TuningNavigationAction> {
  const generation = state.list.generation + 1;
  state.list = { ...state.list, status: "loading", error: null, generation };
  return env.listTuningSessions()
    .map((response): TuningNavigationAction => ({ tag: "listLoaded", generation, response }))
    .catch((error): TuningNavigationAction => ({ tag: "listFailed", generation, error: String(error) }));
}

function requestDetail(state: TuningNavigationState, sessionId: string, env: BenchEnv): Effect<TuningNavigationAction> {
  const generation = state.detail.generation + 1;
  const snapshot = state.detail.sessionId === sessionId ? state.detail.snapshot : null;
  state.detail = { status: "loading", snapshot, error: null, generation, sessionId };
  return env.getTuningSession(sessionId)
    .map((detail): TuningNavigationAction => ({ tag: "detailLoaded", generation, sessionId, detail }))
    .catch((error): TuningNavigationAction => ({ tag: "detailFailed", generation, sessionId, error: String(error) }));
}

function selectAttempt(state: TuningNavigationState, attemptId: string): void {
  state.selection = { ...state.selection, attemptId, trialId: null, pairId: null, gameId: null };
  state.unavailable = null;
}

function selectTrial(state: TuningNavigationState, trialId: string): void {
  const trial = state.detail.snapshot?.trials.find((row) => row.trial_id === trialId);
  state.selection = { ...state.selection, attemptId: trial?.attempt_id ?? null, trialId, pairId: null, gameId: null };
  state.unavailable = null;
}

function selectPair(state: TuningNavigationState, pairId: string): void {
  const trial = state.detail.snapshot?.trials.find((row) => row.pairs.some((pair) => pair.pair_id === pairId));
  state.selection = { ...state.selection, attemptId: trial?.attempt_id ?? null, trialId: trial?.trial_id ?? null, pairId, gameId: null };
  state.unavailable = null;
}

function selectGame(state: TuningNavigationState, gameId: string): void {
  const trial = state.detail.snapshot?.trials.find((row) => row.pairs.some((pair) => pair.games.some((game) => game.game_id === gameId)));
  const pair = trial?.pairs.find((row) => row.games.some((game) => game.game_id === gameId));
  state.selection = { sessionId: state.selection.sessionId, attemptId: trial?.attempt_id ?? null, trialId: trial?.trial_id ?? null, pairId: pair?.pair_id ?? null, gameId };
  state.unavailable = null;
}

function reconcileSelection(state: TuningNavigationState, detail: TuningSessionDetail): void {
  const selected = state.selection;
  const attempt = detail.attempts.find((row) => row.attempt_id === selected.attemptId);
  if (selected.attemptId && !attempt) return unavailable(state, "attempt");
  const trial = detail.trials.find((row) => row.trial_id === selected.trialId && row.attempt_id === selected.attemptId);
  if (selected.trialId && !trial) return unavailable(state, "trial");
  const pair = trial?.pairs.find((row) => row.pair_id === selected.pairId);
  if (selected.pairId && !pair) return unavailable(state, "pair");
  const game = pair?.games.find((row) => row.game_id === selected.gameId);
  if (selected.gameId && !game) return unavailable(state, "game");
  state.unavailable = null;
}

function unavailable(state: TuningNavigationState, entity: "attempt" | "trial" | "pair" | "game"): void {
  const selected = state.selection;
  if (entity === "attempt") state.selection = selectionForSession(selected.sessionId!);
  if (entity === "trial") state.selection = { ...selected, trialId: null, pairId: null, gameId: null };
  if (entity === "pair") state.selection = { ...selected, pairId: null, gameId: null };
  if (entity === "game") state.selection = { ...selected, gameId: null };
  state.unavailable = `${entity} unavailable`;
}

function toggleExpanded(state: TuningNavigationState, id: string): void {
  state.expandedIds = state.expandedIds.includes(id)
    ? state.expandedIds.filter((value) => value !== id)
    : [...state.expandedIds, id];
}

function detailRefresh(state: TuningNavigationState, sessionId: string): Effect<TuningNavigationAction> | null {
  return state.selection.sessionId === sessionId && state.detail.snapshot?.summary.status === "active"
    ? Effect.delay(TUNING_DETAIL_REFRESH_MS, { tag: "detailRefreshTick", sessionId, generation: state.detail.generation })
    : null;
}

type ListAction = Extract<TuningNavigationAction, { tag: "listRequest" | "listLoaded" | "listFailed" }>;
type DetailAction = Extract<TuningNavigationAction, { tag: "detailRequest" | "detailRefreshTick" | "detailLoaded" | "detailFailed" }>;
type SelectionAction = Exclude<TuningNavigationAction, ListAction | DetailAction>;

function reduceList(state: TuningNavigationState, action: ListAction, env: BenchEnv): Effect<TuningNavigationAction> | null {
  if (action.tag === "listRequest") return requestList(state, env);
  if (action.generation !== state.list.generation) return null;
  if (action.tag === "listLoaded") state.list = { ...state.list, status: "done", snapshot: action.response, error: null };
  else state.list = { ...state.list, status: "error", error: action.error };
  return null;
}

function reduceDetail(state: TuningNavigationState, action: DetailAction, env: BenchEnv): Effect<TuningNavigationAction> | null {
  if (action.tag === "detailRequest") {
    return state.selection.sessionId === action.sessionId ? requestDetail(state, action.sessionId, env) : null;
  }
  if (action.tag === "detailRefreshTick") {
    return action.generation === state.detail.generation
      && action.sessionId === state.selection.sessionId
      && state.detail.snapshot?.summary.status === "active"
      ? requestDetail(state, action.sessionId, env)
      : null;
  }
  if (action.generation !== state.detail.generation || action.sessionId !== state.selection.sessionId) return null;
  if (action.tag === "detailLoaded") {
    state.detail = { ...state.detail, status: "done", snapshot: action.detail, error: null };
    reconcileSelection(state, action.detail);
    return detailRefresh(state, action.sessionId);
  }
  state.detail = { ...state.detail, status: "error", error: action.error };
  return null;
}

function reduceSelection(state: TuningNavigationState, action: SelectionAction, env: BenchEnv): Effect<TuningNavigationAction> | null {
  if (action.tag === "selectSession") {
    state.selection = selectionForSession(action.sessionId);
    state.unavailable = null;
    return requestDetail(state, action.sessionId, env);
  }
  if (action.tag === "clearSession") {
    state.selection = selectionForSession(null);
    clearDetail(state);
    state.unavailable = null;
    return null;
  }
  if (action.tag === "selectAttempt") selectAttempt(state, action.attemptId);
  if (action.tag === "selectTrial") selectTrial(state, action.trialId);
  if (action.tag === "selectPair") selectPair(state, action.pairId);
  if (action.tag === "selectGame") selectGame(state, action.gameId);
  if (action.tag === "toggleExpanded") toggleExpanded(state, action.id);
  return null;
}

export function tuningNavigationReducer(
  state: TuningNavigationState,
  action: TuningNavigationAction,
  env: BenchEnv,
): Effect<TuningNavigationAction> | null {
  if (action.tag === "listRequest" || action.tag === "listLoaded" || action.tag === "listFailed") return reduceList(state, action, env);
  if (action.tag === "detailRequest" || action.tag === "detailRefreshTick" || action.tag === "detailLoaded" || action.tag === "detailFailed") return reduceDetail(state, action, env);
  return reduceSelection(state, action, env);
}
