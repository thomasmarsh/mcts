// core/store.ts — Valtio proxy store with SolidJS snapshot bridge + scope.

import { proxy, snapshot, subscribe } from "valtio/vanilla";
import { createMemo, createSignal, onCleanup } from "solid-js";
import type { Effect } from "./effect.js";

// ── Types ────────────────────────────────────────────────────────────────────

export interface Store<AppState, AppAction> {
  /** The raw valtio proxy — reducers mutate this directly. */
  state: AppState;
  /** Dispatch a tagged action through the reducer. */
  dispatch: (action: AppAction) => void;
  /** Returns a reactive SolidJS accessor for the current state snapshot. Call inside a component. */
  getState: () => () => AppState;
}

/**
 * Scoped store — narrowed state, narrowed actions, narrowed env.
 * Created by scope() for prop-drilling into child components.
 */
export interface ScopedStore<NarrowState, NarrowAction> {
  state: NarrowState;
  dispatch: (action: NarrowAction) => void;
  /** Returns a reactive SolidJS accessor for the current state snapshot. Call inside a component. */
  getState: () => () => NarrowState;
}

// ── Snapshot (internal) ──────────────────────────────────────────────────────

function useSnapshot<S extends object>(proxyState: S): () => S {
  // createSignal overload requires Exclude<T, Function>; cast via any since S is always a plain object
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [snap, setSnap] = createSignal<S>(snapshot(proxyState as any) as any);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const unsub = subscribe(proxyState as any, () => setSnap(snapshot(proxyState as any) as any));
  onCleanup(unsub);
  return snap;
}

// ── Scope ────────────────────────────────────────────────────────────────────

/**
 * Create a scoped store from a parent store.
 *
 * - `get`: read narrow state from parent
 * - `widen`: embed narrow action into parent action space
 * - `narrowEnv`: extract narrow env from full env (static, captured at scope time)
 *
 * The scoped store's `dispatch` sends widened actions into the parent.
 * The scoped store's `state` is a reactive getter of the narrow slice.
 */
export function scope<PS extends object, PA, NS, NA, NEnv, AppEnv>(
  parent: Store<PS, PA>,
  get: (app: PS) => NS,
  widen: (a: NA) => PA,
  narrowEnv: (env: AppEnv) => NEnv,
  env: AppEnv,
): ScopedStore<NS, NA> & { env: NEnv } {
  return {
    get state(): NS { return get(parent.state); },
    dispatch: (a: NA) => parent.dispatch(widen(a)),
    env: narrowEnv(env),
    getState: () => { const p = parent.getState(); return createMemo(() => get(p())); },
  };
}

// ── Store factory ────────────────────────────────────────────────────────────

export function createStore<S extends object, A, Env>(
  init: S,
  reducerFn: (draft: S, action: A, env: Env) => Effect<A> | null,
  env: Env,
  onDispatch?: (action: A, state: S) => void,
): Store<S, A> {
  const state = proxy(init) as S;

  function dispatch(action: A): void {
    const effect = reducerFn(state, action, env);
    onDispatch?.(action, state);
    if (effect) {
      effect.execute(dispatch).catch((e: unknown) => console.error("unhandled effect error:", e));
    }
  }

  return { state, dispatch, getState: () => useSnapshot(state) };
}
