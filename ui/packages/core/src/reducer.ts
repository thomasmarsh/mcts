// core/reducer.ts — TCA-style reducer composition with valtio drafts.
//
// pullback() scopes a child Reducer to a parent state via lens mutation.
// combine() runs all reducers in sequence over the same proxy.

import { Effect } from "./effect.js";

export type Dispatch<A> = (a: A) => void;

export type Reducer<S, A, Env> = (draft: S, action: A, env: Env) => Effect<A> | null;

export function pullback<S, A, PS, PA, FEnv, PEnv>(
  child: Reducer<S, A, FEnv>,
  get: (parent: PS) => S,
  match: (action: PA) => A | null,
  widen: (a: A) => PA,
  getEnv: (env: PEnv) => FEnv,
): Reducer<PS, PA, PEnv> {
  return (draft, action, env) => {
    const local = match(action);
    if (!local) return null;
    const eff = child(get(draft), local, getEnv(env));
    return eff ? eff.map(widen) : null;
  };
}

/**
 * Like pullback, but injects `env.navigate` into the child env so reducers
 * can emit navigation side-effects without polluting their own action types.
 * Calls to `env.navigate` are captured synchronously during reduce and emitted
 * as parent-level effects via `widenNav`, interleaved with any feature effects.
 */
export function pullbackWithNav<
  NavAction,
  S,
  A,
  E extends { navigate(a: NavAction): Effect<never> },
  PS,
  PA,
  PEnv,
>(
  child: Reducer<S, A, E>,
  get: (parent: PS) => S,
  match: (action: PA) => A | null,
  widen: (a: A) => PA,
  getEnv: (env: PEnv) => Omit<E, "navigate">,
  widenNav: (nav: NavAction) => PA,
): Reducer<PS, PA, PEnv> {
  return (draft, action, env) => {
    const local = match(action);
    if (!local) return null;
    const pending: PA[] = [];
    const childEnv = {
      ...(getEnv(env) as E),
      navigate: (nav: NavAction): Effect<never> => {
        pending.push(widenNav(nav));
        return Effect.none();
      },
    };
    const eff = child(get(draft), local, childEnv);
    const navEff =
      pending.length > 0 ? Effect.merge(...pending.map((a) => Effect.send<PA>(a))) : null;
    if (!eff && !navEff) return null;
    if (!eff) return navEff;
    if (!navEff) return eff.map(widen);
    return Effect.merge(eff.map(widen), navEff);
  };
}

export function combine<S, A, Env>(...reducers: Reducer<S, A, Env>[]): Reducer<S, A, Env> {
  return (draft, action, env) => {
    const effs: Effect<A>[] = [];
    for (const r of reducers) {
      const eff = r(draft, action, env);
      if (eff) effs.push(eff);
    }
    return effs.length > 0 ? Effect.merge(...effs) : null;
  };
}
