// reducer.ts — App-level reducer: combines the `GameTree` reducer with the
// `aiMove`/`analysis` job-poll wiring (PLAN-UI.md session 3). Owns the `Env`
// type consumed by both this reducer and `api-client.ts`'s `createEnv` --
// mirrors pb/ui/app/src/reducer.ts's convention of defining `AppEnv`
// alongside the reducer that consumes it (`api-client.ts` imports this
// module's `Env` type, not the other way around, so there's no runtime
// circular import between the two files).
//
// `aiMove`/`analysis` are handled as direct branches here, not via
// `pullback`, because -- like pb's `inlineDiagram`/`cfgDiagram` handling in
// its own `reducer.ts` -- each dispatch needs to read `draft.gameKind` and
// the tree's current state to build that call's `JobPollEnv` dynamically;
// `pullback`'s `getEnv` only ever sees `env`, never `draft`. `GameTree` has
// no such need (it never touches the network), so it's wired through
// `pullback` normally.

import {
  Effect,
  jobPollReduce,
  pullback,
  type JobPollAction,
  type JobPollEnv,
  type JobSubmitResult,
} from "@mcts/core";
import { gameTreeReducer, type GameTree, type GameTreeAction } from "./game-tree.js";
import type { AppState } from "./state.js";
import type {
  AiMoveResult,
  AiPresetInfo,
  Analysis,
  GameInfo,
  LegalMovesResult,
  StateAndView,
} from "./types.js";

/** Every network operation a reducer in this package may perform, lifted to
 * `Effect` -- see PLAN-UI.md's "Hard rule": no reducer or component calls
 * `fetch`/`ApiClient` directly, only `env.xxx()`. Each method is generic per
 * call (not per `Env` instance) so this single `Env` type serves every game
 * kind/state/move combination without itself naming one.
 */
export interface Env {
  getGames(): Effect<GameInfo[]>;
  newGame<S, V = unknown>(kind: string, config?: unknown): Effect<StateAndView<S, V>>;
  legalMoves<S, M>(kind: string, state: S): Effect<LegalMovesResult<M>>;
  view<S, V = unknown>(kind: string, state: S): Effect<V>;
  apply<S, M, V = unknown>(kind: string, state: S, move: M): Effect<StateAndView<S, V>>;
  aiPresets(kind: string): Effect<AiPresetInfo[]>;
  aiMove<S, M, V = unknown>(kind: string, state: S, preset: string): Effect<AiMoveResult<S, M, V>>;
  analyze<S, M>(kind: string, state: S, preset: string, budgetMs?: number): Effect<Analysis<M>>;
}

export type AiMoveJobAction<S, M> =
  | { tag: "request"; preset: string }
  | { tag: "job"; action: JobPollAction<AiMoveResult<S, M>> };

export type AnalysisJobAction<M> =
  | { tag: "request"; preset: string; budgetMs?: number }
  | { tag: "job"; action: JobPollAction<Analysis<M>> };

export type AppAction<S, M> =
  | { tag: "tree"; action: GameTreeAction<S, M> }
  | { tag: "aiMove"; action: AiMoveJobAction<S, M> }
  | { tag: "analysis"; action: AnalysisJobAction<M> };

/** `jobPollReduce` only ever calls `submitJob`/`pollJob` for the `"start"`/
 * `"tick"` tags. Every `submitJob` this reducer builds resolves directly to
 * `{status: "done", ...}` (see the "request" branches below), so `"tick"`
 * -- and therefore `pollJob` -- is never reached; nor is `submitJob` reached
 * from the `"job"` branch, since a `"start"` action only ever originates
 * from a `"request"` action (which builds its own real `jobEnv`). This stub
 * exists purely to satisfy `JobPollEnv`'s shape for those unreachable paths,
 * and throws loudly if that assumption is ever wrong. */
function unreachableJobEnv<T>(reason: string): JobPollEnv<T> {
  return {
    submitJob: () => {
      throw new Error(reason);
    },
    pollJob: () => {
      throw new Error(reason);
    },
  };
}

export function appReducer<S, M>(
  draft: AppState<S, M>,
  action: AppAction<S, M>,
  env: Env,
): Effect<AppAction<S, M>> | null {
  const treeReducer = pullback<GameTree<S, M>, GameTreeAction<S, M>, AppState<S, M>, AppAction<S, M>, unknown, Env>(
    gameTreeReducer,
    (s) => s.tree,
    (a) => (a.tag === "tree" ? a.action : null),
    (a): AppAction<S, M> => ({ tag: "tree", action: a }),
    () => undefined,
  );
  const treeEffect = treeReducer(draft, action, env);
  if (treeEffect) return treeEffect;

  if (action.tag === "aiMove") {
    const ja = action.action;
    if (ja.tag === "request") {
      const current = draft.tree.nodes[draft.tree.currentId];
      if (!current) return null;
      const { gameKind } = draft;
      const { preset } = ja;
      const jobEnv: JobPollEnv<AiMoveResult<S, M>> = {
        submitJob: () =>
          env
            .aiMove<S, M>(gameKind, current.state, preset)
            .map((result): JobSubmitResult<AiMoveResult<S, M>> => ({ status: "done", result })),
        pollJob: () => {
          throw new Error("unreachable: ai_move resolves synchronously (see submitJob above)");
        },
      };
      const eff = jobPollReduce(draft.aiMove, { tag: "start" }, jobEnv);
      return eff ? eff.map((a): AppAction<S, M> => ({ tag: "aiMove", action: { tag: "job", action: a } })) : null;
    }
    const eff = jobPollReduce(
      draft.aiMove,
      ja.action,
      unreachableJobEnv("unreachable: a forwarded aiMove/job action never re-submits or polls"),
    );
    return eff ? eff.map((a): AppAction<S, M> => ({ tag: "aiMove", action: { tag: "job", action: a } })) : null;
  }

  if (action.tag === "analysis") {
    const ja = action.action;
    if (ja.tag === "request") {
      const current = draft.tree.nodes[draft.tree.currentId];
      if (!current) return null;
      const { gameKind } = draft;
      const { preset, budgetMs } = ja;
      const jobEnv: JobPollEnv<Analysis<M>> = {
        submitJob: () =>
          env
            .analyze<S, M>(gameKind, current.state, preset, budgetMs)
            .map((result): JobSubmitResult<Analysis<M>> => ({ status: "done", result })),
        pollJob: () => {
          throw new Error("unreachable: analyze resolves synchronously (see submitJob above)");
        },
      };
      const eff = jobPollReduce(draft.analysis, { tag: "start" }, jobEnv);
      return eff ? eff.map((a): AppAction<S, M> => ({ tag: "analysis", action: { tag: "job", action: a } })) : null;
    }
    const eff = jobPollReduce(
      draft.analysis,
      ja.action,
      unreachableJobEnv("unreachable: a forwarded analysis/job action never re-submits or polls"),
    );
    return eff ? eff.map((a): AppAction<S, M> => ({ tag: "analysis", action: { tag: "job", action: a } })) : null;
  }

  return null;
}
