// reducer.ts — App-level reducer: combines the `GameTree` reducer with
// `newGame`/`move`/`aiMove`/`analysis` job-poll wiring and current-position
// (view + legal moves) derivation (PLAN-UI.md sessions 3 and 4).
//
// `newGame`/`move`/`aiMove`/`analysis`/`position` are handled as direct
// branches here, not via `pullback`, because -- like pb's `inlineDiagram`/
// `cfgDiagram` handling in its own `reducer.ts` -- each dispatch needs to
// read `draft.gameKind` and the tree's current state to build that call's
// env dynamically; `pullback`'s `getEnv` only ever sees `env`, never `draft`.
// `GameTree` has no such need (it never touches the network), so it's wired
// through `pullback` normally.

import {
  Effect,
  initialJobPollState,
  jobPollReduce,
  pullback,
  type JobPollAction,
  type JobPollEnv,
  type JobSubmitResult,
} from "@mcts/core";
import { gameTreeReducer, initialGameTree, type GameTree, type GameTreeAction } from "./game-tree.js";
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

/** Runs an `Effect` for its single value, as a `Promise` -- lets a reducer
 * combine two `env.xxx()` calls (e.g. `position`'s `view` + `legalMoves`)
 * with `Promise.all` while still routing every network call through `env`,
 * never `fetch` directly (PLAN-UI.md's hard rule only forbids the latter). */
function toPromise<T>(effect: Effect<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    effect.execute((v) => resolve(v)).catch(reject);
  });
}

export type NewGameJobAction<S, V> =
  | { tag: "request"; config?: unknown }
  | { tag: "job"; action: JobPollAction<StateAndView<S, V>> };

export type MoveJobAction<S, M, V> =
  | { tag: "request"; move: M }
  | { tag: "job"; action: JobPollAction<StateAndView<S, V>> };

export type AiMoveJobAction<S, M, V> =
  | { tag: "request"; preset: string }
  | { tag: "job"; action: JobPollAction<AiMoveResult<S, M, V>> };

export type AnalysisJobAction<M> =
  | { tag: "request"; preset: string; budgetMs?: number }
  | { tag: "job"; action: JobPollAction<Analysis<M>> };

export type PositionAction<V, M> =
  | { tag: "request" }
  /** `epoch` is the epoch this request was issued under (mirrors
   * `aiMove`/`analysis`'s own `epoch`-stamping) -- `nodeId` alone isn't a
   * sufficient staleness guard once `switchGame` (session 8) can change
   * `gameKind` while a request for the *previous* kind's current node is
   * still in flight: that response's `nodeId` can coincidentally still
   * equal `draft.tree.currentId` (nothing else has touched the tree yet),
   * but it carries a view/legal-moves shape the new `gameKind` can't
   * interpret. Checking `epoch` too catches that case, since `switchGame`
   * always changes `epoch`. */
  | { tag: "loaded"; nodeId: string; epoch: number; view: V; moves: M[] };

export type AiPresetsJobAction =
  | { tag: "request" }
  | { tag: "job"; action: JobPollAction<AiPresetInfo[]> };

export type AppAction<S, M, V = unknown> =
  | { tag: "tree"; action: GameTreeAction<S, M> }
  | { tag: "position"; action: PositionAction<V, M> }
  | { tag: "aiPresets"; action: AiPresetsJobAction }
  | { tag: "newGame"; action: NewGameJobAction<S, V>; config?: unknown }
  | { tag: "move"; action: MoveJobAction<S, M, V>; move?: M }
  | { tag: "aiMove"; action: AiMoveJobAction<S, M, V>; epoch?: number }
  | { tag: "analysis"; action: AnalysisJobAction<M>; epoch?: number }
  | { tag: "setSeat"; player: string; control: string }
  | { tag: "setPreset"; preset: string }
  /** Session 8: switch which game kind the New Game dialog is about to
   * start, ahead of the actual `newGame` dispatch. Resets everything
   * `newGame`'s own "done" branch resets (see below) plus `aiPresets`/
   * `seats` -- both are per-`gameKind` metadata that a different kind's
   * dialog needs fetched/cleared fresh, not carried over from whichever
   * kind was previously active. Also drops `epoch` to 0 (see this branch's
   * own comment) until `newGame` bumps it back up, so nothing tries to
   * re-derive a position from `tree`'s now-mismatched-kind state in the
   * meantime. */
  | { tag: "switchGame"; gameKind: string }
  /** Session 7: rehydrate a save file's `{gameKind, config, tree}` wholesale
   * -- fully client-side, no `env` call. Resets the same job-poll slices and
   * bumps `epoch` the same way a completed `newGame` does, so an in-flight
   * `aiMove`/`analysis` from the game being replaced gets dropped rather than
   * grafted onto the loaded one (see `state.ts`'s `epoch` doc). */
  | { tag: "load"; gameKind: string; config: unknown; tree: GameTree<S, M> };

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

export function appReducer<S, M, V = unknown>(
  draft: AppState<S, M, V>,
  action: AppAction<S, M, V>,
  env: Env,
): Effect<AppAction<S, M, V>> | null {
  const treeReducer = pullback<GameTree<S, M>, GameTreeAction<S, M>, AppState<S, M, V>, AppAction<S, M, V>, unknown, Env>(
    gameTreeReducer,
    (s) => s.tree,
    (a) => (a.tag === "tree" ? a.action : null),
    (a): AppAction<S, M, V> => ({ tag: "tree", action: a }),
    () => undefined,
  );
  const treeEffect = treeReducer(draft, action, env);
  if (action.tag === "tree") {
    // Any tree navigation (undo/redo/jumpTo/deleteBranch) can change
    // `currentId` -- a stale `analysis` result would otherwise go on
    // labeling a position it was never computed for, which the heatmap
    // overlay/suggested-move highlight would then silently mis-render
    // against the new position's board. `GameTree`'s own reducer never
    // returns an effect (it's pure/synchronous, see its header comment), so
    // `treeEffect` is always null here -- this reset is the only thing this
    // branch does beyond what `treeReducer` already did as a side effect of
    // the call above.
    draft.analysis = initialJobPollState<Analysis<M>>();
    // Same reasoning extends to `position`: it's kept null (rather than
    // whatever the previous `currentId` had) until the `position/request`
    // effect this triggers re-derives it for the new node. This is what
    // makes "`draft.position` is null or belongs to `draft.tree.currentId`"
    // a true invariant of this reducer -- every branch that can move
    // `currentId` (this one, and `move`/`aiMove`'s "done" branches below)
    // nulls `position` in the same reduction, so nothing downstream needs to
    // separately check `position.nodeId` against `tree.currentId` to know
    // whether a read is stale.
    draft.position = null;
    return treeEffect;
  }

  if (action.tag === "setSeat") {
    draft.seats[action.player] = action.control;
    return null;
  }

  if (action.tag === "setPreset") {
    draft.ui.selectedPreset = action.preset;
    return null;
  }

  if (action.tag === "switchGame") {
    draft.gameKind = action.gameKind;
    draft.position = null;
    draft.move = initialJobPollState<StateAndView<S, V>>();
    draft.aiMove = initialJobPollState<AiMoveResult<S, M, V>>();
    draft.analysis = initialJobPollState<Analysis<M>>();
    draft.newGame = initialJobPollState<StateAndView<S, V>>();
    draft.aiPresets = initialJobPollState<AiPresetInfo[]>();
    draft.seats = {};
    // `draft.tree` is deliberately left as-is here -- the outgoing kind's
    // nodes, still shaped for the outgoing kind's S/M -- since the actual
    // replacement only happens once the `newGame` request dispatched right
    // after this one completes (see that branch's "done" handling below).
    // That leaves a real, not just cosmetic, window where `gameKind` and
    // `tree` disagree about which game's shape the tree's states are in:
    // `GameShell`'s `position/request` effect fires on *every* store update
    // (`useSnapshot`'s valtio subscription re-fires the whole snapshot
    // signal on any mutation, and the effect reads that signal
    // unconditionally before checking `tree.currentId` -- it doesn't
    // actually gate on that value changing), so without an extra guard the
    // very next store update after this one would fire `env.view(newKind,
    // <oldKind-shaped state>)` and hit the server's "invalid state" 400.
    // Resetting `epoch` to 0 reuses the exact guard `App.tsx`'s pre-
    // bootstrap placeholder root already relies on (`if (s.epoch < 1)
    // return;` in that effect) to suppress `position/request` until
    // `newGame`'s completion below bumps it back to >= 1 in the same
    // reduction that replaces `tree` -- at which point kind and tree are
    // atomically consistent again, same as that initial-bootstrap case.
    draft.epoch = 0;
    return null;
  }

  if (action.tag === "load") {
    draft.gameKind = action.gameKind;
    draft.config = action.config;
    draft.tree = action.tree;
    draft.position = null;
    draft.move = initialJobPollState<StateAndView<S, V>>();
    draft.aiMove = initialJobPollState<AiMoveResult<S, M, V>>();
    draft.analysis = initialJobPollState<Analysis<M>>();
    draft.newGame = initialJobPollState<StateAndView<S, V>>();
    draft.epoch += 1;
    return null;
  }

  if (action.tag === "position") {
    const pa = action.action;
    if (pa.tag === "request") {
      const current = draft.tree.nodes[draft.tree.currentId];
      if (!current) return null;
      const { gameKind, epoch } = draft;
      const nodeId = draft.tree.currentId;
      return Effect.fromPromise(async () => {
        const [view, legal] = await Promise.all([
          toPromise(env.view<S, V>(gameKind, current.state)),
          toPromise(env.legalMoves<S, M>(gameKind, current.state)),
        ]);
        return { nodeId, epoch, view, moves: legal.moves };
      }).map((r): AppAction<S, M, V> => ({ tag: "position", action: { tag: "loaded", ...r } }));
    }
    // Drop a `loaded` result for a node that's no longer current (superseded
    // by a later navigation/move, whose own `loaded` will land instead) or
    // whose epoch has since moved on (switchGame/newGame/load -- see
    // `PositionAction`'s doc comment above for why `nodeId` alone isn't
    // enough once a kind switch can leave `tree.currentId` pointing at a
    // now-foreign-shaped node).
    if (pa.nodeId === draft.tree.currentId && pa.epoch === draft.epoch) {
      draft.position = { nodeId: pa.nodeId, view: pa.view, legalMoves: pa.moves };
    }
    return null;
  }

  if (action.tag === "aiPresets") {
    const ja = action.action;
    if (ja.tag === "request") {
      const { gameKind } = draft;
      const jobEnv: JobPollEnv<AiPresetInfo[]> = {
        submitJob: () => env.aiPresets(gameKind).map((result): JobSubmitResult<AiPresetInfo[]> => ({ status: "done", result })),
        pollJob: () => {
          throw new Error("unreachable: ai_presets resolves synchronously (see submitJob above)");
        },
      };
      const eff = jobPollReduce(draft.aiPresets, { tag: "start" }, jobEnv);
      return eff ? eff.map((a): AppAction<S, M, V> => ({ tag: "aiPresets", action: { tag: "job", action: a } })) : null;
    }
    const eff = jobPollReduce(
      draft.aiPresets,
      ja.action,
      unreachableJobEnv("unreachable: a forwarded aiPresets/job action never re-submits or polls"),
    );
    return eff ? eff.map((a): AppAction<S, M, V> => ({ tag: "aiPresets", action: { tag: "job", action: a } })) : null;
  }

  if (action.tag === "newGame") {
    const ja = action.action;
    if (ja.tag === "request") {
      const { gameKind } = draft;
      const jobEnv: JobPollEnv<StateAndView<S, V>> = {
        submitJob: () =>
          env.newGame<S, V>(gameKind, ja.config).map((result): JobSubmitResult<StateAndView<S, V>> => ({ status: "done", result })),
        pollJob: () => {
          throw new Error("unreachable: new resolves synchronously (see submitJob above)");
        },
      };
      const eff = jobPollReduce(draft.newGame, { tag: "start" }, jobEnv);
      return eff ? eff.map((a): AppAction<S, M, V> => ({ tag: "newGame", action: { tag: "job", action: a }, config: ja.config })) : null;
    }
    const eff = jobPollReduce(
      draft.newGame,
      ja.action,
      unreachableJobEnv("unreachable: a forwarded newGame/job action never re-submits or polls"),
    );
    if (draft.newGame.status === "done" && draft.newGame.result) {
      // Fold the new position straight into a fresh tree/epoch in the same
      // reduction that observed "done" -- no separate "did newGame finish"
      // effect for GameShell to watch for. `position` isn't populated here;
      // it's re-derived by the `position/request` effect GameShell fires
      // whenever `tree.currentId` changes (which this assignment triggers).
      draft.tree = initialGameTree<S, M>(draft.newGame.result.state);
      draft.position = null;
      draft.move = initialJobPollState<StateAndView<S, V>>();
      draft.aiMove = initialJobPollState<AiMoveResult<S, M, V>>();
      draft.analysis = initialJobPollState<Analysis<M>>();
      draft.epoch += 1;
      // The request's own `config` (threaded through both effect maps above,
      // since the "start" -> "done" round trip re-dispatches through this
      // same branch) -- see state.ts's `config` doc for why this is what a
      // save file needs alongside `gameKind`/`tree`.
      draft.config = action.config;
      draft.newGame = initialJobPollState<StateAndView<S, V>>();
    }
    return eff ? eff.map((a): AppAction<S, M, V> => ({ tag: "newGame", action: { tag: "job", action: a }, config: action.config })) : null;
  }

  if (action.tag === "move") {
    const ja = action.action;
    if (ja.tag === "request") {
      const current = draft.tree.nodes[draft.tree.currentId];
      if (!current) return null;
      const { gameKind } = draft;
      const { move } = ja;
      const jobEnv: JobPollEnv<StateAndView<S, V>> = {
        submitJob: () =>
          env.apply<S, M, V>(gameKind, current.state, move).map((result): JobSubmitResult<StateAndView<S, V>> => ({ status: "done", result })),
        pollJob: () => {
          throw new Error("unreachable: apply resolves synchronously (see submitJob above)");
        },
      };
      const eff = jobPollReduce(draft.move, { tag: "start" }, jobEnv);
      return eff ? eff.map((a): AppAction<S, M, V> => ({ tag: "move", action: { tag: "job", action: a }, move })) : null;
    }
    const eff = jobPollReduce(
      draft.move,
      ja.action,
      unreachableJobEnv("unreachable: a forwarded move/job action never re-submits or polls"),
    );
    if (draft.move.status === "done" && draft.move.result && action.move !== undefined) {
      gameTreeReducer(draft.tree, { tag: "applyMove", move: action.move, state: draft.move.result.state }, undefined);
      // Same staleness reasoning as the "tree" branch above -- currentId
      // just changed, so both `analysis` and `position` must be dropped in
      // this same reduction to preserve the "`position` is null or matches
      // `currentId`" invariant.
      draft.analysis = initialJobPollState<Analysis<M>>();
      draft.position = null;
    }
    return eff ? eff.map((a): AppAction<S, M, V> => ({ tag: "move", action: { tag: "job", action: a }, move: action.move })) : null;
  }

  if (action.tag === "aiMove") {
    const ja = action.action;
    if (ja.tag === "request") {
      const current = draft.tree.nodes[draft.tree.currentId];
      if (!current) return null;
      const { gameKind } = draft;
      const { preset } = ja;
      const startEpoch = draft.epoch;
      const jobEnv: JobPollEnv<AiMoveResult<S, M, V>> = {
        submitJob: () =>
          env
            .aiMove<S, M, V>(gameKind, current.state, preset)
            .map((result): JobSubmitResult<AiMoveResult<S, M, V>> => ({ status: "done", result })),
        pollJob: () => {
          throw new Error("unreachable: ai_move resolves synchronously (see submitJob above)");
        },
      };
      const eff = jobPollReduce(draft.aiMove, { tag: "start" }, jobEnv);
      return eff
        ? eff.map((a): AppAction<S, M, V> => ({ tag: "aiMove", action: { tag: "job", action: a }, epoch: startEpoch }))
        : null;
    }
    // A response from a game that's since been replaced by "New Game" --
    // drop it rather than grafting a stale move onto the new game's tree.
    if (action.epoch !== undefined && action.epoch !== draft.epoch) return null;
    const eff = jobPollReduce(
      draft.aiMove,
      ja.action,
      unreachableJobEnv("unreachable: a forwarded aiMove/job action never re-submits or polls"),
    );
    if (draft.aiMove.status === "done" && draft.aiMove.result) {
      const { move, state } = draft.aiMove.result;
      gameTreeReducer(draft.tree, { tag: "applyMove", move, state }, undefined);
      // Same staleness reasoning as the "tree" branch above -- currentId
      // just changed, so both `analysis` and `position` must be dropped in
      // this same reduction to preserve the "`position` is null or matches
      // `currentId`" invariant. This is the fix for the race that could
      // otherwise auto-play an AI-controlled seat's move on top of a human
      // seat's turn: `GameShell`'s autoplay effect used to be able to read
      // `position` (and the `currentPlayer` derived from it) one node behind
      // `tree.currentId` immediately after this branch ran, since the
      // `position/request` effect it triggers only *starts* the re-fetch,
      // it doesn't resolve synchronously. Nulling `position` here means that
      // effect now correctly sees "no position yet" instead of stale data
      // describing the wrong player's turn.
      draft.analysis = initialJobPollState<Analysis<M>>();
      draft.position = null;
    }
    return eff
      ? eff.map((a): AppAction<S, M, V> => ({ tag: "aiMove", action: { tag: "job", action: a }, epoch: action.epoch }))
      : null;
  }

  if (action.tag === "analysis") {
    const ja = action.action;
    if (ja.tag === "request") {
      const current = draft.tree.nodes[draft.tree.currentId];
      if (!current) return null;
      const { gameKind } = draft;
      const { preset, budgetMs } = ja;
      const startEpoch = draft.epoch;
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
      return eff
        ? eff.map((a): AppAction<S, M, V> => ({ tag: "analysis", action: { tag: "job", action: a }, epoch: startEpoch }))
        : null;
    }
    if (action.epoch !== undefined && action.epoch !== draft.epoch) return null;
    const eff = jobPollReduce(
      draft.analysis,
      ja.action,
      unreachableJobEnv("unreachable: a forwarded analysis/job action never re-submits or polls"),
    );
    return eff
      ? eff.map((a): AppAction<S, M, V> => ({ tag: "analysis", action: { tag: "job", action: a }, epoch: action.epoch }))
      : null;
  }

  return null;
}
