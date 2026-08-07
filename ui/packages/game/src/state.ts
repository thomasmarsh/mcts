// state.ts — App state: the single state tree shape (mirrors pb/ui/app/src/state.ts's
// flat-feature-slices convention). Generic over a game's state `S`, move `M`,
// and view `V` -- packages/druid's concrete types instantiate this;
// this package itself never names a concrete game.

import { initialJobPollState, type JobPollState } from "@mcts/core";
import { initialGameTree, type GameTree } from "./game-tree.js";
import type { AiMoveResult, AiPresetInfo, Analysis, StateAndView } from "./types.js";

/** Who controls each player -- keyed by the game's own player id (e.g.
 * Druid's "Black"/"White", tic-tac-toe's "X"/"O") rather than a
 * fixed two-seat shape, so this stays game-agnostic. A missing entry means
 * "human" (the default seat for any player nothing has set yet). */
export type SeatsState = Record<string, "human" | string>;

/** Misc UI-only state that doesn't belong to the tree or a job-poll slice.
 * Placeholder for the preset picker; grows as needed,
 * not speculatively here. */
export interface UiState {
  selectedPreset: string | null;
}

/** View + legal moves for `nodeId` -- re-derived by `GameShell`
 * every time `tree.currentId` changes, since `GameTree` itself only stores a
 * node's raw `S`, not its `V`/legal-move-list (see `reducer.ts`'s `position`
 * handling for why: `new`/`apply`/`ai_move` already return `view` for free,
 * but `undo`/`redo`/`jumpTo` are pure client-side moves with no accompanying
 * server round trip, so there's no single call site to grab it from -- one
 * "derive for whatever `currentId` is now" effect covers every case
 * uniformly, at the cost of one redundant local request on the moved-and-
 * already-had-the-view path). `nodeId` guards a `loaded` result actually
 * matching the node it was requested for -- see `reducer.ts`. */
export interface PositionInfo<V, M> {
  nodeId: string;
  view: V;
  legalMoves: M[];
}

export interface AppState<S, M, V = unknown> {
  gameKind: string;
  /** Bumped once per completed `newGame`. Stamped onto in-flight `aiMove`/
   * `analysis` requests so a response arriving after a *new* game has
   * started (a real request, not a bug -- Master's search can take 8s, and
   * "New Game" deliberately stays clickable mid-AI-turn, same as app.js) gets
   * dropped instead of grafting an old game's move onto the new one. See
   * `reducer.ts`'s `aiMove`/`analysis` handling. */
  epoch: number;
  /** The config the current tree's root was created from -- along with
   * `gameKind` and `tree`, exactly what a save file needs (see
   * `save-load.ts`). Set in the same reduction that observes a completed
   * `newGame` or handles a `load` action; `null` for the pre-bootstrap
   * placeholder root (see `App.tsx`'s header comment). */
  config: unknown;
  tree: GameTree<S, M>;
  position: PositionInfo<V, M> | null;
  /** Static per-kind metadata (`GameShell`'s seat pickers/AI-move preset
   * list), fetched once per `gameKind` -- unlike `position`, this never
   * changes as the tree is navigated, only when `gameKind` itself does. */
  aiPresets: JobPollState<AiPresetInfo[]>;
  newGame: JobPollState<StateAndView<S, V>>;
  move: JobPollState<StateAndView<S, V>>;
  aiMove: JobPollState<AiMoveResult<S, M, V>>;
  analysis: JobPollState<Analysis<M>>;
  seats: SeatsState;
  ui: UiState;
}

export function initialAppState<S, M, V = unknown>(gameKind: string, rootState: S): AppState<S, M, V> {
  return {
    gameKind,
    epoch: 0,
    config: null,
    tree: initialGameTree<S, M>(rootState),
    position: null,
    aiPresets: initialJobPollState<AiPresetInfo[]>(),
    newGame: initialJobPollState<StateAndView<S, V>>(),
    move: initialJobPollState<StateAndView<S, V>>(),
    aiMove: initialJobPollState<AiMoveResult<S, M, V>>(),
    analysis: initialJobPollState<Analysis<M>>(),
    seats: {},
    ui: { selectedPreset: null },
  };
}
