// state.ts — App state: the single state tree shape (mirrors pb/ui/app/src/state.ts's
// flat-feature-slices convention). Generic over a game's state `S` and move
// `M` -- packages/druid's concrete types instantiate this in Session 4; this
// package itself never names a concrete game.

import { initialJobPollState, type JobPollState } from "@mcts/core";
import { initialGameTree, type GameTree } from "./game-tree.js";
import type { AiMoveResult, Analysis } from "./types.js";

/** Who controls each player -- keyed by the game's own player id (e.g.
 * Druid's "Black"/"White", tic-tac-toe's "X"/"O" in Session 8) rather than a
 * fixed two-seat shape, so this stays game-agnostic. Placeholder shape for
 * Session 4+'s seat-picker UI; nothing in this session reads or writes it
 * beyond `initialAppState`'s default. */
export type SeatsState = Record<string, "human" | "ai">;

/** Misc UI-only state that doesn't belong to the tree or a job-poll slice.
 * Placeholder for Session 5/6's preset picker; grows as those sessions need
 * fields, not speculatively here. */
export interface UiState {
  selectedPreset: string | null;
}

export interface AppState<S, M> {
  gameKind: string;
  tree: GameTree<S, M>;
  aiMove: JobPollState<AiMoveResult<S, M>>;
  analysis: JobPollState<Analysis<M>>;
  seats: SeatsState;
  ui: UiState;
}

export function initialAppState<S, M>(gameKind: string, rootState: S): AppState<S, M> {
  return {
    gameKind,
    tree: initialGameTree<S, M>(rootState),
    aiMove: initialJobPollState<AiMoveResult<S, M>>(),
    analysis: initialJobPollState<Analysis<M>>(),
    seats: {},
    ui: { selectedPreset: null },
  };
}
