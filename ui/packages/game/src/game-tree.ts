// game-tree.ts — Game-agnostic undo/redo/branch reducer.
// Generic over a game's state `S` and move `M`; knows nothing about any
// particular game, and nothing about the network -- callers already have the
// resulting state in hand (from an `env.apply()` call elsewhere) by the time
// they dispatch `applyMove`. Pure and synchronous: never returns an `Effect`,
// so its `env` parameter is unused (typed `unknown` rather than dropped, to
// keep the same three-argument shape every other `Reducer` in this codebase
// has, for `pullback`/`combine` compatibility).

import type { Effect } from "@mcts/core";
import type { SearchReport } from "./types.js";

export interface GameTreeNode<S, M> {
  id: string;
  state: S;
  /** `null` only for the root -- every other node was reached by a move. */
  move: M | null;
  /** Final search evidence that selected this node's move. Human moves and
   * the root have none; the report therefore describes the parent state. */
  search: SearchReport<M> | null;
  parentId: string | null;
  childIds: string[];
}

export interface GameTree<S, M> {
  nodes: Record<string, GameTreeNode<S, M>>;
  rootId: string;
  currentId: string;
  /** Monotonic node-id counter. A plain counter (not `crypto.randomUUID()`)
   * keeps this reducer deterministic and dependency-free, which matters for
   * save/load round-tripping identical ids and for tests
   * asserting on exact ids. */
  nextId: number;
}

export function initialGameTree<S, M>(rootState: S): GameTree<S, M> {
  const rootId = "n0";
  return {
    nodes: {
      [rootId]: { id: rootId, state: rootState, move: null, search: null, parentId: null, childIds: [] },
    },
    rootId,
    currentId: rootId,
    nextId: 1,
  };
}

export type GameTreeAction<S, M> =
  | { tag: "applyMove"; move: M; state: S; search?: SearchReport<M> | null }
  | { tag: "undo" }
  | { tag: "redo"; childId?: string }
  | { tag: "jumpTo"; id: string }
  | { tag: "deleteBranch"; id: string };

/** Structural move equality via JSON comparison -- the only option available
 * without requiring every game's move type to carry its own `Eq`. Sound for
 * every move type in this codebase (plain tuples/enums that serialize
 * losslessly, same values this package already round-trips over the wire),
 * not sound in general for types with `undefined` fields or `Map`s. Exported
 * for `AnalysisPanel`/`GameShell` to match an analysis
 * candidate's move against `suggested_move`/`hoveredMove` -- same soundness
 * argument applies there. */
export function moveEquals<M>(a: M, b: M): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** True when `tree.currentId` has no children -- the live frontier of play,
 * as opposed to a node reached by navigating back into history (undo/redo/
 * jumpTo). `GameShell`'s autoplay effect gates on this before driving the
 * next AI move: without it, navigating to an ancestor node whose mover
 * happens to be AI-controlled immediately re-triggered an aiMove from there,
 * which either replayed the existing child (undo looking like it "snapped
 * back" to the last move) or forked a new one (a history click going
 * nowhere the user could see). */
export function isFrontier<S, M>(tree: GameTree<S, M>): boolean {
  return tree.nodes[tree.currentId]?.childIds.length === 0;
}

export function gameTreeReducer<S, M>(
  draft: GameTree<S, M>,
  action: GameTreeAction<S, M>,
  _env: unknown,
): Effect<GameTreeAction<S, M>> | null {
  switch (action.tag) {
  case "applyMove": {
    const current = draft.nodes[draft.currentId];
    if (!current) return null;
    const existingChildId = current.childIds.find((id) => {
      const child = draft.nodes[id];
      return child !== undefined && child.move !== null && moveEquals(child.move, action.move);
    });
    if (existingChildId !== undefined) {
      const child = draft.nodes[existingChildId];
      if (child && child.search === null && action.search !== undefined && action.search !== null) {
        child.search = action.search;
      }
      draft.currentId = existingChildId;
      return null;
    }
    const id = `n${draft.nextId}`;
    draft.nextId += 1;
    draft.nodes[id] = {
      id,
      state: action.state,
      move: action.move,
      search: action.search ?? null,
      parentId: current.id,
      childIds: [],
    };
    current.childIds.push(id);
    draft.currentId = id;
    return null;
  }
  case "undo": {
    const current = draft.nodes[draft.currentId];
    if (current?.parentId) draft.currentId = current.parentId;
    return null;
  }
  case "redo": {
    const current = draft.nodes[draft.currentId];
    if (!current || current.childIds.length === 0) return null;
    const targetId = action.childId ?? current.childIds[current.childIds.length - 1];
    if (targetId !== undefined && current.childIds.includes(targetId)) draft.currentId = targetId;
    return null;
  }
  case "jumpTo": {
    if (draft.nodes[action.id]) draft.currentId = action.id;
    return null;
  }
  case "deleteBranch": {
    const target = draft.nodes[action.id];
    if (!target || target.id === draft.rootId) return null; // can't delete the root
    const parentId = target.parentId;
    const toDelete: string[] = [];
    const stack = [target.id];
    while (stack.length > 0) {
      const id = stack.pop();
      if (id === undefined) break;
      toDelete.push(id);
      const node = draft.nodes[id];
      if (node) stack.push(...node.childIds);
    }
    if (parentId) {
      const parent = draft.nodes[parentId];
      if (parent) parent.childIds = parent.childIds.filter((id) => id !== target.id);
    }
    const deletedSet = new Set(toDelete);
    for (const id of toDelete) delete draft.nodes[id];
    if (deletedSet.has(draft.currentId)) draft.currentId = parentId ?? draft.rootId;
    return null;
  }
  }
}
