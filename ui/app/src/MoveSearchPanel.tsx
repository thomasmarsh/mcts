// MoveSearchPanel.tsx — Presents search evidence retained on the currently
// selected history node. The report belongs to the move that reached that
// node, so its formatter receives the parent state rather than the node's
// resulting state.

import { type Component, createMemo, Show } from "solid-js";
import type { GameTree } from "@mcts/game";
import { SearchInspector } from "@mcts/search-inspector";

type S = unknown;
type M = unknown;

export const MoveSearchPanel: Component<{
  tree: GameTree<S, M>;
  formatMove?: (move: M, before: S) => string;
}> = (props) => {
  const current = createMemo(() => props.tree.nodes[props.tree.currentId]);
  const parent = createMemo(() => {
    const node = current();
    return node?.parentId ? props.tree.nodes[node.parentId] : undefined;
  });

  return (
    <section id="move-search-panel" aria-label="Selected move search">
      <Show
        when={current()?.move === null ? undefined : parent()}
        fallback={<p role="status">The starting position has no selected move.</p>}
      >
        {(parentNode) => (
          <>
            <h2>Search that selected this move</h2>
            <Show
              when={current()?.search}
              fallback={
                <p role="status">
                  No retained search report for this move. It was played by a human or comes from
                  legacy history.
                </p>
              }
            >
              {(report) => (
                <SearchInspector
                  report={report()}
                  before={parentNode().state}
                  formatMove={props.formatMove}
                />
              )}
            </Show>
          </>
        )}
      </Show>
    </section>
  );
};
