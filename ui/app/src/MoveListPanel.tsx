// MoveListPanel.tsx — Game-tree navigator (PLAN-UI.md session 5): a linear
// list of moves along the root-to-current path, click-to-jump
// (`GameTree.jumpTo`), and a branch indicator on any path node with more
// than one explored child. A full SGF-style tree widget is explicitly out of
// scope for v1 (see the plan) -- this stays a flat list plus a small popover
// for picking among a fork's alternates.
//
// Per PLAN-UI.md's hard rule, this component never touches the network --
// its only output is `onJump(id)`, which `GameShell` wires to a
// `tree/jumpTo` dispatch.

import { type Component, createMemo, createSignal, For, Show } from "solid-js";
import type { GameTree, GameTreeNode } from "@mcts/game";

type S = unknown;
type M = unknown;

interface PathRow {
  node: GameTreeNode<S, M>;
  ply: number;
  label: string;
}

export const MoveListPanel: Component<{
  tree: GameTree<S, M>;
  formatMove?: (move: M, before: S) => string;
  onJump: (id: string) => void;
}> = (props) => {
  const [openBranchId, setOpenBranchId] = createSignal<string | null>(null);

  const path = createMemo((): PathRow[] => {
    const chain: GameTreeNode<S, M>[] = [];
    let node: GameTreeNode<S, M> | undefined = props.tree.nodes[props.tree.currentId];
    while (node) {
      chain.push(node);
      node = node.parentId ? props.tree.nodes[node.parentId] : undefined;
    }
    chain.reverse();
    return chain.map((n, i) => ({
      node: n,
      ply: i,
      label: i === 0 ? "Start" : (props.formatMove?.(n.move as M, chain[i - 1]!.state) ?? JSON.stringify(n.move)),
    }));
  });

  function jump(id: string): void {
    setOpenBranchId(null);
    props.onJump(id);
  }

  function toggleBranches(id: string): void {
    setOpenBranchId((cur) => (cur === id ? null : id));
  }

  return (
    <div id="move-list">
      <ol>
        <For each={path()}>
          {(row) => (
            <li>
              <button class="move-row" classList={{ current: row.node.id === props.tree.currentId }} onClick={() => jump(row.node.id)}>
                <Show when={row.ply > 0}>
                  <span class="ply">{row.ply}.</span>
                </Show>
                <span class="label">{row.label}</span>
              </button>
              <Show when={row.node.childIds.length > 1}>
                <button
                  class="branch-toggle"
                  title={`${row.node.childIds.length} branches from here`}
                  onClick={() => toggleBranches(row.node.id)}
                >
                  ⑂ {row.node.childIds.length}
                </button>
                <Show when={openBranchId() === row.node.id}>
                  <ul class="branch-list">
                    <For each={row.node.childIds}>
                      {(childId) => {
                        const child = props.tree.nodes[childId];
                        if (!child) return null;
                        const label = props.formatMove?.(child.move as M, row.node.state) ?? JSON.stringify(child.move);
                        return (
                          <li>
                            <button class="branch-option" classList={{ active: path().some((r) => r.node.id === childId) }} onClick={() => jump(childId)}>
                              {label}
                            </button>
                          </li>
                        );
                      }}
                    </For>
                  </ul>
                </Show>
              </Show>
            </li>
          )}
        </For>
      </ol>
    </div>
  );
};
