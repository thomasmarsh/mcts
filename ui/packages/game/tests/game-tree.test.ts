// tests/game-tree.test.ts — Tests for the pure GameTree undo/redo/branch reducer.

import { describe, it, expect } from "vitest";
import { gameTreeReducer, initialGameTree, isFrontier } from "../src/game-tree.js";
import type { SearchReport } from "../src/types.js";

// Test-only state/move types: state is just "how many moves deep", move is a
// label -- GameTree never inspects either, so any shape does.
type S = number;
type M = string;

function searchReport(selectedAction: M): SearchReport<M> {
  return {
    status: "available",
    schema_version: 1,
    reason: null,
    elapsed_seconds: 0.5,
    iteration_limit: 100,
    time_limit_seconds: null,
    completed_iterations: 100,
    termination: "iterations",
    selected_action: selectedAction,
    actions: [],
    principal_variation: [selectedAction],
    root_visits: 100,
    tree_nodes: 101,
    mean_depth: 2,
    max_depth: 4,
    graph_mode: "tree",
    tt_reads: 0,
    tt_writes: 0,
    tt_hits: 0,
    tt_hit_ratio: null,
    iterations_per_second: 200,
    warnings: [],
  };
}

describe("gameTreeReducer", () => {
  it("applyMove creates a new child and advances current", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);

    expect(tree.currentId).not.toBe(tree.rootId);
    const current = tree.nodes[tree.currentId];
    expect(current?.state).toBe(1);
    expect(current?.move).toBe("a");
    expect(current?.parentId).toBe(tree.rootId);
    expect(tree.nodes[tree.rootId]?.childIds).toEqual([tree.currentId]);
  });

  it("keeps root and human moves report-free while retaining an AI move's report", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "human", state: 1 }, undefined);
    const humanId = tree.currentId;
    gameTreeReducer(tree, { tag: "undo" }, undefined);
    const report = searchReport("ai");
    gameTreeReducer(tree, { tag: "applyMove", move: "ai", state: 1, search: report }, undefined);

    expect(tree.nodes[tree.rootId]?.search).toBeNull();
    expect(tree.nodes[humanId]?.search).toBeNull();
    expect(tree.nodes[tree.currentId]?.search).toEqual(report);
  });

  it("undo/redo round-trips", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    const afterFirst = tree.currentId;

    gameTreeReducer(tree, { tag: "undo" }, undefined);
    expect(tree.currentId).toBe(tree.rootId);

    gameTreeReducer(tree, { tag: "redo" }, undefined);
    expect(tree.currentId).toBe(afterFirst);
  });

  it("undo at the root is a no-op", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "undo" }, undefined);
    expect(tree.currentId).toBe(tree.rootId);
  });

  it("redo with no children is a no-op", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "redo" }, undefined);
    expect(tree.currentId).toBe(tree.rootId);
  });

  it("reuses the existing child branch when replaying an already-explored move", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    const firstChildId = tree.currentId;

    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);

    expect(tree.currentId).toBe(firstChildId);
    expect(tree.nodes[tree.rootId]?.childIds).toEqual([firstChildId]);
    expect(Object.keys(tree.nodes)).toHaveLength(2);
  });

  it("fills missing evidence when reusing a child without replacing existing evidence", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    const childId = tree.currentId;
    const firstReport = searchReport("a");
    const laterReport = { ...searchReport("a"), completed_iterations: 200 };

    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1, search: firstReport }, undefined);
    expect(tree.currentId).toBe(childId);
    expect(tree.nodes[childId]?.search).toEqual(firstReport);

    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1, search: laterReport }, undefined);
    expect(tree.nodes[childId]?.search).toEqual(firstReport);
  });

  it("creates a sibling branch for a different move from the same node", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "applyMove", move: "b", state: 2 }, undefined);

    expect(tree.nodes[tree.rootId]?.childIds).toHaveLength(2);
  });

  it("redo defaults to the most recently added child", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "applyMove", move: "b", state: 2 }, undefined);
    const secondChildId = tree.currentId;

    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "redo" }, undefined);

    expect(tree.currentId).toBe(secondChildId);
  });

  it("redo honors an explicit childId", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    const firstChildId = tree.currentId;
    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "applyMove", move: "b", state: 2 }, undefined);

    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "redo", childId: firstChildId }, undefined);

    expect(tree.currentId).toBe(firstChildId);
  });

  it("jumpTo moves current to an arbitrary explored node", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    const childId = tree.currentId;
    gameTreeReducer(tree, { tag: "undo" }, undefined);

    gameTreeReducer(tree, { tag: "jumpTo", id: childId }, undefined);
    expect(tree.currentId).toBe(childId);
  });

  it("jumpTo to an unknown id is a no-op", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "jumpTo", id: "does-not-exist" }, undefined);
    expect(tree.currentId).toBe(tree.rootId);
  });

  it("deleteBranch removes the subtree and reparents an orphaned current to the deleted node's parent", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    const childId = tree.currentId;
    gameTreeReducer(tree, { tag: "applyMove", move: "b", state: 2 }, undefined);
    const grandchildId = tree.currentId;

    gameTreeReducer(tree, { tag: "deleteBranch", id: childId }, undefined);

    expect(tree.nodes[childId]).toBeUndefined();
    expect(tree.nodes[grandchildId]).toBeUndefined();
    expect(tree.currentId).toBe(tree.rootId);
    expect(tree.nodes[tree.rootId]?.childIds).toEqual([]);
  });

  it("deleteBranch leaves current untouched when current is outside the deleted subtree", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    const firstChildId = tree.currentId;
    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "applyMove", move: "b", state: 2 }, undefined);
    const secondChildId = tree.currentId;

    gameTreeReducer(tree, { tag: "deleteBranch", id: firstChildId }, undefined);

    expect(tree.nodes[firstChildId]).toBeUndefined();
    expect(tree.currentId).toBe(secondChildId);
  });

  it("preserves retained evidence through navigation and deletion of another branch", () => {
    const tree = initialGameTree<S, M>(0);
    const report = searchReport("a");
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1, search: report }, undefined);
    const aiChildId = tree.currentId;

    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "redo", childId: aiChildId }, undefined);
    expect(tree.nodes[tree.currentId]?.search).toEqual(report);

    gameTreeReducer(tree, { tag: "undo" }, undefined);
    gameTreeReducer(tree, { tag: "applyMove", move: "human", state: 1 }, undefined);
    const humanChildId = tree.currentId;
    gameTreeReducer(tree, { tag: "deleteBranch", id: humanChildId }, undefined);
    gameTreeReducer(tree, { tag: "jumpTo", id: aiChildId }, undefined);

    expect(tree.nodes[aiChildId]?.search).toEqual(report);
  });

  it("deleteBranch cannot delete the root", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "deleteBranch", id: tree.rootId }, undefined);
    expect(tree.nodes[tree.rootId]).toBeDefined();
    expect(tree.currentId).toBe(tree.rootId);
  });
});

// A regression suite for the bug `isFrontier` fixes: `GameShell`'s autoplay
// effect used to fire an aiMove whenever it was an AI-controlled seat's
// turn, with no check for whether `currentId` was actually the live tip of
// play -- so undo/redo/jumpTo into history, landing on a node whose mover
// happened to be AI-controlled, immediately re-triggered an aiMove *from
// that node*.
describe("isFrontier", () => {
  it("is true at a leaf with no children (a fresh root, or the tip of play)", () => {
    const tree = initialGameTree<S, M>(0);
    expect(isFrontier(tree)).toBe(true);
  });

  it("stays true after applyMove advances current to the new leaf", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    expect(isFrontier(tree)).toBe(true);
  });

  it("is false once current has been navigated back to a node with children", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    gameTreeReducer(tree, { tag: "undo" }, undefined);
    expect(isFrontier(tree)).toBe(false);
  });

  it("is false after jumpTo lands on an ancestor with children", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
    gameTreeReducer(tree, { tag: "jumpTo", id: tree.rootId }, undefined);
    expect(isFrontier(tree)).toBe(false);
  });

  it("is false for a currentId that doesn't exist in the tree", () => {
    const tree = initialGameTree<S, M>(0);
    tree.currentId = "does-not-exist";
    expect(isFrontier(tree)).toBe(false);
  });
});
