// tests/game-tree.test.ts — Tests for the pure GameTree undo/redo/branch reducer.

import { describe, it, expect } from "vitest";
import { gameTreeReducer, initialGameTree } from "../src/game-tree.js";

// Test-only state/move types: state is just "how many moves deep", move is a
// label -- GameTree never inspects either, so any shape does.
type S = number;
type M = string;

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

  it("deleteBranch cannot delete the root", () => {
    const tree = initialGameTree<S, M>(0);
    gameTreeReducer(tree, { tag: "deleteBranch", id: tree.rootId }, undefined);
    expect(tree.nodes[tree.rootId]).toBeDefined();
    expect(tree.currentId).toBe(tree.rootId);
  });
});
