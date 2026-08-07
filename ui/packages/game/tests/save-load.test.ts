// tests/save-load.test.ts — Tests for save-load.ts's serialize/parse round
// trip and malformed-input rejection (PLAN-UI.md session 7's validation
// bullet: "save a game with several branches, reload, confirm the full tree
// -- not just the mainline -- survives intact").

import { describe, it, expect } from "vitest";
import { gameTreeReducer, initialGameTree } from "../src/game-tree.js";
import { parseSave, serializeSave, SAVE_FORMAT_VERSION } from "../src/save-load.js";

// Test-only state/move types -- save-load.ts never inspects their shape.
type S = number;
type M = string;

function branchedTree() {
  const tree = initialGameTree<S, M>(0);
  gameTreeReducer(tree, { tag: "applyMove", move: "a", state: 1 }, undefined);
  gameTreeReducer(tree, { tag: "applyMove", move: "c", state: 3 }, undefined);
  gameTreeReducer(tree, { tag: "undo" }, undefined);
  gameTreeReducer(tree, { tag: "undo" }, undefined);
  gameTreeReducer(tree, { tag: "applyMove", move: "b", state: 2 }, undefined);
  gameTreeReducer(tree, { tag: "undo" }, undefined);
  return tree;
}

describe("serializeSave / parseSave", () => {
  it("round-trips a tree with multiple branches intact, not just the mainline", () => {
    const tree = branchedTree();
    const json = serializeSave("druid", { size: { w: 5, h: 5 } }, tree);
    const loaded = parseSave<S, M>(json);

    expect(loaded.formatVersion).toBe(SAVE_FORMAT_VERSION);
    expect(loaded.gameKind).toBe("druid");
    expect(loaded.config).toEqual({ size: { w: 5, h: 5 } });
    expect(loaded.tree).toEqual(tree);
    expect(Object.keys(loaded.tree.nodes)).toHaveLength(4); // root + a + b + c
    expect(loaded.tree.nodes[loaded.tree.rootId]?.childIds).toHaveLength(2); // both "a" and "b"
  });

  it("preserves a null config", () => {
    const tree = initialGameTree<S, M>(0);
    const json = serializeSave("ttt", null, tree);
    expect(parseSave<S, M>(json).config).toBeNull();
  });

  it("rejects invalid JSON", () => {
    expect(() => parseSave("not json")).toThrow(/valid JSON/);
  });

  it("rejects a non-object payload", () => {
    expect(() => parseSave("42")).toThrow(/JSON object/);
  });

  it("rejects a mismatched format version", () => {
    const bad = JSON.stringify({ formatVersion: 999, gameKind: "druid", config: null, tree: initialGameTree<S, M>(0) });
    expect(() => parseSave(bad)).toThrow(/format version/);
  });

  it("rejects a missing gameKind", () => {
    const bad = JSON.stringify({ formatVersion: SAVE_FORMAT_VERSION, config: null, tree: initialGameTree<S, M>(0) });
    expect(() => parseSave(bad)).toThrow(/gameKind/);
  });

  it("rejects a malformed tree (missing rootId)", () => {
    const bad = JSON.stringify({
      formatVersion: SAVE_FORMAT_VERSION,
      gameKind: "druid",
      config: null,
      tree: { nodes: {}, currentId: "n0", nextId: 1 },
    });
    expect(() => parseSave(bad)).toThrow(/tree/);
  });

  it("rejects a tree whose currentId isn't a real node", () => {
    const tree = initialGameTree<S, M>(0);
    const bad = JSON.stringify({
      formatVersion: SAVE_FORMAT_VERSION,
      gameKind: "druid",
      config: null,
      tree: { ...tree, currentId: "does-not-exist" },
    });
    expect(() => parseSave(bad)).toThrow(/tree/);
  });

  it("rejects a node missing childIds", () => {
    const tree = initialGameTree<S, M>(0);
    const nodes = tree.nodes as Record<string, unknown>;
    delete (nodes[tree.rootId] as { childIds?: unknown }).childIds;
    const bad = JSON.stringify({ formatVersion: SAVE_FORMAT_VERSION, gameKind: "druid", config: null, tree });
    expect(() => parseSave(bad)).toThrow(/tree/);
  });
});
