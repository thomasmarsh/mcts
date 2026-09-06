// save-load.ts — Client-side save/load: serializes
// {formatVersion, gameKind, config, tree} to/from JSON text. Fully local:
// the client already holds the full game tree -- no `env` call,
// no `fetch`. Lives here rather than app/src because it's game-agnostic
// (generic over S/M, same as game-tree.ts); the file download/upload DOM
// mechanics (anchor `download`, `<input type=file>`) belong to app/src's
// SaveLoadPanel instead.

import type { GameTree, GameTreeNode } from "./game-tree.js";

/** The omitted `search` field from older nodes is upgraded to `null`, so this
 * remains compatible with existing v1 save files. */
export const SAVE_FORMAT_VERSION = 1;

export interface SaveFile<S, M> {
  formatVersion: number;
  gameKind: string;
  config: unknown;
  tree: GameTree<S, M>;
}

export function serializeSave<S, M>(
  gameKind: string,
  config: unknown,
  tree: GameTree<S, M>,
): string {
  const file: SaveFile<S, M> = { formatVersion: SAVE_FORMAT_VERSION, gameKind, config, tree };
  return JSON.stringify(file, null, 2);
}

function isGameTreeNode(v: unknown): v is GameTreeNode<unknown, unknown> {
  if (!v || typeof v !== "object") return false;
  const n = v as Record<string, unknown>;
  return (
    typeof n.id === "string" &&
    "state" in n &&
    "move" in n &&
    (n.parentId === null || typeof n.parentId === "string") &&
    Array.isArray(n.childIds) &&
    n.childIds.every((c) => typeof c === "string")
  );
}

function isGameTree(v: unknown): v is GameTree<unknown, unknown> {
  if (!v || typeof v !== "object") return false;
  const t = v as Record<string, unknown>;
  if (
    typeof t.rootId !== "string" ||
    typeof t.currentId !== "string" ||
    typeof t.nextId !== "number"
  )
    return false;
  if (!t.nodes || typeof t.nodes !== "object") return false;
  const nodes = t.nodes as Record<string, unknown>;
  if (!Object.values(nodes).every(isGameTreeNode)) return false;
  return t.rootId in nodes && t.currentId in nodes;
}

/** Parses and validates a save file's JSON text, throwing a descriptive
 * `Error` on any structural mismatch -- callers (app/src's SaveLoadPanel)
 * turn that into a UI-visible message rather than crashing on a corrupt or
 * hand-edited file. Only checks `GameTree`'s own shape (node ids/parent/
 * child links, `rootId`/`currentId` actually present in `nodes`); doesn't
 * validate `state`/`move` payloads against a specific game's real `S`/`M`,
 * same as the rest of this package staying generic over both. */
export function parseSave<S, M>(text: string): SaveFile<S, M> {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("Not valid JSON.");
  }
  if (!data || typeof data !== "object") throw new Error("Save file must be a JSON object.");
  const d = data as Record<string, unknown>;
  if (d.formatVersion !== SAVE_FORMAT_VERSION) {
    throw new Error(
      `Unsupported save format version ${JSON.stringify(d.formatVersion)} (expected ${SAVE_FORMAT_VERSION}).`,
    );
  }
  if (typeof d.gameKind !== "string" || d.gameKind.length === 0) {
    throw new Error('Missing or invalid "gameKind".');
  }
  if (!isGameTree(d.tree)) {
    throw new Error('Missing or malformed "tree".');
  }
  const tree = d.tree as GameTree<S, M>;
  const nodes = Object.fromEntries(
    Object.entries(tree.nodes).map(([id, node]) => [id, { ...node, search: node.search ?? null }]),
  ) as Record<string, GameTreeNode<S, M>>;
  return {
    formatVersion: d.formatVersion,
    gameKind: d.gameKind,
    config: d.config,
    tree: { ...tree, nodes },
  };
}
