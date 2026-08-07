// layers.ts — Pure reconstruction of Druid's *physical* piece stack from a
// move history (PLAN-UI.md session 4). Ported from server/static/app.js's
// `initLayers`/`applyMoveToLayers`, but as a pure function of
// `MoveStep<GameState, Move>[]` (available for free from `GameTree`'s
// root-to-current path) instead of app.js's session-long replay-as-you-go
// bookkeeping. This is strictly more correct than app.js: since a fresh
// Druid game always starts from an empty board (no "load a game already in
// progress" path exists yet -- see PLAN-UI.md's decision 1 and session 7),
// the full history is always available, so there's no need for app.js's
// "seed from whatever the current board looks like and guess" fallback for
// pre-existing stacks.
//
// The server's `Square` model only stores each cell's current top
// owner/height and overwrites the middle cell of a bridging lintel to match
// the endpoints' height -- it has no memory of what was physically built
// underneath. `layers[cellIndex]` reconstructs that memory: an array of
// per-level entries, bottom to top, where each entry is an owner (a real
// placed unit cube), `null` (a gap -- empty air under a bridging lintel), or
// a `{ beam }` marker (a level that's part of a merged lintel beam mesh,
// rendered once -- see `beams` -- rather than as a separate unit cube per
// cell).

import type { MoveStep } from "@mcts/game";
import type { GameState, Move, Player, Size } from "./types.js";

export type LayerEntry = Player | null | { beam: number };

export interface Beam {
  level: number;
  orientation: "Horizontal" | "Vertical";
  cells: [number, number, number];
  color: Player;
}

export interface StackModel {
  layers: LayerEntry[][];
  beams: Beam[];
}

/** The 1 or 3 board cells a move occupies, mirroring app.js's `footprintFor`. */
export function footprintFor(move: Move, w: number): number[] {
  const [piece, index] = move;
  if (piece === "Sarsen") return [index];
  const step = piece.Lintel === "Horizontal" ? 1 : w;
  return [index, index + step, index + 2 * step];
}

/** Replays `history` into a physical stack model. `history[i].before.player`
 * is the mover for `history[i].move` (the player *to move* immediately
 * before that move was applied) -- deriving the mover this way, rather than
 * assuming Black/White strictly alternate, stays correct even if a future
 * rule variant ever allows a pass. */
export function buildStackModel(size: Size, history: MoveStep<GameState, Move>[]): StackModel {
  const layers: LayerEntry[][] = Array.from({ length: size.w * size.h }, () => []);
  const beams: Beam[] = [];
  let nextBeamId = 0;

  for (const step of history) {
    const owner = step.before.player;
    const [piece, index] = step.move;

    if (piece === "Sarsen") {
      layers[index]!.push(owner);
      continue;
    }

    const orientation = piece.Lintel;
    const cells = footprintFor(step.move, size.w) as [number, number, number];
    const level = layers[cells[0]]!.length;
    const beamId = nextBeamId++;

    layers[cells[0]]!.push({ beam: beamId });
    layers[cells[2]]!.push({ beam: beamId });
    while (layers[cells[1]]!.length < level) layers[cells[1]]!.push(null);
    layers[cells[1]]!.push({ beam: beamId });

    beams.push({ level, orientation, cells, color: owner });
  }

  return { layers, beams };
}
