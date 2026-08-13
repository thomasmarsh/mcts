// CongoRenderer.tsx — 7×7 board renderer for Congo.
//
// Uses two-click interaction (click source piece, then click destination),
// same protocol as ChessVariantRenderer, because multiple pieces can reach
// the same destination square. Unlike ChessVariantRenderer's `[src, dst]`
// tuple, a Congo `Move` already carries its own `captures`, which matters
// here: a Monkey's jump-chain move can capture pieces on squares far from
// its destination, so "is this a capture" and "which squares light up as
// captured" both come from `move.captures`, never from occupancy of `to`.
//
// Board orientation: the engine's mailbox has row 0 = Black's home rank
// (see `games/congo/src/lib.rs`'s module doc comment) and row 6 = White's,
// so the display grid maps 1:1 onto the engine index (row-major, no flip)
// with Black's castle at the top of the screen.
//
// Known simplification: Congo's rules allow a Monkey jump-chain to revisit
// squares, so in rare geometries two distinct legal moves can share the same
// (from, to) pair with different capture sets. The two-click UI can only
// name a destination square, not a capture path, so `moveLookup` below just
// keeps the first match for a given (from, to) key.

import { type Component, createEffect, createMemo, createSignal, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import type { Cell, GameState, GameView, Move, PieceCode } from "./types.js";
import { RIVER_ROW, SIZE } from "./types.js";
import "./congo.css";

import lionIcon from "./pieces/lion.svg";
import elephantIcon from "./pieces/elephant.svg";
import giraffeIcon from "./pieces/giraffe.svg";
import crocodileIcon from "./pieces/crocodile.svg";
import zebraIcon from "./pieces/zebra.svg";
import monkeyIcon from "./pieces/monkey.svg";
import pawnIcon from "./pieces/pawn.svg";

const PIECE_ICON: Record<PieceCode, string> = {
  giraffe: giraffeIcon,
  monkey: monkeyIcon,
  elephant: elephantIcon,
  lion: lionIcon,
  crocodile: crocodileIcon,
  zebra: zebraIcon,
  pawn: pawnIcon,
  superpawn: pawnIcon,
};

function inCastle(row: number, col: number): boolean {
  return (row <= 2 || row >= 4) && col >= 2 && col <= 4;
}

function zoneOf(row: number, col: number): "river" | "castle" | "grass" {
  if (row === RIVER_ROW) return "river";
  if (inCastle(row, col)) return "castle";
  return "grass";
}

export const CongoRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
  /** Engine index of the currently selected source piece, or null. */
  const [selectedSource, setSelectedSource] = createSignal<number | null>(null);

  const movesBySrc = createMemo(() => {
    const map = new Map<number, Move[]>();
    for (const mv of props.legalMoves) {
      const list = map.get(mv.from) ?? [];
      list.push(mv);
      map.set(mv.from, list);
    }
    return map;
  });

  const movablePieceSet = createMemo(() => new Set(movesBySrc().keys()));

  const legalDstForSelected = createMemo(() => {
    const src = selectedSource();
    if (src === null) return new Set<number>();
    const moves = movesBySrc().get(src);
    if (!moves) return new Set<number>();
    return new Set(moves.map((mv) => mv.to));
  });

  /** (src,dst) → move. See the module doc comment for the rare-ambiguity
   * caveat this collapses. */
  const moveLookup = createMemo(() => {
    const map = new Map<string, Move>();
    for (const mv of props.legalMoves) {
      const key = `${mv.from},${mv.to}`;
      if (!map.has(key)) map.set(key, mv);
    }
    return map;
  });

  const overlayByDst = createMemo(() => {
    const map = new Map<number, { visitShare: number; isProven: boolean; isSuggested: boolean }>();
    for (const entry of props.analysisOverlay ?? []) map.set(entry.move.to, entry);
    return map;
  });

  /** Squares captured by the currently hovered move, for a preview ring. */
  const hoveredCaptures = createMemo(() => new Set(props.hoveredMove?.captures ?? []));

  createEffect(() => {
    void props.legalMoves; // track this dependency
    setSelectedSource(null);
  });

  function cellAt(idx: number): Cell | null {
    return props.state.squares[idx] ?? null;
  }

  function onCellClick(idx: number): void {
    if (props.busy) return;
    const src = selectedSource();

    if (src === null) {
      if (movablePieceSet().has(idx)) setSelectedSource(idx);
      return;
    }

    if (legalDstForSelected().has(idx)) {
      const mv = moveLookup().get(`${src},${idx}`);
      if (mv) {
        setSelectedSource(null);
        props.onMove(mv);
        return;
      }
    }
    if (src === idx) {
      setSelectedSource(null);
      return;
    }
    if (movablePieceSet().has(idx)) {
      setSelectedSource(idx);
      return;
    }
    setSelectedSource(null);
  }

  function isLegalDst(idx: number): boolean {
    if (props.busy) return false;
    return legalDstForSelected().has(idx);
  }

  return (
    <div class="cg-board">
      <div class="cg-grid">
        <For each={Array.from({ length: SIZE * SIZE }, (_, i) => i)}>
          {(idx) => {
            const row = Math.floor(idx / SIZE);
            const col = idx % SIZE;
            const zone = zoneOf(row, col);
            const cell = () => cellAt(idx);
            const isMovable = () => movablePieceSet().has(idx) && cell() !== null;
            const legal = () => isLegalDst(idx);
            const overlay = () => overlayByDst().get(idx);
            const heat = () => overlay()?.visitShare ?? 0;
            const hovered = () =>
              !props.busy && selectedSource() !== null && props.hoveredMove != null &&
              props.hoveredMove.from === selectedSource() && props.hoveredMove.to === idx && legal();
            const captureTarget = () => hoveredCaptures().has(idx);
            const isGhost = () => !cell() && legal() && hovered();
            const atRisk = () => (props.state.river_since[idx] ?? 0) > 0 && cell() !== null;
            const previewMove = () =>
              selectedSource() !== null ? moveLookup().get(`${selectedSource()},${idx}`) : undefined;
            const isCapture = () => (previewMove()?.captures.length ?? 0) > 0;

            return (
              <button
                type="button"
                class="cg-cell"
                classList={{
                  [`cg-${zone}`]: true,
                  "cg-checker": zone === "grass" && (row + col) % 2 === 0,
                  "cg-movable": !props.busy && selectedSource() === null && isMovable(),
                  "cg-selected-source": selectedSource() === idx,
                  legal: legal(),
                  hovered: hovered(),
                  "capture-legal": legal() && isCapture(),
                  "capture-target": captureTarget(),
                  heat: overlay() !== undefined,
                  proven: overlay()?.isProven ?? false,
                  suggested: overlay()?.isSuggested ?? false,
                }}
                style={{ "--heat": String(heat()) }}
                disabled={
                  props.busy ||
                  (selectedSource() === null
                    ? !isMovable()
                    : !legal() && idx !== selectedSource() && !isMovable())
                }
                onClick={() => onCellClick(idx)}
                onMouseEnter={() => {
                  if (selectedSource() !== null && !props.busy && legal()) {
                    const mv = moveLookup().get(`${selectedSource()},${idx}`);
                    if (mv) props.onHover(mv);
                  }
                }}
                onMouseLeave={() => props.onHover(null)}
              >
                {(() => {
                  const c = cell();
                  if (c) {
                    return (
                      <span class={`cg-piece cg-piece-${c.player.toLowerCase()}`}>
                        <img class="cg-piece-icon" src={PIECE_ICON[c.piece]} alt={c.piece} draggable={false} />
                        {c.piece === "superpawn" ? <span class="cg-badge">★</span> : null}
                        {atRisk() ? <span class="cg-risk-ring" /> : null}
                      </span>
                    );
                  }
                  if (isGhost() && selectedSource() !== null) {
                    const srcCell = cellAt(selectedSource() as number);
                    if (srcCell) {
                      return (
                        <span class={`cg-piece cg-piece-ghost cg-piece-${srcCell.player.toLowerCase()}`}>
                          <img class="cg-piece-icon" src={PIECE_ICON[srcCell.piece]} alt="" draggable={false} />
                        </span>
                      );
                    }
                  }
                  return null;
                })()}
              </button>
            );
          }}
        </For>
      </div>
    </div>
  );
};
