// ChessVariantRenderer.tsx — Shared 8×8 board renderer for Knightthrough,
// Breakthrough, and any future piece-movement chess variants that share the
// same bitboard wire format `{black: u64, white: u64, turn, winner}` with
// `[src, dst]` moves.
//
// Uses two-click interaction (click source piece, then click destination),
// because multiple pieces can reach the same destination in these games.
// A one-click-destination model would be ambiguous.
//
// IMPORTANT (SolidJS): the `selectedSource` signal must be *called directly*
// inside the JSX expressions (classList/disabled/children), not stored in a
// local `const src = selectedSource()` before the `return`. Reading a signal
// into a local variable before the `return` of a `For` callback does NOT
// register a reactive dependency, so the cell style would never update after
// a click. All reads below call `selectedSource()` inline.
//
// Renders a DOM/CSS checkerboard with black/white stone discs on each cell
// occupied by a piece. Legal moves render as translucent dots (empty target)
// or ring highlights (capture). Ghost preview, analysis heatmap, and the
// hover protocol all follow the same pattern as TttRenderer/OthelloRenderer.
//
// Vertical orientation: the Rust engine's bitboard uses bottom-left origin
// (row 0 = south wall), but the CSS grid renders top-left first. We flip the
// row index for display so the the north/top rows (black's starting
// territory) appear at the top of the screen and south/bottom rows (white's
// starting territory) at the bottom.

import { type Component, createEffect, createMemo, createSignal, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";
import "./chess-variant.css";

const COLS = 8;
const ROWS = 8;

/** Convert a display-grid index (0 = top-left) into the engine's bitboard
 * index (0 = south/bottom-left). The bitboard stores row 0 at the bottom;
 * the UI shows row 7 at the top, so we flip the row coordinate. */
function engineIndex(displayIndex: number): number {
  const displayRow = Math.floor(displayIndex / COLS);
  const col = displayIndex % COLS;
  const engineRow = ROWS - 1 - displayRow;
  return engineRow * COLS + col;
}

/** Returns "black", "white", or null for a given engine cell index. */
function occupant(state: GameState, index: number): "black" | "white" | null {
  const black = BigInt(`0x${state.black}`);
  const white = BigInt(`0x${state.white}`);
  const bit = 1n << BigInt(index);
  if (black & bit) return "black";
  if (white & bit) return "white";
  return null;
}

/** Check if a legal move is a capture (lands on an occupied cell). */
function isCapture(move: Move, state: GameState): boolean {
  return occupant(state, move[1]) !== null;
}

function isLightSquare(displayRow: number, col: number): boolean {
  return (displayRow + col) % 2 === 0;
}

export const ChessVariantRenderer: Component<
  GameRendererProps<GameState, Move, GameView>
> = (props) => {
  /** Engine index of the currently selected source piece, or null. */
  const [selectedSource, setSelectedSource] = createSignal<number | null>(null);

  /** Group legal moves by source engine index. */
  const movesBySrc = createMemo(() => {
    const map = new Map<number, Move[]>();
    for (const mv of props.legalMoves) {
      const list = map.get(mv[0]) ?? [];
      list.push(mv);
      map.set(mv[0], list);
    }
    return map;
  });

  /** Set of source engine indices that have at least one legal move. */
  const movablePieceSet = createMemo(() => new Set(movesBySrc().keys()));

  /** Legal destination engine indices for the currently selected source. */
  const legalDstForSelected = createMemo(() => {
    const src = selectedSource();
    if (src === null) return new Set<number>();
    const moves = movesBySrc().get(src);
    if (!moves) return new Set<number>();
    return new Set(moves.map((mv) => mv[1]));
  });

  /** Quick lookup: (src,dst) → full move. */
  const moveLookup = createMemo(() => {
    const map = new Map<string, Move>();
    for (const mv of props.legalMoves) {
      map.set(`${mv[0]},${mv[1]}`, mv);
    }
    return map;
  });

  const overlayByDst = createMemo(() => {
    const map = new Map<number, { visitShare: number; isProven: boolean; isSuggested: boolean }>();
    for (const entry of props.analysisOverlay ?? []) map.set(entry.move[1], entry);
    return map;
  });

  // Reset source selection whenever the set of legal moves changes (e.g.,
  // after a move is made, AI responds, or the board transitions to a new
  // player's turn). Otherwise a stale selection would show destinations
  // from a previous state.
  createEffect(() => {
    void props.legalMoves; // track this dependency
    setSelectedSource(null);
  });

  function onCellClick(displayIndex: number): void {
    if (props.busy) return;
    const idx = engineIndex(displayIndex);
    const src = selectedSource();

    if (src === null) {
      // No source selected yet — try to select this piece if it can move.
      if (movablePieceSet().has(idx) && occupant(props.state, idx) !== null) {
        setSelectedSource(idx);
      }
    } else {
      // Source is selected — check if this click is a legal destination.
      if (legalDstForSelected().has(idx)) {
        const mv = moveLookup().get(`${src},${idx}`);
        if (mv) {
          setSelectedSource(null);
          props.onMove(mv);
          return;
        }
      }
      // Click on the same piece → deselect.
      if (src === idx) {
        setSelectedSource(null);
        return;
      }
      // Click on another movable piece → switch selection.
      if (movablePieceSet().has(idx) && occupant(props.state, idx) !== null) {
        setSelectedSource(idx);
        return;
      }
      // Click on an empty or non-movable cell → deselect.
      setSelectedSource(null);
    }
  }

  /** Returns true if the given engine index is a legal destination for the
   * currently selected source piece. */
  function isLegalDst(idx: number): boolean {
    if (props.busy) return false;
    return legalDstForSelected().has(idx);
  }

  return (
    <div class="cv-board">
      <div class="cv-grid">
        <For each={Array.from({ length: 64 }, (_, i) => i)}>
          {(displayIndex) => {
            const idx = engineIndex(displayIndex);
            const displayRow = Math.floor(displayIndex / COLS);
            const col = displayIndex % COLS;
            const occ = () => occupant(props.state, idx);
            const isMovable = () => movablePieceSet().has(idx) && occ() !== null;
            const legal = () => isLegalDst(idx);
            const overlay = () => overlayByDst().get(idx);
            const heat = () => overlay()?.visitShare ?? 0;
            const hovered = () =>
              !props.busy && selectedSource() !== null && props.hoveredMove != null &&
              props.hoveredMove[0] === selectedSource() && props.hoveredMove[1] === idx && legal();
            const isGhost = () => !occ() && legal() && hovered();

            return (
              <button
                type="button"
                class="cv-cell"
                classList={{
                  "cv-light": isLightSquare(displayRow, col),
                  "cv-dark": !isLightSquare(displayRow, col),
                  "cv-movable": !props.busy && selectedSource() === null && isMovable(),
                  "cv-selected-source": selectedSource() === idx,
                  legal: legal(),
                  hovered: hovered(),
                  "capture-legal": legal() && isCapture(
                    moveLookup().get(`${selectedSource()},${idx}`) ?? [0, 0],
                    props.state,
                  ),
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
                onClick={() => onCellClick(displayIndex)}
                onMouseEnter={() => {
                  if (selectedSource() !== null && !props.busy && legal()) {
                    const mv = moveLookup().get(`${selectedSource()},${idx}`);
                    if (mv) props.onHover(mv);
                  }
                }}
                onMouseLeave={() => props.onHover(null)}
              >
                {occ() === "black" ? (
                  <span class="cv-piece-black" />
                ) : occ() === "white" ? (
                  <span class="cv-piece-white" />
                ) : isGhost() ? (
                  <span class="cv-piece-ghost">{props.state.turn === "Black" ? "●" : "○"}</span>
                ) : null}
              </button>
            );
          }}
        </For>
      </div>
    </div>
  );
};