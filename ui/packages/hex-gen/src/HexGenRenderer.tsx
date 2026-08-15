// HexGenRenderer.tsx — An SVG hex grid, the first genuinely hexagonal board
// renderer in this UI (contrast `@mcts/goban`'s square grid and
// `TttRenderer`'s DOM grid). Pointy-top hexagons laid out with axial
// coordinates `(q, r) = (col, row)`, sheared so row `r+1` sits half a hex to
// the *left* of row `r` -- matching `game_core::bitboard::BitBoard::flood6`'s
// actual hex-adjacent diagonal: `(row, col)` and `(row+1, col+1)` are the
// pair `shift_northeast`/`shift_southwest` connect (proven by that module's
// `test_flood6_uses_northeast_southwest_diagonal_only`), not `(row-1, col+1)`
// as a naive "rows shift right" layout would assume. Shearing left instead
// makes that pair the six-neighbor set actually drawn touching here; see
// `centerOf`'s derivation.

import { type Component, createMemo, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import type { GameState, GameView, Move, Player } from "./types.js";
import { sideOf } from "./types.js";
import "./hex-gen.css";

const HEX_SIZE = 28;
const SQRT3 = Math.sqrt(3);

/** Pixel center of cell `(row, col)`, before the board-wide translation
 * that shifts everything into positive `viewBox` coordinates. */
function centerOf(row: number, col: number): { x: number; y: number } {
  return {
    x: HEX_SIZE * SQRT3 * (col - row / 2),
    y: HEX_SIZE * 1.5 * row,
  };
}

/** The six corners of a pointy-top hexagon centered at `(cx, cy)`, as an
 * SVG `points` attribute value. */
function hexPoints(cx: number, cy: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i - 30);
    pts.push(`${cx + HEX_SIZE * Math.cos(angle)},${cy + HEX_SIZE * Math.sin(angle)}`);
  }
  return pts.join(" ");
}

interface Cell {
  index: number;
  row: number;
  col: number;
  cx: number;
  cy: number;
}

/** All `side * side` cell centers, translated so the whole board (plus one
 * hex's worth of margin) sits inside a non-negative `viewBox` -- computed
 * from the actual min/max of `centerOf`'s output rather than a hand-derived
 * formula, since shearing can push `x` negative for large `row`. */
function layoutCells(side: number): { cells: Cell[]; width: number; height: number } {
  const raw: { row: number; col: number; x: number; y: number }[] = [];
  for (let row = 0; row < side; row++) {
    for (let col = 0; col < side; col++) {
      raw.push({ row, col, ...centerOf(row, col) });
    }
  }
  const margin = HEX_SIZE * 1.5;
  const minX = Math.min(...raw.map((c) => c.x)) - margin;
  const minY = Math.min(...raw.map((c) => c.y)) - margin;
  const maxX = Math.max(...raw.map((c) => c.x)) + margin;
  const maxY = Math.max(...raw.map((c) => c.y)) + margin;
  const cells = raw.map((c) => ({
    index: c.row * side + c.col,
    row: c.row,
    col: c.col,
    cx: c.x - minX,
    cy: c.y - minY,
  }));
  return { cells, width: maxX - minX, height: maxY - minY };
}

/** `P0` owns the top/bottom edges (row 0 / row `side-1`), `P1` owns the
 * left/right edges (col 0 / col `side-1`) -- mirrors
 * `games/hex-gen/src/lib.rs`'s `Position::winner` edge sets. */
function edgeOwner(cell: Cell, side: number): Player | null {
  if (cell.row === 0 || cell.row === side - 1) return "P0";
  if (cell.col === 0 || cell.col === side - 1) return "P1";
  return null;
}

export const HexGenRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
  const legalSet = createMemo(() => new Set(props.legalMoves));
  const side = createMemo(() => sideOf(props.state.cells.length));
  const layout = createMemo(() => layoutCells(side()));

  const overlayByCell = createMemo(() => {
    const map = new Map<number, { visitShare: number; isProven: boolean; isSuggested: boolean }>();
    for (const entry of props.analysisOverlay ?? []) map.set(entry.move, entry);
    return map;
  });

  function onCellClick(cell: number): void {
    if (props.busy || !legalSet().has(cell)) return;
    props.onMove(cell);
  }

  return (
    <div class="hex-gen-board">
      <svg
        class="hex-gen-grid"
        viewBox={`0 0 ${layout().width} ${layout().height}`}
        width={layout().width}
        height={layout().height}
      >
        <For each={layout().cells}>
          {(cell) => {
            const mark = () => props.state.cells[cell.index] ?? null;
            const legal = () => !props.busy && legalSet().has(cell.index);
            const overlay = () => overlayByCell().get(cell.index);
            const heat = () => overlay()?.visitShare ?? 0;
            const owner = edgeOwner(cell, side());
            return (
              <g
                class="hex-gen-cell"
                classList={{
                  legal: legal(),
                  hovered: props.hoveredMove === cell.index && legal(),
                  heat: overlay() !== undefined,
                  proven: overlay()?.isProven ?? false,
                  suggested: overlay()?.isSuggested ?? false,
                  "edge-p0": owner === "P0",
                  "edge-p1": owner === "P1",
                }}
                style={{ "--heat": String(heat()) }}
                onClick={() => onCellClick(cell.index)}
                onMouseEnter={() => legal() && props.onHover(cell.index)}
                onMouseLeave={() => props.onHover(null)}
              >
                <polygon class="hex-gen-hex" points={hexPoints(cell.cx, cell.cy)} />
                {mark() !== null && (
                  <circle class="hex-gen-stone" classList={{ p0: mark() === "P0", p1: mark() === "P1" }}
                    cx={cell.cx} cy={cell.cy} r={HEX_SIZE * 0.55} />
                )}
                {mark() === null && legal() && props.hoveredMove === cell.index && (
                  <circle
                    class="hex-gen-ghost"
                    classList={{ p0: props.state.turn === "P0", p1: props.state.turn === "P1" }}
                    cx={cell.cx}
                    cy={cell.cy}
                    r={HEX_SIZE * 0.55}
                  />
                )}
              </g>
            );
          }}
        </For>
      </svg>
    </div>
  );
};
