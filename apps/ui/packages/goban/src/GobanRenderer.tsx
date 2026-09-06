// GobanRenderer.tsx — Shared N×N Go-style board renderer: stones sit on line
// intersections (not inside cells), with drawn grid lines so the board is
// legible even when empty. Extracted from Tanbo's original renderer, minus
// Tanbo's connector-bar-between-same-colour-stones visual (that stays
// Tanbo-specific — AtariGo/Gonnect have no equivalent concept).
//
// Board size is chosen per new-game (AtariGo: 5/7/9, Gonnect: 9/13/19 — see
// each package's `NewGameFields`), not fixed at module-load time, so it's
// derived reactively from `props.state.cells.length` (always `N * N`)
// rather than being a factory parameter closed over once.

import { type Component, createMemo, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import { cellOf, type GameState, type GameView, type Move } from "./types.js";
import { standardStarPoints } from "./star-points.js";
import "./goban.css";

/** Spacing between adjacent intersections, in px. */
const CELL = 60;
/** Margin from the board edge to the outermost line. */
const PAD = 36;
/** Click/hover hit-box per intersection. Equal to CELL so adjacent hit-boxes
 * tile the board exactly, with no gaps and no overlap. */
const HIT = CELL;

function boardSizeOf(cells: readonly unknown[]): number {
  return Math.round(Math.sqrt(cells.length));
}

export const GobanRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
  const n = createMemo(() => boardSizeOf(props.state.cells));
  const boardSize = createMemo(() => n() * n());
  const boardPx = createMemo(() => PAD * 2 + (n() - 1) * CELL);
  const starPoints = createMemo(() => standardStarPoints(n()));

  function pointX(cell: number): number {
    return PAD + (cell % n()) * CELL;
  }
  function pointY(cell: number): number {
    return PAD + Math.floor(cell / n()) * CELL;
  }

  /** Legal moves keyed by cell — the UI only ever has one legal placement
   * per empty cell, so a click needs to look up the actual `Move` object
   * (cell index alone isn't a valid move; see `types.ts`) rather than
   * constructing one. */
  const legalByCell = createMemo(() => {
    const map = new Map<number, Move>();
    for (const mv of props.legalMoves) map.set(cellOf(mv), mv);
    return map;
  });

  const overlayByCell = createMemo(() => {
    const map = new Map<number, { visitShare: number; isProven: boolean; isSuggested: boolean }>();
    for (const entry of props.analysisOverlay ?? []) map.set(cellOf(entry.move), entry);
    return map;
  });

  const hoveredCell = () => (props.hoveredMove !== null ? cellOf(props.hoveredMove) : null);

  function onCellClick(cell: number): void {
    if (props.busy) return;
    const mv = legalByCell().get(cell);
    if (mv) props.onMove(mv);
  }

  const gridLines = createMemo(() => {
    const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];
    for (let i = 0; i < n(); i++) {
      const p = PAD + i * CELL;
      lines.push({ x1: p, y1: PAD, x2: p, y2: boardPx() - PAD });
      lines.push({ x1: PAD, y1: p, x2: boardPx() - PAD, y2: p });
    }
    return lines;
  });

  return (
    <div class="goban-board">
      <div class="goban-grid" style={{ width: `${boardPx()}px`, height: `${boardPx()}px` }}>
        <svg class="goban-lines" width={boardPx()} height={boardPx()}>
          <For each={gridLines()}>{(l) => <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} />}</For>
          <For each={starPoints()}>
            {(cell) => <circle class="goban-star" cx={pointX(cell)} cy={pointY(cell)} r={4} />}
          </For>
        </svg>

        <For each={Array.from({ length: boardSize() }, (_, i) => i)}>
          {(cell) => {
            const legal = () => !props.busy && legalByCell().has(cell);
            const overlay = () => overlayByCell().get(cell);
            const heat = () => overlay()?.visitShare ?? 0;

            const stone = () => props.state.cells[cell];
            const stoneColor = (): "black" | "white" | null =>
              stone() === "Black" ? "black" : stone() === "White" ? "white" : null;

            const x = pointX(cell);
            const y = pointY(cell);

            return (
              <button
                type="button"
                class="goban-point"
                classList={{
                  legal: legal(),
                  hovered: hoveredCell() === cell && legal(),
                  heat: overlay() !== undefined,
                  proven: overlay()?.isProven ?? false,
                  suggested: overlay()?.isSuggested ?? false,
                }}
                style={{
                  left: `${x}px`,
                  top: `${y}px`,
                  width: `${HIT}px`,
                  height: `${HIT}px`,
                  "--heat": String(heat()),
                }}
                disabled={!legal()}
                onClick={() => onCellClick(cell)}
                onMouseEnter={() => {
                  if (!legal()) return;
                  const mv = legalByCell().get(cell);
                  if (mv) props.onHover(mv);
                }}
                onMouseLeave={() => props.onHover(null)}
              >
                {stoneColor() && <div class={`goban-stone goban-stone-${stoneColor()}`} />}
                {!stoneColor() && legal() && hoveredCell() === cell && (
                  <div
                    class={`goban-ghost goban-stone-${props.state.turn === "Black" ? "black" : "white"}`}
                  />
                )}
              </button>
            );
          }}
        </For>
      </div>
    </div>
  );
};
