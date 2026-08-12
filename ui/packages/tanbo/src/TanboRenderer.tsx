// TanboRenderer.tsx — A 9×9 Go-style board for Tanbo: stones sit on line
// intersections (not inside cells), with drawn grid lines so the board is
// legible even when empty. Orthogonally adjacent same-colour stones are
// linked by a connector bar spanning center-to-center, so it's always fully
// hidden under the two stone circles at both ends and can't show a seam.

import { type Component, createMemo, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";
import "./tanbo.css";

const N = 9;
const BOARD_SIZE = N * N;

/** Spacing between adjacent intersections, in px. */
const CELL = 60;
/** Margin from the board edge to the outermost line. */
const PAD = 36;
const BOARD_PX = PAD * 2 + (N - 1) * CELL;
/** Stone diameter. Connector thickness must stay <= this so a connector's
 * square-cut ends always land inside the stone circle at each end (a
 * rectangle's corner is `thickness / 2` from the point it's anchored to;
 * that's inside the circle as long as it's <= the stone radius). */
const STONE_D = 50;
const CONNECTOR_THICKNESS = STONE_D - 24;
/** Click/hover hit-box per intersection. Equal to CELL so adjacent hit-boxes
 * tile the board exactly, with no gaps and no overlap. */
const HIT = CELL;

/** Standard 9×9 Go star points (hoshi), reused here purely as a visual
 * reference grid for the eye — Tanbo has no rules attached to them. */
const STAR_POINTS = [20, 24, 40, 56, 60];

function pointX(cell: number): number {
  return PAD + (cell % N) * CELL;
}
function pointY(cell: number): number {
  return PAD + Math.floor(cell / N) * CELL;
}

export const TanboRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
  const legalSet = createMemo(() => new Set(props.legalMoves));

  const overlayByCell = createMemo(() => {
    const map = new Map<number, { visitShare: number; isProven: boolean; isSuggested: boolean }>();
    for (const entry of props.analysisOverlay ?? []) map.set(entry.move, entry);
    return map;
  });

  function onCellClick(cell: number): void {
    if (props.busy || !legalSet().has(cell)) return;
    props.onMove(cell);
  }

  const gridLines = createMemo(() => {
    const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];
    for (let i = 0; i < N; i++) {
      const p = PAD + i * CELL;
      lines.push({ x1: p, y1: PAD, x2: p, y2: BOARD_PX - PAD });
      lines.push({ x1: PAD, y1: p, x2: BOARD_PX - PAD, y2: p });
    }
    return lines;
  });

  return (
    <div class="tanbo-board">
      <div class="tanbo-grid" style={{ width: `${BOARD_PX}px`, height: `${BOARD_PX}px` }}>
        <svg class="tanbo-lines" width={BOARD_PX} height={BOARD_PX}>
          <For each={gridLines()}>{(l) => <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} />}</For>
          <For each={STAR_POINTS}>{(cell) => <circle class="tanbo-star" cx={pointX(cell)} cy={pointY(cell)} r={4} />}</For>
        </svg>

        <For each={Array.from({ length: BOARD_SIZE }, (_, i) => i)}>
          {(cell) => {
            const legal = () => !props.busy && legalSet().has(cell);
            const overlay = () => overlayByCell().get(cell);
            const heat = () => overlay()?.visitShare ?? 0;

            const stone = () => props.state.cells[cell];
            const stoneColor = (): "black" | "white" | null =>
              stone() === "Black" ? "black" : stone() === "White" ? "white" : null;

            const x = pointX(cell);
            const y = pointY(cell);

            // Connection to right/bottom neighbour (each pair drawn once).
            const connRight = () => {
              if (cell % N >= N - 1) return false;
              const s = stone();
              return s != null && s === props.state.cells[cell + 1];
            };
            const connDown = () => {
              if (Math.floor(cell / N) >= N - 1) return false;
              const s = stone();
              return s != null && s === props.state.cells[cell + N];
            };

            return (
              <>
                {stoneColor() && connRight() && (
                  <div
                    class={`tanbo-connector conn-${stoneColor()}`}
                    style={{
                      left: `${x}px`,
                      top: `${y - CONNECTOR_THICKNESS / 2}px`,
                      width: `${CELL}px`,
                      height: `${CONNECTOR_THICKNESS}px`,
                    }}
                  />
                )}
                {stoneColor() && connDown() && (
                  <div
                    class={`tanbo-connector conn-${stoneColor()}`}
                    style={{
                      left: `${x - CONNECTOR_THICKNESS / 2}px`,
                      top: `${y}px`,
                      width: `${CONNECTOR_THICKNESS}px`,
                      height: `${CELL}px`,
                    }}
                  />
                )}

                <button
                  type="button"
                  class="tanbo-point"
                  classList={{
                    legal: legal(),
                    hovered: props.hoveredMove === cell && legal(),
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
                  onMouseEnter={() => legal() && props.onHover(cell)}
                  onMouseLeave={() => props.onHover(null)}
                >
                  {stoneColor() && <div class={`tanbo-stone stone-${stoneColor()}`} />}
                  {!stoneColor() && legal() && props.hoveredMove === cell && (
                    <div class={`tanbo-ghost stone-${props.state.turn === "Black" ? "black" : "white"}`} />
                  )}
                </button>
              </>
            );
          }}
        </For>
      </div>
    </div>
  );
};
