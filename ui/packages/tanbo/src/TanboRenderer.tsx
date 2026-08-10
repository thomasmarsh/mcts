// TanboRenderer.tsx — A 9×9 CSS grid for Tanbo with group-connection
// indicators: orthogonally adjacent stones of the same colour are visually
// linked by a coloured bar, making bounded-root captures easy to spot.

import { type Component, createMemo, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";
import "./tanbo.css";

const N = 9;
const BOARD_SIZE = N * N;

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

  return (
    <div class="tanbo-board">
      <div class="tanbo-grid">
        <For each={Array.from({ length: BOARD_SIZE }, (_, i) => i)}>
          {(cell) => {
            const legal = () => !props.busy && legalSet().has(cell);
            const overlay = () => overlayByCell().get(cell);
            const heat = () => overlay()?.visitShare ?? 0;

            const stone = () => props.state.cells[cell];
            const isBlack = () => stone() === "Black";
            const isWhite = () => stone() === "White";

            // Connection to right neighbour
            const connRight = () => {
              if (cell % N >= N - 1) return false;
              const s = stone();
              return s != null && s === props.state.cells[cell + 1];
            };
            // Connection to bottom neighbour
            const connDown = () => {
              if (Math.floor(cell / N) >= N - 1) return false;
              const s = stone();
              return s != null && s === props.state.cells[cell + N];
            };

            const stoneColor = () =>
              isBlack() ? "#1a1a1a" : isWhite() ? "#f0f0f0" : null;

            return (
              <button
                type="button"
                class="tanbo-cell"
                classList={{
                  legal: legal(),
                  hovered: props.hoveredMove === cell && legal(),
                  heat: overlay() !== undefined,
                  proven: overlay()?.isProven ?? false,
                  suggested: overlay()?.isSuggested ?? false,
                }}
                style={{ "--heat": String(heat()) }}
                disabled={!legal()}
                onClick={() => onCellClick(cell)}
                onMouseEnter={() => legal() && props.onHover(cell)}
                onMouseLeave={() => props.onHover(null)}
              >
                {/* Stone circle */}
                {stoneColor() && (
                  <div
                    class="tanbo-stone"
                    style={{ background: stoneColor()! }}
                  />
                )}

                {/* Connector bars between same-colour neighbours */}
                {stoneColor() && connRight() && (
                  <div
                    class="tanbo-connector right"
                    style={{ background: stoneColor()! }}
                  />
                )}
                {stoneColor() && connDown() && (
                  <div
                    class="tanbo-connector down"
                    style={{ background: stoneColor()! }}
                  />
                )}

                {/* Ghost preview */}
                {!stoneColor() && legal() && props.hoveredMove === cell && (
                  <div
                    class="tanbo-ghost"
                    style={{
                      background: props.state.turn === "Black" ? "#1a1a1a" : "#f0f0f0",
                    }}
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