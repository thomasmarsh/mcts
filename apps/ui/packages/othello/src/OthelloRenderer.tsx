// OthelloRenderer.tsx — An 8×8 Othello grid rendered in plain DOM/CSS.
// Each cell shows a black disc (●), white disc (○), or empty with a ghost
// preview for legal moves on hover. A pass button appears below the grid
// when `Move(64)` is legal.

import { type Component, createMemo, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";
import "./othello.css";

export const OthelloRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
  const legalSet = createMemo(() => new Set(props.legalMoves));
  const hasPass = createMemo(() => legalSet().has(64));

  const overlayByCell = createMemo(() => {
    const map = new Map<number, { visitShare: number; isProven: boolean; isSuggested: boolean }>();
    for (const entry of props.analysisOverlay ?? []) map.set(entry.move, entry);
    return map;
  });

  /** Returns "black", "white", or null for a given cell index. */
  function occupant(index: number): "black" | "white" | null {
    const cellBit = 1n << BigInt(index);
    const black = BigInt(`0x${props.state.black}`);
    const white = BigInt(`0x${props.state.white}`);
    if (black & cellBit) return "black";
    if (white & cellBit) return "white";
    return null;
  }

  function onCellClick(cell: number): void {
    if (props.busy || !legalSet().has(cell)) return;
    props.onMove(cell);
  }

  return (
    <div class="othello-board">
      <div class="othello-grid">
        <For each={Array.from({ length: 64 }, (_, i) => i)}>
          {(index) => {
            const occ = () => occupant(index);
            const legal = () => !props.busy && legalSet().has(index);
            const overlay = () => overlayByCell().get(index);
            const heat = () => overlay()?.visitShare ?? 0;
            return (
              <button
                type="button"
                class="othello-cell"
                classList={{
                  "cell-black": occ() === "black",
                  "cell-white": occ() === "white",
                  legal: legal(),
                  hovered: props.hoveredMove === index && legal(),
                  heat: overlay() !== undefined,
                  proven: overlay()?.isProven ?? false,
                  suggested: overlay()?.isSuggested ?? false,
                }}
                style={{ "--heat": String(heat()) }}
                disabled={!legal() && occ() !== null}
                onClick={() => onCellClick(index)}
                onMouseEnter={() => legal() && props.onHover(index)}
                onMouseLeave={() => props.onHover(null)}
              >
                {occ() === "black" ? (
                  "●"
                ) : occ() === "white" ? (
                  "○"
                ) : legal() && props.hoveredMove === index ? (
                  <span class="othello-ghost">{props.state.turn === "Black" ? "●" : "○"}</span>
                ) : null}
              </button>
            );
          }}
        </For>
      </div>
      {hasPass() && (
        <button
          type="button"
          class="othello-pass"
          classList={{ suggested: overlayByCell().get(64)?.isSuggested ?? false }}
          disabled={props.busy}
          onClick={() => onCellClick(64)}
        >
          Pass
        </button>
      )}
    </div>
  );
};
