// TttRenderer.tsx — A plain 3x3 DOM/CSS grid (PLAN-UI.md session 8), no
// three.js: the deliberate contrast with `DruidRenderer`'s WebGL board,
// proving `GameRendererProps` doesn't secretly assume 3D or Druid's own
// move encoding (a `[Piece, index]` tuple vs. tic-tac-toe's bare cell index).

import { type Component, createMemo, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";
import "./ttt.css";

export const TttRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
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
    <div class="ttt-board">
      <div class="ttt-grid">
        <For each={props.state.cells}>
          {(mark, i) => {
            const cell = i();
            const legal = () => !props.busy && legalSet().has(cell);
            const overlay = () => overlayByCell().get(cell);
            const heat = () => overlay()?.visitShare ?? 0;
            return (
              <button
                type="button"
                class="ttt-cell"
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
                {mark ?? (legal() && props.hoveredMove === cell ? <span class="ttt-ghost">{props.state.turn}</span> : "")}
              </button>
            );
          }}
        </For>
      </div>
    </div>
  );
};
