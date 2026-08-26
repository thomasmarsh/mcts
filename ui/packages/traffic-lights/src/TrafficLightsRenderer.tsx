// TrafficLightsRenderer.tsx — A 3x3 board of coloured circles
// (empty → red → yellow → green), rendered as DOM/CSS circles with no
// three.js dependency — same contrast with `DruidRenderer` that
// `TttRenderer` provides, but for a game with coloured cell states
// rather than player marks.

import { type Component, createMemo, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import type { CellState, GameState, GameView, Move } from "./types.js";
import "./traffic-lights.css";

/** Decode the cell index from a raw `Move` (top bits carry the piece
 * encoding; only the `>> 2` part is the cell index). */
function moveCellIndex(mv: Move): number {
  return mv >> 2;
}

export const TrafficLightsRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (
  props,
) => {
  const legalSet = createMemo<Set<number>>(() => {
    const set = new Set<number>();
    for (const mv of props.legalMoves) {
      set.add(moveCellIndex(mv));
    }
    return set;
  });

  /** Map from cell index → the *only* legal Move affecting that cell
   * (each cell has at most one legal move per position, since the piece
   * progression is deterministic: empty→R, R→Y, Y→G). */
  const moveByCell = createMemo<Map<number, Move>>(() => {
    const map = new Map<number, Move>();
    for (const mv of props.legalMoves) {
      map.set(moveCellIndex(mv), mv);
    }
    return map;
  });

  const overlayByCell = createMemo(() => {
    const map = new Map<number, { visitShare: number; isProven: boolean; isSuggested: boolean }>();
    for (const entry of props.analysisOverlay ?? []) map.set(moveCellIndex(entry.move), entry);
    return map;
  });

  function onCellClick(cell: number): void {
    if (props.busy) return;
    const mv = moveByCell().get(cell);
    if (mv === undefined) return;
    props.onMove(mv);
  }

  const cellClass = (cell: number, mark: CellState) => {
    const legal = !props.busy && legalSet().has(cell);
    const overlay = overlayByCell().get(cell);
    const classes: Record<string, boolean | undefined> = {
      "tl-cell": true,
      empty: mark === null,
      legal: legal && mark !== null,
      hovered: props.hoveredMove != null && moveCellIndex(props.hoveredMove) === cell && legal,
      heat: overlay !== undefined,
      proven: overlay?.isProven ?? false,
      suggested: overlay?.isSuggested ?? false,
    };
    if (mark !== null) classes[`piece-${mark}`] = true;
    return classes;
  };

  return (
    <div class="tl-board">
      <div class="tl-grid">
        <For each={props.state.cells}>
          {(mark, i) => {
            const cell = i();
            // Derived values must be thunks called inside the JSX
            // expressions, not plain consts: `<For>` runs this callback
            // once per item, and a plain const freezes at creation time
            // (e.g. a cell recreated during the legalMoves=[] refetch gap
            // would keep `disabled` true forever — the classList stays
            // fresh only because `cellClass()` reads reactive sources
            // inside the JSX effect).
            const legal = () => !props.busy && legalSet().has(cell);
            const overlay = () => overlayByCell().get(cell);
            const heat = () => overlay()?.visitShare ?? 0;
            const isGhost = () =>
              mark === null &&
              legal() &&
              props.hoveredMove != null &&
              moveCellIndex(props.hoveredMove) === cell;

            return (
              <button
                type="button"
                classList={cellClass(cell, mark)}
                style={{ "--heat": String(heat()) }}
                disabled={!legal()}
                onClick={() => onCellClick(cell)}
                onMouseEnter={() => legal() && props.onHover(moveByCell().get(cell) ?? null)}
                onMouseLeave={() => props.onHover(null)}
              >
                {isGhost() ? <span class="tl-ghost" /> : ""}
              </button>
            );
          }}
        </For>
      </div>
    </div>
  );
};
