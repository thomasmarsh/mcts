// FocusRenderer.tsx — Plain DOM/CSS 8x8 board (52 playable cells; the 12
// notched-off corner slots render as inert gaps, see geometry.ts). Each
// playable cell shows its stack as a small bottom-to-top column of colored
// chips.
//
// Two-stage click, mirroring TakRenderer's "select a stack, then its
// destination" pattern -- but simpler, since a Focus slide's (direction,
// count) pair maps to exactly one destination cell per source (no
// drop-schedule ambiguity the way a Tak spread has, see move-codec.ts's
// `destinationCell` doc comment), so there's no candidate-list panel: with
// no source selected, clicking one of your own controllable stacks selects
// it (`selectedSrc`, component-local -- purely a renderer interaction
// detail, not lifted to the store); with a source selected, clicking a
// highlighted destination cell fires that move directly, and clicking the
// source again deselects it.
//
// `GameShell`'s `Place`/`Slide` modes (see summary.ts) already filter
// `legalMoves` down to one move kind at a time, but this component doesn't
// assume that -- it buckets `legalMoves` into placements and slides
// unconditionally, so both interaction paths "just work" regardless of
// which mode (if any) is active.

import { type Component, createEffect, createMemo, createSignal, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import { ALL_CELLS, isValidCell } from "./geometry.js";
import { destinationCell, isSlideMove, moveCell } from "./move-codec.js";
import type { Move } from "./move-codec.js";
import { PLAYER_COLORS } from "./summary.js";
import type { GameState, GameView } from "./types.js";
import "./focus.css";

interface OverlayInfo {
  visitShare: number;
  isProven: boolean;
  isSuggested: boolean;
}

export const FocusRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
  const [selectedSrc, setSelectedSrc] = createSignal<number | null>(null);

  // A new position drops any in-progress source selection -- otherwise a
  // stale `selectedSrc` could point at a cell whose stack just changed
  // entirely (or vanished under an opponent's merge).
  createEffect(() => {
    void props.state;
    setSelectedSrc(null);
  });

  const placeByCell = createMemo(() => {
    const map = new Map<number, Move>();
    for (const m of props.legalMoves) if (!isSlideMove(m)) map.set(moveCell(m), m);
    return map;
  });

  const slideMoves = createMemo(() => props.legalMoves.filter(isSlideMove));
  const slideSources = createMemo(() => new Set(slideMoves().map(moveCell)));

  const destinationsFromSrc = createMemo(() => {
    const src = selectedSrc();
    const map = new Map<number, Move>();
    if (src === null) return map;
    for (const m of slideMoves()) if (moveCell(m) === src) map.set(destinationCell(m), m);
    return map;
  });

  const overlayByCell = createMemo(() => {
    const map = new Map<number, OverlayInfo>();
    for (const entry of props.analysisOverlay ?? []) {
      map.set(destinationCell(entry.move), entry);
    }
    return map;
  });

  const hoveredTargetCell = createMemo(() => {
    const mv = props.hoveredMove;
    return mv === null ? null : destinationCell(mv);
  });

  function stackAt(cell: number): number[] {
    return props.view.board[cell] ?? [];
  }

  function onCellClick(cell: number): void {
    if (props.busy) return;
    const src = selectedSrc();
    if (src !== null) {
      const dest = destinationsFromSrc().get(cell);
      if (dest !== undefined) {
        setSelectedSrc(null);
        props.onMove(dest);
        return;
      }
      if (cell === src) {
        setSelectedSrc(null);
        return;
      }
    }
    if (slideSources().has(cell)) {
      setSelectedSrc(cell);
      return;
    }
    const placeMv = placeByCell().get(cell);
    if (placeMv !== undefined) props.onMove(placeMv);
  }

  function onCellHover(cell: number): void {
    if (props.busy) return;
    const src = selectedSrc();
    if (src !== null) {
      props.onHover(destinationsFromSrc().get(cell) ?? null);
      return;
    }
    if (slideSources().has(cell)) {
      props.onHover(null); // selecting a source isn't itself a move
      return;
    }
    props.onHover(placeByCell().get(cell) ?? null);
  }

  return (
    <div class="focus-board">
      <div class="focus-grid">
        <For each={ALL_CELLS}>
          {(cell) => {
            if (!isValidCell(cell)) return <div class="focus-gap" />;
            const isSelectable = () =>
              !props.busy && slideSources().has(cell) && selectedSrc() === null;
            const isSelected = () => selectedSrc() === cell;
            const isDestination = () => destinationsFromSrc().has(cell);
            const isPlaceable = () =>
              !props.busy && selectedSrc() === null && placeByCell().has(cell);
            const overlay = () => overlayByCell().get(cell);
            const heat = () => overlay()?.visitShare ?? 0;
            return (
              <button
                type="button"
                class="focus-cell"
                classList={{
                  selectable: isSelectable(),
                  selected: isSelected(),
                  destination: isDestination(),
                  placeable: isPlaceable(),
                  hovered: hoveredTargetCell() === cell,
                  heat: overlay() !== undefined,
                  proven: overlay()?.isProven ?? false,
                  suggested: overlay()?.isSuggested ?? false,
                }}
                style={{ "--heat": String(heat()) }}
                disabled={props.busy}
                onClick={() => onCellClick(cell)}
                onMouseEnter={() => onCellHover(cell)}
                onMouseLeave={() => props.onHover(null)}
              >
                <div class="focus-stack">
                  <For each={stackAt(cell)}>
                    {(player) => (
                      <span class="focus-piece" style={{ background: PLAYER_COLORS[player] }} />
                    )}
                  </For>
                </div>
              </button>
            );
          }}
        </For>
      </div>
    </div>
  );
};
