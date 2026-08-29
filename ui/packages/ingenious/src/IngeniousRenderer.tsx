// IngeniousRenderer.tsx — SVG hex-grid board plus a rack/swap side panel.
//
// A tile has two colors and no inherent "up" side, so picking one from the
// rack alone doesn't fully determine a move -- the player also has to say
// which of the tile's two ends goes on which of the two empty cells. This
// renderer lets the player click either visible half of the selected rack tile
// to choose the color that will be anchored first. Every matching legal move
// is then presented through that color at its current board position.
//
// Placement is a two-click gesture, resolved entirely on the board (no popup):
// hover any empty cell to preview every legal domino touching it, click one to
// anchor it, then click its highlighted neighbor to finish. A legal move names
// only the lower-indexed endpoint because the engine stores each board edge
// once, but either endpoint can be the UI anchor. This exposes all six possible
// orientations around a hex without duplicating moves in the engine protocol.

import { type Component, createEffect, createMemo, createSignal, For, Show } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import { centerOf, neighborOf, VALID_CELLS } from "./geometry.js";
import { hexPoints, PieceIcon } from "./pieces.js";
import {
  isPlaceMove,
  type Color,
  type GameState,
  type GameView,
  type Move,
  type PlaceMove,
} from "./types.js";
import "./ingenious.css";

const HEX_SIZE = 22;

interface CellLayout {
  index: number;
  cx: number;
  cy: number;
}

interface PlacementEnd {
  move: PlaceMove;
  anchor: number;
  anchorColor: Color;
  target: number;
  targetColor: Color;
}

function layoutCells(): { cells: CellLayout[]; width: number; height: number } {
  const raw = VALID_CELLS.map((index) => ({ index, ...centerOf(index, HEX_SIZE) }));
  const margin = HEX_SIZE * 1.5;
  const minX = Math.min(...raw.map((c) => c.x)) - margin;
  const minY = Math.min(...raw.map((c) => c.y)) - margin;
  const maxX = Math.max(...raw.map((c) => c.x)) + margin;
  const maxY = Math.max(...raw.map((c) => c.y)) + margin;
  return {
    cells: raw.map((c) => ({ index: c.index, cx: c.x - minX, cy: c.y - minY })),
    width: maxX - minX,
    height: maxY - minY,
  };
}

// The valid-cell set and its layout never change (fixed 2-player board), so
// this is computed once at module load, not per render.
const LAYOUT = layoutCells();
const CELL_BY_INDEX = new Map(LAYOUT.cells.map((c) => [c.index, c]));

export const IngeniousRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (
  props,
) => {
  const [selectedType, setSelectedType] = createSignal<[Color, Color] | null>(null);
  const [anchorColor, setAnchorColor] = createSignal<Color | null>(null);
  // The first-clicked start cell of an in-progress two-click placement, or
  // `null` when no placement is underway.
  const [anchor, setAnchor] = createSignal<number | null>(null);
  // The cell under the pointer, used only for the local hover preview (the
  // renderer doesn't round-trip through the parent's `hoveredMove`).
  const [hoveredCell, setHoveredCell] = createSignal<number | null>(null);

  const currentRack = createMemo(() => props.state.racks[props.state.current_player] ?? []);

  const rackTypes = createMemo(() => {
    const seen = new Set<string>();
    const types: [Color, Color][] = [];
    for (const slot of currentRack()) {
      if (!slot) continue;
      const key = `${slot[0]}|${slot[1]}`;
      if (seen.has(key)) continue;
      seen.add(key);
      types.push(slot);
    }
    return types;
  });

  // Drop the current selection (and any in-progress placement) once it's no
  // longer offered -- a new turn, or this exact type left the rack.
  createMemo(() => {
    const t = selectedType();
    if (t && !rackTypes().some(([a, b]) => a === t[0] && b === t[1])) {
      setSelectedType(null);
      setAnchorColor(null);
      setAnchor(null);
    }
  });

  // Every legal placement for the selected physical tile. Which color goes on
  // the anchor is chosen by clicking that half of the rack tile, not with a
  // separate flip control.
  const placements = createMemo<PlaceMove[]>(() => {
    const t = selectedType();
    if (!t) return [];
    return props.legalMoves
      .filter(isPlaceMove)
      .map((m) => m.Place)
      .filter(
        (mv) =>
          (mv.color_a === t[0] && mv.color_b === t[1]) ||
          (mv.color_a === t[1] && mv.color_b === t[0]),
      );
  });

  // The two endpoint views of every legal move. The engine encodes an edge
  // from its lower-indexed end only; offering both views makes every physical
  // orientation around an empty hex reachable by the same two-click gesture.
  const placementEnds = createMemo<PlacementEnd[]>(() =>
    placements()
      .flatMap((move) => {
        const neighbor = neighborOf(move.cell, move.dir);
        if (neighbor === null) return [];
        return [
          {
            move,
            anchor: move.cell,
            anchorColor: move.color_a,
            target: neighbor,
            targetColor: move.color_b,
          },
          {
            move,
            anchor: neighbor,
            anchorColor: move.color_b,
            target: move.cell,
            targetColor: move.color_a,
          },
        ];
      })
      .filter((placement) => placement.anchorColor === anchorColor()),
  );

  const endsAt = (cell: number): PlacementEnd[] =>
    placementEnds().filter((placement) => placement.anchor === cell);

  // The first click may be either end of any legal domino.
  const validAnchors = createMemo(
    () => new Set(placementEnds().map((placement) => placement.anchor)),
  );

  const anchoredEnds = createMemo<PlacementEnd[]>(() => {
    const a = anchor();
    return a === null ? [] : endsAt(a);
  });

  // Hovering is intentionally only a preview: the normal board remains dark
  // and placed tiles remain easy to distinguish until the player asks for a
  // particular location.
  const previewEnds = createMemo<PlacementEnd[]>(() => {
    const h = hoveredCell();
    return anchor() === null && h !== null ? endsAt(h) : [];
  });

  const canPlace = createMemo(() => props.legalMoves.some(isPlaceMove));
  const canKeep = createMemo(() => props.legalMoves.includes("KeepRack"));
  const canSwap = createMemo(() => props.legalMoves.includes("Swap"));

  let automaticDecision: Move | null = null;
  createEffect(() => {
    const onlyMove = props.legalMoves.length === 1 ? props.legalMoves[0] : null;
    if (props.busy || props.state.phase !== "swap_decision" || !onlyMove || isPlaceMove(onlyMove)) {
      automaticDecision = null;
      return;
    }
    if (automaticDecision !== onlyMove) {
      automaticDecision = onlyMove;
      props.onMove(onlyMove);
    }
  });

  // If the anchor stops being legal (e.g. the turn changed), drop it.
  createMemo(() => {
    const a = anchor();
    if (a !== null && !validAnchors().has(a)) setAnchor(null);
  });

  function selectType(t: [Color, Color], color: Color): void {
    if (props.busy || !canPlace()) return;
    const current = selectedType();
    if (current && current[0] === t[0] && current[1] === t[1] && anchorColor() === color) {
      setSelectedType(null);
      setAnchorColor(null);
    } else {
      setSelectedType(t);
      setAnchorColor(color);
    }
    setAnchor(null);
  }

  function clickCell(c: number): void {
    if (props.busy) return;
    const a = anchor();
    if (a === null) {
      if (validAnchors().has(c)) setAnchor(c);
      return;
    }
    if (c === a) {
      setAnchor(null);
      return;
    }
    const placement = anchoredEnds().find((end) => end.target === c);
    if (placement) {
      props.onMove({ Place: placement.move });
      setAnchor(null);
      return;
    }
    // Clicking a different valid start cell restarts the placement there.
    if (validAnchors().has(c)) setAnchor(c);
  }

  function ghostGlyph(color: Color, cx: number, cy: number) {
    return (
      <>
        <polygon class="ingenious-ghost-hex" points={hexPoints(cx, cy, HEX_SIZE * 0.82)} />
        <PieceIcon color={color} cx={cx} cy={cy} r={HEX_SIZE * 0.82} />
      </>
    );
  }

  return (
    <div class="ingenious-board">
      <svg
        class="ingenious-grid"
        viewBox={`0 0 ${LAYOUT.width} ${LAYOUT.height}`}
        width={LAYOUT.width}
        height={LAYOUT.height}
      >
        <For each={LAYOUT.cells}>
          {(cell) => {
            const color = () => props.state.board[cell.index] ?? null;
            return (
              <>
                <polygon class="ingenious-hex" points={hexPoints(cell.cx, cell.cy, HEX_SIZE)} />
                <Show when={color()}>
                  <PieceIcon color={color() as Color} cx={cell.cx} cy={cell.cy} r={HEX_SIZE} />
                </Show>
              </>
            );
          }}
        </For>

        {/* Invisible hit areas keep the unselected board legible while still
            allowing any empty endpoint to start or preview a placement. */}
        <Show when={selectedType() !== null && anchor() === null}>
          <For each={LAYOUT.cells}>
            {(cell) => (
              <polygon
                class="ingenious-cell-hit"
                data-role="cell-hit"
                data-cell={cell.index}
                points={hexPoints(cell.cx, cell.cy, HEX_SIZE)}
                onClick={() => clickCell(cell.index)}
                onMouseEnter={() => setHoveredCell(cell.index)}
                onMouseLeave={() => setHoveredCell(null)}
              />
            )}
          </For>
        </Show>

        {/* Preview only the placements incident to the hex under the pointer. */}
        <For each={previewEnds()}>
          {(placement) => {
            const aPos = CELL_BY_INDEX.get(placement.anchor)!;
            const tPos = CELL_BY_INDEX.get(placement.target)!;
            return (
              <>
                <line
                  class="ingenious-placement-link preview"
                  x1={aPos.cx}
                  y1={aPos.cy}
                  x2={tPos.cx}
                  y2={tPos.cy}
                />
                <g class="ingenious-preview-anchor">
                  {ghostGlyph(placement.anchorColor, aPos.cx, aPos.cy)}
                </g>
                <g class="ingenious-preview" data-role="preview" data-cell={placement.target}>
                  {ghostGlyph(placement.targetColor, tPos.cx, tPos.cy)}
                </g>
              </>
            );
          }}
        </For>

        {/* The selected anchor remains visible. Each green neighbor is one
            complete, legal orientation of the chosen tile. */}
        <Show when={anchor() !== null}>
          <For each={anchoredEnds()}>
            {(placement, index) => {
              const aPos = CELL_BY_INDEX.get(placement.anchor)!;
              const tPos = CELL_BY_INDEX.get(placement.target)!;
              return (
                <>
                  <line
                    class="ingenious-placement-link"
                    x1={aPos.cx}
                    y1={aPos.cy}
                    x2={tPos.cx}
                    y2={tPos.cy}
                  />
                  <Show when={index() === 0}>
                    <g
                      class="ingenious-anchor"
                      data-role="anchor"
                      data-cell={placement.anchor}
                      onClick={() => clickCell(placement.anchor)}
                    >
                      {ghostGlyph(placement.anchorColor, aPos.cx, aPos.cy)}
                    </g>
                  </Show>
                  <g
                    class="ingenious-target"
                    data-role="target"
                    data-cell={placement.target}
                    onClick={() => clickCell(placement.target)}
                  >
                    <polygon
                      class="ingenious-target-slot"
                      points={hexPoints(tPos.cx, tPos.cy, HEX_SIZE * 0.55)}
                    />
                  </g>
                </>
              );
            }}
          </For>
        </Show>
      </svg>

      <div class="ingenious-panel">
        <Show when={currentRack().some((slot) => slot !== null)}>
          <p class="ingenious-hint">
            Pick the color you want to place first, then hover an empty hex to move the whole tile
            through every legal orientation there. Click to anchor that color; the subtle green
            slots show where the second half can go. Click the anchor again to cancel.
          </p>
          <div class="ingenious-rack">
            <For each={rackTypes()}>
              {(t) => {
                const active = (color: Color) => {
                  const s = selectedType();
                  return s !== null && s[0] === t[0] && s[1] === t[1] && anchorColor() === color;
                };
                return (
                  <div class="ingenious-tile">
                    <button
                      type="button"
                      class="ingenious-tile-half"
                      classList={{ active: active(t[0]) }}
                      aria-label={`Place ${t[0]} first`}
                      disabled={props.busy || !canPlace()}
                      onClick={() => selectType(t, t[0])}
                    >
                      <svg viewBox="0 0 40 40" class="ingenious-tile-icon">
                        <rect class="ingenious-tile-icon-bg" width={40} height={40} />
                        <PieceIcon color={t[0]} cx={20} cy={20} r={16} />
                      </svg>
                    </button>
                    <button
                      type="button"
                      class="ingenious-tile-half"
                      classList={{ active: active(t[1]) }}
                      aria-label={`Place ${t[1]} first`}
                      disabled={props.busy || !canPlace()}
                      onClick={() => selectType(t, t[1])}
                    >
                      <svg viewBox="0 0 40 40" class="ingenious-tile-icon">
                        <rect class="ingenious-tile-icon-bg" width={40} height={40} />
                        <PieceIcon color={t[1]} cx={20} cy={20} r={16} />
                      </svg>
                    </button>
                  </div>
                );
              }}
            </For>
          </div>
        </Show>

        <Show when={canKeep() && canSwap()}>
          <div class="ingenious-swap-decision">
            <button
              type="button"
              disabled={props.busy || !canKeep()}
              onClick={() => props.onMove("KeepRack")}
            >
              Keep &amp; refill
            </button>
            <button
              type="button"
              disabled={props.busy || !canSwap()}
              onClick={() => props.onMove("Swap")}
            >
              Swap rack
            </button>
          </div>
        </Show>
      </div>
    </div>
  );
};
