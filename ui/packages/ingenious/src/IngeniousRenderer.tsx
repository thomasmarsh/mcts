// IngeniousRenderer.tsx — SVG hex-grid board plus a rack/swap side panel.
//
// A tile has two colors and no inherent "up" side, so picking one from the
// rack alone doesn't fully determine a move -- the player also has to say
// which of the tile's two ends goes on which of the two empty cells. This
// renderer resolves that with a tile-type selection (`selectedType`) plus a
// `flipped` toggle that swaps which end is "primary", rather than asking the
// player to disambiguate per board location: once a type and a flip state
// are chosen, `color_a`/`color_b` are pinned, so every remaining
// `legalMoves` entry of that exact type maps to exactly one empty-cell-pair
// location on the board, with no further ambiguity to resolve by clicking.

import { type Component, createMemo, createSignal, For, Show } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import { centerOf, neighborOf, VALID_CELLS } from "./geometry.js";
import { COLOR_HEX } from "./summary.js";
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

function hexPoints(cx: number, cy: number, size: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i - 30);
    pts.push(`${cx + size * Math.cos(angle)},${cy + size * Math.sin(angle)}`);
  }
  return pts.join(" ");
}

interface CellLayout {
  index: number;
  cx: number;
  cy: number;
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

function placeMoveEquals(a: PlaceMove, b: PlaceMove): boolean {
  return a.cell === b.cell && a.dir === b.dir && a.color_a === b.color_a && a.color_b === b.color_b;
}

export const IngeniousRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (
  props,
) => {
  const [selectedType, setSelectedType] = createSignal<[Color, Color] | null>(null);
  const [flipped, setFlipped] = createSignal(false);

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

  // Drop the current selection once it's no longer offered -- a new turn, or
  // this exact type left the rack after being placed.
  createMemo(() => {
    const t = selectedType();
    if (t && !rackTypes().some(([a, b]) => a === t[0] && b === t[1])) {
      setSelectedType(null);
      setFlipped(false);
    }
  });

  const effectiveColors = createMemo(() => {
    const t = selectedType();
    if (!t) return null;
    return flipped() ? { color_a: t[1], color_b: t[0] } : { color_a: t[0], color_b: t[1] };
  });

  const placementMoves = createMemo<PlaceMove[]>(() => {
    const eff = effectiveColors();
    if (!eff) return [];
    return props.legalMoves
      .filter(isPlaceMove)
      .map((m) => m.Place)
      .filter((mv) => mv.color_a === eff.color_a && mv.color_b === eff.color_b);
  });

  const canPlace = createMemo(() => props.legalMoves.some(isPlaceMove));
  const canKeep = createMemo(() => props.legalMoves.includes("KeepRack"));
  const canSwap = createMemo(() => props.legalMoves.includes("Swap"));

  function selectType(t: [Color, Color]): void {
    if (props.busy || !canPlace()) return;
    const current = selectedType();
    if (current && current[0] === t[0] && current[1] === t[1]) {
      setSelectedType(null);
      setFlipped(false);
    } else {
      setSelectedType(t);
      setFlipped(false);
    }
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
              <polygon
                class="ingenious-hex"
                points={hexPoints(cell.cx, cell.cy, HEX_SIZE)}
                style={{ fill: color() ? COLOR_HEX[color() as Color] : undefined }}
              />
            );
          }}
        </For>
        <For each={placementMoves()}>
          {(mv) => {
            const cellPos = CELL_BY_INDEX.get(mv.cell);
            const nbIndex = neighborOf(mv.cell, mv.dir);
            const nbPos = nbIndex !== null ? CELL_BY_INDEX.get(nbIndex) : undefined;
            if (!cellPos || !nbPos) return null;
            const move: Move = { Place: mv };
            const isHovered = () =>
              props.hoveredMove !== null &&
              isPlaceMove(props.hoveredMove) &&
              placeMoveEquals(props.hoveredMove.Place, mv);
            return (
              <g
                class="ingenious-placement"
                classList={{ hovered: isHovered() }}
                onClick={() => !props.busy && props.onMove(move)}
                onMouseEnter={() => props.onHover(move)}
                onMouseLeave={() => props.onHover(null)}
              >
                <line
                  class="ingenious-placement-link"
                  x1={cellPos.cx}
                  y1={cellPos.cy}
                  x2={nbPos.cx}
                  y2={nbPos.cy}
                />
                <circle
                  class="ingenious-ghost"
                  cx={cellPos.cx}
                  cy={cellPos.cy}
                  r={HEX_SIZE * 0.55}
                  style={{ fill: COLOR_HEX[mv.color_a] }}
                />
                <circle
                  class="ingenious-ghost"
                  cx={nbPos.cx}
                  cy={nbPos.cy}
                  r={HEX_SIZE * 0.55}
                  style={{ fill: COLOR_HEX[mv.color_b] }}
                />
              </g>
            );
          }}
        </For>
      </svg>

      <div class="ingenious-panel">
        <Show when={currentRack().some((slot) => slot !== null)}>
          <div class="ingenious-rack">
            <For each={rackTypes()}>
              {(t) => {
                const active = () => {
                  const s = selectedType();
                  return s !== null && s[0] === t[0] && s[1] === t[1];
                };
                return (
                  <button
                    type="button"
                    class="ingenious-tile"
                    classList={{ active: active() }}
                    disabled={props.busy || !canPlace()}
                    onClick={() => selectType(t)}
                  >
                    <span class="ingenious-tile-half" style={{ background: COLOR_HEX[t[0]] }} />
                    <span class="ingenious-tile-half" style={{ background: COLOR_HEX[t[1]] }} />
                  </button>
                );
              }}
            </For>
            <Show when={selectedType() !== null && selectedType()![0] !== selectedType()![1]}>
              <button
                type="button"
                class="ingenious-flip"
                disabled={props.busy}
                onClick={() => setFlipped((f) => !f)}
              >
                Flip
              </button>
            </Show>
          </div>
        </Show>

        <Show when={canKeep() || canSwap()}>
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
