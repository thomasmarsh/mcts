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

  // Ingenious tiles have no rotation control of their own: every legal
  // (cell, direction) pair for the selected type/flip is offered at once as
  // a clickable overlay directly on the board, so picking a location *is*
  // picking the rotation. When several overlays share an anchor cell (the
  // tile can go in more than one direction from the same empty hex), dim
  // every overlay except the one under the pointer so hovering around that
  // anchor reads as "cycling rotations" instead of a solid smear of color.
  const anyGhostHovered = createMemo(
    () => props.hoveredMove !== null && isPlaceMove(props.hoveredMove),
  );

  // How many currently-offered rotations touch each cell. A rotation whose
  // hex appears here more than once is sharing that hex with a sibling
  // rotation -- if both siblings' full domino shapes were independently
  // clickable there, whichever painted last would silently win every click
  // in that spot, making the other rotation unreachable no matter where you
  // aim near it. Used to withhold a contested hex's hit-region from a
  // rotation until it's already the hovered one (see the per-hex handlers
  // below), so disambiguation always has to go through each rotation's own
  // unshared hex first.
  const contestedCounts = createMemo(() => {
    const counts = new Map<number, number>();
    for (const mv of placementMoves()) {
      const nb = neighborOf(mv.cell, mv.dir);
      counts.set(mv.cell, (counts.get(mv.cell) ?? 0) + 1);
      if (nb !== null) counts.set(nb, (counts.get(nb) ?? 0) + 1);
    }
    return counts;
  });

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
              <>
                <polygon class="ingenious-hex" points={hexPoints(cell.cx, cell.cy, HEX_SIZE)} />
                <Show when={color()}>
                  <PieceIcon color={color() as Color} cx={cell.cx} cy={cell.cy} r={HEX_SIZE} />
                </Show>
              </>
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
            const dimmed = () => anyGhostHovered() && !isHovered();

            // A hex only accepts pointer interaction on this rotation's
            // behalf while it isn't contested, or once this exact rotation
            // is already hovered (so the second click, or a click anywhere
            // else on an already-previewed domino, still works normally).
            const counts = contestedCounts();
            const cellLive = () => (counts.get(mv.cell) ?? 0) <= 1 || isHovered();
            const nbLive = () => nbIndex === null || (counts.get(nbIndex) ?? 0) <= 1 || isHovered();

            function pick(live: () => boolean): void {
              if (!props.busy && live()) props.onMove(move);
            }
            function hover(live: () => boolean): void {
              if (live()) props.onHover(move);
            }

            return (
              <g class="ingenious-placement" classList={{ hovered: isHovered(), dimmed: dimmed() }}>
                <line
                  class="ingenious-placement-link"
                  x1={cellPos.cx}
                  y1={cellPos.cy}
                  x2={nbPos.cx}
                  y2={nbPos.cy}
                />
                <g
                  class="ingenious-ghost"
                  classList={{ "ingenious-ghost-hit": cellLive() }}
                  onClick={() => pick(cellLive)}
                  onMouseEnter={() => hover(cellLive)}
                  onMouseLeave={() => props.onHover(null)}
                >
                  <polygon
                    class="ingenious-ghost-hex"
                    points={hexPoints(cellPos.cx, cellPos.cy, HEX_SIZE * 0.82)}
                  />
                  <PieceIcon
                    color={mv.color_a}
                    cx={cellPos.cx}
                    cy={cellPos.cy}
                    r={HEX_SIZE * 0.82}
                  />
                </g>
                <g
                  class="ingenious-ghost"
                  classList={{ "ingenious-ghost-hit": nbLive() }}
                  onClick={() => pick(nbLive)}
                  onMouseEnter={() => hover(nbLive)}
                  onMouseLeave={() => props.onHover(null)}
                >
                  <polygon
                    class="ingenious-ghost-hex"
                    points={hexPoints(nbPos.cx, nbPos.cy, HEX_SIZE * 0.82)}
                  />
                  <PieceIcon color={mv.color_b} cx={nbPos.cx} cy={nbPos.cy} r={HEX_SIZE * 0.82} />
                </g>
              </g>
            );
          }}
        </For>
      </svg>

      <div class="ingenious-panel">
        <Show when={currentRack().some((slot) => slot !== null)}>
          <p class="ingenious-hint">
            Pick a tile below, flip it if it has two colors, then click a highlighted hex pair on
            the board -- each highlighted pair is a different valid rotation for that tile. When
            rotations overlap at one hex, hover the hex that's unique to the one you want first.
          </p>
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
                    <span class="ingenious-tile-half">
                      <svg viewBox="0 0 40 40" class="ingenious-tile-icon">
                        <rect class="ingenious-tile-icon-bg" width={40} height={40} />
                        <PieceIcon color={t[0]} cx={20} cy={20} r={16} />
                      </svg>
                    </span>
                    <span class="ingenious-tile-half">
                      <svg viewBox="0 0 40 40" class="ingenious-tile-icon">
                        <rect class="ingenious-tile-icon-bg" width={40} height={40} />
                        <PieceIcon color={t[1]} cx={20} cy={20} r={16} />
                      </svg>
                    </span>
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
