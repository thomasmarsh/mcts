// CongoRenderer.tsx — 7×7 board renderer for Congo.
//
// Two-click interaction (click source piece, then click destination) for
// every piece but a Monkey mid-capture-chain, same protocol as
// ChessVariantRenderer. A Monkey's jump-chain is different: several distinct
// legal moves can share a (from, to) pair while capturing different pieces
// along the way (the chain can revisit squares and fork), so naming just a
// final destination is ambiguous -- which piece the player keeps alive is a
// real strategic choice, not something a heuristic (e.g. "prefer more
// captures") should make for them. So once a click lands on a square that's
// only reachable through such a fork, the renderer switches into
// click-through-the-chain mode: each click either extends the chain by one
// hop (if there's more than one legal way to continue, or a real choice
// between stopping and continuing) or re-clicking the Monkey's current
// square confirms stopping there. Whenever a click leaves only one possible
// full move, it's submitted immediately without waiting for a redundant
// confirmation click -- this degrades to a plain two-click move for every
// piece that has no forks, which is every piece except a Monkey with
// multiple capture options.
//
// This needs `Move.hops`, the chain's *ordered* landing-square sequence
// (`games/congo/src/lib.rs`'s `Move.captures` is deliberately a sorted set,
// order-independent, so it alone can't tell two forking paths apart).
//
// Board orientation: the engine's mailbox has row 0 = Black's home rank
// (see `games/congo/src/lib.rs`'s module doc comment) and row 6 = White's,
// so the display grid maps 1:1 onto the engine index (row-major, no flip)
// with Black's castle at the top of the screen.

import { type Component, createEffect, createMemo, createSignal, For } from "solid-js";
import type { GameRendererProps } from "@mcts/game";
import type { Cell, GameState, GameView, Move, PieceCode } from "./types.js";
import { RIVER_ROW, SIZE } from "./types.js";
import "./congo.css";

import lionIcon from "./pieces/lion.svg";
import elephantIcon from "./pieces/elephant.svg";
import giraffeIcon from "./pieces/giraffe.svg";
import crocodileIcon from "./pieces/crocodile.svg";
import zebraIcon from "./pieces/zebra.svg";
import monkeyIcon from "./pieces/monkey.svg";
import pawnIcon from "./pieces/pawn.svg";

const PIECE_ICON: Record<PieceCode, string> = {
  giraffe: giraffeIcon,
  monkey: monkeyIcon,
  elephant: elephantIcon,
  lion: lionIcon,
  crocodile: crocodileIcon,
  zebra: zebraIcon,
  pawn: pawnIcon,
  superpawn: pawnIcon,
};

function inCastle(row: number, col: number): boolean {
  return (row <= 2 || row >= 4) && col >= 2 && col <= 4;
}

function zoneOf(row: number, col: number): "river" | "castle" | "grass" {
  if (row === RIVER_ROW) return "river";
  if (inCastle(row, col)) return "castle";
  return "grass";
}

/** Does `hops` start with exactly `prefix` (element-wise)? */
function hasPrefix(hops: number[], prefix: number[]): boolean {
  if (hops.length < prefix.length) return false;
  return prefix.every((sq, i) => hops[i] === sq);
}

/** The captured square between two board indices exactly one jump apart
 * (differing by 0 or ±2 in each of row/col) -- the midpoint. */
function hopMidpoint(a: number, b: number): number {
  const ar = Math.floor(a / SIZE);
  const ac = a % SIZE;
  const br = Math.floor(b / SIZE);
  const bc = b % SIZE;
  return ((ar + br) / 2) * SIZE + (ac + bc) / 2;
}

export const CongoRenderer: Component<GameRendererProps<GameState, Move, GameView>> = (props) => {
  /** Engine index of the currently selected source piece, or null. */
  const [selectedSource, setSelectedSource] = createSignal<number | null>(null);
  /** Landing squares confirmed so far in an in-progress Monkey chain, oldest
   * first. Empty whenever nothing beyond the source has been confirmed yet. */
  const [chainPath, setChainPath] = createSignal<number[]>([]);
  /** Purely visual hover tracking, decoupled from `props.hoveredMove` (which
   * can't represent a still-ambiguous partial chain -- see `onMouseEnter`). */
  const [hoveredSquare, setHoveredSquare] = createSignal<number | null>(null);
  /** Set when hovering a legal next-hop square whose eventual move is still
   * ambiguous, so at least the immediate capture can be previewed. */
  const [hoverExtraCapture, setHoverExtraCapture] = createSignal<number | null>(null);

  const movesBySrc = createMemo(() => {
    const map = new Map<number, Move[]>();
    for (const mv of props.legalMoves) {
      const list = map.get(mv.from) ?? [];
      list.push(mv);
      map.set(mv.from, list);
    }
    return map;
  });

  const movablePieceSet = createMemo(() => new Set(movesBySrc().keys()));

  /** Legal moves from the selected source whose `hops` still match the
   * confirmed `chainPath` prefix -- i.e. still-reachable outcomes from here. */
  const candidates = createMemo(() => {
    const src = selectedSource();
    if (src === null) return [];
    const prefix = chainPath();
    return (movesBySrc().get(src) ?? []).filter((mv) => hasPrefix(mv.hops, prefix));
  });

  /** The Monkey's (or any piece's) current virtual position: the last
   * confirmed hop, or the source square if none confirmed yet. */
  const currentPos = createMemo(() => {
    const path = chainPath();
    if (path.length === 0) return selectedSource();
    return path[path.length - 1] ?? selectedSource();
  });

  /** True if stopping right here (without extending further) is itself a
   * legal move. */
  const canStopHere = createMemo(() => {
    const prefix = chainPath();
    return prefix.length > 0 && candidates().some((mv) => mv.hops.length === prefix.length);
  });

  /** Squares one further hop away that some candidate still reaches. */
  const nextHopSquares = createMemo(() => {
    const prefix = chainPath();
    const set = new Set<number>();
    for (const mv of candidates()) {
      const next = mv.hops[prefix.length];
      if (mv.hops.length > prefix.length && next !== undefined) set.add(next);
    }
    return set;
  });

  const legalDstForSelected = createMemo(() => {
    const set = new Set(nextHopSquares());
    if (canStopHere()) {
      const cur = currentPos();
      if (cur !== null) set.add(cur);
    }
    return set;
  });

  /** Squares already captured by the confirmed part of an in-progress chain. */
  const confirmedCaptures = createMemo(() => {
    const src = selectedSource();
    const path = chainPath();
    const caps = new Set<number>();
    if (src === null) return caps;
    let prev = src;
    for (const step of path) {
      caps.add(hopMidpoint(prev, step));
      prev = step;
    }
    return caps;
  });

  const overlayByDst = createMemo(() => {
    const map = new Map<number, { visitShare: number; isProven: boolean; isSuggested: boolean }>();
    for (const entry of props.analysisOverlay ?? []) map.set(entry.move.to, entry);
    return map;
  });

  /** Squares captured by the currently hovered/resolved move, for a preview
   * ring -- merged with the chain's already-confirmed captures and (when the
   * hovered move is still ambiguous) the immediate hop's own capture. */
  const hoveredCaptures = createMemo(() => {
    const set = new Set(props.hoveredMove?.captures ?? []);
    for (const sq of confirmedCaptures()) set.add(sq);
    const extra = hoverExtraCapture();
    if (extra !== null) set.add(extra);
    return set;
  });

  createEffect(() => {
    void props.legalMoves; // track this dependency
    setSelectedSource(null);
    setChainPath([]);
  });

  function cellAt(idx: number): Cell | null {
    return props.state.squares[idx] ?? null;
  }

  /** Clears selection state, including hover-preview state left over from
   * the click that just submitted a move -- a plain `onMouseLeave` never
   * fires for that click, so without this a capture-preview ring can be
   * left pointing at a now-empty square until the mouse actually moves. */
  function resetSelection(): void {
    setSelectedSource(null);
    setChainPath([]);
    setHoverExtraCapture(null);
    props.onHover(null);
  }

  function onCellClick(idx: number): void {
    if (props.busy) return;
    const src = selectedSource();

    if (src === null) {
      if (movablePieceSet().has(idx)) setSelectedSource(idx);
      return;
    }

    const prefix = chainPath();
    const cur = currentPos();

    // Re-clicking the Monkey's current square confirms stopping the chain
    // here, when that's itself a legal move.
    if (prefix.length > 0 && idx === cur && canStopHere()) {
      const mv = candidates().find((m) => m.hops.length === prefix.length);
      if (mv) {
        props.onMove(mv);
        resetSelection();
        return;
      }
    }

    // Clicking a legal next-hop square extends the chain -- or, if that
    // leaves only one possible full move, submits it immediately.
    if (nextHopSquares().has(idx)) {
      const newPrefix = [...prefix, idx];
      const newCandidates = candidates().filter((mv) => hasPrefix(mv.hops, newPrefix));
      if (newCandidates.length === 1 && newCandidates[0]) {
        props.onMove(newCandidates[0]);
        resetSelection();
      } else {
        setChainPath(newPrefix);
      }
      return;
    }

    if (idx === src && prefix.length === 0) {
      resetSelection();
      return;
    }
    if (movablePieceSet().has(idx)) {
      setSelectedSource(idx);
      setChainPath([]);
      return;
    }
    resetSelection();
  }

  function isLegalDst(idx: number): boolean {
    if (props.busy) return false;
    return legalDstForSelected().has(idx);
  }

  return (
    <div class="cg-board">
      <div class="cg-grid">
        <For each={Array.from({ length: SIZE * SIZE }, (_, i) => i)}>
          {(idx) => {
            const row = Math.floor(idx / SIZE);
            const col = idx % SIZE;
            const zone = zoneOf(row, col);
            const cell = () => cellAt(idx);
            const isMovable = () => movablePieceSet().has(idx) && cell() !== null;
            const legal = () => isLegalDst(idx);
            const overlay = () => overlayByDst().get(idx);
            const heat = () => overlay()?.visitShare ?? 0;
            const hovered = () => !props.busy && hoveredSquare() === idx;
            const captureTarget = () => hoveredCaptures().has(idx);
            const isCurrentPos = () => selectedSource() !== null && currentPos() === idx;
            const isGhost = () => !cell() && legal() && hovered();
            const isVirtualPiece = () => chainPath().length > 0 && isCurrentPos() && !cell();
            const atRisk = () => (props.state.river_since[idx] ?? 0) > 0 && cell() !== null;
            const isCapture = () => {
              const prefix = chainPath();
              return candidates().some(
                (mv) =>
                  mv.hops.length > prefix.length &&
                  mv.hops[prefix.length] === idx &&
                  mv.captures.length > prefix.length,
              );
            };

            return (
              <button
                type="button"
                class="cg-cell"
                classList={{
                  [`cg-${zone}`]: true,
                  "cg-checker": zone === "grass" && (row + col) % 2 === 0,
                  "cg-movable": !props.busy && selectedSource() === null && isMovable(),
                  "cg-selected-source": isCurrentPos(),
                  legal: legal(),
                  hovered: hovered(),
                  "capture-legal": legal() && isCapture(),
                  "capture-target": captureTarget(),
                  heat: overlay() !== undefined,
                  proven: overlay()?.isProven ?? false,
                  suggested: overlay()?.isSuggested ?? false,
                }}
                style={{ "--heat": String(heat()) }}
                disabled={
                  props.busy ||
                  (selectedSource() === null
                    ? !isMovable()
                    : !legal() && idx !== selectedSource() && !isCurrentPos() && !isMovable())
                }
                onClick={() => onCellClick(idx)}
                onMouseEnter={() => {
                  if (props.busy) return;
                  setHoveredSquare(idx);
                  const src = selectedSource();
                  if (src === null) return;
                  const prefix = chainPath();
                  const cur = currentPos();
                  if (idx === cur && canStopHere()) {
                    const mv = candidates().find((m) => m.hops.length === prefix.length);
                    props.onHover(mv ?? null);
                    setHoverExtraCapture(null);
                    return;
                  }
                  if (nextHopSquares().has(idx)) {
                    const matching = candidates().filter(
                      (mv) => mv.hops.length > prefix.length && mv.hops[prefix.length] === idx,
                    );
                    if (matching.length === 1 && matching[0]) {
                      props.onHover(matching[0]);
                      setHoverExtraCapture(null);
                    } else {
                      props.onHover(null);
                      setHoverExtraCapture(cur !== null ? hopMidpoint(cur, idx) : null);
                    }
                    return;
                  }
                  props.onHover(null);
                  setHoverExtraCapture(null);
                }}
                onMouseLeave={() => {
                  setHoveredSquare(null);
                  setHoverExtraCapture(null);
                  props.onHover(null);
                }}
              >
                {(() => {
                  const c = cell();
                  if (c) {
                    return (
                      <span class={`cg-piece cg-piece-${c.player.toLowerCase()}`}>
                        <img class="cg-piece-icon" src={PIECE_ICON[c.piece]} alt={c.piece} draggable={false} />
                        {c.piece === "superpawn" ? <span class="cg-badge">★</span> : null}
                        {atRisk() ? <span class="cg-risk-ring" /> : null}
                      </span>
                    );
                  }
                  if (isVirtualPiece() || (isGhost() && selectedSource() !== null)) {
                    const srcCell = cellAt(selectedSource() as number);
                    if (srcCell) {
                      return (
                        <span class={`cg-piece cg-piece-ghost cg-piece-${srcCell.player.toLowerCase()}`}>
                          <img class="cg-piece-icon" src={PIECE_ICON[srcCell.piece]} alt="" draggable={false} />
                        </span>
                      );
                    }
                  }
                  return null;
                })()}
              </button>
            );
          }}
        </For>
      </div>
    </div>
  );
};
