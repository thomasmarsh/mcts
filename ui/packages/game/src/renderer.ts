// renderer.ts — Game-agnostic renderer/GameShell contract (PLAN-UI.md session
// 4). `packages/druid` (and later `packages/ttt`, session 8) implement
// `GameKindModule` once per game; `app/src/GameShell.tsx` consumes a
// `Record<string, GameKindModule>` registry and never names a concrete S/M/V
// itself -- the TS-side mirror of `GameAdapter`'s `Value` erasure on the Rust
// side (this package's `types.ts` is `S`/`M`/`V` generic for the same reason).

import type { Component } from "solid-js";

/** One step along the root-to-current path: the move applied and the state
 * it was applied *from* (not the resulting state -- that's either the next
 * step's `before`, or the renderer's own `state` prop for the last step).
 * Carrying `before` (not just `move`) lets a renderer derive per-move
 * metadata a bare move list can't -- e.g. Druid's stack reconstruction needs
 * to know which player made each move, and the mover is `before.player`, not
 * something recoverable from `move` alone. */
export interface MoveStep<S, M> {
  move: M;
  before: S;
}

/** One candidate move from `analyze`, positioned for a heatmap overlay.
 * `visitShare` is `visits / total_visits`, pre-divided so renderers never
 * need `total_visits` themselves. Reserved for session 6 -- no renderer
 * reads this yet. */
export interface AnalysisOverlayEntry<M> {
  move: M;
  visitShare: number;
  isProven: boolean;
}

/** Everything a board renderer needs to draw one game and report
 * interaction back up to `GameShell`. `hoveredMove` is a controlled prop
 * (not renderer-local state) on purpose: session 6's analysis panel will
 * want to preview a candidate move on the board by hovering its row, which
 * only works if `GameShell` (not the renderer) owns the value. */
export interface GameRendererProps<S, M, V> {
  /** Current state, after `history`'s last move. */
  state: S;
  /** Display-only companion to `state` -- `terminal`/`winner`-shaped fields
   * that only exist on the view, not the raw state (see
   * `server/adapters/mod.rs`'s `GameAdapter::view`). Druid's renderer needs
   * this for its minimap's turn ring/winner glow; a renderer with no such
   * chrome is free to ignore it. */
  view: V;
  /** Root-to-current path, oldest first. Empty at the root/start of a new game. */
  history: MoveStep<S, M>[];
  /** Legal moves to make pickable -- already mode-filtered by `GameShell`
   * (see `GameModeDef`), so the renderer highlights/picks from exactly what
   * it's given without needing its own move-kind knowledge. */
  legalMoves: M[];
  /** True while a move/AI/analysis request is in flight -- renderers should
   * stop offering picks (no highlights, no ghost) while true, same as
   * app.js's `busy` flag. */
  busy: boolean;
  onMove: (move: M) => void;
  hoveredMove: M | null;
  onHover: (move: M | null) => void;
  analysisOverlay?: AnalysisOverlayEntry<M>[];
}

export type GameRendererComponent<S, M, V> = Component<GameRendererProps<S, M, V>>;

/** One row of a `GameShell` HUD panel (e.g. Druid's "Black — 8 sarsens, 4
 * lintels"). `swatch` is a CSS color for the small dot app.js draws before
 * each hand line -- omit it for rows that don't want one. */
export interface HudLine {
  id: string;
  text: string;
  swatch?: string;
}

/** What `GameShell`'s chrome needs to display for one position -- the
 * game-specific half of app.js's `updateHud`. Computed from a game's `view`
 * (not its raw `state`): `terminal`/`winner` only exist on the view. */
export interface GameSummary {
  turnText: string;
  bannerText: string;
  bannerColor?: string;
  lines: HudLine[];
  /** Whose turn it is (one of the module's own `players` ids), or `null` if
   * the position is terminal. `GameShell` uses this (not `turnText`, which
   * is display-only prose) to look up `seats[currentPlayer]` and decide
   * whether to auto-trigger an AI move. */
  currentPlayer: string | null;
}

/** A placement-mode button (Druid's Sarsen/Lintel-H/Lintel-V). `GameShell`
 * renders one button per mode and filters `legalMoves` down to `filter`'s
 * matches before handing them to the renderer -- the renderer itself never
 * needs to know a game has modes at all. Games with no meaningful subdivision
 * of their move space (tic-tac-toe) simply omit `modes` on their module. */
export interface GameModeDef<M> {
  id: string;
  label: string;
  hotkey?: string;
  filter: (move: M) => boolean;
}

/** Everything `GameShell` needs to host one game kind. One instance per
 * game, registered in `app/src`'s `Record<string, GameKindModule>` (session
 * 8 adds a second entry; nothing here needs to change to support that). */
export interface GameKindModule<S, M, V> {
  kind: string;
  /** Fixed player ids for this game (Druid: `["Black", "White"]`), used to
   * build seat pickers/autoplay generically without `GameShell` knowing what
   * a "player" is called in any particular game. */
  players: string[];
  Renderer: GameRendererComponent<S, M, V>;
  summarize: (view: V) => GameSummary;
  modes?: GameModeDef<M>[];
  /** Board-size/etc. new-game config editor. Omit for games with no
   * meaningful config (a fixed board). */
  NewGameFields?: Component<{ config: unknown; onChange: (config: unknown) => void }>;
  /** Human-readable label for one move (session 5's move-list panel), given
   * the state it was applied *from* -- mirrors `MoveStep`'s own shape, since
   * some games' labels need board context a bare move can't carry (Druid's
   * needs `before.size` to turn a board index into a coordinate). Falls back
   * to `JSON.stringify(move)` for a game module that omits this. */
  formatMove?: (move: M, before: S) => string;
}
