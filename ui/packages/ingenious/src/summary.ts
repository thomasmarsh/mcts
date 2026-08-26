// summary.ts — Ingenious's `GameSummary`/`formatMove`, the per-game half of
// `GameShell`'s HUD chrome. Score is per-color (not a single number): each
// line shows one player's full six-color vector, since a player's true
// total is their *lowest* color, not a sum -- showing the vector, not a
// single number, is what makes that rule legible at a glance.

import type { GameSummary, HudLine } from "@mcts/game";
import {
  COLORS,
  TARGET_SCORE,
  type Color,
  type GameState,
  type GameView,
  type Move,
} from "./types.js";

/** Neutral per-player accent for the HUD swatch -- Ingenious has no inherent
 * per-player board color the way Black/White stone games do. */
const PLAYER_SWATCH = ["#4a4a4a", "#a0a0a0"];

const COLOR_HEX: Record<Color, string> = {
  Red: "#e0483e",
  Green: "#4caf7a",
  Blue: "#3b82c4",
  Orange: "#d98c2b",
  Yellow: "#e6c229",
  Purple: "#8859a8",
};

const COLOR_ABBREV: Record<Color, string> = {
  Red: "R",
  Green: "G",
  Blue: "B",
  Orange: "O",
  Yellow: "Y",
  Purple: "P",
};

function playerLabel(index: number): string {
  return `P${index}`;
}

function scoreText(score: number[]): string {
  return COLORS.map((c, i) => `${COLOR_ABBREV[c]}${score[i]}`).join(" ");
}

function scoreLines(view: GameView): HudLine[] {
  return view.score.map((score, i) => ({
    id: `score-${i}`,
    text: `${playerLabel(i)} — ${scoreText(score)}`,
    swatch: PLAYER_SWATCH[i],
  }));
}

export function summarize(view: GameView): GameSummary {
  const lines = scoreLines(view);
  if (view.terminal) {
    const bannerText =
      view.winner === null ? "No moves left — draw." : `${playerLabel(view.winner)} wins!`;
    return {
      turnText: "Game over",
      bannerText,
      lines,
      currentPlayer: null,
    };
  }

  const phaseText =
    view.phase === "swap_decision"
      ? `${playerLabel(view.current_player)} to decide: keep or swap rack`
      : `${playerLabel(view.current_player)} to place` +
        (view.pending_bonus > 0 ? ` (bonus placement owed)` : "");

  return {
    turnText: phaseText,
    bannerText: "",
    lines,
    currentPlayer: playerLabel(view.current_player),
  };
}

export function formatMove(move: Move, before: GameState): string {
  const mover = playerLabel(before.current_player);
  if (move === "KeepRack") return `${mover} keeps rack`;
  if (move === "Swap") return `${mover} swaps rack`;
  const { color_a, color_b } = move.Place;
  return `${mover} places ${COLOR_ABBREV[color_a]}/${COLOR_ABBREV[color_b]}`;
}

export { COLOR_HEX };

/** Every color's score reaching `TARGET_SCORE` freezes it (see
 * `games/ingenious/src/lib.rs`'s `bonus_used`) -- exported for the renderer's
 * rack/score display to grey out a maxed-out color consistently with the
 * HUD's own numbers. */
export function isMaxed(score: number, target = TARGET_SCORE): boolean {
  return score >= target;
}
