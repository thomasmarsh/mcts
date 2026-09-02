// pair-model.ts — pure derivation behind the Run Evidence pair inspector.
// Turns the two seat-swapped game summaries of one pair into candidate-
// relative rows plus a headline W/D/L / ply / compute total. The tuner
// records no per-ply move traces, so this is summary-only; `hasTrace` is
// the seam a board-playback view would branch on and is always false today.

import type { ProjectionGameRow, ProjectionPairRow } from "../tuner-types.js";

export type PairResult = "win" | "loss" | "draw" | "unknown";

export interface PairGameView {
  gameId: string;
  shortGameId: string;
  /** which seat the candidate held. */
  side: string;
  result: PairResult;
  resultLabel: string;
  plies: number;
  elapsedMs: number;
  candidateIterations: number;
  opponentIterations: number;
  hasTrace: boolean;
}

export interface PairInspectorView {
  pairId: string;
  shortPairId: string;
  phase: string;
  candidateId: string;
  opponentId: string;
  taskId: string;
  pairUtility: number;
  games: PairGameView[];
  wins: number;
  draws: number;
  losses: number;
  totalPlies: number;
  totalElapsedMs: number;
  candidateIterations: number;
  opponentIterations: number;
}

function classify(outcome: string): { result: PairResult; label: string } {
  const o = outcome.toLowerCase();
  if (o === "draw" || o === "tie") return { result: "draw", label: "Draw" };
  if (o.includes("candidate") && o.includes("win")) return { result: "win", label: "Candidate win" };
  if (o.includes("opponent") && o.includes("win")) return { result: "loss", label: "Opponent win" };
  if (o === "win") return { result: "win", label: "Candidate win" };
  if (o === "loss") return { result: "loss", label: "Opponent win" };
  return { result: "unknown", label: outcome };
}

const short = (id: string): string => id.replace(/^(pair|game|candidate|opponent|task)-/, "").slice(0, 12);

export function derivePairInspector(
  pair: ProjectionPairRow,
  games: ProjectionGameRow[],
): PairInspectorView {
  const rows: PairGameView[] = games.map((g) => {
    const { result, label } = classify(g.outcome);
    return {
      gameId: g.game_id,
      shortGameId: short(g.game_id),
      side: g.candidate_side,
      result,
      resultLabel: label,
      plies: g.plies,
      elapsedMs: g.elapsed_ms,
      candidateIterations: g.candidate_iterations_total,
      opponentIterations: g.opponent_iterations_total,
      hasTrace: false,
    };
  });
  return {
    pairId: pair.pair_id,
    shortPairId: short(pair.pair_id),
    phase: pair.phase,
    candidateId: pair.candidate_id,
    opponentId: pair.opponent_id,
    taskId: pair.task_id,
    pairUtility: pair.pair_utility,
    games: rows,
    wins: rows.filter((r) => r.result === "win").length,
    draws: rows.filter((r) => r.result === "draw").length,
    losses: rows.filter((r) => r.result === "loss").length,
    totalPlies: rows.reduce((n, r) => n + r.plies, 0),
    totalElapsedMs: rows.reduce((n, r) => n + r.elapsedMs, 0),
    candidateIterations: rows.reduce((n, r) => n + r.candidateIterations, 0),
    opponentIterations: rows.reduce((n, r) => n + r.opponentIterations, 0),
  };
}
