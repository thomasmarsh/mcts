import type {
  JsonValue,
  TuningAttempt,
  TuningGame,
  TuningPair,
  TuningSessionDetail,
  TuningSessionListItem,
  TuningTrial,
} from "../types.js";
import type { TuningSelection } from "../tuning-navigation.js";

export interface ReplayTarget {
  runId: string;
  game: string;
  gameSeq: number;
  live: boolean;
}

export function sessionLabel(session: TuningSessionListItem): string {
  return session.label ?? session.game ?? `Tuning ${session.session_id.slice(0, 12)}`;
}

export function terminalProgress(session: TuningSessionListItem): string {
  const target = session.target_trial_count;
  return target === null
    ? `${session.counts.terminal} terminal trials`
    : `${session.counts.terminal} / ${target} terminal trials`;
}

export function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function formatScore(value: number | null): string {
  return value === null ? "not recorded" : value.toFixed(3);
}

export function formatRating(mu: number | null, sigma: number | null): string {
  return mu === null || sigma === null ? "not recorded" : `${mu.toFixed(3)} ± ${sigma.toFixed(3)}`;
}

export function pairEvidence(pair: TuningPair): string {
  const count = pair.games.length;
  if (pair.status === "failed") return `failed after ${count} of 2 games`;
  if (pair.status === "complete" && count !== 2) return `incomplete: ${count} of 2 games`;
  return `${count} of 2 games · ${pair.status}`;
}

export function opponentLabel(pair: TuningPair): string {
  return pair.opponent.label ?? pair.opponent.anchor_id;
}

export function trialsForAttempt(detail: TuningSessionDetail, attemptId: string): TuningTrial[] {
  return detail.trials.filter((trial) => trial.attempt_id === attemptId);
}

export function selectedAttempt(detail: TuningSessionDetail, selection: TuningSelection): TuningAttempt | null {
  return detail.attempts.find((attempt) => attempt.attempt_id === selection.attemptId) ?? null;
}

export function selectedTrial(detail: TuningSessionDetail, selection: TuningSelection): TuningTrial | null {
  return detail.trials.find((trial) => trial.trial_id === selection.trialId) ?? null;
}

export function selectedPair(detail: TuningSessionDetail, selection: TuningSelection): TuningPair | null {
  return selectedTrial(detail, selection)?.pairs.find((pair) => pair.pair_id === selection.pairId) ?? null;
}

export function selectedGame(detail: TuningSessionDetail, selection: TuningSelection): TuningGame | null {
  return selectedPair(detail, selection)?.games.find((game) => game.game_id === selection.gameId) ?? null;
}

export function jsonText(value: JsonValue | null): string {
  return value === null ? "not recorded" : JSON.stringify(value, null, 2);
}

export function configurationSummary(value: JsonValue | null): string {
  if (value === null) return "configuration not recorded";
  if (typeof value !== "object" || Array.isArray(value)) return JSON.stringify(value);
  const fields = Object.entries(value).slice(0, 3);
  const summary = fields.map(([key, field]) => `${key}=${JSON.stringify(field)}`).join(", ");
  return Object.keys(value).length > fields.length ? `${summary}, …` : summary;
}

export function replayTarget(
  detail: TuningSessionDetail,
  session: TuningSessionListItem | null,
  selection: TuningSelection,
): ReplayTarget | string {
  const game = selectedGame(detail, selection);
  const attempt = selectedAttempt(detail, selection);
  if (!game) return "Select a recorded game to inspect its replay.";
  if (!attempt?.bench_run_id) return "Replay unavailable: the attempt has no associated physical run.";
  if (game.trace_game_seq === null) return "Replay unavailable: this game has no trace sequence.";
  if (!session?.game) return "Replay unavailable: the session did not record its game kind.";
  return { runId: attempt.bench_run_id, game: session.game, gameSeq: game.trace_game_seq, live: attempt.status === "running" };
}
