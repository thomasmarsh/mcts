import type { ExperimentCell, LeaderboardEntry } from "./types.js";

export function formatRate(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

export function formatInterval(lower: number, upper: number): string {
  return `${formatRate(lower)} – ${formatRate(upper)}`;
}

export function formatWld(
  value:
    | Pick<ExperimentCell, "wins" | "losses" | "draws">
    | Pick<LeaderboardEntry, "wins" | "losses" | "draws">,
): string {
  return `${value.wins}/${value.losses}/${value.draws}`;
}

export function formatProgress(completed: number, planned: number): string {
  return `${completed}/${planned}`;
}

export function formatObservedResult(
  value: Pick<ExperimentCell, "completed_games" | "win_rate" | "ci_lower" | "ci_upper">,
): string {
  return value.completed_games === 0
    ? "No games yet"
    : `${formatRate(value.win_rate)} (95% CI ${formatInterval(value.ci_lower, value.ci_upper)})`;
}

export function formatLeaderboardResult(entry: LeaderboardEntry): string {
  return entry.total === 0
    ? "No games yet"
    : `${formatRate(entry.win_rate)} (${formatInterval(entry.ci_lower, entry.ci_upper)})`;
}

export function statusLabel(status: string): string {
  return status.replaceAll("_", " ");
}

export function formatTime(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString() : "Not recorded";
}
