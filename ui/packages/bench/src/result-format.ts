function formatRate(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

function formatInterval(lower: number, upper: number): string {
  return `${formatRate(lower)} – ${formatRate(upper)}`;
}

export function formatProgress(completed: number, planned: number): string {
  return `${completed}/${planned}`;
}

export function formatObservedResult(value: {
  completed_games: number;
  win_rate: number;
  ci_lower: number;
  ci_upper: number;
}): string {
  return value.completed_games === 0
    ? "No games yet"
    : `${formatRate(value.win_rate)} (95% CI ${formatInterval(value.ci_lower, value.ci_upper)})`;
}

export function statusLabel(status: string): string {
  return status.replaceAll("_", " ");
}

export function formatTime(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString() : "Not recorded";
}
