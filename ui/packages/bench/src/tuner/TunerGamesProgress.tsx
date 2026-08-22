/** TunerGamesProgress — live progress view for an active tuner run.
 *
 * Instead of the log tail (which is always-empty JSONL for optuna runs),
 * this shows:
 *   - Trial completion progress (match_count / expected games per trial)
 *   - Worker count from the launch config
 *   - Recent completed games from `GameTraceSummary[]`
 *   - PID of the running process
 *
 * The games list is updated every tail tick, same as the trial/chart data.
 * Games are shown most-recent-first with opponent/outcome, since they form
 * the "this is what's happening right now" signal. */

import { createMemo, For, Show, type Component } from "solid-js";
import type { GameTraceSummary, RunDetail } from "../index.js";

// ── Helpers ───────────────────────────────────────────────────────

function configOverride(config: unknown, key: string): number | null {
  const overrides = (config as { overrides?: unknown } | null)?.overrides;
  if (!Array.isArray(overrides)) return null;
  for (const override of overrides) {
    if (typeof override !== "string" || !override.startsWith(`${key}=`)) continue;
    const value = Number(override.slice(key.length + 1));
    if (Number.isFinite(value) && value > 0) return value;
  }
  return null;
}

function resolveRounds(config: unknown): number {
  return configOverride(config, "target.rounds") ?? 20;
}

function resolveTrials(config: unknown): number | null {
  return configOverride(config, "optimizer.n_trials");
}

function resolveWorkers(config: unknown): string {
  const w = configOverride(config, "optimizer.n_workers");
  return w !== null ? String(w) : "auto";
}

function outcomeLabel(outcome: string | null): string {
  if (!outcome) return "?";
  // Wire outcomes: "A", "B", "draw", "timeout", ...
  switch (outcome) {
    case "A": return "A wins";
    case "B": return "B wins";
    case "draw": return "draw";
    default: return outcome;
  }
}

// ── Component ─────────────────────────────────────────────────────

export const TunerGamesProgress: Component<{
  detail: RunDetail | null;
  games: GameTraceSummary[];
}> = (props) => {
  const matchCount = createMemo(() => props.detail?.match_count ?? 0);
  const trialCount = createMemo(() => props.detail?.trial_count ?? 0);
  const pid = createMemo(() => props.detail?.pid);
  const rounds = createMemo(() => resolveRounds(props.detail?.config));
  const nTrials = createMemo(() => resolveTrials(props.detail?.config));
  const workers = createMemo(() => resolveWorkers(props.detail?.config));

  // Estimate: per trial, `rounds` game pairs per matchmaking step,
  // with 5-15 steps per trial (matchmaking's min_games..max_games).
  // A rough lower bound is rounds * min_games (5) pairs.
  const gamesPerTrialLow = createMemo(() => rounds() * 5);
  const gamesPerTrialHigh = createMemo(() => rounds() * 15);

  const progressPct = createMemo(() => {
    const total = nTrials();
    if (!total || total <= 0) return null;
    return Math.min(100, Math.round((trialCount() / total) * 100));
  });

  const recentGames = createMemo(() => {
    // Show the 10 most recent games, newest first (they arrive newest-first
    // from the server already, but we truncate to keep it compact).
    const gs = props.games;
    if (gs.length === 0) return [];
    return gs.slice(0, Math.min(10, gs.length));
  });

  const isRunning = createMemo(() => props.detail?.status === "running");

  return (
    <div id="tuner-games-progress">
      <div id="tuner-games-header">
        <span class="tuner-games-title">
          {isRunning() ? "Running" : "Completed"}
        </span>
        <span class="tuner-games-meta">
          <span class="tuner-games-stat">
            <strong>{matchCount()}</strong> game{matchCount() === 1 ? "" : "s"}
          </span>
          <span class="tuner-games-stat">
            <strong>{trialCount()}</strong> / {nTrials() ?? "?"} trial{trialCount() === 1 ? "" : "s"}
          </span>
          <Show when={pid()}>
            <span class="tuner-games-stat">PID {pid()}</span>
          </Show>
          <span class="tuner-games-stat">{workers()} workers</span>
        </span>
        <Show when={progressPct() !== null}>
          <span class="tuner-games-pct">{progressPct()}%</span>
        </Show>
      </div>

      <Show when={trialCount() === 0 && matchCount() === 0}>
        <div class="tuner-games-waiting">
          Starting up…
          <Show when={isRunning()}>
            {" "}preflight check in progress
          </Show>
        </div>
      </Show>

      <Show when={trialCount() === 0 && matchCount() > 0}>
        <div class="tuner-games-waiting">
          Evaluating trial 1 — {matchCount()} game{matchCount() === 1 ? "" : "s"} played
          (≈ {gamesPerTrialLow()}–{gamesPerTrialHigh()} expected for first trial)
        </div>
      </Show>

      <Show when={trialCount() > 0}>
        <div class="tuner-games-active">
          <strong>Trial {trialCount() + 1}</strong> in progress after
          {" "}{trialCount()} completed — {matchCount() - trialCount() * gamesPerTrialHigh()} games into it
        </div>
      </Show>

      <Show when={recentGames().length > 0}>
        <div id="tuner-games-list">
          <For each={recentGames()}>
            {(g) => (
              <div class="tuner-games-item" title={JSON.stringify({ game_seq: g.game_seq, seed: g.seed, ply_count: g.ply_count, started_at: g.started_at })}>
                <span class="tuner-games-item-id">#{g.game_seq}</span>
                <span class="tuner-games-item-outcome">{outcomeLabel(g.outcome)}</span>
                {g.ply_count > 0 ? <span class="tuner-games-item-plies">{g.ply_count} ply</span> : null}
              </div>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
};