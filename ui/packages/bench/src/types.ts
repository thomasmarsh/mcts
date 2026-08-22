// types.ts — Wire types mirroring server/bench/mod.rs's response shapes and
// query parameters. Field names match the JSON wire format exactly
// (snake_case), same convention as @mcts/game's types.ts: the Rust structs
// don't set `#[serde(rename_all)]`, so there's no camelCase translation
// layer anywhere. Kept separate from both api-client.ts and reducer.ts so
// neither needs to import from the other.

/** Values `runs.status` can take (see the schema comment in
 * src/bench/schema.rs: running | completed | crashed | stopped). The open
 * `string` union member keeps a status added server-side later from
 * failing to deserialize here. */
export type RunStatus = "running" | "completed" | "completed_with_errors" | "crashed" | "stopped" | (string & {});

export type Budget =
  | { kind: "iterations"; value: number }
  | { kind: "time_per_move_ms"; value: number };

export interface NamedStrategyConfig { id: string; label: string; config: Record<string, unknown> }
export interface ExperimentGame { game: string; game_config: unknown }
export interface ExperimentSpecV1 {
  version: 1;
  games: ExperimentGame[];
  baseline: NamedStrategyConfig;
  variants: NamedStrategyConfig[];
  budgets: Budget[];
  rounds_per_cell: number;
  base_seed: number;
  max_parallel_cells: number;
}
export interface ValidationField { path: string; message: string }
export interface Project { project_id: string; name: string; description: string; archived: boolean; created_at: string; updated_at: string }
export interface Experiment { experiment_id: string; project_id: string; name: string; description: string; spec: ExperimentSpecV1; created_at: string; updated_at: string }
export interface ExperimentCell {
  cell_id: string; cell_seed: number | null; game: string; game_config: unknown; variant_id: string; variant_label: string;
  candidate_config: Record<string, unknown>; baseline_id: string; baseline_label: string;
  baseline_config: Record<string, unknown>; budget: Budget; rounds: number; planned_games: number;
  completed_games: number; status: string; started_at: string | null; ended_at: string | null; error: string | null;
  wins: number; losses: number; draws: number; win_rate: number; ci_lower: number; ci_upper: number;
}

export interface BenchSpectatorProps {
  runId: string;
  game: string;
  kind: string;
  live: boolean;
  cellId?: string;
  initialGameSeq?: number;
}

/** A run is finished once its status leaves "running". Its log file is
 * complete at that point — the run process writes it directly, so process
 * exit means fully flushed — which is what lets a log tail stop polling
 * instead of re-checking forever. */
export function isTerminalStatus(status: RunStatus): boolean {
  return status !== "running";
}

/** `GET /api/bench/runs` element. */
export interface RunSummary {
  run_id: string;
  kind: string;
  game: string | null;
  project_id: string | null;
  experiment_id: string | null;
  label: string | null;
  git_sha: string;
  git_dirty: boolean;
  host: string;
  pid: number | null;
  started_at: string;
  ended_at: string | null;
  status: RunStatus;
  match_count: number;
  trial_count: number;
}

/** `GET /api/bench/runs/{run_id}` response — `RunSummary` plus the fields
 * only the detail route carries. */
export interface RunDetail {
  run_id: string;
  kind: string;
  game: string | null;
  project_id: string | null;
  experiment_id: string | null;
  experiment_spec: ExperimentSpecV1 | null;
  label: string | null;
  config: unknown;
  git_sha: string;
  git_dirty: boolean;
  host: string;
  pid: number | null;
  started_at: string;
  ended_at: string | null;
  status: RunStatus;
  log_path: string;
  exit_code: number | null;
  match_count: number;
  trial_count: number;
  incumbent: IncumbentInfo | null;
}

/** A tuner run's current incumbent (its own intensifier's tracked best
 * config, not a naive lowest-cost trial -- see `LogRecord::Incumbent`'s doc
 * comment on the Rust side for why that distinction matters once a run uses
 * multiple baseline instances). `config` is already in the exact shape
 * `tune eval --baseline-config` expects. `null` on `RunDetail` for a
 * non-tuner run, or one that hasn't reported an incumbent yet. */
export interface IncumbentInfo {
  config: Record<string, unknown>;
  cost: number;
}

/** `GET /api/bench/runs/{run_id}/log?since=` response. `next_offset` is the
 * byte-offset cursor to pass as `since` on the next poll. */
export interface RunLogResponse {
  lines: string[];
  next_offset: number;
}

/** One traced game, newest first, from `GET /api/bench/runs/{run_id}/games`. */
export interface GameTraceSummary {
  game_seq: number;
  match_seq?: number | null;
  cell_id?: string | null;
  seed?: number | null;
  metrics?: unknown | null;
  ply_count: number;
  started_at: string;
  ended_at: string;
  strategy_a: string | null;
  strategy_b: string | null;
  outcome: string | null;
  winner: string | null;
}

/** One persisted trace position. `state` is wire JSON for round-robin and
 * display text for tuner traces. */
export interface GameMove {
  ply: number;
  ts: string;
  state: unknown;
  mv: unknown | null;
  player: string | null;
}

/** One SSE payload from a run's live trace stream. */
export interface LiveGameMove extends GameMove {
  game_seq: number;
}

/** `GET /api/bench/leaderboard` element. `win_rate` counts draws as
 * half a win; `ci_lower`/`ci_upper` are the Wilson interval. */
export interface LeaderboardEntry {
  strategy: string;
  total: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  ci_lower: number;
  ci_upper: number;
}

/** Map of git SHA to its leaderboard entries — the shape returned by a
 * commit-trends fetch that queries the leaderboard for every SHA that has
 * run data. */
export type CommitTrendData = Record<string, LeaderboardEntry[]>;

/** Sorted list of (sha, entries) pairs, pre-computed so the chart doesn't
 * re-derive it from the map on every render. */
export interface CommitTrendRow {
  sha: string;
  shortSha: string;
  entries: LeaderboardEntry[];
}

/** `POST /api/bench/launch` response. */
export interface LaunchResponse {
  run_id: string;
  pid: number;
  log_path: string;
  /** If the child process exited within 500ms of launch, the contents of
   * its stderr (redirected to stdout.log).  Absent or null means the
   * child was still alive after the check window — the launch succeeded
   * normally. */
  launch_error?: string | null;
}

/** `POST /api/bench/runs/{run_id}/stop` response — the Rust handler builds
 * this ad hoc (`json!`) with slightly different fields per path, so the
 * type is deliberately loose. */
export interface StopResponse {
  run_id: string;
  message?: string;
  status?: string;
  pid?: number | null;
  signal?: string | null;
}

/** Client-side filter shapes for the list/leaderboard queries. `null`
 * means "no filter" (the param is omitted from the query string). These
 * live in state (the reducer owns the current values); the API client maps
 * `gitSha` onto the wire's `git_sha`. */
export interface RunFilters {
  status: string | null;
  game: string | null;
  project_id?: string | null;
  experiment_id?: string | null;
}

export interface LeaderboardFilters {
  game: string | null;
  gitSha: string | null;
  since: string | null;
}

/** `GET /api/bench/kinds` element — metadata for one run kind. */
export interface BenchKindInfo {
  kind: string;
  label: string;
  description: string;
  games: BenchGameInfo[];
}

/** Per-game info within a run kind. */
export interface BenchGameInfo {
  game: string;
  strategies: StrategyInfo[];
}

/** A playable strategy for a game kind. */
export interface StrategyInfo {
  id: string;
  label: string;
  description: string;
}

/** One entry in a tuner's parameter space (`game_host::TunerParameter`).
 * `spec` is `#[serde(flatten)]`ed on the Rust side, so `type`/`bounds`/
 * `choices`/`value`/`default` land as top-level sibling fields of `name`
 * rather than nested — this type mirrors the wire shape directly. Only
 * `type` is guaranteed; which other fields are present depends on it
 * (`float`/`int` -> `bounds`+`default`, `categorical` -> `choices`+
 * `default`, `constant` -> `value`). */
export interface TunerParameter {
  name: string;
  type: "float" | "int" | "categorical" | "constant" | (string & {});
  bounds?: [number, number];
  choices?: string[];
  default?: unknown;
  value?: unknown;
}

/** A conditional activation rule (`game_host::TunerCondition`): when the
 * trial's config matches every `if` entry, every parameter named in `then`
 * is also active (and therefore present in a trial's `config`). `if`'s
 * values are either a single value or a list of values (any-of). */
export interface TunerCondition {
  if: Record<string, unknown>;
  then: string[];
}

/** A game's tunable search space, as reported by `tune describe`
 * (`game_host::TunerInfo`) and surfaced through `GET
 * /api/bench/tuner/kinds`. `baselines` is a list rather than a single id so
 * tuner can evaluate each trial against multiple opponent strengths
 * (`Scenario(instances=...)`) -- most games report one entry; a game with a
 * genuine second, harder preset (e.g. druid's "master") can list it as a
 * second instance. */
export interface TunerInfo {
  id: string;
  baselines: string[];
  eval_rounds: number;
  parameters: TunerParameter[];
  conditions: TunerCondition[];
  /** The game's own `default_config()` -- a game-setup axis (e.g. Druid's
   * board size) tuner never searches over, unlike `parameters`. `{}` means
   * the game's board is fixed at compile time and there's nothing to
   * configure here. */
  game_config: unknown;
}

/** `GET /api/bench/tuner/kinds` element — one tunable game. */
export interface TunerGameInfo {
  game: string;
  tuner: TunerInfo;
}

/** One row from the `trials` table, as reported by `GET
 * /api/bench/runs/{run_id}/trials`. `config` holds only the trial's
 * *active* parameters (inactive ones, per `TunerCondition`, are omitted). */
export interface TrialRow {
  trial_id: number;
  ts: string;
  config: Record<string, unknown>;
  seed: number | null;
  cost: number | null;
  extra: unknown | null;
}

/** One rung of a tuner ladder chain, as reported by `GET
 * /api/bench/runs/{run_id}/chain` -- oldest first. A run with no
 * `ladder_root` (a plain run, or a ladder run whose baseline was never
 * advanced) is a one-element chain containing just itself, so this always
 * has at least one entry for a run that exists. `incumbent` is the cost
 * this rung's baseline was promoted at (the *prior* rung's own incumbent)
 * -- `null` for the chain's root, which has no prior baseline advance
 * behind it. */
export interface ChainRung {
  run_id: string;
  label: string | null;
  status: RunStatus;
  started_at: string;
  ended_at: string | null;
  trial_count: number;
  incumbent: IncumbentInfo | null;
}
