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
export type RunStatus = "running" | "completed" | "crashed" | "stopped" | (string & {});

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
  game: string;
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
  game: string;
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

/** A SMAC3 run's current incumbent (its own intensifier's tracked best
 * config, not a naive lowest-cost trial -- see `LogRecord::Incumbent`'s doc
 * comment on the Rust side for why that distinction matters once a run uses
 * multiple baseline instances). `config` is already in the exact shape
 * `tune eval --baseline-config` expects. `null` on `RunDetail` for a
 * non-SMAC3 run, or one that hasn't reported an incumbent yet. */
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
 * /api/bench/smac3/kinds`. `baselines` is a list rather than a single id so
 * SMAC3 can evaluate each trial against multiple opponent strengths
 * (`Scenario(instances=...)`) -- most games report one entry; a game with a
 * genuine second, harder preset (e.g. druid's "master") can list it as a
 * second instance. */
export interface TunerInfo {
  id: string;
  baselines: string[];
  eval_rounds: number;
  parameters: TunerParameter[];
  conditions: TunerCondition[];
}

/** `GET /api/bench/smac3/kinds` element — one tunable game. */
export interface Smac3GameInfo {
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
