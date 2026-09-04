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
export type RunStatus =
  "running" | "completed" | "completed_with_errors" | "crashed" | "stopped" | (string & {});

export interface BenchSpectatorProps {
  runId: string;
  game: string;
  kind: string;
  live: boolean;
  cellId?: string;
  initialGameSeq?: number;
}

/** JSON carried by persisted tuning manifests and configuration snapshots.
 * Keeping this boundary explicit prevents presentation code from treating
 * lifecycle payloads as arbitrary records. */
export type JsonValue =
  null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

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
  /** Present only on legacy experiment runs (the web UI no longer launches
   * these); kept as opaque JSON since the experiment spec type is gone. */
  experiment_spec: unknown | null;
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
  /** Final search evidence retained for the move that reached this state. */
  search?: import("@mcts/game").SearchReport<unknown> | null;
}

/** One SSE payload from a run's live trace stream. */
export interface LiveGameMove extends GameMove {
  game_seq: number;
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

/** Client-side filter shape for the run-list query. `null` means "no
 * filter" (the param is omitted from the query string). Lives in state —
 * the reducer owns the current values. */
export interface RunFilters {
  status: string | null;
  game: string | null;
  project_id?: string | null;
  experiment_id?: string | null;
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

/** Bounds and types for a game's `game_config` setup axis
 * (`game_host::GameConfigSchema`), reusing the `TunerParameter` /
 * `TunerCondition` shapes verbatim. An empty `parameters` list means the
 * board is fixed at compile time — nothing to configure. Druid's `{w, h}`
 * size is two dotted `size.w` / `size.h` parameters. */
export interface GameConfigSchema {
  parameters: TunerParameter[];
  conditions: TunerCondition[];
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
  /** Bounds and types for the `game_config` axis, so a launch form or the
   * objective editor can accept and validate a non-default value (e.g.
   * AtariGo on 9×9). Absent / empty `parameters` for a fixed-board game. */
  game_config_schema?: GameConfigSchema;
}

/** `GET /api/bench/tuner/kinds` element — one tunable game. */
export interface TunableGame {
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
