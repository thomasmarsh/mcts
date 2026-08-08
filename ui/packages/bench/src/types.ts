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

/** `POST /api/bench/launch` response. */
export interface LaunchResponse {
  run_id: string;
  pid: number;
  log_path: string;
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
