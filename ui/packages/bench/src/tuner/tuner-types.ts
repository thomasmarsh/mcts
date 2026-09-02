// tuner-types.ts — TypeScript mirrors of the version-4 tuner server routes:
// the operational journal (`/api/bench/tuner/runs*`), the read-only
// projection (`/api/bench/tuner/projection/*`), and the launch-log tail.
// Field names and optionality track the Rust `Serialize` structs verbatim
// (`server/src/bench/tuner_runs.rs`, `server/src/bench/tuner_api.rs`); the
// integration test proves they stay in sync.

import type { JsonValue } from "../types.js";

// --- Operational journal -------------------------------------------------

/** `record.terminal_outcome` — how the detached process ended. */
export type TerminalOutcome = "exited" | "signalled" | "spawn_failed";

/** Derived liveness the server reports on each journal row. */
export type TunerRunLiveness = "live" | "exited" | "unknown";

/** One row of `GET /api/bench/tuner/runs`. */
export interface TunerRunView {
  run_id: string;
  argv: string[];
  run_dir: string;
  pid: number | null;
  started_at: string;
  terminal_outcome: TerminalOutcome | null;
  status: TunerRunLiveness;
}

/** `GET /api/bench/tuner/runs/{id}/log?since=N` — tail of `launch.out` plus
 * the full `launch.err` (re-sent each poll; normally empty). */
export interface TunerRunLog {
  lines: string[];
  next_offset: number;
  err_lines: string[];
}

/** One frozen-objective file the server offers, keyed by its filename stem.
 * `GET /api/bench/tuner/objectives`. The absolute path never crosses the
 * API — a launch request carries the `key`. */
export interface TunerObjectiveFile {
  key: string;
  objective_id: string | null;
  game_kind: string | null;
}

/** Body of `POST /api/bench/tuner/runs`. The server resolves `game_kind` to
 * a built-in `game-<kind>` binary and `objective_key` to a file in its
 * configured objectives directory, so no filesystem path is part of the
 * request. Optional fields fall back to the tuner CLI's own defaults when
 * omitted. */
export interface TunerLaunchRequest {
  game_kind: string;
  objective_key: string;
  run_id: string;
  task_seed: number;
  tuning_pair_budget: number;
  validation_pair_budget: number;
  production_validation_pairs: number;
  seed?: number | null;
  cohort_size?: number | null;
  finalists?: number | null;
  bootstrap_candidates?: number | null;
  random_reserve_candidates?: number | null;
  diagnostic_pair_budget?: number | null;
  evaluator_workers?: number | null;
  proposer_policy?: string | null;
  exclude_family?: string[];
}

/** Body of `POST /api/bench/tuner/runs/{id}/extend`. */
export interface TunerBudgetExtension {
  tuning_pair_attempts_delta?: number;
  validation_pair_attempts_delta?: number;
  diagnostic_pair_attempts_delta?: number;
  reason: string;
}

// --- Projection (science) ----------------------------------------------

/** One row of `GET /api/bench/tuner/projection/runs`. */
export interface ProjectionRunListItem {
  run_id: string;
  terminal_status: string | null;
  report_available: boolean;
  ingest_error: string | null;
  game_kind: string | null;
  objective_id: string | null;
  shadow_policy_kind: string | null;
  active_elimination: boolean | null;
  report_status: string | null;
  validation_claim: string | null;
  total_pair_attempts: number;
  total_completed_pairs: number;
}

export interface ProjectionManifestSummary {
  manifest_run_id: string | null;
  manifest_fingerprint: string | null;
  game_kind: string;
  objective_id: string;
  cohort_size: number;
  finalists: number;
  seed: number;
  task_seed: number;
  shadow_policy_kind: string;
  active_elimination: boolean;
}

export interface ProjectionReportSummary {
  schema_version: number;
  status: string;
  validation_claim: string;
}

export interface ProjectionComputePhase {
  phase: string;
  pair_attempts: number;
  completed_pairs: number;
  failed_attempts: number;
  censored_attempts: number;
  physical_games: number;
  search_iterations: number;
  wall_time_ms: number;
}

/** `GET /api/bench/tuner/projection/runs/{id}`. */
export interface ProjectionRunDetail {
  run_id: string;
  terminal_status: string | null;
  report_available: boolean;
  ingest_error: string | null;
  manifest: ProjectionManifestSummary | null;
  report: ProjectionReportSummary | null;
  compute: ProjectionComputePhase[];
}

export interface ProjectionCohort {
  cohort_index: number;
  candidate_ids: string[];
  retained_candidate_ids: string[];
}

export interface ProjectionCandidate {
  candidate_id: string;
  fingerprint: string;
  canonical_config: JsonValue;
  cohort_index: number;
  cohort_slot: number;
  source: string;
  parent_candidate_id: string | null;
}

export interface ProjectionPairRow {
  pair_id: string;
  phase: string;
  candidate_id: string;
  task_id: string;
  opponent_id: string;
  pair_utility: number;
}

export interface ProjectionValidationRow {
  candidate_id: string;
  rank: number;
  estimate: number;
  lower: number;
  upper: number;
  wins: number;
  draws: number;
  losses: number;
}

export interface ProjectionValidation {
  rows: ProjectionValidationRow[];
  unresolved_ties: JsonValue;
}

export interface ProjectionRefreshResult {
  projected: number;
  skipped: number;
  ingest_errors: number;
  pruned: number;
}

export interface ProjectionPairQuery {
  candidate?: string;
  cohort?: number;
  limit?: number;
  offset?: number;
}
