// tuner-types.ts — TypeScript mirrors of the version-4 tuner server routes:
// the operational journal (`/api/bench/tuner/runs*`), the read-only
// projection (`/api/bench/tuner/projection/*`), and the launch-log tail.
// Field names and optionality track the Rust `Serialize` structs verbatim
// (`server/src/bench/tuner_runs.rs`, `server/src/bench/tuner_api.rs`); the
// integration test proves they stay in sync.

import type { JsonValue } from "../types.js";

// --- Operational journal -------------------------------------------------

/** `record.terminal_outcome` — how the detached process ended. `lost` means
 * the pid died with no reaper thread left alive to observe it (the server
 * that launched it restarted, or the machine rebooted) — the server's
 * periodic reaper assigns it after the fact once it notices the pid is gone. */
export type TerminalOutcome = "exited" | "signalled" | "spawn_failed" | "lost";

/** Derived liveness the server reports on each journal row. `failed` = the
 * process ended before it ever wrote a `manifest.json`, so the projection
 * will never describe it; `error_detail` carries its `launch.err`. */
export type TunerRunLiveness = "live" | "exited" | "failed" | "unknown";

/** One row of `GET /api/bench/tuner/runs`. */
export interface TunerRunView {
  run_id: string;
  argv: string[];
  run_dir: string;
  pid: number | null;
  started_at: string;
  terminal_outcome: TerminalOutcome | null;
  status: TunerRunLiveness;
  /** Tail of `launch.err`, present only when `status === "failed"`. */
  error_detail?: string | null;
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
  /** Number of entries in the objective's opponent panel. */
  opponent_count: number;
  /** File mtime, RFC3339; null if the file's metadata could not be read. */
  updated_at: string | null;
  /** The same stem is also shipped in the read-only seed corpus. */
  is_seed: boolean;
}

/** `GET /api/bench/tuner/objectives/{key}` — the objective JSON verbatim plus
 * its metadata. */
export interface TunerObjectiveDetail {
  key: string;
  content: JsonValue;
  updated_at: string | null;
  is_seed: boolean;
}

/** `POST /api/bench/tuner/objectives/{key}/validate` and the 400 body shape of
 * a rejected `PUT` — mirrors the Rust `ObjectiveValidation`. */
export interface ObjectiveValidationResult {
  ok: boolean;
  errors: string[];
  objective_id?: string;
  panel_fingerprint?: string;
}

/** One launch-profile file the server offers, keyed by its filename stem.
 * `GET /api/bench/tuner/profiles`. A launch profile is a saved bundle
 * `{game, objective, constraints, efforts, budgets}` a run is started from;
 * it is not an objective — it only references an `objective_key`. */
export interface TunerProfileFile {
  key: string;
  profile_id: string | null;
  game_kind: string | null;
  objective_key: string | null;
  /** Number of `constraints` entries the profile carries. */
  constraint_count: number;
  /** File mtime, RFC3339; null if the file's metadata could not be read. */
  updated_at: string | null;
  /** The same stem is also shipped in the read-only seed corpus. */
  is_seed: boolean;
}

/** `GET /api/bench/tuner/profiles/{key}` — the profile JSON verbatim plus
 * its metadata. */
export interface TunerProfileDetail {
  key: string;
  content: JsonValue;
  updated_at: string | null;
  is_seed: boolean;
}

/** Body of `POST /api/bench/tuner/runs`. The server resolves `game_kind` to
 * a built-in `game-<kind>` binary and `objective_key` to a file in its
 * configured objectives directory, so no filesystem path is part of the
 * request. Optional fields fall back to the tuner CLI's own defaults when
 * omitted. */
export type SpaceOverride =
  | { fix: string | number | boolean }
  | { range: [number, number] }
  | { choices: Array<string | number | boolean> };

/** One entry of the unified `constraints` wire form: a `set` of per-parameter
 * narrowings, optionally guarded by a `when` predicate over categorical
 * parameters. The bare `Record<string, SpaceOverride>` map is also accepted as
 * sugar for a single un-predicated entry. Only un-predicated entries are wired
 * end to end today. */
export interface Constraint {
  when?: Record<string, Array<string | number | boolean>>;
  set: Record<string, SpaceOverride>;
}

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
  /** Unified run-scoped tuning-space constraints — an array of {@link Constraint}
   * entries or the bare `{ name: SpaceOverride }` map as sugar. Serialised to a
   * single `--constraint <json>`. Constrains, never widens, the game's declared
   * schema; the server preflight is authoritative and the form only blocks
   * obvious local errors. */
  constraints?: Constraint[] | Record<string, SpaceOverride> | null;
  /** Per-phase search effort. Each phase accepts iterations *or* time, never
   * both; omitting a pair falls back to the tuner CLI default. Names match
   * the Rust `TunerLaunchRequest` serde fields verbatim. */
  tuning_max_iterations?: number | null;
  tuning_max_time_ms?: number | null;
  validation_max_iterations?: number | null;
  validation_max_time_ms?: number | null;
  production_max_iterations?: number | null;
  production_max_time_ms?: number | null;
}

/** Body of `POST /api/bench/tuner/runs/{id}/extend`. */
export interface TunerBudgetExtension {
  tuning_pair_attempts_delta?: number;
  validation_pair_attempts_delta?: number;
  diagnostic_pair_attempts_delta?: number;
  reason: string;
}

/** `POST /api/bench/tuner/runs/preflight` — dry-run of every launch check
 * `tuner_cli` applies before it starts a run. `ok: false` means the launch
 * would fail for the reasons in `errors`; the form blocks on it. */
export interface LaunchPreflightResult {
  ok: boolean;
  errors: string[];
}

/** One opponent in a resolved plan's panel — the schema-default source is
 * expanded to its actual `config`, so a hidden `rave` opponent is visible. */
export interface RunPlanOpponent {
  id: string;
  label?: string;
  role: string;
  weight: number;
  source: string;
  /** Canonical JSON of the resolved config. */
  config: string;
  fingerprint?: string;
}

/** One parameter in a resolved plan's tuning space, after `constraints` have
 * been applied. */
export interface RunPlanParameter {
  name: string;
  kind: string;
  bounds: [number, number] | null;
  choices: (string | number | boolean | null)[] | null;
  default: string | number | boolean | null;
  constant_value: string | number | boolean | null;
  /** Human-readable activation condition, e.g. `select in ['rave']`. */
  active_when: string | null;
}

export interface RunPlanSpace {
  schema_id: string;
  /** Residual domain of the root `algorithm` categorical after constraints. */
  algorithms: (string | number | boolean | null)[];
  /** Residual domain of every categorical / bool / constant axis, keyed by
   * axis name (`select`, `simulate`, …) — a narrowed entry means a variant
   * set was restricted for this run. */
  residual_categoricals: Record<string, (string | number | boolean | null)[]>;
  constraints: unknown[];
  parameters: RunPlanParameter[];
}

/** `POST /api/bench/tuner/runs/plan` — the fully resolved shape of the run a
 * launch request would start. `ok`/`errors` mirror the embedded preflight;
 * the rest is present only when resolution got far enough. */
export interface RunPlan {
  ok: boolean;
  errors: string[];
  game_kind?: string;
  objective_id?: string;
  objective_fingerprint?: string;
  game_config?: string;
  game_config_is_override?: boolean;
  opponents?: RunPlanOpponent[];
  panel_fingerprint?: string;
  space?: RunPlanSpace;
  efforts?: Record<"tuning" | "validation" | "production", { kind: string; value: number }>;
  budgets?: {
    cohort_size: number;
    finalists: number;
    bootstrap_candidates: number;
    random_reserve_candidates: number;
    tuning_pairs: number;
    tuning_pair_budget: number;
    validation_pair_budget: number;
    diagnostic_pair_budget: number;
    production_validation_pairs: number;
    proposer_policy: string;
    derived: {
      initial_cohort_pairs: number;
      validation_pairs_per_finalist: number;
      production_pairs: number;
    };
  };
  epoch?: { epoch_id: string; fingerprint: string };
}

// --- Live evidence journal --------------------------------------------

/** The evidence event types the tuner emits (mirrors `event_payloads.py`'s
 * `EventType`). The UI treats the payload as opaque and reads only the
 * shallow fields the ticker / progress fold need. */
export type EvidenceEventType =
  | "proposal_created"
  | "proposal_accepted"
  | "proposal_rejected"
  | "cohort_completed"
  | "pair_started"
  | "pair_completed"
  | "pair_failed"
  | "diagnostic_pair_started"
  | "diagnostic_pair_completed"
  | "diagnostic_pair_failed"
  | "run_interrupted"
  | "run_failed"
  | "observation_completed"
  | "finalists_selected"
  | "run_completed"
  | "allocation_decided"
  | "shadow_race_decided"
  | "candidate_failed"
  | "budget_extended";

/** One line of `evidence.jsonl`, decoded but passed through verbatim. */
export interface EvidenceEnvelope {
  sequence: number;
  type: EvidenceEventType;
  payload: unknown;
}

/** `GET /api/bench/tuner/runs/{id}/evidence?since_seq=N` — a forward tail. */
export interface EvidenceTailResponse {
  events: EvidenceEnvelope[];
  next_seq: number;
  run_status: TunerRunLiveness;
}

/** What `openEvidenceStream` pushes: batches of envelopes, then exactly one
 * terminal `ended` or `error`. */
export type EvidenceStreamMessage =
  | { kind: "events"; events: EvidenceEnvelope[] }
  /** The headless follower committed a projection pass covering this run's
   * newest evidence — re-fetch the science slices, no client refresh POST. */
  | { kind: "projectionUpdated" }
  | { kind: "ended" }
  | { kind: "error"; error: string };

/** `GET /api/bench/tuner/projection/meta` — projection-wide freshness. */
export interface ProjectionMeta {
  last_pass_at: string | null;
}

/** One rendered line in the `<EventTicker>`. */
export interface TickerLine {
  seq: number;
  text: string;
}

export type LivePhase = "proposal" | "tuning" | "validation" | "diagnostic" | "done";

/** Event tallies behind the live progress rail — counts and maxima only, no
 * statistics (those stay in the projection / `report.json`). */
export interface LiveProgress {
  phase: LivePhase;
  cohortIndex: number | null;
  pairs: { started: number; completed: number; failed: number };
  proposals: Record<string, { created: number; accepted: number; rejected: number }>;
  bestSoFar: { candidateId: string; pairUtility: number } | null;
  lastEventSeq: number;
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

/** One row of `GET .../projection/runs/{id}/pairs/{pair_id}/games` — a game
 * *summary* (no per-ply trace; the v4 tuner emits none). */
export interface ProjectionGameRow {
  game_id: string;
  pair_id: string;
  /** which seat the candidate held in this game. */
  candidate_side: string;
  outcome: string;
  plies: number;
  elapsed_ms: number;
  candidate_iterations_total: number;
  opponent_iterations_total: number;
}

/** `GET .../projection/runs/{id}/proposals` — the `proposals` table
 * verbatim. Populated on every projection refresh, so it is live during a
 * run; `disposition` is `null` while the proposal still awaits its cohort's
 * decision. */
export interface ProjectionProposal {
  proposal_index: number;
  cohort_index: number;
  cohort_slot: number;
  candidate_id: string;
  source: string;
  source_attempt: number;
  disposition: string | null;
  frontier_id: string;
  origin: string | null;
  acquisition: number | null;
  prediction: number | null;
  uncertainty: number | null;
  parent_candidate_id: string | null;
  refill_of_candidate_id: string | null;
}

/** `GET .../projection/runs/{id}/observations`. */
export interface ProjectionObservation {
  observation_id: string;
  candidate_id: string;
  phase: string;
  prefix_id: string;
  mean: number;
  lower: number;
  upper: number;
}

/** `GET .../projection/runs/{id}/shadow-decisions`. */
export interface ProjectionShadowDecision {
  race_index: number;
  cohort_index: number;
  prefix_id: string;
  candidate_id: string;
  boundary_candidate_id: string;
  disposition: string;
  policy_kind: string;
  policy_version: string;
}

/** `GET .../projection/runs/{id}/active-eliminations`. */
export interface ProjectionActiveElimination {
  batch_index: number;
  cohort_index: number;
  prefix_id: string;
  candidate_id: string;
  action: string;
  margin_kind: string;
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
