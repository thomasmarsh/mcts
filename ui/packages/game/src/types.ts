// types.ts — Wire types mirroring server/adapters/mod.rs's `GameAdapter`
// contract and server/main.rs's route response shapes -- typed against the
// real contract, not a sketch. Field names
// match the JSON wire format exactly (snake_case): none of the Rust structs
// this mirrors set `#[serde(rename_all = ...)]`, so there's no
// camelCase-translation layer here either. Kept separate from both
// api-client.ts and reducer.ts (rather than folded into either) so neither
// needs to import from the other -- see reducer.ts's `Env` / api-client.ts's
// `createEnv` for why that avoided circular import matters here.

/** `GET /api/games` element. */
export interface GameInfo {
  kind: string;
  label: string;
  description: string;
  config_schema: unknown;
}

/** `GET /api/games/{kind}/ai_presets` element. */
export interface AiPresetInfo {
  id: string;
  label: string;
  description: string;
}

/** One candidate root action from `analyze`, mirroring
 * `server::adapters::AnalysisAction` (action encoded as the game's own move
 * type `M`, not `Value` -- that erasure only exists on the Rust side of the
 * wire). */
export interface AnalysisAction<M> {
  action: M;
  visits: number;
  mean_value: number;
  is_proven: boolean;
}

/** Why a final search report is unavailable or partial. */
export type SearchReportReason = "strategy_unsupported" | "search_not_run" | "root_parallel_pv_single_tree";

/** The condition that stopped the most recent search. */
export type SearchTermination = "iterations" | "time" | "solved" | "unknown";

/** The retained search structure represented by a report. */
export type SearchGraphMode = "tree" | "transpositions" | "dag_edges" | "dag_nodes" | "dag_both";

/** A non-fatal qualification attached to a final report. */
export type SearchWarning =
  | "actions_truncated"
  | "principal_variation_truncated"
  | "structural_diagnostics_omitted"
  | "root_parallel_pv_single_tree";

/** One root action's final-search evidence. */
export interface SearchActionReport<M> {
  action: M;
  visits: number;
  share: number;
  mean_value: number;
  is_proven: boolean;
}

interface SearchReportBase<M> {
  schema_version: number;
  reason: SearchReportReason | null;
  elapsed_seconds: number | null;
  iteration_limit: number | null;
  time_limit_seconds: number | null;
  completed_iterations: number;
  termination: SearchTermination | null;
  selected_action: M | null;
  actions: SearchActionReport<M>[];
  principal_variation: M[];
  root_visits: number;
  tree_nodes: number;
  mean_depth: number | null;
  max_depth: number | null;
  graph_mode: SearchGraphMode | null;
  tt_reads: number;
  tt_writes: number;
  tt_hits: number;
  tt_hit_ratio: number | null;
  iterations_per_second: number | null;
  warnings: SearchWarning[];
}

/** Complete final-search evidence when the selected strategy retained it. */
export interface AvailableSearchReport<M> extends SearchReportBase<M> {
  status: "available";
}

/** Complete final-search evidence with an explicit qualification. */
export interface PartialSearchReport<M> extends SearchReportBase<M> {
  status: "partial";
}

/** A modern strategy that does not expose final-search evidence. */
export interface UnavailableSearchReport<M> extends SearchReportBase<M> {
  status: "unavailable";
}

/** Versioned final-search evidence, discriminated by its availability. */
export type SearchReport<M> = AvailableSearchReport<M> | PartialSearchReport<M> | UnavailableSearchReport<M>;

/** `POST /api/games/{kind}/analyze` response, mirroring `server::adapters::Analysis`. */
export interface Analysis<M> {
  actions: AnalysisAction<M>[];
  principal_variation: M[];
  total_visits: number;
  suggested_move: M | null;
  /** Omitted by legacy producers; `null` is also accepted for compatibility. */
  search?: SearchReport<M> | null;
}

/** `POST /api/games/{kind}/new` and `POST /api/games/{kind}/apply` share this
 * response shape -- see server/main.rs's `post_new`/`post_apply`. */
export interface StateAndView<S, V = unknown> {
  state: S;
  view: V;
}

/** `POST /api/games/{kind}/ai_move` response -- `StateAndView` plus the move
 * the AI chose. */
export interface AiMoveResult<S, M, V = unknown> {
  move: M;
  state: S;
  view: V;
  /** Omitted by legacy producers; `null` is also accepted for compatibility. */
  search?: SearchReport<M> | null;
}

/** `POST /api/games/{kind}/legal_moves` response. */
export interface LegalMovesResult<M> {
  moves: M[];
}

// -- config_ir::SearchSpec wire types (mcts-tune/src/config_ir.rs), for the
// "Custom" strategy path threaded through `ai_move`/`analyze` alongside a
// named `preset` string. Hand-mirrored, same convention as the rest of this
// file -- no Rust->TS codegen tooling exists in this repo. Recursion is
// bounded to exactly one level (`inner: BaseSelectSpec`/`BaseSimulateSpec`),
// matching `config_ir.rs`'s own bound (see that file's doc comment on why).

/** `mcts::select::RaveSchedule`'s wire form. */
export type RaveSchedule =
  | { kind: "hand_selected"; k: number }
  | { kind: "min_mse"; bias: number }
  | { kind: "threshold"; rave: number };

/** `mcts::select::RaveUcb`'s wire form. */
export type RaveUcb =
  | { kind: "none" }
  | { kind: "ucb1"; exploration_constant: number }
  | { kind: "ucb1_tuned"; exploration_constant: number };

/** `mcts::simulate::DecisiveMoveMode`'s wire form. */
export type DecisiveMoveMode =
  | "win"
  | "win_loss"
  | "win_loss_draw"
  | "anti_decisive";

/** `config_ir::BaseSelectSpec` -- the families an `EpsilonGreedy` may wrap. */
export type BaseSelectSpec =
  | { kind: "ucb1"; c: number }
  | { kind: "ucb1_tuned"; c: number }
  | { kind: "amaf"; alpha: number; c: number }
  | { kind: "rave"; threshold: number; schedule: RaveSchedule; ucb: RaveUcb }
  | { kind: "uct_pn"; c: number; c_pn: number }
  | { kind: "progressive_history"; c: number; ph_weight: number }
  | { kind: "bayes_uct1"; c: number }
  | { kind: "bayes_uct2"; c: number };

/** `config_ir::SelectSpec` -- `BaseSelectSpec`'s families plus the
 * `epsilon_greedy` wrapper. */
export type SelectSpec = BaseSelectSpec | { kind: "epsilon_greedy"; epsilon: number; inner: BaseSelectSpec };

/** `config_ir::BaseSimulateSpec` -- the families `epsilon_greedy`/
 * `decisive_move` may wrap. */
export type BaseSimulateSpec =
  | { kind: "uniform" }
  | { kind: "mast" }
  | { kind: "nst"; backoff_threshold: number }
  | { kind: "lgr" }
  | { kind: "lgr2" }
  | { kind: "lgr2_mast" };

/** `config_ir::SimulateSpec` -- `BaseSimulateSpec`'s families plus its
 * wrappers and fixed-shape two-level leaves (`decisive_move_mast`/
 * `decisive_move_nst`/`meta_mcts`, none of which carry a `wraps` schema key
 * -- see `config_ir.rs`'s doc comment on why they're flat, not recursive). */
export type SimulateSpec =
  | BaseSimulateSpec
  | { kind: "epsilon_greedy"; epsilon: number; inner: BaseSimulateSpec }
  | { kind: "decisive_move"; mode: DecisiveMoveMode; inner: BaseSimulateSpec }
  | { kind: "decisive_move_mast"; mode: DecisiveMoveMode; epsilon: number }
  | { kind: "decisive_move_nst"; mode: DecisiveMoveMode; epsilon: number; nst_backoff_threshold: number }
  | { kind: "meta_mcts"; iterations: number };

/** `config_ir::BackpropSpec`. `bayes_gaussian`/`bayes_numeric` are the two
 * Bayesian posterior-propagation modes `select`'s `bayes_uct1`/`bayes_uct2`
 * require (see `config_ir.rs`'s `needs_posterior` doc comment) -- pairing
 * either `select` variant with `classic` here is rejected server-side by
 * `config_ir::validate_search_spec`. */
export type BackpropSpec =
  | { kind: "classic" }
  | { kind: "bayes_gaussian"; prior_variance: number; obs_variance: number }
  | { kind: "bayes_numeric"; prior_variance: number; obs_variance: number; value_lo: number; value_hi: number };

/** `config_ir::FinalActionSpec`. */
export type FinalActionSpec =
  | { kind: "robust_child" }
  | { kind: "max_avg" }
  | { kind: "max_robust_child" }
  | { kind: "secure_child"; a: number };

/** `config_ir::SearchSpec` -- the full four-axis free composition. */
export interface SearchSpec {
  select: SelectSpec;
  simulate: SimulateSpec;
  backprop: BackpropSpec;
  final_action: FinalActionSpec;
}

/** `mcts_tune::presets::CustomStrategySpec`. */
export interface CustomStrategySpec {
  search: SearchSpec;
  max_time_ms?: number;
  max_iterations?: number;
  threads?: number;
  use_transpositions?: boolean;
  q_init?: string;
  /** Monte Carlo *graph* search (transposition-merged DAG) instead of plain
   * tree search. Requires `use_transpositions: true` -- rejected server-side
   * by `mcts_tune::presets::build_custom` otherwise, since graph search only
   * makes sense against a game with a real zobrist hash. */
  mcgs?: boolean;
}

/** Which strategy an `ai_move`/`analyze` call should use for a seat -- a
 * named preset id (the existing wire form) or an inline `CustomStrategySpec`
 * (this phase's addition). Not itself a wire type -- `api-client.ts`
 * flattens this into `{preset, custom?}` at the HTTP boundary, matching
 * `server::main::AiMoveRequest`/`AnalyzeRequest`'s actual shape (`preset`
 * stays a required sentinel string, `"custom"`, alongside `custom` --  see
 * that struct's doc comment). */
export type AiStrategyRef = { kind: "preset"; id: string } | { kind: "custom"; spec: CustomStrategySpec };

/** A field's leaf shape in `AxisSchema` -- mirrors
 * `mcts_tune::config_ir_schema::axis_schema()`'s per-field JSON. `bare`
 * (only ever `true`, only ever on an `"enum"` field) marks a field whose
 * Rust type has no `#[serde(tag = "kind")]` -- `DecisiveMoveMode` is the
 * one example (see `mcts::simulate::DecisiveMoveMode`'s own derive): its
 * wire form is a plain string (`"win"`), not `{kind, ...fields}` the way
 * `RaveSchedule`/`RaveUcb` (real tagged unions, ordinary `"enum"` fields
 * with no `bare`) are. A field with `bare: true` always has every variant's
 * `fields` empty (there is nothing to carry -- see `bare_en`'s doc comment
 * on the Rust side), so rendering/storing it as a plain string loses
 * nothing. Getting this distinction wrong is exactly what produced
 * `CustomStrategySpec` deserialization's "unknown variant `kind`, expected
 * one of `win`, `win_loss`, `win_loss_draw`" error in production: `mode`
 * was being sent as `{"kind":"win"}`, and `serde` read the object's own key
 * as if it were the variant name. */
export type AxisFieldSchema =
  | { name: string; type: "float"; bounds: [number, number]; default: number }
  | { name: string; type: "int"; bounds: [number, number]; default: number }
  | { name: string; type: "bool"; default: boolean }
  | { name: string; type: "enum"; default: string; variants: AxisVariantSchema[]; bare?: boolean };

/** One variant's schema entry -- `wraps` is present only on variants that
 * recurse into another axis's base variant set (`epsilon_greedy`/
 * `decisive_move`), naming which key of `AxisSchema` to look up for the
 * inner picker. */
export interface AxisVariantSchema {
  kind: string;
  fields: AxisFieldSchema[];
  wraps?: "select_base" | "simulate_base";
}

/** `GET /api/strategy-schema` response -- `mcts_tune::config_ir_schema::
 * axis_schema()`'s full JSON shape. */
export interface AxisSchema {
  select: { variants: AxisVariantSchema[] };
  select_base: { variants: AxisVariantSchema[] };
  simulate: { variants: AxisVariantSchema[] };
  simulate_base: { variants: AxisVariantSchema[] };
  backprop: { variants: AxisVariantSchema[] };
  final_action: { variants: AxisVariantSchema[] };
}

/** `game_host::TunerParameter`'s `#[serde(flatten)]` wire shape -- one entry
 * of `TunerInfo.parameters`. */
export type TunerParameter =
  | { name: string; type: "categorical"; choices: string[]; default: string }
  | { name: string; type: "int"; bounds: [number, number]; default: number }
  | { name: string; type: "float"; bounds: [number, number]; default: number }
  | { name: string; type: "bool"; default: boolean };

/** `game_host::TunerCondition` -- `if`'s single entry maps a parent
 * parameter name to the value(s) that activate every name in `then`.
 * `if_` on the Rust side, renamed `if` over the wire. */
export interface TunerCondition {
  if: Record<string, string | boolean | (string | boolean)[]>;
  then: string[];
}

/** `GET /api/games/{kind}/strategy-families` response -- `game_host::
 * TunerInfo`. `null` for a game with no tuning support. */
export interface TunerInfo {
  id: string;
  baselines: string[];
  eval_rounds: number;
  parameters: TunerParameter[];
  conditions: TunerCondition[];
  game_config: unknown;
}
