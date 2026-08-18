//! Generic multi-family MCTS strategy tuning harness shared by every game
//! crate that opts into SMAC3-style hyperparameter search. Everything here
//! is generic over `G: Game` -- picking a concrete game, a baseline preset,
//! and whether that game has a real `zobrist_hash` (see `use_transpositions`
//! below) is the only per-game glue. See `games/traffic-lights/src/main.rs`
//! for the reference wiring.
//!
//! The tunable search space has two levels: a top-level categorical
//! `family` axis choosing *which* `Strategy<G>` (select/simulate/backprop/
//! final-action composition) to run -- the same catalog of named types
//! declared in `mcts::strategies::mcts::strategy` -- and, within the chosen
//! family, that family's own hyperparameters (RAVE's schedule/`c`/epsilon,
//! the UCB families' exploration constant, etc). `q_init` is a `SearchConfig`
//! setting orthogonal to the strategy and applies to every family; the
//! `final_action` axis (`max_avg`/`secure_child`/`robust_child`) applies to
//! every family whose own named type doesn't already fix a different final
//! action (`ucb1_max_robust` and `meta_mcts` each fix their own, matching
//! their `strategy.rs` definitions).
//!
//! `strategy.rs`'s `QuasiBestFirst` is deliberately **not** in the catalog:
//! its `best_child` falls back to a full nested `TreeSearch::choose_action`
//! call whenever its opening book has no entry for the current line, which
//! is unconditionally true here (this harness never populates a book) --
//! and that fallback fires on every tree-descent step of the *outer* search,
//! not once per leaf like `meta_mcts`'s nested search. Its own doc comment
//! says as much: it's designed to pair with an outer `max_iterations: 1`,
//! not the shared many-iteration budget every other family in this harness
//! runs under. Confirmed by hand: wiring it in the same way as the other
//! families hung indefinitely on a two-move `Nim` game.
//!
//! This crate deliberately depends on nothing but `mcts` (core search) and
//! `game-host` (for `TunerInfo`/`HostError`) -- no game crate, and no
//! `mcts-bench` (whose `rayon`/`indicatif`/`duckdb` dependencies the lean
//! per-game subprocess binaries have no reason to pull in). `bench`/
//! `mcts-bench` never depend on this crate either: they only ever talk to a
//! game as an opaque subprocess, and building a candidate/baseline
//! `Box<dyn Search<G>>` requires the concrete `G` in scope at compile time,
//! so that code has to live inside each game's own crate/binary.

use std::str::FromStr;

pub mod config_ir;
pub mod config_ir_schema;
mod family_catalog;
pub mod presets;
pub mod trace;

use family_catalog::{
    condition, dispatch_family, family_choices, family_conditions, param, tunable_field_parameters,
    FamilySpec, TrialParams,
};
use game_host::{
    ConfiguredCandidateSide, ConfiguredMatchResult, ConfiguredOutcome, ConfiguredStrategyMetrics,
    HostError, TunerInfo,
};
use mcts::game::{Game, PlayerIndex};
use mcts::strategies::mcts::{node::QInit, GraphSearch, GraphStats};
use mcts::strategies::Search;
use serde_json::{json, Value};

#[cfg(test)]
use mcts::strategies::mcts::select::{RaveSchedule, RaveUcb};
#[cfg(test)]
use mcts::strategies::mcts::{simulate, strategy};
#[cfg(test)]
use mcts::strategies::mcts::{SearchConfig, TreeSearch};

const PLAYOUT_DEPTH: usize = 200;
const MAX_ITER: usize = 10_000;
const EXPAND_THRESHOLD: u32 = 1;

/// Iteration cap for `meta_mcts`'s inner nested search -- see the comment at
/// its `make_candidate` arm for why this can't just be `TreeSearch::default()`.
/// Deliberately small (not `MAX_ITER`-sized): the outer search's own
/// `MAX_ITER` simulate steps each run a full inner search of this size, so
/// `meta_mcts`'s total per-move cost is already `MAX_ITER *
/// META_MCTS_INNER_ITERATIONS` -- a few dozen iterations is enough for the
/// inner search to be more informed than a uniform rollout without making
/// every `meta_mcts` trial two orders of magnitude more expensive than every
/// other family's. Still real work, though -- see `tests/stress.rs` for why
/// its round-trip test doesn't live in this file's fast suite.
const META_MCTS_INNER_ITERATIONS: usize = 50;

/// A candidate's search-effort ceiling -- orthogonal to `TrialParams`
/// (which family/hyperparameters to run), this is *how much compute* that
/// family gets to run for. Defaults to this harness's historical behavior
/// (`MAX_ITER` iterations, single-threaded, uncapped wall time) -- the
/// right shape for a `baseline_config`-backed opponent (self-play against a
/// discovered config, including the `random`/`flat_mc` floor families
/// below), since both sides of that match are built the same way and so
/// stay symmetric regardless of budget.
///
/// A **named-preset** baseline (e.g. Druid's `strong`/`master`, built by
/// `build_ai` on a wall-clock time budget and every available CPU core, not
/// `MAX_ITER`) is a different story: leaving the candidate at the default
/// here pits a single-threaded, tree-discarding-per-move, fixed-iteration
/// search against a multi-core, tree-persisting, time-budgeted one -- a
/// mismatch severe enough to produce a near-100%-loss streak on its own,
/// independent of which family/hyperparameters SMAC3 samples. A game's own
/// `tune_eval` is responsible for building a `SearchBudget` that mirrors
/// whatever named preset it's dispatching to in that case (see
/// `games/druid/src/main.rs`'s `tune_eval`).
///
/// `max_iterations` is deliberately **not** part of `TrialParams` -- it's a
/// per-*run* compute budget an operator sets once at launch (`--override
/// target.max_iterations=N`, or the launch form's "Iteration budget"
/// field), not a per-*trial* hyperparameter SMAC3 gets to search over
/// (searching it would just reward configs that use the biggest budget
/// available, not the best hyperparameters at a fixed budget). `None` here
/// means "use this crate's historical constant" (`MAX_ITER`) -- see
/// `base_config`. A game's `tune_eval` reads this from its own
/// `max_iterations: Option<usize>` CLI-forwarded argument and threads the
/// *same* value into both the candidate's budget and, for a
/// `baseline_config`-backed opponent, `build_search`'s budget too --
/// leaving one side on the old `MAX_ITER` default while the other honors an
/// operator's override would silently reintroduce the exact asymmetric-
/// budget mismatch this type exists to prevent.
#[derive(Debug, Clone, Copy)]
pub struct SearchBudget {
    pub max_time: Option<std::time::Duration>,
    pub threads: usize,
    pub max_iterations: Option<usize>,
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self {
            max_time: None,
            threads: 1,
            max_iterations: None,
        }
    }
}

impl SearchBudget {
    fn iteration_limit(self) -> usize {
        self.max_iterations
            .or_else(|| self.max_time.map(|_| usize::MAX))
            .unwrap_or(MAX_ITER)
    }
}

/// Converts one trial's `TrialParams` into `config_ir`'s `SearchSpec`/
/// `SearchSettings` -- the `config_ir`-based counterpart of `make_candidate`'s
/// `match p.family.as_str()` dispatch, covering every family except
/// `"random"`/`"flat_mc"` (not a `Compose<..>` `Strategy`, so they stay direct
/// arms in `make_candidate` permanently; see its own comment on why). Per-
/// family construction is `family_catalog::dispatch_family`'s
/// `register_family!` table; this function only handles what's common to
/// every family (`q_init`, `mcgs`, the fixed `SearchSettings` knobs).
/// `make_candidate` calls this and then `config_ir::build_search` for every
/// other family.
fn to_search_spec(
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<(config_ir::SearchSpec, config_ir::SearchSettings), HostError> {
    let q_init = QInit::from_str(&p.q_init)
        .map_err(|_| HostError::bad_request(format!("invalid q_init: {}", p.q_init)))?;
    let mcgs = p.mcgs.unwrap_or(false);
    if mcgs && !use_transpositions {
        return Err(HostError::bad_request(
            "mcgs requires a game with a zobrist hash",
        ));
    }

    let FamilySpec {
        select,
        simulate,
        final_action,
        solver_loss_threshold: solver_loss_threshold_setting,
        contempt_factor: contempt_factor_setting,
    } = dispatch_family(&p.family, p)?;

    let spec = config_ir::SearchSpec {
        select,
        simulate,
        backprop: config_ir::BackpropSpec::Classic {},
        final_action,
    };
    let settings = config_ir::SearchSettings {
        max_iterations: budget.iteration_limit(),
        max_playout_depth: PLAYOUT_DEPTH,
        expand_threshold: EXPAND_THRESHOLD,
        q_init,
        use_transpositions: use_transpositions && !mcgs,
        use_mcts_solver: true,
        reuse_tree: !mcgs,
        num_tree_threads: budget.threads,
        seed,
        max_time: budget.max_time,
        graph_search: mcgs.then_some(GraphSearch::Dag(GraphStats::Both)),
        solver_loss_threshold: solver_loss_threshold_setting,
        contempt_factor: contempt_factor_setting,
    };
    Ok((spec, settings))
}

/// Builds a `Box<dyn Search<G>>` from a raw params JSON object, the same
/// deserialize-then-dispatch path `strategy_tune_eval` uses for the
/// candidate side -- exposed so a caller can also build an *opponent* from
/// an arbitrary discovered config, not just a named preset. See
/// `game_host::GameAdapter::tune_eval`'s `baseline_config` parameter.
///
/// Every caller of `build_search` builds an *opponent* -- a
/// `baseline_config`-backed baseline, or a `--baseline-config` for the
/// ladder driver's own self-play rungs -- never the candidate under tune.
/// That side of the match is already symmetric with the candidate (both go
/// through this exact function), so `budget` should always be the *same*
/// `SearchBudget` the caller is about to pass as `strategy_tune_eval`'s
/// `candidate_budget` -- passing `SearchBudget::default()` here while the
/// candidate runs under an operator's `max_iterations` override would break
/// that symmetry (an opponent quietly capped at the old `MAX_ITER` while
/// the candidate is held to a smaller budget, or vice versa).
pub fn build_search<G: Game + 'static>(
    params: &Value,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<Box<dyn Search<G = G>>, HostError> {
    let trial: TrialParams = serde_json::from_value(params.clone())
        .map_err(|e| HostError::bad_request(format!("invalid tuning params: {e}")))?;
    make_candidate(&trial, seed, use_transpositions, budget)
}

fn make_candidate<G: Game + 'static>(
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<Box<dyn Search<G = G>>, HostError> {
    match p.family.as_str() {
        // Baseline-only floor families -- deliberately *not* in
        // `strategy_tuner_info`'s searchable `family` choices (a candidate
        // sampled as `random`/`flat_mc` would just hover around a ~0.5 cost
        // forever, wasting SMAC3's trial budget). Reachable only via
        // `build_search`/`--baseline-config`, e.g. as a ladder's floor rung.
        // Neither reads `q_init` or any other `TrialParams` field beyond
        // `family` itself. Not a `Compose<..>` `Strategy`, so these two stay
        // outside `to_search_spec`/`config_ir::build_search` permanently.
        "random" => Ok(Box::new(mcts::strategies::random::Random::<G>::new())),
        "flat_mc" => Ok(Box::new(
            mcts::strategies::flat_mc::FlatMonteCarloStrategy::<G>::new(),
        )),
        _ => {
            let (spec, settings) = to_search_spec(p, seed, use_transpositions, budget)?;
            Ok(config_ir::build_search(&spec, &settings))
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Result of one `strategy_tune_eval` call: `cost` is what SMAC3 minimizes
/// (the candidate's loss rate against the baseline); `wins`/`losses`/`draws`
/// are from the candidate's perspective, for display.
#[derive(Debug)]
pub struct TuneEvalOutcome {
    pub cost: f64,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

/// Play `rounds` rounds (each round: one game candidate-first, one game
/// baseline-first) of `params`-built candidate vs `baseline_build()`, and
/// return the aggregate outcome. `baseline_build` is a thunk rather than an
/// already-built search so a fresh, un-treed instance is used for every
/// single game, matching how each preset's `build_*` function is used
/// elsewhere in a game's own crate.
///
/// `use_transpositions` should only be `true` for a game with a real
/// `Game::zobrist_hash` override -- the default hash is a constant `0`, and
/// enabling transpositions against it merges every node in the tree into
/// one, silently corrupting the search rather than erroring. Pass `true`
/// only for games that have actually implemented `zobrist_hash`.
///
/// `initial_state` is the board every game in this call starts from --
/// almost every caller passes `G::S::default()` (`Game::S` requires
/// `Default`), matching this function's behavior before `initial_state`
/// existed. A game with a real runtime-configurable setup (e.g. Druid's
/// board size) builds one from its own `tune_eval`'s `game_config` argument
/// instead; every other game ignores that argument entirely, since its
/// board is fixed at compile time regardless.
/// `candidate_budget` is the compute the *candidate* side gets -- see
/// `SearchBudget`'s doc comment. Pass `SearchBudget::default()` when
/// `baseline_build` builds an opponent already on the same iteration-based
/// footing as the candidate (a `baseline_config`-backed self-play opponent,
/// built via `build_search`, or any of this crate's own floor families);
/// pass a budget mirroring the opponent's own compute when `baseline_build`
/// wraps a named, wall-clock/thread-budgeted preset instead (see
/// `games/druid/src/main.rs`'s `tune_eval`).
#[allow(clippy::too_many_arguments)]
pub fn strategy_tune_eval<G: Game + 'static>(
    params: &Value,
    rounds: u32,
    seed: Option<u64>,
    use_transpositions: bool,
    candidate_budget: SearchBudget,
    baseline_build: impl Fn() -> Box<dyn Search<G = G>>,
    initial_state: G::S,
    trace_path: Option<&std::path::Path>,
    on_game: &mut dyn FnMut(ConfiguredMatchResult) -> Result<(), HostError>,
) -> Result<TuneEvalOutcome, HostError> {
    let trial: TrialParams = serde_json::from_value(params.clone())
        .map_err(|e| HostError::bad_request(format!("invalid tuning params: {e}")))?;
    let seed = seed.unwrap_or(0);

    if rounds == 0 {
        let _ = make_candidate::<G>(&trial, seed, use_transpositions, &candidate_budget)?;
        return Ok(TuneEvalOutcome {
            cost: 0.0,
            wins: 0,
            losses: 0,
            draws: 0,
        });
    }

    let mut tracer = trace_path
        .map(trace::MoveTracer::open)
        .transpose()
        .map_err(|e| HostError::bad_request(format!("failed to open --trace-path: {e}")))?;

    let (mut wins, mut losses, mut draws) = (0u32, 0u32, 0u32);
    let mut seq = 0u64;
    for round in 1..=rounds {
        let mut candidate = make_candidate(&trial, seed, use_transpositions, &candidate_budget)?;
        let mut baseline = baseline_build();

        seq += 1;
        let result = play_one(
            candidate.as_mut(),
            baseline.as_mut(),
            initial_state.clone(),
            tracer.as_mut(),
            round,
            seq,
            seed,
            ConfiguredCandidateSide::First,
        )?;
        match result.outcome {
            ConfiguredOutcome::CandidateWin => wins += 1,
            ConfiguredOutcome::BaselineWin => losses += 1,
            ConfiguredOutcome::Draw => draws += 1,
        }
        on_game(result)?;

        // Swap move order so the candidate plays second half the time.
        let mut candidate = make_candidate(&trial, seed, use_transpositions, &candidate_budget)?;
        let mut baseline = baseline_build();
        seq += 1;
        let result = play_one(
            baseline.as_mut(),
            candidate.as_mut(),
            initial_state.clone(),
            tracer.as_mut(),
            round,
            seq,
            seed,
            ConfiguredCandidateSide::Second,
        )?;
        match result.outcome {
            ConfiguredOutcome::CandidateWin => wins += 1,
            ConfiguredOutcome::BaselineWin => losses += 1,
            ConfiguredOutcome::Draw => draws += 1,
        }
        on_game(result)?;
    }

    Ok(TuneEvalOutcome {
        cost: cost_from_losses(losses, rounds),
        wins,
        losses,
        draws,
    })
}

/// Implements the common shape of `GameAdapter::tune_eval` directly, for a
/// game whose board is fixed at compile time -- it ignores `game_config` and
/// always starts from `G::S::default()` -- and whose only named baseline is
/// a single preset (`baseline_preset`, e.g. `"strong"`). That covers every
/// current game except the handful with a runtime-configurable board (e.g.
/// `games/druid`, whose `tune_eval` builds `initial_state` from its own
/// `game_config` argument instead) or a `tuner()` with more than one
/// baseline; those keep writing out `strategy_tune_eval`/`build_search`
/// directly, matching what this function does internally.
///
/// `presets_source` names the preset file in the panic message if
/// `baseline_preset` fails to build (e.g. `"games/ttt/presets.json"`) --
/// this only fires if that file's own baseline preset is broken, which
/// should never happen since it ships with the crate.
#[allow(clippy::too_many_arguments)]
pub fn generic_tune_eval<G: Game + 'static>(
    presets: &presets::PresetTable,
    baseline_preset: &str,
    presets_source: &str,
    use_transpositions: bool,
    preset_seed: u64,
    params: Value,
    rounds: u32,
    seed: Option<u64>,
    baseline_config: Option<Value>,
    max_iterations: Option<usize>,
    max_time_ms: Option<u64>,
    trace_path: Option<std::path::PathBuf>,
    on_game: &mut dyn FnMut(ConfiguredMatchResult) -> Result<(), HostError>,
) -> Result<Value, HostError> {
    // `use_transpositions: true` requires a real `Game::zobrist_hash`
    // override -- the caller is responsible for only passing `true` when `G`
    // has one, so merging transposed nodes during the candidate's search is
    // safe here (see `strategy_tune_eval`'s doc comment).
    let outcome = if let Some(cfg) = baseline_config {
        let baseline_seed = seed.unwrap_or(0);
        // This opponent is itself a `build_search`-built config, on the
        // same iteration-based footing as the candidate -- both sides get
        // the *same* budget (an operator's `max_iterations` override
        // included) so there's nothing to match asymmetrically (see
        // `SearchBudget`'s and `build_search`'s doc comments).
        let budget = SearchBudget {
            max_iterations,
            max_time: max_time_ms.map(std::time::Duration::from_millis),
            ..Default::default()
        };
        // Fail fast on an invalid baseline config, before any games are
        // played -- mirrors how a bad candidate `params` is already
        // rejected during `TrialParams` deserialization inside
        // `strategy_tune_eval` itself.
        build_search::<G>(&cfg, baseline_seed, use_transpositions, &budget)?;
        strategy_tune_eval::<G>(
            &params,
            rounds,
            seed,
            use_transpositions,
            budget,
            move || {
                build_search::<G>(&cfg, baseline_seed, use_transpositions, &budget)
                    .expect("baseline_config already validated above")
            },
            Default::default(),
            trace_path.as_deref(),
            on_game,
        )?
    } else {
        strategy_tune_eval::<G>(
            &params,
            rounds,
            seed,
            use_transpositions,
            SearchBudget {
                max_iterations,
                max_time: max_time_ms.map(std::time::Duration::from_millis),
                ..Default::default()
            },
            move || {
                presets
                    .build::<G>(baseline_preset, preset_seed)
                    .unwrap_or_else(|e| {
                        panic!("{presets_source}'s {baseline_preset:?} preset must build: {e}")
                    })
            },
            Default::default(),
            trace_path.as_deref(),
            on_game,
        )?
    };
    Ok(json!({
        "cost": outcome.cost,
        "wins": outcome.wins,
        "losses": outcome.losses,
        "draws": outcome.draws,
    }))
}

/// `cost = losses / (2*rounds)`: the candidate's loss rate across the
/// `2*rounds` games it plays (moving first and second each round), the
/// quantity SMAC3 minimizes. Draws and wins both count as "not a loss" --
/// this doesn't reward wins over draws, only penalizes outright losses.
fn cost_from_losses(losses: u32, rounds: u32) -> f64 {
    let total = 2.0 * rounds as f64;
    if total > 0.0 {
        losses as f64 / total
    } else {
        0.0
    }
}

/// Play one game between `first` (moves when `player_to_move` is index 0)
/// and `second` (index 1). Returns `(first_result, second_result, draws)`
/// win/loss counts -- each pair is `(1,0)`/`(0,1)`/`(0,0)` with the third
/// element set on a draw.
///
/// Assumes a 2-player, win/loss/draw game (`G::winner() -> Option<P>` with
/// indices 0/1) -- every current caller is, but a 3+-player game would need
/// this generalized first.
#[allow(clippy::too_many_arguments)]
fn play_one<G: Game>(
    first: &mut dyn Search<G = G>,
    second: &mut dyn Search<G = G>,
    initial_state: G::S,
    mut tracer: Option<&mut trace::MoveTracer>,
    round: u32,
    seq: u64,
    seed: u64,
    candidate_side: ConfiguredCandidateSide,
) -> Result<ConfiguredMatchResult, HostError> {
    let started = std::time::Instant::now();
    let game_seq = tracer.as_mut().map(|t| t.start_game());
    let mut state = initial_state;
    let mut ply = 0u32;
    let mut measurements = Vec::new();

    if let (Some(t), Some(seq)) = (tracer.as_mut(), game_seq) {
        t.write_ply::<G::S, G::A>(seq, ply, &state, None, None);
    }

    while !G::is_terminal(&state) {
        let mover = G::player_to_move(&state).to_index();
        let pre_move_state = state.clone();
        let move_started = std::time::Instant::now();
        let action = if mover == 0 {
            first.choose_action(&state)
        } else {
            second.choose_action(&state)
        };
        let visits = if mover == 0 {
            first.root_report(&pre_move_state).total_visits
        } else {
            second.root_report(&pre_move_state).total_visits
        };
        state = G::apply(state, &action);
        ply += 1;
        measurements.push((mover, visits as u64, move_started.elapsed()));

        if let (Some(t), Some(seq)) = (tracer.as_mut(), game_seq) {
            let player = if mover == 0 {
                if candidate_side == ConfiguredCandidateSide::First {
                    "candidate"
                } else {
                    "baseline"
                }
            } else {
                if candidate_side == ConfiguredCandidateSide::Second {
                    "candidate"
                } else {
                    "baseline"
                }
            };
            t.write_ply(seq, ply, &state, Some(&action), Some(player));
        }
    }
    let outcome = match G::winner(&state) {
        None => ConfiguredOutcome::Draw,
        Some(p) if (p.to_index() == 0) == (candidate_side == ConfiguredCandidateSide::First) => {
            ConfiguredOutcome::CandidateWin
        }
        Some(_) => ConfiguredOutcome::BaselineWin,
    };
    let first_half = ply / 2;
    let mut candidate = ConfiguredStrategyMetrics::default();
    let mut baseline = ConfiguredStrategyMetrics::default();
    for (index, (mover, visits, elapsed)) in measurements.into_iter().enumerate() {
        let metrics = if (mover == 0) == (candidate_side == ConfiguredCandidateSide::First) {
            &mut candidate
        } else {
            &mut baseline
        };
        metrics.iterations_total = metrics.iterations_total.saturating_add(visits);
        metrics.move_time_ms = metrics
            .move_time_ms
            .saturating_add(elapsed.as_millis().min(u64::MAX as u128) as u64);
        if (index as u32 + 1) <= first_half {
            metrics.iterations_first_half = metrics.iterations_first_half.saturating_add(visits);
        }
    }
    Ok(ConfiguredMatchResult {
        record_type: "configured_match_result".into(),
        seq,
        round,
        seed,
        candidate_side,
        outcome,
        trace_game_seq: game_seq,
        plies: ply,
        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        candidate,
        baseline,
    })
}

// ---------------------------------------------------------------------------
// Tuner metadata
// ---------------------------------------------------------------------------

/// Search-space metadata for the full multi-family catalog above, for `tune
/// describe` to report to a SMAC3 harness or launch-form UI. `baselines` is
/// the list of preset ids a caller's `tune_eval` can build a
/// `strategy_tune_eval` `baseline_build` argument for -- most games report
/// exactly one entry; a game with a genuine second, harder preset can list
/// it as a second instance for SMAC3's multi-instance evaluation.
///
/// `game_config` always comes back `{}` here -- this function only knows
/// the strategy search space, not any per-game setup axis (that's
/// `GameAdapter::default_config()`'s job). A game with a real one overrides
/// the field on the returned value via struct-update syntax; see
/// `GameAdapter::tuner()` on e.g. `games/druid/src/main.rs`.
pub fn strategy_tuner_info(baselines: &[&str], eval_rounds: u32) -> TunerInfo {
    strategy_tuner_info_with_mcgs(baselines, eval_rounds, false)
}

/// Tuning schema for a game with a sound Zobrist hash. The `mcgs` boolean
/// selects the combined edge-and-node statistics graph mode; it is omitted
/// entirely for games that cannot safely create transposition tables.
pub fn strategy_tuner_info_with_mcgs(
    baselines: &[&str],
    eval_rounds: u32,
    supports_mcgs: bool,
) -> TunerInfo {
    let mut info = TunerInfo {
        id: "strategy".into(),
        baselines: baselines.iter().map(|s| s.to_string()).collect(),
        game_config: json!({}),
        eval_rounds,
        parameters: {
            let mut parameters = vec![
                param(
                    "family",
                    json!({"type": "categorical", "choices": family_choices(), "default": "rave"}),
                ),
                param(
                    "q_init",
                    json!({"type": "categorical", "choices": ["Draw", "Infinity", "Loss", "Parent", "Win"], "default": "Infinity"}),
                ),
            ];
            parameters.extend(tunable_field_parameters());
            parameters
        },
        conditions: {
            let mut conditions = family_conditions();
            // Gated by another field's own sampled value (`final_action`,
            // `schedule`, `rave_ucb`, `contempt`), not by `family` directly --
            // see `register_family!`'s doc comment in `family_catalog.rs` for
            // why these stay hand-written instead of per-row table entries.
            conditions.extend([
                condition(json!({"final_action": "secure_child"}), &["a"]),
                condition(json!({"schedule": "hand_selected"}), &["k"]),
                condition(json!({"schedule": "min_mse"}), &["bias"]),
                condition(json!({"schedule": "threshold"}), &["rave"]),
                condition(json!({"rave_ucb": ["ucb1", "tuned"]}), &["c"]),
                condition(json!({"contempt": "on"}), &["contempt_factor"]),
            ]);
            conditions
        },
    };
    if supports_mcgs {
        info.parameters
            .push(param("mcgs", json!({"type": "bool", "default": false})));
    }
    info
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use game_nim::Nim;

    #[test]
    fn test_cost_from_losses_hand_verified() {
        // 20 rounds -> 40 games; 15 losses -> cost 0.375.
        assert_eq!(cost_from_losses(15, 20), 0.375);
        assert_eq!(cost_from_losses(0, 20), 0.0);
        assert_eq!(cost_from_losses(40, 20), 1.0);
        // 10 losses out of 4 rounds (8 games) -> 1.25, clamped nowhere --
        // callers are expected to pass a `losses` that's actually <= 2*rounds.
        assert_eq!(cost_from_losses(10, 4), 1.25);
    }

    #[test]
    fn test_cost_from_losses_zero_rounds_is_zero() {
        assert_eq!(cost_from_losses(0, 0), 0.0);
    }

    // Bounded, unlike production `baseline_build` callers (which always pass
    // a real budgeted preset): the missing-field/unknown-value rejection
    // tests below never reach real play, but the family round-trip tests do,
    // and `TreeSearch::default()`'s `max_iterations` is `usize::MAX`.
    fn baseline() -> Box<dyn Search<G = Nim>> {
        Box::new(
            TreeSearch::<Nim, strategy::Ucb1>::new().config(SearchConfig::new().max_iterations(50)),
        )
    }

    fn rave_params() -> Value {
        json!({
            "family": "rave",
            "threshold": 700,
            "c": 0.3,
            "epsilon": 0.1,
            "q_init": "Infinity",
            "final_action": "robust_child",
            "schedule": "threshold",
            "rave": 700,
            "rave_ucb": "tuned",
        })
    }

    fn pn_params() -> Value {
        json!({
            "family": "ucb1_pn",
            "q_init": "Infinity",
            "c": 1.4,
            "c_pn": 1.0,
            "final_action": "robust_child",
            "solver_loss_threshold": 5,
            "contempt": "off",
        })
    }

    fn comparison_params() -> Value {
        json!({
            "family": "ucb1",
            "c": 1.4,
            "q_init": "Infinity",
            "final_action": "robust_child",
        })
    }

    #[test]
    fn mcgs_schema_is_available_only_to_hashing_games() {
        let plain = strategy_tuner_info(&["strong"], 1);
        assert!(!plain.parameters.iter().any(|p| p.name == "mcgs"));

        let graph = strategy_tuner_info_with_mcgs(&["strong"], 1, true);
        let mcgs = graph
            .parameters
            .iter()
            .find(|p| p.name == "mcgs")
            .expect("hashing games expose the MCGS switch");
        assert_eq!(mcgs.spec["type"], json!("bool"));
        assert_eq!(mcgs.spec["default"], json!(false));
    }

    #[test]
    fn configured_eval_streams_alternating_results_and_matches_aggregate() {
        let params = comparison_params();
        let budget = SearchBudget {
            max_iterations: Some(3),
            ..Default::default()
        };
        let mut records = Vec::new();
        let mut sink = |record| {
            records.push(record);
            Ok(())
        };
        let outcome = strategy_tune_eval::<Nim>(
            &params,
            2,
            Some(42),
            false,
            budget,
            || build_search::<Nim>(&params, 0, false, &budget).unwrap(),
            <Nim as Game>::S::default(),
            None,
            &mut sink,
        )
        .unwrap();

        assert_eq!(records.len(), 4);
        assert_eq!(
            records.iter().map(|record| record.seq).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            records
                .iter()
                .map(|record| record.candidate_side)
                .collect::<Vec<_>>(),
            vec![
                ConfiguredCandidateSide::First,
                ConfiguredCandidateSide::Second,
                ConfiguredCandidateSide::First,
                ConfiguredCandidateSide::Second,
            ]
        );
        let wins = records
            .iter()
            .filter(|record| record.outcome == ConfiguredOutcome::CandidateWin)
            .count() as u32;
        let losses = records
            .iter()
            .filter(|record| record.outcome == ConfiguredOutcome::BaselineWin)
            .count() as u32;
        let draws = records
            .iter()
            .filter(|record| record.outcome == ConfiguredOutcome::Draw)
            .count() as u32;
        assert_eq!(
            (wins, losses, draws),
            (outcome.wins, outcome.losses, outcome.draws)
        );
        for record in records {
            assert!(record.candidate.iterations_total > 0);
            assert!(record.baseline.iterations_total > 0);
            assert!(record.candidate.iterations_first_half <= record.candidate.iterations_total);
            assert!(record.baseline.iterations_first_half <= record.baseline.iterations_total);
        }
    }

    #[test]
    fn configured_eval_sink_error_stops_before_later_games() {
        let params = comparison_params();
        let budget = SearchBudget {
            max_iterations: Some(3),
            ..Default::default()
        };
        let mut seen = 0;
        let mut sink = |_record| {
            seen += 1;
            Err(HostError::internal("stop streaming"))
        };
        let err = strategy_tune_eval::<Nim>(
            &params,
            2,
            Some(42),
            false,
            budget,
            || build_search::<Nim>(&params, 0, false, &budget).unwrap(),
            <Nim as Game>::S::default(),
            None,
            &mut sink,
        )
        .expect_err("sink failure should abort the comparison");
        assert_eq!(seen, 1);
        assert_eq!(err.message, "stop streaming");
    }

    #[test]
    fn search_budget_time_and_default_iteration_limits_are_distinct() {
        assert_eq!(SearchBudget::default().iteration_limit(), MAX_ITER);
        assert_eq!(
            SearchBudget {
                max_time: Some(std::time::Duration::from_millis(1)),
                ..Default::default()
            }
            .iteration_limit(),
            usize::MAX
        );
        assert_eq!(
            SearchBudget {
                max_iterations: Some(7),
                max_time: Some(std::time::Duration::from_millis(1)),
                ..Default::default()
            }
            .iteration_limit(),
            7
        );
    }

    #[test]
    fn test_tune_eval_rejects_params_missing_required_field() {
        // "schedule": "threshold" requires "rave", which is absent -- this
        // must fail fast during config construction, before any game is
        // played (no real MCTS search runs in this test).
        let mut params = rave_params();
        params.as_object_mut().unwrap().remove("rave");
        let err = strategy_tune_eval::<Nim>(
            &params,
            1,
            Some(0),
            false,
            SearchBudget::default(),
            baseline,
            <Nim as Game>::S::default(),
            None,
            &mut |_| Ok(()),
        )
        .expect_err("missing `rave` must be rejected");
        assert!(err.message.contains("rave"));
    }

    #[test]
    fn zero_round_internal_validation_builds_candidate_without_playing() {
        let mut params = rave_params();
        params.as_object_mut().unwrap().remove("rave");
        let err = strategy_tune_eval::<Nim>(
            &params,
            0,
            Some(0),
            false,
            SearchBudget {
                max_iterations: Some(1),
                ..Default::default()
            },
            baseline,
            <Nim as Game>::S::default(),
            None,
            &mut |_| Ok(()),
        )
        .expect_err("zero-round validation must reach the strategy builder");
        assert!(err.message.contains("rave"));
    }

    #[test]
    fn test_tune_eval_rejects_unknown_schedule() {
        let mut params = rave_params();
        params["schedule"] = json!("not_a_real_schedule");
        let err = strategy_tune_eval::<Nim>(
            &params,
            1,
            Some(0),
            false,
            SearchBudget::default(),
            baseline,
            <Nim as Game>::S::default(),
            None,
            &mut |_| Ok(()),
        )
        .expect_err("unknown schedule must be rejected");
        assert!(err.message.contains("schedule"));
    }

    #[test]
    fn test_tune_eval_rejects_unknown_final_action() {
        let mut params = rave_params();
        params["final_action"] = json!("not_a_real_final_action");
        let err = strategy_tune_eval::<Nim>(
            &params,
            1,
            Some(0),
            false,
            SearchBudget::default(),
            baseline,
            <Nim as Game>::S::default(),
            None,
            &mut |_| Ok(()),
        )
        .expect_err("unknown final_action must be rejected");
        assert!(err.message.contains("final_action"));
    }

    #[test]
    fn test_tune_eval_rejects_unknown_contempt() {
        let mut params = pn_params();
        params["contempt"] = json!("not_a_real_contempt_mode");
        let err = strategy_tune_eval::<Nim>(
            &params,
            1,
            Some(0),
            false,
            SearchBudget::default(),
            baseline,
            <Nim as Game>::S::default(),
            None,
            &mut |_| Ok(()),
        )
        .expect_err("unknown contempt must be rejected");
        assert!(err.message.contains("contempt"));
    }

    #[test]
    fn test_tune_eval_rejects_contempt_on_missing_contempt_factor() {
        let mut params = pn_params();
        params["contempt"] = json!("on");
        params.as_object_mut().unwrap().remove("contempt_factor");
        let err = strategy_tune_eval::<Nim>(
            &params,
            1,
            Some(0),
            false,
            SearchBudget::default(),
            baseline,
            <Nim as Game>::S::default(),
            None,
            &mut |_| Ok(()),
        )
        .expect_err("contempt=on without contempt_factor must be rejected");
        assert!(err.message.contains("contempt_factor"));
    }

    #[test]
    fn test_tune_eval_rejects_unknown_family() {
        let mut params = rave_params();
        params["family"] = json!("not_a_real_family");
        let err = strategy_tune_eval::<Nim>(
            &params,
            1,
            Some(0),
            false,
            SearchBudget::default(),
            baseline,
            <Nim as Game>::S::default(),
            None,
            &mut |_| Ok(()),
        )
        .expect_err("unknown family must be rejected");
        assert!(err.message.contains("family"));
    }

    /// One hand-verified construction+round-trip test per new family arm,
    /// each playing a single round of `Nim` (fast, deterministic) to prove
    /// the concrete type actually builds and the declared params round-trip
    /// through `make_candidate` without error. `cost_from_losses` itself is
    /// already covered above -- this only exercises dispatch.
    fn assert_family_round_trips(mut params: Value) {
        params["q_init"] = json!("Infinity");
        let outcome = strategy_tune_eval::<Nim>(
            &params,
            1,
            Some(0),
            false,
            SearchBudget::default(),
            baseline,
            <Nim as Game>::S::default(),
            None,
            &mut |_| Ok(()),
        )
        .unwrap_or_else(|e| {
            panic!(
                "family {:?} should round-trip: {}",
                params["family"], e.message
            )
        });
        assert!(outcome.wins + outcome.losses + outcome.draws == 2);
    }

    #[test]
    fn test_family_ucb1_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1", "c": 1.4, "final_action": "robust_child",
        }));
    }

    #[test]
    fn test_family_ucb1_dm_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_dm", "c": 1.4, "final_action": "max_avg",
        }));
    }

    #[test]
    fn test_family_ucb1_mast_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child",
        }));
    }

    #[test]
    fn test_family_ucb1_nst_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_nst", "c": 1.4, "epsilon": 0.2,
            "nst_backoff_threshold": 3, "final_action": "robust_child",
        }));
    }

    #[test]
    fn test_family_ucb1_progressive_history_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_progressive_history", "c": 1.4, "ph_weight": 0.5,
            "final_action": "robust_child",
        }));
    }

    #[test]
    fn test_family_ucb1_max_robust_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_max_robust", "c": 1.4,
        }));
    }

    #[test]
    fn test_family_amaf_round_trips() {
        assert_family_round_trips(json!({
            "family": "amaf", "c": 1.4, "amaf_alpha": 0.5, "final_action": "secure_child", "a": 4.0,
        }));
    }

    #[test]
    fn test_family_amaf_mast_round_trips() {
        assert_family_round_trips(json!({
            "family": "amaf_mast", "c": 1.4, "amaf_alpha": 0.5, "epsilon": 0.2,
            "final_action": "robust_child",
        }));
    }

    #[test]
    fn test_family_ucb1_tuned_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_tuned", "c": 1.4, "final_action": "robust_child",
        }));
    }

    #[test]
    fn test_family_ucb1_tuned_mast_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_tuned_mast", "c": 1.4, "final_action": "robust_child",
        }));
    }

    #[test]
    fn test_family_ucb1_tuned_dm_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_tuned_dm", "c": 1.4, "final_action": "robust_child",
        }));
    }

    #[test]
    fn test_family_ucb1_tuned_dm_mast_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_tuned_dm_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child",
        }));
    }

    #[test]
    fn test_family_ucb1_dm_nst_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_dm_nst", "c": 1.4, "epsilon": 0.2,
            "nst_backoff_threshold": 3, "final_action": "robust_child",
        }));
    }

    // `meta_mcts`'s round trip is proven in `tests/stress.rs` instead of here:
    // its inner nested search makes even one candidate-vs-baseline game
    // noticeably slower than every other family's (multi-second, not the
    // sub-second every sibling test above runs in), so it belongs in the
    // slow/stress suite `cargo test --lib` never compiles, not this fast one.

    #[test]
    fn test_family_rave_round_trips() {
        assert_family_round_trips(rave_params());
    }

    #[test]
    fn test_family_ucb1_pn_round_trips() {
        assert_family_round_trips(pn_params());
    }

    #[test]
    fn test_family_ucb1_pn_mast_round_trips() {
        assert_family_round_trips(json!({
            "family": "ucb1_pn_mast", "c": 1.4, "c_pn": 1.0, "epsilon": 0.2,
            "final_action": "robust_child", "solver_loss_threshold": 5,
            "contempt": "on", "contempt_factor": -0.5,
        }));
    }

    // -----------------------------------------------------------------
    // `to_search_spec` -- config_ir conversion (step 4c). Not yet wired
    // into `make_candidate` (that's step 4d); these tests pin the exact
    // `SearchSpec`/`SearchSettings` shape each family converts to.
    // -----------------------------------------------------------------

    fn trial(params: Value) -> TrialParams {
        serde_json::from_value(params).unwrap()
    }

    #[test]
    fn to_search_spec_ucb1() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1", "c": 1.4, "q_init": "Infinity", "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec,
            config_ir::SearchSpec {
                select: config_ir::SelectSpec::Ucb1 { c: 1.4 },
                simulate: config_ir::SimulateSpec::Uniform {},
                backprop: config_ir::BackpropSpec::Classic {},
                final_action: config_ir::FinalActionSpec::RobustChild {},
            }
        );
    }

    #[test]
    fn to_search_spec_ucb1_dm() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_dm", "c": 1.4, "q_init": "Infinity", "final_action": "max_avg",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec,
            config_ir::SearchSpec {
                select: config_ir::SelectSpec::Ucb1 { c: 1.4 },
                simulate: config_ir::SimulateSpec::DecisiveMove {
                    mode: simulate::DecisiveMoveMode::Win,
                    inner: config_ir::BaseSimulateSpec::Uniform {},
                },
                backprop: config_ir::BackpropSpec::Classic {},
                final_action: config_ir::FinalActionSpec::MaxAvg {},
            }
        );
    }

    #[test]
    fn to_search_spec_ucb1_mast() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_mast", "c": 1.4, "epsilon": 0.2, "q_init": "Infinity",
                "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec.simulate,
            config_ir::SimulateSpec::EpsilonGreedy {
                epsilon: 0.2,
                inner: config_ir::BaseSimulateSpec::Mast {},
            }
        );
    }

    #[test]
    fn to_search_spec_ucb1_nst() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3,
                "q_init": "Infinity", "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec.simulate,
            config_ir::SimulateSpec::EpsilonGreedy {
                epsilon: 0.2,
                inner: config_ir::BaseSimulateSpec::Nst {
                    backoff_threshold: 3
                },
            }
        );
    }

    #[test]
    fn to_search_spec_ucb1_dm_nst() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_dm_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3,
                "q_init": "Infinity", "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec.simulate,
            config_ir::SimulateSpec::DecisiveMoveNst {
                mode: simulate::DecisiveMoveMode::Win,
                epsilon: 0.2,
                nst_backoff_threshold: 3,
            }
        );
    }

    #[test]
    fn to_search_spec_ucb1_progressive_history() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_progressive_history", "c": 1.4, "ph_weight": 0.5,
                "q_init": "Infinity", "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec.select,
            config_ir::SelectSpec::ProgressiveHistory {
                c: 1.4,
                ph_weight: 0.5
            }
        );
    }

    #[test]
    fn to_search_spec_amaf() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "amaf", "c": 1.4, "amaf_alpha": 0.5, "q_init": "Infinity",
                "final_action": "secure_child", "a": 4.0,
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec,
            config_ir::SearchSpec {
                select: config_ir::SelectSpec::Amaf { alpha: 0.5, c: 1.4 },
                simulate: config_ir::SimulateSpec::Uniform {},
                backprop: config_ir::BackpropSpec::Classic {},
                final_action: config_ir::FinalActionSpec::SecureChild { a: 4.0 },
            }
        );
    }

    #[test]
    fn to_search_spec_amaf_mast() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "amaf_mast", "c": 1.4, "amaf_alpha": 0.5, "epsilon": 0.2,
                "q_init": "Infinity", "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec.simulate,
            config_ir::SimulateSpec::EpsilonGreedy {
                epsilon: 0.2,
                inner: config_ir::BaseSimulateSpec::Mast {},
            }
        );
    }

    #[test]
    fn to_search_spec_ucb1_tuned() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_tuned", "c": 1.4, "q_init": "Infinity",
                "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(spec.select, config_ir::SelectSpec::Ucb1Tuned { c: 1.4 });
        assert_eq!(spec.simulate, config_ir::SimulateSpec::Uniform {});
    }

    #[test]
    fn to_search_spec_ucb1_tuned_mast() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_tuned_mast", "c": 1.4, "q_init": "Infinity",
                "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(spec.simulate, config_ir::SimulateSpec::Mast {});
    }

    #[test]
    fn to_search_spec_ucb1_tuned_dm() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_tuned_dm", "c": 1.4, "q_init": "Infinity",
                "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec.simulate,
            config_ir::SimulateSpec::DecisiveMove {
                mode: simulate::DecisiveMoveMode::Win,
                inner: config_ir::BaseSimulateSpec::Uniform {},
            }
        );
    }

    #[test]
    fn to_search_spec_ucb1_tuned_dm_mast() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_tuned_dm_mast", "c": 1.4, "epsilon": 0.2, "q_init": "Infinity",
                "final_action": "robust_child",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec.simulate,
            config_ir::SimulateSpec::DecisiveMoveMast {
                mode: simulate::DecisiveMoveMode::Win,
                epsilon: 0.2,
            }
        );
    }

    #[test]
    fn to_search_spec_rave() {
        let (spec, _) =
            to_search_spec(&trial(rave_params()), 0, false, &SearchBudget::default()).unwrap();
        assert_eq!(
            spec.select,
            config_ir::SelectSpec::Rave {
                threshold: 700,
                schedule: RaveSchedule::Threshold { rave: 700 },
                ucb: RaveUcb::Ucb1Tuned {
                    exploration_constant: 0.3
                },
            }
        );
        assert_eq!(
            spec.simulate,
            config_ir::SimulateSpec::DecisiveMoveMast {
                mode: simulate::DecisiveMoveMode::WinLoss,
                epsilon: 0.1,
            }
        );
    }

    #[test]
    fn to_search_spec_ucb1_pn() {
        let (spec, settings) =
            to_search_spec(&trial(pn_params()), 0, false, &SearchBudget::default()).unwrap();
        assert_eq!(
            spec.select,
            config_ir::SelectSpec::UctPn { c: 1.4, c_pn: 1.0 }
        );
        assert_eq!(settings.solver_loss_threshold, Some(5));
        assert_eq!(settings.contempt_factor, None);
    }

    #[test]
    fn to_search_spec_ucb1_pn_mast() {
        let (spec, settings) = to_search_spec(
            &trial(json!({
                "family": "ucb1_pn_mast", "c": 1.4, "c_pn": 1.0, "epsilon": 0.2,
                "q_init": "Infinity", "final_action": "robust_child",
                "solver_loss_threshold": 5, "contempt": "on", "contempt_factor": -0.5,
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec.simulate,
            config_ir::SimulateSpec::EpsilonGreedy {
                epsilon: 0.2,
                inner: config_ir::BaseSimulateSpec::Mast {},
            }
        );
        assert_eq!(settings.solver_loss_threshold, Some(5));
        assert_eq!(settings.contempt_factor, Some(-0.5));
    }

    #[test]
    fn to_search_spec_ucb1_max_robust() {
        let (spec, _) = to_search_spec(
            &trial(json!({
                "family": "ucb1_max_robust", "c": 1.4, "q_init": "Infinity",
            })),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec,
            config_ir::SearchSpec {
                select: config_ir::SelectSpec::Ucb1 { c: 1.4 },
                simulate: config_ir::SimulateSpec::Uniform {},
                backprop: config_ir::BackpropSpec::Classic {},
                final_action: config_ir::FinalActionSpec::MaxRobustChild {},
            }
        );
    }

    #[test]
    fn to_search_spec_meta_mcts() {
        let (spec, _) = to_search_spec(
            &trial(json!({"family": "meta_mcts", "c": 1.4, "q_init": "Infinity"})),
            0,
            false,
            &SearchBudget::default(),
        )
        .unwrap();
        assert_eq!(
            spec,
            config_ir::SearchSpec {
                select: config_ir::SelectSpec::Ucb1 { c: 1.4 },
                simulate: config_ir::SimulateSpec::MetaMcts {
                    iterations: META_MCTS_INNER_ITERATIONS
                },
                backprop: config_ir::BackpropSpec::Classic {},
                final_action: config_ir::FinalActionSpec::MaxAvg {},
            }
        );
    }

    #[test]
    fn to_search_spec_settings_mirror_base_config() {
        let (_, settings) = to_search_spec(
            &trial(comparison_params()),
            7,
            true,
            &SearchBudget {
                max_iterations: Some(123),
                threads: 4,
                max_time: Some(std::time::Duration::from_secs(1)),
            },
        )
        .unwrap();
        assert_eq!(settings.max_iterations, 123);
        assert_eq!(settings.max_playout_depth, PLAYOUT_DEPTH);
        assert_eq!(settings.expand_threshold, EXPAND_THRESHOLD);
        assert!(matches!(settings.q_init, QInit::Infinity));
        assert!(settings.use_transpositions);
        assert!(settings.use_mcts_solver);
        assert!(settings.reuse_tree);
        assert_eq!(settings.num_tree_threads, 4);
        assert_eq!(settings.seed, 7);
        assert_eq!(settings.max_time, Some(std::time::Duration::from_secs(1)));
        assert_eq!(settings.graph_search, None);
    }

    #[test]
    fn to_search_spec_mcgs_sets_graph_search_and_disables_transpositions_and_reuse() {
        let mut params = comparison_params();
        params["mcgs"] = json!(true);
        let (_, settings) =
            to_search_spec(&trial(params), 0, true, &SearchBudget::default()).unwrap();
        assert_eq!(
            settings.graph_search,
            Some(GraphSearch::Dag(GraphStats::Both))
        );
        assert!(!settings.use_transpositions);
        assert!(!settings.reuse_tree);
    }

    #[test]
    fn to_search_spec_mcgs_without_transpositions_is_rejected() {
        let mut params = comparison_params();
        params["mcgs"] = json!(true);
        // `(SearchSpec, SearchSettings)` isn't `Debug`, so `expect_err` doesn't
        // apply here -- match instead (see `test_build_search_rejects_unknown_family`).
        let err = match to_search_spec(&trial(params), 0, false, &SearchBudget::default()) {
            Err(e) => e,
            Ok(_) => panic!("mcgs without a zobrist hash must be rejected"),
        };
        assert!(err.message.contains("zobrist"));
    }

    #[test]
    fn to_search_spec_rejects_missing_required_field() {
        let mut params = rave_params();
        params.as_object_mut().unwrap().remove("rave");
        let err = match to_search_spec(&trial(params), 0, false, &SearchBudget::default()) {
            Err(e) => e,
            Ok(_) => panic!("missing `rave` must be rejected"),
        };
        assert!(err.message.contains("rave"));
    }

    #[test]
    fn to_search_spec_rejects_unknown_family() {
        let mut params = rave_params();
        params["family"] = json!("not_a_real_family");
        let err = match to_search_spec(&trial(params), 0, false, &SearchBudget::default()) {
            Err(e) => e,
            Ok(_) => panic!("unknown family must be rejected"),
        };
        assert!(err.message.contains("family"));
    }

    /// Proves `build_search` (the public entry point `GameAdapter::
    /// tune_eval`'s `baseline_config` path uses) works as a
    /// `strategy_tune_eval` `baseline_build` source, not just as a
    /// standalone constructor -- a UCB1-built opponent played against a RAVE
    /// candidate for one round.
    #[test]
    fn test_strategy_tune_eval_with_config_built_baseline_round_trips() {
        let baseline_params = json!({
            "family": "ucb1", "c": 1.4, "final_action": "robust_child", "q_init": "Infinity",
        });
        let outcome = strategy_tune_eval::<Nim>(
            &rave_params(),
            1,
            Some(0),
            false,
            SearchBudget::default(),
            || {
                build_search::<Nim>(&baseline_params, 0, false, &SearchBudget::default())
                    .expect("baseline_params is a valid ucb1 config")
            },
            <Nim as Game>::S::default(),
            None,
            &mut |_| Ok(()),
        )
        .expect("candidate vs config-built baseline should round-trip");
        assert_eq!(outcome.wins + outcome.losses + outcome.draws, 2);
    }

    /// `random`/`flat_mc` are floor families reachable only via
    /// `build_search`/`--baseline-config` (a ladder's floor rung), never
    /// sampled as a SMAC3 candidate -- proven by their absence from
    /// `strategy_tuner_info().parameters`'s `family` choices below.
    #[test]
    fn test_build_search_builds_random_floor_family() {
        build_search::<Nim>(
            &json!({"family": "random", "q_init": "Infinity"}),
            0,
            false,
            &SearchBudget::default(),
        )
        .expect("random should build with just family/q_init");
    }

    #[test]
    fn test_build_search_builds_flat_mc_floor_family() {
        build_search::<Nim>(
            &json!({"family": "flat_mc", "q_init": "Infinity"}),
            0,
            false,
            &SearchBudget::default(),
        )
        .expect("flat_mc should build with just family/q_init");
    }

    #[test]
    fn test_strategy_tuner_info_excludes_floor_families_from_searchable_choices() {
        let tuner = strategy_tuner_info(&["strong"], 1);
        let family = tuner
            .parameters
            .iter()
            .find(|p| p.name == "family")
            .expect("family param must exist");
        let choices = family.spec["choices"]
            .as_array()
            .expect("family choices must be an array")
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(
            !choices.contains(&"random") && !choices.contains(&"flat_mc"),
            "floor families must never be SMAC3-searchable candidates: {choices:?}"
        );
    }

    #[test]
    fn test_build_search_rejects_unknown_family() {
        let mut params = rave_params();
        params["family"] = json!("not_a_real_family");
        // `Box<dyn Search<G>>` isn't `Debug`, so `Result::expect_err` doesn't
        // apply here -- match instead.
        let err = match build_search::<Nim>(&params, 0, false, &SearchBudget::default()) {
            Err(e) => e,
            Ok(_) => panic!("unknown family must be rejected"),
        };
        assert!(err.message.contains("family"));
    }

    /// The parameter set each family's `make_candidate` arm actually
    /// requires -- mirrors the literals already passed to
    /// `assert_family_round_trips` above, plus `meta_mcts` (whose own
    /// round-trip lives in `tests/stress.rs` for cost reasons, but this
    /// check is pure metadata with no MCTS search, so it's cheap to include
    /// here too).
    ///
    /// Deliberately still hand-written rather than generated from
    /// `register_family!`'s per-row field lists (`family_conditions()`):
    /// those rows only name *which* top-level fields a family reads, not
    /// concrete values, so they can't exercise the nested conditions this
    /// test also needs to cover -- `rave`'s `schedule`/`rave_ucb`-gated
    /// fields, `final_action: secure_child`'s `a`, `contempt: on`'s
    /// `contempt_factor` -- all of which are hand-written conditions
    /// `strategy_tuner_info_with_mcgs` appends precisely because they
    /// depend on a *child* field's own sampled value, not on `family`
    /// alone (see `family_catalog.rs`'s `register_family!` doc comment).
    /// Generating a fixture from the field-name list alone would only be
    /// able to assert "this field is active", which `family_conditions()`
    /// already guarantees by construction -- a tautology, not a check.
    /// What would still silently drift is a *new* family being added to
    /// `register_family!` without a matching entry here; that's covered by
    /// `test_family_required_params_covers_every_registered_family` below
    /// instead, which needs no concrete values.
    fn family_required_params() -> Vec<(&'static str, Value)> {
        vec![
            (
                "ucb1",
                json!({"family": "ucb1", "c": 1.4, "final_action": "robust_child"}),
            ),
            (
                "ucb1_dm",
                json!({"family": "ucb1_dm", "c": 1.4, "final_action": "max_avg"}),
            ),
            (
                "ucb1_mast",
                json!({"family": "ucb1_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child"}),
            ),
            (
                "ucb1_nst",
                json!({"family": "ucb1_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3, "final_action": "robust_child"}),
            ),
            (
                "ucb1_progressive_history",
                json!({"family": "ucb1_progressive_history", "c": 1.4, "ph_weight": 0.5, "final_action": "robust_child"}),
            ),
            (
                "ucb1_max_robust",
                json!({"family": "ucb1_max_robust", "c": 1.4}),
            ),
            (
                "amaf",
                json!({"family": "amaf", "c": 1.4, "amaf_alpha": 0.5, "final_action": "secure_child", "a": 4.0}),
            ),
            (
                "amaf_mast",
                json!({"family": "amaf_mast", "c": 1.4, "amaf_alpha": 0.5, "epsilon": 0.2, "final_action": "robust_child"}),
            ),
            (
                "ucb1_tuned",
                json!({"family": "ucb1_tuned", "c": 1.4, "final_action": "robust_child"}),
            ),
            (
                "ucb1_tuned_mast",
                json!({"family": "ucb1_tuned_mast", "c": 1.4, "final_action": "robust_child"}),
            ),
            (
                "ucb1_tuned_dm",
                json!({"family": "ucb1_tuned_dm", "c": 1.4, "final_action": "robust_child"}),
            ),
            (
                "ucb1_tuned_dm_mast",
                json!({"family": "ucb1_tuned_dm_mast", "c": 1.4, "epsilon": 0.2, "final_action": "robust_child"}),
            ),
            ("rave", rave_params()),
            (
                "ucb1_dm_nst",
                json!({"family": "ucb1_dm_nst", "c": 1.4, "epsilon": 0.2, "nst_backoff_threshold": 3, "final_action": "robust_child"}),
            ),
            ("meta_mcts", json!({"family": "meta_mcts", "c": 1.4})),
            ("ucb1_pn", pn_params()),
            (
                "ucb1_pn_mast",
                json!({
                    "family": "ucb1_pn_mast", "c": 1.4, "c_pn": 1.0, "epsilon": 0.2,
                    "final_action": "robust_child", "solver_loss_threshold": 5,
                    "contempt": "on", "contempt_factor": -0.5,
                }),
            ),
        ]
    }

    #[test]
    fn test_family_required_params_covers_every_registered_family() {
        // The gap `family_required_params()`'s own doc comment identifies:
        // a family added to `register_family!` without a matching fixture
        // here wouldn't fail anything, it would just silently skip that
        // family in `test_tuner_info_conditions_cover_every_family_param_make_candidate_needs`
        // below. Comparing the two name sets closes that gap without
        // needing `family_required_params()` to become generated data.
        let registered: std::collections::HashSet<&str> = family_choices().into_iter().collect();
        let covered: std::collections::HashSet<&str> = family_required_params()
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            registered, covered,
            "family_required_params() must have exactly one fixture per family_catalog::family_choices() entry"
        );
    }

    /// The fixed point of "active" parameter names implied by
    /// `TunerInfo.conditions` for one fully-assigned trial config -- the
    /// same any-of/if-then evaluation a SMAC3 `ConfigSpace` performs,
    /// chasing multi-level conditions (e.g. `family: rave` activates
    /// `schedule`, whose own sampled value in turn activates one of
    /// `rave`/`k`/`bias`).
    fn active_params(tuner: &TunerInfo, chosen: &Value) -> std::collections::HashSet<String> {
        let chosen = chosen.as_object().expect("params must be an object");
        let mut active: std::collections::HashSet<String> =
            ["family", "q_init"].iter().map(|s| s.to_string()).collect();
        loop {
            let mut added = false;
            for cond in &tuner.conditions {
                let (parent, expected) = cond
                    .if_
                    .as_object()
                    .and_then(|m| m.iter().next())
                    .expect("condition `if` is a single-entry object");
                if !active.contains(parent) {
                    continue;
                }
                let Some(actual) = chosen.get(parent) else {
                    continue;
                };
                let matches = match expected {
                    Value::Array(vals) => vals.contains(actual),
                    other => other == actual,
                };
                if matches {
                    for name in &cond.then {
                        if active.insert(name.clone()) {
                            added = true;
                        }
                    }
                }
            }
            if !added {
                break;
            }
        }
        active
    }

    #[test]
    fn test_tuner_info_conditions_cover_every_family_param_make_candidate_needs() {
        // Regression coverage for a real bug: `make_candidate`'s `rave` arm
        // always required `epsilon`, but `strategy_tuner_info`'s conditions
        // never activated `epsilon` for `family: rave` -- so a real SMAC3
        // search built from this metadata could (and did) sample seemingly
        // valid `rave` configs the binary then rejected as missing a param.
        // For every family, every key its own round-trip fixture supplies
        // must be reachable as "active" from `strategy_tuner_info`'s
        // declared conditions given that exact assignment, catching any
        // future family where a hand-written fixture and the declared
        // schema's activation drift apart the same way.
        let tuner = strategy_tuner_info(&["strong"], 1);
        for (family, params) in family_required_params() {
            let active = active_params(&tuner, &params);
            for key in params.as_object().unwrap().keys() {
                assert!(
                    active.contains(key),
                    "family {family:?}: param {key:?} is required by make_candidate but \
                     strategy_tuner_info's conditions never mark it active for this config"
                );
            }
        }
    }
}
