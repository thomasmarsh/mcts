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

use game_host::{HostError, TunerCondition, TunerInfo, TunerParameter};
use mcts::game::{Game, PlayerIndex};
use mcts::strategies::mcts::select::{self, RaveSchedule, RaveUcb, SelectStrategy};
use mcts::strategies::mcts::simulate::SimulateStrategy;
use mcts::strategies::mcts::strategy::{self, Compose};
use mcts::strategies::mcts::{backprop, node::QInit, simulate, SearchConfig, Strategy, TreeSearch};
use mcts::strategies::Search;
use serde_json::{json, Value};

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

/// PN-MCTS families (Kowalski, Doe, Winands, Górski & Soemers, "Proof
/// Number Based Monte-Carlo Tree Search", 2023): `select::UctPn` wraps plain
/// UCB1 with a rank-based bonus from proof/disproof numbers, only meaningful
/// with MCTS-Solver on (see `select::UctPn`'s doc comment) -- so both arms
/// below force `use_mcts_solver(true)` and `reuse_tree(true)` rather than
/// exposing either as a tunable (PNS-style search always wants its
/// proof/disproof numbers carried across moves, not rebuilt from scratch
/// every `choose_action` call), and expose the paper's own per-game knobs
/// (`c_pn`, the proven-loss selection threshold `T` as
/// `solver_loss_threshold`, and the final-move-selection contempt factor)
/// as tunable params instead of assuming the paper's published values
/// transfer to this repo's games.
const PN_FAMILIES: &[&str] = &["ucb1_pn", "ucb1_pn_mast"];

/// Families whose own named `strategy.rs` type leaves `final_action`
/// configurable (the common `RobustChild`/`SecureChild` slot) rather than
/// fixing something else -- these are the ones `tune eval`'s `final_action`
/// param applies to.
const FINAL_ACTION_FAMILIES: &[&str] = &[
    "ucb1",
    "ucb1_dm",
    "ucb1_mast",
    "ucb1_nst",
    "ucb1_progressive_history",
    "amaf",
    "amaf_mast",
    "ucb1_tuned",
    "ucb1_tuned_mast",
    "ucb1_tuned_dm",
    "ucb1_tuned_dm_mast",
    "rave",
    "ucb1_pn",
    "ucb1_pn_mast",
];

/// Families that share the plain exploration-constant `c` parameter (every
/// family whose `Select` is `select::Ucb1`, `select::Ucb1Tuned`, or
/// `select::Amaf`/`select::ProgressiveHistory`, all of which wrap one).
/// `rave`'s own `c` is gated separately, by `rave_ucb`, since it's only
/// meaningful for two of `rave`'s three UCB modes.
const C_FAMILIES: &[&str] = &[
    "ucb1",
    "ucb1_dm",
    "ucb1_mast",
    "ucb1_nst",
    "ucb1_progressive_history",
    "amaf",
    "amaf_mast",
    "ucb1_tuned",
    "ucb1_tuned_mast",
    "ucb1_tuned_dm",
    "ucb1_tuned_dm_mast",
    "ucb1_max_robust",
    "meta_mcts",
    "ucb1_pn",
    "ucb1_pn_mast",
];

/// Families whose simulate step is (or wraps) an epsilon-greedy policy.
/// `rave`'s own simulate step (`DecisiveMove<EpsilonGreedy<Mast>>`, see its
/// `make_candidate` arm) wraps one too, same as `ucb1_tuned_dm_mast`'s, so
/// it belongs here: `make_candidate`'s `rave` arm requires `epsilon`
/// unconditionally, and this list is what makes that requirement visible to
/// callers (SMAC3's ConfigSpace, the launch-form UI) as an active parameter.
const EPSILON_FAMILIES: &[&str] = &[
    "ucb1_mast",
    "ucb1_nst",
    "amaf_mast",
    "ucb1_tuned_dm_mast",
    "rave",
    "ucb1_pn_mast",
];

fn base_config<G: Game, S: Strategy<G> + Default>(
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<SearchConfig<G, S>, HostError> {
    let q_init = QInit::from_str(&p.q_init)
        .map_err(|_| HostError::bad_request(format!("invalid q_init: {}", p.q_init)))?;
    let mut config = SearchConfig::new()
        .max_iterations(MAX_ITER)
        .max_playout_depth(PLAYOUT_DEPTH)
        .expand_threshold(EXPAND_THRESHOLD)
        .q_init(q_init)
        .use_transpositions(use_transpositions)
        // Every hand-built preset in every game turns both of these on --
        // a tuned candidate that doesn't is being measured under search
        // conditions nothing actually deployed ever runs with. Only the PN
        // families used to force these (their own `select::UctPn` needs
        // both to be meaningful at all); there's no reason every other
        // family shouldn't get the same tactical sharpness and continuity.
        .use_mcts_solver(true)
        .reuse_tree(true)
        .num_tree_threads(budget.threads)
        .seed(seed);
    if let Some(max_time) = budget.max_time {
        config = config.max_time(max_time);
    }
    Ok(config)
}

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
#[derive(Debug, Clone, Copy)]
pub struct SearchBudget {
    pub max_time: Option<std::time::Duration>,
    pub threads: usize,
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self {
            max_time: None,
            threads: 1,
        }
    }
}

/// One trial's candidate parameters, deserialized from the `params` JSON
/// object `strategy_tune_eval` receives -- the merged active-parameter set a
/// SMAC3 harness builds from its search-space YAML. `family` selects which
/// of the fields below are actually required; everything except `family`/
/// `q_init` is `Option` because it's only meaningful for a subset of
/// families (validated per-family in `make_candidate`, the same way missing
/// required fields were already rejected before `family` existed).
#[derive(Debug, serde::Deserialize)]
pub struct TrialParams {
    family: String,
    q_init: String,
    final_action: Option<String>,
    a: Option<f64>,
    c: Option<f64>,
    epsilon: Option<f64>,
    amaf_alpha: Option<f64>,
    ph_weight: Option<f64>,
    nst_backoff_threshold: Option<u32>,
    threshold: Option<u32>,
    bias: Option<f64>,
    schedule: Option<String>,
    k: Option<u32>,
    rave: Option<u32>,
    rave_ucb: Option<String>,
    c_pn: Option<f64>,
    solver_loss_threshold: Option<u32>,
    contempt: Option<String>,
    contempt_factor: Option<f64>,
}

/// Builds a candidate for any family whose `strategy.rs` counterpart leaves
/// `final_action` configurable: dispatches on `p.final_action` and
/// monomorphizes `Compose<Sel, Sim, backprop::Classic, FA>` for whichever of
/// `SecureChild`/`RobustChild` was chosen, mirroring the three-arm dispatch
/// `rave`'s own `make_candidate` used before other families existed.
fn build_with_final_action<G, Sel, Sim>(
    sel: Sel,
    sim: Sim,
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<Box<dyn Search<G = G>>, HostError>
where
    G: Game + 'static,
    Sel: SelectStrategy<G> + 'static,
    Sim: SimulateStrategy<G> + 'static,
{
    let fa = p
        .final_action
        .as_deref()
        .ok_or_else(|| HostError::bad_request("missing param: final_action"))?;
    match fa {
        "max_avg" => {
            let config =
                base_config::<G, Compose<Sel, Sim, backprop::Classic, select::SecureChild>>(
                    p,
                    seed,
                    use_transpositions,
                    budget,
                )?
                .select(sel)
                .simulate(sim);
            Ok(Box::new(
                TreeSearch::<G, Compose<Sel, Sim, backprop::Classic, select::SecureChild>>::new()
                    .config(config),
            ))
        }
        "secure_child" => {
            let a =
                p.a.ok_or_else(|| HostError::bad_request("missing param: a"))?;
            let mut config = base_config::<
                G,
                Compose<Sel, Sim, backprop::Classic, select::SecureChild>,
            >(p, seed, use_transpositions, budget)?
            .select(sel)
            .simulate(sim);
            config.final_action.a = a;
            Ok(Box::new(
                TreeSearch::<G, Compose<Sel, Sim, backprop::Classic, select::SecureChild>>::new()
                    .config(config),
            ))
        }
        "robust_child" => {
            let config =
                base_config::<G, Compose<Sel, Sim, backprop::Classic, select::RobustChild>>(
                    p,
                    seed,
                    use_transpositions,
                    budget,
                )?
                .select(sel)
                .simulate(sim);
            Ok(Box::new(
                TreeSearch::<G, Compose<Sel, Sim, backprop::Classic, select::RobustChild>>::new()
                    .config(config),
            ))
        }
        other => Err(HostError::bad_request(format!(
            "unknown final_action: {other}"
        ))),
    }
}

/// Builds a candidate for one of the PN-MCTS families (`ucb1_pn`/
/// `ucb1_pn_mast`): like `build_with_final_action`, `final_action` stays
/// configurable. `use_mcts_solver`/`reuse_tree` are no longer set here
/// explicitly -- `base_config` now turns both on unconditionally for every
/// family (PNS-style search's need for them was always the general case,
/// not a PN-specific one) -- only the solver's own tunable knobs
/// (`solver_loss_threshold`, `contempt_factor`) stay applied here, alongside
/// `UctPn`'s `c_pn`.
/// The solver-side knobs `build_pn_with_final_action` applies on top of
/// `select::UctPn` -- grouped into one struct rather than two more bare
/// arguments to stay under clippy's `too_many_arguments` threshold.
struct PnSolverParams {
    solver_loss_threshold: u32,
    contempt_factor: Option<f64>,
}

fn build_pn_with_final_action<G, Sim>(
    c: f64,
    c_pn: f64,
    sim: Sim,
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
    solver: PnSolverParams,
) -> Result<Box<dyn Search<G = G>>, HostError>
where
    G: Game + 'static,
    Sim: SimulateStrategy<G> + 'static,
{
    let sel = select::UctPn::with_c(c, c_pn);
    let PnSolverParams {
        solver_loss_threshold,
        contempt_factor,
    } = solver;
    let fa = p
        .final_action
        .as_deref()
        .ok_or_else(|| HostError::bad_request("missing param: final_action"))?;
    match fa {
        "max_avg" => {
            let config = base_config::<
                G,
                Compose<select::UctPn, Sim, backprop::Classic, select::SecureChild>,
            >(p, seed, use_transpositions, budget)?
            .select(sel)
            .simulate(sim)
            .solver_loss_threshold(solver_loss_threshold)
            .contempt_factor(contempt_factor);
            Ok(
                Box::new(
                    TreeSearch::<
                        G,
                        Compose<select::UctPn, Sim, backprop::Classic, select::SecureChild>,
                    >::new()
                    .config(config),
                ),
            )
        }
        "secure_child" => {
            let a =
                p.a.ok_or_else(|| HostError::bad_request("missing param: a"))?;
            let mut config = base_config::<
                G,
                Compose<select::UctPn, Sim, backprop::Classic, select::SecureChild>,
            >(p, seed, use_transpositions, budget)?
            .select(sel)
            .simulate(sim)
            .solver_loss_threshold(solver_loss_threshold)
            .contempt_factor(contempt_factor);
            config.final_action.a = a;
            Ok(
                Box::new(
                    TreeSearch::<
                        G,
                        Compose<select::UctPn, Sim, backprop::Classic, select::SecureChild>,
                    >::new()
                    .config(config),
                ),
            )
        }
        "robust_child" => {
            let config = base_config::<
                G,
                Compose<select::UctPn, Sim, backprop::Classic, select::RobustChild>,
            >(p, seed, use_transpositions, budget)?
            .select(sel)
            .simulate(sim)
            .solver_loss_threshold(solver_loss_threshold)
            .contempt_factor(contempt_factor);
            Ok(
                Box::new(
                    TreeSearch::<
                        G,
                        Compose<select::UctPn, Sim, backprop::Classic, select::RobustChild>,
                    >::new()
                    .config(config),
                ),
            )
        }
        other => Err(HostError::bad_request(format!(
            "unknown final_action: {other}"
        ))),
    }
}

/// Builds a candidate for a family whose `strategy.rs` counterpart fixes its
/// own `final_action` (`ucb1_max_robust`, `meta_mcts`) -- `p.final_action` is
/// ignored (the search-space YAML never activates it for these families, per
/// `strategy_tuner_info`'s conditions).
fn build_fixed<G, Sel, Sim, FA>(
    sel: Sel,
    sim: Sim,
    fa: FA,
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<Box<dyn Search<G = G>>, HostError>
where
    G: Game + 'static,
    Sel: SelectStrategy<G> + 'static,
    Sim: SimulateStrategy<G> + 'static,
    FA: SelectStrategy<G> + 'static,
{
    let config = base_config::<G, Compose<Sel, Sim, backprop::Classic, FA>>(
        p,
        seed,
        use_transpositions,
        budget,
    )?
    .select(sel)
    .simulate(sim)
    .final_action(fa);
    Ok(Box::new(
        TreeSearch::<G, Compose<Sel, Sim, backprop::Classic, FA>>::new().config(config),
    ))
}

/// Builds a `Box<dyn Search<G>>` from a raw params JSON object, the same
/// deserialize-then-dispatch path `strategy_tune_eval` uses for the
/// candidate side -- exposed so a caller can also build an *opponent* from
/// an arbitrary discovered config, not just a named preset. See
/// `game_host::GameAdapter::tune_eval`'s `baseline_config` parameter.
pub fn build_search<G: Game + 'static>(
    params: &Value,
    seed: u64,
    use_transpositions: bool,
) -> Result<Box<dyn Search<G = G>>, HostError> {
    let trial: TrialParams = serde_json::from_value(params.clone())
        .map_err(|e| HostError::bad_request(format!("invalid tuning params: {e}")))?;
    // Every caller of `build_search` builds an *opponent* -- a
    // `baseline_config`-backed baseline, or a `--baseline-config` for the
    // ladder driver's own self-play rungs -- never the candidate under
    // tune. That side of the match is already symmetric with the candidate
    // (both go through this exact function), so there's nothing to match a
    // budget to; the default (`MAX_ITER` iterations, single-threaded,
    // uncapped time) is always correct here.
    make_candidate(&trial, seed, use_transpositions, &SearchBudget::default())
}

fn make_candidate<G: Game + 'static>(
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<Box<dyn Search<G = G>>, HostError> {
    let missing = |field: &str| HostError::bad_request(format!("missing param: {field}"));
    let c = || p.c.ok_or_else(|| missing("c"));
    let epsilon = || p.epsilon.ok_or_else(|| missing("epsilon"));
    let c_pn = || p.c_pn.ok_or_else(|| missing("c_pn"));
    let solver_loss_threshold = || {
        p.solver_loss_threshold
            .ok_or_else(|| missing("solver_loss_threshold"))
    };
    // `contempt` gates `contempt_factor` the same way `rave_ucb`'s
    // "none"/"ucb1"/"tuned" gates `c` -- an explicit "off" choice rather
    // than treating the field's mere absence as off, so a trial that forgot
    // the gate entirely is rejected the same way a missing required field
    // always is here, not silently treated as "off".
    let contempt_factor = || match p.contempt.as_deref() {
        Some("off") => Ok(None),
        Some("on") => Ok(Some(
            p.contempt_factor
                .ok_or_else(|| missing("contempt_factor"))?,
        )),
        Some(other) => Err(HostError::bad_request(format!("unknown contempt: {other}"))),
        None => Err(missing("contempt")),
    };

    match p.family.as_str() {
        // Baseline-only floor families -- deliberately *not* in
        // `strategy_tuner_info`'s searchable `family` choices (a candidate
        // sampled as `random`/`flat_mc` would just hover around a ~0.5 cost
        // forever, wasting SMAC3's trial budget). Reachable only via
        // `build_search`/`--baseline-config`, e.g. as a ladder's floor rung.
        // Neither reads `q_init` or any other `TrialParams` field beyond
        // `family` itself.
        "random" => Ok(Box::new(mcts::strategies::random::Random::<G>::new())),
        "flat_mc" => Ok(Box::new(
            mcts::strategies::flat_mc::FlatMonteCarloStrategy::<G>::new(),
        )),
        "ucb1" => build_with_final_action(
            select::Ucb1::with_c(c()?),
            simulate::Uniform,
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "ucb1_dm" => build_with_final_action(
            select::Ucb1::with_c(c()?),
            simulate::DecisiveMove::<G>::new(),
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "ucb1_mast" => build_with_final_action(
            select::Ucb1::with_c(c()?),
            simulate::EpsilonGreedy::<G, simulate::Mast>::with_epsilon(epsilon()?),
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "ucb1_nst" => {
            let nst = simulate::Nst::new().backoff_threshold(
                p.nst_backoff_threshold
                    .ok_or_else(|| missing("nst_backoff_threshold"))?,
            );
            build_with_final_action(
                select::Ucb1::with_c(c()?),
                simulate::EpsilonGreedy::<G, simulate::Nst>::with_epsilon(epsilon()?).inner(nst),
                p,
                seed,
                use_transpositions,
                budget,
            )
        }
        "ucb1_progressive_history" => build_with_final_action(
            select::ProgressiveHistory::new(
                select::Ucb1::with_c(c()?),
                p.ph_weight.ok_or_else(|| missing("ph_weight"))?,
            ),
            simulate::Uniform,
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "amaf" => build_with_final_action(
            select::Amaf::new()
                .alpha(p.amaf_alpha.ok_or_else(|| missing("amaf_alpha"))?)
                .exploration_constant(c()?),
            simulate::Uniform,
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "amaf_mast" => build_with_final_action(
            select::Amaf::new()
                .alpha(p.amaf_alpha.ok_or_else(|| missing("amaf_alpha"))?)
                .exploration_constant(c()?),
            simulate::EpsilonGreedy::<G, simulate::Mast>::with_epsilon(epsilon()?),
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "ucb1_tuned" => build_with_final_action(
            select::Ucb1Tuned::with_c(c()?),
            simulate::Uniform,
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "ucb1_tuned_mast" => build_with_final_action(
            select::Ucb1Tuned::with_c(c()?),
            simulate::Mast,
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "ucb1_tuned_dm" => build_with_final_action(
            select::Ucb1Tuned::with_c(c()?),
            simulate::DecisiveMove::<G>::new(),
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "ucb1_tuned_dm_mast" => build_with_final_action(
            select::Ucb1Tuned::with_c(c()?),
            simulate::DecisiveMove::<G, simulate::EpsilonGreedy<G, simulate::Mast>>::new().inner(
                simulate::EpsilonGreedy::<G, simulate::Mast>::with_epsilon(epsilon()?),
            ),
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "rave" => {
            let schedule = match p.schedule.as_deref().ok_or_else(|| missing("schedule"))? {
                "hand_selected" => RaveSchedule::HandSelected {
                    k: p.k.ok_or_else(|| missing("k"))?,
                },
                "min_mse" => RaveSchedule::MinMSE {
                    bias: p.bias.ok_or_else(|| missing("bias"))?,
                },
                "threshold" => RaveSchedule::Threshold {
                    rave: p.rave.ok_or_else(|| missing("rave"))?,
                },
                other => return Err(HostError::bad_request(format!("unknown schedule: {other}"))),
            };
            let ucb = match p.rave_ucb.as_deref().ok_or_else(|| missing("rave_ucb"))? {
                "none" => RaveUcb::None,
                "ucb1" => RaveUcb::Ucb1 {
                    exploration_constant: c()?,
                },
                "tuned" => RaveUcb::Ucb1Tuned {
                    exploration_constant: c()?,
                },
                other => return Err(HostError::bad_request(format!("unknown rave_ucb: {other}"))),
            };
            build_with_final_action(
                select::Rave::new(
                    p.threshold.ok_or_else(|| missing("threshold"))?,
                    schedule,
                    ucb,
                ),
                simulate::DecisiveMove::<G, simulate::EpsilonGreedy<G, simulate::Mast>>::new()
                    .mode(simulate::DecisiveMoveMode::WinLoss)
                    .inner(simulate::EpsilonGreedy::<G, simulate::Mast>::with_epsilon(
                        epsilon()?,
                    )),
                p,
                seed,
                use_transpositions,
                budget,
            )
        }
        "ucb1_pn" => build_pn_with_final_action(
            c()?,
            c_pn()?,
            simulate::Uniform,
            p,
            seed,
            use_transpositions,
            budget,
            PnSolverParams {
                solver_loss_threshold: solver_loss_threshold()?,
                contempt_factor: contempt_factor()?,
            },
        ),
        "ucb1_pn_mast" => build_pn_with_final_action(
            c()?,
            c_pn()?,
            simulate::EpsilonGreedy::<G, simulate::Mast>::with_epsilon(epsilon()?),
            p,
            seed,
            use_transpositions,
            budget,
            PnSolverParams {
                solver_loss_threshold: solver_loss_threshold()?,
                contempt_factor: contempt_factor()?,
            },
        ),
        "ucb1_max_robust" => build_fixed(
            select::Ucb1::with_c(c()?),
            simulate::Uniform,
            select::MaxRobustChild,
            p,
            seed,
            use_transpositions,
            budget,
        ),
        "meta_mcts" => {
            // `simulate::MetaMcts`'s inner search has no default iteration
            // cap of its own (`TreeSearch::default()`'s `max_iterations` is
            // `usize::MAX`, meant to be paired with a time budget this
            // harness doesn't set) -- every simulate step would otherwise
            // run an effectively unbounded nested search. Cap it explicitly
            // instead of relying on `Default`.
            let inner = TreeSearch::<G, strategy::Ucb1>::new().config(
                SearchConfig::<G, strategy::Ucb1>::new().max_iterations(META_MCTS_INNER_ITERATIONS),
            );
            build_fixed(
                select::Ucb1::with_c(c()?),
                simulate::MetaMcts { inner },
                select::MaxAvgScore,
                p,
                seed,
                use_transpositions,
                budget,
            )
        }
        other => Err(HostError::bad_request(format!("unknown family: {other}"))),
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
pub fn strategy_tune_eval<G: Game + 'static>(
    params: &Value,
    rounds: u32,
    seed: Option<u64>,
    use_transpositions: bool,
    candidate_budget: SearchBudget,
    baseline_build: impl Fn() -> Box<dyn Search<G = G>>,
    initial_state: G::S,
) -> Result<TuneEvalOutcome, HostError> {
    let trial: TrialParams = serde_json::from_value(params.clone())
        .map_err(|e| HostError::bad_request(format!("invalid tuning params: {e}")))?;
    let seed = seed.unwrap_or(0);

    let (mut wins, mut losses, mut draws) = (0u32, 0u32, 0u32);
    for _ in 0..rounds {
        let mut candidate = make_candidate(&trial, seed, use_transpositions, &candidate_budget)?;
        let mut baseline = baseline_build();

        let (c, b, d) = play_one(candidate.as_mut(), baseline.as_mut(), initial_state.clone());
        wins += c;
        losses += b;
        draws += d;

        // Swap move order so the candidate plays second half the time.
        let mut candidate = make_candidate(&trial, seed, use_transpositions, &candidate_budget)?;
        let mut baseline = baseline_build();
        let (b, c, d) = play_one(baseline.as_mut(), candidate.as_mut(), initial_state.clone());
        wins += c;
        losses += b;
        draws += d;
    }

    Ok(TuneEvalOutcome {
        cost: cost_from_losses(losses, rounds),
        wins,
        losses,
        draws,
    })
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
fn play_one<G: Game>(
    first: &mut dyn Search<G = G>,
    second: &mut dyn Search<G = G>,
    initial_state: G::S,
) -> (u32, u32, u32) {
    let mut state = initial_state;
    while !G::is_terminal(&state) {
        let mover = G::player_to_move(&state).to_index();
        let action = if mover == 0 {
            first.choose_action(&state)
        } else {
            second.choose_action(&state)
        };
        state = G::apply(state, &action);
    }
    match G::winner(&state) {
        None => (0, 0, 1),
        Some(p) if p.to_index() == 0 => (1, 0, 0),
        Some(_) => (0, 1, 0),
    }
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
    TunerInfo {
        id: "strategy".into(),
        baselines: baselines.iter().map(|s| s.to_string()).collect(),
        game_config: json!({}),
        eval_rounds,
        parameters: vec![
            param(
                "family",
                json!({"type": "categorical", "choices": [
                    "ucb1", "ucb1_dm", "ucb1_mast", "ucb1_nst",
                    "ucb1_progressive_history", "ucb1_max_robust",
                    "amaf", "amaf_mast",
                    "ucb1_tuned", "ucb1_tuned_mast", "ucb1_tuned_dm", "ucb1_tuned_dm_mast",
                    "meta_mcts", "rave", "ucb1_pn", "ucb1_pn_mast",
                ], "default": "rave"}),
            ),
            param(
                "q_init",
                json!({"type": "categorical", "choices": ["Draw", "Infinity", "Loss", "Parent", "Win"], "default": "Infinity"}),
            ),
            param(
                "final_action",
                json!({"type": "categorical", "choices": ["max_avg", "secure_child", "robust_child"], "default": "robust_child"}),
            ),
            param(
                "a",
                json!({"type": "float", "bounds": [0, 10], "default": 4.0}),
            ),
            param(
                "c",
                json!({"type": "float", "bounds": [0, 3], "default": std::f64::consts::SQRT_2}),
            ),
            param(
                "epsilon",
                json!({"type": "float", "bounds": [0, 1], "default": 0.1}),
            ),
            param(
                "amaf_alpha",
                json!({"type": "float", "bounds": [0, 1], "default": 1.0}),
            ),
            param(
                "ph_weight",
                json!({"type": "float", "bounds": [0, 5], "default": 1.0}),
            ),
            param(
                "nst_backoff_threshold",
                json!({"type": "int", "bounds": [0, 100], "default": 5}),
            ),
            param(
                "bias",
                json!({"type": "float", "bounds": [0, 10], "default": 0.00001}),
            ),
            param(
                "k",
                json!({"type": "int", "bounds": [0, 2000], "default": 1000}),
            ),
            param(
                "rave",
                json!({"type": "int", "bounds": [0, 2000], "default": 700}),
            ),
            param(
                "schedule",
                json!({"type": "categorical", "choices": ["hand_selected", "min_mse", "threshold"], "default": "threshold"}),
            ),
            param(
                "threshold",
                json!({"type": "int", "bounds": [0, 2000], "default": 700}),
            ),
            param(
                "rave_ucb",
                json!({"type": "categorical", "choices": ["none", "ucb1", "tuned"], "default": "tuned"}),
            ),
            param(
                // Kowalski et al. 2023 Eq. 4: clustered 1.0-2.0 in the
                // paper's own experiments, domain-dependent.
                "c_pn",
                json!({"type": "float", "bounds": [0, 3], "default": 1.0}),
            ),
            param(
                // MCTS-Solver's proven-loss selection threshold `T`
                // (Kowalski et al. 2023 Section III.B); the paper uses
                // T=5 throughout.
                "solver_loss_threshold",
                json!({"type": "int", "bounds": [0, 50], "default": 5}),
            ),
            param(
                "contempt",
                json!({"type": "categorical", "choices": ["off", "on"], "default": "off"}),
            ),
            param(
                // Compared against `Node::expected_score`, whose default
                // range (`Game::compute_utilities`'s default) is [-1, 1].
                "contempt_factor",
                json!({"type": "float", "bounds": [-1, 1], "default": 0.0}),
            ),
        ],
        conditions: vec![
            condition(json!({"family": FINAL_ACTION_FAMILIES}), &["final_action"]),
            condition(json!({"final_action": "secure_child"}), &["a"]),
            condition(json!({"family": C_FAMILIES}), &["c"]),
            condition(json!({"family": EPSILON_FAMILIES}), &["epsilon"]),
            condition(json!({"family": ["amaf", "amaf_mast"]}), &["amaf_alpha"]),
            condition(
                json!({"family": "ucb1_progressive_history"}),
                &["ph_weight"],
            ),
            condition(json!({"family": "ucb1_nst"}), &["nst_backoff_threshold"]),
            condition(
                json!({"family": "rave"}),
                &["threshold", "schedule", "rave_ucb"],
            ),
            condition(json!({"schedule": "hand_selected"}), &["k"]),
            condition(json!({"schedule": "min_mse"}), &["bias"]),
            condition(json!({"schedule": "threshold"}), &["rave"]),
            condition(json!({"rave_ucb": ["ucb1", "tuned"]}), &["c"]),
            condition(
                json!({"family": PN_FAMILIES}),
                &["c_pn", "solver_loss_threshold", "contempt"],
            ),
            condition(json!({"contempt": "on"}), &["contempt_factor"]),
        ],
    }
}

fn param(name: &str, spec: Value) -> TunerParameter {
    TunerParameter {
        name: name.into(),
        spec,
    }
}

fn condition(if_: Value, then: &[&str]) -> TunerCondition {
    TunerCondition {
        if_,
        then: then.iter().map(|s| s.to_string()).collect(),
    }
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
        )
        .expect_err("missing `rave` must be rejected");
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
                build_search::<Nim>(&baseline_params, 0, false)
                    .expect("baseline_params is a valid ucb1 config")
            },
            <Nim as Game>::S::default(),
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
        build_search::<Nim>(&json!({"family": "random", "q_init": "Infinity"}), 0, false)
            .expect("random should build with just family/q_init");
    }

    #[test]
    fn test_build_search_builds_flat_mc_floor_family() {
        build_search::<Nim>(
            &json!({"family": "flat_mc", "q_init": "Infinity"}),
            0,
            false,
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
        let err = match build_search::<Nim>(&params, 0, false) {
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
