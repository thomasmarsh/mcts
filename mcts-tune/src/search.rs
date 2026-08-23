use std::str::FromStr;

use game_host::HostError;
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, GraphSearch, GraphStats};
use mcts::strategies::Search;
use serde_json::Value;

use crate::{
    config_ir,
    family_catalog::{dispatch_family, FamilySpec, TrialParams},
};

pub(crate) const PLAYOUT_DEPTH: usize = 200;
pub(crate) const MAX_ITER: usize = 10_000;
pub(crate) const EXPAND_THRESHOLD: u32 = 1;

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
pub(crate) const META_MCTS_INNER_ITERATIONS: usize = 50;

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
/// independent of which family/hyperparameters tuner samples. A game's own
/// `tune_eval` is responsible for building a `SearchBudget` that mirrors
/// whatever named preset it's dispatching to in that case (see
/// `games/druid/src/main.rs`'s `tune_eval`).
///
/// `max_iterations` is deliberately **not** part of `TrialParams` -- it's a
/// per-*run* compute budget an operator sets once at launch (`--override
/// target.max_iterations=N`, or the launch form's "Iteration budget"
/// field), not a per-*trial* hyperparameter tuner gets to search over
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
    pub(crate) fn iteration_limit(self) -> usize {
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
pub(crate) fn to_search_spec(
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<(config_ir::SearchSpec, config_ir::SearchSettings), HostError> {
    let q_init = QInit::from_str(&p.q_init)
        .map_err(|_| HostError::bad_request(format!("invalid q_init: {}", p.q_init)))?;
    let mcgs = p.mcgs.unwrap_or(false);
    let (use_transpositions, reuse_tree, graph_search) =
        resolve_graph_search(mcgs, use_transpositions)?;

    let FamilySpec {
        select,
        simulate,
        final_action,
        backprop,
        solver_loss_threshold: solver_loss_threshold_setting,
        contempt_factor: contempt_factor_setting,
    } = dispatch_family(&p.family, p)?;

    let spec = config_ir::SearchSpec {
        select,
        simulate,
        backprop,
        final_action,
    };
    let settings = config_ir::SearchSettings {
        max_iterations: budget.iteration_limit(),
        max_playout_depth: PLAYOUT_DEPTH,
        expand_threshold: EXPAND_THRESHOLD,
        q_init,
        use_transpositions,
        use_mcts_solver: true,
        reuse_tree,
        num_tree_threads: budget.threads,
        seed,
        max_time: budget.max_time,
        graph_search,
        solver_loss_threshold: solver_loss_threshold_setting,
        contempt_factor: contempt_factor_setting,
    };
    Ok((spec, settings))
}

/// Derives `SearchSettings`'s `use_transpositions`/`reuse_tree`/
/// `graph_search` from a requested `mcgs` flag and whether the game supports
/// transpositions at all -- the one place "`mcgs` implies `Dag(Both)`, turns
/// off the plain transposition table and tree reuse, and requires a real
/// zobrist hash" is decided. Both [`to_search_spec`] and
/// [`presets::build_custom`] call this rather than each re-deriving the same
/// three fields from `mcgs`, so that mapping can't drift into two different
/// answers as either caller changes independently -- see this repo's
/// `AGENTS.md` on why config axes like this one need to be correct by
/// construction rather than duplicated by convention.
pub(crate) fn resolve_graph_search(
    mcgs: bool,
    use_transpositions: bool,
) -> Result<(bool, bool, Option<GraphSearch>), HostError> {
    if mcgs && !use_transpositions {
        return Err(HostError::bad_request(
            "mcgs requires a game with a zobrist hash",
        ));
    }
    Ok((
        use_transpositions && !mcgs,
        !mcgs,
        mcgs.then_some(GraphSearch::Dag(GraphStats::Both)),
    ))
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

pub(crate) fn make_candidate<G: Game + 'static>(
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<Box<dyn Search<G = G>>, HostError> {
    match p.family.as_str() {
        // Baseline-only floor families -- deliberately *not* in
        // `strategy_tuner_info`'s searchable `family` choices (a candidate
        // sampled as `random`/`flat_mc` would just hover around a ~0.5 cost
        // forever, wasting the tuner's trial budget). Reachable only via
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
            config_ir::validate_search_spec::<G>(&spec).map_err(HostError::bad_request)?;
            Ok(config_ir::build_search(&spec, &settings))
        }
    }
}
