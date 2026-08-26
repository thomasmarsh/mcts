use std::str::FromStr;

use game_host::{
    Analysis, AnalysisAction, HostError, SearchActionReport, SearchGraphMode, SearchReport,
    SearchReportReason, SearchReportStatus, SearchTermination, SearchWarning,
};
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, GraphSearch, GraphStats, TranspositionKeying};
use mcts::strategies::{
    Search, SearchGraphMode as EngineSearchGraphMode, SearchReport as EngineSearchReport,
    SearchReportReason as EngineSearchReportReason, SearchReportStatus as EngineSearchReportStatus,
    SearchTermination as EngineSearchTermination, SearchWarning as EngineSearchWarning,
};
use serde_json::Value;

use crate::{
    config_ir,
    direct_search::build_direct,
    family_catalog::{dispatch_family, ComposeSpec, FamilySpec, TrialParams},
};

pub(crate) const PLAYOUT_DEPTH: usize = 200;
pub(crate) const MAX_ITER: usize = 10_000;
pub(crate) const EXPAND_THRESHOLD: u32 = 1;

/// Chooses an action and converts the evidence retained for that exact call
/// into the canonical game-host wire format. The caller supplies the same
/// move encoder it uses at its protocol boundary, keeping game-specific wire
/// shapes (such as Tak PTN) out of the generic search layer.
pub fn choose_action_with_report<G, F>(
    search: &mut dyn Search<G = G>,
    state: &G::S,
    encode_action: F,
) -> (G::A, SearchReport)
where
    G: Game,
    F: Fn(&G::A) -> serde_json::Value,
{
    let selected_action = search.choose_action(state);
    let report = wire_search_report(
        search.search_report(state, &selected_action),
        &encode_action,
    );
    (selected_action, report)
}

/// Projects the existing root summary into the legacy analysis fields while
/// retaining the separately versioned final report. The selected action is
/// authoritative: a principal variation may be absent or partial, but the
/// move that was actually chosen is always known.
pub fn legacy_analysis_with_report<G, F>(
    search: &dyn Search<G = G>,
    state: &G::S,
    selected_action: &G::A,
    report: SearchReport,
    encode_action: F,
) -> Analysis
where
    G: Game,
    F: Fn(&G::A) -> serde_json::Value,
{
    let root = search.root_report(state);
    let actions: Vec<AnalysisAction> = root
        .actions
        .into_iter()
        .map(|action| AnalysisAction {
            action: encode_action(&action.action),
            visits: action.visits,
            mean_value: action.mean_value,
            is_proven: action.is_proven,
        })
        .collect();

    if !matches!(report.status, SearchReportStatus::Unavailable) {
        for retained in &report.actions {
            assert!(
                actions.iter().any(|legacy| {
                    legacy.action == retained.action
                        && legacy.visits == retained.visits
                        && legacy.mean_value == retained.mean_value
                        && legacy.is_proven == retained.is_proven
                }),
                "final report action must agree with legacy root summary"
            );
        }
    }

    Analysis {
        actions,
        principal_variation: root
            .principal_variation
            .iter()
            .map(&encode_action)
            .collect(),
        total_visits: root.total_visits,
        suggested_move: Some(encode_action(selected_action)),
        search: Some(report),
    }
}

fn wire_search_report<A>(
    report: EngineSearchReport<A>,
    encode_action: &impl Fn(&A) -> serde_json::Value,
) -> SearchReport {
    SearchReport {
        schema_version: report.schema_version,
        status: match report.status {
            EngineSearchReportStatus::Available => SearchReportStatus::Available,
            EngineSearchReportStatus::Partial => SearchReportStatus::Partial,
            EngineSearchReportStatus::Unavailable => SearchReportStatus::Unavailable,
        },
        reason: report.reason.map(|reason| match reason {
            EngineSearchReportReason::StrategyUnsupported => {
                SearchReportReason::StrategyUnsupported
            }
            EngineSearchReportReason::SearchNotRun => SearchReportReason::SearchNotRun,
            EngineSearchReportReason::RootParallelPvSingleTree => {
                SearchReportReason::RootParallelPvSingleTree
            }
        }),
        elapsed_seconds: report.elapsed_seconds,
        iteration_limit: report.iteration_limit,
        time_limit_seconds: report.time_limit_seconds,
        completed_iterations: report.completed_iterations,
        termination: report.termination.map(|termination| match termination {
            EngineSearchTermination::Iterations => SearchTermination::Iterations,
            EngineSearchTermination::Time => SearchTermination::Time,
            EngineSearchTermination::Solved => SearchTermination::Solved,
            EngineSearchTermination::Unknown => SearchTermination::Unknown,
        }),
        selected_action: report.selected_action.as_ref().map(encode_action),
        actions: report
            .actions
            .into_iter()
            .map(|action| SearchActionReport {
                action: encode_action(&action.action),
                visits: action.visits,
                share: action.share,
                mean_value: action.mean_value,
                is_proven: action.is_proven,
            })
            .collect(),
        principal_variation: report
            .principal_variation
            .iter()
            .map(encode_action)
            .collect(),
        root_visits: report.root_visits,
        tree_nodes: report.tree_nodes,
        mean_depth: report.mean_depth,
        max_depth: report.max_depth,
        graph_mode: report.graph_mode.map(|mode| match mode {
            EngineSearchGraphMode::Tree => SearchGraphMode::Tree,
            EngineSearchGraphMode::Transpositions => SearchGraphMode::Transpositions,
            EngineSearchGraphMode::DagEdges => SearchGraphMode::DagEdges,
            EngineSearchGraphMode::DagNodes => SearchGraphMode::DagNodes,
            EngineSearchGraphMode::DagBoth => SearchGraphMode::DagBoth,
        }),
        tt_reads: report.tt_reads,
        tt_writes: report.tt_writes,
        tt_hits: report.tt_hits,
        tt_hit_ratio: report.tt_hit_ratio,
        iterations_per_second: report.iterations_per_second,
        warnings: report
            .warnings
            .into_iter()
            .map(|warning| match warning {
                EngineSearchWarning::ActionsTruncated => SearchWarning::ActionsTruncated,
                EngineSearchWarning::PrincipalVariationTruncated => {
                    SearchWarning::PrincipalVariationTruncated
                }
                EngineSearchWarning::StructuralDiagnosticsOmitted => {
                    SearchWarning::StructuralDiagnosticsOmitted
                }
                EngineSearchWarning::RootParallelPvSingleTree => {
                    SearchWarning::RootParallelPvSingleTree
                }
            })
            .collect(),
    }
}

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
/// discovered config, including a `random`/`flat_mc` baseline), since both
/// sides of that match are built the same way and so stay symmetric
/// regardless of budget.
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

/// Converts one trial's `TrialParams` and its already-dispatched
/// `ComposeSpec` into `config_ir`'s `SearchSpec`/`SearchSettings` -- the
/// part of candidate construction common to every `FamilySpec::Compose`
/// family (`q_init`, `mcgs`, the fixed `SearchSettings` knobs), factored out
/// of `to_search_spec` so `make_candidate` can call it directly for a
/// `Compose` family without re-dispatching `p.family`.
fn compose_settings(
    cs: ComposeSpec,
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<(config_ir::SearchSpec, config_ir::SearchSettings), HostError> {
    let q_init_str = p
        .q_init
        .as_deref()
        .ok_or_else(|| HostError::bad_request("missing param: q_init".to_string()))?;
    let q_init = QInit::from_str(q_init_str)
        .map_err(|_| HostError::bad_request(format!("invalid q_init: {q_init_str}")))?;
    let mcgs = p.mcgs.unwrap_or(false);
    let state_only_keying = p.state_only_keying.unwrap_or(false);
    let (use_transpositions, reuse_tree, graph_search, transposition_keying) =
        resolve_graph_search(mcgs, use_transpositions, state_only_keying)?;

    let ComposeSpec {
        select,
        simulate,
        final_action,
        backprop,
        solver_loss_threshold: solver_loss_threshold_setting,
        contempt_factor: contempt_factor_setting,
    } = cs;

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
        transposition_keying,
        solver_loss_threshold: solver_loss_threshold_setting,
        contempt_factor: contempt_factor_setting,
    };
    Ok((spec, settings))
}

/// Converts one trial's `TrialParams` into `config_ir`'s `SearchSpec`/
/// `SearchSettings`, valid only for a family that resolves to
/// `FamilySpec::Compose` (a `Direct` family, e.g. `"random"`, has no such
/// representation; calling this with one returns the `Err` below). Per-family
/// construction is `family_catalog::dispatch_family`'s `register_family!`
/// table; `compose_settings` handles what's common to every `Compose` family.
/// `make_candidate` builds its own `Box<dyn Search<G>>` straight from
/// `compose_settings`/`build_direct` rather than through this function, so
/// this is exercised only by `tests.rs`'s direct `SearchSpec`/
/// `SearchSettings`-level assertions.
#[cfg(test)]
pub(crate) fn to_search_spec(
    p: &TrialParams,
    seed: u64,
    use_transpositions: bool,
    budget: &SearchBudget,
) -> Result<(config_ir::SearchSpec, config_ir::SearchSettings), HostError> {
    match dispatch_family(&p.family, p)? {
        FamilySpec::Compose(cs) => compose_settings(cs, p, seed, use_transpositions, budget),
        FamilySpec::Direct(_) => Err(HostError::bad_request(format!(
            "family {:?} has no config_ir::SearchSpec representation",
            p.family
        ))),
    }
}

/// Derives `SearchSettings`'s `use_transpositions`/`reuse_tree`/
/// `graph_search`/`transposition_keying` from a requested `mcgs` flag and
/// whether the game supports transpositions at all -- the one place "`mcgs`
/// implies `Dag(Both)`, turns off the plain transposition table and tree
/// reuse, and requires a real zobrist hash" is decided. Both
/// [`to_search_spec`] and [`presets::build_custom`] call this rather than
/// each re-deriving the same fields from `mcgs`, so that mapping can't drift
/// into two different answers as either caller changes independently -- see
/// this repo's `AGENTS.md` on why config axes like this one need to be
/// correct by construction rather than duplicated by convention.
///
/// `state_only_keying` selects `TranspositionKeying::StateOnly` over the
/// default `PerPly` and is rejected unless `mcgs` is also `true` -- it's
/// meaningless without graph search on, and enabling it asserts the game's
/// zobrist hash meets `TranspositionKeying::StateOnly`'s stricter GHI
/// precondition (see that type's doc comment), which is a per-game claim
/// only the caller (a specific game's preset/tuner wiring) can make, not
/// something this shared helper can verify.
pub(crate) fn resolve_graph_search(
    mcgs: bool,
    use_transpositions: bool,
    state_only_keying: bool,
) -> Result<(bool, bool, Option<GraphSearch>, TranspositionKeying), HostError> {
    if mcgs && !use_transpositions {
        return Err(HostError::bad_request(
            "mcgs requires a game with a zobrist hash",
        ));
    }
    if state_only_keying && !mcgs {
        return Err(HostError::bad_request("state_only_keying requires mcgs"));
    }
    Ok((
        use_transpositions && !mcgs,
        !mcgs,
        mcgs.then_some(GraphSearch::Dag(GraphStats::Both)),
        if state_only_keying {
            TranspositionKeying::StateOnly
        } else {
            TranspositionKeying::PerPly
        },
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
    match dispatch_family(&p.family, p)? {
        FamilySpec::Direct(direct) => Ok(build_direct::<G>(&direct, budget)),
        FamilySpec::Compose(cs) => {
            let (spec, settings) = compose_settings(cs, p, seed, use_transpositions, budget)?;
            config_ir::validate_search_spec::<G>(&spec).map_err(HostError::bad_request)?;
            Ok(config_ir::build_search(&spec, &settings))
        }
    }
}
