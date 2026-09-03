use crate::game::Game;
use crate::game::PlayerIndex;
use crate::algorithms::mcts::config::GraphSearch;
use crate::algorithms::mcts::config::GraphStats;
use crate::algorithms::mcts::node::NodeStats;
use crate::algorithms::mcts::node::Proven;
use crate::algorithms::mcts::search::shared::TreeIndex;
use crate::algorithms::mcts::search::SearchRun;
use crate::algorithms::mcts::search::TreeStats;
use crate::algorithms::{
    SearchActionReport, SearchGraphMode, SearchReport, SearchReportReason, SearchReportStatus,
    SearchWarning,
};

use std::sync::atomic::Ordering::Relaxed;

const ACTION_REPORT_LIMIT: usize = 1_024;
const PRINCIPAL_VARIATION_LIMIT: usize = 256;

fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// Build a [`SearchReport`] from the per-call evidence collected by a
/// `TreeSearch`.  Monomorphized only over `G` (not `(G, S)`), so the heavy
/// collection/sorting/PV logic compiles once per game type rather than once
/// per `(select, backprop)` strategy pair.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_search_report<G: Game>(
    run: &SearchRun<G::A>,
    index: &TreeIndex<G::A>,
    root_id: crate::algorithms::mcts::index::Id,
    pv: &[G::A],
    root_stats: &NodeStats,
    stats: &TreeStats<G>,
    graph_search: GraphSearch,
    use_transpositions: bool,
    max_iterations: usize,
    max_time: std::time::Duration,
    state: &G::S,
    selected_action: &G::A,
) -> SearchReport<G::A> {
    let graph_stats = graph_stats_from(graph_search, use_transpositions);

    let player = G::player_to_move(state).to_index();
    let root_parallel = run.root_parallel.as_ref();
    let mut actions: Vec<_> = if let Some(root_parallel) = root_parallel {
        root_parallel
            .actions
            .iter()
            .enumerate()
            .map(|(i, (action, visits, scores))| {
                (
                    i,
                    action.clone(),
                    *visits,
                    finite(scores[player] / *visits as f64),
                    false,
                )
            })
            .collect()
    } else {
        let root = index.get(root_id);
        let children = root.children();
        (0..children.len())
            .filter(|&i| children.is_explored(i))
            .map(|i| {
                let child_id = children.node_id(i).unwrap();
                let is_proven = index.get(child_id).proven() != Proven::Unproven;
                let snap = if matches!(graph_stats, Some(GraphStats::Nodes)) {
                    index.get(child_id).stats.snapshot(player)
                } else {
                    children.snapshot(i, player)
                };
                (
                    i,
                    children.action(i).clone(),
                    snap.num_visits,
                    finite(snap.expected_score()),
                    is_proven,
                )
            })
            .collect()
    };
    let all_action_visits: u64 = actions
        .iter()
        .map(|(_, _, visits, _, _)| *visits as u64)
        .sum();
    let selected_row = actions
        .iter()
        .find(|(_, action, _, _, _)| action == selected_action)
        .cloned();
    actions.sort_by_key(|action| (std::cmp::Reverse(action.2), action.0));
    let actions_truncated = actions.len() > ACTION_REPORT_LIMIT;
    actions.truncate(ACTION_REPORT_LIMIT);
    if actions_truncated
        && !actions
            .iter()
            .any(|(_, action, _, _, _)| action == selected_action)
    {
        if let Some(selected) = selected_row {
            let _ = actions.pop();
            actions.push(selected);
            // Restore root-order tie breaking after retaining a selected
            // row that was outside the report cap.
            actions.sort_by_key(|action| (std::cmp::Reverse(action.2), action.0));
        }
    }
    let actions = actions
        .into_iter()
        .map(
            |(_, action, visits, mean_value, is_proven)| SearchActionReport {
                action,
                visits,
                share: finite(if all_action_visits == 0 {
                    0.0
                } else {
                    visits as f64 / all_action_visits as f64
                }),
                mean_value: mean_value.clamp(-1.0, 1.0),
                is_proven,
            },
        )
        .collect();

    let pv = root_parallel.map_or(pv, |report| &report.principal_variation);
    let pv_truncated = pv.len() > PRINCIPAL_VARIATION_LIMIT;
    let mut principal_variation = pv.to_vec();
    principal_variation.truncate(PRINCIPAL_VARIATION_LIMIT);
    let completed_iterations = root_parallel.map_or_else(
        || stats.iter_count.load(Relaxed),
        |report| report.completed_iterations,
    );
    let mean_depth = (completed_iterations > 0).then(|| {
        finite(
            root_parallel.map_or_else(
                || stats.accum_depth.load(Relaxed),
                |report| report.accum_depth,
            ) as f64
                / completed_iterations as f64,
        )
    });
    let tt_hit_ratio = (run.tt_reads > 0).then(|| finite(run.tt_hits as f64 / run.tt_reads as f64));
    let iterations_per_second = (run.elapsed_seconds > 0.0)
        .then(|| completed_iterations as f64 / run.elapsed_seconds)
        .filter(|rate| rate.is_finite());
    let graph_mode = match graph_search {
        GraphSearch::Tree if use_transpositions => SearchGraphMode::Transpositions,
        GraphSearch::Tree => SearchGraphMode::Tree,
        GraphSearch::Dag(GraphStats::Edges) => SearchGraphMode::DagEdges,
        GraphSearch::Dag(GraphStats::Nodes) => SearchGraphMode::DagNodes,
        GraphSearch::Dag(GraphStats::Both) => SearchGraphMode::DagBoth,
    };
    let mut warnings = vec![SearchWarning::StructuralDiagnosticsOmitted];
    if root_parallel.is_some() {
        warnings.push(SearchWarning::RootParallelPvSingleTree);
    }
    if actions_truncated {
        warnings.push(SearchWarning::ActionsTruncated);
    }
    if pv_truncated {
        warnings.push(SearchWarning::PrincipalVariationTruncated);
    }
    SearchReport {
        schema_version: 1,
        status: if root_parallel.is_some() {
            SearchReportStatus::Partial
        } else {
            SearchReportStatus::Available
        },
        reason: run
            .root_parallel
            .is_some()
            .then_some(SearchReportReason::RootParallelPvSingleTree),
        elapsed_seconds: Some(run.elapsed_seconds),
        iteration_limit: (max_iterations != usize::MAX).then_some(max_iterations),
        time_limit_seconds: (max_time != std::time::Duration::default())
            .then(|| finite(max_time.as_secs_f64())),
        completed_iterations,
        termination: Some(run.termination),
        selected_action: Some(selected_action.clone()),
        actions,
        principal_variation,
        root_visits: root_parallel.map_or_else(
            || {
                if graph_stats.is_some_and(GraphStats::uses_nodes) {
                    index.get(root_id).stats.num_visits()
                } else {
                    root_stats.num_visits()
                }
            },
            |report| {
                report
                    .actions
                    .iter()
                    .fold(0_u32, |total, (_, visits, _)| total.saturating_add(*visits))
            },
        ),
        tree_nodes: root_parallel.map_or_else(|| index.len(), |report| report.tree_nodes),
        mean_depth,
        max_depth: (completed_iterations > 0).then(|| {
            root_parallel.map_or_else(|| stats.max_depth.load(Relaxed), |report| report.max_depth)
        }),
        graph_mode: Some(graph_mode),
        tt_reads: run.tt_reads,
        tt_writes: run.tt_writes,
        tt_hits: run.tt_hits,
        tt_hit_ratio,
        iterations_per_second,
        warnings,
    }
}

/// Equivalent to `SearchConfig::graph_stats()` without the `S` dependency:
/// the graph-stats variant in play, if any.
fn graph_stats_from(graph_search: GraphSearch, use_transpositions: bool) -> Option<GraphStats> {
    match graph_search {
        GraphSearch::Tree if use_transpositions => Some(GraphStats::Edges),
        GraphSearch::Tree => None,
        GraphSearch::Dag(stats) => Some(stats),
    }
}
