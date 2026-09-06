//! TreeStrap harvest: after one MCTS search, read the searched value at
//! every internal node of the just-completed tree and emit it as a training
//! target.
//!
//! This is the arm-C label source for the searched-value "signal bake-off":
//! arms A and B keep one target per search (the played root position, from
//! the game outcome or from the root's own searched value); arm C keeps the
//! searched value at *every* internal node the search touched. All three
//! arms consume the identical set of searches -- they differ only in what
//! they retain.
//!
//! ## What "searched value" means
//!
//! Post-search, an MCTS node's value estimate for the player to move at that
//! node is `ChildArray::expected_score(idx, player_index)`, read from the
//! parent's child array. It already lives in `[-1, 1]` (±1 per playout,
//! divided by the visit count) -- exactly the [`Record::target`] convention
//! (`+1` win, `-1` loss, side-to-move perspective), so it is written through
//! unscaled.
//!
//! ## Tree, not DAG
//!
//! The label search is a genuine tree (`GraphSearch::Tree`,
//! `use_transpositions = false`): no shared nodes, no cycles, so the walk is
//! a plain depth-first recursion with no visited-set. [`harvest_tree`]
//! `debug_assert!`s that assumption against the live config; a transposition
//! search would need `Id`-keyed dedup instead.
//!
//! Othello canonicalises positions under its D4 symmetry group only when the
//! search actually keys nodes by canonical hash (i.e. under transpositions);
//! with the tree-search label config it does not, so `real_action` collapses
//! to `children.action(idx)`. The walk still routes every action through
//! [`real_action`] with a freshly recomputed `incoming_sym`, matching
//! `search/reroot.rs`, so enabling canonicalisation later cannot silently
//! corrupt the harvested positions.

use mcts::algorithms::mcts::config::GraphSearch;
use mcts::algorithms::mcts::index::Id;
use mcts::algorithms::mcts::node::{real_action, NodeState};
use mcts::algorithms::mcts::search::TreeIndex;
use mcts::algorithms::mcts::{profile, select, simulate, IsmctsMode, SearchConfig, TreeSearch};
use mcts::game::{Game, PlayerIndex, Real};
use mcts::symmetry::incoming_sym;

use crate::dump::{record_for, Record};
use crate::{Othello, State};

/// The MCTS recipe used for every label search: the `strong` preset's
/// core (UCB1, uniform playout, `q_init = Loss`), matching the baseline of
/// `examples/ntuple_match.rs` so the bake-off's label searches and its gate
/// searches are the same engine.
pub type LabelProfile = profile::Mcts<select::Ucb1, simulate::Uniform>;

/// Concrete label-search type. Built directly rather than through
/// `PresetTable` because the harvest needs the concrete `TreeSearch` to walk
/// its arena afterward, which a `Box<dyn Search>` hides.
pub type LabelSearch = TreeSearch<Othello, LabelProfile>;

/// Which internal nodes of a searched tree are worth keeping as targets.
#[derive(Debug, Clone)]
pub struct HarvestFilter {
    /// Skip a node the search visited fewer than this many times -- its
    /// `expected_score` is too noisy to be a useful target.
    pub min_visits: u32,
    /// Keep at most this many nodes per search. When the tree has more, the
    /// highest-visit nodes win (their targets are the most trustworthy).
    pub max_per_search: usize,
    /// Optional depth cap (root is depth 0); `None` means no cap.
    pub max_depth: Option<u32>,
}

impl Default for HarvestFilter {
    fn default() -> Self {
        Self {
            min_visits: 20,
            max_per_search: 512,
            max_depth: None,
        }
    }
}

/// Whether this config keys nodes by canonical-symmetry hash. Public fields
/// only, so this can be read from outside the `mcts` crate (the crate-private
/// `SearchConfig::canonicalizes()` cannot).
fn canonicalizes<S>(cfg: &SearchConfig<Othello, S>) -> bool
where
    S: mcts::algorithms::mcts::PolicyProfile<Othello>,
{
    let uses_transpositions = match cfg.graph_search {
        GraphSearch::Dag(_) => true,
        GraphSearch::Tree => cfg.use_transpositions,
    };
    uses_transpositions && cfg.ismcts_mode == IsmctsMode::Off
}

/// Every internal node of `search`'s just-completed tree that passes
/// `filter`, each paired with its post-search visit count (highest first
/// once the `max_per_search` cap has been applied). The root itself is
/// included when it passes `min_visits`, so a whole-game harvest is a strict
/// superset of the played-root positions arm B keeps.
pub fn harvest_tree_scored(
    search: &LabelSearch,
    root_state: &State,
    filter: &HarvestFilter,
) -> Vec<(u32, Record)> {
    debug_assert!(
        !matches!(search.config.graph_search, GraphSearch::Dag(_)) && !search.config.use_transpositions,
        "harvest_tree assumes a genuine tree; a transposition search needs Id-keyed dedup"
    );
    let canon = canonicalizes(&search.config);

    let mut out: Vec<HarvestHit> = Vec::new();

    // The root, from the side-to-move-at-root perspective -- identical to
    // arm B's target for this search.
    let root_visits = search.root_stats.num_visits();
    if root_visits >= filter.min_visits {
        let pidx = Othello::player_to_move(root_state).to_index();
        let value = root_value(search, root_state, pidx);
        let mut rec = record_for(root_state, None);
        rec.target = value as f32;
        out.push(HarvestHit {
            visits: root_visits,
            rec,
        });
    }

    walk(
        &search.index,
        search.root_id,
        *root_state,
        true,
        0,
        canon,
        filter,
        &mut out,
    );

    if out.len() > filter.max_per_search {
        out.sort_by_key(|h| std::cmp::Reverse(h.visits));
        out.truncate(filter.max_per_search);
    }
    out.into_iter().map(|h| (h.visits, h.rec)).collect()
}

/// [`harvest_tree_scored`] without the visit counts.
pub fn harvest_tree(search: &LabelSearch, root_state: &State, filter: &HarvestFilter) -> Vec<Record> {
    harvest_tree_scored(search, root_state, filter)
        .into_iter()
        .map(|(_, r)| r)
        .collect()
}

/// The root's own searched value for `pidx`. Reads `root_stats` when it has
/// been accumulated; falls back to the best child's `expected_score` (the
/// root can carry zero direct visits when the search only ever scored its
/// children).
pub fn root_value(search: &LabelSearch, root_state: &State, pidx: usize) -> f64 {
    if search.root_stats.num_visits() > 0 {
        return search.root_stats.expected_score(pidx);
    }
    let node = search.index.get(search.root_id);
    let Some(NodeState::Expanded(children)) = node.status() else {
        return 0.0;
    };
    let mut best = (0u32, 0.0f64);
    for i in 0..children.len() {
        let v = children.num_visits(i);
        if v > best.0 {
            best = (v, children.expected_score(i, pidx));
        }
    }
    let _ = root_state;
    best.1
}

struct HarvestHit {
    visits: u32,
    rec: Record,
}

#[allow(clippy::too_many_arguments)]
fn walk(
    index: &TreeIndex<crate::Move>,
    id: Id,
    state: State,
    is_root: bool,
    depth: u32,
    canon: bool,
    filter: &HarvestFilter,
    out: &mut Vec<HarvestHit>,
) {
    if let Some(cap) = filter.max_depth {
        if depth >= cap {
            return;
        }
    }
    let node = index.get(id);
    let Some(NodeState::Expanded(children)) = node.status() else {
        return;
    };
    let incoming = incoming_sym::<Othello>(canon, is_root, Real(&state));
    for i in 0..children.len() {
        let Some(child_id) = children.node_id(i) else {
            continue;
        };
        let visits = children.num_visits(i);
        if visits < filter.min_visits {
            continue;
        }
        let action = real_action::<Othello>(children, i, incoming);
        let child_state = Othello::apply(state, &action);
        let pidx = Othello::player_to_move(&child_state).to_index();
        let value = children.expected_score(i, pidx) as f32;
        let mut rec = record_for(&child_state, None);
        rec.target = value;
        out.push(HarvestHit {
            visits,
            rec,
        });
        walk(index, child_id, child_state, false, depth + 1, canon, filter, out);
    }
}

/// Build a label search from the `strong` recipe at `label_iters` iterations.
pub fn label_search(label_iters: usize, seed: u64) -> LabelSearch {
    use mcts::algorithms::mcts::node::QInit;
    TreeSearch::<Othello, LabelProfile>::new().config(
        SearchConfig::new()
            .name("ntuple/label")
            .expand_threshold(1)
            .q_init(QInit::Loss)
            .max_iterations(label_iters)
            .seed(seed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::algorithms::Search;

    /// Independently count the internal nodes a full harvest should keep, by
    /// walking the arena directly with no shared code path.
    fn count_internal(
        index: &TreeIndex<crate::Move>,
        id: Id,
        state: State,
        min_visits: u32,
        canon: bool,
        is_root: bool,
    ) -> usize {
        let node = index.get(id);
        let Some(NodeState::Expanded(children)) = node.status() else {
            return 0;
        };
        let incoming = incoming_sym::<Othello>(canon, is_root, Real(&state));
        let mut n = 0;
        for i in 0..children.len() {
            let Some(child_id) = children.node_id(i) else { continue };
            if children.num_visits(i) < min_visits {
                continue;
            }
            n += 1;
            let action = real_action::<Othello>(children, i, incoming);
            let child = Othello::apply(state, &action);
            n += count_internal(index, child_id, child, min_visits, canon, false);
        }
        n
    }

    fn searched(iters: usize, seed: u64) -> (LabelSearch, State) {
        let root = State::default();
        let mut s = label_search(iters, seed);
        let _ = s.choose_action(&root);
        (s, root)
    }

    #[test]
    fn harvest_count_matches_an_independent_arena_walk() {
        let (s, root) = searched(64, 1);
        let filter = HarvestFilter {
            min_visits: 2,
            max_per_search: 100_000,
            max_depth: None,
        };
        let hits = harvest_tree_scored(&s, &root, &filter);
        let independent = count_internal(
            &s.index,
            s.root_id,
            root,
            filter.min_visits,
            canonicalizes(&s.config),
            true,
        );
        // `independent` counts children only; the harvest also keeps the
        // root when it passes `min_visits`.
        let root_kept = usize::from(s.root_stats.num_visits() >= filter.min_visits);
        assert_eq!(hits.len(), independent + root_kept);
        assert!(hits.len() > 1, "a 64-iteration search should expand a few nodes");
    }

    #[test]
    fn every_target_matches_a_fresh_child_array_read_and_the_right_perspective() {
        let (s, root) = searched(96, 2);
        let filter = HarvestFilter {
            min_visits: 1,
            max_per_search: 100_000,
            max_depth: None,
        };
        let canon = canonicalizes(&s.config);
        // Re-derive the depth-1 nodes' targets straight off the root's
        // ChildArray and confirm the harvest agrees, with the target read
        // from the *child's* side-to-move perspective (== the child node's
        // own `player_idx`), never the parent's.
        let node = s.index.get(s.root_id);
        let NodeState::Expanded(children) = node.status().unwrap() else {
            panic!("root not expanded")
        };
        let incoming = incoming_sym::<Othello>(canon, true, Real(&root));
        let hits = harvest_tree_scored(&s, &root, &filter);
        for i in 0..children.len() {
            let Some(child_id) = children.node_id(i) else { continue };
            if children.num_visits(i) == 0 {
                continue;
            }
            let action = real_action::<Othello>(children, i, incoming);
            let child_state = Othello::apply(root, &action);
            let child_pidx = Othello::player_to_move(&child_state).to_index();
            assert_eq!(
                s.index.get(child_id).player_idx,
                child_pidx,
                "child node player_idx disagrees with its state's side to move"
            );
            let want = children.expected_score(i, child_pidx) as f32;
            let got = hits
                .iter()
                .find(|(_, r)| r.black == child_state.black.bits() && r.white == child_state.white.bits())
                .map(|(_, r)| r.target)
                .expect("depth-1 node missing from harvest");
            assert!((want - got).abs() < 1e-6, "target skew: {want} vs {got}");
        }
    }

    #[test]
    fn max_per_search_keeps_exactly_the_cap_and_the_highest_visit_nodes() {
        let (s, root) = searched(128, 3);
        let uncapped = harvest_tree_scored(
            &s,
            &root,
            &HarvestFilter {
                min_visits: 1,
                max_per_search: 100_000,
                max_depth: None,
            },
        );
        let cap = 5.min(uncapped.len().saturating_sub(1)).max(1);
        assert!(uncapped.len() > cap, "need more nodes than the cap to test it");
        let capped = harvest_tree_scored(
            &s,
            &root,
            &HarvestFilter {
                min_visits: 1,
                max_per_search: cap,
                max_depth: None,
            },
        );
        assert_eq!(capped.len(), cap);
        let mut visits: Vec<u32> = uncapped.iter().map(|(v, _)| *v).collect();
        visits.sort_unstable_by(|a, b| b.cmp(a));
        let threshold = visits[cap - 1];
        for (v, _) in &capped {
            assert!(*v >= threshold, "capped harvest kept a low-visit node: {v} < {threshold}");
        }
    }

    #[test]
    fn depth_cap_bounds_the_walk() {
        let (s, root) = searched(128, 4);
        let shallow = harvest_tree(
            &s,
            &root,
            &HarvestFilter {
                min_visits: 1,
                max_per_search: 100_000,
                max_depth: Some(1),
            },
        );
        // Depth 0 (root) + depth 1 (root's children) only: every kept
        // position is at most one non-opening ply past the root.
        let root_ply = root.black.count_ones() + root.white.count_ones();
        for r in &shallow {
            let ply = (r.ply as u32) + 4;
            assert!(ply <= root_ply + 1, "depth cap leaked a deeper node");
        }
    }
}
