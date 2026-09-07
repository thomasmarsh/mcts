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

/// Where a harvested node's training target comes from.
///
/// The tree walk always uses the MCTS tree for *structure* (which nodes
/// exist, their visit counts, the [`HarvestFilter`]); the oracle only
/// supplies the number written into each [`Record::target`].
/// [`McSearchedValue`] reads that number from the search itself; the
/// `EdaxLabel` oracle in `dump.rs` reads it from an independent Edax
/// evaluation ([`crate::edax::EdaxEval`]) instead.
pub trait TargetOracle {
    /// Target for `state`, from the side-to-move perspective, in `[-1, 1]`.
    fn target(&mut self, state: &State) -> f32;
}

/// [`TargetOracle`] backed by a lookup table of searched values keyed by
/// `(black, white, side)`. Built from a pre-pass over the same tree the
/// walk traverses, so it reproduces [`harvest_tree_scored`]'s targets
/// through the [`TargetOracle`] seam.
pub struct McSearchedValue {
    table: std::collections::HashMap<(u64, u64, u8), f32>,
}

impl McSearchedValue {
    /// Read every filtered node's searched value out of `search` into a
    /// lookup table.
    pub fn from_search(search: &LabelSearch, root_state: &State, filter: &HarvestFilter) -> Self {
        let mut table = std::collections::HashMap::new();
        for (_, r) in harvest_tree_scored(search, root_state, filter) {
            table.insert((r.black, r.white, r.side), r.target);
        }
        Self { table }
    }
}

impl TargetOracle for McSearchedValue {
    fn target(&mut self, state: &State) -> f32 {
        let side = u8::from(state.turn == crate::Player::White);
        *self
            .table
            .get(&(state.black.bits(), state.white.bits(), side))
            .unwrap_or(&0.0)
    }
}

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
        !matches!(search.config.graph_search, GraphSearch::Dag(_))
            && !search.config.use_transpositions,
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
pub fn harvest_tree(
    search: &LabelSearch,
    root_state: &State,
    filter: &HarvestFilter,
) -> Vec<Record> {
    harvest_tree_scored(search, root_state, filter)
        .into_iter()
        .map(|(_, r)| r)
        .collect()
}

/// Like [`harvest_tree_scored`], but every kept node's target comes from
/// `oracle` (evaluated on that node's reconstructed position) instead of
/// from the MCTS child array. Node *selection* -- the tree structure, visit
/// counts, and `filter` -- is identical to the MCTS path.
pub fn harvest_tree_scored_oracle<O: TargetOracle>(
    search: &LabelSearch,
    root_state: &State,
    filter: &HarvestFilter,
    oracle: &mut O,
) -> Vec<(u32, Record)> {
    let canon = canonicalizes(&search.config);
    let mut out: Vec<HarvestHit> = Vec::new();

    let root_visits = search.root_stats.num_visits();
    if root_visits >= filter.min_visits {
        let mut rec = record_for(root_state, None);
        rec.target = oracle.target(root_state);
        out.push(HarvestHit {
            visits: root_visits,
            rec,
        });
    }
    walk_oracle(
        &search.index,
        search.root_id,
        *root_state,
        true,
        0,
        canon,
        filter,
        oracle,
        &mut out,
    );

    if out.len() > filter.max_per_search {
        out.sort_by_key(|h| std::cmp::Reverse(h.visits));
        out.truncate(filter.max_per_search);
    }
    out.into_iter().map(|h| (h.visits, h.rec)).collect()
}

#[allow(clippy::too_many_arguments)]
fn walk_oracle<O: TargetOracle>(
    index: &TreeIndex<crate::Move>,
    id: Id,
    state: State,
    is_root: bool,
    depth: u32,
    canon: bool,
    filter: &HarvestFilter,
    oracle: &mut O,
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
        let mut rec = record_for(&child_state, None);
        rec.target = oracle.target(&child_state);
        out.push(HarvestHit { visits, rec });
        walk_oracle(
            index,
            child_id,
            child_state,
            false,
            depth + 1,
            canon,
            filter,
            oracle,
            out,
        );
    }
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
        out.push(HarvestHit { visits, rec });
        walk(
            index,
            child_id,
            child_state,
            false,
            depth + 1,
            canon,
            filter,
            out,
        );
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
            let Some(child_id) = children.node_id(i) else {
                continue;
            };
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
        assert!(
            hits.len() > 1,
            "a 64-iteration search should expand a few nodes"
        );
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
            let Some(child_id) = children.node_id(i) else {
                continue;
            };
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
                .find(|(_, r)| {
                    r.black == child_state.black.bits() && r.white == child_state.white.bits()
                })
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
        assert!(
            uncapped.len() > cap,
            "need more nodes than the cap to test it"
        );
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
            assert!(
                *v >= threshold,
                "capped harvest kept a low-visit node: {v} < {threshold}"
            );
        }
    }

    /// A stub [`TargetOracle`] returning a fixed, position-dependent
    /// function -- so the seam test can assert every harvested record got
    /// exactly its own position's value, never a neighbour's.
    struct StubOracle;
    fn stub_value(state: &State) -> f32 {
        let b = state.black.bits();
        let w = state.white.bits();
        let side = u8::from(state.turn == crate::Player::White) as u64;
        (((b ^ w).wrapping_mul(2_654_435_761).wrapping_add(side)) % 2000) as f32 / 1000.0 - 1.0
    }
    impl TargetOracle for StubOracle {
        fn target(&mut self, state: &State) -> f32 {
            stub_value(state)
        }
    }

    #[test]
    fn oracle_seam_associates_each_target_with_its_own_position() {
        let (s, root) = searched(96, 5);
        let filter = HarvestFilter {
            min_visits: 1,
            max_per_search: 100_000,
            max_depth: None,
        };
        let hits = harvest_tree_scored_oracle(&s, &root, &filter, &mut StubOracle);
        assert!(hits.len() > 2);
        for (_, r) in &hits {
            let st = State {
                black: crate::BB::from_bits(r.black),
                white: crate::BB::from_bits(r.white),
                turn: if r.side == 0 {
                    crate::Player::Black
                } else {
                    crate::Player::White
                },
                ..State::default()
            };
            assert!(
                (r.target - stub_value(&st)).abs() < 1e-6,
                "harvested record carries the wrong position's target"
            );
        }
        // Same tree structure as the MCTS-scored walk: identical position set.
        let mcts = harvest_tree_scored(&s, &root, &filter);
        let key = |v: &[(u32, Record)]| {
            let mut k: Vec<_> = v.iter().map(|(_, r)| (r.black, r.white, r.side)).collect();
            k.sort_unstable();
            k
        };
        assert_eq!(key(&hits), key(&mcts));
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
