use crate::game::Canonical;
use crate::game::Game;
use crate::game::Real;
use crate::strategies::mcts::config::GraphStats;
use crate::strategies::mcts::config::TranspositionKeying;
use crate::strategies::mcts::index;
use crate::strategies::mcts::index::Id;
use crate::strategies::mcts::node;
use crate::strategies::mcts::node::NodeState;
use crate::strategies::mcts::search::shared::MAX_REROOT_DEPTH;
use crate::strategies::mcts::search::TreeSearch;
use crate::strategies::mcts::table::TranspositionKey;
use crate::symmetry::incoming_sym;

use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::atomic::Ordering::Relaxed;

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: crate::strategies::mcts::Strategy<G>,
    crate::strategies::mcts::SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    /// `reuse_or_reset`'s counterpart for explicit `GraphSearch::Dag`
    /// re-rooting. Distinct from `reuse.rs`'s single-parent tree promotion
    /// path in two ways that mode requires: a promoted node can have more
    /// than one surviving parent (there is no single incoming edge to
    /// unhook, only an incoming-edge *count* -- see `Node::incoming_edges`),
    /// and every retained node's `ply` is root-relative, so promoting the
    /// root forces a rebase of every reachable node's `ply` and a full
    /// rebuild of the ply-keyed graph table (`table.rs`'s
    /// `TranspositionKey`) rather than an in-place key remap.
    ///
    /// Deliberately does not touch any retained node's `incoming_edges`
    /// count (`Node::add_incoming_edge`/`is_transposition`) even when
    /// pruning a now-unreachable sibling parent drops it below what's
    /// structurally true post-reroot: that count's one consumer
    /// (`shared::mcgs_correction_at_edge`, `GraphStats::Both` only) uses it
    /// to decide whether a node's `NodeStats` were ever pooled from more
    /// than one edge, which is a fact about *history*, not current graph
    /// shape -- `rebuild_reachable_graph` below never resets or splits a
    /// surviving node's own `NodeStats` back apart per-parent, so whatever
    /// pooling already happened is permanent and the resulting edge/node Q
    /// divergence the correction looks for stays real and worth checking
    /// even after a contributing parent becomes unreachable. The direction
    /// that would actually be unsound -- undercounting a node that still
    /// has two *live* incoming edges, silently suppressing a correction
    /// that should fire -- can't happen here: compaction only ever removes
    /// edges, and a new one is only ever added through `add_incoming_edge`.
    pub fn reuse_or_reset_graph(&mut self, player_idx: usize, state: &G::S) -> Id {
        if self.config.reuse_tree {
            if let Some(root_id) = self.try_promote_graph_root(state) {
                self.root_state = Some(state.clone());
                return root_id;
            }
        }
        let hash = G::zobrist_hash(state);
        let root_id = self.reset(player_idx, hash);
        self.root_state = Some(state.clone());
        root_id
    }

    /// Breadth-first search, bounded to `MAX_REROOT_DEPTH` plies, for a real
    /// game state reachable from the current root by following only
    /// resolved child edges. Unlike `reuse.rs`'s `find_reachable` (which
    /// matches candidates by hash and defers state verification to a
    /// separate replay), this tracks the real board state at every frontier
    /// node directly -- required here because a non-root node's `ChildArray`
    /// actions are stored in *that node's own* canonical orientation (see
    /// `node::real_action`'s doc comment), not the literal board's, so a
    /// candidate can only be found by translating each edge through
    /// `crate::symmetry::incoming_sym` as the walk descends, not by comparing a node's
    /// stored (canonical) hash against `target`'s literal hash. Comparing
    /// the replayed state directly also makes a separate post-hoc
    /// verification pass unnecessary: a match here is already proven, not
    /// just a hash-collision candidate.
    ///
    /// Returns `Some((None, root_id))` if the current root already matches
    /// `target` (0 plies). Returns `Some((Some(parent_id), matched_id))`
    /// otherwise, where `parent_id` is `matched_id`'s immediate predecessor
    /// on the path found -- needed only by `GraphStats::Edges`, to read the
    /// traversed edge's accumulated stats. `None` if nothing matches within
    /// the depth bound.
    fn find_reachable_graph(&self, target: &G::S) -> Option<(Option<Id>, Id)> {
        let root_state = self.root_state.as_ref()?;
        if root_state == target {
            return Some((None, self.root_id));
        }
        let canonicalizes = self.config.uses_transpositions();
        let mut frontier: Vec<(Id, G::S)> = vec![(self.root_id, root_state.clone())];
        for _ in 0..MAX_REROOT_DEPTH {
            let mut next = Vec::new();
            for (id, real_state) in frontier {
                let node = self.index.get(id);
                if !node.is_expanded() {
                    continue;
                }
                let children = node.children();
                let incoming_sym =
                    incoming_sym::<G>(canonicalizes, node.is_root(), Real(&real_state));
                for i in 0..children.len() {
                    let Some(child_id) = children.node_id(i) else {
                        continue;
                    };
                    let action = node::real_action::<G>(children, i, incoming_sym);
                    let child_state = G::apply(real_state.clone(), &action);
                    if child_state == *target {
                        return Some((Some(id), child_id));
                    }
                    next.push((child_id, child_state));
                }
            }
            frontier = next;
        }
        None
    }

    /// The promote half of `reuse_or_reset_graph`. `None` means "no
    /// verified match", i.e. the caller should fall back to `reset()`.
    fn try_promote_graph_root(&mut self, state: &G::S) -> Option<Id> {
        let (parent_id, matched_id) = self.find_reachable_graph(state)?;
        let Some(parent_id) = parent_id else {
            // The current root already matches -- nothing moved, so there is
            // nothing to rebase.
            return Some(self.root_id);
        };

        // `matched_id`'s `ChildArray` (if it has one) was generated in
        // *its own* canonical orientation -- correct for a non-root node,
        // but `crate::symmetry::incoming_sym` hard-codes identity once
        // `is_root()` is true, on the assumption a root's actions are
        // already literal (see `expand`'s doc comment). Translate them back
        // to the literal board now, before flipping `is_root` below makes
        // every future `incoming_sym`/`real_action` call for this node stop
        // translating at all -- otherwise the very first descent through it
        // after promotion applies a still-canonical action straight to the
        // real board and corrupts play.
        let canonicalizes = self.config.uses_transpositions();
        let sym = incoming_sym::<G>(canonicalizes, false, Real(state));
        if let Some(children) = self.index.get_mut(matched_id).children_mut() {
            children
                .retranslate_actions(|a| G::invert_action(Canonical(a.clone()), sym).into_inner());
        }

        // `GraphStats::Edges` still has its real per-edge statistics living
        // in the traversed parent's `ChildArray` row, exactly like legacy
        // tree reuse -- lift them out before repointing the root. `Nodes`/
        // `Both` need no such step: `stack::StatsRef` already reads a graph
        // node's own accumulated `NodeStats` at the root (see
        // `NodeStack::current_stats`), so promoting only has to flip
        // `is_root`, not copy anything.
        if !self
            .config
            .graph_stats()
            .is_some_and(GraphStats::uses_nodes)
        {
            let parent = self.index.get(parent_id);
            let idx = parent.child_index(matched_id);
            let children = parent.children();
            debug_assert_eq!(
                children.virtual_loss(idx),
                0,
                "a reused graph root should never carry virtual loss in flight -- \
                 it must have been released symmetrically by the search that \
                 produced it"
            );
            let edge_stats = children.extract_stats(idx);
            self.root_stats = edge_stats;
        }

        self.index.get_mut(self.root_id).is_root = false;
        self.root_id = matched_id;
        self.index.get_mut(self.root_id).is_root = true;

        // Under `PerPly`, the new root's `ply` is exactly how many edges the
        // walk crossed to reach it (root-relative, incremented by exactly
        // one per edge -- see `shared::new_child`), so it *is* the rebase
        // amount, with no need to separately count the path length.
        // `StateOnly` doesn't have this property -- a shared node's `ply` is
        // just whichever depth first created it, not necessarily the depth
        // this promotion just walked -- so `rebuild_reachable_graph` ignores
        // `depth` under that keying and recomputes every node's `ply` from
        // its own BFS distance to the new root instead.
        let depth = self.index.get(self.root_id).ply;
        self.rebuild_reachable_graph(depth);

        self.stats.accum_depth.store(0, Relaxed);
        self.stats.max_depth.store(0, Relaxed);
        self.stats.iter_count.store(0, Relaxed);
        Some(self.root_id)
    }

    /// Rebuilds `self.index` to contain only the subtree reachable from the
    /// (already updated) `self.root_id`, exactly like `search/compact.rs`'s
    /// `compact` (a DAG walk, deduping any node reached through more than
    /// one surviving parent) -- but, unlike `compact`, unconditional rather
    /// than gated on `max_arena_len`: every retained node's `ply` also gets
    /// recomputed here, and a stale `ply` isn't just wasted memory, it
    /// corrupts every future `TranspositionKey` lookup at this depth or
    /// deeper. The ply-keyed graph table is therefore rebuilt from scratch
    /// alongside the arena rather than remapped in place -- there is no way
    /// to "fix up" an old `TranspositionKey` without recomputing it.
    ///
    /// Under `PerPly`, `depth` (the number of real plies just advanced) is
    /// subtracted from every retained node's `ply` -- sound because `ply` is
    /// unique per `PerPly` node: every path from the old root to a given
    /// node has the same length, since a different length would have kept
    /// the two occurrences as separate nodes in the first place (that's the
    /// keying's whole acyclicity argument, see `TranspositionKeying`'s doc
    /// comment). `StateOnly` has no such guarantee -- a shared node's stored
    /// `ply` is only ever whichever depth first created it, not necessarily
    /// the depth this promotion's own walk just crossed -- so `depth` is
    /// ignored for it and `ply` is instead recomputed as this walk's own BFS
    /// distance from the new root, the only value that's still well-defined
    /// once the root has moved.
    fn rebuild_reachable_graph(&mut self, depth: u32) {
        let keying = self.config.transposition_keying;
        let mut order: Vec<Id> = Vec::new();
        let mut distance: FxHashMap<Id, u32> = FxHashMap::default();
        let mut queue: VecDeque<Id> = VecDeque::new();
        queue.push_back(self.root_id);
        distance.insert(self.root_id, 0);
        while let Some(id) = queue.pop_front() {
            let node_distance = distance[&id];
            if let Some(NodeState::Expanded(children)) = self.index.get(id).status() {
                for i in 0..children.len() {
                    if let Some(child_id) = children.node_id(i) {
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            distance.entry(child_id)
                        {
                            e.insert(node_distance + 1);
                            queue.push_back(child_id);
                        }
                    }
                }
            }
            order.push(id);
        }

        let mut new_index = index::Arena::new();
        let mut old_to_new: FxHashMap<Id, Id> = FxHashMap::default();
        old_to_new.reserve(order.len());
        for &old_id in &order {
            let mut node = self.index.get(old_id).clone();
            node.ply = match keying {
                TranspositionKeying::PerPly => node.ply - depth,
                TranspositionKeying::StateOnly => distance[&old_id],
            };
            let new_id = new_index.insert(node);
            old_to_new.insert(old_id, new_id);
        }

        // Every retained non-root node's key becomes the rebuilt graph
        // table -- the root deliberately gets no entry here: its own key
        // uses the literal-board hash the caller computes from the real
        // state (see `choose_action`'s `table.insert_graph` call), not the
        // canonical hash a non-root `Node::hash` stores (see
        // `search::shared::resolve_child_id`). Under `PerPly` no other node
        // can ever collide with it (every non-root node's `ply` is >= 1,
        // the root's is always 0); under `StateOnly` the same is instead
        // guaranteed by the per-game contract that a root position never
        // recurs within a search (see `cycle_game_tests`'s doc comment).
        let mut graph_entries: Vec<(TranspositionKey, Id)> = Vec::with_capacity(order.len());
        for &old_id in &order {
            let new_id = old_to_new[&old_id];
            let node = new_index.get_mut(new_id);
            if let Some(children) = node.children_mut() {
                children.remap_child_ids(&old_to_new);
            }
            if !node.is_root {
                graph_entries.push((TranspositionKey::new(keying, node.hash, node.ply), new_id));
            }
        }

        self.root_id = old_to_new[&self.root_id];
        self.index = new_index;
        self.table.rebuild_graph(graph_entries);
    }
}
