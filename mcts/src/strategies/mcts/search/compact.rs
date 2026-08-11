use crate::game::Game;
use crate::strategies::mcts::index;
use crate::strategies::mcts::index::Id;
use crate::strategies::mcts::node::NodeState;
use crate::strategies::mcts::search::TreeSearch;

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: crate::strategies::mcts::Strategy<G>,
    crate::strategies::mcts::SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    /// Bounded pruning after re-rooting (`SearchConfig::max_arena_len`):
    /// rebuilds `self.index` to contain only the subtree reachable from the
    /// current root via *explored* children, discarding every unreachable
    /// sibling that `reuse_tree`'s promote-in-place otherwise leaves as
    /// garbage forever, and filters `self.table` to match. Only ever called
    /// from `reuse_or_reset`, single-threaded with no concurrent search in
    /// flight (`choose_action_tree_parallel`'s worker threads haven't been
    /// spawned yet at that point), so no locking beyond what `Arena`/
    /// `ChildArray` already do internally is needed here.
    ///
    /// Reachability, not depth: unlike `find_reachable` (deliberately bounded
    /// to `MAX_REROOT_DEPTH`, for a different purpose -- finding a promotion
    /// candidate quickly), this walks the *entire* reachable subtree however
    /// deep, since anything it missed would be a live node getting silently
    /// discarded.
    pub fn compact(&mut self) {
        // Every node reachable from the current root by following only
        // resolved (explored) child slots, in BFS order -- a DAG walk, not a
        // tree walk, since transpositions can make two different parents'
        // child slots resolve to the same node; `visited` dedupes so a
        // shared node is only visited (and only gets one entry in the
        // rebuilt arena) once.
        let mut order: Vec<Id> = Vec::new();
        let mut visited: FxHashSet<Id> = FxHashSet::default();
        let mut queue: VecDeque<Id> = VecDeque::new();
        queue.push_back(self.root_id);
        visited.insert(self.root_id);
        while let Some(id) = queue.pop_front() {
            if let Some(NodeState::Expanded(children)) = self.index.get(id).status() {
                for i in 0..children.len() {
                    if let Some(child_id) = children.node_id(i) {
                        if visited.insert(child_id) {
                            queue.push_back(child_id);
                        }
                    }
                }
            }
            order.push(id);
        }

        // Rebuild into a fresh arena, in the same order -- an `Arena`'s
        // insertion order *is* its id assignment, so `old_to_new` can be
        // read straight off the ids `insert` hands back, with no separate
        // id-allocation scheme needed. Each cloned node's `ChildArray` (via
        // `Node`/`ChildArray`'s ordinary `Clone`) still has its child ids
        // pointing at the *old* arena at this point -- fixed up below, once
        // every reachable node has a new id assigned.
        let mut new_index = index::Arena::new();
        let mut old_to_new: FxHashMap<Id, Id> = FxHashMap::default();
        old_to_new.reserve(order.len());
        for &old_id in &order {
            let node = self.index.get(old_id).clone();
            let new_id = new_index.insert(node);
            old_to_new.insert(old_id, new_id);
        }

        for &old_id in &order {
            let new_id = old_to_new[&old_id];
            if let Some(children) = new_index.get_mut(new_id).children_mut() {
                children.remap_child_ids(&old_to_new);
            }
        }

        self.root_id = old_to_new[&self.root_id];
        self.index = new_index;
        self.table.compact(&old_to_new);
    }
}
