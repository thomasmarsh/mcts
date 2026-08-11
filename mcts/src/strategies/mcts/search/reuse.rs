use crate::game::Game;
use crate::strategies::mcts::index::Id;
use crate::strategies::mcts::search::shared::MAX_REROOT_DEPTH;
use crate::strategies::mcts::search::TreeSearch;

use std::sync::atomic::Ordering::Relaxed;

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: crate::strategies::mcts::Strategy<G>,
    crate::strategies::mcts::SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    /// Breadth-first search, bounded to `MAX_REROOT_DEPTH` plies, for a node
    /// reachable from `start` (inclusive) whose position hash is
    /// `target_hash`, alongside the sequence of actions that reaches it.
    /// Matching by hash rather than by "the action played" means this works
    /// uniformly regardless of how many plies happened since `start` was
    /// last searched -- 0 (repeated call on an unchanged position), 1 (this
    /// search's own move, e.g. `self_play`'s single-engine-both-sides
    /// pattern), or more (this engine's own move plus one or more other
    /// movers' replies, e.g. `round_robin`'s alternating instances) -- with
    /// no need to change `Search::choose_action`'s signature to thread an
    /// explicit played action through from the caller.
    ///
    /// A hash match here is a *candidate*, not proof -- see
    /// `reuse_or_reset`, which replays the returned action path against the
    /// real state it knows `start` represents and verifies full equality
    /// before trusting it.
    ///
    /// Returns `Some((None, start, vec![]))` if `start` itself already
    /// matches (0 plies). Returns `Some((Some(parent_id), matched_id,
    /// path))` otherwise, where `parent_id` is `matched_id`'s immediate
    /// parent on the path found (needed to read the edge whose accumulated
    /// stats become the new root's `root_stats`) and `path` is the actions
    /// from `start` to `matched_id` in order. `None` if nothing matches
    /// within the depth bound.
    fn find_reachable(&self, start: Id, target_hash: u64) -> Option<(Option<Id>, Id, Vec<G::A>)> {
        if self.index.get(start).hash == target_hash {
            return Some((None, start, Vec::new()));
        }
        let mut frontier: Vec<(Id, Vec<G::A>)> = vec![(start, Vec::new())];
        for _ in 0..MAX_REROOT_DEPTH {
            let mut next = Vec::new();
            for (id, path) in frontier {
                let node = self.index.get(id);
                if !node.is_expanded() {
                    continue;
                }
                let children = node.children();
                for i in 0..children.len() {
                    if let Some(child_id) = children.node_id(i) {
                        let mut child_path = path.clone();
                        child_path.push(children.action(i).clone());
                        if self.index.get(child_id).hash == target_hash {
                            return Some((Some(id), child_id, child_path));
                        }
                        next.push((child_id, child_path));
                    }
                }
            }
            frontier = next;
        }
        None
    }

    /// Tree reuse across moves ("re-rooting", see `SearchConfig::
    /// reuse_tree`): tries to find the node
    /// matching `state` within `MAX_REROOT_DEPTH` plies of the current root
    /// and promote it in place -- repointing `root_id`, moving its incoming
    /// edge's accumulated stats onto `root_stats`, and flipping `is_root`
    /// off the old root / on the new one -- instead of discarding the whole
    /// tree. Falls back to the untouched full `reset()` when reuse is
    /// disabled, this is the very first call (`root_state` still `None`, so
    /// there's nothing to replay a candidate path against), or no verified
    /// match is found (first move of a game, the actual play went somewhere
    /// this side's own search never reached, or -- vanishingly unlikely,
    /// but checked rather than assumed impossible -- a candidate's hash
    /// matched by pure 64-bit collision and `find_reachable`'s path replay
    /// caught it).
    ///
    /// Deliberately leaves every other arena node and transposition-table
    /// entry untouched rather than compacting: they're either still
    /// reachable from the new root (still exactly as valid as before) or
    /// unreachable siblings of the played line (dead weight, not incorrect
    /// -- a `TableEntry` or `Node` doesn't stop meaning what it means just
    /// because the tree walk that would find it again got shorter).
    pub fn reuse_or_reset(&mut self, player_idx: usize, state: &G::S) -> Id {
        let hash = G::zobrist_hash(state);
        if self.config.reuse_tree && self.try_promote(state, hash).is_some() {
            self.root_state = Some(state.clone());
            // Bounded pruning (`SearchConfig::max_arena_len`): only checked
            // on the promote path -- a fresh `reset()` below already starts
            // from a single-node arena, nothing to compact.
            if let Some(max_len) = self.config.max_arena_len {
                if self.index.len() > max_len {
                    self.compact();
                }
            }
            return self.root_id;
        }
        let root_id = self.reset(player_idx, hash);
        self.root_state = Some(state.clone());
        root_id
    }

    /// The promote half of `reuse_or_reset` -- split out so the state-replay
    /// verification below has an early-return-friendly home. `None` means
    /// "no verified match", i.e. the caller should fall back to `reset()`.
    fn try_promote(&mut self, state: &G::S, hash: u64) -> Option<Id> {
        let root_state = self.root_state.clone()?;
        let (parent_id, matched_id, path) = self.find_reachable(self.root_id, hash)?;

        let replayed = path.iter().fold(root_state, |s, a| G::apply(s, a));
        if replayed != *state {
            return None;
        }

        if let Some(parent_id) = parent_id {
            let parent = self.index.get(parent_id);
            let idx = parent.child_index(matched_id);
            let children = parent.children();
            debug_assert_eq!(
                children.virtual_loss(idx),
                0,
                "a reused root should never carry virtual loss in flight -- \
                 it must have been released symmetrically by the search that \
                 produced it"
            );
            let edge_stats = children.extract_stats(idx);
            self.index.get_mut(self.root_id).is_root = false;
            self.root_stats = edge_stats;
            self.root_id = matched_id;
            self.index.get_mut(self.root_id).is_root = true;
        }
        self.stats.accum_depth.store(0, Relaxed);
        self.stats.iter_count.store(0, Relaxed);
        Some(self.root_id)
    }
}
