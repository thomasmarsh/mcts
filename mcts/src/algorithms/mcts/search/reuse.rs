use crate::game::Canonical;
use crate::game::Game;
use crate::game::Real;
use crate::algorithms::mcts::index::Id;
use crate::algorithms::mcts::node;
use crate::algorithms::mcts::search::shared::MAX_REROOT_DEPTH;
use crate::algorithms::mcts::search::TreeSearch;
use crate::symmetry::incoming_sym;

use std::sync::atomic::Ordering::Relaxed;

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: crate::algorithms::mcts::PolicyProfile<G>,
    crate::algorithms::mcts::SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    /// Breadth-first search, bounded to `MAX_REROOT_DEPTH` plies, for a real
    /// game state reachable from `start_state` (inclusive) whose position
    /// hash is `target_hash`. Tracks the real board state at every frontier
    /// node directly, translating each edge through `crate::symmetry::
    /// incoming_sym` as the walk descends -- a non-root node's `ChildArray`
    /// actions are stored in *that node's own* canonical orientation (see
    /// `node::real_action`'s doc comment), not the literal board's, so
    /// applying `children.action(i)` straight to a real state (as this used
    /// to) can silently target the wrong cell whenever a hash collision
    /// happens to land on a differently-oriented state. See `reroot.rs`'s
    /// `find_reachable_graph`, which this mirrors.
    ///
    /// A hash match is a *candidate*, not proof by itself -- `try_promote`
    /// still compares the returned state against the caller's own real
    /// `state` before trusting it, but that comparison is now checking a
    /// genuinely-replayed real state rather than re-deriving one from a
    /// canonical action path.
    ///
    /// Returns `Some((None, start, start_state))` if `start` itself already
    /// matches (0 plies). Returns `Some((Some(parent_id), matched_id,
    /// replayed_state))` otherwise, where `parent_id` is `matched_id`'s
    /// immediate parent on the path found (needed to read the edge whose
    /// accumulated stats become the new root's `root_stats`) and
    /// `replayed_state` is the real state reached by following that path.
    /// `None` if nothing matches within the depth bound.
    fn find_reachable(
        &self,
        start: Id,
        start_state: &G::S,
        target_hash: u64,
    ) -> Option<(Option<Id>, Id, G::S)> {
        if self.index.get(start).hash == target_hash {
            return Some((None, start, start_state.clone()));
        }
        let canonicalizes = self.config.canonicalizes();
        let mut frontier: Vec<(Id, G::S)> = vec![(start, start_state.clone())];
        for _ in 0..MAX_REROOT_DEPTH {
            let mut next = Vec::new();
            for (id, real_state) in frontier {
                let node = self.index.get(id);
                if !node.is_expanded() {
                    continue;
                }
                let children = node.children();
                let sym = incoming_sym::<G>(canonicalizes, node.is_root(), Real(&real_state));
                for i in 0..children.len() {
                    if let Some(child_id) = children.node_id(i) {
                        let action = node::real_action::<G>(children, i, sym);
                        let child_state = G::apply(real_state.clone(), &action);
                        if self.index.get(child_id).hash == target_hash {
                            return Some((Some(id), child_id, child_state));
                        }
                        next.push((child_id, child_state));
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
        // A canonicalized node can be shared by several literal orientations
        // through the transposition table. Promoting it changes the root's
        // action-orientation convention, but the same node may still be
        // reached later through a different orientation. Until the arena
        // records the orientation each ChildArray was created in, retaining
        // that shared subtree is unsound: its actions can be applied to an
        // occupied cell. Start a fresh literal root instead.
        if self.config.canonicalizes() {
            let root_id = self.reset(player_idx, G::zobrist_hash(state));
            self.root_state = Some(state.clone());
            return root_id;
        }
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
        let (parent_id, matched_id, replayed) =
            self.find_reachable(self.root_id, &root_state, hash)?;

        if replayed != *state {
            return None;
        }

        if let Some(parent_id) = parent_id {
            // `matched_id`'s own `ChildArray` (if any) was generated in its
            // own canonical orientation -- correct while it had a parent,
            // but `crate::symmetry::incoming_sym` hard-codes identity once
            // `is_root()` is true. Retranslate its stored actions to the
            // literal board now, before flipping `is_root` below makes every
            // future `real_action` call for this node stop translating at
            // all -- otherwise the first descent through it after promotion
            // applies a still-canonical action straight to the real board.
            let canonicalizes = self.config.canonicalizes();
            let sym = incoming_sym::<G>(canonicalizes, false, Real(state));
            if let Some(children) = self.index.get_mut(matched_id).children_mut() {
                children.retranslate_actions(|a| {
                    G::invert_action(Canonical(a.clone()), sym).into_inner()
                });
            }

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
        self.stats.max_depth.store(0, Relaxed);
        self.stats.iter_count.store(0, Relaxed);
        Some(self.root_id)
    }
}
