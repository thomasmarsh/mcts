use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::Transform;
use crate::strategies::mcts::config::McgsCorrection;
use crate::strategies::mcts::config::TranspositionKeying;
use crate::strategies::mcts::index::Id;
use crate::strategies::mcts::node::Node;
use crate::strategies::mcts::node::NodeState;
use crate::strategies::mcts::node::NodeStats;
use crate::strategies::mcts::node::QInit;
use crate::strategies::mcts::search::shared::backprop_step;
use crate::strategies::mcts::search::shared::expand;
use crate::strategies::mcts::search::shared::last_tree_action;
use crate::strategies::mcts::search::shared::simulate_step;
use crate::strategies::mcts::search::shared::Shared;
use crate::strategies::mcts::search::shared::TreeIndex;
use crate::strategies::mcts::search::shared::TreeStats;
use crate::strategies::mcts::search::TreeSearch;
use crate::strategies::mcts::select;
use crate::strategies::mcts::select::SelectContext;
use crate::strategies::mcts::select::SelectStrategy;
use crate::strategies::mcts::simulate::SimulateStrategy;
use crate::strategies::mcts::stack::NodeStack;
use crate::strategies::mcts::table::TranspositionTable;

use rand::rngs::SmallRng;

/// One player's own tree under MO-ISMCTS (`IsmctsMode::MultiTree`) -- the
/// same shape `TreeSearch` itself keeps for every other mode
/// (`index`/`root_id`/`root_stats`/`stack`, plus its own `TreeStats` for
/// whatever `select`/`simulate`/`backprop` accumulate along the way), just
/// not living directly on `TreeSearch` since there are `G::num_players()` of
/// them instead of one.
pub(crate) struct PlayerTree<G: Game> {
    index: TreeIndex<G::A>,
    root_id: Id,
    root_stats: NodeStats,
    stack: Vec<(Id, usize)>,
    stats: TreeStats<G>,
}

impl<G: Game> PlayerTree<G> {
    /// `mover` is the real root position's own player to move -- the same
    /// value for every one of the `num_players` trees, since they all
    /// record the same underlying sequence of real actions and only start
    /// diverging in which statistics get consulted, not in whose turn it
    /// is at a given depth.
    fn new(mover: usize, num_players: usize, has_amaf: bool) -> Self {
        let index = TreeIndex::new();
        let root_id = index.insert(Node::new_root(mover, num_players, 0, has_amaf, false));
        Self {
            index,
            root_id,
            root_stats: NodeStats::new(num_players, has_amaf),
            stack: Vec::new(),
            stats: TreeStats::default(),
        }
    }
}

/// Descends every tree in `trees` together, one real ply at a time, from
/// each tree's own current position (tracked here, not on `PlayerTree`
/// itself, since it only matters for the duration of one iteration) to a
/// leaf ready for a rollout -- MO-ISMCTS's generalization of
/// `search/shared.rs::select_step` to one tree per player (Cowling, Powley &
/// Whitehouse 2012, Section IV-G).
///
/// At each node, only the tree belonging to whichever player is about to
/// move there is ever *selected* from (via its own availability-scored
/// `Ucb1`, restricted to this iteration's determinized legal actions,
/// exactly as `select_step` restricts its single tree); every other
/// player's corresponding node is still widened for that same action and
/// advanced onto it, so their own tree accumulates a matching node to
/// select from once it becomes *their* turn somewhere else in the game.
/// Because every tree is grown with the exact same ordered legal-action
/// list at every node they all visit together, a slot index resolved in one
/// tree's `ChildArray` names the same action in every other tree's
/// `ChildArray` at their corresponding node -- so the mover's own `best_idx`
/// is reused directly to advance every tree, with no separate lookup by
/// action identity needed.
///
/// `ctx_state` is the one real determinized state every tree's descent
/// shares; it's mutated in place, ending at the leaf's own state. Each
/// tree's `stack` is left holding its own root->leaf path, ready for
/// `backprop_step`.
#[allow(clippy::too_many_arguments)]
fn select_multi_tree<G: Game>(
    trees: &mut [PlayerTree<G>],
    ctx_state: &mut G::S,
    root_state: &G::S,
    table: &TranspositionTable,
    expand_threshold: u32,
    q_init: QInit,
    has_amaf: bool,
    ismcts_redeterminize: bool,
    select_strategy: &mut impl SelectStrategy<G>,
    rng: &mut SmallRng,
) {
    let mut current_ids: Vec<Id> = trees.iter().map(|t| t.root_id).collect();
    let mut incoming_idx = 0usize;
    loop {
        for (tree, &id) in trees.iter_mut().zip(current_ids.iter()) {
            tree.stack.push((id, incoming_idx));
        }

        // RIS-MCTS-style re-determinization (Goodman 2019), same as
        // `select_step`'s: redraw the one shared real state fresh from
        // this node's own mover's point of view, rather than trusting
        // whatever this iteration's descent carried down from an ancestor.
        if ismcts_redeterminize {
            *ctx_state = G::determinize(ctx_state.clone(), rng);
        }

        let mover = G::player_to_move(ctx_state).to_index();
        let mover_id = current_ids[mover];

        let num_visits = {
            let node_stack = NodeStack::new(trees[mover].stack.clone());
            node_stack
                .current_stats(&trees[mover].index, &trees[mover].root_stats, None)
                .num_visits()
        };

        // Every tree is expanded/widened together (see this function's doc
        // comment), so the mover's own tree's status/visit count at this
        // position already speaks for all of them -- no need to check the
        // others separately.
        let stop = match trees[mover].index.get(mover_id).status() {
            Some(NodeState::Terminal) => true,
            Some(NodeState::Expanded(_)) => num_visits < expand_threshold,
            None => {
                if num_visits < expand_threshold {
                    true
                } else {
                    for (p, &id) in current_ids.iter().enumerate() {
                        let _ = expand::<G>(
                            &trees[p].index,
                            id,
                            ctx_state,
                            false,
                            has_amaf,
                            false,
                            true,
                            None,
                        );
                    }
                    matches!(
                        trees[mover].index.get(mover_id).status(),
                        Some(NodeState::Terminal)
                    )
                }
            }
        };
        if stop {
            return;
        }

        let mut legal_actions = Vec::new();
        G::generate_actions(ctx_state, &mut legal_actions);
        let mut mover_legal_idxs = Vec::new();
        for (p, &id) in current_ids.iter().enumerate() {
            let children = trees[p].index.get(id).children();
            let idxs = children.grow(&legal_actions);
            for &idx in &idxs {
                children.add_availability(idx);
            }
            if p == mover {
                mover_legal_idxs = idxs;
            }
        }

        let (best_idx, is_new, action) = {
            let node_stack = NodeStack::new(trees[mover].stack.clone());
            let grave = trees[mover].stats.grave.read().unwrap();
            let mover_children = trees[mover].index.get(mover_id).children();
            let select_ctx = SelectContext {
                q_init,
                stack: &node_stack,
                root_stats: &trees[mover].root_stats,
                root_state,
                canonicalizes: false,
                state: ctx_state,
                player: mover,
                index: &trees[mover].index,
                table,
                grave: &grave,
                global: &trees[mover].stats,
                use_transpositions: false,
                graph_stats: None,
                solver_loss_threshold: 0,
                incoming_sym: Transform::IDENTITY,
            };
            let best_idx = select::ismcts_best_child(
                &select_ctx,
                mover_children,
                &mover_legal_idxs,
                select_strategy,
                rng,
            );
            (
                best_idx,
                mover_children.node_id(best_idx).is_none(),
                mover_children.action(best_idx),
            )
        };

        let new_state = G::apply(ctx_state.clone(), &action);
        let new_hash = G::zobrist_hash(&new_state);
        let new_mover = G::player_to_move(&new_state).to_index();
        let num_players = trees.len();

        for p in 0..num_players {
            let id = current_ids[p];
            let ply = trees[p].index.get(id).ply;
            let children = trees[p].index.get(id).children();
            // Claimed here so every tree's own `backprop_step` call later
            // has a matching `add_virtual_loss` to pair with the
            // `remove_virtual_loss` it unconditionally issues for every
            // edge on its stack -- true bookkeeping for lock-free tree
            // parallelism elsewhere, but still a paired invariant that must
            // hold even in this single-threaded loop.
            children.add_virtual_loss(best_idx);
            let child_id = children.get_or_create_child(best_idx, || {
                let child_id = trees[p].index.insert(Node::new_at_ply(
                    new_mover,
                    new_hash,
                    ply + 1,
                    num_players,
                    has_amaf,
                    false,
                ));
                trees[p].index.get(child_id).add_incoming_edge();
                child_id
            });
            current_ids[p] = child_id;
        }
        *ctx_state = new_state;
        incoming_idx = best_idx;

        if is_new && expand_threshold > 0 {
            for (tree, &id) in trees.iter_mut().zip(current_ids.iter()) {
                tree.stack.push((id, incoming_idx));
            }
            return;
        }
    }
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: crate::strategies::mcts::Strategy<G>,
    crate::strategies::mcts::SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    /// MO-ISMCTS (`SearchConfig::ismcts_mode == IsmctsMode::MultiTree`): one
    /// full tree per player, all descended together every iteration along
    /// the same real sequence of actions (`select_multi_tree`), so each
    /// player's own decisions accumulate statistics independent of every
    /// other player's -- see `SearchConfig::ismcts_mode`'s doc comment.
    ///
    /// Only ever reached from `choose_action`'s own `IsmctsMode::MultiTree`
    /// dispatch: `validate()` already requires a single-threaded,
    /// DAG/solver/prior/reuse-free configuration for either `IsmctsMode`
    /// variant, so none of that machinery needs a counterpart here.
    ///
    /// Ends by moving the root player's own tree into `self`'s usual
    /// single-tree fields (`index`/`root_id`/`root_stats`/`stack`/`stats`),
    /// so every existing post-search path (`select_final_action`,
    /// `compute_pv`, `verbose_summary`, `root_report`, `search_report`)
    /// reads it exactly as it would read `IsmctsMode::SingleTree`'s own
    /// tree, unchanged.
    pub(crate) fn choose_action_multi_tree(&mut self, state: &G::S) -> G::A {
        let num_players = G::num_players();
        let root_player = G::player_to_move(state).to_index();
        let has_amaf = self.config.requirements().amaf;

        // `self.index`/`root_id`/`root_stats`/`stack`/`stats` are overwritten
        // wholesale from the root player's own tree once the loop below
        // finishes -- only `self.table` needs clearing up front, since
        // `select_final_action` reads it (unused by `Ucb1`, but left tidy
        // rather than full of another mode's stale entries).
        self.table.clear();
        let mut trees: Vec<PlayerTree<G>> = (0..num_players)
            .map(|_| PlayerTree::new(root_player, num_players, has_amaf))
            .collect();

        // The root's own position is never hidden from any player -- unlike
        // every other node, whose first expansion legitimately reads
        // whichever iteration happens to reach it first, each tree's root
        // legal-action list and terminal status must be resolved against the
        // literal caller-supplied `state`, not a per-iteration
        // `G::determinize`d guess. Without this, a game whose terminal check
        // can read hidden information (e.g. Phantom's win check against a
        // guessed board) could have its very first iteration's guess
        // permanently -- `expand`'s `OnceLock` only ever resolves once --
        // and wrongly mark a real, ongoing root position `Terminal`, which
        // `select_final_action` has no fallback for.
        for tree in &trees {
            let _ = expand::<G>(
                &tree.index,
                tree.root_id,
                state,
                false,
                has_amaf,
                false,
                true,
                None,
            );
        }

        let table = TranspositionTable::default();

        self.timer.start(self.config.max_time);
        for _ in 0..self.config.max_iterations {
            if self.timer.done() {
                break;
            }
            for tree in &mut trees {
                tree.stack.clear();
            }

            let mut ctx_state = if !self.config.ismcts_redeterminize {
                G::determinize(state.clone(), &mut self.config.rng)
            } else {
                state.clone()
            };

            select_multi_tree(
                &mut trees,
                &mut ctx_state,
                state,
                &table,
                self.config.expand_threshold,
                self.config.q_init,
                has_amaf,
                self.config.ismcts_redeterminize,
                &mut self.config.select,
                &mut self.config.rng,
            );

            let prev_action = last_tree_action::<G>(
                &trees[root_player].index,
                &trees[root_player].stack,
                state,
                false,
            );
            let k = self.config.num_rollouts_per_leaf.max(1);
            for _ in 0..k {
                let trial = simulate_step(
                    self.config.max_playout_depth,
                    &trees[root_player].stats,
                    &mut self.config.simulate,
                    &ctx_state,
                    prev_action.clone(),
                    &mut self.config.rng,
                );
                for tree in &trees {
                    backprop_step(
                        &Shared {
                            index: &tree.index,
                            root_state: state,
                            root_stats: &tree.root_stats,
                            table: &table,
                            global: &tree.stats,
                            expand_threshold: self.config.expand_threshold,
                            q_init: self.config.q_init,
                            use_transpositions: false,
                            graph_stats: None,
                            explicit_dag: false,
                            keying: TranspositionKeying::default(),
                            use_mcts_solver: false,
                            max_playout_depth: self.config.max_playout_depth,
                            solver_loss_threshold: 0,
                            has_amaf,
                            mcgs_correction: McgsCorrection::default(),
                            use_ismcts: true,
                            ismcts_redeterminize: self.config.ismcts_redeterminize,
                        },
                        &tree.stack,
                        &self.config.backprop,
                        trial.clone(),
                        self.config.select.backprop_flags() | self.config.simulate.backprop_flags(),
                    );
                }
            }
        }

        let root_tree = trees.swap_remove(root_player);
        self.index = root_tree.index;
        self.root_id = root_tree.root_id;
        self.root_stats = root_tree.root_stats;
        self.stack = root_tree.stack;
        self.stats = root_tree.stats;
        self.root_state = Some(state.clone());

        self.select_final_action(state)
    }
}
