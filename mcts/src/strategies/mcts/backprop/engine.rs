use super::super::node::{self, ChildArray, NodeStats};
use super::super::stack::NodeStack;
use super::super::*;
use super::*;
use crate::game::{Game, Transform};
use rustc_hash::FxHashMap;

type BackpropInitialization<A> = (Vec<f64>, Vec<(A, usize)>, FxHashMap<index::Id, Transform>);

struct PathNode {
    node_id: index::Id,
    node_idx: usize,
    parent_id_opt: Option<index::Id>,
}

pub trait BackpropStrategy: Clone + Sync + Send + Default {
    /// Whether this strategy populates `PlayerStats::posterior_mean`/
    /// `posterior_variance` (and, for `BayesNumeric`, `posterior_grid`) via
    /// `update_posterior` below. `BayesUct1`/`BayesUct2`
    /// (`select/bayes.rs`) set `Requirements::needs_posterior`, checked
    /// against this at `SearchConfig::validate()`-time so that pairing is
    /// caught as a config error rather than silently reading zeroed fields.
    fn provides_posterior(&self) -> bool {
        false
    }

    /// Whether this strategy replaces each ancestor's arithmetic-mean value
    /// with a soft-Bellman (mellowmax) aggregate via `recompute_value`.
    /// `select::Ments` sets `Requirements::needs_softmax_value`, checked
    /// against this at `SearchConfig::validate()`-time so the E2W / soft-
    /// backup pairing is a config error rather than MENTS silently selecting
    /// on plain Monte-Carlo means. Overridden only by `SoftmaxBackprop`.
    fn provides_softmax_value(&self) -> bool {
        false
    }

    /// Recomputes and writes one node's Bayesian posterior for `player`:
    /// the normal-normal conjugate posterior from the node's own
    /// observations if it has no expanded children yet, otherwise the
    /// extremum distribution over its (already-updated-this-call, or
    /// stale-from-a-previous-call for off-path siblings -- tolerated the
    /// same way `derive_pn_dpn`'s partial recomputation already is)
    /// children's posteriors. The extremum direction depends on `mover`
    /// (this node's own `Node::player_idx`, i.e. which player actually
    /// chooses among these children): MAX when `player == mover` (that
    /// player picks their own best child), MIN otherwise (the mover's
    /// choice is adversarial to every other player, in this codebase's
    /// zero-sum two-player convention -- picking a MAX unconditionally
    /// here previously gave every non-mover player a systematically
    /// over-optimistic posterior, since it credited them with a choice
    /// only the actual mover gets to make). Called once per player, per
    /// ancestor node, from the default `update` body below -- default
    /// no-op, only overridden by `BayesGaussian`/`BayesNumeric`.
    fn update_posterior<A: crate::game::Action>(
        &self,
        _player: usize,
        _mover: usize,
        _slot: &PosteriorSlot<A>,
        _own_children: Option<&ChildArray<A>>,
    ) {
    }

    /// How many plies of ancestors, counting from (but not including) the
    /// just-backpropagated leaf, get their own per-player value *recomputed
    /// from their children and overwritten in place* by `recompute_value`
    /// below, instead of being left as the ordinary Monte-Carlo average
    /// `update`'s per-node block already wrote earlier this same call. `0`
    /// (the default) disables the pass entirely -- every ancestor keeps the
    /// plain averaging behavior. `u32::MAX` means "every ancestor".
    /// Overridden by `MinimaxBackprop` (MCTS-MB-n) and `PowerMeanBackprop`
    /// (Power-UCT).
    fn recompute_depth(&self) -> u32 {
        0
    }

    /// Recompute `node`'s own per-player value from its (already-updated-
    /// this-call) children and overwrite it in place via `slot`. Called once
    /// per ancestor within `recompute_depth()` plies of the leaf, leaf-to-
    /// root, from `update` below. Default no-op. `index` is passed for
    /// operators that need a child's proven status (`PowerMeanBackprop`);
    /// operators that don't (`MinimaxBackprop`) ignore it.
    fn recompute_value<A: crate::game::Action>(
        &self,
        _node: &node::Node<A>,
        _slot: &PosteriorSlot<A>,
        _index: &TreeIndex<A>,
        _num_players: usize,
    ) {
    }

    /// When `Some((lambda, max_child))`, `update` replaces the ordinary mean
    /// backup with a truncated λ-return: each ancestor accumulates its own
    /// bootstrapped target `G_t = (1 − λ)·V(s_{t+1}) + λ·G_{t+1}` (base case
    /// `G_L = z`, the rollout return) instead of the shared terminal return
    /// `z`. `lambda == 1.0` is identical to the mean backup, so `TdBackprop`
    /// returns `None` there and `update` takes the untouched `Classic` path.
    /// `.1` is the MaxMCTS(λ) flag: bootstrap from `max` over the node's
    /// children rather than the on-path child. Default `None` (every non-TD
    /// strategy); overridden only by `TdBackprop`.
    fn td_lambda(&self) -> Option<(f64, bool)> {
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn update<G>(
        &self,
        stack: &NodeStack<G::A>,
        global: &TreeStats<G>,
        index: &TreeIndex<G::A>,
        root_stats: &NodeStats,
        root_state: &G::S,
        canonicalizes: bool,
        trial: simulate::Trial<G>,
        flags: BackpropFlags,
        use_mcts_solver: bool,
        graph_stats: Option<GraphStats>,
    ) where
        G: Game,
    {
        let (utilities, mut actions, incoming_syms) =
            initialize_actions(stack, index, root_state, canonicalizes, &trial, &flags);
        let td = self.td_lambda();
        let mut td_return = if td.is_some() {
            utilities.clone()
        } else {
            Vec::new()
        };
        let mut child_value = None;
        let recompute_depth = self.recompute_depth();
        let mut is_leaf = true;

        for (ply_from_leaf, (parent_entry_opt, (node_id, node_idx))) in
            (0u32..).zip(stack.reverse_pairs2())
        {
            let parent_id_opt = parent_entry_opt.map(|(id, _)| *id);
            let path = PathNode {
                node_id: *node_id,
                node_idx: *node_idx,
                parent_id_opt,
            };
            debug_assert!(
                (path.parent_id_opt.is_some() && !index.get(path.node_id).is_root())
                    || (path.parent_id_opt.is_none() && index.get(path.node_id).is_root())
            );
            let step_utilities = td_target::<G>(
                td,
                &mut td_return,
                child_value.as_deref(),
                index.get(path.node_id),
                &utilities,
            );
            update_statistics(index, root_stats, &path, step_utilities, graph_stats);
            child_value = td
                .is_some()
                .then(|| updated_child_value::<G>(index, root_stats, &path, graph_stats));
            refresh_posterior::<G, Self>(self, index, root_stats, &path);
            recompute_configured_value::<G, Self>(
                self,
                index,
                root_stats,
                &path,
                ply_from_leaf,
                recompute_depth,
            );
            propagate_solver::<G>(index, path.node_id, is_leaf, &trial, use_mcts_solver);
            is_leaf = false;
            collect_tree_action::<G>(
                &flags,
                index,
                &path,
                &incoming_syms,
                &utilities,
                global,
                &mut actions,
            );
        }
        update_action_tables(flags, global, &actions, &trial.actions, &utilities);
    }
}

fn initialize_actions<G: Game>(
    stack: &NodeStack<G::A>,
    index: &TreeIndex<G::A>,
    root_state: &G::S,
    canonicalizes: bool,
    trial: &simulate::Trial<G>,
    flags: &BackpropFlags,
) -> BackpropInitialization<G::A> {
    let needs_actions =
        flags.amaf() || flags.grave() || flags.global() || flags.lgr() || flags.lgr2();
    let actions = if needs_actions {
        trial.actions.clone()
    } else {
        Vec::new()
    };
    // Every stack node's own incoming symmetry is replayed from the root.
    // It cannot be cached on an edge, and is only needed while reading tree actions.
    let incoming_syms = if needs_actions {
        stack.incoming_syms::<G>(index, root_state, canonicalizes).0
    } else {
        FxHashMap::default()
    };
    // A natural terminal playout already has its utilities. Cutoff trials use their
    // evaluator result before falling back to the game's winner-based score.
    let utilities = trial
        .terminal
        .utilities(G::num_players())
        .or_else(|| trial.cutoff_utilities.clone())
        .unwrap_or_else(|| G::compute_utilities(&trial.state));
    (utilities, actions, incoming_syms)
}

fn td_target<'a, G: Game>(
    td: Option<(f64, bool)>,
    td_return: &'a mut Vec<f64>,
    child_value: Option<&[f64]>,
    node: &node::Node<G::A>,
    utilities: &'a [f64],
) -> &'a [f64] {
    let Some((lambda, max_child)) = td else {
        return utilities;
    };
    if let Some(child_value) = child_value {
        let bootstrap = if max_child {
            max_child_bootstrap(node, G::num_players()).unwrap_or_else(|| child_value.to_vec())
        } else {
            child_value.to_vec()
        };
        td_lambda_step(td_return, &bootstrap, lambda);
    }
    td_return
}

fn update_statistics<A: crate::game::Action>(
    index: &TreeIndex<A>,
    root_stats: &NodeStats,
    path: &PathNode,
    utilities: &[f64],
    graph_stats: Option<GraphStats>,
) {
    if index.get(path.node_id).is_root() {
        if graph_stats.is_some_and(GraphStats::uses_nodes) {
            index.get(path.node_id).stats.update(utilities);
        } else {
            root_stats.update(utilities);
        }
        return;
    }
    let parent_id = path.parent_id_opt.unwrap();
    debug_assert_ne!(parent_id, path.node_id);
    let parent = index.get(parent_id);
    if graph_stats.is_none_or(GraphStats::uses_edges) {
        parent.children().update(path.node_idx, utilities);
        parent.children().remove_virtual_loss(path.node_idx);
    }
    if graph_stats.is_some_and(GraphStats::uses_nodes) {
        let node = index.get(path.node_id);
        node.stats.update(utilities);
        node.stats.remove_virtual_loss();
    }
}

fn updated_child_value<G: Game>(
    index: &TreeIndex<G::A>,
    root_stats: &NodeStats,
    path: &PathNode,
    graph_stats: Option<GraphStats>,
) -> Vec<f64> {
    let players = 0..G::num_players();
    if graph_stats.is_some_and(GraphStats::uses_nodes) {
        let node = index.get(path.node_id);
        players.map(|p| node.stats.expected_score(p)).collect()
    } else if index.get(path.node_id).is_root() {
        players.map(|p| root_stats.expected_score(p)).collect()
    } else {
        let children = index.get(path.parent_id_opt.unwrap()).children();
        players
            .map(|p| children.expected_score(path.node_idx, p))
            .collect()
    }
}

fn refresh_posterior<G: Game, B: BackpropStrategy>(
    strategy: &B,
    index: &TreeIndex<G::A>,
    root_stats: &NodeStats,
    path: &PathNode,
) {
    if !strategy.provides_posterior() {
        return;
    }
    let node = index.get(path.node_id);
    let own_children = node.is_expanded().then(|| node.children());
    let mover = node.player_idx;
    if node.is_root() {
        let slot = PosteriorSlot::Root(root_stats);
        for player in 0..G::num_players() {
            strategy.update_posterior(player, mover, &slot, own_children);
        }
    } else {
        let slot = PosteriorSlot::Edge(
            index.get(path.parent_id_opt.unwrap()).children(),
            path.node_idx,
        );
        for player in 0..G::num_players() {
            strategy.update_posterior(player, mover, &slot, own_children);
        }
    }
}

fn recompute_configured_value<G: Game, B: BackpropStrategy>(
    strategy: &B,
    index: &TreeIndex<G::A>,
    root_stats: &NodeStats,
    path: &PathNode,
    ply_from_leaf: u32,
    recompute_depth: u32,
) {
    if recompute_depth == 0 || ply_from_leaf == 0 || ply_from_leaf > recompute_depth {
        return;
    }
    let node = index.get(path.node_id);
    let slot = if node.is_root() {
        PosteriorSlot::Root(root_stats)
    } else {
        PosteriorSlot::Edge(
            index.get(path.parent_id_opt.unwrap()).children(),
            path.node_idx,
        )
    };
    strategy.recompute_value(node, &slot, index, G::num_players());
}

fn propagate_solver<G: Game>(
    index: &TreeIndex<G::A>,
    node_id: index::Id,
    is_leaf: bool,
    trial: &simulate::Trial<G>,
    enabled: bool,
) {
    if !enabled {
        return;
    }
    let node = index.get(node_id);
    // Only an unresolved tree leaf can also be the terminal state found by playout.
    if is_leaf && trial.depth == 0 {
        if let Some(proven) = proven_from_terminal(&trial.terminal) {
            node.try_prove(proven);
        }
        if G::score_bounds().is_some() {
            if let Some(score) = G::terminal_score(&trial.state) {
                node.set_terminal_score(score);
            }
        }
    }
    derive_proven(node, index);
    derive_pn_dpn(node, index);
    derive_pn_dpn2(node, index);
    derive_player_pn(node, index, G::num_players());
    if G::num_players() == 2 {
        if let Some((min, max)) = G::score_bounds() {
            derive_score_bounds(node, index, min, max, node.player_idx == 0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_tree_action<G: Game>(
    flags: &BackpropFlags,
    index: &TreeIndex<G::A>,
    path: &PathNode,
    incoming_syms: &FxHashMap<index::Id, Transform>,
    utilities: &[f64],
    global: &TreeStats<G>,
    actions: &mut Vec<(G::A, usize)>,
) {
    if flags.amaf() {
        let sym = path
            .parent_id_opt
            .and_then(|id| incoming_syms.get(&id))
            .copied()
            .unwrap_or(Transform::IDENTITY);
        update_amaf::<G>(
            path.parent_id_opt,
            sym,
            actions,
            index,
            path.node_id,
            utilities,
        );
    } else if flags.grave() {
        update_grave::<G>(actions, index, global, path.node_id, utilities);
    }
    if !(flags.amaf() || flags.grave() || flags.global() || flags.lgr() || flags.lgr2())
        || index.get(path.node_id).is_root()
    {
        return;
    }
    let parent_id = path.parent_id_opt.unwrap();
    let sym = incoming_syms
        .get(&parent_id)
        .copied()
        .unwrap_or(Transform::IDENTITY);
    let action = node::real_action::<G>(index.get(parent_id).children(), path.node_idx, sym);
    actions.push((action, index.get(parent_id).player_idx));
}
