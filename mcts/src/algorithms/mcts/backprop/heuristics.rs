//! AMAF and GRAVE side-table updates used during backpropagation.

use super::super::node;
use super::super::*;
use crate::game::{Game, Transform};

use rustc_hash::FxHashMap;

#[allow(clippy::too_many_arguments)]
pub fn update_amaf<G: Game>(
    parent_id_opt: Option<index::Id>,
    parent_incoming_sym: Transform,
    trace: &[(G::A, usize)],
    index: &TreeIndex<G::A>,
    node_id: index::Id,
    utilities: &[f64],
) {
    // NOTE: O(n) here, but amaf could be calculated top down
    let node = index.get(node_id);
    if !node.is_root() {
        // parent_id must come from the caller's (parent, node) pair for
        // *this* node, not `stack.parent_id()` (always the leaf's
        // parent) — using the latter silently attributed AMAF updates
        // to the wrong parent for every non-leaf node in a multi-level
        // stack.
        let parent_id = parent_id_opt.unwrap();
        debug_assert_ne!(parent_id, node_id);
        debug_assert!(index.get(parent_id).is_expanded());
        let parent = index.get(parent_id);
        let children = parent.children();
        // Maps directly to the sibling's index in `children`, rather
        // than to its arena `Id`, so a match below can call
        // `add_amaf` without a second (previously O(n)) reverse lookup
        // from `Id` back to array position. `parent_incoming_sym` (not
        // `children.sym(i)`, a different value per `real_action`'s doc
        // comment) is `parent`'s own incoming symmetry.
        let sibling_actions: FxHashMap<_, usize> = (0..children.len())
            .filter_map(|i| {
                children
                    .node_id(i)
                    .map(|_| (node::real_action::<G>(children, i, parent_incoming_sym), i))
            })
            .collect();

        // The player who could have chosen any of `parent_id`'s sibling
        // actions is `parent_id`'s own mover -- not the resulting
        // child's `player_idx`, which names the mover of the *next* ply
        // (the opposite player in an alternating game). Matching against
        // the child's `player_idx` here inverted the check.
        let mover = parent.player_idx;
        for (action, p) in trace {
            if *p == mover {
                if let Some(&idx) = sibling_actions.get(action) {
                    (0..G::num_players()).for_each(|i| {
                        children.add_amaf(idx, i, utilities[i]);
                    })
                }
            }
        }
    }
}

pub(crate) fn update_grave<G: Game>(
    trace: &[(G::A, usize)],
    index: &TreeIndex<G::A>,
    global: &TreeStats<G>,
    node_id: index::Id,
    utilities: &[f64],
) {
    let node = index.get(node_id);
    if !node.is_root() {
        let mut grave = global.grave.write().unwrap();
        for (action, p) in trace {
            let players = grave
                .entry(node.hash)
                .or_insert_with(|| vec![Default::default(); G::num_players()]);
            let player = players.get_mut(*p).unwrap();
            let grave_stats = player.entry(action.clone()).or_default();
            grave_stats.num_visits += 1;
            grave_stats.score += utilities[*p];
        }
    }
}

/// Applies the action-history heuristics after the traversal has assembled
/// the complete tree-path-plus-playout action trace.
pub(crate) fn update_action_tables<G: Game>(
    flags: BackpropFlags,
    global: &TreeStats<G>,
    actions: &[(G::A, usize)],
    playout_actions: &[(G::A, usize)],
    utilities: &[f64],
) {
    if flags.global() {
        update_global(global, actions, utilities);
    }
    let chronological = (flags.nst() || flags.lgr() || flags.lgr2())
        .then(|| chronological_actions(actions, playout_actions));
    if flags.nst() {
        update_nst(global, chronological.as_ref().unwrap(), utilities);
    }
    if flags.lgr() {
        update_lgr(global, chronological.as_ref().unwrap(), utilities);
    }
    if flags.lgr2() {
        update_lgr2(global, chronological.as_ref().unwrap(), utilities);
    }
}

fn update_global<G: Game>(global: &TreeStats<G>, actions: &[(G::A, usize)], utilities: &[f64]) {
    let mut global_actions = global.actions.write().unwrap();
    for (action, player) in actions {
        let stats = global_actions.entry(action.clone()).or_default();
        stats.num_visits += 1;
        stats.score += utilities[*player];

        let mut player_actions = global.player_actions[*player].write().unwrap();
        let stats = player_actions.entry(action.clone()).or_default();
        stats.num_visits += 1;
        stats.score += utilities[*player];
    }
}

fn chronological_actions<A: crate::game::Action>(
    actions: &[(A, usize)],
    playout_actions: &[(A, usize)],
) -> Vec<(A, usize)> {
    let mut chronological = actions[playout_actions.len()..].to_vec();
    chronological.reverse();
    chronological.extend(playout_actions.iter().cloned());
    chronological
}

fn update_nst<G: Game>(global: &TreeStats<G>, actions: &[(G::A, usize)], utilities: &[f64]) {
    for pair in actions.windows(2) {
        let (previous, _) = &pair[0];
        let (action, player) = &pair[1];
        let mut bigrams = global.player_bigram_actions[*player].write().unwrap();
        let stats = bigrams
            .entry((previous.clone(), action.clone()))
            .or_default();
        stats.num_visits += 1;
        stats.score += utilities[*player];
    }
}

fn update_lgr<G: Game>(global: &TreeStats<G>, actions: &[(G::A, usize)], utilities: &[f64]) {
    let best = utilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for pair in actions.windows(2) {
        let (previous, _) = &pair[0];
        let (action, player) = &pair[1];
        if utilities[*player] >= best {
            global.player_replies[*player]
                .write()
                .unwrap()
                .insert(previous.clone(), action.clone());
        }
    }
}

fn update_lgr2<G: Game>(global: &TreeStats<G>, actions: &[(G::A, usize)], utilities: &[f64]) {
    let best = utilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for triple in actions.windows(3) {
        let (own_previous, first_player) = &triple[0];
        let (opponent, _) = &triple[1];
        let (action, player) = &triple[2];
        if first_player != player {
            continue;
        }
        let context = (own_previous.clone(), opponent.clone());
        let mut replies = global.player_replies2[*player].write().unwrap();
        if utilities[*player] >= best {
            replies.insert(context, action.clone());
        } else if replies.get(&context) == Some(action) {
            replies.remove(&context);
        }
    }
}
