//! Gumbel root selection, batched -- the training-time exploration variant
//! of `crate::search`, ported from AlphaZero.jl's `BatchedMcts.
//! {get_considered_visits_sequence,get_considered_visits_table,
//! get_penality,gumbel_select_root_action,gumbel_select,gumbel_explore}`,
//! itself inspired by DeepMind's `mctx`. `explore` and `gumbel_explore`
//! differ only in how the *root's* action is chosen each simulation;
//! interior selection below the root is `crate::search::select_from`,
//! unmodified, in both.

use rand::rngs::SmallRng;
use rand::Rng;
use rayon::prelude::*;

use crate::oracle::EnvOracle;
use crate::search::{create_tree, eval_batch, select_from, target_policy, Config};
use crate::tree::{Tree, NO_ACTION, ROOT, UNVISITED};

/// Precompute, for `max_num_actions` candidates spread over `num_simulations`
/// simulations, the required root-child visit count at each simulation --
/// the Sequential Halving schedule, flattened into a lookup table so root
/// selection is branch-free and identical in shape across the whole batch
/// (`get_penalty` masks out every action whose current visit count doesn't
/// match this simulation's required count).
///
/// See AlphaZero.jl `BatchedMcts.get_considered_visits_sequence` / DeepMind
/// `mctx`'s `get_considered_visits_sequence` (this is a verbatim algorithm
/// port -- see that source for the derivation, not repeated here).
pub fn get_considered_visits_sequence(max_num_actions: usize, num_simulations: usize) -> Vec<i32> {
    if max_num_actions <= 1 {
        return (0..num_simulations as i32).collect();
    }
    let num_halving_steps = (max_num_actions as f64).log2().ceil() as usize;
    let mut sequence: Vec<i32> = Vec::with_capacity(num_simulations);
    let mut visits = vec![0i32; max_num_actions];
    let mut num_actions = max_num_actions;
    while sequence.len() < num_simulations {
        let num_extra_visits = (num_simulations / (num_halving_steps * num_actions)).max(1);
        for _ in 0..num_extra_visits {
            sequence.extend_from_slice(&visits[..num_actions]);
            for v in &mut visits[..num_actions] {
                *v += 1;
            }
        }
        num_actions = (num_actions / 2).max(2);
    }
    sequence.truncate(num_simulations);
    sequence
}

/// `get_considered_visits_sequence` for every `num_considered_actions` from
/// `1` to `num_actions`, indexed `table[num_considered_actions - 1]`.
///
/// See AlphaZero.jl `BatchedMcts.get_considered_visits_table`.
pub fn get_considered_visits_table(num_simulations: usize, num_actions: usize) -> Vec<Vec<i32>> {
    (1..=num_actions).map(|k| get_considered_visits_sequence(k, num_simulations)).collect()
}

/// `-Inf` for every root action whose current visit count doesn't match
/// simulation `t`'s (0-indexed) required count from the schedule, `0`
/// otherwise -- this is what forces root exploration to visit exactly the
/// schedule's candidates in order instead of degenerating into plain UCB.
///
/// See AlphaZero.jl `BatchedMcts.get_penality`.
fn get_penalty<Env: Clone>(
    cfg: &Config,
    tree: &Tree<Env>,
    bid: usize,
    considered_visits_table: &[Vec<i32>],
    t: usize,
) -> Vec<f32> {
    let num_valid_actions =
        (0..tree.num_actions()).filter(|&aid| tree.is_valid_action(bid, ROOT, aid as u16)).count();
    let num_considered = cfg.num_considered_actions.min(num_valid_actions).max(1);
    let child_visits = tree.child_visits(bid, ROOT);
    let considered_visits = considered_visits_table[num_considered - 1][t];
    child_visits.into_iter().map(|v| if v == considered_visits { 0.0 } else { f32::NEG_INFINITY }).collect()
}

/// Root-only action selection under Gumbel noise plus the Sequential
/// Halving penalty -- unlike `crate::search::select_action`, the raw
/// `target_policy` score is used directly (no softmax): the Gumbel-top-k
/// argmax needs the unnormalized log-prior-plus-completed-Q score, not a
/// probability.
///
/// See AlphaZero.jl `BatchedMcts.gumbel_select_root_action`.
fn gumbel_select_root_action<Env: Clone>(
    cfg: &Config,
    tree: &Tree<Env>,
    bid: usize,
    gumbel_noise: &[f32],
    considered_visits_table: &[Vec<i32>],
    t: usize,
) -> u16 {
    let policy = target_policy(cfg, tree, bid, ROOT);
    let penalty = get_penalty(cfg, tree, bid, considered_visits_table, t);
    let mut best_aid = NO_ACTION;
    let mut best_score = f32::NEG_INFINITY;
    for aid in 0..tree.num_actions() {
        let score = gumbel_noise[aid] + policy[aid] + penalty[aid];
        if score > best_score {
            best_score = score;
            best_aid = aid as u16;
        }
    }
    best_aid
}

/// One batch item's selection step under `gumbel_explore`: pick the root
/// action per `gumbel_select_root_action`, then fall back to the ordinary
/// interior walk (`crate::search::select_from`) if that root child has
/// already been expanded.
///
/// See AlphaZero.jl `BatchedMcts.gumbel_select` (single-tree body).
fn gumbel_select_from<Env: Clone>(
    cfg: &Config,
    tree: &Tree<Env>,
    bid: usize,
    gumbel_noise: &[f32],
    considered_visits_table: &[Vec<i32>],
    t: usize,
) -> (i32, u16) {
    let aid = gumbel_select_root_action(cfg, tree, bid, gumbel_noise, considered_visits_table, t);
    debug_assert_ne!(aid, NO_ACTION);
    let cnid = tree.child(bid, ROOT, aid);
    if cnid != UNVISITED {
        select_from(cfg, tree, bid, cnid)
    } else {
        (ROOT, aid)
    }
}

/// Sample one standard `Gumbel(0, 1)` variate: `-ln(-ln(U))`, `U ~
/// Uniform(0, 1)` excluding `0` (which would make the inner `ln` diverge).
fn sample_gumbel(rng: &mut SmallRng) -> f32 {
    let u: f32 = rng.gen_range(f32::EPSILON..1.0);
    -(-u.ln()).ln()
}

/// Run `cfg.num_simulations` Gumbel-root simulations against a fresh batch
/// of trees, one per `envs` entry -- the exploration variant `dump.rs`'s
/// self-play recording wants (diverse, policy-improving training targets),
/// as opposed to `crate::search::explore`'s noise-free inference variant.
///
/// See AlphaZero.jl `BatchedMcts.gumbel_explore`.
pub fn gumbel_explore<Env: Clone + Default + Sync + Send>(
    cfg: &Config,
    oracle: &dyn EnvOracle<Env>,
    envs: &[Env],
    rng: &mut SmallRng,
) -> Tree<Env> {
    gumbel_explore_with_noise(cfg, oracle, envs, rng).0
}

/// [`gumbel_explore`], also returning each tree's root Gumbel noise (`noise[bid][aid]`), which
/// [`gumbel_selected_action`] needs to name the search's final action.
pub fn gumbel_explore_with_noise<Env: Clone + Default + Sync + Send>(
    cfg: &Config,
    oracle: &dyn EnvOracle<Env>,
    envs: &[Env],
    rng: &mut SmallRng,
) -> (Tree<Env>, Vec<Vec<f32>>) {
    let mut tree = create_tree(oracle, envs, cfg.num_simulations);
    let a = tree.num_actions();
    let b = tree.batch_size();

    let gumbel: Vec<Vec<f32>> = (0..b).map(|_| (0..a).map(|_| sample_gumbel(rng)).collect()).collect();
    let considered_visits_table = get_considered_visits_table(cfg.num_simulations, a);

    for simnum in 2..=cfg.num_simulations as i32 {
        let t = (simnum - 2) as usize;
        let parent_frontier: Vec<(i32, u16)> = (0..b)
            .into_par_iter()
            .map(|bid| gumbel_select_from(cfg, &tree, bid, &gumbel[bid], &considered_visits_table, t))
            .collect();
        let frontier = eval_batch(oracle, &mut tree, simnum, &parent_frontier);
        crate::search::backpropagate_batch(&mut tree, &frontier);
    }
    (tree, gumbel)
}

/// The action Sequential Halving leaves standing at the root: among the root actions with the
/// most visits (the last survivors), the one with the highest `gumbel + log prior + sigma(q)`
/// (Danihelka et al. 2022, the action a Gumbel search plays; `gumbel` is that tree's
/// [`gumbel_explore_with_noise`] noise row).
pub fn gumbel_selected_action<Env: Clone>(cfg: &Config, tree: &Tree<Env>, bid: usize, gumbel: &[f32]) -> u16 {
    let visits = tree.child_visits(bid, ROOT);
    let most = visits.iter().copied().max().unwrap_or(0);
    let score = target_policy(cfg, tree, bid, ROOT);
    let mut best = NO_ACTION;
    let mut best_score = f32::NEG_INFINITY;
    for aid in 0..tree.num_actions() {
        if visits[aid] != most || !tree.is_valid_action(bid, ROOT, aid as u16) {
            continue;
        }
        let s = gumbel[aid] + score[aid];
        if best == NO_ACTION || s > best_score {
            best = aid as u16;
            best_score = s;
        }
    }
    debug_assert_ne!(best, NO_ACTION, "a root with a legal action has a most-visited legal action");
    best
}
