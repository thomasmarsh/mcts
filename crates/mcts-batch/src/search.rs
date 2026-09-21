//! The deterministic (non-Gumbel) batched search: tree creation, selection,
//! expansion, and backpropagation, ported from AlphaZero.jl's
//! `BatchedMcts.{create_tree,select,eval!,backpropagate!,explore}`. See
//! `crate::tree` for the array layout and `crate::gumbel` for the
//! Gumbel-root variant that shares everything here except how the root's
//! action is chosen.

use rayon::prelude::*;

use crate::oracle::EnvOracle;
use crate::tree::{Tree, NO_ACTION, NO_PARENT, ROOT, UNVISITED};

/// `value_scale`/`max_visit_init` govern how quickly a node's completed-Q
/// estimate outweighs its raw policy prior as visits accumulate -- see
/// `qcoeff`. `num_considered_actions` only matters for
/// `crate::gumbel::gumbel_explore`; it is threaded through here too so a
/// single config serves both entry points, matching AlphaZero.jl's own
/// single `Policy` struct.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub num_simulations: usize,
    pub num_considered_actions: usize,
    pub value_scale: f32,
    pub max_visit_init: i32,
}

impl Default for Config {
    fn default() -> Self {
        Config { num_simulations: 64, num_considered_actions: 8, value_scale: 0.1, max_visit_init: 50 }
    }
}

/// `softmax` over a small fixed-size slice -- every caller here operates on
/// one node's `num_actions`-length row, never anything larger.
pub(crate) fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

/// L1-normalize a policy prior after masking out illegal actions -- the
/// oracle is free to return an arbitrary (even non-normalized) prior over
/// all `num_actions` slots; this is what the tree actually stores, exactly
/// once, at node-creation time.
///
/// See AlphaZero.jl `BatchedMcts.validate_prior`.
fn validate_prior(policy_prior: &[f32], valid_actions: &[bool]) -> Vec<f32> {
    let masked: Vec<f32> =
        policy_prior.iter().zip(valid_actions).map(|(&p, &v)| if v { p } else { 0.0 }).collect();
    let total: f32 = masked.iter().map(|p| p.abs()).sum();
    if total > 0.0 {
        masked.into_iter().map(|p| p / total).collect()
    } else {
        masked
    }
}

/// Build a fresh batch of trees at their root, one `oracle.init` call for
/// the whole batch.
///
/// See AlphaZero.jl `BatchedMcts.create_tree`.
pub fn create_tree<Env: Clone + Default>(
    oracle: &dyn EnvOracle<Env>,
    envs: &[Env],
    num_simulations: usize,
) -> Tree<Env> {
    assert!(!envs.is_empty(), "there should be at least one environment");
    let a = oracle.num_actions();
    let b = envs.len();
    let cap = num_simulations + 1;

    let info = oracle.init(envs);

    let mut tree = Tree {
        num_actions: a,
        cap,
        batch_size: b,
        parent: vec![NO_PARENT; b * cap],
        num_visits: vec![UNVISITED; b * cap],
        total_values: vec![0.0; b * cap],
        children: vec![UNVISITED; b * cap * a],
        state: vec![Env::default(); b * cap],
        terminal: vec![false; b * cap],
        valid_actions: vec![false; b * cap * a],
        prev_action: vec![NO_ACTION; b * cap],
        prev_reward: vec![0.0; b * cap],
        prev_switched: vec![false; b * cap],
        policy_prior: vec![0.0; b * cap * a],
        value_prior: vec![0.0; b * cap],
    };

    for bid in 0..b {
        let root_i = tree.node_idx(bid, ROOT);
        tree.num_visits[root_i] = 1;
        tree.state[root_i] = info.states[bid].clone();
        tree.terminal[root_i] = info.terminal[bid];
        tree.value_prior[root_i] = info.value_prior[bid];
        tree.total_values[root_i] = info.value_prior[bid];
        let row = &info.valid_actions[bid * a..(bid + 1) * a];
        let prior = validate_prior(&info.policy_prior[bid * a..(bid + 1) * a], row);
        for aid in 0..a {
            let idx = tree.action_idx(bid, ROOT, aid as u16);
            tree.valid_actions[idx] = row[aid];
            tree.policy_prior[idx] = prior[aid];
        }
    }
    tree
}

/// `qcoeff`: how strongly completed-Q outweighs the raw policy prior in
/// `target_policy`, growing with the most-visited child so early visits
/// don't overreact to a single noisy value.
///
/// See AlphaZero.jl `BatchedMcts.qcoeff`.
fn qcoeff<Env: Clone>(cfg: &Config, tree: &Tree<Env>, bid: usize, nid: i32) -> f32 {
    let max_child_visit = tree.child_visits(bid, nid).into_iter().max().unwrap_or(0);
    cfg.value_scale * (cfg.max_visit_init + max_child_visit) as f32
}

/// The score behind action selection: log-prior plus a completed-Q term
/// scaled by `qcoeff`, matching the Gumbel paper's deterministic interior
/// selection rule (used both for non-root selection in the plain `explore`
/// and, unmodified, for interior nodes reached by `gumbel_explore`).
///
/// See AlphaZero.jl `BatchedMcts.target_policy`.
pub(crate) fn target_policy<Env: Clone>(cfg: &Config, tree: &Tree<Env>, bid: usize, nid: i32) -> Vec<f32> {
    let qs = tree.completed_qvalues(bid, nid);
    let coeff = qcoeff(cfg, tree, bid, nid);
    (0..tree.num_actions())
        .map(|aid| {
            let prior = tree.policy_prior[tree.action_idx(bid, nid, aid as u16)];
            prior.ln() + coeff * qs[aid]
        })
        .collect()
}

/// The completed-Q improved-policy target at a node, softmax-normalized --
/// this crate's analogue of the per-node engine's own `mcts::algorithms::
/// mcts::gumbel::improved_policy` (same log-prior-plus-completed-Q shape,
/// via [`target_policy`]/[`qcoeff`], just parameterized by this crate's own
/// `Config` instead of `GumbelConfig`). An illegal action's `policy_prior`
/// is `0.0` (see [`validate_prior`]), so its `target_policy` score is `-inf`
/// and it softmaxes to exactly `0.0` here -- callers don't need a separate
/// legality mask.
pub fn improved_policy<Env: Clone>(cfg: &Config, tree: &Tree<Env>, bid: usize, nid: i32) -> Vec<f32> {
    softmax(&target_policy(cfg, tree, bid, nid))
}

/// Pick the action whose realized visit share most lags its target-policy
/// share -- the same deterministic rule Sequential Halving falls back to
/// once past the root (and the only rule `explore`, without Gumbel, ever
/// uses).
///
/// See AlphaZero.jl `BatchedMcts.select_action`.
fn select_action<Env: Clone>(cfg: &Config, tree: &Tree<Env>, bid: usize, nid: i32) -> u16 {
    let policy = softmax(&target_policy(cfg, tree, bid, nid));
    let child_visits = tree.child_visits(bid, nid);
    let total_visits: i32 = child_visits.iter().sum();
    let mut best_aid = NO_ACTION;
    let mut best_score = f32::NEG_INFINITY;
    for aid in 0..tree.num_actions() {
        let score = policy[aid] - child_visits[aid] as f32 / (total_visits + 1) as f32;
        if score > best_score {
            best_score = score;
            best_aid = aid as u16;
        }
    }
    best_aid
}

/// Walk down from `start` (default: the root) choosing `select_action` at
/// each step until a leaf (an unvisited action) or a terminal node is
/// reached. Returns `(parent, action)` for a leaf to expand, or `(node,
/// NO_ACTION)` if the walk hit a terminal node directly.
///
/// See AlphaZero.jl `BatchedMcts.select` (single-tree variant).
pub(crate) fn select_from<Env: Clone>(cfg: &Config, tree: &Tree<Env>, bid: usize, start: i32) -> (i32, u16) {
    let mut cur = start;
    loop {
        if tree.is_terminal(bid, cur) {
            return (cur, NO_ACTION);
        }
        let aid = select_action(cfg, tree, bid, cur);
        debug_assert_ne!(aid, NO_ACTION);
        let cnid = tree.child(bid, cur, aid);
        if cnid != UNVISITED {
            cur = cnid;
        } else {
            return (cur, aid);
        }
    }
}

/// `select_from` from the root, for every batch item, in parallel.
///
/// See AlphaZero.jl `BatchedMcts.select` (batched variant).
pub fn select_batch<Env: Clone + Sync>(cfg: &Config, tree: &Tree<Env>) -> Vec<(i32, u16)> {
    (0..tree.batch_size()).into_par_iter().map(|bid| select_from(cfg, tree, bid, ROOT)).collect()
}

/// Expand every non-terminal entry of `parent_frontier` in one batched
/// `oracle.transition` call (the entire point of this crate -- a single
/// evaluator call scores every batch item's newly-selected leaf together),
/// write the results into `tree` at node index `simnum`, and return the new
/// frontier (each batch item's newly created node, or its already-terminal
/// node unchanged).
///
/// See AlphaZero.jl `BatchedMcts.eval!`.
pub fn eval_batch<Env: Clone + Default>(
    oracle: &dyn EnvOracle<Env>,
    tree: &mut Tree<Env>,
    simnum: i32,
    parent_frontier: &[(i32, u16)],
) -> Vec<i32> {
    let a = tree.num_actions();
    let mut frontier: Vec<i32> = parent_frontier.iter().map(|&(parent, _)| parent).collect();

    let non_terminal: Vec<usize> =
        (0..parent_frontier.len()).filter(|&bid| parent_frontier[bid].1 != NO_ACTION).collect();
    if non_terminal.is_empty() {
        return frontier;
    }

    let parent_states: Vec<Env> =
        non_terminal.iter().map(|&bid| tree.state(bid, parent_frontier[bid].0).clone()).collect();
    let actions: Vec<u16> = non_terminal.iter().map(|&bid| parent_frontier[bid].1).collect();
    let info = oracle.transition(&parent_states, &actions);

    for (row, &bid) in non_terminal.iter().enumerate() {
        let (parent, aid) = parent_frontier[bid];
        let node_i = tree.node_idx(bid, simnum);
        let child_idx = tree.action_idx(bid, parent, aid);
        tree.parent[node_i] = parent;
        tree.children[child_idx] = simnum;
        tree.state[node_i] = info.step.states[row].clone();
        tree.terminal[node_i] = info.step.terminal[row];
        tree.prev_action[node_i] = aid;
        tree.prev_reward[node_i] = info.rewards[row];
        tree.prev_switched[node_i] = info.player_switched[row];
        tree.value_prior[node_i] = info.step.value_prior[row];
        let valid_row = &info.step.valid_actions[row * a..(row + 1) * a];
        let prior = validate_prior(&info.step.policy_prior[row * a..(row + 1) * a], valid_row);
        for aid2 in 0..a {
            let idx = tree.action_idx(bid, simnum, aid2 as u16);
            tree.valid_actions[idx] = valid_row[aid2];
            tree.policy_prior[idx] = prior[aid2];
        }
        frontier[bid] = simnum;
    }
    frontier
}

/// Walk each batch item's frontier node back to its root, incrementing
/// `num_visits` and accumulating `total_values` (each node holds values from
/// the perspective of its own mover, as `Tree::value` promises). Moving up
/// to the parent adds the transition's reward and negates the value exactly
/// when the mover switched -- the same nega-convention flip `Tree::qvalue`
/// uses for a single node, applied along the whole path.
///
/// See AlphaZero.jl `BatchedMcts.backpropagate!`.
pub fn backpropagate_batch<Env: Clone + Sync + Send>(tree: &mut Tree<Env>, frontier: &[i32]) {
    let cap = tree.cap;
    // Safe: `bid` partitions `parent`/`num_visits`/`total_values`/
    // `prev_reward`/`prev_switched` into disjoint `cap`-sized chunks, one
    // per batch item, so distinct `bid`s never touch the same index --
    // `chunks_exact_mut` hands each parallel task its own chunk instead of
    // requiring per-index synchronization.
    let parent = &tree.parent;
    let prev_reward = &tree.prev_reward;
    let prev_switched = &tree.prev_switched;
    let value_prior = &tree.value_prior;
    tree.num_visits
        .par_chunks_exact_mut(cap)
        .zip(tree.total_values.par_chunks_exact_mut(cap))
        .enumerate()
        .for_each(|(bid, (num_visits, total_values))| {
            let base = bid * cap;
            let mut cid = frontier[bid];
            let mut val = value_prior[base + cid as usize];
            loop {
                let i = cid as usize;
                // `val` is from the perspective of the mover at node `cid`, which is what
                // `Tree::value` promises; the reward and switch of the transition into `cid`
                // only matter once the value moves up to the parent.
                num_visits[i] += 1;
                total_values[i] += val;
                let p = parent[base + i];
                if p == NO_PARENT {
                    break;
                }
                val = if prev_switched[base + i] { prev_reward[base + i] - val } else { prev_reward[base + i] + val };
                cid = p;
            }
        });
}

/// Run `cfg.num_simulations` deterministic (non-Gumbel) simulations against
/// a fresh batch of trees, one per `envs` entry.
///
/// See AlphaZero.jl `BatchedMcts.explore`.
pub fn explore<Env: Clone + Default + Sync + Send>(
    cfg: &Config,
    oracle: &dyn EnvOracle<Env>,
    envs: &[Env],
) -> Tree<Env> {
    let mut tree = create_tree(oracle, envs, cfg.num_simulations);
    for simnum in 2..=cfg.num_simulations as i32 {
        let parent_frontier = select_batch(cfg, &tree);
        let frontier = eval_batch(oracle, &mut tree, simnum, &parent_frontier);
        backpropagate_batch(&mut tree, &frontier);
    }
    tree
}
