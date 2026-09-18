//! Correctness of `create_tree`/`select`/`eval`/`backpropagate` against a
//! tiny hand-verifiable two-armed-bandit environment: a root with 2
//! actions, each leading immediately to a terminal state with a fixed,
//! known value. No neural net, no Othello -- exactly the kind of small
//! hand-worked example `AGENTS.md`'s instrumentation-logic rule calls for,
//! isolating the batched-tree algorithm itself from `crate::othello`'s
//! wiring (covered separately in `othello_smoke.rs`).

use mcts_batch::{explore, gumbel_explore, Config, EnvOracle, StepOutput, TransitionOutput};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// `0` is the root; `1`/`2` are the two possible terminal children (chosen
/// by action `0`/`1` respectively). A real game's state would carry a
/// board; here the id alone is enough to know everything about it.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Bandit(u8);

/// Action `0` leads to a state worth `+0.8`; action `1` to `-0.8`. Values
/// are returned as-is (`player_switched = false` throughout, so `qvalue ==
/// value` everywhere and no perspective-flip arithmetic needs verifying
/// separately from the tree-walk/backprop logic itself).
const REWARD: [f32; 2] = [0.8, -0.8];

struct TwoArmedBandit;

impl EnvOracle<Bandit> for TwoArmedBandit {
    fn num_actions(&self) -> usize {
        2
    }

    fn init(&self, envs: &[Bandit]) -> StepOutput<Bandit> {
        let n = envs.len();
        StepOutput {
            states: envs.to_vec(),
            terminal: vec![false; n],
            valid_actions: vec![true; n * 2],
            policy_prior: vec![0.5; n * 2],
            value_prior: vec![0.0; n],
        }
    }

    fn transition(&self, states: &[Bandit], actions: &[u16]) -> TransitionOutput<Bandit> {
        let n = states.len();
        let out_states = actions.iter().map(|&a| Bandit(1 + a as u8)).collect();
        TransitionOutput {
            step: StepOutput {
                states: out_states,
                terminal: vec![true; n],
                valid_actions: vec![false; n * 2],
                policy_prior: vec![0.0; n * 2],
                value_prior: actions.iter().map(|&a| REWARD[a as usize]).collect(),
            },
            rewards: vec![0.0; n],
            player_switched: vec![false; n],
        }
    }
}

#[test]
fn explore_prefers_the_better_arm_but_still_visits_the_worse_one() {
    let oracle = TwoArmedBandit;
    let cfg = Config { num_simulations: 40, ..Config::default() };
    let tree = explore(&cfg, &oracle, &[Bandit(0)]);

    let visits = tree.child_visits(0, tree.root());
    assert!(visits[0] > visits[1], "the +0.8 arm should be visited more than the -0.8 arm: {visits:?}");
    assert!(visits[1] > 0, "select_action's visit/policy-share tracking should still explore the worse arm at least once: {visits:?}");

    let qs = tree.completed_qvalues(0, tree.root());
    assert!((qs[0] - 0.8).abs() < 1e-5, "qs = {qs:?}");
    assert!((qs[1] - (-0.8)).abs() < 1e-5, "qs = {qs:?}");
}

#[test]
fn explore_is_deterministic_given_a_fixed_oracle() {
    let oracle = TwoArmedBandit;
    let cfg = Config { num_simulations: 40, ..Config::default() };
    let a = explore(&cfg, &oracle, &[Bandit(0)]);
    let b = explore(&cfg, &oracle, &[Bandit(0)]);
    assert_eq!(a.child_visits(0, a.root()), b.child_visits(0, b.root()));
}

#[test]
fn explore_runs_multiple_independent_trees_in_one_batch() {
    let oracle = TwoArmedBandit;
    let cfg = Config { num_simulations: 40, ..Config::default() };
    let tree = explore(&cfg, &oracle, &[Bandit(0), Bandit(0), Bandit(0)]);
    assert_eq!(tree.batch_size(), 3);
    for bid in 0..3 {
        let visits = tree.child_visits(bid, tree.root());
        assert!(visits[0] > visits[1], "batch item {bid}: {visits:?}");
    }
}

#[test]
fn gumbel_explore_does_not_panic_and_prefers_the_better_arm() {
    let oracle = TwoArmedBandit;
    let cfg = Config { num_simulations: 16, num_considered_actions: 2, ..Config::default() };
    let mut rng = SmallRng::seed_from_u64(42);
    let tree = gumbel_explore(&cfg, &oracle, &[Bandit(0)], &mut rng);

    let visits = tree.child_visits(0, tree.root());
    // Gumbel forces both root candidates to be tried at least once by
    // Sequential Halving's own schedule (`get_considered_visits_sequence`
    // starts every considered action at required-visits 0) before the
    // (fixed) value gap can bias later simulations.
    assert!(visits[0] > 0 && visits[1] > 0, "both arms should get at least one Gumbel-forced visit: {visits:?}");
    assert!(visits[0] >= visits[1], "the +0.8 arm should not be visited less than the -0.8 arm: {visits:?}");
}
