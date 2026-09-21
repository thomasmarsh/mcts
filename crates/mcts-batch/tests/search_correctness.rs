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

/// Same two-armed shape, but the mover switches on entering the terminal state, as in every
/// two-player game: the terminal value is reported from the *child's* mover's side. Action 0
/// leaves the opponent lost (`-0.8` for them), action 1 leaves them winning (`+0.8`), so from the
/// root mover's side the arms are worth `+0.8` and `-0.8`.
struct SwitchedBandit;

impl EnvOracle<Bandit> for SwitchedBandit {
    fn num_actions(&self) -> usize {
        2
    }

    fn init(&self, envs: &[Bandit]) -> StepOutput<Bandit> {
        TwoArmedBandit.init(envs)
    }

    fn transition(&self, states: &[Bandit], actions: &[u16]) -> TransitionOutput<Bandit> {
        let mut out = TwoArmedBandit.transition(states, actions);
        for v in out.step.value_prior.iter_mut() {
            *v = -*v;
        }
        out.player_switched = vec![true; states.len()];
        out
    }
}

#[test]
fn a_switched_mover_is_valued_from_the_parents_side() {
    let cfg = Config { num_simulations: 40, ..Config::default() };
    let tree = explore(&cfg, &SwitchedBandit, &[Bandit(0)]);
    let visits = tree.child_visits(0, tree.root());
    assert!(visits[0] > visits[1], "the arm that beats the opponent must win the visits: {visits:?}");
    let qs = tree.completed_qvalues(0, tree.root());
    assert!((qs[0] - 0.8).abs() < 1e-5 && (qs[1] + 0.8).abs() < 1e-5, "qs = {qs:?}");
}

/// Depth two: the root mover's arm 0 gives the opponent a choice of two terminals (worth `-0.5`
/// or `+0.9` to the opponent), arm 1 is a certain loss for the opponent (`-0.8`). A minimax player
/// picks arm 1 (the opponent takes the `+0.9` reply after arm 0). An engine with the perspective
/// flipped at every level would have the opponent take the reply that is best for the root mover,
/// value arm 0 at `-0.5` and arm 1 at `-0.8`, and so pick arm 0.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Deep(u8);

struct DeepGame;

impl EnvOracle<Deep> for DeepGame {
    fn num_actions(&self) -> usize {
        2
    }

    fn init(&self, envs: &[Deep]) -> StepOutput<Deep> {
        let n = envs.len();
        StepOutput {
            states: envs.to_vec(),
            terminal: vec![false; n],
            valid_actions: vec![true; n * 2],
            policy_prior: vec![0.5; n * 2],
            value_prior: vec![0.0; n],
        }
    }

    // States: 0 root; 1 = after arm 0 (opponent to move); 2 = after arm 1 (terminal); 3, 4 = the
    // opponent's replies from 1 (terminal, valued from the root mover's side after the switch).
    fn transition(&self, states: &[Deep], actions: &[u16]) -> TransitionOutput<Deep> {
        let n = states.len();
        let (mut out_states, mut terminal, mut value) = (Vec::new(), Vec::new(), Vec::new());
        let mut valid = vec![false; n * 2];
        for (i, (s, &a)) in states.iter().zip(actions).enumerate() {
            match (s.0, a) {
                (0, 0) => {
                    out_states.push(Deep(1));
                    terminal.push(false);
                    value.push(0.0);
                    valid[i * 2] = true;
                    valid[i * 2 + 1] = true;
                }
                (0, _) => {
                    out_states.push(Deep(2));
                    terminal.push(true);
                    value.push(-0.8);
                }
                (1, 0) => {
                    out_states.push(Deep(3));
                    terminal.push(true);
                    value.push(-(-0.5));
                }
                (_, _) => {
                    out_states.push(Deep(4));
                    terminal.push(true);
                    value.push(-0.9);
                }
            }
        }
        TransitionOutput {
            step: StepOutput { states: out_states, terminal, valid_actions: valid, policy_prior: vec![0.5; n * 2], value_prior: value },
            rewards: vec![0.0; n],
            player_switched: vec![true; n],
        }
    }
}

#[test]
fn minimax_over_two_plies_is_recovered() {
    let cfg = Config { num_simulations: 200, ..Config::default() };
    let tree = explore(&cfg, &DeepGame, &[Deep(0)]);
    let visits = tree.child_visits(0, tree.root());
    assert!(visits[1] > visits[0], "arm 1 (opponent -0.8) beats arm 0 (opponent replies +0.9): {visits:?}");
    let qs = tree.completed_qvalues(0, tree.root());
    assert!(qs[1] > qs[0], "qs = {qs:?}");
    assert!((qs[1] - 0.8).abs() < 1e-5, "qs = {qs:?}");
}

#[test]
fn the_gumbel_selected_action_is_a_most_visited_survivor_and_prefers_the_better_arm() {
    use mcts_batch::{gumbel_explore_with_noise, gumbel_selected_action};
    let cfg = Config { num_simulations: 32, num_considered_actions: 2, ..Config::default() };
    let mut better = 0;
    for seed in 0..40 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let (tree, noise) = gumbel_explore_with_noise(&cfg, &SwitchedBandit, &[Bandit(0)], &mut rng);
        let aid = gumbel_selected_action(&cfg, &tree, 0, &noise[0]);
        let visits = tree.child_visits(0, tree.root());
        assert_eq!(visits[aid as usize], *visits.iter().max().unwrap(), "seed {seed}: {visits:?}");
        better += usize::from(aid == 0);
    }
    assert!(better >= 35, "the winning arm should almost always survive halving: {better}/40");
}

#[test]
fn gumbel_explore_and_its_noise_variant_agree() {
    use mcts_batch::gumbel_explore_with_noise;
    let cfg = Config { num_simulations: 24, num_considered_actions: 2, ..Config::default() };
    let a = gumbel_explore(&cfg, &SwitchedBandit, &[Bandit(0)], &mut SmallRng::seed_from_u64(9));
    let (b, _) = gumbel_explore_with_noise(&cfg, &SwitchedBandit, &[Bandit(0)], &mut SmallRng::seed_from_u64(9));
    assert_eq!(a.child_visits(0, a.root()), b.child_visits(0, b.root()));
}
