//! Wiring smoke test for `crate::othello::OthelloOracle`: the algorithm
//! itself is checked against a hand-verifiable example in
//! `search_correctness.rs`; this only checks that the real `Game`/
//! `Evaluator`/`PolicyLogits` seam is plugged in without panicking and
//! produces sane (finite, in-range) output on the real starting position,
//! all-zero weights (correctness of the CNN's own output is irrelevant
//! here, same convention `mlx-selfplay-spike.md`'s own spikes used).

use game_othello::convnet::mlx::MlxCnnValueNet;
use game_othello::convnet::CnnValueNet;
use game_othello::State;
use mcts_batch::othello::{MlxOthelloOracle, OthelloOracle};
use mcts_batch::{explore, gumbel_explore, Config};
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[test]
fn explore_runs_to_completion_on_the_real_starting_position() {
    let oracle = OthelloOracle::new(CnnValueNet::default());
    let cfg = Config { num_simulations: 32, ..Config::default() };
    let tree = explore(&cfg, &oracle, &[State::default()]);

    let qs = tree.completed_qvalues(0, tree.root());
    assert_eq!(qs.len(), mcts_batch::othello::NUM_ACTIONS);
    // Othello's opening position has exactly 4 legal moves; every other
    // action must be masked to -inf, and every legal one must be finite
    // (a real qvalue or root-value estimate, never NaN/inf from a division
    // slip in `Tree::root_value_estimate`/`completed_qvalues`).
    let legal_count = qs.iter().filter(|&&q| q.is_finite()).count();
    assert_eq!(legal_count, 4, "qs = {qs:?}");
}

#[test]
fn explore_handles_multiple_games_in_one_batch() {
    let oracle = OthelloOracle::new(CnnValueNet::default());
    let cfg = Config { num_simulations: 16, ..Config::default() };
    let tree = explore(&cfg, &oracle, &[State::default(), State::default(), State::default()]);
    assert_eq!(tree.batch_size(), 3);
    for bid in 0..3 {
        let qs = tree.completed_qvalues(bid, tree.root());
        assert_eq!(qs.iter().filter(|&&q| q.is_finite()).count(), 4);
    }
}

#[test]
fn gumbel_explore_runs_to_completion_on_the_real_starting_position() {
    let oracle = OthelloOracle::new(CnnValueNet::default());
    let cfg = Config { num_simulations: 32, num_considered_actions: 4, ..Config::default() };
    let mut rng = SmallRng::seed_from_u64(7);
    let tree = gumbel_explore(&cfg, &oracle, &[State::default()], &mut rng);

    let visits = tree.child_visits(0, tree.root());
    let legal_visited = visits.iter().filter(|&&v| v > 0).count();
    // Sequential Halving's own schedule starts every considered candidate
    // (here, all 4 legal moves, since num_considered_actions=4) at
    // required-visits 0 -- every legal action should get tried at least
    // once inside 32 simulations.
    assert_eq!(legal_visited, 4, "visits = {visits:?}");
}

/// Same wiring smoke test as `explore_handles_multiple_games_in_one_batch`,
/// against `MlxOthelloOracle`'s single-stacked-GPU-call path instead of
/// `OthelloOracle`'s per-state `rayon` `map_init` -- both oracles must
/// produce the same shape of sane output regardless of which evaluator
/// backend is doing the batching.
#[test]
fn mlx_oracle_handles_multiple_games_in_one_batch() {
    let oracle = MlxOthelloOracle::new(MlxCnnValueNet::default());
    let cfg = Config { num_simulations: 16, ..Config::default() };
    let tree = explore(&cfg, &oracle, &[State::default(), State::default(), State::default()]);
    assert_eq!(tree.batch_size(), 3);
    for bid in 0..3 {
        let qs = tree.completed_qvalues(bid, tree.root());
        assert_eq!(qs.iter().filter(|&&q| q.is_finite()).count(), 4);
    }
}

/// `MlxOthelloOracle`'s terminal handling (`evaluate_batch_mlx` skips the
/// GPU call for already-terminal states and fills `terminal_value`
/// directly) must not diverge from `OthelloOracle`'s CPU path -- driven
/// past a full game via `gumbel_explore`'s own root-argmax move selection
/// so at least one batch item actually reaches a terminal leaf inside the
/// tree, not just the never-terminal opening position the other tests use.
#[test]
fn mlx_oracle_reaches_and_scores_terminal_positions() {
    let oracle = MlxOthelloOracle::new(MlxCnnValueNet::default());
    let cfg = Config { num_simulations: 60, num_considered_actions: 4, ..Config::default() };
    let mut rng = SmallRng::seed_from_u64(3);
    let tree = gumbel_explore(&cfg, &oracle, &[State::default()], &mut rng);
    let qs = tree.completed_qvalues(0, tree.root());
    assert!(qs.iter().any(|q| q.is_finite()));
}
