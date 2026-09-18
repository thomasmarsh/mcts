//! Hand-verified checks of the Sequential Halving schedule table, per
//! `AGENTS.md`'s instrumentation-logic rule: this is exactly the kind of
//! counting/aggregation logic that should have a fast deterministic test
//! on a small hand-worked input, independent of any real search run.

use mcts_batch::gumbel::get_considered_visits_sequence;

/// Hand-traced from the algorithm itself (AlphaZero.jl
/// `BatchedMctsUtility.get_considered_visits_sequence`):
/// `max_num_actions=4, num_simulations=8` ->
/// `num_halving_steps = ceil(log2(4)) = 2`.
/// Round 1 (`num_actions=4`): `num_extra_visits = max(1, 8/(2*4)) = 1` ->
/// appends `[0,0,0,0]`, `num_actions` halves to `2`.
/// Round 2 (`num_actions=2`): `num_extra_visits = max(1, 8/(2*2)) = 2` ->
/// appends `[1,1]` then `[2,2]`.
/// Total: `[0,0,0,0,1,1,2,2]`.
#[test]
fn four_actions_eight_simulations_matches_hand_trace() {
    assert_eq!(get_considered_visits_sequence(4, 8), vec![0, 0, 0, 0, 1, 1, 2, 2]);
}

/// `max_num_actions <= 1` is the trivial base case: every simulation just
/// increments its own counter, `0..num_simulations`.
#[test]
fn single_action_is_the_identity_sequence() {
    assert_eq!(get_considered_visits_sequence(1, 5), vec![0, 1, 2, 3, 4]);
    assert_eq!(get_considered_visits_sequence(0, 3), vec![0, 1, 2]);
}

/// The sequence always has exactly `num_simulations` entries, and every
/// entry is non-negative -- true for any `(max_num_actions,
/// num_simulations)` pair, so worth checking across a spread of values
/// rather than only the one hand-traced case above.
#[test]
fn sequence_length_and_nonnegativity_hold_broadly() {
    for max_num_actions in 1..=16 {
        for num_simulations in [1, 2, 8, 16, 32, 64] {
            let seq = get_considered_visits_sequence(max_num_actions, num_simulations);
            assert_eq!(seq.len(), num_simulations, "max_num_actions={max_num_actions}");
            assert!(seq.iter().all(|&v| v >= 0));
        }
    }
}
