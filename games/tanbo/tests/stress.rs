//! Many-game random-playout stress tests for Tanbo.
//!
//! These drive thousands of seeded random games to end-to-end termination,
//! which is both slow and (being effectively exhaustive over move choices)
//! liable to hit rare states that a handful of unit tests would miss. That
//! combination belongs in a separate `tests/stress.rs` integration binary,
//! not `cargo test --lib` -- see AGENTS.md.
//!
//! Run explicitly with `cargo test -p game-tanbo --test stress --release`.

use game_tanbo::{State, Tanbo};
use mcts::game::Game;

/// A tiny deterministic xorshift RNG, so failures reproduce from just the
/// seed printed in the panic message.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Play `state` to termination with uniformly random legal moves, asserting
/// at every step that a legal move exists (Tanbo's rules guarantee a
/// non-terminal state always has one) and that the game terminates within
/// `max_steps`. Also asserts the final state has a winner: Tanbo has no
/// draws or ties.
fn play_random_game<const N: usize>(state: &mut State<N>, seed: u64, max_steps: usize) {
    let mut rng = Rng::new(seed);

    for step in 0..max_steps {
        if Tanbo::<N>::is_terminal(state) {
            assert!(
                Tanbo::<N>::winner(state).is_some(),
                "seed {seed}: terminal state has no winner -- Tanbo has no draws or ties"
            );
            return;
        }

        let mut actions = Vec::new();
        Tanbo::<N>::generate_actions(state, &mut actions);
        assert!(
            !actions.is_empty(),
            "seed {seed} step {step}: non-terminal state with no legal actions \
             (current/non-current root capture in games/tanbo/src/lib.rs should make \
             this impossible)"
        );

        let idx = (rng.next() as usize) % actions.len();
        *state = Tanbo::<N>::apply(state.clone(), &actions[idx]);
    }

    panic!("seed {seed}: did not terminate within {max_steps} random-play steps");
}

#[test]
fn stress_tanbo_9_dense() {
    for seed in 0u64..2000 {
        let mut state = State::<9>::new_dense();
        play_random_game(&mut state, seed, 20_000);
    }
}

#[test]
fn stress_tanbo_11_sparse() {
    for seed in 0u64..2000 {
        let mut state = State::<11>::new_sparse();
        play_random_game(&mut state, seed, 20_000);
    }
}

#[test]
fn stress_tanbo_13_dense() {
    for seed in 0u64..500 {
        let mut state = State::<13>::new_dense();
        play_random_game(&mut state, seed, 40_000);
    }
}

#[test]
fn stress_tanbo_19_dense() {
    for seed in 0u64..50 {
        let mut state = State::<19>::new_dense();
        play_random_game(&mut state, seed, 100_000);
    }
}
