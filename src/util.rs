use rand::Rng;

use crate::*;

const PRIMES: [usize; 16] = [
    14323, 18713, 19463, 30553, 33469, 45343, 50221, 51991, 53201, 56923, 64891, 72763, 74471,
    81647, 92581, 94693,
];

pub(super) fn random_best<T, F: Fn(&T) -> f32>(set: &[T], score_fn: F) -> Option<&T> {
    // To make the choice more uniformly random among the best moves,
    // start at a random offset and stride by a random amount.
    // The stride must be coprime with n, so pick from a set of 5 digit primes.

    let n = set.len();
    // Combine both random numbers into a single rng call.
    let r = rand::thread_rng().gen_range(0..n * PRIMES.len());
    let mut i = r / PRIMES.len();
    let stride = PRIMES[r % PRIMES.len()];

    let mut best_score = f32::NEG_INFINITY;
    let mut best = None;
    for _ in 0..n {
        let score = score_fn(&set[i]);
        debug_assert!(!score.is_nan());
        if score > best_score {
            best_score = score;
            best = Some(&set[i]);
        }
        i = (i + stride) % n;
    }
    best
}

/// Play a complete, new game with players using the two provided strategies.
///
/// Returns `None` if the game ends in a draw, or `Some(0)`, `Some(1)` if the
/// first or second strategy won, respectively.
pub fn battle_royale<G, S1, S2>(s1: &mut S1, s2: &mut S2) -> Option<usize>
where
    G: game::Game,
    G::S: Default + Clone,
    S1: strategies::Strategy<G>,
    S2: strategies::Strategy<G>,
{
    let mut state = G::S::default();
    let mut strategies: [&mut dyn strategies::Strategy<G>; 2] = [s1, s2];
    let mut s = 0;
    loop {
        if let Some(winner) = G::get_winner(&state) {
            return match winner {
                game::Winner::Draw => None,
                game::Winner::PlayerJustMoved => Some(1 - s),
                game::Winner::PlayerToMove => Some(s),
            };
        }
        let strategy = &mut strategies[s];
        match strategy.choose_move(&state) {
            Some(m) => {
                if let Some(new_state) = G::apply(&mut state, m) {
                    state = new_state;
                }
            }
            None => return None,
        }
        s = 1 - s;
    }
}
