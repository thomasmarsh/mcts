use rand::Rng;

type XorShiftRng = rand_xorshift::XorShiftRng;

use crate::game::Game;
use crate::strategies;

const PRIMES: [usize; 16] = [
    14323, 18713, 19463, 30553, 33469, 45343, 50221, 51991, 53201, 56923, 64891, 72763, 74471,
    81647, 92581, 94693,
];

pub(super) fn random_best<'a, T, F: Fn(&T) -> f32>(
    set: &'a [T],
    rng: &'a mut XorShiftRng,
    score_fn: F,
) -> Option<&'a T> {
    // To make the choice more uniformly random among the best moves,
    // start at a random offset and stride by a random amount.
    // The stride must be coprime with n, so pick from a set of 5 digit primes.

    let n = set.len();
    // Combine both random numbers into a single rng call.

    let r = rng.gen_range(0..n * PRIMES.len());
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
    G: Game,
    G::S: Default + Clone,
    S1: strategies::Strategy<G>,
    S2: strategies::Strategy<G>,
{
    let mut state = G::S::default();
    let mut strategies: [&mut dyn strategies::Strategy<G>; 2] = [s1, s2];
    let mut s = 0;
    loop {
        if G::is_terminal(&state) {
            let current_player = G::player_to_move(&state);
            let winner = G::winner(&state);
            return winner.map(|p| if current_player == p { s } else { 1 - s });
        }
        let strategy = &mut strategies[s];
        match strategy.choose_move(&state) {
            Some(m) => {
                state = G::apply(&state, m);
            }
            None => return None,
        }
        s = 1 - s;
    }
}

// Return a unique id for humans for this move.
pub(super) fn move_id<G: Game>(s: &<G as Game>::S, m: Option<<G as Game>::M>) -> String {
    if let Some(mov) = m {
        G::notation(s, &mov)
    } else {
        "none".to_string()
    }
}

pub(super) fn pv_string<G: Game>(path: &[<G as Game>::M], state: &<G as Game>::S) -> String
where
    <G as Game>::M: Clone,
    <G as Game>::S: Clone,
{
    let mut state = state.clone();
    let mut out = String::new();
    for (i, m) in (0..).zip(path.iter()) {
        if i > 0 {
            out.push_str("; ");
        }
        out.push_str(move_id::<G>(&state, Some(m.clone())).as_str());
        state = G::apply(&state, m.clone());
    }
    out
}
