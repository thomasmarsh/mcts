use rand::Rng;

use rand::rngs::SmallRng;

use crate::game::{Game, PlayerIndex};
use crate::strategies;

use crate::strategies::Search;
use rayon::prelude::*;
use std::ops::AddAssign;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Clone)]
pub struct AnySearch<'a, G: Game + Clone>(pub Arc<Mutex<Box<dyn strategies::Search<G = G> + 'a>>>);

impl<'a, G> AnySearch<'a, G>
where
    G: Game + Clone,
{
    pub fn new<S: strategies::Search<G = G> + 'a>(search: S) -> Self {
        Self(Arc::new(Mutex::new(Box::new(search))))
    }
}

impl<'a, G: Game + Clone> strategies::Search for AnySearch<'a, G> {
    type G = G;

    fn friendly_name(&self) -> String {
        self.0.lock().unwrap().friendly_name()
    }

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A {
        self.0.lock().unwrap().choose_action(state)
    }
}

const PRIMES: [usize; 16] = [
    14323, 18713, 19463, 30553, 33469, 45343, 50221, 51991, 53201, 56923, 64891, 72763, 74471,
    81647, 92581, 94693,
];

#[inline]
pub(super) fn random_best<'a, T, F: Fn(&T) -> f64>(
    set: &'a [T],
    rng: &'a mut SmallRng,
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

    let mut best_score = f64::NEG_INFINITY;
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
    S1: strategies::Search<G = G>,
    S2: strategies::Search<G = G>,
{
    let mut state = G::S::default();
    let mut strategies: [&mut dyn strategies::Search<G = G>; 2] = [s1, s2];
    let mut s = 0;
    loop {
        if G::is_terminal(&state) {
            let current_player = G::player_to_move(&state);
            let winner = G::winner(&state);
            return winner.map(|p| {
                if current_player.to_index() == p.to_index() {
                    s
                } else {
                    1 - s
                }
            });
        }
        let strategy = &mut strategies[s];
        let m = strategy.choose_action(&state);
        state = G::apply(state, &m);
        s = 1 - s;
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Result {
    wins: usize,
    losses: usize,
    draws: usize,
}

use std::ops::Add;

impl Add for Result {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Result {
            wins: self.wins + rhs.wins,
            losses: self.losses + rhs.losses,
            draws: self.draws + rhs.draws,
        }
    }
}

impl AddAssign for Result {
    fn add_assign(&mut self, rhs: Self) {
        self.wins += rhs.wins;
        self.losses += rhs.losses;
        self.draws += rhs.draws;
    }
}

/// Play a round-robin tournament with the provided strategies.
fn round_robin<G>(strategies: &mut Vec<AnySearch<'_, G>>, init: &G::S) -> Vec<Result>
where
    G: Game + Clone,
    G::S: Sync,
{
    let mut pairs = Vec::new();
    for i in 0..strategies.len() {
        for j in 0..strategies.len() {
            if i != j {
                pairs.push((i, j));
            }
        }
    }
    let len = pairs.len();

    use indicatif::ParallelProgressIterator;

    pairs
        .into_par_iter()
        .map(|(i, j)| {
            let mut results = vec![Result::default(); strategies.len()];
            let mut si = strategies[i].clone();
            let mut sj = strategies[j].clone();
            println!("{} vs. {}", si.friendly_name(), sj.friendly_name(),);
            let mut state = init.clone();
            let mut current_strategy = 0;

            let players = [i, j];

            loop {
                if G::is_terminal(&state) {
                    match G::winner(&state) {
                        None => {
                            results[i].draws += 1;
                            results[j].draws += 1;
                        }
                        Some(p) => {
                            let winner = players[p.to_index()];
                            let loser = players[1 - p.to_index()];

                            results[winner].wins += 1;
                            results[loser].losses += 1;
                        }
                    }
                    break;
                }

                let index = players[current_strategy];
                let action = if index == i {
                    si.choose_action(&state)
                } else {
                    sj.choose_action(&state)
                };

                state = G::apply(state, &action);
                current_strategy = 1 - current_strategy;
            }
            println!("finish: {} vs. {}", si.friendly_name(), sj.friendly_name());
            results
        })
        .progress_count(len as u64)
        .reduce_with(|acc, x| {
            acc.into_iter()
                .zip(x.iter())
                .map(|(r1, r2)| r1 + *r2)
                .collect()
        })
        .unwrap_or_else(|| panic!())
}

/// Play a round-robin tournament multiple times with the provided strategies.
pub fn round_robin_multiple<G, S>(
    strategies: &mut Vec<AnySearch<'_, G>>,
    rounds: usize,
    init: &G::S,
) -> Vec<Result>
where
    G: Game + Clone,
    S: strategies::Search<G = G>,
{
    let mut results = vec![Result::default(); strategies.len()];

    for _ in 0..rounds {
        for (index, result) in round_robin::<G>(strategies, init).iter().enumerate() {
            results[index] += *result;

            println!("=======================================");
            println!(
                "{}: wins={}, losses={}, draws={}",
                strategies[index].friendly_name(),
                results[index].wins,
                results[index].losses,
                results[index].draws,
            );
        }
    }

    results
}

/*

use itertools::Itertools;

/// Perform a parameter grid search for a given strategy.
/// Returns the best parameters and their corresponding performance.
pub fn parameter_grid_search<G, S>(
    base_strategy: &mut S,
    parameter_values: Vec<Vec<S::Param>>,
) -> (Vec<S::Param>, usize)
where
    G: Game,
    G::S: Default + Clone,
    S: strategies::Search<G = G>,
{
    let mut best_params = Vec::new();
    let mut best_score = 0;

    for params in parameter_values.into_iter().multi_cartesian_product() {
        let mut strategy = base_strategy.clone_with_params(params.clone());

        let score = evaluate_strategy::<G, S>(&mut strategy);

        if score > best_score {
            best_score = score;
            best_params = params;
        }
    }

    (best_params, best_score)
}

/// Evaluate a strategy's performance using a round-robin tournament.
fn evaluate_strategy<G, S>(strategy: &mut S) -> usize
where
    G: Game,
    G::S: Default + Clone,
    S: strategies::Search<G = G>,
{
    let mut wins = 0;
    let mut num_games = 0;

    for _ in 0..NUM_ROUNDS {
        let mut opponent_strategy = S::default();
        let winner = battle_royale::<G, _, _>(strategy, &mut opponent_strategy);

        if let Some(w) = winner {
            if w == 0 {
                wins += 1;
            }
            num_games += 1;
        }
    }

    wins * 100 / num_games // Assuming 100 is the maximum possible score
}

const NUM_ROUNDS: usize = 100; // Number of rounds for evaluation
*/

// Return a unique id for humans for this move.
pub(super) fn move_id<G: Game>(s: &<G as Game>::S, m: Option<<G as Game>::A>) -> String {
    if let Some(mov) = m {
        G::notation(s, &mov)
    } else {
        "none".to_string()
    }
}

pub(super) fn pv_string<G: Game>(path: &[<G as Game>::A], state: &<G as Game>::S) -> String {
    let mut state = state.clone();
    let mut out = String::new();
    for (i, m) in (0..).zip(path.iter()) {
        if i > 0 {
            out.push_str("; ");
        }
        out.push_str(move_id::<G>(&state, Some(m.clone())).as_str());
        state = G::apply(state, &m);
    }
    out
}
