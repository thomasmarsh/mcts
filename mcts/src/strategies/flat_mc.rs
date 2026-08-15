use rand::rngs::SmallRng;
use rand::Rng;
use rand_core::SeedableRng;

use crate::game::Game;
use crate::game::PlayerIndex;
use crate::strategies::Search;
use crate::util::random_best;

use std::marker::PhantomData;

pub struct FlatMonteCarloStrategy<G: Game> {
    pub samples_per_move: u32, // TODO: also suppose samples per state
    pub max_rollout_depth: u32,
    pub max_rollouts: u32,
    pub verbose: bool,
    pub game_type: PhantomData<G>,
    pub ucb1: Option<f64>,
    pub name: String,
}

impl<G: Game> FlatMonteCarloStrategy<G> {
    pub fn new() -> Self {
        Self {
            samples_per_move: 100,
            max_rollout_depth: 100,
            max_rollouts: u32::MAX,
            verbose: false,
            game_type: PhantomData,
            ucb1: None,
            name: "flat_mc".into(),
        }
    }

    pub fn set_samples_per_move(mut self, samples_per_move: u32) -> Self {
        self.samples_per_move = samples_per_move;
        self
    }

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

impl<G: Game> Default for FlatMonteCarloStrategy<G> {
    fn default() -> Self {
        Self::new()
    }
}

/// Rolls out a uniformly-random playout from `init_state` and returns the
/// reward for `player` (the index of the player who is choosing the move
/// this rollout is scoring, *not* `player_to_move(init_state)` -- `init_state`
/// here is already the state *after* that move, so its own mover is the
/// opponent, and `Game::get_reward`'s init-relative convention would score
/// the opponent's outcome instead of the candidate move's own).
fn rollout<G: Game>(
    max_rollout_depth: u32,
    player: usize,
    init_state: &G::S,
    rng: &mut SmallRng,
) -> f64
where
    G::S: Clone,
{
    let mut state = init_state.clone();
    let mut actions = Vec::new();
    for _ in 0..max_rollout_depth {
        if G::is_terminal(&state) {
            return G::compute_utilities(&state)[player];
        }
        actions.clear();
        G::generate_actions(&state, &mut actions);
        if actions.is_empty() {
            return 0.;
        }
        let m = actions[rng.gen_range(0..actions.len())].clone();

        state = G::apply(state, &m);
    }
    0.
}

impl<G: Game + Sync + Send> Search for FlatMonteCarloStrategy<G> {
    type G = G;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.into();
    }

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A {
        if G::is_terminal(state) {
            panic!();
        }

        let mut rng = SmallRng::from_entropy();
        let player = G::player_to_move(state).to_index();

        let mut actions = Vec::new();
        G::generate_actions(state, &mut actions);
        let wins = actions
            .iter()
            .map(|m| {
                let mut tmp = state.clone();
                let new_state = G::apply(tmp, m);
                tmp = new_state;
                let mut n = 0;
                for _ in 0..self.samples_per_move {
                    let result = rollout::<G>(self.max_rollout_depth, player, &tmp, &mut rng);
                    if result > 0. {
                        n += 1;
                    }
                }
                (n, m.clone())
            })
            .collect::<Vec<_>>();

        if self.verbose {
            let mut w = wins.clone();
            w.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprintln!("Flat MC:");
            for (n, m) in w.into_iter().take(10) {
                let pct = 100. * (n as f64 / self.samples_per_move as f64);
                let notation = G::notation(state, &m);
                eprintln!(
                    "- {:0.2}% {} ({}/{} wins)",
                    pct, notation, n, self.samples_per_move
                );
            }
        }

        let ucb1 = |w: f64, n: f64, c: f64| w / n + c * (n.ln() / n);

        if let Some(c) = self.ucb1 {
            random_best(wins.as_slice(), &mut rng, |x| {
                ucb1(x.0 as f64, self.samples_per_move as f64, c)
            })
            .map(|x| x.1.clone())
            .unwrap()
        } else {
            random_best(wins.as_slice(), &mut rng, |x| x.0 as f64)
                .map(|x| x.1.clone())
                .unwrap()
        }
    }
}
