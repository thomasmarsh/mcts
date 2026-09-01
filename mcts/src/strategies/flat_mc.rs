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
    /// Rollout RNG. Seeded deterministically so repeated searches of the
    /// same position with the same configuration produce the same estimate;
    /// callers that want run-to-run variation pass a varying seed to
    /// [`FlatMonteCarloStrategy::with_seed`].
    rng: SmallRng,
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
            rng: SmallRng::seed_from_u64(0),
        }
    }

    /// Reseeds the rollout RNG. Two strategies built with the same seed and
    /// configuration play identically.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    pub fn set_samples_per_move(mut self, samples_per_move: u32) -> Self {
        self.samples_per_move = samples_per_move;
        self
    }

    pub fn set_max_rollout_depth(mut self, max_rollout_depth: u32) -> Self {
        self.max_rollout_depth = max_rollout_depth;
        self
    }

    /// `Some(c)` selects the UCB1 move-selection rule with exploration
    /// constant `c`; `None` selects plain win-rate.
    pub fn set_ucb1(mut self, ucb1: Option<f64>) -> Self {
        self.ucb1 = ucb1;
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

        let player = G::player_to_move(state).to_index();

        let mut actions = Vec::new();
        G::generate_actions(state, &mut actions);
        let mut wins = Vec::with_capacity(actions.len());
        for m in &actions {
            let tmp = G::apply(state.clone(), m);
            let mut n = 0;
            for _ in 0..self.samples_per_move {
                if rollout::<G>(self.max_rollout_depth, player, &tmp, &mut self.rng) > 0. {
                    n += 1;
                }
            }
            wins.push((n, m.clone()));
        }

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

        let samples_per_move = self.samples_per_move;
        if let Some(c) = self.ucb1 {
            random_best(wins.as_slice(), &mut self.rng, |x| {
                ucb1(x.0 as f64, samples_per_move as f64, c)
            })
            .map(|x| x.1.clone())
            .unwrap()
        } else {
            random_best(wins.as_slice(), &mut self.rng, |x| x.0 as f64)
                .map(|x| x.1.clone())
                .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PlayerIndex;

    /// Race to exactly `TARGET`: each turn a player adds 1, 2, or 3; landing
    /// on `TARGET` wins, overshooting loses. Short enough that flat Monte
    /// Carlo's uniformly-random rollouts drive the estimate, so the move it
    /// picks is a direct function of its RNG stream.
    const TARGET: u8 = 12;

    #[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
    struct State {
        total: u8,
        plies: u8,
    }

    impl std::fmt::Display for State {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.total)
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize)]
    struct Step(u8);

    #[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
    struct Player(usize);
    impl PlayerIndex for Player {
        fn to_index(&self) -> usize {
            self.0
        }
    }

    #[derive(Clone)]
    struct Race;

    impl Game for Race {
        type S = State;
        type A = Step;
        type P = Player;

        fn apply(state: Self::S, action: &Self::A) -> Self::S {
            State {
                total: state.total + action.0,
                plies: state.plies + 1,
            }
        }

        fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
            if state.total >= TARGET {
                return;
            }
            actions.extend([Step(1), Step(2), Step(3)]);
        }

        fn winner(state: &Self::S) -> Option<Self::P> {
            if state.total < TARGET {
                return None;
            }
            // The player who just moved is `(plies - 1) % 2`; they win on an
            // exact landing and lose on an overshoot.
            let mover = (state.plies as usize + 1) % 2;
            Some(Player(if state.total == TARGET { mover } else { 1 - mover }))
        }

        fn player_to_move(state: &Self::S) -> Self::P {
            Player((state.plies % 2) as usize)
        }
    }

    fn strategy(seed: u64) -> FlatMonteCarloStrategy<Race> {
        FlatMonteCarloStrategy::<Race>::new()
            .set_samples_per_move(64)
            .set_max_rollout_depth(32)
            .with_seed(seed)
    }

    #[test]
    fn same_seed_plays_identically_across_constructions() {
        let moves = |seed| {
            let mut s = strategy(seed);
            let mut state = State::default();
            let mut seq = Vec::new();
            while !Race::is_terminal(&state) {
                let m = s.choose_action(&state);
                seq.push(m);
                state = Race::apply(state, &m);
            }
            seq
        };
        assert_eq!(moves(7), moves(7), "a fixed seed must be reproducible");
        assert_eq!(moves(0), moves(0));
    }

    #[test]
    fn choose_action_does_not_reseed_per_call() {
        // Two calls on the same strategy consume distinct RNG state; a fresh
        // same-seed strategy reproduces the first call, not the second.
        let state = State::default();
        let mut a = strategy(3);
        let first = a.choose_action(&state);
        let next_state = Race::apply(state, &first);
        let _second = a.choose_action(&next_state);

        let mut b = strategy(3);
        assert_eq!(b.choose_action(&state), first);
    }
}
