use rand::rngs::SmallRng;
use rand::Rng;
use rand_core::SeedableRng;
use rand_distr::{Beta, Distribution};

use crate::game::Game;
use crate::game::PlayerIndex;
use crate::algorithms::Search;
use crate::util::random_best;

use std::marker::PhantomData;

/// Per-arm running statistics: how many rollouts an action has been given
/// and how many of those rollouts counted as a win (`rollout` returns a
/// positive reward). `mean()` is the empirical win rate, `[0, 1]`.
#[derive(Clone, Copy, Default)]
pub struct ArmStats {
    pub pulls: u32,
    pub wins: f64,
}

impl ArmStats {
    pub fn mean(&self) -> f64 {
        if self.pulls == 0 {
            0.
        } else {
            self.wins / self.pulls as f64
        }
    }
}

/// Chooses which arm to spend the next rollout on, given every arm's
/// current `(pulls, wins)` and the total rollouts spent so far across all
/// arms. Implementations own their own cold start: an unpulled arm
/// (`stats[i].pulls == 0`) should normally be at least as attractive as any
/// pulled one, since `mean()` reads `0.` for it -- there is no separate
/// "pull every arm once" warm-up phase, this is the only call that ever
/// picks an arm.
///
/// Object-safe so `mcts-tune` can select a policy at runtime from a
/// `bandit_policy` config string, the same way it boxes a `DynSelect<G>`
/// tree-search axis.
pub trait BanditPolicy {
    fn label(&self) -> &'static str;
    fn choose_arm(&self, stats: &[ArmStats], total_pulls: u32, rng: &mut SmallRng) -> usize;
}

/// Uniform allocation: every rollout goes to a uniformly random arm,
/// regardless of history. The literal "flat" Monte Carlo baseline --
/// spends the whole budget with no adaptive allocation at all, useful as a
/// control for how much the other policies actually buy.
#[derive(Clone, Copy, Default)]
pub struct Random;

impl BanditPolicy for Random {
    fn label(&self) -> &'static str {
        "random"
    }

    fn choose_arm(&self, stats: &[ArmStats], _total_pulls: u32, rng: &mut SmallRng) -> usize {
        rng.gen_range(0..stats.len())
    }
}

/// With probability `epsilon`, pull a uniformly random arm; otherwise pull
/// the best empirical mean so far (unpulled arms read `mean() == 0.`, same
/// as any arm that has only lost so far -- there's no separate forced
/// warm-up, so a low `epsilon` can leave some arms unpulled for a while).
#[derive(Clone, Copy)]
pub struct EpsilonGreedy {
    pub epsilon: f64,
}

impl Default for EpsilonGreedy {
    fn default() -> Self {
        Self { epsilon: 0.1 }
    }
}

impl BanditPolicy for EpsilonGreedy {
    fn label(&self) -> &'static str {
        "epsilon_greedy"
    }

    fn choose_arm(&self, stats: &[ArmStats], _total_pulls: u32, rng: &mut SmallRng) -> usize {
        if rng.gen::<f64>() < self.epsilon {
            return rng.gen_range(0..stats.len());
        }
        let indices: Vec<usize> = (0..stats.len()).collect();
        *random_best(&indices, rng, |&i| {
            if stats[i].pulls == 0 {
                f64::INFINITY
            } else {
                stats[i].mean()
            }
        })
        .unwrap()
    }
}

/// Classic UCB1 (Auer, Cesa-Bianchi, Fischer 2002): `mean + c *
/// sqrt(ln(total_pulls) / pulls)`. An unpulled arm scores `+inf`, so every
/// arm is guaranteed one pull before exploitation can dominate -- the same
/// effect the old code got from an explicit "spend `samples_per_move` on
/// every arm" pass, but adaptive past that first pull instead of fixed.
#[derive(Clone, Copy)]
pub struct Ucb1 {
    pub exploration_constant: f64,
}

impl Default for Ucb1 {
    fn default() -> Self {
        Self {
            exploration_constant: std::f64::consts::SQRT_2,
        }
    }
}

impl BanditPolicy for Ucb1 {
    fn label(&self) -> &'static str {
        "ucb1"
    }

    fn choose_arm(&self, stats: &[ArmStats], total_pulls: u32, rng: &mut SmallRng) -> usize {
        let indices: Vec<usize> = (0..stats.len()).collect();
        *random_best(&indices, rng, |&i| {
            let s = &stats[i];
            if s.pulls == 0 {
                f64::INFINITY
            } else {
                s.mean()
                    + self.exploration_constant
                        * ((total_pulls.max(1) as f64).ln() / s.pulls as f64).sqrt()
            }
        })
        .unwrap()
    }
}

/// Thompson sampling over a Beta-Bernoulli model: each arm's win rate has a
/// `Beta(alpha0 + wins, beta0 + losses)` posterior; every pull draws one
/// sample per arm from its posterior and plays the arm with the highest
/// draw. `alpha0 == beta0 == 1` (the default) is the uniform prior.
#[derive(Clone, Copy)]
pub struct ThompsonSampling {
    pub alpha0: f64,
    pub beta0: f64,
}

impl Default for ThompsonSampling {
    fn default() -> Self {
        Self {
            alpha0: 1.,
            beta0: 1.,
        }
    }
}

impl BanditPolicy for ThompsonSampling {
    fn label(&self) -> &'static str {
        "thompson"
    }

    fn choose_arm(&self, stats: &[ArmStats], _total_pulls: u32, rng: &mut SmallRng) -> usize {
        let mut best = 0;
        let mut best_draw = f64::NEG_INFINITY;
        for (i, s) in stats.iter().enumerate() {
            let losses = s.pulls as f64 - s.wins;
            let dist = Beta::new(self.alpha0 + s.wins, self.beta0 + losses)
                .expect("alpha0/beta0 and win/loss counts are always positive");
            let draw = dist.sample(rng);
            if draw > best_draw {
                best_draw = draw;
                best = i;
            }
        }
        best
    }
}

pub struct BanditStrategy<G: Game> {
    /// Total rollouts to spend choosing one move, shared adaptively across
    /// every legal action by `policy` -- the search's whole compute budget,
    /// not a per-action figure. Contrast the removed `samples_per_move`,
    /// which spent that many rollouts on *every* action unconditionally
    /// regardless of how hopeless it looked.
    pub budget: u32,
    pub max_rollout_depth: u32,
    pub policy: Box<dyn BanditPolicy + Send + Sync>,
    pub verbose: bool,
    pub game_type: PhantomData<G>,
    pub name: String,
    /// Rollout RNG. Seeded deterministically so repeated searches of the
    /// same position with the same configuration produce the same estimate;
    /// callers that want run-to-run variation pass a varying seed to
    /// [`BanditStrategy::with_seed`].
    rng: SmallRng,
}

impl<G: Game> BanditStrategy<G> {
    pub fn new() -> Self {
        Self {
            budget: 1000,
            max_rollout_depth: 100,
            policy: Box::new(Ucb1::default()),
            verbose: false,
            game_type: PhantomData,
            name: "bandit".into(),
            rng: SmallRng::seed_from_u64(0),
        }
    }

    /// Reseeds the rollout RNG. Two strategies built with the same seed and
    /// configuration play identically.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    pub fn set_budget(mut self, budget: u32) -> Self {
        self.budget = budget;
        self
    }

    pub fn set_max_rollout_depth(mut self, max_rollout_depth: u32) -> Self {
        self.max_rollout_depth = max_rollout_depth;
        self
    }

    pub fn set_policy(mut self, policy: Box<dyn BanditPolicy + Send + Sync>) -> Self {
        self.policy = policy;
        self
    }

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

impl<G: Game> Default for BanditStrategy<G> {
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

impl<G: Game + Sync + Send> Search for BanditStrategy<G> {
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
        let children: Vec<_> = actions
            .iter()
            .map(|m| G::apply(state.clone(), m))
            .collect();
        let mut stats = vec![ArmStats::default(); actions.len()];

        // The budget is a total across every arm, but every arm still needs
        // at least one pull to have a defined estimate at all.
        let budget = self.budget.max(actions.len() as u32);
        for total_pulls in 0..budget {
            let idx = self.policy.choose_arm(&stats, total_pulls, &mut self.rng);
            let r = rollout::<G>(self.max_rollout_depth, player, &children[idx], &mut self.rng);
            stats[idx].pulls += 1;
            if r > 0. {
                stats[idx].wins += 1.;
            }
        }

        if self.verbose {
            let mut ranked: Vec<usize> = (0..actions.len()).collect();
            ranked.sort_by(|&a, &b| stats[b].mean().partial_cmp(&stats[a].mean()).unwrap());
            eprintln!("Bandit ({}):", self.policy.label());
            for &i in ranked.iter().take(10) {
                let notation = G::notation(state, &actions[i]);
                eprintln!(
                    "- {:0.2}% {} ({}/{} wins)",
                    100. * stats[i].mean(),
                    notation,
                    stats[i].wins,
                    stats[i].pulls
                );
            }
        }

        // Final choice: the most-pulled arm (robust-child, same convention
        // as this codebase's `final_action: robust_child` for tree search),
        // tie-broken by mean, tie-broken randomly. Not the policy's own
        // `choose_arm` -- that call balances explore/exploit mid-search,
        // where an under-sampled arm can still look best by luck.
        let indices: Vec<usize> = (0..actions.len()).collect();
        let best = random_best(&indices, &mut self.rng, |&i| {
            stats[i].pulls as f64 + stats[i].mean() * 0.5
        })
        .unwrap();
        actions[*best].clone()
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

    fn strategy(seed: u64) -> BanditStrategy<Race> {
        BanditStrategy::<Race>::new()
            .set_budget(64 * 3)
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

    #[test]
    fn every_arm_gets_pulled_at_least_once_even_under_epsilon_zero_greedy() {
        // A budget smaller than the arm count still must not panic or leave
        // an arm's stats entirely undefined -- `choose_action` clamps the
        // effective budget up to `actions.len()`.
        let mut s = BanditStrategy::<Race>::new()
            .set_budget(1)
            .set_max_rollout_depth(8)
            .set_policy(Box::new(EpsilonGreedy { epsilon: 0. }))
            .with_seed(1);
        let _ = s.choose_action(&State::default());
    }

    #[test]
    fn all_policies_produce_a_legal_move() {
        let policies: Vec<Box<dyn BanditPolicy + Send + Sync>> = vec![
            Box::new(Random),
            Box::new(EpsilonGreedy::default()),
            Box::new(Ucb1::default()),
            Box::new(ThompsonSampling::default()),
        ];
        for policy in policies {
            let label = policy.label();
            let mut s = BanditStrategy::<Race>::new()
                .set_budget(30)
                .set_max_rollout_depth(16)
                .set_policy(policy)
                .with_seed(42);
            let state = State::default();
            let mut legal = Vec::new();
            Race::generate_actions(&state, &mut legal);
            let m = s.choose_action(&state);
            assert!(legal.contains(&m), "{label} chose an illegal move");
        }
    }
}
