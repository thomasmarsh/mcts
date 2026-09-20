//! Epsilon-greedy self-play with a 1-ply lookahead, feeding [`Learner`], plus
//! the greedy player and match helpers used to measure the result.
//!
//! No tree search is involved: a move is chosen by applying every legal action
//! and taking the successor the model likes best (from the mover's point of
//! view), or a uniformly random action with probability epsilon.

use std::sync::Arc;

use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::config::TrainConfig;
use crate::geometry::{random_walk_tuples, CellFeatures, Geometry};
use crate::td::Learner;
use crate::weights::Model;

type State<F> = <<F as CellFeatures>::G as Game>::S;
type Action<F> = <<F as CellFeatures>::G as Game>::A;

/// Result of a finished game for `viewer`: +1 win, 0 draw, -1 loss. Only call
/// on a terminal state.
pub fn terminal_value<G: Game>(state: &G::S, viewer: usize) -> f32 {
    match G::winner(state) {
        Some(p) if p.to_index() == viewer => 1.0,
        Some(_) => -1.0,
        None => 0.0,
    }
}

/// The value of each action for the player about to move: the successor's
/// exact result if the game ends, else the model's value of the successor
/// signed toward the mover.
fn action_values<F: CellFeatures>(
    feats: &F,
    model: &Model,
    state: &State<F>,
    actions: &[Action<F>],
    codes: &mut [u8],
    out: &mut Vec<f32>,
) {
    let actor = F::G::player_to_move(state).to_index();
    let spc = model.geometry().states_per_cell();
    out.clear();
    for a in actions {
        let next = F::G::apply(state.clone(), a);
        let v = if F::G::is_terminal(&next) {
            terminal_value::<F::G>(&next, actor)
        } else {
            feats.cell_codes(&next, spc, codes);
            let v = model.value(codes);
            if F::G::player_to_move(&next).to_index() == actor {
                v
            } else {
                -v
            }
        };
        out.push(v);
    }
}

fn first_argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

fn random_argmax(values: &[f32], rng: &mut SmallRng) -> usize {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ties = values.iter().filter(|&&v| v == max).count();
    let mut pick = rng.gen_range(0..ties);
    for (i, &v) in values.iter().enumerate() {
        if v == max {
            if pick == 0 {
                return i;
            }
            pick -= 1;
        }
    }
    unreachable!()
}

/// What one training episode produced, for logging.
#[derive(Clone, Copy, Debug, Default)]
pub struct EpisodeStats {
    pub moves: u32,
    pub explored: u32,
    pub td_count: u32,
    pub td_sum: f64,
    pub td_abs_sum: f64,
}

pub struct Trainer<F: CellFeatures> {
    feats: F,
    cfg: TrainConfig,
    learner: Learner,
    rng: SmallRng,
    codes: Vec<u8>,
    idx: Vec<u32>,
    actions: Vec<Action<F>>,
    values: Vec<f32>,
    episode: u64,
    stats: EpisodeStats,
}

impl<F: CellFeatures> Trainer<F> {
    /// Generate the random-walk geometry from `cfg.seed` and start from
    /// all-zero weights.
    pub fn new(feats: F, cfg: TrainConfig) -> Trainer<F> {
        assert!(
            (3..=feats.max_states_per_cell()).contains(&cfg.states_per_cell),
            "states_per_cell {} not supported by this game (3..={})",
            cfg.states_per_cell,
            feats.max_states_per_cell()
        );
        assert_eq!(F::G::num_players(), 2, "the trainer handles two-player games");
        let mut rng = SmallRng::seed_from_u64(cfg.seed);
        let neighbors: Vec<Vec<usize>> = (0..feats.num_cells()).map(|c| feats.neighbors(c)).collect();
        let tuples = random_walk_tuples(&neighbors, cfg.n_tuples, cfg.tuple_len, &mut rng);
        let geom = Geometry::from_tuples(cfg.states_per_cell, tuples, &feats.orientations());
        let codes = vec![0u8; feats.num_cells()];
        let learner = Learner::new(Model::zeros(geom), cfg.td_params(), 2);
        Trainer {
            feats,
            cfg,
            learner,
            rng,
            codes,
            idx: Vec::new(),
            actions: Vec::new(),
            values: Vec::new(),
            episode: 0,
            stats: EpisodeStats::default(),
        }
    }

    pub fn model(&self) -> &Model {
        self.learner.model()
    }

    pub fn learner(&self) -> &Learner {
        &self.learner
    }

    pub fn feats(&self) -> &F {
        &self.feats
    }

    pub fn config(&self) -> &TrainConfig {
        &self.cfg
    }

    pub fn episodes_done(&self) -> u64 {
        self.episode
    }

    /// A new non-terminal state `s` was reached (or the episode's first state,
    /// when no chain has anything pending): step the mover-to-be's chain toward
    /// the discounted value of `s`, then make `s` that chain's pending state.
    /// `explored` marks that the move into `s` was random.
    pub fn observe(&mut self, s: &State<F>, explored: bool) {
        let viewer = F::G::player_to_move(s).to_index();
        let spc = self.cfg.states_per_cell;
        self.feats.cell_codes(s, spc, &mut self.codes);
        let v = self.learner.model().value(&self.codes);
        if let Some(d) = self.learner.step(viewer, self.cfg.gamma * v) {
            self.stats.td_count += 1;
            self.stats.td_sum += d as f64;
            self.stats.td_abs_sum += d.abs() as f64;
        }
        self.learner
            .model()
            .geometry()
            .active_indices(&self.codes, &mut self.idx);
        self.learner.set_prev(viewer, &self.idx);
        if explored && self.cfg.reset_traces_on_explore {
            self.learner.clear_trace(viewer);
        }
    }

    /// The game ended in `terminal`. Each chain's pending state is stepped
    /// toward the true result for its own viewer: for the player whose chain
    /// the terminal state belongs to this is the last real update, for the
    /// other player it is the final-adaptation step, which without a next
    /// state of its own would otherwise never see the outcome.
    pub fn finish(&mut self, terminal: &State<F>) {
        for viewer in 0..2 {
            let result = terminal_value::<F::G>(terminal, viewer);
            if let Some(d) = self.learner.step(viewer, result) {
                self.stats.td_count += 1;
                self.stats.td_sum += d as f64;
                self.stats.td_abs_sum += d.abs() as f64;
            }
            self.learner.clear_trace(viewer);
        }
    }

    /// Play one self-play game from the game's default state, learning online.
    pub fn run_episode(&mut self) -> EpisodeStats {
        let eps = self.cfg.epsilon_at(self.episode);
        self.stats = EpisodeStats::default();
        let mut s = State::<F>::default();
        self.observe(&s, false);
        loop {
            self.actions.clear();
            F::G::generate_actions(&s, &mut self.actions);
            let n = self.actions.len();
            assert!(n > 0, "a non-terminal state must have an action");
            let random = self.rng.gen::<f32>() < eps;
            let explored = random && n > 1;
            let k = if random {
                self.rng.gen_range(0..n)
            } else {
                action_values(
                    &self.feats,
                    self.learner.model(),
                    &s,
                    &self.actions,
                    &mut self.codes,
                    &mut self.values,
                );
                random_argmax(&self.values, &mut self.rng)
            };
            s = F::G::apply(s, &self.actions[k]);
            self.stats.moves += 1;
            if explored {
                self.stats.explored += 1;
            }
            if F::G::is_terminal(&s) {
                self.finish(&s);
                break;
            }
            self.observe(&s, explored);
        }
        self.episode += 1;
        self.stats
    }
}

/// Greedy 1-ply player over a trained model: the [`Search`] adapter that lets
/// any harness gate the raw agent. Deterministic (first best action).
pub struct GreedyPlayer<F: CellFeatures> {
    feats: F,
    model: Arc<Model>,
    name: String,
    codes: Vec<u8>,
    actions: Vec<Action<F>>,
    values: Vec<f32>,
}

impl<F: CellFeatures> GreedyPlayer<F> {
    pub fn new(feats: F, model: Arc<Model>) -> GreedyPlayer<F> {
        let codes = vec![0u8; feats.num_cells()];
        GreedyPlayer {
            feats,
            model,
            name: "ntuple-greedy".to_string(),
            codes,
            actions: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn best_action(&mut self, state: &State<F>) -> Action<F> {
        self.actions.clear();
        F::G::generate_actions(state, &mut self.actions);
        if self.actions.len() == 1 {
            return self.actions[0].clone();
        }
        action_values(
            &self.feats,
            &self.model,
            state,
            &self.actions,
            &mut self.codes,
            &mut self.values,
        );
        self.actions[first_argmax(&self.values)].clone()
    }
}

impl<F: CellFeatures> Search for GreedyPlayer<F> {
    type G = F::G;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &State<F>) -> Action<F> {
        self.best_action(state)
    }
}

/// Wins, draws and losses of the model's greedy player.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatchRecord {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

impl MatchRecord {
    pub fn games(&self) -> u32 {
        self.wins + self.draws + self.losses
    }

    /// Wins plus half of the draws, over games played.
    pub fn score(&self) -> f64 {
        (self.wins as f64 + 0.5 * self.draws as f64) / self.games().max(1) as f64
    }
}

/// Play `games` games of the model's greedy player against `opponent`, from
/// the game's default state, alternating which colour the model plays. The
/// opponent picks an index into the legal actions it is handed.
pub fn play_match<F: CellFeatures>(
    feats: &F,
    model: &Model,
    games: u32,
    seed: u64,
    mut opponent: impl FnMut(&State<F>, &[Action<F>], &mut SmallRng) -> usize,
) -> MatchRecord {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut codes = vec![0u8; feats.num_cells()];
    let mut actions: Vec<Action<F>> = Vec::new();
    let mut values = Vec::new();
    let mut rec = MatchRecord::default();
    for game in 0..games {
        let agent = (game % 2) as usize;
        let mut s = State::<F>::default();
        while !F::G::is_terminal(&s) {
            actions.clear();
            F::G::generate_actions(&s, &mut actions);
            let k = if F::G::player_to_move(&s).to_index() == agent {
                if actions.len() == 1 {
                    0
                } else {
                    action_values(feats, model, &s, &actions, &mut codes, &mut values);
                    first_argmax(&values)
                }
            } else {
                opponent(&s, &actions, &mut rng)
            };
            s = F::G::apply(s, &actions[k]);
        }
        match terminal_value::<F::G>(&s, agent) {
            v if v > 0.0 => rec.wins += 1,
            v if v < 0.0 => rec.losses += 1,
            _ => rec.draws += 1,
        }
    }
    rec
}

/// Opponent that plays a uniformly random legal action.
pub fn uniform_random<S, A>(_: &S, actions: &[A], rng: &mut SmallRng) -> usize {
    rng.gen_range(0..actions.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TttCells;
    use game_ttt::{Piece, TicTacToe};

    fn cfg(seed: u64, episodes: u64) -> TrainConfig {
        TrainConfig {
            seed,
            episodes,
            n_tuples: 12,
            tuple_len: 4,
            states_per_cell: 3,
            alpha: 0.2,
            lambda: 0.5,
            gamma: 1.0,
            epsilon_start: 0.3,
            epsilon_end: 0.1,
            tcl: true,
            reset_traces_on_explore: true,
            trace_cutoff: 1e-4,
            log_every: 100,
            eval_games: 100,
            out_dir: String::new(),
        }
    }

    fn state_after(moves: &[u8]) -> Vec<<TicTacToe as Game>::S> {
        let mut s = <TicTacToe as Game>::S::default();
        let mut out = vec![s];
        for &m in moves {
            s = TicTacToe::apply(s, &game_ttt::Move(m));
            out.push(s);
        }
        out
    }

    #[test]
    fn same_seed_gives_bit_identical_weights_and_a_different_seed_does_not() {
        let run = |seed| {
            let mut t = Trainer::new(TttCells, cfg(seed, 60));
            for _ in 0..60 {
                t.run_episode();
            }
            t.model().weights().to_vec()
        };
        let a = run(3);
        assert_eq!(a, run(3));
        assert_ne!(a, run(4));
        assert!(a.iter().any(|&w| w != 0.0), "training changed the weights");
    }

    #[test]
    fn final_adaptation_carries_the_result_to_both_players() {
        // X plays 0, O 3, X 1, O 4, X 2 and wins on the top row. The terminal
        // state belongs to O's chain (O is to move); X's chain has nothing after
        // its last state and only learns the +1 through the final-adaptation step.
        let states = state_after(&[0, 3, 1, 4, 2]);
        let mut t = Trainer::new(TttCells, cfg(1, 1));
        for s in &states[..states.len() - 1] {
            t.observe(s, false);
        }
        assert!(t.learner().pending(Piece::X.to_index()));
        assert!(t.learner().pending(Piece::O.to_index()));
        t.finish(states.last().unwrap());
        assert!(!t.learner().pending(0) && !t.learner().pending(1));
        assert_eq!(t.learner().trace_len(0) + t.learner().trace_len(1), 0);

        let value = |s: &<TicTacToe as Game>::S| {
            let mut codes = vec![0u8; 9];
            TttCells.cell_codes(s, 3, &mut codes);
            t.model().value(&codes)
        };
        // Last X-to-move state before the win (X to move after O's 4th move).
        assert!(value(&states[4]) > 0.0, "the winner's last state moved up");
        // Last O-to-move state (after X's 3rd move... states[3]).
        assert!(value(&states[3]) < 0.0, "the loser's last state moved down");
    }

    #[test]
    fn training_beats_random_play_at_tic_tac_toe_and_the_player_is_deterministic() {
        let mut t = Trainer::new(TttCells, cfg(11, 1500));
        for _ in 0..1500 {
            t.run_episode();
        }
        let model = Arc::new(t.model().clone());
        let rec = play_match(&TttCells, &model, 200, 5, uniform_random);
        assert!(rec.score() > 0.85, "score {} record {rec:?}", rec.score());

        let mut p = GreedyPlayer::new(TttCells, model.clone());
        let mut q = GreedyPlayer::new(TttCells, model);
        let s = <TicTacToe as Game>::S::default();
        assert_eq!(p.choose_action(&s), q.choose_action(&s));
    }

    #[test]
    fn epsilon_falls_linearly_over_the_run() {
        let c = cfg(1, 101);
        assert!((c.epsilon_at(0) - 0.3).abs() < 1e-6);
        assert!((c.epsilon_at(50) - 0.2).abs() < 1e-6);
        assert!((c.epsilon_at(100) - 0.1).abs() < 1e-6);
    }
}
