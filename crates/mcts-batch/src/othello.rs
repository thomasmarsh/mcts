//! The concrete [`EnvOracle`] for Othello, wired to whichever evaluator
//! already implements both `mcts::evaluator::Evaluator` and
//! `mcts::algorithms::mcts::policy::PolicyLogits` -- the same two traits
//! `game_othello::selfplay::CnnGumbelPlayer<E>` is already generic over, so
//! `game_othello::convnet::CnnValueNet` (CPU) or
//! `game_othello::convnet::mlx::MlxCnnValueNet` plug in unchanged.
//!
//! `PolicyLogits::logits` takes `&mut self` even though neither existing
//! Othello evaluator actually mutates anything on a call (the signature is
//! generic over strategies that do, e.g. ones with an internal cache).
//! [`OthelloOracle`] evaluates every state in a batch with `rayon`'s
//! `map_init` -- one cloned `E` per worker thread (not per state; `E::
//! clone` is cheap relative to a forward pass but not free, so cloning once
//! per thread rather than once per state matters at real batch sizes) --
//! spreading the whole batch's CPU-bound forward passes across every core
//! this machine has, rather than the one core a purely sequential
//! evaluation loop would use.

use game_othello::convnet::mlx::MlxCnnValueNet;
use game_othello::{Move, Othello, State};
use mcts::algorithms::mcts::policy::PolicyLogits;
use mcts::evaluator::{Evaluator, EVAL_MAGNITUDE_LIMIT};
use mcts::game::Game;
use rayon::prelude::*;

use crate::oracle::{EnvOracle, StepOutput, TransitionOutput};

/// 64 board squares plus `Move::PASS` -- Othello's action ids map directly
/// onto `Move(u8)`'s own encoding (`aid == move.0`), so no separate
/// action-id table is needed.
pub const NUM_ACTIONS: usize = 65;

pub struct OthelloOracle<E> {
    net: E,
}

impl<E> OthelloOracle<E> {
    pub fn new(net: E) -> Self {
        OthelloOracle { net }
    }
}

/// The real game outcome from `state.turn`'s perspective -- `+1`/`-1`/`0`,
/// the standard negamax terminal convention (matches
/// `mcts::game::TerminalStatus::utilities`'s own win/draw/loss values).
/// Used instead of a net evaluation at a terminal leaf: the outcome is
/// exactly known there, and a CNN forward pass on a full/passed-out board
/// has no reason to agree with it.
fn terminal_value(state: &State) -> f32 {
    match Othello::winner(state) {
        Some(w) if w == state.turn => 1.0,
        Some(_) => -1.0,
        None => 0.0,
    }
}

/// Evaluate one state: `(terminal, valid_actions[NUM_ACTIONS],
/// policy_prior[NUM_ACTIONS], value_prior)`. `policy_prior` is a softmax
/// over `net.logits`' legal-action scores (unmasked entries are `0.0` and
/// `search::validate_prior` re-masks/renormalizes anyway, but computing the
/// softmax over only the legal actions here avoids an illegal action's
/// arbitrary logit value skewing the normalization the way including it in
/// the softmax denominator would).
fn evaluate_state<E: Evaluator<Othello> + PolicyLogits<Othello>>(
    net: &mut E,
    state: &State,
) -> (bool, Vec<bool>, Vec<f32>, f32) {
    if Othello::is_terminal(state) {
        return (true, vec![false; NUM_ACTIONS], vec![0.0; NUM_ACTIONS], terminal_value(state));
    }
    let mut actions = Vec::new();
    Othello::generate_actions(state, &mut actions);
    let logits = net.logits(state, &actions);
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f64 = exp.iter().sum();

    let mut valid = vec![false; NUM_ACTIONS];
    let mut prior = vec![0.0f32; NUM_ACTIONS];
    for (i, &mv) in actions.iter().enumerate() {
        let aid = mv.0 as usize;
        valid[aid] = true;
        prior[aid] = (exp[i] / sum) as f32;
    }
    let value = net.evaluate(state) as f32 / EVAL_MAGNITUDE_LIMIT as f32;
    (false, valid, prior, value)
}

/// Evaluate a whole batch of states in parallel, one cloned `E` per worker
/// thread, and flatten the per-state `(bool, Vec<bool>, Vec<f32>, f32)`
/// tuples `evaluate_state` returns into the flat `(terminal, valid_actions,
/// policy_prior, value_prior)` arrays `StepOutput` expects.
fn evaluate_batch<E: Evaluator<Othello> + PolicyLogits<Othello> + Clone + Sync>(
    net: &E,
    states: &[State],
) -> (Vec<bool>, Vec<bool>, Vec<f32>, Vec<f32>) {
    let results: Vec<(bool, Vec<bool>, Vec<f32>, f32)> =
        states.par_iter().map_init(|| net.clone(), |local, state| evaluate_state(local, state)).collect();

    let n = results.len();
    let mut terminal = Vec::with_capacity(n);
    let mut valid_actions = Vec::with_capacity(n * NUM_ACTIONS);
    let mut policy_prior = Vec::with_capacity(n * NUM_ACTIONS);
    let mut value_prior = Vec::with_capacity(n);
    for (t, v, p, val) in results {
        terminal.push(t);
        valid_actions.extend(v);
        policy_prior.extend(p);
        value_prior.push(val);
    }
    (terminal, valid_actions, policy_prior, value_prior)
}

impl<E: Evaluator<Othello> + PolicyLogits<Othello> + Clone + Sync> EnvOracle<State> for OthelloOracle<E> {
    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }

    fn init(&self, envs: &[State]) -> StepOutput<State> {
        let (terminal, valid_actions, policy_prior, value_prior) = evaluate_batch(&self.net, envs);
        StepOutput { states: envs.to_vec(), terminal, valid_actions, policy_prior, value_prior }
    }

    fn transition(&self, states: &[State], actions: &[u16]) -> TransitionOutput<State> {
        let out_states: Vec<State> =
            states.iter().zip(actions).map(|(s, &aid)| Othello::apply(*s, &Move(aid as u8))).collect();
        let (terminal, valid_actions, policy_prior, value_prior) = evaluate_batch(&self.net, &out_states);
        TransitionOutput {
            step: StepOutput { states: out_states, terminal, valid_actions, policy_prior, value_prior },
            // `State::apply` always advances `turn`, PASS included, so the
            // mover switches on every transition -- no intermediate reward
            // in Othello either.
            rewards: vec![0.0; states.len()],
            player_switched: vec![true; states.len()],
        }
    }
}

/// Evaluate a whole batch of states with MLX calls of at most `chunk_size`
/// states each (`game_othello::convnet::mlx::evaluate_batch`), rather than
/// `OthelloOracle<E>`'s `rayon`-across-CPU-threads `map_init` -- routing
/// GPU work through 8 CPU threads each opening their own MLX stream would
/// just serialize on the same physical GPU (see `convnet::mlx`'s own
/// thread-local-stream docs), so this oracle stays single-threaded on the
/// CPU side. `chunk_size` bounds a single MLX call's transient working set,
/// which scales with `states.len() * CHANNELS` steeply enough that handing
/// the whole live batch to one call can exceed a real machine's memory long
/// before the net itself looks unreasonably large (measured: a 128-channel/
/// 6-block net's single-call peak went from ~1.8GB at 50 states to ~7.1GB
/// at 200 on an 8GB machine).
fn evaluate_batch_mlx(net: &MlxCnnValueNet, states: &[State], chunk_size: usize) -> (Vec<bool>, Vec<bool>, Vec<f32>, Vec<f32>) {
    let n = states.len();
    let mut terminal = vec![false; n];
    let mut valid_actions = vec![false; n * NUM_ACTIONS];
    let mut policy_prior = vec![0.0f32; n * NUM_ACTIONS];
    let mut value_prior = vec![0.0f32; n];

    let mut live: Vec<usize> = Vec::with_capacity(n);
    let mut actions_per_state: Vec<Vec<Move>> = Vec::new();
    for (i, s) in states.iter().enumerate() {
        if Othello::is_terminal(s) {
            terminal[i] = true;
            value_prior[i] = terminal_value(s);
            continue;
        }
        live.push(i);
        let mut actions = Vec::new();
        Othello::generate_actions(s, &mut actions);
        actions_per_state.push(actions);
    }

    let live_states: Vec<State> = live.iter().map(|&i| states[i]).collect();
    let (values, policies) = game_othello::convnet::mlx::evaluate_batch(net.inner(), &live_states, chunk_size);

    for (row, &i) in live.iter().enumerate() {
        value_prior[i] = values[row];
        let all = &policies[row];
        // Same PASS-as-mean-of-the-64-square-logits convention as
        // `CnnValueNet`/`MlxCnnValueNet`'s own `PolicyLogits::logits`
        // (`all` here has no dedicated PASS slot -- it's the raw 64
        // per-square logits `all_policy_logits`/`evaluate_batch` return).
        let logit_of = |mv: Move| {
            if mv == Move::PASS { all.iter().sum::<f64>() / all.len() as f64 } else { all[mv.0 as usize] }
        };
        // Softmax over legal-action logits only, same as `evaluate_state`'s
        // CPU convention -- storing raw logits here would leave the
        // (common, all-zero-weight test/throughput) case where every legal
        // logit is exactly 0.0 with a literal all-zero prior, which
        // `search::validate_prior` then treats as "no signal" and masks
        // out entirely instead of falling back to a uniform distribution.
        let actions = &actions_per_state[row];
        let max = actions.iter().map(|&mv| logit_of(mv)).fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = actions.iter().map(|&mv| (logit_of(mv) - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        for (k, &mv) in actions.iter().enumerate() {
            let aid = mv.0 as usize;
            valid_actions[i * NUM_ACTIONS + aid] = true;
            policy_prior[i * NUM_ACTIONS + aid] = (exp[k] / sum) as f32;
        }
    }
    (terminal, valid_actions, policy_prior, value_prior)
}

/// The GPU-backed [`EnvOracle`] for Othello: same shape as [`OthelloOracle`]
/// but wired to [`MlxCnnValueNet`] through the batched `convnet::mlx::
/// evaluate_batch` entry point instead of the generic `Evaluator`/
/// `PolicyLogits` traits' one-state-at-a-time contract, so a whole
/// simulation round's live batch becomes a handful of GPU calls (bounded by
/// `chunk_size`, see [`evaluate_batch_mlx`]) instead of one call per state
/// spread across CPU threads.
pub struct MlxOthelloOracle {
    net: MlxCnnValueNet,
    chunk_size: usize,
}

impl MlxOthelloOracle {
    /// `chunk_size` bounds a single MLX call's live batch -- pick it small
    /// enough that `chunk_size` states at this net's geometry fit
    /// comfortably in the running machine's memory (see [`evaluate_batch_mlx`]'s
    /// own docs for measured examples); a too-large value leaves the same
    /// out-of-memory risk this parameter exists to bound, while a
    /// too-small one pays more, smaller GPU calls than necessary.
    pub fn new(net: MlxCnnValueNet, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be positive");
        MlxOthelloOracle { net, chunk_size }
    }
}

impl EnvOracle<State> for MlxOthelloOracle {
    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }

    fn init(&self, envs: &[State]) -> StepOutput<State> {
        let (terminal, valid_actions, policy_prior, value_prior) = evaluate_batch_mlx(&self.net, envs, self.chunk_size);
        StepOutput { states: envs.to_vec(), terminal, valid_actions, policy_prior, value_prior }
    }

    fn transition(&self, states: &[State], actions: &[u16]) -> TransitionOutput<State> {
        let out_states: Vec<State> =
            states.iter().zip(actions).map(|(s, &aid)| Othello::apply(*s, &Move(aid as u8))).collect();
        let (terminal, valid_actions, policy_prior, value_prior) =
            evaluate_batch_mlx(&self.net, &out_states, self.chunk_size);
        TransitionOutput {
            step: StepOutput { states: out_states, terminal, valid_actions, policy_prior, value_prior },
            rewards: vec![0.0; states.len()],
            player_switched: vec![true; states.len()],
        }
    }
}
