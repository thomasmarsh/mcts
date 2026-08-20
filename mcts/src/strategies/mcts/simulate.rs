use super::*;
use crate::evaluator::{Evaluator, MaterialBlind, EVAL_MAGNITUDE_LIMIT};
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::TerminalStatus;
use crate::strategies::negamax::{Negamax, NegamaxOptions};
use crate::strategies::Search;
use crate::util::random_best;

use rand::rngs::SmallRng;
use rand::Rng;
use rustc_hash::FxHashMap;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub enum EndType {
    NaturalEnd,
    // MoveLimit,
    TurnLimit,
}

#[derive(Debug, Clone)]
pub struct Status {
    pub end_type: Option<EndType>,
}

#[derive(Debug, Clone)]
pub struct Trial<G: Game> {
    pub actions: Vec<(G::A, usize)>,
    pub state: G::S,
    pub status: Status,
    pub depth: usize,
    /// The terminal check already performed on `state` to end the playout
    /// loop (when it ended naturally rather than via the depth cutoff) --
    /// consumers that need the winner/utilities of `state` should check
    /// this before calling `Game::winner`/`Game::compute_utilities` again,
    /// to avoid redoing whatever work `Game::terminal_status` did.
    pub terminal: TerminalStatus<G::P>,
    /// A `SimulateStrategy`-supplied leaf value for a playout that ended via
    /// the depth cutoff (`EndType::TurnLimit`, `terminal` is `NotTerminal`)
    /// -- MCTS-IC-E/-M's hook (`EvaluatedCutoff`, below). `None` for every
    /// plain strategy, and for any naturally-ending playout, in which case
    /// backprop falls back to `terminal.utilities()`/`Game::compute_utilities`
    /// exactly as before this field existed.
    pub cutoff_utilities: Option<Vec<f64>>,
}

pub trait SimulateStrategy<G>: Clone + Sync + Send + Default
where
    G: Game,
{
    // The default implementation is a uniform selection
    #[allow(unused_variables)]
    #[allow(clippy::too_many_arguments)]
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
        own_prev_action: Option<&G::A>,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        &available[rng.gen_range(0..available.len())]
    }

    /// `prev_action` is the most recent action played before this playout's
    /// first ply -- the last edge of the tree-descent path that selected
    /// this leaf, or `None` if the leaf is the tree root itself (no descent
    /// happened this iteration). Only consumed by context-sensitive
    /// strategies (`Nst`); every other strategy ignores it. Scoped to the
    /// current search only -- there is deliberately no attempt to thread in
    /// whatever move preceded the tree root in the real game, since `G::S`
    /// doesn't generally retain that history (see `Nst`'s doc comment).
    ///
    /// `own_prev_action` is this same player's own most recent move earlier
    /// in this playout -- the context `Lgr2` needs on top of `prev_action`
    /// (the opponent's reply to it). Unlike `prev_action`, it is *not*
    /// reconstructed from the tree-descent path -- it starts `None` and is
    /// only filled in once this player has moved at least once within the
    /// current `playout` call, so it's always `None` for a player's first
    /// move of a playout even if they moved earlier during tree descent.
    fn playout(
        &mut self,
        mut state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        mut prev_action: Option<G::A>,
        rng: &mut SmallRng,
    ) -> Trial<G> {
        let mut actions = Vec::new();
        let mut available = Vec::new();
        let mut depth = 0;
        let mut own_prev_action: Vec<Option<G::A>> = vec![None; G::num_players()];
        let end_type;
        let terminal;
        loop {
            let status = G::terminal_status(&state);
            if !matches!(status, TerminalStatus::NotTerminal) {
                end_type = Some(EndType::NaturalEnd);
                terminal = status;
                break;
            }
            if depth >= max_playout_depth {
                end_type = Some(EndType::TurnLimit);
                terminal = TerminalStatus::NotTerminal;
                break;
            }
            available.clear();
            G::generate_actions(&state, &mut available);
            if available.is_empty() {
                end_type = Some(EndType::NaturalEnd);
                terminal = TerminalStatus::NotTerminal;
                break;
            }
            let player = G::player_to_move(&state).to_index();
            let action: &G::A = self.select_move(
                &state,
                &available,
                stats,
                player,
                prev_action.as_ref(),
                own_prev_action[player].as_ref(),
                rng,
            );
            let action = action.clone();
            actions.push((action.clone(), player));
            prev_action = Some(action.clone());
            own_prev_action[player] = Some(action.clone());
            state = G::apply(state, &action);
            depth += 1;
        }

        Trial {
            actions,
            state,
            status: Status { end_type },
            depth,
            terminal,
            cutoff_utilities: None,
        }
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(0)
    }

    /// See `select::SelectStrategy::requirements`'s doc comment -- same
    /// default-from-`backprop_flags` reasoning, mirrored here since
    /// `SimulateStrategy` is a separate trait.
    fn requirements(&self) -> config::Requirements {
        config::Requirements::from_backprop_flags(self.backprop_flags())
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Default, Clone)]
pub struct Uniform;

impl<G: Game> SimulateStrategy<G> for Uniform {
    /// Overrides the default `playout` (rather than just `select_move`) so
    /// each ply can skip materializing the full `available` action list via
    /// `G::generate_actions` -- `Uniform::select_move` would just pick
    /// uniformly from it anyway, and `Uniform` never reads `prev_action`/
    /// `own_prev_action`/`stats`, so nothing here needs the tree-descent
    /// bookkeeping the default loop threads through for context-sensitive
    /// strategies. `G::random_action`'s default still falls back to
    /// `generate_actions` + uniform pick, so this is behavior-preserving for
    /// every game; only a game overriding `random_action` (e.g. with
    /// rejection sampling) actually gets cheaper per-ply cost.
    fn playout(
        &mut self,
        mut state: G::S,
        max_playout_depth: usize,
        _stats: &TreeStats<G>,
        _prev_action: Option<G::A>,
        rng: &mut SmallRng,
    ) -> Trial<G> {
        let mut actions = Vec::new();
        let mut depth = 0;
        let end_type;
        let terminal;
        loop {
            let status = G::terminal_status(&state);
            if !matches!(status, TerminalStatus::NotTerminal) {
                end_type = Some(EndType::NaturalEnd);
                terminal = status;
                break;
            }
            if depth >= max_playout_depth {
                end_type = Some(EndType::TurnLimit);
                terminal = TerminalStatus::NotTerminal;
                break;
            }
            let Some(action) = G::random_action(&state, rng) else {
                end_type = Some(EndType::NaturalEnd);
                terminal = TerminalStatus::NotTerminal;
                break;
            };
            let player = G::player_to_move(&state).to_index();
            actions.push((action.clone(), player));
            state = G::apply(state, &action);
            depth += 1;
        }

        Trial {
            actions,
            state,
            status: Status { end_type },
            depth,
            terminal,
            cutoff_utilities: None,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// MCTS-MR-n (minimax rollouts; the domain-independent predecessor to Baier
/// & Winands' MCTS-IR-M): a uniform-random rollout for every ply, except the
/// last `n` plies before the playout's depth cutoff (`max_playout_depth`),
/// which are instead chosen by an exact bounded-negamax search of however
/// much depth remains. Needs no real heuristic `Evaluator` at all when `n` is
/// small enough that the bounded search reaches a true terminal state before
/// its depth budget runs out -- the default `E = MaterialBlind` covers that
/// case; only a game whose remaining-depth search can still bottom out
/// non-terminal needs a real one plugged in via `MinimaxRollout::<G, E>`.
///
/// Overrides `playout` rather than `select_move`, unlike every other
/// strategy above except `Uniform`: deciding whether a given ply falls
/// within the last `n` needs `depth` and `max_playout_depth`, neither of
/// which `select_move`'s signature carries.
pub struct MinimaxRollout<G, E = MaterialBlind>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    pub n: u32,
    negamax: Negamax<G, E>,
}

/// Hand-written for the same reason as `Negamax`'s own `Clone` impl: a
/// derive would add an `E: Clone` bound that `Negamax<G, E>`'s real
/// requirements don't need.
impl<G, E> Clone for MinimaxRollout<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            negamax: self.negamax.clone(),
        }
    }
}

impl<G, E> MinimaxRollout<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n(mut self, n: u32) -> Self {
        self.n = n;
        self
    }
}

impl<G, E> Default for MinimaxRollout<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    fn default() -> Self {
        Self {
            n: 1,
            // No transposition table: each of the last `n` plies searches a
            // different, shrinking remaining depth from a state that (unlike
            // `Negamax::choose_action`'s iterative deepening) is never
            // revisited at another depth, so a table would only add locking
            // overhead across the many rollouts a single MCTS search runs,
            // never pay for itself with a hit.
            negamax: Negamax::new_with_options(
                E::default(),
                NegamaxOptions::default().with_table_bits(0),
            ),
        }
    }
}

impl<G, E> SimulateStrategy<G> for MinimaxRollout<G, E>
where
    G: Game,
    E: Evaluator<G> + Default,
{
    fn playout(
        &mut self,
        mut state: G::S,
        max_playout_depth: usize,
        _stats: &TreeStats<G>,
        _prev_action: Option<G::A>,
        rng: &mut SmallRng,
    ) -> Trial<G> {
        let mut actions = Vec::new();
        let mut available = Vec::new();
        let mut depth = 0;
        let end_type;
        let terminal;
        loop {
            let status = G::terminal_status(&state);
            if !matches!(status, TerminalStatus::NotTerminal) {
                end_type = Some(EndType::NaturalEnd);
                terminal = status;
                break;
            }
            if depth >= max_playout_depth {
                end_type = Some(EndType::TurnLimit);
                terminal = TerminalStatus::NotTerminal;
                break;
            }
            available.clear();
            G::generate_actions(&state, &mut available);
            if available.is_empty() {
                end_type = Some(EndType::NaturalEnd);
                terminal = TerminalStatus::NotTerminal;
                break;
            }
            let remaining = (max_playout_depth - depth) as u32;
            let player = G::player_to_move(&state).to_index();
            let action = if remaining <= self.n {
                self.negamax.bounded_negamax(&state, remaining).0
            } else {
                available[rng.gen_range(0..available.len())].clone()
            };
            actions.push((action.clone(), player));
            state = G::apply(state, &action);
            depth += 1;
        }

        Trial {
            actions,
            state,
            status: Status { end_type },
            depth,
            terminal,
            cutoff_utilities: None,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// MCTS-IC-E (informed cutoffs, evaluator variant; Baier & Winands): wraps an
/// inner `SimulateStrategy` unchanged except that a playout ending via the
/// depth cutoff (`EndType::TurnLimit`, i.e. `trial.terminal` is
/// `NotTerminal`) gets its leaf value from `Evaluator::evaluate` instead of
/// whatever `backprop::update`'s fallback chain would otherwise compute
/// (`Game::compute_utilities`, which is `winner`-based and so silently
/// scores every cutoff as a draw for a game that hasn't overridden it -- see
/// `evaluator::Evaluator`'s doc comment). Every naturally-ending playout (a
/// real win/loss/draw) is untouched: `Trial::terminal` already carries its
/// true utilities, which that same fallback chain always prefers over this
/// field.
///
/// `Evaluator::evaluate`'s score is from the perspective of
/// `Game::player_to_move(&trial.state)`; converted here to a per-player
/// utilities vector the same "nega" way `negamax`'s scoring does, which is
/// only sound for the two-player zero-sum games that convention assumes --
/// same restriction as `negamax::supports::<G>()` and `use_mcts_solver`.
pub struct EvaluatedCutoff<G, E, S = Uniform>
where
    G: Game,
    E: Evaluator<G> + Default,
    S: SimulateStrategy<G> + Default,
{
    evaluator: E,
    inner: S,
    marker: PhantomData<G>,
}

/// Hand-written (rather than `#[derive(Clone)]`) so only `E: Clone` -- not
/// also whatever bound a derive would add transitively -- is required; every
/// real `Evaluator` this crate ships (`MaterialBlind`, `breakthrough::
/// Heuristic`) is already `Clone`.
impl<G, E, S> Clone for EvaluatedCutoff<G, E, S>
where
    G: Game,
    E: Evaluator<G> + Default + Clone,
    S: SimulateStrategy<G> + Default,
{
    fn clone(&self) -> Self {
        Self {
            evaluator: self.evaluator.clone(),
            inner: self.inner.clone(),
            marker: PhantomData,
        }
    }
}

impl<G, E, S> EvaluatedCutoff<G, E, S>
where
    G: Game,
    E: Evaluator<G> + Default,
    S: SimulateStrategy<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inner(mut self, inner: S) -> Self {
        self.inner = inner;
        self
    }
}

impl<G, E, S> Default for EvaluatedCutoff<G, E, S>
where
    G: Game,
    E: Evaluator<G> + Default,
    S: SimulateStrategy<G> + Default,
{
    fn default() -> Self {
        Self {
            evaluator: E::default(),
            inner: S::default(),
            marker: PhantomData,
        }
    }
}

impl<G, E, S> SimulateStrategy<G> for EvaluatedCutoff<G, E, S>
where
    G: Game,
    E: Evaluator<G> + Default + Clone,
    S: SimulateStrategy<G> + Default,
{
    fn playout(
        &mut self,
        state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        prev_action: Option<G::A>,
        rng: &mut SmallRng,
    ) -> Trial<G> {
        let mut trial = self
            .inner
            .playout(state, max_playout_depth, stats, prev_action, rng);
        if matches!(trial.status.end_type, Some(EndType::TurnLimit)) {
            debug_assert!(
                G::num_players() <= 2,
                "EvaluatedCutoff's nega-style utility conversion assumes a \
                 two-player zero-sum game"
            );
            let score = self.evaluator.evaluate(&trial.state) as f64 / EVAL_MAGNITUDE_LIMIT as f64;
            let mover = G::player_to_move(&trial.state).to_index();
            trial.cutoff_utilities = Some(
                (0..G::num_players())
                    .map(|i| if i == mover { score } else { -score })
                    .collect(),
            );
        }
        trial
    }

    fn backprop_flags(&self) -> BackpropFlags {
        self.inner.backprop_flags()
    }

    /// See `EpsilonGreedy::requirements`'s doc comment -- same reason: this
    /// wraps `inner` without changing its storage requirements.
    fn requirements(&self) -> config::Requirements {
        self.inner.requirements()
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct EpsilonGreedy<G, S>
where
    G: Game,
    S: SimulateStrategy<G>,
{
    pub epsilon: f64,
    inner: S,
    marker: PhantomData<G>,
}

impl<G, S> EpsilonGreedy<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self {
            epsilon,
            ..Default::default()
        }
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn inner(mut self, inner: S) -> Self {
        self.inner = inner;
        self
    }
}

impl<G, S> Default for EpsilonGreedy<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            inner: Default::default(),
            marker: PhantomData,
        }
    }
}

impl<G, S> SimulateStrategy<G> for EpsilonGreedy<G, S>
where
    G: Game,
    S: SimulateStrategy<G>,
{
    #[allow(clippy::too_many_arguments)]
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
        own_prev_action: Option<&G::A>,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        if rng.gen::<f64>() < self.epsilon {
            <Uniform as SimulateStrategy<G>>::select_move(
                &mut Uniform,
                state,
                available,
                stats,
                player,
                prev_action,
                own_prev_action,
                rng,
            )
        } else {
            self.inner.select_move(
                state,
                available,
                stats,
                player,
                prev_action,
                own_prev_action,
                rng,
            )
        }
    }

    fn backprop_flags(&self) -> BackpropFlags {
        self.inner.backprop_flags()
    }

    /// Delegates to `inner.requirements()` directly (not the default's
    /// `from_backprop_flags(self.backprop_flags())`) -- a wrapped component
    /// whose `requirements()` carries something `backprop_flags` can't
    /// express (e.g. `select::UctPn`'s `solver`/`max_players`) needs that to
    /// survive being wrapped, not just its backprop bits.
    fn requirements(&self) -> config::Requirements {
        self.inner.requirements()
    }
}

/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisiveMoveMode {
    #[default]
    Win, // Decisive move
    WinLoss,     // Decisive move + anti-decisive move
    WinLossDraw, // Any terminal state
    /// Teytaud & Teytaud 2010's Algorithm 4 (DM+ADM), the real "anti-decisive
    /// move" rather than `WinLoss`'s same-ply terminal check: if no move
    /// wins immediately, look one ply further and prefer a move that leaves
    /// the opponent with no immediate winning reply. Strictly more work than
    /// the other modes (a second `generate_actions` + scan per candidate),
    /// so it's the right default only for playouts with a small enough
    /// branching factor that the extra ply is cheap relative to a full
    /// rollout.
    AntiDecisive,
}

#[derive(Clone)]
pub struct DecisiveMove<G, S = Uniform>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    mode: DecisiveMoveMode,
    inner: S,
    marker: PhantomData<G>,
}

impl<G, S> DecisiveMove<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mode(mut self, mode: DecisiveMoveMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn inner(mut self, inner: S) -> Self {
        self.inner = inner;
        self
    }

    fn choose<'a>(
        &self,
        state: &<G as Game>::S,
        available: &'a [<G as Game>::A],
        player: usize,
    ) -> Option<&'a <G as Game>::A> {
        use DecisiveMoveMode::*;

        let mut draw = None;
        let mut loser = None;
        match self.mode {
            WinLossDraw => {
                for action in available {
                    let child_state = G::apply(state.clone(), action);
                    if !matches!(
                        G::terminal_status(&child_state),
                        TerminalStatus::NotTerminal
                    ) {
                        return Some(action);
                    }
                }
                None
            }

            WinLoss => {
                for action in available {
                    let child_state = G::apply(state.clone(), action);
                    match G::terminal_status(&child_state) {
                        TerminalStatus::Winner(_) => return Some(action),
                        TerminalStatus::Draw => draw = Some(action),
                        TerminalStatus::NotTerminal => {}
                    }
                }
                draw
            }

            Win => {
                for action in available {
                    let child_state = G::apply(state.clone(), action);
                    match G::terminal_status(&child_state) {
                        TerminalStatus::Winner(winner) => {
                            if winner.to_index() == player {
                                return Some(action);
                            }
                            loser = Some(action);
                        }
                        TerminalStatus::Draw => draw = Some(action),
                        TerminalStatus::NotTerminal => {}
                    }
                }
                loser.or(draw)
            }

            AntiDecisive => {
                // Pass 1: an immediate win always takes priority, regardless
                // of where it falls in `available` -- same rule as `Win`.
                for action in available {
                    let child_state = G::apply(state.clone(), action);
                    if matches!(G::terminal_status(&child_state), TerminalStatus::Winner(w) if w.to_index() == player)
                    {
                        return Some(action);
                    }
                }
                // Pass 2 (only reached once pass 1 has ruled out a win
                // anywhere in the list): the real anti-decisive check --
                // look one ply further and take the first move that leaves
                // the opponent no immediate winning reply. `opponent_actions`
                // is reused across candidates rather than reallocated, and
                // this returns on the first acceptable candidate instead of
                // scoring all of them, both in the spirit of Soemers et al.
                // 2021's playout implementations: do the least work needed
                // to find *a* good move, not the full board evaluation.
                let mut opponent_actions = Vec::new();
                for action in available {
                    let child_state = G::apply(state.clone(), action);
                    if !matches!(
                        G::terminal_status(&child_state),
                        TerminalStatus::NotTerminal
                    ) {
                        // Already a loss or draw for us -- ruled out above as
                        // a win, so never worth preferring over an unproven
                        // continuation.
                        continue;
                    }
                    opponent_actions.clear();
                    G::generate_actions(&child_state, &mut opponent_actions);
                    let opponent_can_win = opponent_actions.iter().any(|reply| {
                        matches!(
                            G::terminal_status(&G::apply(child_state.clone(), reply)),
                            TerminalStatus::Winner(_)
                        )
                    });
                    if !opponent_can_win {
                        return Some(action);
                    }
                }
                None
            }
        }
    }
}

impl<G, S> Default for DecisiveMove<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    fn default() -> Self {
        Self {
            mode: DecisiveMoveMode::default(),
            inner: S::default(),
            marker: PhantomData,
        }
    }
}

impl<G, S> SimulateStrategy<G> for DecisiveMove<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    #[allow(clippy::too_many_arguments)]
    fn select_move<'a>(
        &mut self,
        state: &<G as Game>::S,
        available: &'a [<G as Game>::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
        own_prev_action: Option<&G::A>,
        rng: &mut SmallRng,
    ) -> &'a <G as Game>::A {
        self.choose(state, available, player).unwrap_or_else(|| {
            self.inner.select_move(
                state,
                available,
                stats,
                player,
                prev_action,
                own_prev_action,
                rng,
            )
        })
    }

    fn backprop_flags(&self) -> BackpropFlags {
        self.inner.backprop_flags()
    }

    /// See `EpsilonGreedy::requirements`'s doc comment above -- same reason.
    fn requirements(&self) -> config::Requirements {
        self.inner.requirements()
    }
}

/// Picks among `available` by `score_of`, ties broken randomly -- the shared
/// scoring/selection shape `Mast` and `Nst` both reduce to, since NST's only
/// real difference from MAST is *which* table a per-action score comes from.
fn select_by_score<'a, A>(
    available: &'a [A],
    rng: &mut SmallRng,
    mut score_of: impl FnMut(&A) -> f64,
) -> &'a A {
    let scored: Vec<(f64, &A)> = available.iter().map(|a| (score_of(a), a)).collect();
    random_best(&scored, rng, |(score, _)| *score).unwrap().1
}

/// A MAST-table lookup for one action: unvisited actions default to `1.`
/// (matches MAST's original optimistic-untried-move behavior), visited ones
/// score their running average utility.
fn unigram_score<A: crate::game::Action>(
    player_actions: &FxHashMap<A, node::ActionStats>,
    action: &A,
) -> f64 {
    player_actions.get(action).map_or(1., |stats| {
        if stats.num_visits > 0 {
            stats.score / stats.num_visits as f64
        } else {
            1.
        }
    })
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Default, Clone)]
pub struct Mast;

impl<G> SimulateStrategy<G> for Mast
where
    G: Game,
{
    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(GLOBAL)
    }

    #[allow(clippy::too_many_arguments)]
    fn select_move<'a>(
        &mut self,
        _state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        _prev_action: Option<&G::A>,
        _own_prev_action: Option<&G::A>,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        let player_actions = stats.player_actions[player].read().unwrap();
        select_by_score(available, rng, |action| {
            unigram_score(&player_actions, action)
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

/// N-gram Selection Technique (NST, Tak & Winands 2012): a bigram extension
/// of `Mast` -- instead of scoring a candidate action only by its own
/// context-free running average (`Mast`'s `player_actions` table), it first
/// looks up the pair `(prev_action, action)` in `player_bigram_actions` and
/// uses that conditional average once it has at least `backoff_threshold`
/// samples, falling back to the plain unigram/MAST score otherwise (a hard
/// cutover, not the paper's continuous blend).
///
/// The "previous action" context is scoped to the current search only: it's
/// the last edge of the tree-descent path that selected the playout's
/// starting leaf, or the previous ply within the same playout -- never a
/// real move from before the tree root, since `G::S` doesn't generally
/// retain that history and the paper's own formulation is within-rollout
/// anyway. At the very first ply of a playout rolled out directly from the
/// tree root (no descent happened), there is no context and this falls back
/// to the unigram score unconditionally.
#[derive(Clone, Copy, Debug)]
pub struct Nst {
    pub backoff_threshold: u32,
}

impl Default for Nst {
    fn default() -> Self {
        Self {
            backoff_threshold: 5,
        }
    }
}

impl Nst {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn backoff_threshold(mut self, backoff_threshold: u32) -> Self {
        self.backoff_threshold = backoff_threshold;
        self
    }
}

impl<G> SimulateStrategy<G> for Nst
where
    G: Game,
{
    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(GLOBAL | NST)
    }

    #[allow(clippy::too_many_arguments)]
    fn select_move<'a>(
        &mut self,
        _state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
        _own_prev_action: Option<&G::A>,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        let player_actions = stats.player_actions[player].read().unwrap();
        let Some(prev) = prev_action else {
            return select_by_score(available, rng, |action| {
                unigram_score(&player_actions, action)
            });
        };
        let bigram_actions = stats.player_bigram_actions[player].read().unwrap();
        select_by_score(available, rng, |action| {
            match bigram_actions.get(&(prev.clone(), action.clone())) {
                Some(bigram_stats) if bigram_stats.num_visits >= self.backoff_threshold => {
                    bigram_stats.score / bigram_stats.num_visits as f64
                }
                _ => unigram_score(&player_actions, action),
            }
        })
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Last Good Reply (LGR-1, Baier & Drake 2010): a per-player reply table,
/// keyed by the opponent's preceding move, that always plays the most
/// recent move this player replied with in that context *and* went on to
/// win the playout with -- a deterministic override, not a score like
/// `Mast`/`Nst`, falling back to `inner` whenever no reply is recorded yet
/// or the recorded reply isn't currently legal. Unlike `Nst`'s bigram table
/// (a running average with a visit-count backoff), LGR's table is plain
/// last-write-wins with no unlearning -- that's LGRF-2's refinement, not
/// this one.
///
/// Same playout-scoped `prev_action` caveat as `Nst`: the context is always
/// within the current search, never a real move from before the tree root.
#[derive(Clone)]
pub struct Lgr<G, S = Uniform>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    inner: S,
    marker: PhantomData<G>,
}

impl<G, S> Lgr<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inner(mut self, inner: S) -> Self {
        self.inner = inner;
        self
    }
}

impl<G, S> Default for Lgr<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    fn default() -> Self {
        Self {
            inner: S::default(),
            marker: PhantomData,
        }
    }
}

impl<G, S> SimulateStrategy<G> for Lgr<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(LGR) | self.inner.backprop_flags()
    }

    /// See `EpsilonGreedy::requirements`'s doc comment -- same reason: this
    /// needs to add its own `lgr` bit on top of whatever `inner` needs,
    /// which the default `from_backprop_flags(self.backprop_flags())` would
    /// already get right here, but an explicit union keeps this consistent
    /// with `inner`'s own overridden `requirements()` (e.g. an `inner` with
    /// `max_players`/`solver` set, which `backprop_flags()` alone can't
    /// express).
    fn requirements(&self) -> config::Requirements {
        config::Requirements {
            lgr: true,
            ..self.inner.requirements()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
        _own_prev_action: Option<&G::A>,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        if let Some(prev) = prev_action {
            let replies = stats.player_replies[player].read().unwrap();
            if let Some(reply) = replies.get(prev) {
                if let Some(found) = available.iter().find(|a| *a == reply) {
                    return found;
                }
            }
        }
        self.inner.select_move(
            state,
            available,
            stats,
            player,
            prev_action,
            _own_prev_action,
            rng,
        )
    }
}

////////////////////////////////////////////////////////////////////////////////

/// LGRF-2 (Last Good Reply with Forgetting, level 2; Baier & Drake 2010):
/// `Lgr`'s single-ply reply table extended with a second table keyed by
/// *both* preceding moves -- this player's own last move and the
/// opponent's reply to it -- which is where the "forgetting" half of
/// LGRF-2 lives: a 2-ply reply that goes on to lose is actively removed
/// from this table (`backprop::BackpropStrategy::update`'s `flags.lgr2()`
/// block), rather than just being left unwritten the way a losing trial
/// already is for `Lgr`'s plain table.
///
/// Falls back to `inner` (default `Lgr<G>`, i.e. LGR-1) whenever there's no
/// 2-ply context yet (a player's first move of a playout), no recorded
/// reply for that context, or the recorded reply isn't currently legal --
/// so `Lgr2<G>`'s default composition (`Lgr2<G, Lgr<G, Uniform>>`) gets the
/// full LGRF-2 -> LGR-1 -> uniform fallback chain the paper describes for
/// free, by literally nesting `Lgr` rather than re-deriving its behavior.
///
/// The 1-ply table `inner: Lgr` reads/writes (`TreeStats::player_replies`)
/// keeps `Lgr`'s existing plain last-write-wins semantics -- no forgetting
/// -- so composing `Lgr` directly (LGR-1) is unaffected by this type
/// existing at all.
#[derive(Clone)]
pub struct Lgr2<G, S = Lgr<G>>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    inner: S,
    marker: PhantomData<G>,
}

impl<G, S> Lgr2<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inner(mut self, inner: S) -> Self {
        self.inner = inner;
        self
    }
}

impl<G, S> Default for Lgr2<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    fn default() -> Self {
        Self {
            inner: S::default(),
            marker: PhantomData,
        }
    }
}

impl<G, S> SimulateStrategy<G> for Lgr2<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(LGR2) | self.inner.backprop_flags()
    }

    /// See `EpsilonGreedy::requirements`'s doc comment -- same reason: adds
    /// this type's own `lgr2` bit on top of whatever `inner` needs.
    fn requirements(&self) -> config::Requirements {
        config::Requirements {
            lgr2: true,
            ..self.inner.requirements()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
        own_prev_action: Option<&G::A>,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        if let (Some(own), Some(opp)) = (own_prev_action, prev_action) {
            let replies2 = stats.player_replies2[player].read().unwrap();
            if let Some(reply) = replies2.get(&(own.clone(), opp.clone())) {
                if let Some(found) = available.iter().find(|a| *a == reply) {
                    return found;
                }
            }
        }
        self.inner.select_move(
            state,
            available,
            stats,
            player,
            prev_action,
            own_prev_action,
            rng,
        )
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct MetaMcts<G: Game, S: Strategy<G>> {
    pub inner: TreeSearch<G, S>,
}

impl<G, S> Default for MetaMcts<G, S>
where
    G: Game,
    S: Strategy<G>,
{
    fn default() -> Self {
        Self {
            inner: TreeSearch::default(),
        }
    }
}

impl<G, S> SimulateStrategy<G> for MetaMcts<G, S>
where
    G: Game,
    S: Strategy<G>,
{
    #[allow(clippy::too_many_arguments)]
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [<G as Game>::A],
        _stats: &TreeStats<G>,
        _player: usize,
        _prev_action: Option<&G::A>,
        _own_prev_action: Option<&G::A>,
        _rng: &mut SmallRng,
    ) -> &'a <G as Game>::A {
        let action = self.inner.choose_action(state);
        let index = available.iter().position(|p| *p == action).unwrap();
        &available[index]
    }
}
