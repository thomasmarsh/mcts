use super::*;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::TerminalStatus;
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
}

pub trait SimulateStrategy<G>: Clone + Sync + Send + Default
where
    G: Game,
{
    // The default implementation is a uniform selection
    #[allow(unused_variables)]
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
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
            let action: &G::A =
                self.select_move(&state, &available, stats, player, prev_action.as_ref(), rng);
            let action = action.clone();
            actions.push((action.clone(), player));
            prev_action = Some(action.clone());
            state = G::apply(state, &action);
            depth += 1;
        }

        Trial {
            actions,
            state,
            status: Status { end_type },
            depth,
            terminal,
        }
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(0)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Default, Clone)]
pub struct Uniform;

impl<G: Game> SimulateStrategy<G> for Uniform {}

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
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
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
                rng,
            )
        } else {
            self.inner
                .select_move(state, available, stats, player, prev_action, rng)
        }
    }

    fn backprop_flags(&self) -> BackpropFlags {
        self.inner.backprop_flags()
    }
}

/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
pub enum DecisiveMoveMode {
    #[default]
    Win, // Decisive move
    WinLoss,     // Decisive move + anti-decisive move
    WinLossDraw, // Any terminal state
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
                    if !matches!(G::terminal_status(&child_state), TerminalStatus::NotTerminal) {
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
    fn select_move<'a>(
        &mut self,
        state: &<G as Game>::S,
        available: &'a [<G as Game>::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
        rng: &mut SmallRng,
    ) -> &'a <G as Game>::A {
        self.choose(state, available, player).unwrap_or_else(|| {
            self.inner
                .select_move(state, available, stats, player, prev_action, rng)
        })
    }

    fn backprop_flags(&self) -> BackpropFlags {
        self.inner.backprop_flags()
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

    fn select_move<'a>(
        &mut self,
        _state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        _prev_action: Option<&G::A>,
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
/// cutover, not the paper's continuous blend -- see PLAN-WORK.md session
/// 12's design note for why).
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

    fn select_move<'a>(
        &mut self,
        _state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        prev_action: Option<&G::A>,
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
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [<G as Game>::A],
        _stats: &TreeStats<G>,
        _player: usize,
        _prev_action: Option<&G::A>,
        _rng: &mut SmallRng,
    ) -> &'a <G as Game>::A {
        let action = self.inner.choose_action(state);
        let index = available.iter().position(|p| *p == action).unwrap();
        &available[index]
    }
}
