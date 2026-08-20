use rand::rngs::SmallRng;
use rand::Rng;
use serde::Serialize;

// Refers to a player index. Expectation is that these values
// are small and monotonically increasing. Stored as a usize for ease
// of use as an array index.
pub trait PlayerIndex {
    fn to_index(&self) -> usize;
}

// A proxy trait to simplify some implementation.
//
// NOTE: the `Hash` requirement is less strong than the Zobrist requirement for
// transposition tables. However, it would be nice to use the zobrist hash if it
// is available since it may be cheaper.
pub trait Action: Clone + Eq + std::hash::Hash + std::fmt::Debug + Serialize + Sync + Send {}

// Blanket implementation
impl<T: Clone + Eq + std::hash::Hash + std::fmt::Debug + Serialize + Sync + Send> Action for T {}

/// Index of an element in a game's symmetry group, as reported by
/// `Game::canonical_representation` and consumed by `Game::apply_to_action`/
/// `invert_action` -- which orientation a canonicalized state or action sits
/// in, relative to the literal board. `Transform::IDENTITY` (index `0`) is
/// always the no-op element, the same convention `game_core::symmetry::
/// SymmetryGroup` uses for its own group elements; `is_identity` lets a
/// caller fast-path the common case (no symmetry, or the root's own
/// never-canonicalized action list) instead of unconditionally composing/
/// inverting through a transform that turns out to do nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Transform(usize);

impl Transform {
    pub const IDENTITY: Transform = Transform(0);

    #[inline]
    pub const fn new(index: usize) -> Self {
        Transform(index)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0
    }

    #[inline]
    pub const fn is_identity(self) -> bool {
        self.0 == 0
    }
}

impl From<usize> for Transform {
    #[inline]
    fn from(index: usize) -> Self {
        Transform(index)
    }
}

impl From<Transform> for usize {
    #[inline]
    fn from(sym: Transform) -> usize {
        sym.0
    }
}

/// A value expressed in the literal, physical orientation of the game
/// actually being played -- directly legal against `Game::S`/playable via
/// `Game::apply`, or (for an action) directly present in `Game::
/// generate_actions`'s output for such a state. Every consumer outside the
/// canonicalization/graph-merge machinery itself (rollout continuation,
/// applying a move to the real game, the UI/server boundary) deals in
/// `Real` values.
///
/// Contrast [`Canonical`]. The two wrappers exist so a signature states
/// which frame a state or action is in, rather than leaving it to a
/// parameter name or doc comment: a caller that mixes up `Real` and
/// `Canonical` gets a compile error at the call site instead of silently
/// applying/keying a canonical-orientation value against a literal-board
/// state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Real<T>(pub T);

/// A value expressed in the canonicalized orientation `Game::
/// canonical_representation` chose for a position's equivalence class --
/// not directly legal against the real game state until translated back via
/// `Game::invert_action` (see [`Real`]). A `ChildArray`'s own action list is
/// stored in `Canonical` terms for every node except the root (which has no
/// incoming edge to canonicalize against, and so stays `Real`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Canonical<T>(pub T);

impl<T> Real<T> {
    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> Canonical<T> {
    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

/// The outcome of checking whether a state is terminal, bundled with the
/// winner when it is -- so a caller that needs both `is_terminal` and
/// `winner` on the same state (the common case at the end of a rollout) can
/// get them from a single underlying check instead of two, when a `Game`
/// overrides `Game::terminal_status` to compute them together (see
/// `Druid::terminal_status`, which computes the win condition once instead
/// of via separate `is_terminal`/`winner` calls that each redo the same
/// connectivity scan).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalStatus<P> {
    NotTerminal,
    Draw,
    Winner(P),
}

impl<P: PlayerIndex> TerminalStatus<P> {
    /// The utilities this status implies, or `None` if it isn't terminal --
    /// matching `Game::compute_utilities`'s default (1./-1. for the winner,
    /// 0. for a draw) without touching the state again. Callers fall back to
    /// `Game::compute_utilities` on `None` since a non-terminal cutoff (e.g.
    /// a playout depth limit) still needs a utility.
    pub fn utilities(&self, num_players: usize) -> Option<Vec<f64>> {
        match self {
            TerminalStatus::NotTerminal => None,
            TerminalStatus::Draw => Some(vec![0.; num_players]),
            TerminalStatus::Winner(w) => {
                let wi = w.to_index();
                Some(
                    (0..num_players)
                        .map(|i| if i == wi { 1. } else { -1. })
                        .collect(),
                )
            }
        }
    }
}

pub trait Game: Sized + Clone + Sync + Send {
    /// The type representing the state of your game. Ideally, this
    /// should be as small as possible and have a cheap Clone or Copy
    /// implementation.
    type S: Clone + Default + std::fmt::Debug + Sized + Sync + Send + Eq + std::fmt::Display;

    /// The type representing actions, or moves, in your game. These
    /// also should be very cheap to clone.
    type A: Action;

    /// The player type. This value only needs to conform to PlayerIndex.
    type P: PlayerIndex + Clone + std::fmt::Debug + Sync + Send;

    /// Given a state, apply an action to it producing a new state.
    fn apply(state: Self::S, action: &Self::A) -> Self::S;

    /// All possible actions from a given state. This is expected to
    /// be deterministic. (Subsequent invocations on the same state
    /// should produce the same set of actions.) This will not be
    /// invoked if `is_terminal` returns `true`.
    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>);

    /// A single uniformly-random legal action, or `None` if there are none
    /// (mirrors `generate_actions` returning empty on a non-terminal state --
    /// see its callers). The default just materializes the full list via
    /// `generate_actions` and picks one, so this is correct for every `Game`
    /// for free; it exists so a game whose legality check is cheap per
    /// candidate (e.g. an incremental engine) can override it with rejection
    /// sampling -- draw a random cell, check just that one, retry on
    /// failure -- instead of enumerating every candidate on every rollout
    /// ply. `SimulateStrategy::playout`'s default (uniform) path is the only
    /// caller; context-sensitive strategies (`Mast`/`Nst`/`Lgr`/`Lgr2`) still
    /// need the full list for their policy lookups and don't use this.
    fn random_action(state: &Self::S, rng: &mut SmallRng) -> Option<Self::A> {
        let mut actions = Vec::new();
        Self::generate_actions(state, &mut actions);
        if actions.is_empty() {
            None
        } else {
            Some(actions[rng.gen_range(0..actions.len())].clone())
        }
    }

    /// Returns `true` if the game has ended and there are no more
    /// possible actions. The default implementation calls
    /// `generate_actions` which may be expensive. Ideally this can
    /// be computed more cheaply.
    fn is_terminal(state: &Self::S) -> bool {
        let mut actions = Vec::new();
        Self::generate_actions(state, &mut actions);
        actions.is_empty()
    }

    /// `is_terminal` and `winner`, bundled. The default just calls both in
    /// sequence (so behavior is unchanged for every `Game` that doesn't
    /// override this), but a game whose terminal check and win check share
    /// underlying work -- e.g. Druid, where both are answered by the same
    /// board connectivity scan -- can override this to do that work once.
    fn terminal_status(state: &Self::S) -> TerminalStatus<Self::P> {
        if Self::is_terminal(state) {
            match Self::winner(state) {
                Some(w) => TerminalStatus::Winner(w),
                None => TerminalStatus::Draw,
            }
        } else {
            TerminalStatus::NotTerminal
        }
    }

    /// For games with hidden information, state may be determinized
    /// for the sake of sampling via a playout. Essentially, this
    /// amounts to shuffling the hidden state around. Please note,
    /// however, that determinization can be difficult to perform
    /// uniformly and may introduce bias in the the playouts.
    #[allow(unused_variables)]
    fn determinize(state: Self::S, rng: &mut SmallRng) -> Self::S {
        state
    }

    /// Assuming a zero-sum game, the player who has won.
    fn winner(state: &Self::S) -> Option<Self::P>;

    /// Returns the rank of the player in a given game state. The
    /// current implementation assumes a two-player game. Rank is
    /// a value between 1.0 and num_players, with 1.0 being best
    /// and higher numbers being worse.
    //
    // NOTE: this is too expensive. Maybe `rank(S) -> Vec<f64>`
    fn rank(state: &Self::S, player_index: usize) -> f64 {
        match Self::winner(state) {
            Some(w) if w.to_index() == player_index => 1.,
            Some(_) => 2.,
            None => 1.5,
        }
    }

    /// Returns the play whose turn it is to move for the given
    /// state.
    fn player_to_move(state: &Self::S) -> Self::P;

    /// A constant value that indicates the number of players
    /// in the game.
    fn num_players() -> usize {
        2
    }

    /// Move notation for a given move relative to a given state.
    #[allow(unused)]
    fn notation(state: &Self::S, action: &Self::A) -> String {
        "??".into()
    }

    #[inline]
    fn get_reward(init: &Self::S, term: &Self::S) -> f64 {
        Self::compute_utilities(term)[Self::player_to_move(init).to_index()]
    }

    #[allow(unused_variables)]
    fn parse_action(state: &Self::S, input: &str) -> Option<Self::A> {
        unimplemented!();
    }

    // #[inline]
    // fn rank_to_util(rank: f64, num_players: usize) -> f64 {
    //     let n = num_players as f64;

    //     if n == 1. {
    //         2. * rank - 1.
    //     } else {
    //         1. - ((rank - 1.) * (2. / (n - 1.)))
    //     }
    // }

    #[inline]
    fn compute_utilities(state: &Self::S) -> Vec<f64> {
        let winner = Self::winner(state).map(|p| p.to_index());
        (0..Self::num_players())
            .map(|i| match winner {
                None => 0.,
                Some(w) if w == i => 1.,
                _ => -1.,
            })
            .collect()

        // TODO: think about the best way to handle ranking
        //
        // (0..Self::num_players())
        //     .map(|i| {
        //         let n = Self::num_players();
        //         let rank = Self::rank(state, i);
        //         rank_to_util(rank, n)
        //     })
        //     .collect()
    }

    /// A canonical representation of the state, paired with the index of the
    /// symmetry group element that was applied to reach it (`0` is always
    /// the identity). Many board games exhibit some form of symmetry;
    /// canonicalizing the state lets the engine recognize equivalent
    /// positions reached through different move orders. The symmetry index
    /// is what a caller needs to translate actions between the literal
    /// board and the canonicalized state via `apply_to_action`/
    /// `invert_action` below -- the canonicalized state alone isn't enough,
    /// since two different callers can canonicalize to the same state via
    /// different symmetry elements.
    ///
    /// Games that haven't characterized their symmetries return the state
    /// unchanged with symmetry index `0`; this is indistinguishable from
    /// "characterized as having no symmetry", which is fine, since both are
    /// legitimate uses of the identity element.
    fn canonical_representation(state: Real<Self::S>) -> (Canonical<Self::S>, Transform) {
        (Canonical(state.0), Transform::IDENTITY)
    }

    /// Map an action through symmetry element `sym` -- e.g. to translate a
    /// legal action on the literal board into the orientation of a
    /// canonicalized state produced by `canonical_representation`. The
    /// default is the identity, which is correct as long as
    /// `canonical_representation` hasn't been overridden to report anything
    /// but `Transform::IDENTITY`.
    #[allow(unused_variables)]
    fn apply_to_action(action: Real<Self::A>, sym: Transform) -> Canonical<Self::A> {
        Canonical(action.0)
    }

    /// The inverse of `apply_to_action`: `invert_action(apply_to_action(a,
    /// s), s) == a` for every legal action `a` and every symmetry index `s`
    /// the game's `canonical_representation` can report.
    #[allow(unused_variables)]
    fn invert_action(action: Canonical<Self::A>, sym: Transform) -> Real<Self::A> {
        Real(action.0)
    }

    /// The ply (pieces/stones already placed on the board `state`
    /// represents) past which `canonical_representation` should stop
    /// attempting to canonicalize. A position's stabilizer under a game's
    /// symmetry group generally shrinks toward the identity as more
    /// distinguishing detail accumulates on the board, so canonicalizing
    /// well past that point mostly pays its recomputation cost (once per
    /// *visit* -- see `crate::symmetry::incoming_sym`, called from `select_step` and
    /// every `backprop`/PV/render walk, not once per node expansion) for
    /// diminishing transposition-table hits. The default, `usize::MAX`,
    /// means "always attempt it" -- today's behavior for every game that hasn't
    /// characterized a cutoff (including games too small for one to
    /// matter, like tic-tac-toe's whole 9-ply game).
    ///
    /// Takes `state`, not just `Self`, because the right cutoff can depend
    /// on the specific board an instance was built on, not just the game
    /// type: a fixed-board game (tic-tac-toe, Othello) can ignore the
    /// argument and return a constant, but a game whose board size varies
    /// per instance needs to derive its threshold from that instance's own
    /// dimensions rather than the engine imposing one number for every
    /// size.
    #[allow(unused_variables)]
    fn symmetry_ply_limit(state: &Self::S) -> usize {
        usize::MAX
    }

    /// A zobrist hash is expected to be cheap and precomputed upon move
    /// application.
    #[allow(unused_variables)]
    fn zobrist_hash(state: &Self::S) -> u64 {
        0
    }

    /// Whether the game has chance events (dice, shuffled draws, etc.)
    /// beyond what `determinize`'s single-sample resampling papers over --
    /// i.e. whether a deterministic tree search over `Self::S` alone, with
    /// no chance-node branching, can see the whole picture. Defaults to
    /// `false` (deterministic), matching every game in this repo today.
    /// Consulted opportunistically by strategies that assume determinism
    /// (see `strategies::negamax::supports`) as a capability check, not a
    /// compile-time guarantee -- a `true` here doesn't add or remove any
    /// method on `Game` itself.
    fn is_stochastic() -> bool {
        false
    }

    /// Whether a player's own `Self::S` fails to fully determine what
    /// every other player can observe (cards in hand, fog of war, etc.).
    /// Defaults to `false` (perfect information).
    fn has_hidden_information() -> bool {
        false
    }

    /// Whether players strictly alternate single moves, with no
    /// simultaneous-move or multi-action-per-turn phases. Defaults to
    /// `true`, matching every game in this repo today.
    fn alternating_moves() -> bool {
        true
    }
}
