/// An assessment of a game state from the perspective of the player whose turn it is to play.
/// Higher values mean a more favorable state.
/// A draw is defined as a score of zero.
pub type Evaluation = i16;

// These definitions ensure that they negate to each other, but it leaves
// i16::MIN as a valid value less than WORST_EVAL. Don't use this value, and
// any Strategy will panic when it tries to negate it.

// TODO: +1/-1/0 are better for UCT

/// An absolutely wonderful outcome, e.g. a win.
pub const BEST_EVAL: Evaluation = i16::MAX;
/// An absolutely disastrous outcome, e.g. a loss.
pub const WORST_EVAL: Evaluation = -BEST_EVAL;

/// The result of playing a game until it finishes.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Winner {
    /// The player who made the last move won.
    PlayerJustMoved,
    /// Nobody won.
    Draw,
    /// The player who made the last move lost.
    ///
    /// This is uncommon, and many games (chess, checkers, tic-tac-toe, etc)
    /// do not have this possibility.
    PlayerToMove,
}

impl Winner {
    /// Canonical evaluations for end states.
    pub fn evaluate(&self) -> Evaluation {
        match *self {
            Winner::PlayerJustMoved => WORST_EVAL,
            Winner::PlayerToMove => BEST_EVAL,
            Winner::Draw => 0,
        }
    }
}

/// Defines the rules for a two-player, perfect-knowledge game.
///
/// A game ties together types for the state and moves, generates the possible
/// moves from a particular state, and determines whether a state is terminal.
///
/// This is meant to be defined on an empty newtype so that a game engine can
/// be implemented in a separate crate without having to know about these
/// `minimax` traits.
pub trait Game: Sized {
    /// The type of the game state.
    type S;
    /// The type of game moves.
    type M: Clone;

    /// Generate moves at the given state.
    fn generate_moves(state: &Self::S, moves: &mut Vec<Self::M>);

    /// Apply a move to get a new state.
    ///
    /// If the method returns a new state, the caller should use that. If the
    /// method returns None, the caller should use the original.
    /// This enables two different implementation strategies:
    ///
    /// 1) Games with large state that want to update in place.
    /// ```
    /// struct BigBoard([u8; 4096]);
    /// struct BigMove(u16);
    /// fn apply(state: &mut BigBoard, m: BigMove) -> Option<BigBoard> {
    ///     state.0[m.0 as usize] += 1;
    ///     None
    /// }
    /// fn undo(state: &mut BigBoard, m: BigMove) {
    ///     state.0[m.0 as usize] -= 1;
    /// }
    /// ```
    ///
    /// 2) Games with small state that don't want to implement undo.
    /// ```
    /// struct SmallBoard(u64);
    /// struct SmallMove(u8);
    /// fn apply(state: &mut SmallBoard, m: SmallMove) -> Option<SmallBoard> {
    ///     Some(SmallBoard(state.0 | (1<<m.0)))
    /// }
    /// ```
    fn apply(state: &mut Self::S, m: Self::M) -> Option<Self::S>;

    /// Undo mutation done in apply, if any.
    fn undo(_state: &mut Self::S, _m: Self::M) {}

    /// Returns `Some(PlayerJustMoved)` or `Some(PlayerToMove)` if there's a winner,
    /// `Some(Draw)` if the state is terminal without a winner, and `None` if
    /// the state is non-terminal.
    fn get_winner(state: &Self::S) -> Option<Winner>;

    /// Hash of the game state.
    /// Expected to be pre-calculated and cheaply updated with each apply.
    fn zobrist_hash(_state: &Self::S) -> u64 {
        unimplemented!("game has not implemented zobrist hash");
    }

    /// Optional method to return a move that does not change the board state.
    /// This does not need to be a legal move from this position, but it is
    /// used in some strategies to reject a position early if even passing gives
    /// a good position for the opponent.
    fn null_move(_state: &Self::S) -> Option<Self::M> {
        None
    }

    /// Return a human-readable notation for this move in this game state.
    fn notation(_state: &Self::S, _move: Self::M) -> Option<String> {
        None
    }
    /// Return a small index for this move for position-independent tables.
    fn table_index(_: Self::M) -> u16 {
        0
    }
    /// Maximum index value.
    fn max_table_index() -> u16 {
        0
    }
}
