//! A static evaluation of a non-terminal state -- opt-in, per-game, and
//! independent of any particular search strategy, so any strategy (not
//! just `strategies::negamax`, its original and so-far-only consumer) can
//! depend on it without depending on that module. `strategies::negamax`
//! re-exports everything in this module, so existing `negamax::Evaluator`/
//! `negamax::MaterialBlind`/etc. call sites are unaffected by where it
//! actually lives.

use crate::game::Game;

/// Score type returned by [`Evaluator::evaluate`], always from the
/// perspective of the player about to move in the state being scored (the
/// "nega" convention: a child's value is negated to become its parent's).
pub type Score = i32;

/// A proven win for the player to move. Kept well below `i32::MAX` so
/// mate-distance adjustments (`WIN_SCORE - ply`) and aspiration-window
/// arithmetic (`target +/- window`) can't overflow.
pub const WIN_SCORE: Score = 1_000_000;
pub const LOSS_SCORE: Score = -WIN_SCORE;
pub const DRAW_SCORE: Score = 0;

/// Evaluators should stay within this band so a heuristic score can never
/// be confused with a mate-distance-adjusted `WIN_SCORE`/`LOSS_SCORE`
/// (which live in `[WIN_SCORE - max_depth, WIN_SCORE]` and the mirror image
/// below zero, for any `max_depth` this crate would realistically be
/// configured with).
pub const EVAL_MAGNITUDE_LIMIT: Score = 900_000;

/// A static evaluation of a non-terminal state, from the perspective of
/// `Game::player_to_move(state)`. Only meant to be consulted at a search's
/// depth cutoff (terminal states should always be scored from
/// `Game::terminal_status` instead) -- a game small enough to search out to
/// a terminal state at whatever depth it's configured with doesn't need one
/// at all (see [`MaterialBlind`] below).
///
/// This is intentionally not part of `Game` itself: most games plugged into
/// this crate have no static evaluator (they're built for MCTS, whose
/// rollouts don't need one), and folding an `evaluate` method into `Game`
/// would force every one of them to grow a stub. Implement this trait only
/// for the games (and only in the crates) that actually want it.
pub trait Evaluator<G: Game>: Sync + Send {
    fn evaluate(&self, state: &G::S) -> Score;
}

/// An [`Evaluator`] that always returns a draw score, for a game whose
/// state space is small enough that a depth cutoff never actually fires --
/// the value returned then doesn't matter. Also useful as a placeholder
/// while a real evaluator is still being written.
#[derive(Clone, Copy, Default)]
pub struct MaterialBlind;

impl<G: Game> Evaluator<G> for MaterialBlind {
    fn evaluate(&self, _state: &G::S) -> Score {
        DRAW_SCORE
    }
}
