//! A static evaluation function for Breakthrough (`mcts::evaluator::
//! Evaluator`), consulted at a search's depth cutoff in place of playing
//! out to a terminal state.
//!
//! Two terms, matching the shape of the evaluation functions used in the
//! Breakthrough literature this hybridization work is based on (Baier &
//! Winands' MCTS-minimax papers use Breakthrough as their test game):
//! material (piece count difference) and advancement (how close each
//! piece is to its own goal row, weighted more heavily the closer it
//! gets -- a single further step can end the game outright, unlike in
//! most other board games). This is a straightforward hand-written
//! heuristic, not a transcription of either paper's exact tuned weights.

use bitboard::{Board, Const};

use mcts::evaluator::{Evaluator, Score, DRAW_SCORE, EVAL_MAGNITUDE_LIMIT};

use crate::{Breakthrough, Player, State};

type BitBoard<const N: usize, const M: usize> = Board<u64, Const<N>, Const<M>>;

/// Value of a single piece, in the same units as [`advancement_value`]'s
/// output -- chosen so early-game material dominates over advancement
/// (which starts near 0 for every piece at its start row), and only
/// pieces that have actually pushed deep into enemy territory start to
/// compete with material.
const MATERIAL_WEIGHT: i32 = 100;

/// Quadratic in `progress` (0 at the start row, `goal_distance - 1` one
/// step from winning) so a piece's danger grows faster than linearly as it
/// nears the goal -- a lone breakthrough on the last rank can win outright
/// regardless of material elsewhere on the board.
fn advancement_value(progress: usize) -> i32 {
    (progress * progress) as i32
}

/// Sum of [`advancement_value`] over every set cell in `pieces`, where
/// `progress(row)` is `row`'s distance already traveled toward `goal_row`
/// (0 at the piece's own start side, increasing toward the opponent's edge).
fn advancement_total<const N: usize, const M: usize>(
    pieces: BitBoard<N, M>,
    progress: impl Fn(usize) -> usize,
) -> i32 {
    pieces
        .iter_set()
        .map(|i| {
            let (row, _col) = BitBoard::<N, M>::to_coord(i);
            advancement_value(progress(row))
        })
        .sum()
}

/// `state`'s value from Black's perspective: positive favors Black,
/// negative favors White. Black starts near row `0` and advances toward
/// row `N - 1`; White is the mirror image.
fn black_relative_value<const N: usize, const M: usize>(state: &State<N, M>) -> i32 {
    let black = state.black();
    let white = state.white();

    let material = MATERIAL_WEIGHT * (black.count_ones() as i32 - white.count_ones() as i32);
    let advancement =
        advancement_total(black, |row| row) - advancement_total(white, |row| N - 1 - row);

    material + advancement
}

/// [`Evaluator`] for [`Breakthrough`]: material plus weighted advancement
/// (see the module docs), clamped to `EVAL_MAGNITUDE_LIMIT` so it can never
/// be confused with a mate-distance-adjusted `WIN_SCORE`/`LOSS_SCORE`.
#[derive(Clone, Copy, Default)]
pub struct Heuristic;

impl<const N: usize, const M: usize> Evaluator<Breakthrough<N, M>> for Heuristic {
    fn evaluate(&self, state: &State<N, M>) -> Score {
        let black_relative = black_relative_value(state);
        let mover_relative = match state.turn() {
            Player::Black => black_relative,
            Player::White => -black_relative,
        };
        mover_relative.clamp(-EVAL_MAGNITUDE_LIMIT, EVAL_MAGNITUDE_LIMIT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_position_is_symmetric() {
        let state = State::<8, 8>::default();
        assert_eq!(Heuristic.evaluate(&state), DRAW_SCORE);
    }

    #[test]
    fn material_deficit_is_worse_for_its_owner() {
        let state = State::<8, 8>::default();
        let removed = state.white().iter_set().next().unwrap();
        let thin_white = state.white() & !BitBoard::<8, 8>::from_index(removed);
        let down_a_piece = State::new(state.black(), thin_white, Player::White, false);

        // It's White to move and White is down a piece -- worse for the
        // mover than the balanced start position.
        assert!(Heuristic.evaluate(&down_a_piece) < Heuristic.evaluate(&state));
    }

    #[test]
    fn advanced_piece_is_worth_more_than_a_piece_at_home() {
        let advanced = State::new(
            BitBoard::<8, 8>::from_coord(6, 0),
            BitBoard::<8, 8>::EMPTY,
            Player::Black,
            false,
        );
        let at_home = State::new(
            BitBoard::<8, 8>::from_coord(1, 0),
            BitBoard::<8, 8>::EMPTY,
            Player::Black,
            false,
        );
        assert!(Heuristic.evaluate(&advanced) > Heuristic.evaluate(&at_home));
    }
}
