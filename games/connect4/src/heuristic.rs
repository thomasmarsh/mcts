//! A static evaluation function for Connect Four (`mcts::evaluator::
//! Evaluator`), consulted at a search's depth cutoff in place of playing
//! out to a terminal state.
//!
//! Classic "window scoring": every length-4 line on the board (horizontal,
//! vertical, both diagonals) is a potential win for whichever color holds
//! it uncontested. A window already containing both colors can never
//! become a win for either, so it scores 0; a window holding only one
//! color scores by how many of that color are already in it (a 3-in-a-row
//! with the 4th cell still open is far more dangerous than a lone piece).
//! This is the standard textbook Connect Four heuristic, not a tuned or
//! literature-sourced one -- unlike Breakthrough (whose `Heuristic`
//! mirrors specific published weights), Connect Four's MCTS-minimax
//! hybrid literature doesn't fix a canonical evaluator to match.

use bitboard::{Board, Const};

use mcts::evaluator::{Evaluator, Score, DRAW_SCORE, EVAL_MAGNITUDE_LIMIT};

use crate::{Connect4, Player, State};

type BitBoard<const R: usize, const C: usize> = Board<u64, Const<R>, Const<C>>;

/// Score contributed by a window containing exactly `n` of one color and no
/// discs of the other, indexed by `n` (index 0 unused: an empty window
/// contributes nothing). `n == 4` would already be a real win, caught by
/// `Game::is_terminal` before evaluation is ever consulted -- included
/// anyway so a window that happens to hold 4 doesn't silently score as if
/// it held 3.
const WINDOW_WEIGHT: [i32; 5] = [0, 1, 10, 50, 100_000];

/// Small bonus per own piece in the center column, minus the same for the
/// opponent's -- the center column participates in more length-4 windows
/// (horizontal, both diagonals) than any other, so pieces there are worth
/// slightly more even before any of those windows fills in. Chosen well
/// below `WINDOW_WEIGHT[1]` so it only breaks ties among otherwise-equal
/// window scores, not override them.
const CENTER_WEIGHT: i32 = 1;

/// Scores every length-4 window on the board from `mine`'s perspective:
/// positive favors `mine`, negative favors `theirs`.
fn window_score<const R: usize, const C: usize>(
    mine: BitBoard<R, C>,
    theirs: BitBoard<R, C>,
) -> i32 {
    let score_window = |cells: [(usize, usize); 4]| -> i32 {
        let mut my_count = 0;
        let mut their_count = 0;
        for (row, col) in cells {
            if mine.get(row, col) {
                my_count += 1;
            } else if theirs.get(row, col) {
                their_count += 1;
            }
        }
        match (my_count, their_count) {
            (m, 0) if m > 0 => WINDOW_WEIGHT[m],
            (0, t) if t > 0 => -WINDOW_WEIGHT[t],
            _ => 0,
        }
    };

    let mut score = 0;

    if C >= 4 {
        for row in 0..R {
            for col in 0..=(C - 4) {
                score += score_window([(row, col), (row, col + 1), (row, col + 2), (row, col + 3)]);
            }
        }
    }
    if R >= 4 {
        for col in 0..C {
            for row in 0..=(R - 4) {
                score += score_window([(row, col), (row + 1, col), (row + 2, col), (row + 3, col)]);
            }
        }
    }
    if R >= 4 && C >= 4 {
        for row in 0..=(R - 4) {
            for col in 0..=(C - 4) {
                score += score_window([
                    (row, col),
                    (row + 1, col + 1),
                    (row + 2, col + 2),
                    (row + 3, col + 3),
                ]);
                score += score_window([
                    (row, col + 3),
                    (row + 1, col + 2),
                    (row + 2, col + 1),
                    (row + 3, col),
                ]);
            }
        }
    }

    score
}

fn center_score<const R: usize, const C: usize>(
    mine: BitBoard<R, C>,
    theirs: BitBoard<R, C>,
) -> i32 {
    let center = C / 2;
    let count = |b: BitBoard<R, C>| (0..R).filter(|&row| b.get(row, center)).count() as i32;
    CENTER_WEIGHT * (count(mine) - count(theirs))
}

/// `state`'s value from Black's perspective: positive favors Black,
/// negative favors White.
fn black_relative_value<const R: usize, const C: usize>(state: &State<R, C>) -> i32 {
    let black = state.black();
    let white = state.white();
    window_score(black, white) + center_score(black, white)
}

/// [`Evaluator`] for [`Connect4`]: window scoring plus a small center-column
/// bonus (see the module docs), clamped to `EVAL_MAGNITUDE_LIMIT` so it can
/// never be confused with a mate-distance-adjusted `WIN_SCORE`/`LOSS_SCORE`.
#[derive(Clone, Copy, Default)]
pub struct Heuristic;

impl<const R: usize, const C: usize> Evaluator<Connect4<R, C>> for Heuristic {
    fn evaluate(&self, state: &State<R, C>) -> Score {
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

    type Std = BitBoard<6, 7>;

    #[test]
    fn empty_board_is_symmetric() {
        let state = State::<6, 7>::default();
        assert_eq!(Heuristic.evaluate(&state), DRAW_SCORE);
    }

    #[test]
    fn three_in_a_row_with_an_open_end_favors_its_owner() {
        let empty = State::<6, 7>::default();

        // Black holds an open three on row 0 (columns 0-2, column 3 still
        // empty) with no other pieces on the board -- built directly via
        // `from_parts` rather than simulated alternating play, so nothing
        // confounds the comparison with an incidental white formation.
        let mut black = Std::EMPTY;
        black.set(0, 0);
        black.set(0, 1);
        black.set(0, 2);
        let state = State::from_parts(black, Std::EMPTY, Player::Black, false);

        // It's Black to move next; Black's open three should score better
        // for the mover than the empty board.
        assert!(Heuristic.evaluate(&state) > Heuristic.evaluate(&empty));
    }

    #[test]
    fn center_column_piece_is_worth_more_than_an_edge_piece() {
        let mut center_board = Std::EMPTY;
        center_board.set(0, 3);
        let center = State::from_parts(center_board, Std::EMPTY, Player::White, false);

        let mut edge_board = Std::EMPTY;
        edge_board.set(0, 0);
        let edge = State::from_parts(edge_board, Std::EMPTY, Player::White, false);

        // Both are White to move next with Black down neither material nor
        // any completed window -- the only difference is which column
        // Black's lone piece sits in.
        assert!(Heuristic.evaluate(&center) < Heuristic.evaluate(&edge));
    }
}
