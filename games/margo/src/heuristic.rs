//! A static evaluation function for Margo (`mcts::evaluator::Evaluator`),
//! consulted at a search's depth cutoff in place of playing out to a
//! terminal state.
//!
//! Two terms, both cheap to compute from the same visible-board/flood-fill
//! machinery [`crate::resolve_captures`] already uses: height-weighted
//! material (a piece on a higher level is harder to dislodge and blocks
//! more of the board below it, so it's worth more than a level-0 piece) and
//! a liberty term over each group's board-level freedoms (see the module
//! docs on "Freedoms only exist on the board level") -- a heavy penalty for
//! a group down to its last liberty (in atari, capturable by a single enemy
//! placement) and a small bonus for a group with three or more. Plain piece
//! counting sees neither a piece about to be captured nor a piece worth
//! more for the connections it cuts.

use pyramid::{self, TouchingAdjacency};

use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};

use crate::{ground_mask, visible_boards, GoBoard, Margo, Player, State};

/// Value of a single level-0 piece; a piece on level `l` is worth
/// `MATERIAL_WEIGHT * (l + 1)`, so e.g. a level-2 piece is worth 3x as much
/// as a piece resting on the board level.
const MATERIAL_WEIGHT: i32 = 100;

/// Penalty applied per own group sitting at exactly one liberty (atari --
/// capturable by a single enemy placement next).
const ATARI_PENALTY: i32 = 150;

/// Bonus applied per own group with three or more liberties (safely
/// connected, in no near-term danger of capture).
const SAFE_GROUP_BONUS: i32 = 20;

/// Sum of [`MATERIAL_WEIGHT`]-per-level over every occupied cell, positive
/// for Black's pieces and negative for White's -- including buried/zombie
/// cells, which still count toward the piece-count win condition
/// ([`Margo::winner`]) even though they're excluded from connectivity.
fn height_weighted_material(state: &State) -> i32 {
    let mut total = 0i32;
    for index in state.occupied.iter_set() {
        let (_, _, level) = state.occupied.to_coord(index);
        let weight = MATERIAL_WEIGHT * (level as i32 + 1);
        total += if state.black.get_index(index) {
            weight
        } else {
            -weight
        };
    }
    total
}

/// Sum of an atari penalty/safe-group bonus over every group in `own`,
/// flood-filled over the touching-adjacency graph exactly as
/// [`crate::resolve_captures`] does, with liberties counted the same way
/// (empty board-level cells adjacent to the group, `own`/`opp` both
/// restricted to visible -- non-buried, non-zombie -- stones).
fn liberty_term(own: GoBoard, opp: GoBoard, ground: GoBoard, adjacency: &TouchingAdjacency) -> i32 {
    let occupied = own | opp;
    let mut seen = own.empty_like();
    let mut score = 0;
    for point in own {
        if seen.get_index(point) {
            continue;
        }
        let group = bitboard::table_flood(own, adjacency, point);
        seen |= group;
        let group_adjacent = bitboard::table_neighbor_mask(group, adjacency);
        let liberties = (!occupied & group_adjacent & ground).count_ones();
        score += match liberties {
            0 | 2 => 0,
            1 => -ATARI_PENALTY,
            _ => SAFE_GROUP_BONUS,
        };
    }
    score
}

/// `state`'s value from Black's perspective: positive favors Black,
/// negative favors White.
fn black_relative_value(state: &State) -> i32 {
    let material = height_weighted_material(state);

    let adjacency = pyramid::get_adjacency(state.occupied.n());
    let ground = ground_mask(state.occupied.n(), state.occupied.total_cells());
    let (black_board, white_board) = visible_boards(&state.occupied, &state.black);
    let liberties = liberty_term(black_board, white_board, ground, adjacency)
        - liberty_term(white_board, black_board, ground, adjacency);

    material + liberties
}

/// [`Evaluator`] for [`Margo`]: height-weighted material plus a liberty
/// term (see the module docs), clamped to `EVAL_MAGNITUDE_LIMIT` so it can
/// never be confused with a mate-distance-adjusted `WIN_SCORE`/`LOSS_SCORE`.
#[derive(Clone, Copy, Default)]
pub struct Heuristic;

impl Evaluator<Margo> for Heuristic {
    fn evaluate(&self, state: &State) -> Score {
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
    use mcts::evaluator::DRAW_SCORE;

    #[test]
    fn empty_board_is_symmetric() {
        let state = State::new(crate::DEFAULT_N);
        assert_eq!(Heuristic.evaluate(&state), DRAW_SCORE);
    }

    #[test]
    fn material_deficit_is_worse_for_its_owner() {
        let mut state = State::new(crate::DEFAULT_N);
        let black_cell = state.occupied.index(0, 0, 0);
        state.occupied.set_index(black_cell);
        state.black.set_index(black_cell);
        state.turn = Player::White;

        // White to move, down a piece -- worse than the balanced empty board.
        assert!(Heuristic.evaluate(&state) < Heuristic.evaluate(&State::new(crate::DEFAULT_N)));
    }

    #[test]
    fn a_group_in_atari_is_penalized() {
        let mut state = State::new(crate::DEFAULT_N);
        // A lone white stone at a corner has two liberties (its two lateral
        // neighbors); surrounding one with black drops it to one liberty
        // (atari) without capturing it.
        let white_cell = state.occupied.index(0, 0, 0);
        state.occupied.set_index(white_cell);
        let safe = black_relative_value(&state);

        let black_cell = state.occupied.index(1, 0, 0);
        state.occupied.set_index(black_cell);
        state.black.set_index(black_cell);
        let in_atari = black_relative_value(&state);

        // White's group went from two liberties to one: worse for White,
        // i.e. the black-relative score should rise by more than the
        // material swing alone (one black stone at level 0) would explain.
        assert!(in_atari - safe > MATERIAL_WEIGHT);
    }
}
