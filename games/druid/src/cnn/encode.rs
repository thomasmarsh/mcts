//! Input encoding. A position becomes `IN_PLANES` `size x size` planes from the side to move's
//! point of view. Druid's split encoding makes every sub-decision of a turn (piece kind,
//! lintel orientation, cell) its own ply, so the same board is encoded again at each phase, with
//! the phase carried by constant planes. Non-spatial features (hand counts, phase, colour) are
//! constant planes too, which the plain grid net already accepts, so `gridcnn.py` needs no change.
//!
//! | plane | content |
//! |---|---|
//! | 0 | cells whose top piece is the side to move's |
//! | 1 | cells whose top piece is the opponent's |
//! | 2 | stack height / [`HEIGHT_SCALE`] |
//! | 3 | cells legal for the pending cell decision (0 while choosing a piece or an orientation) |
//! | 4 | constant 1 when Black is to move (Black joins top and bottom, White left and right) |
//! | 5 | side to move's sarsens left / `2 * cells` (the starting hand) |
//! | 6 | side to move's lintels left / `cells` |
//! | 7 | opponent's sarsens left / `2 * cells` |
//! | 8 | opponent's lintels left / `cells` |
//! | 9 | constant 1 once a sarsen is chosen |
//! | 10 | constant 1 once a lintel is chosen (orientation still to pick) |
//! | 11 | constant 1 once a horizontal lintel is chosen (cell still to pick) |
//! | 12 | constant 1 once a vertical lintel is chosen (cell still to pick) |
//! | 13 | constant 1 (lets the net see the board edge past the zero padding) |
//!
//! No connectivity information is encoded: the net has to learn it.
//!
//! Policy actions are the cells (`row * size + col`), then the four non-cell sub-decisions:
//! choose sarsen, choose lintel, orient horizontal, orient vertical. Which of them is legal
//! depends on the phase, so the legal-id list (never the plane alone) masks the policy.

use crate::{DruidSplit, HashedState, Move, Orientation, Pending, PieceKind, Player};
use mcts::game::Game;

pub const IN_PLANES: usize = 14;

/// Stack heights are divided by this so typical stacks land in [0, 1]; taller ones are not clamped.
pub const HEIGHT_SCALE: f32 = 8.0;

pub fn num_actions(size: usize) -> usize {
    size * size + 4
}

pub fn choose_sarsen_id(size: usize) -> u16 {
    (size * size) as u16
}

pub fn choose_lintel_id(size: usize) -> u16 {
    (size * size + 1) as u16
}

pub fn orient_horizontal_id(size: usize) -> u16 {
    (size * size + 2) as u16
}

pub fn orient_vertical_id(size: usize) -> u16 {
    (size * size + 3) as u16
}

pub fn action_id(mv: &Move, size: usize) -> u16 {
    match *mv {
        Move::Cell(c) => u16::from(c),
        Move::Piece(PieceKind::Sarsen) => choose_sarsen_id(size),
        Move::Piece(PieceKind::Lintel) => choose_lintel_id(size),
        Move::Orientation(Orientation::Horizontal) => orient_horizontal_id(size),
        Move::Orientation(Orientation::Vertical) => orient_vertical_id(size),
    }
}

/// The move with action id `id`; whether it is legal in a given position is the caller's business.
pub fn move_from_id(id: u16, size: usize) -> Move {
    let cells = (size * size) as u16;
    match id.checked_sub(cells) {
        None => Move::Cell(id as u8),
        Some(0) => Move::Piece(PieceKind::Sarsen),
        Some(1) => Move::Piece(PieceKind::Lintel),
        Some(2) => Move::Orientation(Orientation::Horizontal),
        Some(3) => Move::Orientation(Orientation::Vertical),
        Some(_) => panic!("action id {id} out of range for a {size}x{size} board"),
    }
}

/// The position's legal moves (in `generate_actions` order) and their ascending action ids.
pub fn legal_moves(state: &HashedState) -> (Vec<Move>, Vec<u16>) {
    let size = usize::from(state.state().size.w);
    let mut moves = Vec::new();
    DruidSplit::generate_actions(state, &mut moves);
    let mut ids: Vec<u16> = moves.iter().map(|m| action_id(m, size)).collect();
    ids.sort_unstable();
    (moves, ids)
}

/// The planes as `(row, col, plane)` floats, the layout `grid_cnn::Net::forward` takes. The board
/// must be square (the net is).
pub fn planes(state: &HashedState) -> Vec<f32> {
    let s = state.state();
    assert_eq!(s.size.w, s.size.h, "the grid net needs a square board");
    let size = usize::from(s.size.w);
    let cells = size * size;
    let (_, ids) = legal_moves(state);
    let mut legal = vec![false; cells];
    if matches!(s.pending, Pending::Piece(PieceKind::Sarsen) | Pending::Oriented(_)) {
        for &id in ids.iter().filter(|&&id| usize::from(id) < cells) {
            legal[usize::from(id)] = true;
        }
    }

    let mover = s.player;
    let (own_hand, opp_hand) = match mover {
        Player::Black => (&s.hand_black, &s.hand_white),
        Player::White => (&s.hand_white, &s.hand_black),
    };
    let sarsen_start = (2 * cells) as f32;
    let lintel_start = cells as f32;
    let hands = [
        f32::from(own_hand.sarsens) / sarsen_start,
        f32::from(own_hand.lintels) / lintel_start,
        f32::from(opp_hand.sarsens) / sarsen_start,
        f32::from(opp_hand.lintels) / lintel_start,
    ];
    let black = f32::from(mover == Player::Black);
    let phase = [
        f32::from(s.pending == Pending::Piece(PieceKind::Sarsen)),
        f32::from(s.pending == Pending::Piece(PieceKind::Lintel)),
        f32::from(s.pending == Pending::Oriented(Orientation::Horizontal)),
        f32::from(s.pending == Pending::Oriented(Orientation::Vertical)),
    ];

    let mut out = vec![0.0f32; cells * IN_PLANES];
    for (cell, sq) in s.board.iter().enumerate() {
        let at = cell * IN_PLANES;
        out[at] = f32::from(sq.piece == Some(mover));
        out[at + 1] = f32::from(sq.piece.is_some() && sq.piece != Some(mover));
        out[at + 2] = f32::from(sq.height) / HEIGHT_SCALE;
        out[at + 3] = f32::from(legal[cell]);
        out[at + 4] = black;
        out[at + 5..at + 9].copy_from_slice(&hands);
        out[at + 9..at + 13].copy_from_slice(&phase);
        out[at + 13] = 1.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Size;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn size_of(state: &HashedState) -> usize {
        usize::from(state.state().size.w)
    }

    #[test]
    fn action_ids_round_trip_and_are_dense() {
        for size in [5usize, 7, 9] {
            for id in 0..num_actions(size) as u16 {
                assert_eq!(action_id(&move_from_id(id, size), size), id);
            }
        }
    }

    #[test]
    fn opening_planes_show_hands_black_and_the_edge() {
        let size = 5;
        let s = HashedState::new(Size { w: 5, h: 5 });
        let p = planes(&s);
        assert_eq!(p.len(), size * size * IN_PLANES);
        let at = |cell: usize, plane: usize| p[cell * IN_PLANES + plane];
        assert_eq!(
            (at(3, 0), at(3, 1), at(3, 2), at(3, 3)),
            (0.0, 0.0, 0.0, 0.0),
            "empty board, and no cell decision pending"
        );
        assert_eq!((at(3, 4), at(3, 13)), (1.0, 1.0));
        assert_eq!((at(3, 5), at(3, 6), at(3, 7), at(3, 8)), (1.0, 1.0, 1.0, 1.0));
        assert_eq!((at(3, 9), at(3, 10), at(3, 11), at(3, 12)), (0.0, 0.0, 0.0, 0.0));
        let (moves, ids) = legal_moves(&s);
        assert_eq!(moves.len(), 1, "only sarsens are playable on an empty board");
        assert_eq!(ids, vec![choose_sarsen_id(size)]);
    }

    #[test]
    fn the_cell_plane_and_phase_planes_follow_the_pending_decision() {
        let size = 5;
        let s = HashedState::new(Size { w: 5, h: 5 });
        let s = DruidSplit::apply(s, &Move::Piece(PieceKind::Sarsen));
        let p = planes(&s);
        assert!((0..size * size).all(|c| p[c * IN_PLANES + 3] == 1.0));
        assert_eq!(p[9], 1.0);
        let s = DruidSplit::apply(s, &Move::Cell(12));
        let p = planes(&s);
        let at = |cell: usize, plane: usize| p[cell * IN_PLANES + plane];
        assert_eq!(at(12, 0), 0.0);
        assert_eq!((at(12, 1), at(12, 2)), (1.0, 1.0 / HEIGHT_SCALE), "White to move sees Black's sarsen");
        assert_eq!(at(0, 4), 0.0, "White to move");
        assert_eq!((at(0, 7), at(0, 5)), (1.0 - 1.0 / 50.0, 1.0), "Black spent one sarsen");
    }

    #[test]
    fn legal_ids_always_match_generate_actions_over_random_games() {
        let mut rng = SmallRng::seed_from_u64(5);
        for n in [5u8, 7] {
            for _ in 0..4 {
                let mut s = HashedState::new(Size { w: n, h: n });
                while !DruidSplit::is_terminal(&s) {
                    let (moves, ids) = legal_moves(&s);
                    assert_eq!(ids.len(), moves.len());
                    let p = planes(&s);
                    let cells = size_of(&s).pow(2);
                    let cell_ids = ids.iter().filter(|&&i| usize::from(i) < cells).count();
                    let plane_cells = (0..cells).filter(|&c| p[c * IN_PLANES + 3] == 1.0).count();
                    assert_eq!(plane_cells, cell_ids);
                    let mv = moves[rng.gen_range(0..moves.len())];
                    assert_eq!(move_from_id(action_id(&mv, size_of(&s)), size_of(&s)), mv);
                    s = DruidSplit::apply(s, &mv);
                }
            }
        }
    }
}
