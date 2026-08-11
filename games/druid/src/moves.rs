//! The two move encodings of the same Druid game, selected by `Druid<M>`'s
//! type parameter. Both drive the shared board `State`/`MoveCache` core; they
//! differ only in how a whole-turn placement is exposed as `Game` actions:
//!
//! - `Split` (shipped): the linearized `Piece`/`Orientation`/`Cell`
//!   sub-action sequence, tracked by `State::pending`. This is what the
//!   server binary and presets play.
//! - `Flat` (pre-move-splitting snapshot): a `PlacedPiece` is the whole
//!   action. Kept solely for `examples/strength_move_splitting.rs` to pit the
//!   two representations against each other in one binary; not wired into the
//!   server. Do not add features here.

use mcts::game::Action;

use crate::game::{apply_turn, HashedState};
use crate::heuristics::{max_heuristic_for_cells, DruidHeuristicWeights};
use crate::types::{Orientation, Pending, Piece, PieceKind, PlacedPiece, Player, Pos};
use crate::zobrist::pending_zobrist;

/// Everything a `Game` impl needs from the move encoding that isn't shared
/// board logic: how to enumerate, apply, notate, and heuristic-score actions.
pub trait MoveEncoding: Copy + Default + Send + Sync + 'static {
    type Action: Action;

    fn generate_actions(state: &HashedState, out: &mut Vec<Self::Action>);
    fn apply(state: HashedState, action: &Self::Action) -> HashedState;
    fn notation(state: &HashedState, action: &Self::Action) -> String;

    /// Per-action heuristic score for `mover` to play, higher is better --
    /// the playout-policy side of `moves` (flat scores the whole-turn
    /// placement; split scores each sub-action by the placements it can
    /// represent). See `heuristics::heuristic_scores`.
    fn score_action(
        state: &HashedState,
        mover: Player,
        action: &Self::Action,
        w: &DruidHeuristicWeights,
    ) -> f64;
}

// ---------------------------------------------------------------------------
// Split (the shipped Piece/Orientation/Cell sub-action encoding)
// ---------------------------------------------------------------------------

/// Marker selecting the move-split encoding.
#[derive(Clone, Copy, Debug, Default)]
pub struct Split;

/// A single move-split sub-action: choose a piece kind, choose a lintel
/// orientation, or place on a cell. `Piece`/`Orientation` only advance
/// `State::pending`; `Cell` completes the turn's placement.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Move {
    Piece(PieceKind),
    Orientation(Orientation),
    Cell(u8),
}

impl MoveEncoding for Split {
    type Action = Move;

    fn generate_actions(state: &HashedState, actions: &mut Vec<Move>) {
        let s = &state.0;
        let hand = s.current_hand();
        let candidates = state.3.candidates(s.player);
        match s.pending {
            Pending::None => {
                if hand.sarsens > 0 && candidates.has_any_sarsen() {
                    actions.push(Move::Piece(PieceKind::Sarsen));
                }
                if hand.lintels > 0 && candidates.has_any_lintel() {
                    actions.push(Move::Piece(PieceKind::Lintel));
                }
            }
            Pending::Piece(PieceKind::Sarsen) => {
                for i in 0..s.size.area() as usize {
                    if candidates.sarsen(i) {
                        actions.push(Move::Cell(i as u8));
                    }
                }
            }
            Pending::Piece(PieceKind::Lintel) => {
                if candidates.has_any_lintel_orient(Orientation::Horizontal) {
                    actions.push(Move::Orientation(Orientation::Horizontal));
                }
                if candidates.has_any_lintel_orient(Orientation::Vertical) {
                    actions.push(Move::Orientation(Orientation::Vertical));
                }
            }
            Pending::Oriented(o) => {
                for i in 0..s.size.area() as usize {
                    if candidates.lintel(o, i) {
                        actions.push(Move::Cell(i as u8));
                    }
                }
            }
        }
    }

    fn apply(mut state: HashedState, m: &Move) -> HashedState {
        match *m {
            Move::Piece(kind) => {
                debug_assert_eq!(state.0.pending, Pending::None);
                let old = state.0.pending;
                state.0.pending = Pending::Piece(kind);
                let mut hash = state.1;
                hash ^= pending_zobrist(old);
                hash ^= pending_zobrist(state.0.pending);
                state.1 = hash;
                state
            }
            Move::Orientation(o) => {
                debug_assert_eq!(state.0.pending, Pending::Piece(PieceKind::Lintel));
                let old = state.0.pending;
                state.0.pending = Pending::Oriented(o);
                let mut hash = state.1;
                hash ^= pending_zobrist(old);
                hash ^= pending_zobrist(state.0.pending);
                state.1 = hash;
                state
            }
            Move::Cell(idx) => {
                let piece = match state.0.pending {
                    Pending::Piece(PieceKind::Sarsen) => Piece::Sarsen,
                    Pending::Oriented(o) => Piece::Lintel(o),
                    _ => unreachable!("Cell action with pending {:?}", state.0.pending),
                };
                let placed = PlacedPiece(piece, idx);
                let old_pending = state.0.pending;
                let mut state = apply_turn(state, placed);
                // `apply_turn` flipped the player and mutated the board/hands,
                // but left `pending` and its hash contribution untouched; now
                // reset it to `None` and pay the phase-transition hash delta.
                let mut hash = state.1;
                hash ^= pending_zobrist(old_pending);
                hash ^= pending_zobrist(Pending::None);
                state.1 = hash;
                state.0.pending = Pending::None;
                state
            }
        }
    }

    fn notation(state: &HashedState, m: &Move) -> String {
        match *m {
            Move::Piece(k) => format!("choose {:?}", k),
            Move::Orientation(o) => format!("orient {:?}", o),
            Move::Cell(idx) => {
                let Pos(x, y) = Pos::from(idx as usize, state.0.size);
                let piece = match state.0.pending {
                    Pending::Piece(PieceKind::Sarsen) => Piece::Sarsen,
                    Pending::Oriented(o) => Piece::Lintel(o),
                    _ => return format!("Cell({},{})", x + 1, y + 1),
                };
                match piece {
                    Piece::Sarsen => format!("S({},{})", x + 1, y + 1),
                    Piece::Lintel(Orientation::Horizontal) => format!("L({},{},H)", x + 1, y + 1),
                    Piece::Lintel(Orientation::Vertical) => format!("L({},{},V)", x + 1, y + 1),
                }
            }
        }
    }

    fn score_action(
        state: &HashedState,
        mover: Player,
        m: &Move,
        w: &DruidHeuristicWeights,
    ) -> f64 {
        let s = &state.0;
        let candidates = state.3.candidates(mover);
        match *m {
            Move::Piece(PieceKind::Sarsen) => {
                let cells: Vec<usize> = (0..s.size.area() as usize)
                    .filter(|&i| candidates.sarsen(i))
                    .collect();
                max_heuristic_for_cells(state, mover, Piece::Sarsen, &cells, w)
            }
            Move::Piece(PieceKind::Lintel) => {
                let mut best = f64::NEG_INFINITY;
                for o in [Orientation::Horizontal, Orientation::Vertical] {
                    let piece = Piece::Lintel(o);
                    let cells: Vec<usize> = (0..s.size.area() as usize)
                        .filter(|&i| candidates.lintel(o, i))
                        .collect();
                    let v = max_heuristic_for_cells(state, mover, piece, &cells, w);
                    if v > best {
                        best = v;
                    }
                }
                if best == f64::NEG_INFINITY {
                    0.0
                } else {
                    best
                }
            }
            Move::Orientation(o) => {
                let piece = Piece::Lintel(o);
                let cells: Vec<usize> = (0..s.size.area() as usize)
                    .filter(|&i| candidates.lintel(o, i))
                    .collect();
                max_heuristic_for_cells(state, mover, piece, &cells, w)
            }
            Move::Cell(idx) => {
                let piece = match s.pending {
                    Pending::Piece(PieceKind::Sarsen) => Piece::Sarsen,
                    Pending::Oriented(o) => Piece::Lintel(o),
                    _ => Piece::Sarsen,
                };
                let placed = [PlacedPiece(piece, idx)];
                super::heuristics::heuristic_scores(state, mover, &placed, w)[0]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flat (whole-turn `PlacedPiece` encoding)
// ---------------------------------------------------------------------------

/// Marker selecting the flat (whole-turn `PlacedPiece`) encoding.
#[derive(Clone, Copy, Debug, Default)]
pub struct Flat;

impl MoveEncoding for Flat {
    type Action = PlacedPiece;

    fn generate_actions(state: &HashedState, actions: &mut Vec<PlacedPiece>) {
        let s = &state.0;
        let hand = s.current_hand();
        let candidates = state.3.candidates(s.player);
        for i in 0..s.size.area() as usize {
            if hand.sarsens > 0 && candidates.sarsen(i) {
                actions.push(PlacedPiece(Piece::Sarsen, i as u8));
            }
            if hand.lintels > 0 {
                if candidates.lintel(Orientation::Horizontal, i) {
                    actions.push(PlacedPiece(Piece::Lintel(Orientation::Horizontal), i as u8));
                }
                if candidates.lintel(Orientation::Vertical, i) {
                    actions.push(PlacedPiece(Piece::Lintel(Orientation::Vertical), i as u8));
                }
            }
        }
    }

    fn apply(state: HashedState, m: &PlacedPiece) -> HashedState {
        apply_turn(state, *m)
    }

    fn notation(state: &HashedState, m: &PlacedPiece) -> String {
        let Pos(x, y) = Pos::from(m.1 as usize, state.0.size);
        match m.0 {
            Piece::Sarsen => format!("S({},{})", x + 1, y + 1),
            Piece::Lintel(Orientation::Horizontal) => format!("L({},{},H)", x + 1, y + 1),
            Piece::Lintel(Orientation::Vertical) => format!("L({},{},V)", x + 1, y + 1),
        }
    }

    fn score_action(
        state: &HashedState,
        mover: Player,
        m: &PlacedPiece,
        w: &DruidHeuristicWeights,
    ) -> f64 {
        super::heuristics::heuristic_scores(state, mover, &[*m], w)[0]
    }
}