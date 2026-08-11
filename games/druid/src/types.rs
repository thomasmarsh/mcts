//! Druid's data types: board geometry, pieces, hands, and the per-turn
//! sub-action phase. Pure data (plus a couple of tiny helpers) with no logic
//! dependencies on the rest of the crate, so `State`/`zobrist`/`moves` can
//! all build on it independently.

use serde::{Deserialize, Serialize};

use crate::zobrist::{zobrist_height_bits, HAND_HASHES_LEN, HASHES_LEN};
use mcts::game::PlayerIndex;

// NOTE: the standard game is 10x10 (and 9x9 for Trilith). Board size lives on
// `State` (see `Size::is_supported` below for the ceiling this is checked
// against) rather than here; this constant now only supplies the default
// size for `State::default()` / existing tests and demo binaries.
pub const DEFAULT_SIZE: Size = Size { w: 5, h: 5 };

#[derive(PartialEq, Clone, Copy, Debug, Serialize, Deserialize, Hash, Eq)]
pub enum Player {
    Black,
    White,
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

impl Player {
    /// Advance the player whose turn it is, in place.
    pub(crate) fn next(&mut self) {
        *self = match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Size {
    pub w: u8,
    pub h: u8,
}

impl Size {
    pub(crate) fn area(self) -> u16 {
        (self.w * self.h) as u16
    }

    /// Whether this size is safe to build a game on: big enough for a lintel
    /// to fit in either orientation, and small enough that the Zobrist hash
    /// (see `HASHES` in `zobrist`) can address every (position, color,
    /// height-bit) slot it needs without going out of bounds.
    pub fn is_supported(self) -> bool {
        if self.w < 3 || self.h < 3 {
            return false;
        }
        let area = self.area() as usize;
        area * 2 * zobrist_height_bits(self) <= HASHES_LEN
            && 4 * zobrist_height_bits(self) <= HAND_HASHES_LEN
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Pos(pub u8, pub u8);

impl Pos {
    pub fn from(i: usize, size: Size) -> Pos {
        Pos(i as u8 % size.w, i as u8 / size.w)
    }

    pub fn index(self, width: u8) -> usize {
        (self.1 * width + self.0) as usize
    }

    pub(crate) fn adjacent(&self, size: Size) -> impl Iterator<Item = Pos> {
        let &Pos(x, y) = self;

        [(-1, 0), (1, 0), (0, -1), (0, 1)]
            .into_iter()
            .filter_map(move |(dx, dy)| {
                let nx = x as i8 + dx;
                let ny = y as i8 + dy;
                if (0..size.w as i8).contains(&nx) && (0..size.h as i8).contains(&ny) {
                    Some(Pos(nx as u8, ny as u8))
                } else {
                    None
                }
            })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

impl Orientation {
    pub(crate) fn delta(self) -> (u8, u8) {
        match self {
            Orientation::Horizontal => (1, 0),
            Orientation::Vertical => (0, 1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Piece {
    Sarsen,
    Lintel(Orientation),
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Square {
    pub height: u16,
    pub piece: Option<Player>,
}

impl Square {
    pub(crate) fn matches(&self, color: Player) -> bool {
        self.piece.is_some_and(|p| p == color)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceKind {
    Sarsen,
    Lintel,
}

impl Piece {
    pub fn kind(self) -> PieceKind {
        match self {
            Piece::Sarsen => PieceKind::Sarsen,
            Piece::Lintel(_) => PieceKind::Lintel,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlacedPiece(pub Piece, pub u8);

/// The phase of a turn in the move-split representation: how many of a
/// whole-turn placement's sub-actions (`Piece`? `Orientation`? `Cell`) the
/// current node has committed to. The flat representation always carries
/// `Pending::None` (it applies a whole turn as one action), which is exactly
/// the state a real position is in between turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Pending {
    #[default]
    None,
    Piece(PieceKind),
    Oriented(Orientation),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hand {
    pub sarsens: u8,
    pub lintels: u8,
}

impl Hand {
    pub fn new(size: Size) -> Hand {
        let n = size.w * size.h;
        // Trilith provides 48 sarsens and 20 lintels for a 9x9 board, which
        // is probably too few.
        //
        // Cameron Browne says:
        // > For a 10x10 board you'll need at least 100 cubes in
        // > total (enough to cover the board). A good distribution is 20x1 unit
        // > and 10x3 unit blocks per player.
        // >
        // > This will be sufficient for games that don't go on too long. If the
        // > games  get really involved, however, you'll run out of pieces in
        // > which case you  might:
        // > 1) Pick up a piece already on the board (provided that it's
        // >    reachable) and place it elsewhere, or
        // > 2) Use twice as many pieces :)
        // >
        // > If you're playing the "Druid's Walk" option each player will also
        // > require one pawn.
        //
        // For this game, for an NxM board we use N*M sarsens and half as
        // many lintels.
        Hand {
            sarsens: n * 2,
            lintels: n,
        }
    }
}
