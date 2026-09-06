// Flood fill

impl<S: Storage, R: Dim, C: Dim> Board<S, R, C> {
    /// Performs a four-way floodfill traversing set bits, starting from
    /// `start` (a row-major index, matching `iter_set`'s output). It might
    /// seem more natural to fill unset bits, but that requires one
    /// additional operation in this function, so that decision is up to the
    /// client.
    pub fn flood4(self, start: usize) -> Self {
        debug_assert!(start < self.len());
        let mut seed = Self::new(self.rows, self.cols);
        seed.set_index(start);
        let mut flood = seed & self;

        if flood.bits_empty() {
            return flood;
        }

        loop {
            let temp = flood;
            flood = flood
                | flood.shift_north()
                | flood.shift_east()
                | flood.shift_south()
                | flood.shift_west();
            flood &= self;
            if flood == temp {
                break;
            }
        }
        flood
    }

    /// Performs an eight-way floodfill traversing set bits, starting from
    /// `start`. Mirrors `BitBoard::flood8` (`BigBitBoard` doesn't implement
    /// this): unlike `flood4`, the north/south shifts are OR'd into `flood`
    /// in one statement *before* the east/west shifts are computed off that
    /// updated value, in a second statement -- so `flood.shift_east()` in
    /// the second statement also reaches cells diagonal to the original
    /// `flood` (`east(north(x))` = northeast of `x`, etc.), giving full
    /// 8-way connectivity through compounding, without a real diagonal
    /// shift primitive.
    pub fn flood8(self, start: usize) -> Self {
        debug_assert!(start < self.len());
        let mut seed = Self::new(self.rows, self.cols);
        seed.set_index(start);
        let mut flood = seed & self;

        if flood.bits_empty() {
            return flood;
        }

        loop {
            let temp = flood;
            flood = flood | flood.shift_north() | flood.shift_south();
            flood = flood | flood.shift_east() | flood.shift_west();
            flood &= self;
            if flood == temp {
                break;
            }
        }
        flood
    }

    /// Performs a six-way (hex) floodfill traversing set bits, seeded from
    /// every set bit of `seed` rather than a single start index -- unlike
    /// `flood4`/`flood8`, a hex connection check (e.g. "does the mover's
    /// group touch both of their board edges") needs to seed from an entire
    /// edge region, which may contain several disconnected stones.
    ///
    /// All six shifts must be OR'd into `flood` in a single expression, as
    /// `flood4` does (unlike `flood8`, which splits its four shifts across
    /// two statements) -- splitting them here would let one direction's
    /// shift compound off a *previous statement's* unmasked result within
    /// the same iteration, bridging through a cell not actually in `self`.
    /// Six-way adjacency deliberately excludes the northwest/southeast
    /// diagonal, so that compounding would wrongly treat it as connected.
    pub fn flood6(self, seed: Self) -> Self {
        let mut flood = seed & self;

        if flood.bits_empty() {
            return flood;
        }

        loop {
            let temp = flood;
            flood = flood
                | flood.shift_north()
                | flood.shift_south()
                | flood.shift_east()
                | flood.shift_west()
                | flood.shift_northeast()
                | flood.shift_southwest();
            flood &= self;
            if flood == temp {
                break;
            }
        }
        flood
    }
}

use super::{Board, Direction};
use crate::dim::Dim;
use crate::storage::Storage;

//////////////////////////////////////////////////////////////////////////////////////////////////

// Connectivity tests

impl<S: Storage, R: Dim, C: Dim> Board<S, R, C> {
    #[inline]
    pub fn has_opposite_connection4(self, start: usize) -> bool {
        let n = self.wall(Direction::North);
        let e = self.wall(Direction::East);
        let s = self.wall(Direction::South);
        let w = self.wall(Direction::West);

        let mut seed = Self::new(self.rows, self.cols);
        seed.set_index(start);
        let mut flood = seed & self;

        if flood.bits_empty() {
            return false;
        }

        loop {
            let temp = flood;
            flood = flood
                | flood.shift_north()
                | flood.shift_east()
                | flood.shift_south()
                | flood.shift_west();
            flood &= self;
            if (flood.intersects(n) && flood.intersects(s))
                || (flood.intersects(e) && flood.intersects(w))
            {
                return true;
            } else if flood == temp {
                return false;
            }
        }
    }

    /// The eight-way counterpart to `has_opposite_connection4`. Reaches
    /// diagonal neighbors the same way `flood8` does -- see its doc comment.
    #[inline]
    pub fn has_opposite_connection8(self, start: usize) -> bool {
        let n = self.wall(Direction::North);
        let e = self.wall(Direction::East);
        let s = self.wall(Direction::South);
        let w = self.wall(Direction::West);

        let mut seed = Self::new(self.rows, self.cols);
        seed.set_index(start);
        let mut flood = seed & self;

        if flood.bits_empty() {
            return false;
        }

        loop {
            let temp = flood;
            flood = flood | flood.shift_north() | flood.shift_south();
            flood = flood | flood.shift_east() | flood.shift_west();
            flood &= self;
            if (flood.intersects(n) && flood.intersects(s))
                || (flood.intersects(e) && flood.intersects(w))
            {
                return true;
            } else if flood == temp {
                return false;
            }
        }
    }
}
