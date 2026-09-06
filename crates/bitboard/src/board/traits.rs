use std::ops::{BitAnd, BitOr, BitXor, Not};

use serde::{Deserialize, Serialize};

use super::Board;
use crate::dim::Dim;
use crate::storage::Storage;

impl<S: Storage, R: Dim, C: Dim> PartialEq for Board<S, R, C> {
    fn eq(&self, other: &Self) -> bool {
        self.rows() == other.rows()
            && self.cols() == other.cols()
            && (0..S::CAPACITY_WORDS).all(|w| self.bits.word(w) == other.bits.word(w))
    }
}

impl<S: Storage, R: Dim, C: Dim> Eq for Board<S, R, C> {}

/// Renders a board as a compact rectangular grid. Rows are printed from the
/// highest row index to the lowest so the first row in the output is the
/// board's north edge; set cells use `X`, unset cells use `.`.
impl<S: Storage, R: Dim, C: Dim> std::fmt::Display for Board<S, R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in (0..self.rows()).rev() {
            for col in 0..self.cols() {
                write!(f, "{}", if self.get(row, col) { 'X' } else { '.' })?;
            }
            if row != 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// Mirrors `GoEngine`'s own `Hash` impl: fold every backing word plus
/// `rows`/`cols` so a `Dyn`-dimensioned board's runtime size participates
/// (two boards with the same bits but different dims must hash differently,
/// matching `PartialEq`, which already compares dims).
impl<S: Storage, R: Dim, C: Dim> std::hash::Hash for Board<S, R, C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for w in 0..S::CAPACITY_WORDS {
            self.bits.word(w).hash(state);
        }
        self.rows().hash(state);
        self.cols().hash(state);
    }
}

impl<S: Storage, R: Dim, C: Dim> BitAnd for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.combine(rhs, |a, b| a & b)
    }
}

impl<S: Storage, R: Dim, C: Dim> BitOr for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.combine(rhs, |a, b| a | b)
    }
}

impl<S: Storage, R: Dim, C: Dim> BitXor for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.combine(rhs, |a, b| a ^ b)
    }
}

impl<S: Storage, R: Dim, C: Dim> Not for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        let mut out = self;
        for w in 0..S::CAPACITY_WORDS {
            let mask = self.word_mask(w);
            *out.bits.word_mut(w) = !self.bits.word(w) & mask;
        }
        out
    }
}

impl<S: Storage, R: Dim, C: Dim> std::ops::BitAndAssign for Board<S, R, C> {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<S: Storage, R: Dim, C: Dim> std::ops::BitOrAssign for Board<S, R, C> {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<S: Storage, R: Dim, C: Dim> std::ops::BitXorAssign for Board<S, R, C> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

/// Consumes set bits lowest-index-first, same idiom as
/// `bitboard::Board`'s `Iterator` impl -- lets a `for src in
/// board` loop (over a `Copy` board, so the loop body's `board` binding is
/// unaffected) walk row-major indices without going through `iter_set`'s
/// borrow.
impl<S: Storage, R: Dim, C: Dim> Iterator for Board<S, R, C> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        for w in 0..S::CAPACITY_WORDS {
            let word = self.bits.word(w);
            if word != 0 {
                let bit = word.trailing_zeros() as usize;
                *self.bits.word_mut(w) = word & (word - 1);
                return Some(w * 64 + bit);
            }
        }
        None
    }
}

// Serde. `S` (`u64` or `[u64; WORDS]`) doesn't itself implement
// `Serialize`/`Deserialize` generically over a const `WORDS`, so words are
// collected into a plain `Vec<u64>` via `Storage::word`/`word_mut` instead.
// `rows`/`cols` ride along as plain `usize`s so a `Dyn`-dimensioned board's
// runtime size survives the round trip; `Const<N>` verifies the
// deserialized length still matches `N` via `Dim::from_len`.

#[derive(Serialize)]
struct BoardDataRef {
    rows: usize,
    cols: usize,
    words: Vec<u64>,
}

#[derive(Deserialize)]
struct BoardDataOwned {
    rows: usize,
    cols: usize,
    words: Vec<u64>,
}

impl<S: Storage, R: Dim, C: Dim> Serialize for Board<S, R, C> {
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        BoardDataRef {
            rows: self.rows(),
            cols: self.cols(),
            words: (0..S::CAPACITY_WORDS).map(|w| self.bits.word(w)).collect(),
        }
        .serialize(serializer)
    }
}

impl<'de, S: Storage, R: Dim, C: Dim> Deserialize<'de> for Board<S, R, C> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let data = BoardDataOwned::deserialize(deserializer)?;
        if data.words.len() != S::CAPACITY_WORDS {
            return Err(D::Error::invalid_length(
                data.words.len(),
                &format!("{} words", S::CAPACITY_WORDS).as_str(),
            ));
        }

        let mut bits = S::zero();
        for (w, word) in data.words.into_iter().enumerate() {
            *bits.word_mut(w) = word;
        }

        let rows = R::from_len(data.rows);
        let cols = C::from_len(data.cols);
        // `Const<N>::from_len` ignores its argument (it has no runtime state
        // to restore), so a mismatched `data.rows`/`data.cols` -- e.g.
        // deserializing a 11x11 board's JSON as `Const<9>` -- must be caught
        // here instead, by comparing what was actually reconstructed against
        // what was on the wire.
        if rows.get() != data.rows || cols.get() != data.cols {
            return Err(D::Error::custom(format!(
                "Board: dims on the wire ({}x{}) don't match the target type's dims ({}x{})",
                data.rows,
                data.cols,
                rows.get(),
                cols.get()
            )));
        }

        Ok(Board { bits, rows, cols })
    }
}
