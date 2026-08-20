//! A fixed-size transposition table keyed by `Game::zobrist_hash`, used by
//! `Negamax` to reuse alpha-beta bounds across both transposing move orders
//! and successive iterative-deepening passes.
//!
//! `Game::zobrist_hash` defaults to `0` for games that haven't implemented
//! it (see `game.rs`), which would make every state collide in a
//! hash-only table -- silently wrong, not just slow, since a stale entry
//! for an unrelated state could get returned as if it were an exact score.
//! Storing the full state alongside the hash and verifying equality on
//! lookup (`Game::S: Eq` is already required by `Game`) closes that hole
//! for free: an un-hashed game just gets a table that's *useless* (every
//! lookup misses once two different states land in the same slot) rather
//! than one that's wrong.

use crate::game::Game;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Bound {
    /// `score` is the exact minimax value.
    Exact,
    /// `score` is a lower bound (search failed high / beta cutoff).
    Lower,
    /// `score` is an upper bound (search failed low, nothing beat alpha).
    Upper,
}

#[derive(Clone)]
pub(crate) struct TTEntry<G: Game> {
    pub(crate) hash: u64,
    pub(crate) state: G::S,
    pub(crate) depth: u32,
    pub(crate) score: i32,
    pub(crate) bound: Bound,
    pub(crate) best_action: Option<G::A>,
}

pub(crate) struct TranspositionTable<G: Game> {
    slots: Vec<Option<TTEntry<G>>>,
    mask: usize,
}

impl<G: Game> TranspositionTable<G> {
    /// `bits` gives a table of `1 << bits` slots. Zero is rejected by the
    /// caller (`NegamaxOptions::with_table_bits` -- table absence is
    /// represented by `Negamax.table: None`, not a zero-size table here).
    pub(crate) fn new(bits: u32) -> Self {
        let size = 1usize << bits;
        let mut slots = Vec::with_capacity(size);
        slots.resize_with(size, || None);
        Self {
            slots,
            mask: size - 1,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub(crate) fn lookup(&self, hash: u64, state: &G::S) -> Option<&TTEntry<G>> {
        let entry = self.slots[(hash as usize) & self.mask].as_ref()?;
        (entry.hash == hash && &entry.state == state).then_some(entry)
    }

    /// Depth-preferred replacement within a slot: a deeper result is more
    /// expensive to reproduce and prunes more on a future hit, so it's kept
    /// over a shallower one for the *same* key. A different key colliding
    /// into the same slot always takes over the slot outright (there's
    /// nowhere else to put it in a single-slot-per-index table, and an
    /// evicted entry just goes back to being an ordinary re-searchable
    /// position).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store(
        &mut self,
        hash: u64,
        state: &G::S,
        depth: u32,
        score: i32,
        bound: Bound,
        best_action: Option<G::A>,
    ) {
        let idx = (hash as usize) & self.mask;
        let replace = match &self.slots[idx] {
            None => true,
            Some(e) if e.hash != hash => true,
            Some(e) => depth >= e.depth,
        };
        if replace {
            self.slots[idx] = Some(TTEntry {
                hash,
                state: state.clone(),
                depth,
                score,
                bound,
                best_action,
            });
        }
    }
}
