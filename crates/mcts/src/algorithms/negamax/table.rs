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

/// How a slot's existing entry is chosen to be kept or overwritten when a
/// new one collides into it. See `NegamaxOptions::replacement`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Replacement {
    /// Every store overwrites the slot outright, regardless of the
    /// existing entry's depth. Cheapest, and favors the most recent
    /// position over the most expensive one to reproduce -- rarely a good
    /// trade once a search is deep enough that entries differ a lot in
    /// how much work they represent.
    Always,
    /// A single slot per index, kept over a shallower same-key update but
    /// always taken over outright by a colliding *different* key (there's
    /// nowhere else in a one-slot table to put it). This crate's original
    /// (and still default) policy.
    #[default]
    DepthPreferred,
    /// Two slots per index: a depth-preferred tier (replaced only by an
    /// entry at least as deep, whether or not the key matches) backed by
    /// an always-replace tier that catches whatever the depth-preferred
    /// tier just refused, so a shallow-but-recent position doesn't get
    /// evicted from the table entirely just because a deep search happens
    /// to hash into the same slot.
    TwoTier,
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

/// A slot's tiers. `Always`/`DepthPreferred` only ever use `tiers[0]`;
/// `TwoTier` uses both. Keeping one slot shape for every policy (rather
/// than a separately-typed table per policy) keeps `Negamax` itself
/// policy-agnostic.
type Slot<G> = [Option<TTEntry<G>>; 2];

pub(crate) struct TranspositionTable<G: Game> {
    slots: Vec<Slot<G>>,
    mask: usize,
    replacement: Replacement,
}

impl<G: Game> TranspositionTable<G> {
    /// `bits` gives a table of `1 << bits` slots. Zero is rejected by the
    /// caller (`NegamaxOptions::with_table_bits` -- table absence is
    /// represented by `Negamax.table: None`, not a zero-size table here).
    pub(crate) fn new(bits: u32, replacement: Replacement) -> Self {
        let size = 1usize << bits;
        let mut slots = Vec::with_capacity(size);
        slots.resize_with(size, || [None, None]);
        Self {
            slots,
            mask: size - 1,
            replacement,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.iter().flatten().filter(|s| s.is_some()).count()
    }

    /// Returns an owned copy of the matching entry, not a borrow: once the
    /// table is shared across threads behind a lock (see `Negamax`'s
    /// parallel root splitting), a caller can't hold a read guard open
    /// across the rest of `negamax_search`'s recursion, so every lookup
    /// pays one clone up front instead.
    pub(crate) fn lookup(&self, hash: u64, state: &G::S) -> Option<TTEntry<G>> {
        let slot = &self.slots[(hash as usize) & self.mask];
        let tiers = if self.replacement == Replacement::TwoTier {
            &slot[..]
        } else {
            &slot[..1]
        };
        tiers
            .iter()
            .flatten()
            .find(|e| e.hash == hash && &e.state == state)
            .cloned()
    }

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
        let entry = TTEntry {
            hash,
            state: state.clone(),
            depth,
            score,
            bound,
            best_action,
        };
        let [depth_tier, always_tier] = &mut self.slots[idx];

        match self.replacement {
            Replacement::Always => *depth_tier = Some(entry),
            Replacement::DepthPreferred => {
                // A different key colliding into the slot always takes it
                // over outright: there's nowhere else to put it in a
                // single-tier table, and the evicted entry just goes back
                // to being an ordinary re-searchable position.
                let replace = match depth_tier {
                    None => true,
                    Some(e) if e.hash != hash => true,
                    Some(e) => depth >= e.depth,
                };
                if replace {
                    *depth_tier = Some(entry);
                }
            }
            Replacement::TwoTier => {
                // The depth tier only yields to an entry at least as deep
                // as what it already holds, whether or not the key
                // matches; anything it refuses falls through to the
                // always-replace tier instead of being dropped.
                let replace_depth_tier = match depth_tier {
                    None => true,
                    Some(e) => depth >= e.depth,
                };
                if replace_depth_tier {
                    *depth_tier = Some(entry);
                } else {
                    *always_tier = Some(entry);
                }
            }
        }
    }
}
