//! Incremental legality (`MoveCache`): per-color bitset replacements for the
//! legality half of `State::moves()`, so `generate_actions` reads cached
//! bools instead of recomputing per-anchor legality every ply.

use crate::state::State;
use crate::types::{Orientation, Player, Pos};

/// Per-color legality bits, ignoring hand count (callers filter that
/// separately -- see `moves::generate_actions`): `sarsen[i]` mirrors
/// `State::sarsen_legal_at(i, color)`, `lintel_h[i]`/`lintel_v[i]` mirror
/// `State::lintel_legal_at(i, orientation, color).is_some()`. The `*_any`
/// flags are maintained aggregates so the move-split `generate_actions` can
/// decide whether a piece-kind/orientation sub-action is on offer without
/// scanning the vec. One of these per color, held by `MoveCache` below.
#[derive(Clone, Debug)]
pub(crate) struct MoveCandidates {
    pub(crate) sarsen: Vec<bool>,
    pub(crate) lintel_h: Vec<bool>,
    pub(crate) lintel_v: Vec<bool>,
    sarsen_any: bool,
    lintel_h_any: bool,
    lintel_v_any: bool,
}

impl MoveCandidates {
    fn new(area: usize) -> Self {
        MoveCandidates {
            sarsen: vec![false; area],
            lintel_h: vec![false; area],
            lintel_v: vec![false; area],
            sarsen_any: false,
            lintel_h_any: false,
            lintel_v_any: false,
        }
    }

    fn lintel_mut(&mut self, orientation: Orientation) -> &mut Vec<bool> {
        match orientation {
            Orientation::Horizontal => &mut self.lintel_h,
            Orientation::Vertical => &mut self.lintel_v,
        }
    }

    pub(crate) fn sarsen(&self, i: usize) -> bool {
        self.sarsen[i]
    }

    pub(crate) fn lintel(&self, orientation: Orientation, i: usize) -> bool {
        match orientation {
            Orientation::Horizontal => self.lintel_h[i],
            Orientation::Vertical => self.lintel_v[i],
        }
    }

    pub(crate) fn has_any_sarsen(&self) -> bool {
        self.sarsen_any
    }
    pub(crate) fn has_any_lintel(&self) -> bool {
        self.lintel_h_any || self.lintel_v_any
    }
    pub(crate) fn has_any_lintel_orient(&self, o: Orientation) -> bool {
        match o {
            Orientation::Horizontal => self.lintel_h_any,
            Orientation::Vertical => self.lintel_v_any,
        }
    }
}

/// Incremental replacement for the legality half of `State::moves()`
/// (hand-count filtering stays a read-time check in `generate_actions`,
/// since it's a single hand-wide condition, not a per-cell one) -- same
/// role `Connectivity` plays for `State::connection()`.
///
/// One `MoveCandidates` per color, each a `Vec<bool>` indexed by cell/anchor
/// rather than an enumerable set (`HashSet<PlacedPiece>` or similar): board area is
/// capped at ~100 cells by `Size::is_supported`/`HASHES_LEN`, so a linear
/// scan over the bits in `generate_actions` is already cheap (a handful of
/// bool reads per cell) -- the actual cost this eliminates is the
/// *legality computation* itself (per-anchor height/color comparisons
/// across up to 3 cells), which `moves()` used to redo for every cell on
/// every call. `MoveCache::update` pays that computation only for the
/// bounded recheck set a move can actually affect, regardless of board
/// size, and `generate_actions` reads the resulting bits directly.
///
/// Unlike `Connectivity` (whose union-find root assignment is
/// path-dependent, so two logically-equal states can carry different
/// internal bytes), `MoveCache` is a pure function of `State` -- both
/// colors' bits are fully determined by board contents via
/// `sarsen_legal_at`/`lintel_legal_at`, regardless of move order. It's
/// still excluded from `HashedState`'s `PartialEq`/`Eq` (see the comment
/// there), but for a different reason: not unsoundness, just redundancy --
/// comparing it can never disagree with comparing `State` once `.0` already
/// matches, so it would only add cost, not discriminating power.
#[derive(Clone, Debug)]
pub(crate) struct MoveCache {
    black: MoveCandidates,
    white: MoveCandidates,
}

impl MoveCache {
    pub(crate) fn new(state: &State) -> Self {
        let area = state.size.area() as usize;
        let mut cache = MoveCache {
            black: MoveCandidates::new(area),
            white: MoveCandidates::new(area),
        };
        cache.rebuild(state);
        cache
    }

    pub(crate) fn candidates(&self, color: Player) -> &MoveCandidates {
        match color {
            Player::Black => &self.black,
            Player::White => &self.white,
        }
    }

    fn candidates_mut(&mut self, color: Player) -> &mut MoveCandidates {
        match color {
            Player::Black => &mut self.black,
            Player::White => &mut self.white,
        }
    }

    /// Full from-scratch recompute against `state`'s current board -- used
    /// to build a fresh cache and, in tests, to resync one after `.board`
    /// was poked directly, bypassing `apply`.
    pub(crate) fn rebuild(&mut self, state: &State) {
        let area = state.size.area() as usize;
        for color in [Player::Black, Player::White] {
            let candidates = self.candidates_mut(color);
            for i in 0..area {
                candidates.sarsen[i] = state.sarsen_legal_at(i, color);
                for orientation in [Orientation::Horizontal, Orientation::Vertical] {
                    candidates.lintel_mut(orientation)[i] =
                        state.lintel_legal_at(i, orientation, color).is_some();
                }
            }
            candidates.sarsen_any = candidates.sarsen.iter().any(|&b| b);
            candidates.lintel_h_any = candidates.lintel_h.iter().any(|&b| b);
            candidates.lintel_v_any = candidates.lintel_v.iter().any(|&b| b);
        }
    }

    /// Patch the cache for a move that just touched `cells` on `state`'s
    /// (post-move) board. `lintel_legal_at(anchor, ...)` only ever reads
    /// `anchor`'s own <=3 cells, so a touched cell `j` can only change:
    /// sarsen legality at `j` itself, and lintel legality at the <=3 anchors
    /// per orientation whose triple includes `j` -- anchor `j` (`j` at
    /// triple-index 0), `j - d` (index 1), and `j - 2d` (index 2), for each
    /// orientation's delta `d`. Both colors get rechecked at every touched
    /// cell regardless of which color moved, since a cell's occupant/height
    /// affects both colors' legality (usually oppositely for color, but
    /// identically for the height-only case of a same-color sarsen stack).
    pub(crate) fn update(&mut self, state: &State, cells: &[usize]) {
        let size = state.size;
        for &j in cells {
            for color in [Player::Black, Player::White] {
                self.candidates_mut(color).sarsen[j] = state.sarsen_legal_at(j, color);
            }
            let Pos(jx, jy) = crate::types::Pos::from(j, size);
            for orientation in [Orientation::Horizontal, Orientation::Vertical] {
                let (dx, dy) = orientation.delta();
                for k in 0..3i16 {
                    let ax = jx as i16 - k * dx as i16;
                    let ay = jy as i16 - k * dy as i16;
                    if ax < 0 || ay < 0 || ax as u8 >= size.w || ay as u8 >= size.h {
                        continue;
                    }
                    let anchor = crate::types::Pos(ax as u8, ay as u8).index(size.w);
                    for color in [Player::Black, Player::White] {
                        let legal = state.lintel_legal_at(anchor, orientation, color).is_some();
                        self.candidates_mut(color).lintel_mut(orientation)[anchor] = legal;
                    }
                }
            }
        }
        for color in [Player::Black, Player::White] {
            let c = self.candidates_mut(color);
            c.sarsen_any = c.sarsen.iter().any(|&b| b);
            c.lintel_h_any = c.lintel_h.iter().any(|&b| b);
            c.lintel_v_any = c.lintel_v.iter().any(|&b| b);
        }
    }
}

impl Default for MoveCache {
    fn default() -> Self {
        MoveCache::new(&State::default())
    }
}