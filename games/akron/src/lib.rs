//! Akron (pyramidal connection game on `pyramid::Pyramid`).
//!
//! A player either adds a piece from their pile ([`Action::Add`], always
//! level 0 -- see its doc comment for why), relocates one of their own
//! pieces already on the board ([`Action::Move`]), or -- Black's reply to
//! White's opening move only -- swaps colours instead ([`Action::Swap`],
//! [`State::can_swap`]). `Action::Move` legality and the win condition
//! ([`State::has_span`]) both use [`connectivity::Groups`]' cut-aware
//! connectivity (not raw touching adjacency): a piece may only relocate to a
//! cell touching its own *unbroken* group, and a span only counts as a win
//! if it isn't cut partway by an opponent's overpass. [`Game::winner`] checks
//! the opponent's span before the mover's own (award-on-reveal priority) and
//! also awards a win by forfeit when the player to move has no legal action
//! at all. There is deliberately no repetition-draw rule in this `Game`
//! impl: the published rules make repetition draws player-invoked (by mutual
//! agreement), which has no natural meaning against a self-play search that
//! has no "player" to invoke one -- if it's ever surfaced at all, it belongs
//! at the UI layer, not here.

use std::fmt;

use bitboard::{Adjacency, Dyn};
use mcts::game::{Canonical, Game, PlayerIndex, Real, TerminalStatus, Transform};
use mcts::zobrist::LazyZobristTable;
use pyramid::{get_adjacency, Pyramid, PyramidD4};
use rand::rngs::SmallRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub mod connectivity;
use connectivity::Groups;

/// Smallest supported base width -- same range as `games/margo`, since both
/// sit on the same `pyramid::Pyramid` foundation.
pub const MIN_N: usize = 4;

/// Largest supported base width -- fixes `Cells`' storage width (`[u64;
/// 7]`, since `pyramid::total_cells(10) == 385` needs 7 words).
pub const MAX_N: usize = 10;

/// `State::default()`'s board size -- the published rules' "advanced"
/// 10x10/50-marble option is `MAX_N`; 8x8/32 is the standard size, but this
/// crate follows `games/margo`'s own default choice of 7 for consistency
/// across the pyramidal games.
pub const DEFAULT_N: usize = 7;

type Cells = Pyramid<[u64; 7], Dyn>;

/// A player's starting pile size for a base-`n` board: `n^2 / 2`, "enough
/// to cover the board surface" per the published rules (an 8x8 board gets
/// 32 marbles per player; this crate's other supported sizes scale the same
/// way, rounding down for odd `n`, which the published rules don't cover
/// directly).
pub const fn pile_size(n: usize) -> u32 {
    (n * n / 2) as u32
}

// ── Symmetry / Zobrist hashing ──────────────────────────────────────────
//
// Whole-pyramid D4 (`pyramid::PyramidD4`), array-of-hashes per symmetry
// element -- the same shape `games/margo`/`games/gonnect`/`games/atarigo`
// use, adapted to `PyramidD4`'s own API (a flat `index_symmetries`/
// `invert_symmetry` pair, no separate rows/cols).
//
// Unlike `games/margo` (which tracks `zombie`/`previous` as extra geometric
// fields alongside `occupied`/`black`), Akron's `State` has no such fields:
// `occupied`/`black` alone are the entire board position, and `white_pile`/
// `black_pile` are non-geometric (`pile_size(n) - <that colour's board
// count>` always, an invariant every `Game::apply` arm preserves --
// including `Action::Swap`, which shifts one piece's colour but keeps its
// pile-plus-count sum fixed per colour -- so a symmetric image never needs
// to recompute them, only carry them along unchanged). `Groups` (see
// `connectivity.rs`) is likewise not stored on `State` at all, always
// recomputed on demand, so `canonical_representation` has nothing keyed on
// pre-transform indices to rebuild -- simpler than Margo's `groups` field.

/// A cell can only belong to `occupied`/`black` -- see the module docs above
/// for why there's no `Zombie`/`Previous`-style third or fourth channel the
/// way `games/margo` needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel {
    Occupied = 0,
    Black = 1,
}

const HASH_CHANNELS: usize = 2;

/// Largest board this game serves is `MAX_N`'s `total_cells(MAX_N) == 385`
/// cells, `HASH_CHANNELS` channels per cell, plus one slot each for `turn`
/// and `can_swap`.
const MAX_CELLS: usize = pyramid::total_cells(MAX_N);
pub const ZOBRIST_ENTRIES: usize = MAX_CELLS * HASH_CHANNELS + 2;
pub const ZOBRIST_TURN: usize = MAX_CELLS * HASH_CHANNELS;
pub const ZOBRIST_CAN_SWAP: usize = MAX_CELLS * HASH_CHANNELS + 1;

/// Random Zobrist table, lazily initialised.
pub static HASHES: LazyZobristTable<ZOBRIST_ENTRIES> = LazyZobristTable::new(0x41CE05F1D2B7E39A);

#[inline]
fn zobrist_cell(pos: usize, channel: Channel) -> usize {
    pos * HASH_CHANNELS + channel as usize
}

/// XOR the hash contribution for a single cell into all 8 symmetry hashes.
#[inline]
fn xor_cell(hashes: &mut [u64; 8], pos: usize, channel: Channel, sym: &PyramidD4) {
    for (s, &sym_pos) in sym.index_symmetries(pos).iter().enumerate() {
        hashes[s] ^= HASHES.hash(zobrist_cell(sym_pos, channel));
    }
}

/// XOR the hash contribution for every set bit of `cells` on `channel`.
fn xor_cells(hashes: &mut [u64; 8], cells: &Cells, channel: Channel, sym: &PyramidD4) {
    for pos in cells.iter_set() {
        xor_cell(hashes, pos, channel, sym);
    }
}

/// XOR a position-independent constant (turn, can_swap) into all 8 hashes.
#[inline]
fn xor_const(hashes: &mut [u64; 8], table_idx: usize) {
    let v = HASHES.hash(table_idx);
    for h in hashes.iter_mut() {
        *h ^= v;
    }
}

/// Rebuilds all 8 symmetry hashes from scratch -- see this section's own
/// doc comment for why there's nothing incrementally maintained on `State`
/// to rebuild from here.
fn rebuild_hashes(occupied: &Cells, black: &Cells, turn: Player, can_swap: bool) -> [u64; 8] {
    let sym = PyramidD4::new(occupied.n());
    let mut hashes = [0u64; 8];
    xor_cells(&mut hashes, occupied, Channel::Occupied, &sym);
    xor_cells(&mut hashes, black, Channel::Black, &sym);
    if turn == Player::Black {
        xor_const(&mut hashes, ZOBRIST_TURN);
    }
    if can_swap {
        xor_const(&mut hashes, ZOBRIST_CAN_SWAP);
    }
    hashes
}

/// The symmetric image of `cells` under symmetry element `sym_idx`.
fn transform_cells(cells: &Cells, sym: &PyramidD4, sym_idx: usize) -> Cells {
    let mut out = Cells::new(Dyn(cells.n()));
    for pos in cells.iter_set() {
        out.set_index(sym.index_symmetries(pos)[sym_idx]);
    }
    out
}

/// A comparable key for a board's set-cell pattern, for picking the
/// lexicographically minimal symmetric image -- `Cells` has no `Ord` impl of
/// its own. Copies the raw backing words into a fixed-size array (mirrors
/// `games/margo`'s identically-named helper) rather than collecting
/// `iter_set` into a `Vec<usize>`, since this runs on every candidate
/// symmetry, for every geometric channel, on every node of the MCTS
/// selection path whenever transpositions are enabled.
fn cells_key(cells: &Cells) -> [u64; 7] {
    let mut out = [0u64; 7];
    for (i, w) in cells.words().enumerate() {
        out[i] = w;
    }
    out
}

/// Index of the symmetry whose image of the entire geometric state
/// (`occupied`, `black`) is lexicographically minimal -- the canonical
/// orientation for the position. Unlike `games/margo` (whose `zombie`/
/// `previous` fields can be invariant under a different subgroup than
/// `(occupied, black)` alone, forcing every geometric field into the
/// tie-break -- see that module's own doc comment on the lesson), Akron's
/// `State` has only these two geometric fields, so tying the break to both
/// of them together is already exhaustive: no other field could disagree
/// with whichever symmetry they jointly pick out.
fn canonical_symmetry(occupied: &Cells, black: &Cells) -> usize {
    let sym = PyramidD4::new(occupied.n());
    (0..8)
        .min_by_key(|&sym_idx| {
            (
                cells_key(&transform_cells(occupied, &sym, sym_idx)),
                cells_key(&transform_cells(black, &sym, sym_idx)),
            )
        })
        .unwrap()
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Default, Hash, PartialEq, Eq)]
pub enum Player {
    #[default]
    White,
    Black,
}

impl Player {
    fn next(self) -> Player {
        match self {
            Player::White => Player::Black,
            Player::Black => Player::White,
        }
    }
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

/// A move: place a piece from the mover's pile. `.0` is the flat pyramid
/// index of a level-0 (board-level) cell -- see `pyramid::Pyramid::index`/
/// `to_coord`. Pile placements never land above level 0 (the published
/// rules: "pieces added from the pile must be placed directly on the board
/// and not stacked on existing pieces") -- unlike `Pyramid::can_place`
/// itself, which allows any supported cell regardless of how it's reached,
/// `State::generate_actions` filters candidates to level 0 specifically for
/// this action, since `can_place` has no notion of "placed from pile" vs.
/// "relocated". A moved piece is the only way a piece ever reaches a
/// higher level. `.1` is the board's base width `n`, carried along because
/// `Game::apply_to_action`/`invert_action` need it to build a `PyramidD4`
/// symmetry and their trait signature carries only the action and a
/// `Transform` index, no state -- mirrors `games/margo`'s identically-
/// shaped `Action::Place(u16, u8)`.
#[derive(Copy, Clone, Serialize, Deserialize, Debug, Hash, PartialEq, Eq)]
pub enum Action {
    Add(u16, u8),
    /// Relocate the mover's own piece from `.0` to `.1` (both flat pyramid
    /// indices), per the published rules' movement clause: the destination
    /// must be empty, supported, and touch some other cell already in the
    /// mover's own connected group (using [`connectivity::Groups`]'
    /// cut-aware connectivity, excluding the mover's own vacated cell and
    /// anything the resulting cascade itself relocates this turn -- see
    /// `State::move_destinations`). A piece that supports exactly one other
    /// piece drags that piece down to fill the vacated gap
    /// (`pyramid::Pyramid::relocate`'s cascade), possibly recursively. `.2`
    /// is the board's base width `n`, for the same reason `Add`'s `.1`
    /// carries it.
    Move(u16, u16, u8),
    /// Pie-rule reply to White's opening placement: recolour the single
    /// piece on the board instead of adding one of Black's own, mirroring
    /// `games/margo`'s identically-named `Action::Swap`/`State::can_swap`
    /// shape. Legal only when [`State::can_swap`] holds. A fixed point of
    /// every symmetry element, like `games/margo`'s `Action::Swap`.
    Swap,
}

/// Board state: `occupied` is every placed piece regardless of colour;
/// `black` marks which of those cells belong to Black -- White's pieces are
/// `occupied & !black`, derived rather than stored separately so the two
/// boards can't drift out of sync (same split `games/margo::State` uses).
/// `white_pile`/`black_pile` count each player's remaining unplaced pieces
/// (see [`pile_size`]); a player with an empty pile can no longer add, but
/// may still have a legal `Action::Move`.
/// `can_swap` tracks whether the pie-rule swap window is still open --
/// true from the empty board until the piece count first leaves 1, the
/// same `games/margo`-style pattern: closed by any placement that isn't
/// the very first, and explicitly by `Action::Swap` itself, so it can
/// never reopen even if a later move happens to drop the piece count back
/// to 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    occupied: Cells,
    black: Cells,
    white_pile: u32,
    black_pile: u32,
    turn: Player,
    can_swap: bool,
}

impl Default for State {
    fn default() -> Self {
        Self::new(DEFAULT_N)
    }
}

impl State {
    /// A fresh empty base-`n` board, both piles full. `n` must be within
    /// `MIN_N..=MAX_N`.
    pub fn new(n: usize) -> Self {
        assert!(
            (MIN_N..=MAX_N).contains(&n),
            "Akron board size must be between {MIN_N} and {MAX_N}, got {n}"
        );
        Self {
            occupied: Cells::new(Dyn(n)),
            black: Cells::new(Dyn(n)),
            white_pile: pile_size(n),
            black_pile: pile_size(n),
            turn: Player::default(),
            can_swap: true,
        }
    }

    #[inline]
    pub fn is_occupied(&self, index: usize) -> bool {
        self.occupied.get_index(index)
    }

    #[inline]
    pub fn is_black(&self, index: usize) -> bool {
        self.black.get_index(index)
    }

    #[inline]
    pub fn is_white(&self, index: usize) -> bool {
        self.is_occupied(index) && !self.is_black(index)
    }

    #[inline]
    pub fn turn(&self) -> Player {
        self.turn
    }

    /// This board's base width -- see `MIN_N`/`MAX_N`.
    #[inline]
    pub fn n(&self) -> usize {
        self.occupied.n()
    }

    /// Total addressable cells for this board's size (see
    /// `pyramid::total_cells`).
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.occupied.total_cells()
    }

    /// Remaining unplaced pieces for `player` -- see [`pile_size`].
    #[inline]
    pub fn pile(&self, player: Player) -> u32 {
        match player {
            Player::White => self.white_pile,
            Player::Black => self.black_pile,
        }
    }

    /// Whether `Action::Swap` is currently legal: the swap window (see the
    /// `can_swap` field's doc comment) is still open and exactly one piece
    /// -- White's opening placement -- is on the board.
    #[inline]
    pub fn can_swap(&self) -> bool {
        self.can_swap && self.occupied.count_ones() == 1
    }

    /// The raw swap-window flag, ungated by the current piece count -- unlike
    /// [`State::can_swap`], which collapses to `false` on an empty board even
    /// though the window is still open, this is what a wire adapter must
    /// round-trip via [`State::from_parts`]: serializing the gated value
    /// instead would permanently close the window the moment it's read back
    /// on a still-empty board.
    #[inline]
    pub fn swap_window_open(&self) -> bool {
        self.can_swap
    }

    /// Every occupied cell's flat index, for a wire adapter to serialize.
    pub fn occupied_indices(&self) -> Vec<usize> {
        self.occupied.iter_set().collect()
    }

    /// Every Black-occupied cell's flat index, for a wire adapter to
    /// serialize.
    pub fn black_indices(&self) -> Vec<usize> {
        self.black.iter_set().collect()
    }

    /// Reconstructs a `State` from flat-index lists -- the inverse of
    /// `occupied_indices`/`black_indices`, for a wire adapter to deserialize
    /// a JSON request back into a real `State` without going through legal
    /// play. No legality checking is done here: the caller (a `GameAdapter`
    /// round-tripping its own previously emitted wire format) is trusted to
    /// pass back a state this crate itself produced.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        n: usize,
        occupied: &[usize],
        black: &[usize],
        white_pile: u32,
        black_pile: u32,
        turn: Player,
        can_swap: bool,
    ) -> Self {
        let fill = |indices: &[usize]| {
            let mut cells = Cells::new(Dyn(n));
            for &index in indices {
                cells.set_index(index);
            }
            cells
        };
        Self {
            occupied: fill(occupied),
            black: fill(black),
            white_pile,
            black_pile,
            turn,
            can_swap,
        }
    }

    /// The chain of cells a relocation starting at `from` would vacate,
    /// read-only (mirrors, without mutating, the drop chain
    /// `pyramid::Pyramid::relocate`'s cascade performs internally):
    /// `chain[0]` is `from` itself, and each following entry is the single
    /// dependent that drops into the previous entry's gap, one level up.
    /// `None` if `from` (or some link further up the chain) is pinned --
    /// two or more dependents, so there's no single unambiguous drop.
    /// Every entry except the last ends up reoccupied once the whole
    /// cascade settles; only `chain.last()` is actually empty afterwards.
    fn vacated_chain(&self, from: usize) -> Option<Vec<usize>> {
        let mut chain = vec![from];
        let (mut col, mut row, mut level) = self.occupied.to_coord(from);
        loop {
            match self.occupied.dependents(col, row, level).as_slice() {
                [] => return Some(chain),
                &[(dc, dr)] => {
                    col = dc;
                    row = dr;
                    level += 1;
                    chain.push(self.occupied.index(col, row, level));
                }
                _ => return None,
            }
        }
    }

    /// Whether the piece at `from` is a *freedom* of its own group: removing
    /// it doesn't break the rest of that group's connectivity into more than
    /// one piece (a singleton, or already-disconnected, piece is trivially a
    /// freedom). Purely a derived property of the current board -- see the
    /// published rules' "Freedoms" clause -- computed by actually trying the
    /// removal against [`connectivity::Groups`]' cut-aware connectivity.
    ///
    /// `before` is the caller's already-computed connectivity for the
    /// current (unmodified) board -- shared across every candidate `from` a
    /// caller like `generate_actions` checks in the same pass, since it's
    /// identical for all of them and gives `O(1)` access (via
    /// [`connectivity::Groups::group_members`]) to exactly the cells that
    /// need to stay connected, without a per-candidate board scan.
    ///
    /// The actual removal check ([`connectivity::survives_removal`]) is a
    /// flood fill seeded from those members, not a whole-board rebuild --
    /// see its own doc comment for why that's both cheaper and still exactly
    /// correct (removing `from` can newly cut a shielded connection just as
    /// easily as it can newly restore a cut one, so this can't be shortcut
    /// by only inspecting `before`'s existing group structure).
    fn is_freedom(&self, from: usize, before: &mut Groups) -> bool {
        let members: Vec<usize> = before
            .group_members(from)
            .iter()
            .copied()
            .filter(|&i| i != from)
            .collect();
        if members.len() <= 1 {
            return true;
        }
        let mut after_occupied = self.occupied;
        after_occupied.clear_index(from);
        connectivity::survives_removal(self.n(), &after_occupied, &self.black, &members)
    }

    /// Legal destinations for relocating the piece at `from`, per the
    /// published rules' movement clause: an empty cell, not part of this
    /// turn's own cascade (`chain`), that touches some *other*, not-also-
    /// moving-this-turn cell already in `from`'s own cut-aware connected
    /// group -- then confirmed physically placeable (support, and a
    /// successful cascade with no pinning further up) via
    /// `pyramid::Pyramid::relocate` itself against a scratch copy.
    ///
    /// `groups` is the caller's connectivity for the current board, shared
    /// the same way [`State::is_freedom`]'s `before` is -- see that method's
    /// doc comment. Candidate anchors come from
    /// [`connectivity::Groups::group_members`] rather than a full board
    /// scan, since that's already exactly "every occupied, same-coloured
    /// cell in `from`'s group".
    ///
    /// The board `from`'s own cascade settles into is independent of which
    /// destination is being tried -- `chain` (already computed by the
    /// caller) fully determines it -- so it's built once up front rather
    /// than re-derived (via a fresh `Pyramid::relocate`, cascade recursion
    /// and all) for every `(anchor, destination)` pair this checks. Each
    /// candidate destination then only needs the cheap physical check
    /// (`Pyramid::can_place`) against that already-settled board, not a
    /// full trial relocation.
    fn move_destinations(&self, from: usize, chain: &[usize], groups: &mut Groups) -> Vec<usize> {
        let n = self.n();
        let adjacency = get_adjacency(n);

        let mut settled = self.occupied;
        settled.clear_index(chain[0]);
        for i in 1..chain.len() {
            settled.clear_index(chain[i]);
            settled.set_index(chain[i - 1]);
        }

        let mut candidates: Vec<usize> = Vec::new();
        let anchors: Vec<usize> = groups.group_members(from).to_vec();
        for anchor in anchors {
            if anchor == from || chain.contains(&anchor) {
                continue;
            }
            for to in adjacency.neighbors(anchor) {
                if chain.contains(&to) || candidates.contains(&to) {
                    continue;
                }
                let (tc, tr, tl) = settled.to_coord(to);
                if settled.can_place(tc, tr, tl) {
                    candidates.push(to);
                }
            }
        }
        candidates
    }

    /// Whether `player` has completed a *span*: an unbroken, cut-aware
    /// (see [`connectivity::Groups`]) chain of their own pieces connecting
    /// their two assigned board sides. Black spans rows (row 0 to row
    /// `n-1`); White spans columns (col 0 to col `n-1`) -- a fixed
    /// assignment, arbitrary but consistent, since the published rules say
    /// only that each player owns "their edges of the board" without
    /// specifying which axis is whose. Only level-0 cells lie on the
    /// board's physical perimeter -- every higher level is strictly
    /// interior, sitting over a smaller inset square (see
    /// `pyramid::level_side`) -- so only level-0 cells are checked as
    /// endpoints, though the connecting chain itself may pass through any
    /// level.
    fn has_span(&self, player: Player) -> bool {
        let n = self.n();
        let black = player == Player::Black;
        let mut groups = Groups::compute(n, &self.occupied, &self.black);
        let (side_a, side_b): (Vec<usize>, Vec<usize>) = if black {
            (
                (0..n).map(|col| self.occupied.index(col, 0, 0)).collect(),
                (0..n)
                    .map(|col| self.occupied.index(col, n - 1, 0))
                    .collect(),
            )
        } else {
            (
                (0..n).map(|row| self.occupied.index(0, row, 0)).collect(),
                (0..n)
                    .map(|row| self.occupied.index(n - 1, row, 0))
                    .collect(),
            )
        };
        side_a.iter().any(|&a| {
            groups.color_of(a) == Some(black) && side_b.iter().any(|&b| groups.same_group(a, b))
        })
    }

    /// Whether the player to move has at least one legal action -- `Swap`,
    /// `Add`, or `Move` -- without materializing the full list the way
    /// [`Game::generate_actions`] does. Early-exits at the first legal
    /// action found: [`Game::winner`]'s no-legal-move forfeit check is the
    /// only caller, and it only needs a yes/no answer, not the list itself.
    /// An `Add` is checked first and is cheap to confirm (any empty
    /// level-0 cell while the pile is non-empty) -- the overwhelmingly
    /// common case, since most reachable states have plenty of empty board
    /// left. The `Move` scan (one [`connectivity::Groups::compute`] plus a
    /// per-piece freedom/destination check, mirroring `generate_actions`'
    /// own move loop) only runs once that's ruled out, i.e. once a pile is
    /// empty and the board is full -- the same "pay for it only when it can
    /// actually apply" shape as `games/druid`'s analogous
    /// `Game::terminal_status` fallback check.
    fn has_legal_action(&self) -> bool {
        if self.can_swap() {
            return true;
        }
        let n = self.n();
        if self.pile(self.turn) > 0 && (0..(n * n)).any(|index| !self.is_occupied(index)) {
            return true;
        }
        let color = self.turn == Player::Black;
        let mut groups = Groups::compute(n, &self.occupied, &self.black);
        for from in 0..self.total_cells() {
            if !self.is_occupied(from) || self.is_black(from) != color {
                continue;
            }
            let Some(chain) = self.vacated_chain(from) else {
                continue;
            };
            if !self.is_freedom(from, &mut groups) {
                continue;
            }
            if !self.move_destinations(from, &chain, &mut groups).is_empty() {
                return true;
            }
        }
        false
    }

    /// This state's Zobrist hash, symmetry-invariant: two states that are
    /// symmetric images of each other under `PyramidD4` hash identically,
    /// since both pick out the same slot of `rebuild_hashes`'s per-symmetry
    /// array via [`canonical_symmetry`] (see `games/margo`/`games/gonnect`'s
    /// identical `State::zobrist_hash`/`State::hash` for why this is the
    /// trick that makes the array-of-hashes design work).
    fn zobrist_hash(&self) -> u64 {
        let hashes = rebuild_hashes(&self.occupied, &self.black, self.turn, self.can_swap);
        hashes[canonical_symmetry(&self.occupied, &self.black)]
    }
}

#[derive(Clone)]
pub struct Akron;

impl Game for Akron {
    type S = State;
    type A = Action;
    type P = Player;

    fn apply(mut state: State, action: &Action) -> State {
        match *action {
            Action::Add(index, _n) => {
                let index = index as usize;
                debug_assert!(
                    state.occupied.to_coord(index).2 == 0,
                    "Action::Add must target a level-0 cell"
                );
                debug_assert!(
                    !state.is_occupied(index),
                    "action generated by generate_actions must be legal"
                );
                state.occupied.set_index(index);
                match state.turn {
                    Player::White => {
                        state.white_pile -= 1;
                    }
                    Player::Black => {
                        state.black.set_index(index);
                        state.black_pile -= 1;
                    }
                }
            }
            Action::Move(from, to, _n) => {
                let (from, to) = (from as usize, to as usize);
                let color_from = state.is_black(from);
                debug_assert!(
                    state.is_occupied(from) && color_from == (state.turn == Player::Black),
                    "action generated by generate_actions must be legal"
                );
                // `chain` (see `State::vacated_chain`) is the pillar of
                // single-dependents above `from` that the cascade drops one
                // level each -- its colours have to shift in lockstep with
                // `relocate`'s occupancy-only cascade, since `black` is a
                // separate bitboard `relocate` knows nothing about. Capture
                // each link's colour before mutating anything.
                let chain = state
                    .vacated_chain(from)
                    .expect("action generated by generate_actions must be legal");
                let chain_colors: Vec<bool> = chain.iter().map(|&i| state.is_black(i)).collect();
                let ok = state
                    .occupied
                    .relocate(state.occupied.to_coord(from), state.occupied.to_coord(to));
                debug_assert!(ok, "action generated by generate_actions must be legal");
                for i in 1..chain.len() {
                    if chain_colors[i] {
                        state.black.set_index(chain[i - 1]);
                    } else {
                        state.black.clear_index(chain[i - 1]);
                    }
                }
                // The chain's last link is the one cell that ends up
                // genuinely empty once the whole cascade settles.
                state.black.clear_index(*chain.last().unwrap());
                if color_from {
                    state.black.set_index(to);
                } else {
                    state.black.clear_index(to);
                }
            }
            Action::Swap => {
                debug_assert!(
                    state.can_swap(),
                    "action generated by generate_actions must be legal"
                );
                // Exactly one piece is on the board when this is legal --
                // recolour it and hand it to the pile it came from, mirroring
                // `games/margo`'s general "flip every occupied cell's
                // colour" swap even though only one cell is ever occupied
                // here.
                let index = state
                    .occupied
                    .iter_set()
                    .next()
                    .expect("can_swap implies exactly one occupied cell");
                state.black.set_index(index);
                state.white_pile += 1;
                state.black_pile -= 1;
                state.can_swap = false;
            }
        }
        // The swap window closes for good the moment the piece count leaves
        // 1 -- see `State`'s `can_swap` field doc comment.
        if state.occupied.count_ones() != 1 {
            state.can_swap = false;
        }
        state.turn = state.turn.next();
        state
    }

    fn generate_actions(state: &State, actions: &mut Vec<Action>) {
        let n = state.occupied.n();
        if state.can_swap() {
            actions.push(Action::Swap);
        }

        if state.pile(state.turn) > 0 {
            for index in 0..(n * n) {
                if !state.is_occupied(index) {
                    actions.push(Action::Add(index as u16, n as u8));
                }
            }
        }

        // Shared across every candidate `from` below -- see `is_freedom`/
        // `move_destinations`' doc comments for why a single
        // `Groups::compute` here (instead of one per candidate) is the
        // difference that matters as board size grows.
        let color = state.turn == Player::Black;
        let mut groups = Groups::compute(n, &state.occupied, &state.black);
        for from in 0..state.total_cells() {
            if !state.is_occupied(from) || state.is_black(from) != color {
                continue;
            }
            let Some(chain) = state.vacated_chain(from) else {
                continue;
            };
            if !state.is_freedom(from, &mut groups) {
                continue;
            }
            for to in state.move_destinations(from, &chain, &mut groups) {
                actions.push(Action::Move(from as u16, to as u16, n as u8));
            }
        }
    }

    /// Rejection-sampling fast path for `SimulateStrategy::playout`'s
    /// uniform rollouts -- same idea as `games/margo`/`games/gonnect`/
    /// `games/atarigo`'s `random_action` overrides: an `Add` candidate is
    /// `O(1)` to confirm (any empty level-0 cell while the pile is
    /// non-empty, no connectivity involved), unlike a `Move`, whose
    /// legality is inseparable from a full [`connectivity::Groups::compute`]
    /// -- unlike `games/margo`'s own per-candidate legality check, there's
    /// no way to cheaply spot-check "is *this one* relocation legal" here,
    /// since [`State::is_freedom`]/[`State::move_destinations`] both need
    /// the same whole-board connectivity regardless of which single
    /// candidate is ultimately asked about.
    ///
    /// Consequently, this deliberately departs from true uniform sampling
    /// over the full action set: whenever the mover's pile is non-empty, it
    /// draws only from `Add`, never from `Move`, even on the (typically
    /// rarer) states where some freedom piece also has a legal
    /// destination -- rather than paying for a full [`Game::generate_actions`]
    /// on every rollout ply just to weight the two types correctly. This
    /// changes rollout *character* (playouts favour building outward with
    /// fresh pieces over relocating existing ones while pile remains), not
    /// legality -- every action this returns is genuinely legal, and once
    /// the pile is actually empty (or already exhausted several random
    /// probes against a nearly-full board), this still falls back to the
    /// exact `generate_actions`-backed uniform draw, so `Move`/`Swap` are
    /// never permanently unreachable, only de-weighted while `Add` remains
    /// cheap and available.
    fn random_action(state: &State, rng: &mut SmallRng) -> Option<Action> {
        if state.can_swap() {
            let mut actions = Vec::new();
            Self::generate_actions(state, &mut actions);
            return Some(actions[rng.gen_range(0..actions.len())]);
        }

        let n = state.n();
        if state.pile(state.turn) > 0 {
            let max_attempts = 64;
            for _ in 0..max_attempts {
                let index = rng.gen_range(0..(n * n));
                if !state.is_occupied(index) {
                    return Some(Action::Add(index as u16, n as u8));
                }
            }
        }

        let mut actions = Vec::new();
        Self::generate_actions(state, &mut actions);
        if actions.is_empty() {
            None
        } else {
            Some(actions[rng.gen_range(0..actions.len())])
        }
    }

    /// Terminal exactly when [`Game::winner`] finds one: either a completed
    /// span (checked first) or the no-legal-move loss (see `winner`'s own
    /// doc comment) -- there is no other way for the game to end, since a
    /// pile running out still leaves `Action::Move` available as long as any
    /// of the player's pieces are a freedom with a legal destination.
    ///
    /// Delegates to [`Game::terminal_status`] (see its own doc comment for
    /// why) rather than repeating the span/forfeit logic here -- calling
    /// `is_terminal` alone still pays for one full check, same as before.
    fn is_terminal(state: &State) -> bool {
        !matches!(Self::terminal_status(state), TerminalStatus::NotTerminal)
    }

    fn player_to_move(state: &State) -> Player {
        state.turn
    }

    /// A completed span (see [`State::has_span`]) wins; failing that, a
    /// player with no legal move on their turn loses immediately (the
    /// published rules' explicit forfeit clause -- unlike `games/margo`,
    /// where the analogous "no legal placement" case is instead scored by
    /// piece count, Akron's rules state the forfeit directly).
    ///
    /// Span priority is "opponent's exposed win first": `state.turn` is
    /// whoever is about to move *next*, i.e. the opponent of whoever just
    /// played the move that produced this state (`state.turn.next()`, since
    /// there are only two players and turn always flips after `apply`).
    /// Checking `state.turn`'s span before the mover's own matches the
    /// published rules' award-on-reveal clause: a move that uncovers the
    /// opponent's pre-existing win is scored as the opponent's win even if
    /// the same move also happens to complete the mover's own span.
    fn winner(state: &State) -> Option<Player> {
        let opponent = state.turn;
        let mover = state.turn.next();
        if state.has_span(opponent) {
            return Some(opponent);
        }
        if state.has_span(mover) {
            return Some(mover);
        }
        if !state.has_legal_action() {
            return Some(mover);
        }
        None
    }

    /// Single source of truth for both `is_terminal` and `winner`, mirroring
    /// `games/druid`'s identically-motivated override: without this, the
    /// default `Game::terminal_status` answers `is_terminal` and `winner`
    /// as two independent calls, each of which redoes the same
    /// `has_span`/`has_legal_action` work above from scratch -- doubling the
    /// cost of every terminal check along an MCTS rollout, which calls
    /// `terminal_status` once per ply (see `mcts::strategies::mcts::simulate`).
    /// `is_terminal`/`winner` above still each do their own read when called
    /// standalone, same as before this override existed.
    fn terminal_status(state: &State) -> TerminalStatus<Player> {
        let opponent = state.turn;
        let mover = state.turn.next();
        if state.has_span(opponent) {
            return TerminalStatus::Winner(opponent);
        }
        if state.has_span(mover) {
            return TerminalStatus::Winner(mover);
        }
        if !state.has_legal_action() {
            return TerminalStatus::Winner(mover);
        }
        TerminalStatus::NotTerminal
    }

    fn notation(state: &Self::S, action: &Self::A) -> String {
        match *action {
            Action::Add(index, _n) => {
                let (col, row, _level) = state.occupied.to_coord(index as usize);
                format!("({col},{row})")
            }
            Action::Move(from, to, _n) => {
                let (fc, fr, fl) = state.occupied.to_coord(from as usize);
                let (tc, tr, tl) = state.occupied.to_coord(to as usize);
                format!("({fc},{fr},{fl})->({tc},{tr},{tl})")
            }
            Action::Swap => "swap".to_string(),
        }
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.zobrist_hash()
    }

    fn canonical_representation(state: Real<Self::S>) -> (Canonical<Self::S>, Transform) {
        let state = state.0;
        let sym_idx = canonical_symmetry(&state.occupied, &state.black);
        let sym = PyramidD4::new(state.occupied.n());
        let occupied = transform_cells(&state.occupied, &sym, sym_idx);
        let black = transform_cells(&state.black, &sym, sym_idx);
        (
            Canonical(State {
                occupied,
                black,
                white_pile: state.white_pile,
                black_pile: state.black_pile,
                turn: state.turn,
                can_swap: state.can_swap,
            }),
            Transform::new(sym_idx),
        )
    }

    /// `Action::Swap` is a fixed point of every symmetry (see the variant's
    /// own doc comment); `Add`/`Move` transform their cell indices through
    /// the `PyramidD4` group for the board size they each carry.
    fn apply_to_action(action: Real<Self::A>, sym: Transform) -> Canonical<Self::A> {
        match action.0 {
            Action::Swap => Canonical(Action::Swap),
            Action::Add(index, n) => {
                let pyramid_sym = PyramidD4::new(n as usize);
                let image = pyramid_sym.index_symmetries(index as usize)[sym.index()];
                Canonical(Action::Add(image as u16, n))
            }
            Action::Move(from, to, n) => {
                let pyramid_sym = PyramidD4::new(n as usize);
                let from_image = pyramid_sym.index_symmetries(from as usize)[sym.index()];
                let to_image = pyramid_sym.index_symmetries(to as usize)[sym.index()];
                Canonical(Action::Move(from_image as u16, to_image as u16, n))
            }
        }
    }

    fn invert_action(action: Canonical<Self::A>, sym: Transform) -> Real<Self::A> {
        match action.0 {
            Action::Swap => Real(Action::Swap),
            Action::Add(index, n) => {
                let pyramid_sym = PyramidD4::new(n as usize);
                let original = pyramid_sym.invert_symmetry(index as usize, sym.index());
                Real(Action::Add(original as u16, n))
            }
            Action::Move(from, to, n) => {
                let pyramid_sym = PyramidD4::new(n as usize);
                let from_original = pyramid_sym.invert_symmetry(from as usize, sym.index());
                let to_original = pyramid_sym.invert_symmetry(to as usize, sym.index());
                Real(Action::Move(from_original as u16, to_original as u16, n))
            }
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for level in 0..self.occupied.n() {
            let side = self.occupied.level_side(level);
            writeln!(f, "L{level}:")?;
            for row in (0..side).rev() {
                for col in 0..side {
                    let index = self.occupied.index(col, row, level);
                    let ch = if !self.is_occupied(index) {
                        '.'
                    } else if self.is_black(index) {
                        'X'
                    } else {
                        'O'
                    };
                    write!(f, "{ch} ")?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::util::random_play;
    use pyramid::crossing::get_crossing_table;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[test]
    fn random_play_smoke_test() {
        random_play::<Akron>();
    }

    #[test]
    fn add_actions_only_target_level_zero_cells() {
        let state = State::new(DEFAULT_N);
        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        let n = state.n();
        assert_eq!(actions.len(), n * n);
        for action in actions {
            let Action::Add(index, _n) = action else {
                panic!("empty board must only offer Add actions");
            };
            let (_, _, level) = state.occupied.to_coord(index as usize);
            assert_eq!(level, 0, "Add must only target level-0 cells");
        }
    }

    #[test]
    fn occupied_cell_is_not_offered_again() {
        let mut state = State::new(DEFAULT_N);
        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        let first = actions[0];
        state = Akron::apply(state, &first);

        actions.clear();
        Akron::generate_actions(&state, &mut actions);
        let Action::Add(placed, placed_n) = first else {
            panic!("first move on an empty board must be an Add");
        };
        assert!(
            !actions.contains(&Action::Add(placed, placed_n)),
            "a just-occupied cell must not be offered again"
        );
        // One fewer Add (the just-occupied cell), plus the pie-rule Swap
        // now on offer for Black's reply to White's opening placement (see
        // `State::can_swap`) -- net unchanged from `n * n`.
        assert!(state.can_swap());
        assert!(actions.contains(&Action::Swap));
        assert_eq!(actions.len(), state.n() * state.n());
    }

    /// Once both piles are empty, `Action::Add` disappears from
    /// `generate_actions` but the game is not terminal while a freedom piece
    /// still has a legal `Action::Move` destination -- unlike `games/margo`,
    /// where an emptied board is scored immediately by piece count, Akron's
    /// no-legal-move forfeit (see `Game::winner`) only fires once *no*
    /// action at all -- Add or Move -- remains.
    #[test]
    fn empty_pile_does_not_end_the_game_while_a_move_remains() {
        let mut state = state_with(4, Player::White);
        state.white_pile = 0;
        state.black_pile = 0;
        // Two touching White pieces: each is a freedom (removing either
        // leaves the other, still trivially one group), and each has the
        // other as an anchor to touch a fresh destination against -- a lone
        // piece, with no other cell in its group to touch, would have none.
        place(&mut state, 0, 0, 0, false);
        place(&mut state, 1, 0, 0, false);

        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        assert!(
            actions.iter().all(|a| matches!(a, Action::Move(_, _, _))),
            "an emptied pile must offer no Add actions"
        );
        assert!(!actions.is_empty(), "the lone piece has legal relocations");
        assert!(!Akron::is_terminal(&state));
    }

    /// A player with an empty pile and no piece of their own on the board at
    /// all has no legal action whatsoever -- the published rules' explicit
    /// forfeit clause: the opponent wins immediately.
    #[test]
    fn no_legal_move_forfeits_the_game_to_the_opponent() {
        let mut state = state_with(4, Player::Black);
        state.black_pile = 0;
        place(&mut state, 0, 0, 0, false);

        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        assert!(
            actions.is_empty(),
            "Black has no pile and no piece of its own to move"
        );
        assert!(Akron::is_terminal(&state));
        assert_eq!(Akron::winner(&state), Some(Player::White));
    }

    /// `Action::Swap` recolours White's opening piece to Black, refunds it
    /// to White's pile (it was never really Black's placement), debits
    /// Black's pile by one to account for the piece Black now owns on the
    /// board, closes the swap window for good, and still passes the turn
    /// like any other action.
    #[test]
    fn swap_recolours_the_opening_piece_and_rebalances_the_piles() {
        let state = State::new(4);
        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        let Action::Add(opening, opening_n) = actions[0] else {
            panic!("the empty board's first offered action must be an Add");
        };
        let state = Akron::apply(state, &Action::Add(opening, opening_n));
        assert!(state.can_swap());

        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        assert!(actions.contains(&Action::Swap));

        let state = Akron::apply(state, &Action::Swap);
        assert!(
            state.is_black(opening as usize),
            "the opening piece now belongs to Black"
        );
        assert_eq!(state.pile(Player::White), pile_size(4));
        assert_eq!(state.pile(Player::Black), pile_size(4) - 1);
        assert!(!state.can_swap(), "the swap window closes for good");
        assert_eq!(state.turn(), Player::White);
    }

    /// The published rules' award-on-reveal clause: if the same board has a
    /// completed span for *both* players at once, `winner` must resolve it
    /// in favour of whoever is about to move next (`state.turn`) -- the
    /// opponent of whoever just played the move that produced this
    /// position -- not the mover's own newly-completed span. Both spans
    /// coexist here without either being cut: Black's is a straight column
    /// wall (col 4, every row), and White's is a chain that only *touches*
    /// (never crosses) that wall via colour-blind support -- a two-cell
    /// level-1 bridge resting on top of Black's wall, connecting a White
    /// chain on col 4's left side to one on its right side. Since the
    /// bridge only ever touches Black's wall through support edges, never
    /// through a same-colour edge crossing one of Black's own, no over/under
    /// cut ever applies to either side -- confirmed directly below via
    /// `has_span`, rather than assumed.
    #[test]
    fn simultaneous_spans_resolve_to_whoever_is_about_to_move() {
        let n = 10;
        let mut state = state_with(n, Player::White);
        for row in 0..n {
            place(&mut state, 4, row, 0, true);
        }
        for col in 0..=3 {
            place(&mut state, col, 3, 0, false);
        }
        for col in 5..n {
            place(&mut state, col, 4, 0, false);
        }
        place(&mut state, 3, 4, 0, false);
        place(&mut state, 3, 5, 0, false);
        place(&mut state, 5, 5, 0, false);
        place(&mut state, 3, 3, 1, false);
        place(&mut state, 3, 4, 1, false);
        place(&mut state, 4, 4, 1, false);

        assert!(
            state.has_span(Player::Black),
            "Black's wall is a genuine, uncut row-to-row chain"
        );
        assert!(
            state.has_span(Player::White),
            "White's bridge genuinely connects col0 to col9 without being cut"
        );

        // `state.turn` is whoever is about to move next, i.e. the opponent
        // of whoever just moved -- `winner` must credit that opponent's
        // span first regardless of which colour it is.
        let mut black_to_move = state.clone();
        black_to_move.turn = Player::Black;
        assert_eq!(Akron::winner(&black_to_move), Some(Player::Black));

        let mut white_to_move = state.clone();
        white_to_move.turn = Player::White;
        assert_eq!(Akron::winner(&white_to_move), Some(Player::White));
    }

    #[test]
    fn rejects_board_size_outside_supported_range() {
        assert!(std::panic::catch_unwind(|| State::new(MIN_N - 1)).is_err());
        assert!(std::panic::catch_unwind(|| State::new(MAX_N + 1)).is_err());
    }

    /// Builds a bare `State` for hand-placed movement fixtures: an empty
    /// `n`-board with arbitrary piles (irrelevant to `Action::Move`) and
    /// `turn` set so the caller's mover is legal to act.
    fn state_with(n: usize, turn: Player) -> State {
        State {
            occupied: Pyramid::new(Dyn(n)),
            black: Pyramid::new(Dyn(n)),
            white_pile: pile_size(n),
            black_pile: pile_size(n),
            turn,
            // These fixtures place pieces directly, bypassing `apply`, so
            // leave the swap window closed -- open, it would spuriously
            // offer `Action::Swap` whenever a fixture happens to land on a
            // one-piece board.
            can_swap: false,
        }
    }

    fn place(state: &mut State, col: usize, row: usize, level: usize, black: bool) -> usize {
        state.occupied.set(col, row, level);
        if black {
            state.black.set(col, row, level);
        }
        state.occupied.index(col, row, level)
    }

    /// A relocation that leaves the mover's own group connected is legal
    /// and, once applied, actually moves the piece.
    #[test]
    fn legal_relocation_that_preserves_group_connectivity() {
        let mut state = state_with(4, Player::White);
        // A row of three touching White pieces: (2,0,0) is an endpoint, so
        // removing it leaves (0,0,0)-(1,0,0) still connected.
        place(&mut state, 0, 0, 0, false);
        place(&mut state, 1, 0, 0, false);
        let from = place(&mut state, 2, 0, 0, false);
        let to = state.occupied.index(1, 1, 0);

        let n = state.n() as u8;
        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        assert!(
            actions.contains(&Action::Move(from as u16, to as u16, n)),
            "an endpoint piece must be able to relocate to a cell touching the rest of its group"
        );

        let state = Akron::apply(state, &Action::Move(from as u16, to as u16, n));
        assert!(!state.is_occupied(from), "the source cell is now empty");
        assert!(state.is_occupied(to) && !state.is_black(to));
        assert_eq!(state.turn(), Player::Black);
    }

    /// A relocation that would disconnect the mover's own group -- moving
    /// the middle piece out of a three-in-a-row chain, with no other path
    /// between the two endpoints -- is never offered, per the published
    /// rules' "Freedoms" clause.
    #[test]
    fn rejected_relocation_that_would_break_group_connectivity() {
        let mut state = state_with(4, Player::White);
        place(&mut state, 0, 0, 0, false);
        let middle = place(&mut state, 1, 0, 0, false);
        place(&mut state, 2, 0, 0, false);

        let mut groups = Groups::compute(state.n(), &state.occupied, &state.black);
        assert!(
            !state.is_freedom(middle, &mut groups),
            "removing the middle piece disconnects the two endpoints"
        );

        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::Move(from, _, _) if *from as usize == middle)),
            "a non-freedom piece must never be offered as a Move source"
        );
    }

    /// Brute-force oracle for `State::is_freedom`: rebuilds `before`'s
    /// members via a full board scan (rather than
    /// `connectivity::Groups::group_members`'s cache) and the "after"
    /// connectivity via a whole-board `Groups::compute` (rather than
    /// `connectivity::survives_removal`'s scoped flood fill) -- an
    /// independent-in-implementation, identical-in-definition derivation of
    /// the same property, to cross-check the faster path against.
    fn is_freedom_oracle(state: &State, from: usize) -> bool {
        let mut before = Groups::compute(state.n(), &state.occupied, &state.black);
        let members: Vec<usize> = (0..state.total_cells())
            .filter(|&i| i != from && before.same_group(from, i))
            .collect();
        if members.len() <= 1 {
            return true;
        }
        let mut after_occupied = state.occupied;
        after_occupied.clear_index(from);
        let mut after = Groups::compute(state.n(), &after_occupied, &state.black);
        members.windows(2).all(|w| after.same_group(w[0], w[1]))
    }

    /// `State::is_freedom` (backed by `connectivity::Groups::group_members`
    /// and `connectivity::survives_removal`'s scoped flood fill) must agree
    /// with the whole-board brute-force oracle above for every candidate it
    /// checks, across random play at board sizes large enough to have real
    /// over/under crossing pillars -- not just the plain-touching case, but
    /// specifically the case `connectivity::survives_removal`'s doc comment
    /// calls out: removing a piece can newly *cut* a connection it had been
    /// shielding, not just newly restore one it had been blocking. A random
    /// sample of occupied cells (rather than every occupied cell) is checked
    /// each ply, to keep this within `cargo test --lib`'s speed budget while
    /// still exercising many distinct board shapes across seeds/plies.
    fn is_freedom_matches_oracle(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = state.total_cells() * 2;

        for _ in 0..max_plies {
            if Akron::is_terminal(&state) {
                return;
            }
            let occupied: Vec<usize> = state.occupied_indices();
            for _ in 0..3.min(occupied.len()) {
                let index = occupied[rng.gen_range(0..occupied.len())];
                let mut groups = Groups::compute(n, &state.occupied, &state.black);
                assert_eq!(
                    state.is_freedom(index, &mut groups),
                    is_freedom_oracle(&state, index),
                    "n={n} seed={seed}: is_freedom disagrees with the brute-force oracle at cell {index}"
                );
            }
            let mut actions = Vec::new();
            Akron::generate_actions(&state, &mut actions);
            if actions.is_empty() {
                return;
            }
            let action = actions[rng.gen_range(0..actions.len())];
            state = Akron::apply(state, &action);
        }
    }

    /// `random_action`'s `Add`-biased fast path must always agree with
    /// `generate_actions`'s full enumeration: every draw is either `None`
    /// exactly when `generate_actions` is empty, or an action also present
    /// in `generate_actions`'s output. Mirrors `games/margo`'s
    /// `random_action_matches_generate_actions` -- this doesn't (and can't)
    /// check that the draw is uniform, only that it's always legal, since
    /// `random_action`'s own doc comment is explicit that the two
    /// distributions differ on purpose.
    fn random_action_matches_generate_actions(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = state.total_cells() * 2;

        for _ in 0..max_plies {
            if Akron::is_terminal(&state) {
                return;
            }
            let mut actions = Vec::new();
            Akron::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "n={n} seed={seed}: no legal moves on a non-terminal state"
            );
            for _ in 0..8 {
                let drawn = Akron::random_action(&state, &mut rng).expect(
                    "random_action must return Some whenever generate_actions is non-empty",
                );
                assert!(
                    actions.contains(&drawn),
                    "n={n} seed={seed}: random_action drew {drawn:?}, not present in \
                     generate_actions {actions:?}"
                );
            }
            let action = actions[rng.gen_range(0..actions.len())];
            state = Akron::apply(state, &action);
        }
    }

    #[test]
    fn test_random_action_matches_generate_actions() {
        for seed in 0..8 {
            random_action_matches_generate_actions(4, seed);
        }
        for seed in 0..4 {
            random_action_matches_generate_actions(7, seed);
        }
    }

    #[test]
    fn is_freedom_matches_brute_force_oracle_across_random_play() {
        for seed in 0..3 {
            is_freedom_matches_oracle(4, seed);
        }
        for seed in 0..3 {
            is_freedom_matches_oracle(8, seed);
        }
    }

    /// Relocating a piece that uniquely supports another piece drags the
    /// dependent down to fill the vacated gap -- including when the
    /// dependent belongs to the *other* player, exercising the colour-bit
    /// bookkeeping `Game::apply`'s `Action::Move` arm has to do in lockstep
    /// with `pyramid::Pyramid::relocate`'s occupancy-only cascade.
    #[test]
    fn relocation_triggers_a_cascade_drop() {
        let mut state = state_with(4, Player::White);
        // A White 2x2 base block supporting a single Black dependent --
        // support is colour-blind, so this is physically valid even though
        // the dependent belongs to the other player.
        place(&mut state, 0, 0, 0, false);
        place(&mut state, 1, 0, 0, false);
        place(&mut state, 0, 1, 0, false);
        let from = place(&mut state, 1, 1, 0, false);
        let dependent = place(&mut state, 0, 0, 1, true);
        let to = state.occupied.index(2, 0, 0);

        assert_eq!(state.vacated_chain(from), Some(vec![from, dependent]));

        let n = state.n() as u8;
        let state = Akron::apply(state, &Action::Move(from as u16, to as u16, n));

        assert!(
            state.is_occupied(from) && state.is_black(from),
            "the Black dependent drops down to fill the vacated gap"
        );
        assert!(
            !state.is_occupied(dependent),
            "the dependent's old cell is genuinely empty now"
        );
        assert!(
            state.is_occupied(to) && !state.is_black(to),
            "the White mover lands at `to`"
        );
    }

    /// A piece that only touches the rest of the mover's group through a
    /// cell that is *itself* dropping this turn (as another player's piece
    /// resting on the mover, in this fixture) has no legal destination at
    /// all: that cell can't be used as the "touches a connected cell"
    /// anchor, per the published rules' cascade caveat, even though it
    /// would otherwise be one (and even though some other, physically
    /// supported cell touches it).
    #[test]
    fn cascade_dropped_piece_cannot_anchor_a_destination() {
        let mut state = state_with(4, Player::White);
        let from = place(&mut state, 0, 0, 0, false);
        let dependent = place(&mut state, 0, 0, 1, true);
        // Support for a same-level neighbour of `dependent`, entirely in
        // Black -- physically valid (support doesn't care about colour) but
        // contributes no White anchor of its own.
        place(&mut state, 1, 0, 0, true);
        place(&mut state, 2, 0, 0, true);
        place(&mut state, 1, 1, 0, true);
        place(&mut state, 2, 1, 0, true);
        let would_be_destination = state.occupied.index(1, 0, 1);

        assert!(
            state.occupied.can_place(1, 0, 1),
            "the candidate destination must be physically placeable"
        );
        assert!(
            get_adjacency(state.n())
                .neighbors(dependent)
                .into_iter()
                .any(|n| n == would_be_destination),
            "the candidate destination must genuinely touch the dropping piece"
        );

        let chain = state
            .vacated_chain(from)
            .expect("a single dependent cascades cleanly");
        let mut groups = Groups::compute(state.n(), &state.occupied, &state.black);
        assert!(
            state.is_freedom(from, &mut groups),
            "the sole White piece is trivially a freedom"
        );
        let destinations = state.move_destinations(from, &chain, &mut groups);
        assert!(
            !destinations.contains(&would_be_destination),
            "a cell reachable only through this turn's own dropping piece must not be offered"
        );
        assert!(
            destinations.is_empty(),
            "with the dropping piece excluded, this isolated mover has no anchor left at all"
        );
    }

    /// A straight, unbroken row-to-row chain is a Black win, and a
    /// column-to-column chain is a White win -- the two axes are checked
    /// independently and don't cross-credit the other player.
    #[test]
    fn straight_unbroken_chain_completes_a_span() {
        let n = 5;
        let mut black_state = state_with(n, Player::White);
        for row in 0..n {
            place(&mut black_state, 2, row, 0, true);
        }
        assert!(black_state.has_span(Player::Black));
        assert!(!black_state.has_span(Player::White));
        assert_eq!(Akron::winner(&black_state), Some(Player::Black));
        assert!(Akron::is_terminal(&black_state));

        let mut white_state = state_with(n, Player::White);
        for col in 0..n {
            place(&mut white_state, col, 2, 0, false);
        }
        assert!(white_state.has_span(Player::White));
        assert!(!white_state.has_span(Player::Black));
        assert_eq!(Akron::winner(&white_state), Some(Player::White));
    }

    /// A board with pieces from both players but no completed span for
    /// either is not terminal and has no winner.
    #[test]
    fn incomplete_chain_is_not_a_win() {
        let n = 5;
        let mut state = state_with(n, Player::White);
        for row in 0..(n - 1) {
            place(&mut state, 2, row, 0, true);
        }
        assert!(!state.has_span(Player::Black));
        assert_eq!(Akron::winner(&state), None);
        assert!(!Akron::is_terminal(&state));
    }

    /// A straight, physically-touching row-to-row chain does *not* count as
    /// a span once an opposing overpass cuts one of its edges -- the win
    /// condition uses [`connectivity::Groups`]' cut-aware connectivity, not
    /// raw touching adjacency. The cutting position is derived from
    /// `pyramid::crossing::get_crossing_table` itself (the same oracle
    /// `connectivity`'s own tests trust), rather than hand-derived, per
    /// this repo's "don't hand-derive capture/crossing geometry" lesson.
    #[test]
    fn cut_connection_does_not_count_as_a_span() {
        let n = 10;
        let mut state = state_with(n, Player::White);
        for row in 0..n {
            place(&mut state, 4, row, 0, true);
        }
        let a = state.occupied.index(4, 4, 0);
        let b = state.occupied.index(4, 5, 0);
        let edge = (a.min(b), a.max(b));
        let partner = get_crossing_table(n)[&edge][0];
        let (pc, pr, pl) = state.occupied.to_coord(partner.0);
        place(&mut state, pc, pr, pl, false);
        let (qc, qr, ql) = state.occupied.to_coord(partner.1);
        place(&mut state, qc, qr, ql, false);

        assert!(
            !state.has_span(Player::Black),
            "the straight chain is cut partway by White's overpass"
        );
        assert_eq!(Akron::winner(&state), None);
        assert!(!Akron::is_terminal(&state));
    }

    /// The same cut position as above, but with an uncut detour around the
    /// cut edge -- the span is completed through the detour, since the win
    /// condition only requires *some* unbroken path between the two sides,
    /// not that the most direct one survives.
    #[test]
    fn detour_around_a_cut_edge_still_completes_the_span() {
        let n = 10;
        let mut state = state_with(n, Player::White);
        for row in 0..n {
            place(&mut state, 4, row, 0, true);
        }
        let a = state.occupied.index(4, 4, 0);
        let b = state.occupied.index(4, 5, 0);
        let edge = (a.min(b), a.max(b));
        let partner = get_crossing_table(n)[&edge][0];
        let (pc, pr, pl) = state.occupied.to_coord(partner.0);
        place(&mut state, pc, pr, pl, false);
        let (qc, qr, ql) = state.occupied.to_coord(partner.1);
        place(&mut state, qc, qr, ql, false);
        // Detour: (4,4,0) - (3,4,0) - (3,5,0) - (4,5,0), none of whose
        // edges are cut, bypassing the direct edge cut above.
        place(&mut state, 3, 4, 0, true);
        place(&mut state, 3, 5, 0, true);

        assert!(
            state.has_span(Player::Black),
            "an uncut detour around the cut edge still completes the span"
        );
        assert_eq!(Akron::winner(&state), Some(Player::Black));
    }

    /// Re-cutting the cutter (mirroring `connectivity`'s own Figure 5/6
    /// narrative test) restores a previously-cut span: the direct chain
    /// from `cut_connection_does_not_count_as_a_span` above is not a win
    /// while White's overpass cuts it, but becomes one again once Black
    /// places an even-higher overpass that cuts White's cutter.
    #[test]
    fn recutting_the_cutter_restores_a_previously_cut_span() {
        let n = 10;
        let mut state = state_with(n, Player::White);
        for row in 0..n {
            place(&mut state, 4, row, 0, true);
        }
        let a = state.occupied.index(4, 4, 0);
        let b = state.occupied.index(4, 5, 0);
        let edge = (a.min(b), a.max(b));
        let chain = &get_crossing_table(n)[&edge];
        let cutter = chain[0];
        let (pc, pr, pl) = state.occupied.to_coord(cutter.0);
        place(&mut state, pc, pr, pl, false);
        let (qc, qr, ql) = state.occupied.to_coord(cutter.1);
        place(&mut state, qc, qr, ql, false);
        assert!(
            !state.has_span(Player::Black),
            "precondition: White's overpass cuts the direct chain"
        );

        // The cutter's own pillar (from the crossing table keyed on the
        // cutter edge itself) gives a still-higher partner to re-cut it
        // with -- Black, restoring the original span underneath.
        let cutter_key = (cutter.0.min(cutter.1), cutter.0.max(cutter.1));
        let recutter = get_crossing_table(n)[&cutter_key]
            .first()
            .copied()
            .expect("the cutter edge used here has a further crossing partner");
        let (rc, rr, rl) = state.occupied.to_coord(recutter.0);
        place(&mut state, rc, rr, rl, true);
        let (sc, sr, sl) = state.occupied.to_coord(recutter.1);
        place(&mut state, sc, sr, sl, true);

        assert!(
            state.has_span(Player::Black),
            "cutting White's cutter restores Black's original span"
        );
    }

    /////////////////////////////////////////////////////////////////////////
    // Symmetry: `apply_to_action`/`invert_action` round-trip,
    // `canonical_representation` invariance across every symmetric image of
    // a state, `invert_action` always producing a legal real action along
    // random play, and hash consistency -- mirroring `games/margo`'s own
    // symmetry test suite.

    /// Both geometric fields (`occupied`/`black`) transformed through each
    /// of the 8 `PyramidD4` elements -- the full set of states that must all
    /// canonicalize identically, since they're the same position viewed
    /// from 8 different orientations. `white_pile`/`black_pile`/`turn`/
    /// `can_swap` aren't geometric, so they ride along unchanged.
    fn state_symmetries(state: &State) -> [State; 8] {
        let n = state.occupied.n();
        let sym = PyramidD4::new(n);
        std::array::from_fn(|i| State {
            occupied: transform_cells(&state.occupied, &sym, i),
            black: transform_cells(&state.black, &sym, i),
            white_pile: state.white_pile,
            black_pile: state.black_pile,
            turn: state.turn,
            can_swap: state.can_swap,
        })
    }

    #[test]
    fn action_transform_round_trip() {
        let n = DEFAULT_N as u8;
        let total = pyramid::total_cells(DEFAULT_N);
        for index in 0..total {
            for sym in 0..8usize {
                let sym = Transform::new(sym);
                let add = Action::Add(index as u16, n);
                let back = Akron::invert_action(Akron::apply_to_action(Real(add), sym), sym);
                assert_eq!(back.into_inner(), add);

                let to = (index + 1) % total;
                let mv = Action::Move(index as u16, to as u16, n);
                let back = Akron::invert_action(Akron::apply_to_action(Real(mv), sym), sym);
                assert_eq!(back.into_inner(), mv);
            }
        }
        for sym in 0..8usize {
            let sym = Transform::new(sym);
            assert_eq!(
                Akron::apply_to_action(Real(Action::Swap), sym).into_inner(),
                Action::Swap
            );
            assert_eq!(
                Akron::invert_action(Canonical(Action::Swap), sym).into_inner(),
                Action::Swap
            );
        }
    }

    /// Plays a short random game (adds/moves/swap all in scope), and at
    /// every reached state, checks that `canonical_representation` agrees
    /// across all 8 of that state's own symmetric images.
    fn check_canonical_representation_invariant(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let mut reachable = vec![state.clone()];
        let max_plies = state.occupied.total_cells() / 2;
        for _ in 0..max_plies {
            if Akron::is_terminal(&state) {
                break;
            }
            let mut actions = Vec::new();
            Akron::generate_actions(&state, &mut actions);
            let action = actions[rng.gen_range(0..actions.len())];
            state = Akron::apply(state, &action);
            reachable.push(state.clone());
        }

        for state in reachable {
            let (canon, _sym) = Akron::canonical_representation(Real(state.clone()));
            let canon_state = canon.into_inner();

            for variant in state_symmetries(&state) {
                let (canon2, _) = Akron::canonical_representation(Real(variant));
                assert_eq!(
                    canon2.into_inner(),
                    canon_state,
                    "canonical_representation disagreed across symmetric images \
                     (n={n}, seed={seed})"
                );
            }
        }
    }

    #[test]
    fn canonical_representation_invariant_under_symmetry() {
        for seed in 0..20 {
            check_canonical_representation_invariant(MIN_N, seed);
        }
        for seed in 0..10 {
            check_canonical_representation_invariant(DEFAULT_N, seed);
        }
    }

    /// Along random play, every action `generate_actions` offers on the
    /// canonicalized state, translated back via `invert_action`, must be
    /// present in the real state's own `generate_actions` output.
    fn check_invert_action_legal_along_random_game(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = state.occupied.total_cells() + 2;

        for _ in 0..max_plies {
            if Akron::is_terminal(&state) {
                return;
            }
            let mut real_actions = Vec::new();
            Akron::generate_actions(&state, &mut real_actions);

            let (canon, sym) = Akron::canonical_representation(Real(state.clone()));
            let canon = canon.into_inner();
            let mut canon_actions = Vec::new();
            Akron::generate_actions(&canon, &mut canon_actions);

            for &canon_action in &canon_actions {
                let translated = Akron::invert_action(Canonical(canon_action), sym).into_inner();
                assert!(
                    real_actions.contains(&translated),
                    "seed {seed}, n={n}: invert_action produced {translated:?} (from \
                     canonical {canon_action:?}, sym {sym:?}), not present in real \
                     generate_actions {real_actions:?}"
                );
            }

            let action = real_actions[rng.gen_range(0..real_actions.len())];
            state = Akron::apply(state, &action);
        }
    }

    #[test]
    fn invert_action_produces_legal_real_actions() {
        for seed in 0..30 {
            check_invert_action_legal_along_random_game(MIN_N, seed);
        }
        for seed in 0..15 {
            check_invert_action_legal_along_random_game(DEFAULT_N, seed);
        }
    }

    /// Random-sampled hash-consistency check: any two states sharing a
    /// `zobrist_hash` must have the same `canonical_representation` output.
    #[test]
    fn random_games_hash_consistency() {
        use std::collections::HashMap;

        let mut rng = SmallRng::seed_from_u64(9);
        let mut by_hash: HashMap<u64, State> = HashMap::new();
        let mut mismatches = 0;
        for _game in 0..200 {
            let mut state = State::new(MIN_N);
            for _ in 0..40 {
                if Akron::is_terminal(&state) {
                    break;
                }
                let mut actions = Vec::new();
                Akron::generate_actions(&state, &mut actions);
                let action = actions[rng.gen_range(0..actions.len())];
                state = Akron::apply(state, &action);

                let h = state.zobrist_hash();
                let (canon, _sym) = Akron::canonical_representation(Real(state.clone()));
                let canon_state = canon.into_inner();
                match by_hash.get(&h) {
                    Some(prev) if *prev != canon_state => {
                        mismatches += 1;
                        println!("MISMATCH at hash {h}");
                    }
                    Some(_) => {}
                    None => {
                        by_hash.insert(h, canon_state);
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "hash collided across different equivalence classes"
        );
    }
}
