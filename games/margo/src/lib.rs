//! Margo (pyramidal marble game) on a base-`n` pyramid (`pyramid::Pyramid`).
//!
//! Board size is a runtime parameter, not a distinct compiled type per size
//! (mirroring `games/gonnect`'s variable board size): the published
//! recommended sizes run 4x4 ("Spargo", the Shibumi-set size) through 10x10+
//! ("Extreme"), with 7x7 ("Standard") the most common recommendation and
//! this crate's default. `MAX_N` (10) fixes the storage width (`[u64; 7]`,
//! 448 bits, comfortably covering `total_cells(10) == 385`); every smaller
//! size reuses the same monomorphization with a runtime-sized `Pyramid`.
//!
//! A piece may be placed on any empty, supported cell (`Pyramid::can_place`).
//! Capture is Go-style over the *visible* (non-buried, non-zombie)
//! touching-adjacency graph, but a liberty is specifically an empty
//! *board-level* (level 0) cell reachable through that graph -- an empty
//! higher-level stacking point, even a fully-supported one, gives no group a
//! liberty (Margo Basics, "Freedoms only exist on the board level"). After a
//! placement, any enemy group without a liberty is removed -- except a
//! member pinned by a capturing-colour piece
//! resting directly on top of it, which survives in place as a "zombie"
//! (still occupying its cell and still counted for scoring, but permanently
//! excluded from future connectivity, like a buried piece). Suicide is
//! illegal unless the placement itself captures enough to give the new
//! piece a liberty, and a piece hidden by another piece directly above it
//! doesn't count toward any group's connectivity or liberties even though
//! it's still physically on the board. A placement is also illegal if its
//! resulting `(occupied, black)` pair would exactly recreate the position
//! as it stood immediately before the previous move (single-position ko,
//! not full positional superko). White moves first; Black may, in lieu of
//! their first move, swap colours instead (the pie rule) -- see
//! `Action::Swap`/`State::can_swap`. The game ends when the player to move
//! has no legal placement, and the winner is whoever has more pieces on the
//! board (a draw if equal).

mod heuristic;

use std::fmt;

use bitboard::{Board, Dyn};
use mcts::game::{Canonical, Game, PlayerIndex, Real, Transform};
use mcts::zobrist::LazyZobristTable;
use pyramid::{Pyramid, PyramidD4, TouchingAdjacency};
use serde::{Deserialize, Serialize};

pub use heuristic::Heuristic;

mod raster;

/// Smallest supported base width -- "Spargo", the size a physical Shibumi
/// ball set (4x4x4) plays Margo on.
pub const MIN_N: usize = 4;

/// Largest supported base width -- the published rules' "Extreme" size.
/// Fixes `Cells`/`GoBoard`'s storage width (`[u64; 7]`, since
/// `pyramid::total_cells(10) == 385` needs 7 words); every smaller size
/// reuses the same storage at runtime, unused high bits simply staying zero.
pub const MAX_N: usize = 10;

/// `State::default()`'s board size -- "Standard", the rules' own
/// recommendation for most games.
pub const DEFAULT_N: usize = 7;

type Cells = Pyramid<[u64; 7], Dyn>;

/// A flat bitset addressed by the same index as `Cells`, used only as a
/// container for [`resolve_captures`]'s Go-style flood fill over
/// `TouchingAdjacency` -- a single-row `Dyn` board, since the pyramid's flat
/// index has no natural row/col split of its own.
type GoBoard = Board<[u64; 7], Dyn, Dyn>;

fn go_board(cells: usize) -> GoBoard {
    GoBoard::new(Dyn(1), Dyn(cells))
}

/// The level-0 (board-level) cells of a base-`n` pyramid, as a [`GoBoard`]
/// mask -- "Freedoms only exist on the board level" (Margo Basics, Groups):
/// an empty higher-level stacking point, even a fully-supported one, gives
/// no group a liberty, only an empty *board hole* does. Level 0 is always
/// flat indices `[0, n * n)` (`pyramid::level_offset(n, 0) == 0`, and level
/// 0's side is `n` -- see `pyramid::level_side`).
fn ground_mask(n: usize, cells: usize) -> GoBoard {
    let mut mask = go_board(cells);
    for index in 0..(n * n) {
        mask.set_index(index);
    }
    mask
}

/// Build per-level raster-indexed color masks: `own[l]` has bits set
/// at raster positions where the mover has a piece at level `l`.
fn build_color_masks(
    n: usize,
    state: &State,
) -> (Vec<raster::LevelBoard>, Vec<raster::LevelBoard>) {
    use bitboard::{Board, Dyn};
    let mut own: Vec<Board<[u64; 2], Dyn, Dyn>> =
        (0..n).map(|_| Board::new(Dyn(n), Dyn(n))).collect();
    let mut opp: Vec<Board<[u64; 2], Dyn, Dyn>> =
        (0..n).map(|_| Board::new(Dyn(n), Dyn(n))).collect();
    let mover_black = state.turn == Player::Black;
    for idx in 0..state.occupied.total_cells() {
        if !state.occupied.get_index(idx) {
            continue;
        }
        let (col, row, level) = state.occupied.to_coord(idx);
        let pos = row * n + col;
        let is_black = state.black.get_index(idx);
        if (is_black && mover_black) || (!is_black && !mover_black) {
            own[level].set_index(pos);
        } else {
            opp[level].set_index(pos);
        }
    }
    (own, opp)
}

/// The non-zombie subset of `black`/`white` occupancy, split into per-colour
/// [`GoBoard`]s -- the input [`resolve_captures`] runs its touching-adjacency
/// flood fill over. Zombie cells are NOT excluded: a zombie still physically
/// touches its neighbors and participates in group connectivity. The zombie
/// mask only determines which group members survive capture (zombies stay,
/// non-zombies are removed).
///
/// "Buried" (visually occluded from two levels up) does *not* affect
/// connectivity -- a buried piece still physically touches its neighbors,
/// and only the touching graph determines group membership.
fn visible_boards(occupied: &Cells, black: &Cells) -> (GoBoard, GoBoard) {
    let cells = occupied.total_cells();
    let mut black_board = go_board(cells);
    let mut white_board = go_board(cells);
    for index in occupied.iter_set() {
        // Note: we do NOT filter out zombies here. A zombie still
        // physically touches its neighbors and participates in group
        // connectivity. The zombie mask only determines which group
        // members survive capture (zombies stay, non-zombies are removed).
        if black.get_index(index) {
            black_board.set_index(index);
        } else {
            white_board.set_index(index);
        }
    }
    (black_board, white_board)
}

/// Go-style legality/capture check for a stone already placed at `index` in
/// `own` (mirrors `bitboard::check_go_move`'s algorithm exactly, but over
/// `TouchingAdjacency`'s table via `table_flood`/`table_neighbor_mask`
/// instead of `Board::flood4`'s rectangular shift math, and taking `own`
/// with the candidate stone already set rather than seeding it internally --
/// a placement can newly bury an existing, already-placed piece elsewhere
/// (occluding it from two levels up), so `own`/`opp` are always rebuilt
/// from scratch against the post-placement buried set rather than patched
/// incrementally, leaving no separate pre-placement board to seed from).
/// Returns the mask
/// of opponent cells to capture if legal, or `None` if the placement is
/// suicide.
///
/// Closing a captured cell's last liberty by occupying its dependent two
/// levels above one of its own supporters has the same retroactive-burial
/// effect: the newly-placed piece buries that supporter (it's exactly the
/// occluder position for it), which was also one of the captured cell's own
/// neighbors, permanently reopening a liberty. This makes some captures --
/// specifically, ones whose last liberty is two levels above an existing
/// piece the captured cell also depends on -- structurally unreachable, not
/// just hard to set up by hand (see `apply_captures_splits_pinned_and_
/// removed_members`, which tests the zombie/removal split directly against
/// a hand-built capture mask rather than fighting this to build one).
fn resolve_captures(
    own: GoBoard,
    opp: GoBoard,
    index: usize,
    adjacency: &TouchingAdjacency,
    ground: GoBoard,
) -> Option<GoBoard> {
    debug_assert!(own.get_index(index));
    debug_assert!(!opp.get_index(index));
    let occupied = own | opp;
    let group = bitboard::table_flood(own, adjacency, index);
    let group_adjacent = bitboard::table_neighbor_mask(group, adjacency);
    let empty_adjacent = !occupied & group_adjacent & ground;
    let safe = !empty_adjacent.none_set();

    let occupied_adjacent = occupied & group_adjacent;
    let mut seen = own.empty_like();
    let mut will_capture = own.empty_like();
    for point in occupied_adjacent {
        debug_assert!(
            opp.get_index(point),
            "own's flood already covers same-colour neighbors"
        );
        if !seen.get_index(point) {
            let enemy_group = bitboard::table_flood(opp, adjacency, point);
            seen |= enemy_group;
            let enemy_adjacent = bitboard::table_neighbor_mask(enemy_group, adjacency);
            let enemy_empty_adjacent = !occupied & enemy_adjacent & ground;
            if enemy_empty_adjacent.none_set() {
                will_capture |= enemy_group;
            }
        }
    }

    (safe || !will_capture.none_set()).then_some(will_capture)
}

/// Splits a computed capture mask into zombies and outright removals: a
/// captured cell survives in place, keeping its colour, iff a
/// capturing-colour piece rests directly on top of it (one of its own
/// dependents belongs to the mover) -- removing it would leave that piece
/// floating. `mover_black` identifies the capturing colour. Factored out of
/// [`State::resolve_place`] so the zombie/removal split can be unit tested
/// directly against a hand-built capture mask, without needing a
/// [`resolve_captures`]-derived one -- mirroring how
/// `placement_that_captures_to_gain_a_liberty_is_accepted` tests
/// `resolve_captures` directly when a full `State`-reachable scenario isn't
/// tractable to construct by hand.
fn apply_captures(
    mut occupied: Cells,
    mut black: Cells,
    mut zombie: Cells,
    captured: GoBoard,
    mover_black: bool,
) -> (Cells, Cells, Cells) {
    for cell in captured {
        if zombie.get_index(cell) {
            // An existing zombie permanently survives every future
            // capture it participates in -- it's pinned for good,
            // regardless of whether its original pinning piece is
            // still on the board.
            continue;
        }
        let (c, r, l) = occupied.to_coord(cell);
        let pinned_by_capturer = occupied
            .dependents(c, r, l)
            .into_iter()
            .any(|(dc, dr)| black.get(dc, dr, l + 1) == mover_black);
        if pinned_by_capturer {
            zombie.set_index(cell);
        } else {
            occupied.clear_index(cell);
            black.clear_index(cell);
        }
    }
    (occupied, black, zombie)
}

// ── Symmetry / Zobrist hashing ──────────────────────────────────────────
//
// Whole-pyramid D4 (`pyramid::PyramidD4`), array-of-hashes per symmetry
// element -- the same shape `games/gonnect`/`games/atarigo` use, adapted to
// `PyramidD4`'s own API (a flat `index_symmetries`/`invert_symmetry` pair,
// no separate rows/cols).
//
// Unlike Gonnect (which incrementally XORs its `hashes: [u64; 8]` field on
// every `apply`), Margo recomputes the whole array from scratch whenever
// it's needed (`State::zobrist_hash`), never storing it -- the same
// rebuild-rather-than-incrementally-maintain choice `visible_boards` above
// already makes for buried/visible occupancy, on the same reasoning: the
// board is small (<= 385 cells), and captures, zombie transitions, and the
// wholesale swap of `previous` on every move already touch most of the
// board's geometric fields, so incremental XOR maintenance here would save
// little over a full rebuild while adding a second place (alongside
// `resolve_place`/`apply_captures`) that has to get every one of those
// transitions exactly right.

/// A cell can independently belong to `occupied`/`black`/`zombie` (the
/// current position) and to the `previous` ko snapshot's own `occupied`/
/// `black` pair -- see `State`'s doc comment for why `previous` is a
/// geometric field (must rotate/reflect with everything else under
/// `Game::canonical_representation`) rather than a non-geometric one like
/// `turn`/`can_swap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel {
    Occupied = 0,
    Black = 1,
    Zombie = 2,
    PrevOccupied = 3,
    PrevBlack = 4,
}

const HASH_CHANNELS: usize = 5;

/// Largest board this game serves is `MAX_N`'s `total_cells(MAX_N) == 385`
/// cells, `HASH_CHANNELS` channels per cell, plus one slot each for `turn`,
/// `can_swap`, and whether `previous` is present at all (`Some` vs. `None`
/// is itself game-relevant: see `ZOBRIST_HAS_PREVIOUS`'s own doc comment).
const MAX_CELLS: usize = pyramid::total_cells(MAX_N);
pub const ZOBRIST_ENTRIES: usize = MAX_CELLS * HASH_CHANNELS + 3;
pub const ZOBRIST_TURN: usize = MAX_CELLS * HASH_CHANNELS;
pub const ZOBRIST_CAN_SWAP: usize = MAX_CELLS * HASH_CHANNELS + 1;
/// `previous == None` (the initial empty board) and `previous == Some((empty,
/// empty))` (reached after White's first placement, before anything has
/// been captured) are different game states -- the first can never trigger
/// ko, the second forbids ever recreating an empty board again -- but both
/// would hash identically on the `PrevOccupied`/`PrevBlack` channels alone
/// (all-zero either way) without this extra bit distinguishing "no
/// previous" from "previous happens to be all-zero".
pub const ZOBRIST_HAS_PREVIOUS: usize = MAX_CELLS * HASH_CHANNELS + 2;

/// Random Zobrist table, lazily initialised.
pub static HASHES: LazyZobristTable<ZOBRIST_ENTRIES> = LazyZobristTable::new(0xD34D9F1C7A02E6B5);

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

/// XOR a position-independent constant (turn, can_swap, has-previous) into
/// all 8 hashes.
#[inline]
fn xor_const(hashes: &mut [u64; 8], table_idx: usize) {
    let v = HASHES.hash(table_idx);
    for h in hashes.iter_mut() {
        *h ^= v;
    }
}

/// Rebuilds all 8 symmetry hashes from scratch -- see this section's own
/// doc comment for why this is called fresh rather than incrementally
/// maintained as a `State` field.
fn rebuild_hashes(
    occupied: &Cells,
    black: &Cells,
    zombie: &Cells,
    previous: Option<(Cells, Cells)>,
    turn: Player,
    can_swap: bool,
) -> [u64; 8] {
    let sym = PyramidD4::new(occupied.n());
    let mut hashes = [0u64; 8];
    xor_cells(&mut hashes, occupied, Channel::Occupied, &sym);
    xor_cells(&mut hashes, black, Channel::Black, &sym);
    xor_cells(&mut hashes, zombie, Channel::Zombie, &sym);
    if let Some((prev_occupied, prev_black)) = previous {
        xor_cells(&mut hashes, &prev_occupied, Channel::PrevOccupied, &sym);
        xor_cells(&mut hashes, &prev_black, Channel::PrevBlack, &sym);
        xor_const(&mut hashes, ZOBRIST_HAS_PREVIOUS);
    }
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
/// its own (a bare `[u64; 7]` word pattern isn't otherwise meaningful to
/// compare), so this uses the ascending list of set indices instead, which
/// `iter_set` already produces in order.
fn cells_key(cells: &Cells) -> Vec<usize> {
    cells.iter_set().collect()
}

/// Index of the symmetry whose image of the *entire* geometric state
/// (`occupied`, `black`, `zombie`, `previous`) is lexicographically
/// minimal -- the canonical orientation for the position.
///
/// Unlike `games/gonnect`'s `canonical_symmetry` (which ties-break on
/// `(black, white)` alone, reasoning that every other geometric field rides
/// along under the same chosen symmetry regardless), Margo's own version of
/// that shortcut is unsound: when `(occupied, black)` happens to be
/// invariant under some non-identity element `g` (a genuine tie), `zombie`/
/// `previous` need not *also* be `g`-invariant, so `min_by_key`'s
/// first-minimum tie-break picks whichever `g` sorts first in `0..8` --
/// which, relative to a symmetric *image* of the same position, is a
/// different index than relative to the original, sending `zombie`/
/// `previous` to inconsistent results across symmetric inputs. Caught by
/// `canonical_representation_invariant_under_symmetry` (a 4x4 board with a
/// self-symmetric `(occupied, black)` pattern but an asymmetric `previous`)
/// rather than reasoned out by hand. Including every geometric field in
/// the tie-break makes the minimizer's *image* unique even when several
/// symmetry indices achieve it, which is what `canonical_representation`
/// actually needs.
fn canonical_symmetry(
    occupied: &Cells,
    black: &Cells,
    zombie: &Cells,
    previous: Option<(Cells, Cells)>,
) -> usize {
    let sym = PyramidD4::new(occupied.n());
    (0..8)
        .min_by_key(|&sym_idx| {
            let previous_key = previous.map(|(prev_occupied, prev_black)| {
                (
                    cells_key(&transform_cells(&prev_occupied, &sym, sym_idx)),
                    cells_key(&transform_cells(&prev_black, &sym, sym_idx)),
                )
            });
            (
                cells_key(&transform_cells(occupied, &sym, sym_idx)),
                cells_key(&transform_cells(black, &sym, sym_idx)),
                cells_key(&transform_cells(zombie, &sym, sym_idx)),
                previous_key,
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

/// A move: place a piece, or (Black's pie-rule reply to White's opening
/// move only, see [`State::can_swap`]) swap colours instead.
#[derive(Copy, Clone, Serialize, Deserialize, Debug, Hash, PartialEq, Eq)]
pub enum Action {
    /// Place a piece at flat pyramid index `.0` (see `pyramid::Pyramid::
    /// index`/`to_coord`). `u16` because `total_cells(MAX_N) == 385`
    /// overflows `u8`. `.1` is the board's base width `n` -- carried along
    /// because `Game::apply_to_action`/`invert_action` need it to build a
    /// `PyramidD4` symmetry, and (unlike `apply`, which always has a
    /// `State` in scope) their trait signature carries only the action and
    /// a `Transform` index, no state. Mirrors `games/gonnect`'s `Move`,
    /// which carries its own board size for the identical reason.
    Place(u16, u8),
    /// Recolour the single piece on the board to the opposite colour,
    /// taken instead of a placement. Legal only when [`State::can_swap`]
    /// holds, i.e. exactly one piece (White's opening move) is on the
    /// board and nothing has closed the swap window yet. A fixed point of
    /// every symmetry element, mirroring how `games/gonnect`/`games/
    /// atarigo` treat their own `SWAP`/`PASS` sentinels.
    Swap,
}

/// Board state: `occupied` is every placed piece regardless of colour (the
/// bitset `Pyramid::can_place`/`is_supported` operate over, since support is
/// colour-agnostic); `black` marks which of those cells belong to Black --
/// White's pieces are `occupied & !black`, derived rather than stored
/// separately so the two boards can't drift out of sync with each other.
/// `zombie` marks cells that survived capture pinned by a capturing-colour
/// piece resting directly on top (see the module docs) -- still set in
/// `occupied`/`black`, but permanently excluded from visible connectivity
/// the same way a buried cell is, since the support structure that pins them
/// never un-forms. Board size lives entirely in `occupied`/`black`'s own
/// `Dyn` dimension (`Pyramid::n`), not a separate field, so it can never
/// disagree with them. `previous` is the `(occupied, black)` pair as it
/// stood immediately before the move that produced this state -- together
/// they fully determine the position (white is always `occupied & !black`),
/// so this is the position ko forbids the player to move from recreating
/// (single-position ko against the immediately preceding position, not full
/// positional superko: see the module docs). `can_swap` tracks whether the
/// pie-rule swap window is still open -- true from the empty board until
/// the piece count first leaves 1 (i.e. until someone other than White's
/// opening placement moves), following `games/gonnect`'s identical
/// `can_swap` pattern: closed by any placement that isn't the very first,
/// and explicitly by `Action::Swap` itself, so it can never reopen even if
/// a later capture happens to drop the piece count back to 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    occupied: Cells,
    black: Cells,
    zombie: Cells,
    previous: Option<(Cells, Cells)>,
    turn: Player,
    can_swap: bool,
}

impl Default for State {
    fn default() -> Self {
        Self::new(DEFAULT_N)
    }
}

impl State {
    /// A fresh empty base-`n` board. `n` must be within `MIN_N..=MAX_N`.
    pub fn new(n: usize) -> Self {
        assert!(
            (MIN_N..=MAX_N).contains(&n),
            "Margo board size must be between {MIN_N} and {MAX_N}, got {n}"
        );
        Self {
            occupied: Cells::new(Dyn(n)),
            black: Cells::new(Dyn(n)),
            zombie: Cells::new(Dyn(n)),
            previous: None,
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
    pub fn is_zombie(&self, index: usize) -> bool {
        self.zombie.get_index(index)
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

    /// The `previous` ko snapshot as `(occupied, black)` flat-index lists,
    /// for a wire adapter to serialize -- mirrors `occupied_indices`/
    /// `black_indices` below, since `previous` stores the same shape of data
    /// one ply back.
    pub fn previous_indices(&self) -> Option<(Vec<usize>, Vec<usize>)> {
        self.previous
            .map(|(o, b)| (o.iter_set().collect(), b.iter_set().collect()))
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

    /// Every zombie cell's flat index, for a wire adapter to serialize.
    pub fn zombie_indices(&self) -> Vec<usize> {
        self.zombie.iter_set().collect()
    }

    /// Reconstructs a `State` from flat-index lists -- the inverse of
    /// `occupied_indices`/`black_indices`/`zombie_indices`/
    /// `previous_indices`, for a wire adapter to deserialize a JSON request
    /// back into a real `State` without going through legal play. No
    /// legality checking is done here: the caller (a `GameAdapter`
    /// round-tripping its own previously emitted wire format) is trusted to
    /// pass back a state this crate itself produced.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        n: usize,
        occupied: &[usize],
        black: &[usize],
        zombie: &[usize],
        previous: Option<(&[usize], &[usize])>,
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
            zombie: fill(zombie),
            previous: previous.map(|(o, b)| (fill(o), fill(b))),
            turn,
            can_swap,
        }
    }

    /// Whether `Action::Swap` is currently legal -- see the field's own doc
    /// comment on [`State`].
    #[inline]
    pub fn can_swap(&self) -> bool {
        self.can_swap && self.occupied.count_ones() == 1
    }

    /// Whether the swap window has been permanently closed yet (the raw
    /// `can_swap` field, distinct from [`can_swap`](Self::can_swap)'s
    /// additional "exactly one piece on the board right now" check) -- a
    /// wire adapter needs this to serialize/reconstruct the window's
    /// open/closed bit itself, since `can_swap()`'s piece-count check would
    /// otherwise make every non-ply-1 state round-trip as permanently
    /// closed regardless of whether the window had actually been used yet.
    #[inline]
    pub fn swap_window_open(&self) -> bool {
        self.can_swap
    }

    /// Attempts to place the player to move's piece at flat `index`,
    /// returning the resulting `(occupied, black, zombie)` boards (captures
    /// already applied) if legal, or `None` if the cell isn't a legal
    /// placement (unsupported/occupied) or would be suicide.
    fn resolve_place(
        &self,
        index: usize,
        adjacency: &TouchingAdjacency,
    ) -> Option<(Cells, Cells, Cells)> {
        let (col, row, level) = self.occupied.to_coord(index);
        if !self.occupied.can_place(col, row, level) {
            return None;
        }
        let mover_black = self.turn == Player::Black;
        let (black_board, white_board) = visible_boards(&self.occupied, &self.black);
        let (own, opp) = if mover_black {
            let mut own = black_board;
            own.set_index(index);
            (own, white_board)
        } else {
            let mut own = white_board;
            own.set_index(index);
            (own, black_board)
        };
        let ground = ground_mask(self.occupied.n(), self.occupied.total_cells());
        self.resolve_place_inner(index, adjacency, own, opp, ground)
    }

    /// Core of [`resolve_place`]: assumes `index` is already physically
    /// placeable (`can_place` already passed, cell not occupied). Checks
    /// capture legality (suicide), applies captures/zombification, and
    /// enforces ko. The caller supplies precomputed `own`/`opp` visible
    /// boards (with the candidate stone already set in `own`) and `ground`
    /// mask so `generate_actions` can compute them once instead of per
    /// candidate.
    fn resolve_place_inner(
        &self,
        index: usize,
        adjacency: &TouchingAdjacency,
        own: GoBoard,
        opp: GoBoard,
        ground: GoBoard,
    ) -> Option<(Cells, Cells, Cells)> {
        let mover_black = self.turn == Player::Black;
        let mut new_occupied = self.occupied;
        new_occupied.set_index(index);
        let mut new_black = self.black;
        if mover_black {
            new_black.set_index(index);
        }

        let captured = resolve_captures(own, opp, index, adjacency, ground)?;
        let (new_occupied, new_black, new_zombie) =
            apply_captures(new_occupied, new_black, self.zombie, captured, mover_black);

        if self.previous == Some((new_occupied, new_black)) {
            return None;
        }

        Some((new_occupied, new_black, new_zombie))
    }

    fn piece_count(&self, player: Player) -> u32 {
        match player {
            Player::Black => self.black.count_ones(),
            Player::White => self.occupied.count_ones() - self.black.count_ones(),
        }
    }

    /// This state's Zobrist hash, symmetry-invariant: two states that are
    /// symmetric images of each other under `PyramidD4` hash identically,
    /// since both pick out the same slot of `rebuild_hashes`'s per-symmetry
    /// array via [`canonical_symmetry`] (see `games/gonnect`'s identical
    /// `State::hash` for why this is the trick that makes the array-of-
    /// hashes design work).
    fn zobrist_hash(&self) -> u64 {
        let hashes = rebuild_hashes(
            &self.occupied,
            &self.black,
            &self.zombie,
            self.previous,
            self.turn,
            self.can_swap,
        );
        hashes[canonical_symmetry(&self.occupied, &self.black, &self.zombie, self.previous)]
    }
}

#[derive(Clone)]
pub struct Margo;

impl Game for Margo {
    type S = State;
    type A = Action;
    type P = Player;

    fn apply(mut state: State, action: &Action) -> State {
        let previous = (state.occupied, state.black);
        match *action {
            Action::Place(index, _n) => {
                let adjacency = pyramid::get_adjacency(state.occupied.n());
                let (new_occupied, new_black, new_zombie) = state
                    .resolve_place(index as usize, adjacency)
                    .expect("action generated by generate_actions must be legal");
                state.occupied = new_occupied;
                state.black = new_black;
                state.zombie = new_zombie;
            }
            Action::Swap => {
                debug_assert!(
                    state.can_swap(),
                    "action generated by generate_actions must be legal"
                );
                // Recolour the single piece on the board (White's opening
                // move) to Black -- general "flip every occupied cell's
                // colour" rather than a single-bit special case, since
                // that's what a swap means regardless of how many pieces
                // happen to be down when it's legal.
                for index in 0..state.occupied.total_cells() {
                    if !state.occupied.get_index(index) {
                        continue;
                    }
                    if state.black.get_index(index) {
                        state.black.clear_index(index);
                    } else {
                        state.black.set_index(index);
                    }
                }
                state.can_swap = false;
            }
        }
        state.previous = Some(previous);
        // The swap window closes for good the moment the piece count
        // leaves 1 -- see `State`'s `can_swap` doc comment.
        if state.occupied.count_ones() != 1 {
            state.can_swap = false;
        }
        state.turn = state.turn.next();
        state
    }

    fn generate_actions(state: &State, actions: &mut Vec<Action>) {
        if state.can_swap() {
            actions.push(Action::Swap);
        }
        let adjacency = pyramid::get_adjacency(state.occupied.n());
        let n = state.occupied.n() as u8;
        let n_usize = state.occupied.n();
        let (base_black, base_white) = visible_boards(&state.occupied, &state.black);
        let ground = ground_mask(n_usize, state.occupied.total_cells());
        let mover_black = state.turn == Player::Black;

        // Raster fast path: mutable, place/remove per candidate.
        let mut raster = raster::Raster::from_pyramid(n_usize, &state.occupied);
        let (mut own_color, opp_color) = build_color_masks(n_usize, state);

        for index in 0..state.occupied.total_cells() {
            if state.is_occupied(index) {
                continue;
            }
            let (col, row, level) = state.occupied.to_coord(index);
            if !state.occupied.can_place(col, row, level) {
                continue;
            }

            // Temporarily place on raster, check, then roll back.
            let raster_pos = raster.place(col, row, level);
            own_color[level].set_index(raster_pos);

            let group = raster.flood(col, row, level, &own_color);
            let libs = raster.count_liberties(&group);
            let fast_ok = if libs < 2 {
                false
            } else {
                let seeds = raster.enemies_touching(&group, &own_color);
                // Deduplicate: flood each enemy group only once.
                let mut enemy_seen: Vec<raster::LevelBoard> = (0..n_usize)
                    .map(|_| raster::LevelBoard::new(Dyn(n_usize), Dyn(n_usize)))
                    .collect();
                seeds.iter().all(|&(ec, er, el)| {
                    let ep = raster.raster_index(ec, er);
                    if enemy_seen[el].get_index(ep) {
                        return true;
                    }
                    let eg = raster.flood(ec, er, el, &opp_color);
                    for l2 in 0..n_usize {
                        enemy_seen[l2] |= eg[l2];
                    }
                    raster.count_liberties(&eg) > 0
                })
            };

            // Roll back.
            own_color[level].clear_index(raster_pos);
            raster.remove(col, row, level);

            // Raster says legal, but we must also check ko: if placing
            // here recreates the previous board state.
            let ko_safe = if let Some((ref prev_occ, _)) = state.previous {
                // Candidate was NOT in prev_occupied → can't recreate it.
                // If it WAS, last turn captured a stone here; re-placing it
                // might be ko.
                !prev_occ.get_index(index)
            } else {
                true // no previous state, ko impossible
            };

            if fast_ok && ko_safe {
                actions.push(Action::Place(index as u16, n));
                continue;
            }

            // Full 3D path for edge cases (suicide, capture, ko).
            let (own, opp) = if mover_black {
                let mut own = base_black;
                own.set_index(index);
                (own, base_white)
            } else {
                let mut own = base_white;
                own.set_index(index);
                (own, base_black)
            };
            if state
                .resolve_place_inner(index, adjacency, own, opp, ground)
                .is_some()
            {
                actions.push(Action::Place(index as u16, n));
            }
        }
    }

    fn player_to_move(state: &State) -> Player {
        state.turn
    }

    /// Piece-count race: the player with more pieces on the board wins once
    /// the mover has no legal placement left. Equal counts are a draw
    /// (`None`), matching the published rule ("draw if equal").
    fn winner(state: &State) -> Option<Player> {
        let black = state.piece_count(Player::Black);
        let white = state.piece_count(Player::White);
        match black.cmp(&white) {
            std::cmp::Ordering::Greater => Some(Player::Black),
            std::cmp::Ordering::Less => Some(Player::White),
            std::cmp::Ordering::Equal => None,
        }
    }

    fn notation(state: &Self::S, action: &Self::A) -> String {
        match *action {
            Action::Place(index, _n) => {
                let (col, row, level) = state.occupied.to_coord(index as usize);
                format!("({col},{row},L{level})")
            }
            Action::Swap => "swap".to_string(),
        }
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.zobrist_hash()
    }

    fn canonical_representation(state: Real<Self::S>) -> (Canonical<Self::S>, Transform) {
        let state = state.0;
        let sym_idx =
            canonical_symmetry(&state.occupied, &state.black, &state.zombie, state.previous);
        let sym = PyramidD4::new(state.occupied.n());
        let previous = state.previous.map(|(o, b)| {
            (
                transform_cells(&o, &sym, sym_idx),
                transform_cells(&b, &sym, sym_idx),
            )
        });
        (
            Canonical(State {
                occupied: transform_cells(&state.occupied, &sym, sym_idx),
                black: transform_cells(&state.black, &sym, sym_idx),
                zombie: transform_cells(&state.zombie, &sym, sym_idx),
                previous,
                turn: state.turn,
                can_swap: state.can_swap,
            }),
            Transform::new(sym_idx),
        )
    }

    /// `Action::Swap` is a fixed point of every symmetry (see the variant's
    /// own doc comment); only `Place`'s index transforms, through the
    /// `PyramidD4` group for the board size it carries.
    fn apply_to_action(action: Real<Self::A>, sym: Transform) -> Canonical<Self::A> {
        match action.0 {
            Action::Swap => Canonical(Action::Swap),
            Action::Place(index, n) => {
                let pyramid_sym = PyramidD4::new(n as usize);
                let image = pyramid_sym.index_symmetries(index as usize)[sym.index()];
                Canonical(Action::Place(image as u16, n))
            }
        }
    }

    fn invert_action(action: Canonical<Self::A>, sym: Transform) -> Real<Self::A> {
        match action.0 {
            Action::Swap => Real(Action::Swap),
            Action::Place(index, n) => {
                let pyramid_sym = PyramidD4::new(n as usize);
                let original = pyramid_sym.invert_symmetry(index as usize, sym.index());
                Real(Action::Place(original as u16, n))
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
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    fn can_place(state: &State, index: usize) -> bool {
        let (col, row, level) = state.occupied.to_coord(index);
        state.occupied.can_place(col, row, level)
    }

    #[test]
    fn random_play_smoke_test() {
        random_play::<Margo>();
    }

    #[test]
    fn base_level_needs_no_support() {
        let state = State::new(DEFAULT_N);
        assert!(can_place(&state, state.occupied.index(0, 0, 0)));
    }

    #[test]
    fn higher_level_needs_all_four_supporters() {
        let mut state = State::new(DEFAULT_N);
        let target = state.occupied.index(0, 0, 1);
        assert!(!can_place(&state, target));

        let adjacency = TouchingAdjacency::new(state.occupied.n());
        for &(col, row) in &[(0, 0), (1, 0), (0, 1)] {
            let idx = state.occupied.index(col, row, 0);
            let (occupied, black, zombie) = state.resolve_place(idx, &adjacency).unwrap();
            state.occupied = occupied;
            state.black = black;
            state.zombie = zombie;
            state.turn = state.turn.next();
        }
        assert!(
            !can_place(&state, target),
            "three of four supporters is not enough"
        );

        let idx = state.occupied.index(1, 1, 0);
        let (occupied, black, zombie) = state.resolve_place(idx, &adjacency).unwrap();
        state.occupied = occupied;
        state.black = black;
        state.zombie = zombie;
        assert!(can_place(&state, target));
    }

    #[test]
    fn occupied_cell_cannot_be_placed_into_again() {
        let mut state = State::new(DEFAULT_N);
        let adjacency = TouchingAdjacency::new(state.occupied.n());
        let idx = state.occupied.index(2, 2, 0);
        let (occupied, black, zombie) = state.resolve_place(idx, &adjacency).unwrap();
        state.occupied = occupied;
        state.black = black;
        state.zombie = zombie;
        assert!(!can_place(&state, idx));
    }

    #[test]
    fn rejects_board_size_outside_supported_range() {
        assert!(std::panic::catch_unwind(|| State::new(MIN_N - 1)).is_err());
        assert!(std::panic::catch_unwind(|| State::new(MAX_N + 1)).is_err());
    }

    #[test]
    fn every_recommended_board_size_supports_random_play() {
        for n in MIN_N..=MAX_N {
            let mut rng = SmallRng::seed_from_u64(n as u64);
            let mut state = State::new(n);
            let max_plies = state.occupied.total_cells() + 2;
            for _ in 0..max_plies {
                if Margo::is_terminal(&state) {
                    break;
                }
                let mut actions = Vec::new();
                Margo::generate_actions(&state, &mut actions);
                assert!(
                    !actions.is_empty(),
                    "n={n}: no legal moves on a non-terminal state"
                );
                let action = actions[rng.gen_range(0..actions.len())];
                state = Margo::apply(state, &action);
            }
        }
    }

    /// A single-member group with one liberty is captured when its last
    /// liberty is filled -- but here that liberty happens to be the group's
    /// own dependent, so the closing black stone rests directly on top of
    /// it, pinning it: the captured piece stays on the board as a zombie
    /// (original colour kept, but excluded from future connectivity) rather
    /// than being removed. Constructed via direct field construction, not
    /// simulated play, so the scenario is exact regardless of
    /// `generate_actions`'s own behaviour.
    ///
    /// Uses the base corner `(0, 0, 0)` rather than an interior cell: a
    /// touching-adjacency neighbor set includes not just the (up to four)
    /// same-level orthogonal cells but also the (up to four) level-`+1`
    /// cells resting on top -- an interior level-0 cell has as many as four
    /// of those extra vertical neighbors, but a corner cell has exactly one
    /// (`(0, 0, 1)`, its only possible dependent -- see
    /// `pyramid::dependent_positions`), and that one is only ever fillable
    /// once `(0, 0, 0)` itself is occupied (since it's one of `(0, 0, 1)`'s
    /// four supporters). So surrounding a corner needs exactly its two
    /// lateral neighbors filled, then the capturing move is the corner's own
    /// dependent -- newly placeable precisely because the target itself
    /// supports it, and unavoidably a pin: a captured cell's last liberty
    /// can only ever be closed by occupying its dependent (every other
    /// neighbor a corner cell has is a lateral, already accounted for), and
    /// closing that specific slot with the mover's own colour is exactly
    /// what "pinned" means. See [`apply_captures`]'s own tests for the
    /// plain-removal case (a captured cell whose dependent, if any, isn't
    /// occupied by the mover) -- reproducing that end-to-end through a real
    /// `State` turns out to need a fully captured multi-level group, and
    /// closing one always buries an existing piece it also depends on (see
    /// `resolve_captures`'s neighboring doc comment on this), reopening a
    /// liberty no further move can close.
    #[test]
    fn scripted_zombie_survives_pinned_capture() {
        let mut state = State::new(DEFAULT_N);
        let white_cell = state.occupied.index(0, 0, 0);
        state.occupied.set_index(white_cell);

        let black_cells = [(1, 0, 0), (0, 1, 0), (1, 1, 0)];
        for &(col, row, level) in &black_cells {
            let idx = state.occupied.index(col, row, level);
            state.occupied.set_index(idx);
            state.black.set_index(idx);
        }
        state.turn = Player::Black;

        let adjacency = TouchingAdjacency::new(state.occupied.n());
        let last = state.occupied.index(0, 0, 1);
        let (new_occupied, new_black, new_zombie) = state.resolve_place(last, &adjacency).unwrap();

        assert!(
            new_occupied.get_index(white_cell),
            "pinned white stone must survive as a zombie, not be removed"
        );
        assert!(
            !new_black.get_index(white_cell),
            "a zombie keeps its original colour"
        );
        assert!(
            new_zombie.get_index(white_cell),
            "captured-but-pinned stone must be marked zombie"
        );
        assert!(
            new_occupied.get_index(last),
            "black's placement must remain"
        );
        assert!(new_black.get_index(last));
    }

    /// [`apply_captures`]'s zombie/removal split, exercised directly against
    /// a hand-built capture mask rather than one [`resolve_captures`]
    /// derives from a real encirclement: two unrelated white cells, one with
    /// a black piece resting directly on top of it (pinned), one with no
    /// piece above it at all (not pinned). Landed this way -- rather than as
    /// a single `State`-reachable captured *group* with one pinned and one
    /// plain-removed member -- because every way of constructing that
    /// requires closing a captured cell's liberty by occupying its
    /// dependent, and doing that two levels above any base cell always
    /// buries an existing supporter that same cell also depends on for its
    /// own liberties (see `resolve_captures`'s doc comment), permanently
    /// reopening a liberty no further move can close. `apply_captures`
    /// itself has no such constraint -- it only inspects a precomputed
    /// capture mask.
    #[test]
    fn apply_captures_splits_pinned_and_removed_members() {
        let mut occupied = Cells::new(Dyn(DEFAULT_N));
        let mut black = Cells::new(Dyn(DEFAULT_N));

        let pinned = occupied.index(0, 0, 0);
        let pin = occupied.index(0, 0, 1);
        let removed = occupied.index(2, 2, 0);

        occupied.set_index(pinned);
        occupied.set_index(pin);
        occupied.set_index(removed);
        black.set_index(pin);

        let mut captured = go_board(occupied.total_cells());
        captured.set_index(pinned);
        captured.set_index(removed);

        let (new_occupied, new_black, new_zombie) =
            apply_captures(occupied, black, Cells::new(Dyn(DEFAULT_N)), captured, true);

        assert!(
            new_occupied.get_index(pinned),
            "pinned member stays on the board"
        );
        assert!(
            !new_black.get_index(pinned),
            "a zombie keeps its original colour"
        );
        assert!(new_zombie.get_index(pinned));

        assert!(
            !new_occupied.get_index(removed),
            "unpinned member is removed"
        );
        assert!(!new_zombie.get_index(removed));
    }

    /// A zombie that was already pinned before the current capture (from
    /// an earlier move) unconditionally survives every future capture it
    /// participates in -- it is permanently excluded from removal, even if
    /// its original pinning piece is no longer on the board.
    #[test]
    fn existing_zombie_survives_later_capture_of_its_group() {
        let mut occupied = Cells::new(Dyn(DEFAULT_N));
        let black = Cells::new(Dyn(DEFAULT_N));
        let mut zombie = Cells::new(Dyn(DEFAULT_N));

        // Two adjacent white pieces at level 0, one is already a zombie.
        let zombie_cell = occupied.index(0, 0, 0);
        let non_zombie = occupied.index(1, 0, 0);
        occupied.set_index(zombie_cell);
        occupied.set_index(non_zombie);
        zombie.set_index(zombie_cell);

        // Capture both together. The zombie must survive, the non-zombie
        // must be removed (no pinning piece above this time).
        let mut captured = go_board(occupied.total_cells());
        captured.set_index(zombie_cell);
        captured.set_index(non_zombie);

        let (new_occupied, _new_black, new_zombie) =
            apply_captures(occupied, black, zombie, captured, true);

        assert!(
            new_occupied.get_index(zombie_cell),
            "pre-existing zombie survives the capture"
        );
        assert!(new_zombie.get_index(zombie_cell));

        assert!(
            !new_occupied.get_index(non_zombie),
            "non-zombie group member is removed"
        );
        assert!(!new_zombie.get_index(non_zombie));
    }

    /// "Buried" is a visual aid (a piece hidden by another two levels
    /// above), not a connectivity rule -- a buried piece still physically
    /// touches its neighbors via the touching-adjacency graph, so it must
    /// appear in `visible_boards` alongside every other non-zombie piece.
    #[test]
    fn buried_piece_still_present_in_visible_boards() {
        let mut state = State::new(DEFAULT_N);
        // Build a full pyramid tip: base 2x2, one level-1, then occluder
        // at level 2 which buries the level-0 cell at (1, 1).
        let base = [(1, 1), (2, 1), (1, 2), (2, 2)];
        for &(col, row) in &base {
            state.occupied.set_index(state.occupied.index(col, row, 0));
        }
        for &(col, row) in &[(1, 1), (2, 1), (1, 2)] {
            state.occupied.set_index(state.occupied.index(col, row, 1));
        }
        state.occupied.set_index(state.occupied.index(0, 0, 2));

        let target = state.occupied.index(1, 1, 0);
        assert!(state.occupied.is_buried(1, 1, 0));

        let (_black_board, white_board) = visible_boards(&state.occupied, &state.black);
        // All occupied cells (no zombies, no black) are white. The buried
        // one must be present -- it still touches its neighbors.
        assert!(white_board.get_index(target));
    }

    /// A zombie piece still touches its same-colour neighbors, so it must
    /// participate in group connectivity. Only during capture does the
    /// zombie mask matter (zombies survive, non-zombies are removed).
    #[test]
    fn zombie_participates_in_visible_connectivity() {
        let mut state = State::new(DEFAULT_N);
        // Two adjacent white pieces. Mark one as a zombie.
        let a = state.occupied.index(0, 0, 0);
        let b = state.occupied.index(1, 0, 0);
        state.occupied.set_index(a);
        state.occupied.set_index(b);
        state.zombie.set_index(b);

        let (_black_board, white_board) = visible_boards(&state.occupied, &state.black);
        // Both cells must appear in white_board -- the zombie still touches
        // cell a and participates in the touching-adjacency group.
        assert!(white_board.get_index(a));
        assert!(white_board.get_index(b));
    }

    /// Placing into a spot with zero liberties and no capture is illegal.
    ///
    /// Uses the apex of a `MIN_N` (4x4) pyramid rather than a base cell: the
    /// apex's *only* touching neighbors are its four level-2 supporters (no
    /// same-level neighbors -- the apex level is a single cell -- and
    /// nothing rests above it), and by the time it's placeable at all those
    /// four must already be filled (support requirement), so "surround the
    /// apex" is just "fill its four supporters" with no lateral cells to
    /// separately account for. With those four supporters left with open
    /// liberties of their own below (level 1 is untouched here), capturing
    /// them isn't possible either, so placing white at the apex is plain
    /// suicide.
    /// From the game `margo-2026-08-24T23-46-45-862Z.game.json` n28→n29:
    /// a row of four black stones at level 0 (indices 29, 30, 31, 32),
    /// with the middle two (30, 31) pinned as zombies under a white stone
    /// at level 1 (index 69). White placing at the group's last empty
    /// ground-level liberty (index 39, (4,5)) must capture the entire
    /// group — zombies 30 and 31 survive, 29 and 32 are removed.
    #[test]
    fn zombie_group_captured_when_last_liberty_filled() {
        let mut state = State::new(7);
        state.turn = Player::White;

        // Build the n28 position (Black's turn — Black just placed at 32).
        let occupied: &[(usize, usize, usize)] = &[
            (1, 0, 0), //  1  black
            (2, 0, 0), //  2  black
            (3, 0, 0), //  3  black
            (4, 0, 0), //  4  black
            (1, 3, 0), // 22  white
            (2, 3, 0), // 23  white
            (3, 3, 0), // 24  white
            (4, 3, 0), // 25  white
            (0, 4, 0), // 28  white
            (1, 4, 0), // 29  black
            (2, 4, 0), // 30  black (zombie)
            (3, 4, 0), // 31  black (zombie)
            (4, 4, 0), // 32  black
            (5, 4, 0), // 33  white
            (0, 5, 0), // 35  white
            (1, 5, 0), // 36  white
            (2, 5, 0), // 37  white
            (3, 5, 0), // 38  white
            (5, 5, 0), // 40  white
            (1, 6, 0), // 43  white
            (4, 6, 0), // 46  white
            (2, 3, 1), // 69  white (level 1, pins 30 & 31)
        ];
        for &(col, row, level) in occupied {
            let idx = state.occupied.index(col, row, level);
            state.occupied.set_index(idx);
        }
        let black: &[(usize, usize, usize)] = &[
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 0, 0),
            (1, 4, 0),
            (2, 4, 0),
            (3, 4, 0),
            (4, 4, 0),
        ];
        for &(col, row, level) in black {
            let idx = state.occupied.index(col, row, level);
            state.black.set_index(idx);
        }
        state.zombie.set_index(state.occupied.index(2, 4, 0));
        state.zombie.set_index(state.occupied.index(3, 4, 0));

        // n28 was Black's turn. White should now be able to place at 39
        // (4, 5, 0) and capture the black group.
        let adjacency = TouchingAdjacency::new(7);
        let idx = state.occupied.index(4, 5, 0);
        let (new_occupied, new_black, new_zombie) = state
            .resolve_place(idx, &adjacency)
            .expect("placement at (4,5) must be legal");

        // The non-zombie black stones at 29 and 32 must be removed.
        assert!(
            !new_occupied.get_index(state.occupied.index(1, 4, 0)),
            "black (1,4)=29 must be captured"
        );
        assert!(
            !new_occupied.get_index(state.occupied.index(4, 4, 0)),
            "black (4,4)=32 must be captured"
        );

        // Zombies 30 and 31 must survive.
        assert!(
            new_occupied.get_index(state.occupied.index(2, 4, 0)),
            "zombie (2,4)=30 must survive"
        );
        assert!(
            new_zombie.get_index(state.occupied.index(2, 4, 0)),
            "(2,4)=30 must still be a zombie"
        );
        assert!(
            new_occupied.get_index(state.occupied.index(3, 4, 0)),
            "zombie (3,4)=31 must survive"
        );
        assert!(
            new_zombie.get_index(state.occupied.index(3, 4, 0)),
            "(3,4)=31 must still be a zombie"
        );

        // The white stone at (4,5) must be on the board.
        assert!(new_occupied.get_index(idx));
        assert!(!new_black.get_index(idx));
    }

    /// Black group B4-C4-D4-E4 (indices 22,23,24,25) on row 4 has exactly
    /// one ground-level liberty at E3 (4,2). When White plays E3, the group
    /// has no liberties and must be captured. Zombies C4 and D4 (pinned by
    /// the White stone at C4↑ level 1) survive; B4 and E4 are removed.
    #[test]
    fn capture_of_group_on_row_4() {
        let mut state = State::new(7);
        state.turn = Player::White;

        // Set up the board as described.
        // Row 2:  B2=W                             E2=W
        // Row 3:  A3=W  B3=W  C3=W  D3=W         F3=W
        // Row 4:  A4=W  B4=B  C4=Bz D4=Bz E4=B   F4=W
        // Row 5:        B5=W  C5=W  D5=W  E5=W
        let occupied: &[(usize, usize, usize)] = &[
            (1, 1, 0), // B2  white
            (4, 1, 0), // E2  white
            (0, 2, 0), // A3  white
            (1, 2, 0), // B3  white
            (2, 2, 0), // C3  white
            (3, 2, 0), // D3  white
            (5, 2, 0), // F3  white
            (0, 3, 0), // A4  white
            (1, 3, 0), // B4  black
            (2, 3, 0), // C4  black (zombie)
            (3, 3, 0), // D4  black (zombie)
            (4, 3, 0), // E4  black
            (5, 3, 0), // F4  white
            (1, 4, 0), // B5  white
            (2, 4, 0), // C5  white
            (3, 4, 0), // D5  white
            (4, 4, 0), // E5  white
            (2, 3, 1), // C4↑ level 1 white (pins C4 and D4)
        ];
        for &(col, row, level) in occupied {
            let idx = state.occupied.index(col, row, level);
            state.occupied.set_index(idx);
        }
        for &(col, row) in &[(1, 3), (2, 3), (3, 3), (4, 3)] {
            state.black.set_index(state.occupied.index(col, row, 0));
        }
        state.zombie.set_index(state.occupied.index(2, 3, 0));
        state.zombie.set_index(state.occupied.index(3, 3, 0));

        let adjacency = TouchingAdjacency::new(7);
        let e3 = state.occupied.index(4, 2, 0);
        let (new_occupied, new_black, new_zombie) = state
            .resolve_place(e3, &adjacency)
            .expect("E3 must be a legal placement");

        // B4 and E4 must be removed.
        assert!(
            !new_occupied.get_index(state.occupied.index(1, 3, 0)),
            "B4 must be captured"
        );
        assert!(
            !new_occupied.get_index(state.occupied.index(4, 3, 0)),
            "E4 must be captured"
        );

        // C4 and D4 survive as zombies.
        assert!(
            new_occupied.get_index(state.occupied.index(2, 3, 0)),
            "C4 must survive"
        );
        assert!(
            new_zombie.get_index(state.occupied.index(2, 3, 0)),
            "C4 must remain zombie"
        );
        assert!(
            new_occupied.get_index(state.occupied.index(3, 3, 0)),
            "D4 must survive"
        );
        assert!(
            new_zombie.get_index(state.occupied.index(3, 3, 0)),
            "D4 must remain zombie"
        );

        // E3 is on the board.
        assert!(new_occupied.get_index(e3));
        assert!(!new_black.get_index(e3));
    }

    /// Ko rejects a placement whose resulting `(occupied, black)` pair
    /// exactly matches `previous`, and allows the same placement once
    /// `previous` no longer matches. Reuses
    /// `scripted_zombie_survives_pinned_capture`'s already-verified capture
    /// (a real placement computed by [`State::resolve_place`], not a hand-
    /// predicted one) as the candidate move, rather than hand-deriving a
    /// full two-move capture/recapture pair through this crate's touching
    /// graph: every non-apex single-stone capture pins its target (see that
    /// test's structural-finding note), so the piece a recapture would need
    /// to remove again never becomes fully vacant the way a plain Go ko
    /// does -- the mechanism under test here is purely `resolve_place`'s
    /// comparison against `self.previous`, which doesn't care whether
    /// `previous` came from two real plies back or was set directly.
    #[test]
    fn ko_rejects_immediate_repeat_of_previous_position() {
        let mut state = State::new(DEFAULT_N);
        let white_cell = state.occupied.index(0, 0, 0);
        state.occupied.set_index(white_cell);

        let black_cells = [(1, 0, 0), (0, 1, 0), (1, 1, 0)];
        for &(col, row, level) in &black_cells {
            let idx = state.occupied.index(col, row, level);
            state.occupied.set_index(idx);
            state.black.set_index(idx);
        }
        state.turn = Player::Black;

        let adjacency = TouchingAdjacency::new(state.occupied.n());
        let last = state.occupied.index(0, 0, 1);
        let (new_occupied, new_black, _) = state.resolve_place(last, &adjacency).unwrap();

        let mut ko_state = state.clone();
        ko_state.previous = Some((new_occupied, new_black));
        assert!(
            ko_state.resolve_place(last, &adjacency).is_none(),
            "recreating the stored previous position must be rejected"
        );

        let mut free_state = state.clone();
        free_state.previous = Some((State::new(DEFAULT_N).occupied, State::new(DEFAULT_N).black));
        assert!(
            free_state.resolve_place(last, &adjacency).is_some(),
            "the same placement is legal once previous no longer matches"
        );
    }

    /// [`Game::apply`] threads `previous` forward as the pre-move
    /// `(occupied, black)` pair, so a later ko check compares against the
    /// position as it stood immediately before the just-applied move.
    #[test]
    fn apply_records_pre_move_position_as_previous() {
        let state = State::new(DEFAULT_N);
        let pre_move = (state.occupied, state.black);
        let mut actions = Vec::new();
        Margo::generate_actions(&state, &mut actions);
        let action = actions[0];
        let next = Margo::apply(state, &action);
        assert_eq!(next.previous, Some(pre_move));
    }

    /// Black fills every cell below the apex -- levels 0 through 2 -- except
    /// one level-0 hole, so the whole black mass is one group with a real
    /// board-level liberty (the hole) and stays alive. White's apex
    /// placement only touches that black mass, gains no liberty of its own,
    /// and can't capture it (the hole keeps black's liberty open), so it
    /// must be rejected as suicide.
    #[test]
    fn suicide_placement_rejected() {
        let mut state = State::new(MIN_N);
        let hole = state.occupied.index(3, 3, 0);
        for level in 0..3 {
            let side = state.occupied.level_side(level);
            for row in 0..side {
                for col in 0..side {
                    let idx = state.occupied.index(col, row, level);
                    if idx == hole {
                        continue;
                    }
                    state.occupied.set_index(idx);
                    state.black.set_index(idx);
                }
            }
        }
        state.turn = Player::White;

        let adjacency = TouchingAdjacency::new(state.occupied.n());
        let apex = state.occupied.index(0, 0, 3);
        assert!(state.resolve_place(apex, &adjacency).is_none());
    }

    /// A placement that itself captures an enemy group is legal even though
    /// the placed stone would otherwise have zero liberties, since the
    /// capture opens up a liberty for it.
    ///
    /// Exercises [`resolve_captures`] directly rather than through
    /// `State`/`Cells`: the apex is the *only* cell whose full neighbor set
    /// can be closed off at all (every other cell has a dependent above it
    /// that, per the support rule, can't possibly be occupied yet, since it
    /// needs the candidate cell itself as one of its own four supporters --
    /// so every non-apex placement always keeps a guaranteed-empty neighbor
    /// and can never be true suicide), but even the apex's four supporters
    /// are so densely cross-connected through the touching graph that
    /// closing every one of their liberties requires occupying essentially
    /// the entire rest of the board (verified empirically for `MIN_N`'s
    /// 30-cell board, not derived by hand) -- and on a real `Cells` board
    /// that dense a fill buries several cells retroactively (a placement
    /// can occlude an existing piece two levels below it), reopening
    /// liberties `resolve_captures` alone doesn't know about. Working
    /// directly against a synthetic `GoBoard` pair sidesteps that
    /// unrelated interaction and isolates the thing this test is actually
    /// about: the "safe or captures" legality check itself, over the real
    /// `TouchingAdjacency` table.
    #[test]
    fn placement_that_captures_to_gain_a_liberty_is_accepted() {
        let n = MIN_N;
        let cells = pyramid::total_cells(n);
        let apex = pyramid::index(n, 0, 0, n - 1);

        let mut own = go_board(cells);
        own.set_index(apex);
        let mut opp = go_board(cells);
        for i in 0..cells {
            if i != apex {
                opp.set_index(i);
            }
        }

        let adjacency = TouchingAdjacency::new(n);
        let ground = ground_mask(n, cells);
        let captured = resolve_captures(own, opp, apex, &adjacency, ground)
            .expect("capturing placement must be legal");
        for i in 0..cells {
            if i != apex {
                assert!(
                    captured.get_index(i),
                    "every non-apex cell must be captured"
                );
            }
        }
        assert!(!captured.get_index(apex));
    }

    /// `Action::Swap` is offered exactly once per game: never on White's
    /// opening move (the board is empty, so `can_swap()`'s `occupied ==
    /// 1` guard fails), always as one of Black's options on the very next
    /// move, and never again afterward -- whether Black takes it or plays
    /// a normal placement instead. A property test over many random board
    /// sizes/seeds rather than a single hand-picked trajectory: the claim
    /// is about the swap bookkeeping alone, not about any particular
    /// position, so there's nothing to gain from tracing one by hand.
    #[test]
    fn swap_offered_exactly_once_as_black_first_reply() {
        for n in MIN_N..=MAX_N {
            for seed in 0..8u64 {
                let mut rng = SmallRng::seed_from_u64(n as u64 * 100 + seed);
                let mut state = State::new(n);

                let mut actions = Vec::new();
                Margo::generate_actions(&state, &mut actions);
                assert!(
                    actions.iter().all(|a| *a != Action::Swap),
                    "n={n} seed={seed}: swap must not be offered on White's opening move"
                );

                let max_plies = state.occupied.total_cells() + 2;
                let mut swap_offers_seen = 0;
                for ply in 0..max_plies {
                    if Margo::is_terminal(&state) {
                        break;
                    }
                    actions.clear();
                    Margo::generate_actions(&state, &mut actions);
                    assert!(!actions.is_empty(), "n={n} seed={seed}: no legal moves");
                    let swap_offered = actions.contains(&Action::Swap);
                    if swap_offered {
                        swap_offers_seen += 1;
                        assert_eq!(
                            ply, 1,
                            "n={n} seed={seed}: swap must be offered only as Black's \
                             first reply (ply 1), not ply {ply}"
                        );
                    }
                    let action = actions[rng.gen_range(0..actions.len())];
                    state = Margo::apply(state, &action);
                }
                assert!(
                    swap_offers_seen <= 1,
                    "n={n} seed={seed}: swap must never be offered more than once"
                );
            }
        }
    }

    /// Applying `Action::Swap` recolours the single piece on the board
    /// (White's opening placement becomes Black's), hands the move back to
    /// White, closes the swap window for good, and records `previous` as
    /// the pre-swap position -- same bookkeeping any other action gets.
    #[test]
    fn swap_recolours_the_single_piece_and_closes_the_window() {
        let state = State::new(DEFAULT_N);
        let mut actions = Vec::new();
        Margo::generate_actions(&state, &mut actions);
        let opening = actions[0];
        let Action::Place(opening_index, _n) = opening else {
            panic!("White's opening move must be a placement");
        };
        let after_white = Margo::apply(state, &opening);
        assert_eq!(after_white.turn, Player::Black);
        assert!(after_white.can_swap());

        let pre_swap = (after_white.occupied, after_white.black);
        let after_swap = Margo::apply(after_white, &Action::Swap);

        assert!(
            after_swap.is_black(opening_index as usize),
            "the opening piece must now belong to Black"
        );
        assert_eq!(
            after_swap.occupied.count_ones(),
            1,
            "swap changes colour, not piece count"
        );
        assert_eq!(after_swap.turn, Player::White);
        assert!(!after_swap.can_swap(), "the swap window closes for good");
        assert_eq!(after_swap.previous, Some(pre_swap));
    }

    /// A normal placement as Black's first reply (declining the swap)
    /// closes the swap window just as permanently as taking it does.
    #[test]
    fn declining_swap_also_closes_the_window() {
        let state = State::new(DEFAULT_N);
        let mut actions = Vec::new();
        Margo::generate_actions(&state, &mut actions);
        let opening = actions[0];
        let after_white = Margo::apply(state, &opening);

        actions.clear();
        Margo::generate_actions(&after_white, &mut actions);
        let placement = actions
            .iter()
            .find(|a| **a != Action::Swap)
            .copied()
            .expect("a normal placement must be available alongside swap");
        let after_black = Margo::apply(after_white, &placement);
        assert!(!after_black.can_swap());

        actions.clear();
        Margo::generate_actions(&after_black, &mut actions);
        assert!(actions.iter().all(|a| *a != Action::Swap));
    }

    /////////////////////////////////////////////////////////////////////////
    // Symmetry: `apply_to_action`/`invert_action` round-trip,
    // `canonical_representation` invariance across every symmetric image of
    // a state, `invert_action` always producing a legal real action along
    // random play, and hash consistency -- mirroring `games/gonnect`'s/
    // `games/atarigo`'s own symmetry test suites.

    /// Every geometric field (`occupied`/`black`/`zombie`/`previous`)
    /// transformed through each of the 8 `PyramidD4` elements -- the full
    /// set of states that must all canonicalize identically, since they're
    /// the same position viewed from 8 different orientations. `turn`/
    /// `can_swap` aren't geometric, so they ride along unchanged.
    fn state_symmetries(state: &State) -> [State; 8] {
        let sym = PyramidD4::new(state.occupied.n());
        std::array::from_fn(|i| State {
            occupied: transform_cells(&state.occupied, &sym, i),
            black: transform_cells(&state.black, &sym, i),
            zombie: transform_cells(&state.zombie, &sym, i),
            previous: state
                .previous
                .map(|(o, b)| (transform_cells(&o, &sym, i), transform_cells(&b, &sym, i))),
            turn: state.turn,
            can_swap: state.can_swap,
        })
    }

    /// A canonicalized state's fields, used as the "same equivalence class"
    /// comparison key in the invariance/hash-consistency checks below.
    fn canon_key(state: &State) -> (Cells, Cells, Cells, Option<(Cells, Cells)>, Player, bool) {
        (
            state.occupied,
            state.black,
            state.zombie,
            state.previous,
            state.turn,
            state.can_swap,
        )
    }

    #[test]
    fn action_transform_round_trip() {
        let n = DEFAULT_N as u8;
        let total = pyramid::total_cells(DEFAULT_N);
        for index in 0..total {
            for sym in 0..8usize {
                let action = Action::Place(index as u16, n);
                let sym = Transform::new(sym);
                let transformed = Margo::apply_to_action(Real(action), sym);
                let back = Margo::invert_action(transformed, sym);
                assert_eq!(back.into_inner(), action);
            }
        }
        for sym in 0..8usize {
            let sym = Transform::new(sym);
            assert_eq!(
                Margo::apply_to_action(Real(Action::Swap), sym).into_inner(),
                Action::Swap
            );
            assert_eq!(
                Margo::invert_action(Canonical(Action::Swap), sym).into_inner(),
                Action::Swap
            );
        }
    }

    /// Plays a short random game (captures/zombies/ko/swap all in scope,
    /// not just placements), and at every reached state, checks that
    /// `canonical_representation` agrees across all 8 of that state's own
    /// symmetric images.
    fn check_canonical_representation_invariant(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let mut reachable = vec![state.clone()];
        let max_plies = state.occupied.total_cells() / 2;
        for _ in 0..max_plies {
            if Margo::is_terminal(&state) {
                break;
            }
            let mut actions = Vec::new();
            Margo::generate_actions(&state, &mut actions);
            let action = actions[rng.gen_range(0..actions.len())];
            state = Margo::apply(state, &action);
            reachable.push(state.clone());
        }

        for state in reachable {
            let (canon, _sym) = Margo::canonical_representation(Real(state.clone()));
            let canon_key_value = canon_key(&canon.into_inner());

            for variant in state_symmetries(&state) {
                let (canon2, _) = Margo::canonical_representation(Real(variant));
                assert_eq!(
                    canon_key(&canon2.into_inner()),
                    canon_key_value,
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
            if Margo::is_terminal(&state) {
                return;
            }
            let mut real_actions = Vec::new();
            Margo::generate_actions(&state, &mut real_actions);

            let (canon, sym) = Margo::canonical_representation(Real(state.clone()));
            let canon = canon.into_inner();
            let mut canon_actions = Vec::new();
            Margo::generate_actions(&canon, &mut canon_actions);

            for &canon_action in &canon_actions {
                let translated = Margo::invert_action(Canonical(canon_action), sym).into_inner();
                assert!(
                    real_actions.contains(&translated),
                    "seed {seed}, n={n}: invert_action produced {translated:?} (from \
                     canonical {canon_action:?}, sym {sym:?}), not present in real \
                     generate_actions {real_actions:?}"
                );
            }

            let action = real_actions[rng.gen_range(0..real_actions.len())];
            state = Margo::apply(state, &action);
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
    /// Random sampling rather than an exhaustive BFS of the reachable-state
    /// graph -- Margo's `previous`/ko field keeps positions from deduping
    /// the way Go-style capture-heavy games often do, so (per this repo's
    /// `AGENTS.md` note on `cargo test --lib` memory safety) an exhaustive
    /// walk even from `MIN_N` isn't bounded the way Gonnect's 3x3 exhaustive
    /// check is.
    #[test]
    fn random_games_hash_consistency() {
        use std::collections::HashMap;

        let mut rng = SmallRng::seed_from_u64(9);
        let mut by_hash: HashMap<u64, _> = HashMap::new();
        let mut mismatches = 0;
        for _game in 0..200 {
            let mut state = State::new(MIN_N);
            for _ in 0..40 {
                if Margo::is_terminal(&state) {
                    break;
                }
                let mut actions = Vec::new();
                Margo::generate_actions(&state, &mut actions);
                let action = actions[rng.gen_range(0..actions.len())];
                state = Margo::apply(state, &action);

                let h = state.zobrist_hash();
                let (canon, _sym) = Margo::canonical_representation(Real(state.clone()));
                let key = canon_key(&canon.into_inner());
                match by_hash.get(&h) {
                    Some(prev) if *prev != key => {
                        mismatches += 1;
                        println!("MISMATCH at hash {h}");
                    }
                    Some(_) => {}
                    None => {
                        by_hash.insert(h, key);
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "hash collided across different equivalence classes"
        );
    }

    /// [`Margo::winner`] is a piece-count race, not connectivity/territory --
    /// built via direct field construction so the counts are exact
    /// regardless of `generate_actions`'s behaviour, rather than relying on
    /// a specific game reaching this ratio through legal play.
    #[test]
    fn winner_is_whoever_has_more_pieces() {
        let mut state = State::new(MIN_N);
        for &(col, row) in &[(0, 0), (1, 0), (2, 0)] {
            state.occupied.set_index(state.occupied.index(col, row, 0));
            state.black.set_index(state.occupied.index(col, row, 0));
        }
        for &(col, row) in &[(3, 0), (0, 1)] {
            state.occupied.set_index(state.occupied.index(col, row, 0));
        }
        assert_eq!(Margo::winner(&state), Some(Player::Black));

        let mut tied = State::new(MIN_N);
        for &(col, row) in &[(0, 0), (1, 0)] {
            tied.occupied.set_index(tied.occupied.index(col, row, 0));
            tied.black.set_index(tied.occupied.index(col, row, 0));
        }
        for &(col, row) in &[(3, 0), (0, 1)] {
            tied.occupied.set_index(tied.occupied.index(col, row, 0));
        }
        assert_eq!(Margo::winner(&tied), None);
    }
}
