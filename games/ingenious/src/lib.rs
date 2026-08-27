//! Ingenious (Einfach Genial), Reiner Knizia, 2004.
//!
//! Supports 2- and 3-player games via `Ingenious<const P: usize>` (aliased as
//! [`Ingenious2`]/[`Ingenious3`]). `State` stores every player's rack in
//! full, but only the mover's own rack is real information to a search over
//! this game: `Ingenious::determinize` resamples every other rack from the
//! pool of tiles the mover can't see, and `Ingenious::has_hidden_information`
//! reports this so search strategies can use that. Tile draws are a
//! state-embedded PRNG stream rather than true hidden chance nodes.
//!
//! The board is a hexagon-of-hexagons embedded in an `SIDE x SIDE` square
//! grid, using an offset coordinate system where the six hex-adjacency
//! directions are (in `(row, col)` deltas): N=(+1,0), S=(-1,0), E=(0,+1),
//! W=(0,-1), NE=(+1,+1), SW=(-1,-1). A cell's hex distance from the board
//! center is `max(|dc|, |dr|, |dr - dc|)` for `dr = row - CENTER`,
//! `dc = col - CENTER`. Every player count shares the same underlying grid
//! and center cell; a `P`-player board is playable out to `playable_radius(P)`
//! hexes from center, so a smaller player count simply leaves the grid's
//! outer ring(s) unused -- the same way the physical board reserves its
//! outer rings for higher player counts.

use game_core::display::{RectangularBoard, RectangularBoardDisplay};
use mcts::game::{Game, PlayerIndex};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;

pub const NUM_COLORS: usize = 6;
pub const RACK_SIZE: usize = 6;
pub const TARGET_SCORE: u8 = 18;

// Board storage is sized for the largest player count this crate currently
// supports (3). Bump `BOARD_RADIUS` (and `playable_radius`'s mapping) to add
// a 4-player board.
const BOARD_RADIUS: usize = 6;
pub const SIDE: usize = 2 * BOARD_RADIUS + 1;
pub const NUM_CELLS: usize = SIDE * SIDE;

const CENTER: i32 = BOARD_RADIUS as i32;
const DEFAULT_SEED: u64 = 0x1CE0_1DEA_C0FF_EE42;

/// How far from center a `players`-player board's playable disc extends --
/// each added player unlocks one more ring, matching the real board's
/// radius-per-player-count table.
pub const fn playable_radius(players: usize) -> usize {
    players + 3
}

// Six hex-adjacency directions as (row, col) deltas. Opposite directions are
// adjacent pairs: (N, S) = (0, 1), (E, W) = (2, 3), (NE, SW) = (4, 5) -- so
// `dir ^ 1` gives the opposite direction.
const DELTAS: [(i32, i32); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)];
const DIR_N: usize = 0;
const DIR_E: usize = 2;
const DIR_NE: usize = 4;
// The three directions that, taken from the lower-indexed endpoint of any
// hex-adjacent pair, enumerate each undirected board edge exactly once
// (their opposites S/W/SW are reached from the *other* endpoint instead).
const CANONICAL_DIRS: [usize; 3] = [DIR_N, DIR_E, DIR_NE];

const NO_NEIGHBOR: u8 = u8::MAX;

#[inline]
const fn opposite(dir: usize) -> usize {
    dir ^ 1
}

#[inline]
fn hex_distance(row: i32, col: i32) -> i32 {
    let dr = row - CENTER;
    let dc = col - CENTER;
    dc.abs().max(dr.abs()).max((dr - dc).abs())
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Color {
    Red,
    Green,
    Blue,
    Orange,
    Yellow,
    Purple,
}

impl Color {
    pub const ALL: [Color; NUM_COLORS] = [
        Color::Red,
        Color::Green,
        Color::Blue,
        Color::Orange,
        Color::Yellow,
        Color::Purple,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    fn char(self) -> char {
        match self {
            Color::Red => 'R',
            Color::Green => 'G',
            Color::Blue => 'B',
            Color::Orange => 'O',
            Color::Yellow => 'Y',
            Color::Purple => 'P',
        }
    }
}

/// Total physical count of a tile type: 6 copies of each of the 15
/// different-color pairs, 5 copies of each of the 6 same-color doubles.
#[inline]
fn tile_type_total(a: Color, b: Color) -> u8 {
    if a == b {
        5
    } else {
        6
    }
}

/// Canonical (a.index() <= b.index()) ordering for an unordered tile-color
/// pair, so equal tile types always compare/hash/sort identically regardless
/// of which color was named first.
#[inline]
fn normalize(a: Color, b: Color) -> (Color, Color) {
    if a.index() <= b.index() {
        (a, b)
    } else {
        (b, a)
    }
}

////////////////////////////////////////////////////////////////////////////////////////

/// Precomputed board geometry -- neighbor table, valid-cell mask, and the six
/// pre-printed starting symbol cells -- for one player count `P`. Computed
/// once per `P` and shared process-wide; `NUM_CELLS` is tiny (169) so this
/// costs nothing to keep around.
struct Geometry {
    valid: Vec<u8>,
    valid_mask: [bool; NUM_CELLS],
    neighbors: [[u8; 6]; NUM_CELLS],
    /// `symbol_cell[color.index()]` is the board cell bearing that color's
    /// pre-printed starting symbol.
    symbol_cell: [u8; NUM_COLORS],
    /// Reverse lookup: `symbol_of[cell]` is `Some(color_index)` if `cell` is
    /// a pre-printed symbol cell.
    symbol_of: [Option<u8>; NUM_CELLS],
}

fn geometry<const P: usize>() -> &'static Geometry {
    // Board storage (`NUM_CELLS`) below is only sized for these two arities.
    const { assert!(P == 2 || P == 3, "Ingenious only supports 2 or 3 players") };

    // A `static` declared inside a generic function is a single shared item,
    // not one instance per monomorphization -- so the cache is keyed
    // explicitly by player count instead of relying on `P` to separate it.
    static CACHE: [OnceLock<Geometry>; 4] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    CACHE[P].get_or_init(|| {
        let radius = playable_radius(P) as i32;
        let mut valid = Vec::new();
        let mut valid_mask = [false; NUM_CELLS];
        for row in 0..SIDE {
            for col in 0..SIDE {
                if hex_distance(row as i32, col as i32) <= radius {
                    let idx = row * SIDE + col;
                    valid_mask[idx] = true;
                    valid.push(idx as u8);
                }
            }
        }

        let mut neighbors = [[NO_NEIGHBOR; 6]; NUM_CELLS];
        for row in 0..SIDE {
            for col in 0..SIDE {
                let idx = row * SIDE + col;
                if !valid_mask[idx] {
                    continue;
                }
                for (dir, &(dr, dc)) in DELTAS.iter().enumerate() {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr < 0 || nc < 0 || nr >= SIDE as i32 || nc >= SIDE as i32 {
                        continue;
                    }
                    let nidx = (nr as usize) * SIDE + nc as usize;
                    if valid_mask[nidx] {
                        neighbors[idx][dir] = nidx as u8;
                    }
                }
            }
        }

        // The six starting symbols sit at radius 3 from center, one per hex
        // direction -- a 6-fold-symmetric placement, the same for every
        // player count. The real board's exact printed layout isn't
        // available in any text-searchable source; this is a symmetric
        // stand-in that preserves the rule it exists to support (each
        // opening move is equally good, up to rotation) without claiming to
        // reproduce the physical artwork.
        let mut symbol_cell = [0u8; NUM_COLORS];
        let mut symbol_of = [None; NUM_CELLS];
        for (i, &(dr, dc)) in DELTAS.iter().enumerate() {
            let row = (CENTER + 3 * dr) as usize;
            let col = (CENTER + 3 * dc) as usize;
            let idx = row * SIDE + col;
            symbol_cell[i] = idx as u8;
            symbol_of[idx] = Some(i as u8);
        }

        Geometry {
            valid,
            valid_mask,
            neighbors,
            symbol_cell,
            symbol_of,
        }
    })
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Player(pub u8);

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        self.0 as usize
    }
}

impl Player {
    fn from_index(index: usize) -> Self {
        Player(index as u8)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// The player to move must place a tile (their turn's mandatory first
    /// placement, or a bonus placement owed from a color that just reached
    /// `TARGET_SCORE`).
    Place,
    /// All placements (mandatory + any bonus chain) are resolved; the player
    /// to move chooses whether to swap their whole rack before refilling.
    SwapDecision,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub struct PlaceMove {
    pub cell: u8,
    pub dir: u8,
    pub color_a: Color,
    pub color_b: Color,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum Action {
    Place(PlaceMove),
    /// Decline the optional rack swap; refill up to `RACK_SIZE` and end the
    /// turn.
    KeepRack,
    /// Discard the whole rack back into the bag, draw a fresh `RACK_SIZE`,
    /// and end the turn. Only legal when `State::swap_eligible`.
    Swap,
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct State<const P: usize> {
    pub board: [Option<Color>; NUM_CELLS],
    /// `board_tile_counts[i][j]` (i <= j only) is how many physical tiles of
    /// that color-pair type have been placed on the board so far.
    pub board_tile_counts: [[u8; NUM_COLORS]; NUM_COLORS],
    pub racks: [[Option<(Color, Color)>; RACK_SIZE]; P],
    pub score: [[u8; NUM_COLORS]; P],
    /// Once true, this color is frozen at `TARGET_SCORE` for this player:
    /// the bonus play for it has already been granted and used, and it can
    /// never score (or grant another bonus) again.
    pub bonus_used: [[bool; NUM_COLORS]; P],
    pub has_moved: [bool; P],
    pub claimed_symbols: [bool; NUM_COLORS],
    pub current_player: usize,
    pub phase: Phase,
    /// Extra placements still owed to `current_player` from colors that hit
    /// `TARGET_SCORE` earlier in this same turn (may chain: a bonus
    /// placement can itself trigger more bonuses).
    pub pending_bonus: u8,
    pub winner_immediate: Option<usize>,
    /// Deterministic PRNG stream driving tile draws (splitmix64). Re-rolled
    /// by `Game::determinize` before each playout so simulations sample
    /// different future draws even from an identical tree node.
    pub rng: u64,
}

impl<const P: usize> Default for State<P> {
    fn default() -> Self {
        Self::new(DEFAULT_SEED)
    }
}

impl<const P: usize> State<P> {
    pub fn new(seed: u64) -> Self {
        let mut state = State {
            board: [None; NUM_CELLS],
            board_tile_counts: [[0; NUM_COLORS]; NUM_COLORS],
            racks: [[None; RACK_SIZE]; P],
            score: [[0; NUM_COLORS]; P],
            bonus_used: [[false; NUM_COLORS]; P],
            has_moved: [false; P],
            claimed_symbols: [false; NUM_COLORS],
            current_player: 0,
            phase: Phase::Place,
            pending_bonus: 0,
            winner_immediate: None,
            rng: seed,
        };

        let g = geometry::<P>();
        for (color_idx, &cell) in g.symbol_cell.iter().enumerate() {
            state.board[cell as usize] = Some(Color::ALL[color_idx]);
        }

        for p in 0..P {
            for slot in 0..RACK_SIZE {
                state.racks[p][slot] = state.draw_one();
            }
            sort_rack(&mut state.racks[p]);
        }

        state
    }

    #[inline]
    fn next_rng(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// How many physical tiles of type (a, b) are currently accounted for
    /// outside the bag -- on the board, or in any player's rack.
    fn in_play_count(&self, a: Color, b: Color) -> u8 {
        let (a, b) = normalize(a, b);
        let mut n = self.board_tile_counts[a.index()][b.index()];
        for rack in &self.racks {
            for slot in rack.iter().flatten() {
                if *slot == (a, b) {
                    n += 1;
                }
            }
        }
        n
    }

    /// Draws one tile uniformly at random from whatever remains in the bag
    /// (the fixed 120-tile distribution minus everything currently on the
    /// board or in a rack), or `None` if the bag is empty.
    fn draw_one(&mut self) -> Option<(Color, Color)> {
        let mut candidates: [(Color, Color, u32); 21] = [(Color::Red, Color::Red, 0); 21];
        let mut n = 0;
        let mut total = 0u32;
        for i in 0..NUM_COLORS {
            for j in i..NUM_COLORS {
                let a = Color::ALL[i];
                let b = Color::ALL[j];
                let remaining =
                    tile_type_total(a, b).saturating_sub(self.in_play_count(a, b)) as u32;
                if remaining > 0 {
                    candidates[n] = (a, b, remaining);
                    n += 1;
                    total += remaining;
                }
            }
        }
        if total == 0 {
            return None;
        }
        let mut r = self.next_rng() % total as u64;
        for &(a, b, w) in &candidates[..n] {
            if r < w as u64 {
                return Some((a, b));
            }
            r -= w as u64;
        }
        unreachable!("weighted draw covered fewer than `total` candidates")
    }

    fn refill_rack(&mut self, player: usize) {
        for slot in 0..RACK_SIZE {
            if self.racks[player][slot].is_none() {
                self.racks[player][slot] = self.draw_one();
            }
        }
        sort_rack(&mut self.racks[player]);
    }

    fn rack_has_any_tile(&self, player: usize) -> bool {
        self.racks[player].iter().any(Option::is_some)
    }

    #[inline]
    fn occupied(&self, cell: u8) -> bool {
        self.board[cell as usize].is_some()
    }

    /// The color index of an unclaimed pre-printed symbol adjacent to
    /// `cell`, if any -- used to legalize/resolve a player's first move.
    fn adjacent_unclaimed_symbol(&self, cell: u8) -> Option<usize> {
        let g = geometry::<P>();
        for dir in 0..6 {
            let nb = g.neighbors[cell as usize][dir];
            if nb == NO_NEIGHBOR {
                continue;
            }
            if let Some(color_idx) = g.symbol_of[nb as usize] {
                if !self.claimed_symbols[color_idx as usize] {
                    return Some(color_idx as usize);
                }
            }
        }
        None
    }

    /// Sum of same-colored-run points scored from `cell` (which was just
    /// given `color`) looking outward along every direction except
    /// `exclude_dir` (the direction toward this tile's other half).
    fn run_score(&self, cell: u8, exclude_dir: usize, color: Color) -> u32 {
        let g = geometry::<P>();
        let mut pts = 0u32;
        for dir in 0..6 {
            if dir == exclude_dir {
                continue;
            }
            let mut at = cell;
            loop {
                let nb = g.neighbors[at as usize][dir];
                if nb == NO_NEIGHBOR {
                    break;
                }
                match self.board[nb as usize] {
                    Some(c) if c == color => {
                        pts += 1;
                        at = nb;
                    }
                    _ => break,
                }
            }
        }
        pts
    }

    /// Applies newly-scored points to `player`'s per-color totals, capping
    /// each at `TARGET_SCORE` and freezing (`bonus_used`) any color that
    /// just crossed it. Returns how many new bonus plays were granted.
    fn apply_scoring(&mut self, player: usize, gained: [u32; NUM_COLORS]) -> u8 {
        let mut bonus = 0;
        for (c, &g) in gained.iter().enumerate() {
            if g == 0 || self.bonus_used[player][c] {
                continue;
            }
            let old = self.score[player][c];
            let new = (old as u32 + g).min(TARGET_SCORE as u32) as u8;
            self.score[player][c] = new;
            if old < TARGET_SCORE && new >= TARGET_SCORE {
                self.bonus_used[player][c] = true;
                bonus += 1;
            }
        }
        bonus
    }

    fn board_has_legal_placement(&self) -> bool {
        let g = geometry::<P>();
        for &cell in &g.valid {
            if self.occupied(cell) {
                continue;
            }
            for &dir in &CANONICAL_DIRS {
                let nb = g.neighbors[cell as usize][dir];
                if nb != NO_NEIGHBOR && !self.occupied(nb) {
                    return true;
                }
            }
        }
        false
    }

    fn generate_place_actions(&self, actions: &mut Vec<Action>) {
        let g = geometry::<P>();
        let player = self.current_player;

        let mut types: Vec<(Color, Color)> = Vec::with_capacity(RACK_SIZE);
        for &slot in &self.racks[player] {
            if let Some(tile) = slot {
                if types.last() != Some(&tile) {
                    types.push(tile);
                }
            }
        }

        let first_move = !self.has_moved[player];

        for &cell in &g.valid {
            if self.occupied(cell) {
                continue;
            }
            for &dir in &CANONICAL_DIRS {
                let nb = g.neighbors[cell as usize][dir];
                if nb == NO_NEIGHBOR || self.occupied(nb) {
                    continue;
                }
                if first_move
                    && self.adjacent_unclaimed_symbol(cell).is_none()
                    && self.adjacent_unclaimed_symbol(nb).is_none()
                {
                    continue;
                }
                for &(a, b) in &types {
                    actions.push(Action::Place(PlaceMove {
                        cell,
                        dir: dir as u8,
                        color_a: a,
                        color_b: b,
                    }));
                    if a != b {
                        actions.push(Action::Place(PlaceMove {
                            cell,
                            dir: dir as u8,
                            color_a: b,
                            color_b: a,
                        }));
                    }
                }
            }
        }
    }

    fn swap_eligible(&self, player: usize) -> bool {
        let scores = self.score[player];
        let min = *scores.iter().min().unwrap();
        let mut low = [false; NUM_COLORS];
        for i in 0..NUM_COLORS {
            low[i] = scores[i] == min;
        }
        !self.racks[player]
            .iter()
            .flatten()
            .any(|&(a, b)| low[a.index()] || low[b.index()])
    }

    fn generate_swap_actions(&self, actions: &mut Vec<Action>) {
        actions.push(Action::KeepRack);
        if self.swap_eligible(self.current_player) {
            actions.push(Action::Swap);
        }
    }

    fn apply_place(&mut self, mv: &PlaceMove) {
        let g = geometry::<P>();
        let nb = g.neighbors[mv.cell as usize][mv.dir as usize];
        debug_assert_ne!(nb, NO_NEIGHBOR);
        let player = self.current_player;

        let first_move = !self.has_moved[player];
        let claimed_by_this_move = if first_move {
            self.adjacent_unclaimed_symbol(mv.cell)
                .or_else(|| self.adjacent_unclaimed_symbol(nb))
        } else {
            None
        };

        self.board[mv.cell as usize] = Some(mv.color_a);
        self.board[nb as usize] = Some(mv.color_b);

        let placed_type = normalize(mv.color_a, mv.color_b);
        self.board_tile_counts[placed_type.0.index()][placed_type.1.index()] += 1;

        for slot in &mut self.racks[player] {
            if *slot == Some(placed_type) {
                *slot = None;
                break;
            }
        }
        sort_rack(&mut self.racks[player]);

        let mut gained = [0u32; NUM_COLORS];
        gained[mv.color_a.index()] += self.run_score(mv.cell, mv.dir as usize, mv.color_a);
        gained[mv.color_b.index()] += self.run_score(nb, opposite(mv.dir as usize), mv.color_b);
        let new_bonus = self.apply_scoring(player, gained);

        if first_move {
            if let Some(color_idx) = claimed_by_this_move {
                self.claimed_symbols[color_idx] = true;
            }
            self.has_moved[player] = true;
        }

        if self.score[player] == [TARGET_SCORE; NUM_COLORS] {
            self.winner_immediate = Some(player);
        }

        self.pending_bonus += new_bonus;
        if self.pending_bonus > 0
            && self.rack_has_any_tile(player)
            && self.board_has_legal_placement()
        {
            self.pending_bonus -= 1;
            self.phase = Phase::Place;
        } else {
            self.pending_bonus = 0;
            self.phase = Phase::SwapDecision;
        }
    }

    fn end_turn(&mut self) {
        self.current_player = (self.current_player + 1) % P;
        self.phase = Phase::Place;
        self.pending_bonus = 0;
    }

    fn apply_action(&mut self, action: &Action) {
        match action {
            Action::Place(mv) => self.apply_place(mv),
            Action::KeepRack => {
                self.refill_rack(self.current_player);
                self.end_turn();
            }
            Action::Swap => {
                let player = self.current_player;
                self.racks[player] = [None; RACK_SIZE];
                self.refill_rack(player);
                self.end_turn();
            }
        }
    }

    /// Compares every player's score vector sorted ascending (lowest color
    /// first); the highest such vector wins outright, and a tie for the
    /// highest is a draw.
    fn compute_winner(&self) -> Option<Player> {
        if let Some(p) = self.winner_immediate {
            return Some(Player::from_index(p));
        }
        let mut sorted = self.score;
        for s in &mut sorted {
            s.sort_unstable();
        }
        let best = *sorted.iter().max().unwrap();
        let mut winner = None;
        for (i, s) in sorted.iter().enumerate() {
            if *s == best {
                if winner.is_some() {
                    return None;
                }
                winner = Some(i);
            }
        }
        winner.map(Player::from_index)
    }

    fn compute_hash(&self) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        let mix = |h: u64, v: u64| -> u64 { (h ^ v).wrapping_mul(0x0100_0000_01B3) };
        for (i, cell) in self.board.iter().enumerate() {
            if let Some(c) = cell {
                h = mix(h, (i as u64) << 8 | c.index() as u64);
            }
        }
        for (p, rack) in self.racks.iter().enumerate() {
            for slot in rack.iter().flatten() {
                h = mix(
                    h,
                    0x1000 | (p as u64) << 8 | (slot.0.index() as u64) << 4 | slot.1.index() as u64,
                );
            }
        }
        for (p, scores) in self.score.iter().enumerate() {
            for (c, &s) in scores.iter().enumerate() {
                h = mix(h, 0x2000 | (p as u64) << 8 | (c as u64) << 4 | s as u64);
            }
        }
        h = mix(h, 0x3000 | self.current_player as u64);
        h = mix(h, 0x4000 | matches!(self.phase, Phase::SwapDecision) as u64);
        h = mix(h, 0x5000 | self.pending_bonus as u64);
        h
    }

    /// Hash of everything every player can see: the board, all scores,
    /// claimed symbols, whose turn it is, and the turn phase -- but not any
    /// rack. Two states that differ only in who holds which tiles are the
    /// same information set from an onlooker's perspective and hash equal
    /// here, unlike `compute_hash`, which folds racks in and so treats them
    /// as different states.
    pub fn public_hash(&self) -> u64 {
        let mut h: u64 = 0xD1B5_4A32_D192_ED03;
        let mix = |h: u64, v: u64| -> u64 { (h ^ v).wrapping_mul(0x0100_0000_01B3) };
        for (i, cell) in self.board.iter().enumerate() {
            if let Some(c) = cell {
                h = mix(h, (i as u64) << 8 | c.index() as u64);
            }
        }
        for (p, scores) in self.score.iter().enumerate() {
            for (c, &s) in scores.iter().enumerate() {
                h = mix(h, 0x2000 | (p as u64) << 8 | (c as u64) << 4 | s as u64);
            }
        }
        for (c, &claimed) in self.claimed_symbols.iter().enumerate() {
            h = mix(h, 0x6000 | (c as u64) << 4 | claimed as u64);
        }
        h = mix(h, 0x3000 | self.current_player as u64);
        h = mix(h, 0x4000 | matches!(self.phase, Phase::SwapDecision) as u64);
        h = mix(h, 0x5000 | self.pending_bonus as u64);
        h
    }

    /// Resamples every rack except `current_player`'s from the pool of
    /// tiles that player can't already see: every tile not on the board and
    /// not in their own rack, which mixes together the bag and every other
    /// player's rack contents (indistinguishable to the mover, since tiles
    /// are anonymous by color pair rather than individually tracked). Each
    /// opponent keeps their current rack size -- hand size is public even
    /// though contents aren't -- and only which tiles fill those slots
    /// changes.
    fn redeal_hidden_racks(&mut self, rng: &mut SmallRng) {
        let observer = self.current_player;

        let mut pool: Vec<(Color, Color)> = Vec::new();
        for i in 0..NUM_COLORS {
            for j in i..NUM_COLORS {
                let a = Color::ALL[i];
                let b = Color::ALL[j];
                let mut remaining = tile_type_total(a, b) - self.board_tile_counts[i][j];
                for slot in self.racks[observer].iter().flatten() {
                    if *slot == (a, b) {
                        remaining -= 1;
                    }
                }
                for _ in 0..remaining {
                    pool.push((a, b));
                }
            }
        }
        pool.shuffle(rng);

        let mut drawn = 0;
        for p in 0..P {
            if p == observer {
                continue;
            }
            let rack_len = self.racks[p].iter().flatten().count();
            self.racks[p] = [None; RACK_SIZE];
            for slot in self.racks[p].iter_mut().take(rack_len) {
                *slot = pool.get(drawn).copied();
                drawn += 1;
            }
            sort_rack(&mut self.racks[p]);
        }
    }
}

/// Sorts a rack so identical tile types cluster (`None`s last), independent
/// of draw/removal order -- keeps two racks holding the same multiset of
/// tiles structurally `Eq`, and lets `generate_place_actions` dedupe
/// same-type tiles with a cheap adjacent-elements check.
fn sort_rack(rack: &mut [Option<(Color, Color)>; RACK_SIZE]) {
    rack.sort_unstable_by_key(|slot| match slot {
        Some((a, b)) => (0u8, a.index(), b.index()),
        None => (1u8, 0, 0),
    });
}

////////////////////////////////////////////////////////////////////////////////////////

impl<const P: usize> RectangularBoard for State<P> {
    const NUM_DISPLAY_ROWS: usize = SIDE;
    const NUM_DISPLAY_COLS: usize = SIDE;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        let idx = row * SIDE + col;
        if !geometry::<P>().valid_mask[idx] {
            ' '
        } else {
            match self.board[idx] {
                Some(c) => c.char(),
                None => '.',
            }
        }
    }
}

impl<const P: usize> fmt::Display for State<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)?;
        writeln!(
            f,
            "player {} to move, phase {:?}, pending_bonus {}",
            self.current_player, self.phase, self.pending_bonus
        )?;
        for p in 0..P {
            writeln!(f, "P{} score: {:?}", p, self.score[p])?;
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone)]
pub struct Ingenious<const P: usize>;

/// The 2-player game as it ships today: radius-5 board (91 cells).
pub type Ingenious2 = Ingenious<2>;
/// The 3-player game: radius-6 board (127 cells), one more rack and one more
/// opening symbol claimed than the 2-player game.
pub type Ingenious3 = Ingenious<3>;

impl<const P: usize> Game for Ingenious<P> {
    type S = State<P>;
    type A = Action;
    type P = Player;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        match state.phase {
            Phase::Place => state.generate_place_actions(actions),
            Phase::SwapDecision => state.generate_swap_actions(actions),
        }
    }

    fn apply(mut state: Self::S, action: &Self::A) -> Self::S {
        state.apply_action(action);
        state
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.winner_immediate.is_some() || !state.board_has_legal_placement()
    }

    fn winner(state: &Self::S) -> Option<Self::P> {
        state.compute_winner()
    }

    fn player_to_move(state: &Self::S) -> Self::P {
        Player::from_index(state.current_player)
    }

    fn num_players() -> usize {
        P
    }

    fn is_stochastic() -> bool {
        true
    }

    fn has_hidden_information() -> bool {
        true
    }

    fn alternating_moves() -> bool {
        false
    }

    /// Resamples the tile-draw stream and every rack but `current_player`'s
    /// own, so a search rooted at their move only ever sees information they
    /// actually have: their own tiles for certain, everyone else's as one
    /// consistent guess.
    fn determinize(mut state: Self::S, rng: &mut SmallRng) -> Self::S {
        state.rng = rng.gen();
        state.redeal_hidden_racks(rng);
        state
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.compute_hash()
    }

    fn notation(_state: &Self::S, action: &Self::A) -> String {
        match action {
            Action::Place(mv) => format!(
                "Place({},{} dir={} {:?}/{:?})",
                mv.cell % SIDE as u8,
                mv.cell / SIDE as u8,
                mv.dir,
                mv.color_a,
                mv.color_b
            ),
            Action::KeepRack => "KeepRack".into(),
            Action::Swap => "Swap".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::strategies::{
        mcts::{strategy, SearchConfig, TreeSearch},
        Search,
    };
    use mcts::util::random_play;
    use rand::SeedableRng;

    #[test]
    fn geometry_has_91_valid_cells_for_2p_and_127_for_3p() {
        assert_eq!(geometry::<2>().valid.len(), 91);
        assert_eq!(geometry::<3>().valid.len(), 127);
    }

    #[test]
    fn symbol_cells_are_distinct_and_at_radius_3_for_every_player_count() {
        for &cells in &[geometry::<2>(), geometry::<3>()] {
            let mut seen = std::collections::HashSet::new();
            for &cell in &cells.symbol_cell {
                assert!(seen.insert(cell), "duplicate symbol cell {cell}");
                let row = (cell as usize) / SIDE;
                let col = (cell as usize) % SIDE;
                assert_eq!(hex_distance(row as i32, col as i32), 3);
            }
        }
        // Player count only changes which outer ring is unlocked -- the
        // opening symbols themselves sit well inside every supported board.
        assert_eq!(geometry::<2>().symbol_cell, geometry::<3>().symbol_cell);
    }

    #[test]
    fn each_valid_cell_has_at_least_two_neighbors() {
        for g in [geometry::<2>(), geometry::<3>()] {
            for &cell in &g.valid {
                let n = g.neighbors[cell as usize]
                    .iter()
                    .filter(|&&n| n != NO_NEIGHBOR)
                    .count();
                assert!(n >= 2, "cell {cell} has only {n} neighbors");
            }
        }
    }

    #[test]
    fn initial_state_deals_full_racks_and_seeds_symbols() {
        let two = State::<2>::new(1);
        for p in 0..2 {
            assert_eq!(two.racks[p].iter().flatten().count(), RACK_SIZE);
        }

        let three = State::<3>::new(1);
        for p in 0..3 {
            assert_eq!(three.racks[p].iter().flatten().count(), RACK_SIZE);
        }

        let g = geometry::<3>();
        for (i, &cell) in g.symbol_cell.iter().enumerate() {
            assert_eq!(three.board[cell as usize], Some(Color::ALL[i]));
        }
    }

    // Every tile is always accounted for exactly once: on the board, in a
    // rack, or in the (implicit) bag -- total 120 across every type,
    // regardless of player count.
    fn check_tile_supply_is_conserved<const P: usize>(seed: u64) {
        let state = State::<P>::new(seed);
        let mut total = 0u32;
        for i in 0..NUM_COLORS {
            for j in i..NUM_COLORS {
                let a = Color::ALL[i];
                let b = Color::ALL[j];
                let cap = tile_type_total(a, b) as u32;
                assert!(state.in_play_count(a, b) as u32 <= cap);
                total += cap;
            }
        }
        assert_eq!(total, 120);
    }

    #[test]
    fn tile_supply_is_conserved() {
        check_tile_supply_is_conserved::<2>(2);
        check_tile_supply_is_conserved::<3>(2);
    }

    #[test]
    fn determinize_keeps_mover_rack_and_own_tile_supply_fixed() {
        let mut state = State::<3>::new(9);
        // Take a few turns off the initial deal so racks aren't full.
        state.racks[1][0] = None;
        state.racks[1][1] = None;
        state.racks[2][0] = None;

        let mover_rack_before = state.racks[0];
        let opponent_rack_lens: Vec<usize> = (1..3)
            .map(|p| state.racks[p].iter().flatten().count())
            .collect();

        let mut rng = SmallRng::seed_from_u64(1);
        let determinized = Ingenious::<3>::determinize(state, &mut rng);

        assert_eq!(determinized.racks[0], mover_rack_before);
        for (i, p) in (1..3).enumerate() {
            assert_eq!(
                determinized.racks[p].iter().flatten().count(),
                opponent_rack_lens[i]
            );
        }

        // Every tile is still accounted for exactly once, same as any other
        // reachable state.
        let mut total = 0u32;
        for i in 0..NUM_COLORS {
            for j in i..NUM_COLORS {
                let a = Color::ALL[i];
                let b = Color::ALL[j];
                let cap = tile_type_total(a, b) as u32;
                assert!(determinized.in_play_count(a, b) as u32 <= cap);
                total += cap;
            }
        }
        assert_eq!(total, 120);
    }

    #[test]
    fn determinize_resamples_opponent_racks_across_repeated_calls() {
        let state = State::<2>::new(9);
        let mut rng = SmallRng::seed_from_u64(1);
        let mut saw_a_change = false;
        for _ in 0..20 {
            let determinized = Ingenious::<2>::determinize(state, &mut rng);
            if determinized.racks[1] != state.racks[1] {
                saw_a_change = true;
                break;
            }
        }
        assert!(saw_a_change, "determinize never changed the hidden rack");
    }

    #[test]
    fn public_hash_ignores_racks_but_distinguishes_public_state() {
        let mut state = State::<2>::new(9);
        let with_original_racks = state.public_hash();

        let mut rng = SmallRng::seed_from_u64(1);
        let redealt = Ingenious::<2>::determinize(state, &mut rng);
        assert_eq!(with_original_racks, redealt.public_hash());

        state.current_player = 1;
        assert_ne!(with_original_racks, state.public_hash());
    }

    #[test]
    fn straight_line_run_scores_one_point_per_matching_hex() {
        // Hand-built board (bypassing move legality/turn order, per this
        // repo's guidance for isolating scoring math): three red hexes laid
        // out east of the board's center, then scored as if `center` were a
        // freshly-placed red tile-half whose sibling sits to the west (so
        // the run is scored purely eastward). The board's shared coordinate
        // system means this is identical for every player count.
        let mut state = State::<2>::new(3);
        let g = geometry::<2>();
        let center = (CENTER as usize * SIDE + CENTER as usize) as u8;
        let mut cell = center;
        for _ in 0..3 {
            let nb = g.neighbors[cell as usize][DIR_E];
            assert_ne!(nb, NO_NEIGHBOR);
            state.board[nb as usize] = Some(Color::Red);
            cell = nb;
        }
        let dir_w = opposite(DIR_E);
        let pts = state.run_score(center, dir_w, Color::Red);
        assert_eq!(pts, 3);
    }

    #[test]
    fn scoring_freezes_at_target_and_grants_one_bonus_on_crossing() {
        let mut state = State::<2>::new(4);
        state.score[0] = [10, 0, 0, 0, 0, 0];
        let bonus = state.apply_scoring(0, [20, 0, 0, 0, 0, 0]);
        assert_eq!(bonus, 1);
        assert_eq!(state.score[0][0], TARGET_SCORE);
        assert!(state.bonus_used[0][0]);

        // Further gains in that color are ignored now that it's frozen.
        let bonus2 = state.apply_scoring(0, [5, 0, 0, 0, 0, 0]);
        assert_eq!(bonus2, 0);
        assert_eq!(state.score[0][0], TARGET_SCORE);
    }

    #[test]
    fn winner_compares_sorted_score_vectors_lowest_first() {
        let mut state = State::<2>::new(5);
        state.score[0] = [5, 5, 5, 5, 5, 5];
        state.score[1] = [1, 9, 9, 9, 9, 9];
        assert_eq!(state.compute_winner(), Some(Player(0)));

        state.score[1] = [5, 5, 5, 5, 5, 6];
        assert_eq!(state.compute_winner(), Some(Player(1)));

        state.score[1] = [5, 5, 5, 5, 5, 5];
        assert_eq!(state.compute_winner(), None);
    }

    #[test]
    fn winner_is_n_way_among_three_players() {
        let mut state = State::<3>::new(5);
        state.score[0] = [5, 5, 5, 5, 5, 5];
        state.score[1] = [4, 9, 9, 9, 9, 9];
        state.score[2] = [3, 9, 9, 9, 9, 9];
        assert_eq!(state.compute_winner(), Some(Player(0)));

        // A tie between two of the three players, with the third strictly
        // behind both, is still a draw -- it isn't decided by the third
        // player's worse score.
        state.score[1] = [5, 5, 5, 5, 5, 5];
        assert_eq!(state.compute_winner(), None);
    }

    #[test]
    fn random_playouts_terminate() {
        random_play::<Ingenious2>();
        random_play::<Ingenious3>();
    }

    fn check_random_playout_invariants<const P: usize>(seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::<P>::new(rng.gen());
        let mut actions = Vec::new();
        for _ in 0..500 {
            if Ingenious::<P>::is_terminal(&state) {
                break;
            }
            actions.clear();
            Ingenious::<P>::generate_actions(&state, &mut actions);
            assert!(!actions.is_empty());
            let m = &actions[rng.gen_range(0..actions.len())];
            state = Ingenious::<P>::apply(state, m);

            for p in 0..P {
                for &s in &state.score[p] {
                    assert!(s <= TARGET_SCORE);
                }
            }
            let mut total = 0u32;
            for i in 0..NUM_COLORS {
                for j in i..NUM_COLORS {
                    let a = Color::ALL[i];
                    let b = Color::ALL[j];
                    total += state.in_play_count(a, b) as u32;
                }
            }
            assert!(total <= 120);
        }
    }

    #[test]
    fn random_playouts_never_exceed_target_score_and_conserve_tiles_2p() {
        for seed in 0..20 {
            check_random_playout_invariants::<2>(seed);
        }
    }

    #[test]
    fn random_playouts_never_exceed_target_score_and_conserve_tiles_3p() {
        for seed in 0..20 {
            check_random_playout_invariants::<3>(seed);
        }
    }

    // A short plain-UCT (no solver, no prior, no cutoff-evaluator) self-play
    // run against the 3-player board -- the search core takes the same
    // per-player-vector backprop path for any player count, so this is
    // mainly a smoke test that a 3-player `Ingenious` wires into that path
    // correctly end to end, not a strength benchmark.
    #[test]
    fn three_player_self_play_smoke() {
        let mut search: TreeSearch<Ingenious3, strategy::Ucb1> =
            TreeSearch::new().config(SearchConfig::new().max_iterations(64).seed(7));

        let mut state = State::<3>::new(7);
        for _ in 0..12 {
            if Ingenious3::is_terminal(&state) {
                break;
            }
            let action = search.choose_action(&state);
            state = Ingenious3::apply(state, &action);
        }
    }

    // ISMCTS (`SearchConfig::use_ismcts`) self-play against 2-player
    // Ingenious's real hidden racks: every iteration searches its own
    // `Ingenious::determinize`d guess at the opponent's rack, widening and
    // scoring the root's `ChildArray` against that per-iteration sample
    // instead of the one sample its first expansion happened to see. Checks
    // the plumbing actually runs end to end (growable root, real
    // availability counts accumulating, still-legal final choice) --
    // not a claim that any specific inner node's action set diverges across
    // iterations, which is `games/ingenious`'s own rack-legality logic to
    // exercise, not this search-engine wiring test's job.
    #[test]
    fn ismcts_self_play_stays_legal_and_tracks_availability() {
        let mut search: TreeSearch<Ingenious2, strategy::Ucb1> = TreeSearch::new().config(
            SearchConfig::new()
                .use_ismcts(true)
                .max_iterations(40)
                .seed(11),
        );

        let mut state = State::<2>::new(13);
        for _ in 0..4 {
            if Ingenious2::is_terminal(&state) {
                break;
            }
            let action = search.choose_action(&state);

            let mut legal = Vec::new();
            Ingenious2::generate_actions(&state, &mut legal);
            assert!(
                legal.contains(&action),
                "ISMCTS chose an action illegal against the real state"
            );

            let root = search.index.get(search.root_id);
            let children = root.children();
            assert!(children.is_growable());
            assert!(children.len() >= legal.len());
            // The root is visited by every iteration, so every one of its
            // legal children's availability should reflect that -- the
            // count `Ucb1::score_child` uses in place of a shared parent
            // visit count under ISMCTS is actually being written, not left
            // at its zero default.
            let root_idx = (0..children.len())
                .find(|&i| children.action(i) == action)
                .unwrap();
            assert!(children.availability(root_idx) > 0);

            state = Ingenious2::apply(state, &action);
        }
    }

    #[test]
    fn ismcts_redeterminize_self_play_stays_legal_and_tracks_availability() {
        let mut search: TreeSearch<Ingenious2, strategy::Ucb1> = TreeSearch::new().config(
            SearchConfig::new()
                .use_ismcts(true)
                .ismcts_redeterminize(true)
                .max_iterations(40)
                .seed(17),
        );

        let mut state = State::<2>::new(23);
        for _ in 0..4 {
            if Ingenious2::is_terminal(&state) {
                break;
            }
            let action = search.choose_action(&state);

            let mut legal = Vec::new();
            Ingenious2::generate_actions(&state, &mut legal);
            assert!(
                legal.contains(&action),
                "re-determinizing ISMCTS chose an action illegal against the real state"
            );

            let root = search.index.get(search.root_id);
            let children = root.children();
            assert!(children.is_growable());
            assert!(children.len() >= legal.len());
            let root_idx = (0..children.len())
                .find(|&i| children.action(i) == action)
                .unwrap();
            assert!(children.availability(root_idx) > 0);

            state = Ingenious2::apply(state, &action);
        }
    }
}
