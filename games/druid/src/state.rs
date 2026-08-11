//! The board `State` and every primitive that reads or mutates it: legality
//! tests, whole-turn move enumeration, connectivity (BFS `connection()` /
//! `connect_distance`), and rendering. This is the shared ground-truth layer
//! both the flat and move-split `Game` encodings drive -- they differ only in
//! how they expose actions (see `moves`), never in board semantics.

use std::collections::VecDeque;

use rustc_hash::FxHashSet as HashSet;
use serde::{Deserialize, Serialize};

use crate::types::{Hand, Orientation, Pending, Piece, Player, PlacedPiece, Pos, Size, Square};

/// A Druid position: whose turn, the stacked board, each player's remaining
/// hand, and (move-split only) which sub-action of the current turn is in
/// progress. The flat encoding leaves `pending` at `Pending::None`, which is
/// exactly what a real, between-turns position holds.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    pub player: Player,
    pub board: Vec<Square>,
    pub hand_black: Hand,
    pub hand_white: Hand,
    pub size: Size,
    #[serde(default)]
    pub pending: Pending,
}

// TODO:
//
// A move can be implemented as a u16 to support up to 128x128 board sizes:
//
// PlacedPiece: u16
// - orientation: 1  bit
// - piece_type:  1  bit
// - location:    14 bits (up to 128 * 128 = 16384)
//
// State has some optimal packings depending on the board size. Note that
// above 9x9 the board state no longer fits in a 64 byte cache line. For
// purposes of board state packing, we have to assume a max height. We will
// take log2(N*M). For example, a 10x10 board would have a max height of 7.

impl Default for State {
    fn default() -> Self {
        Self::new(crate::DEFAULT_SIZE)
    }
}

impl State {
    pub fn new(size: Size) -> Self {
        State {
            player: Player::Black,
            board: vec![
                Square {
                    height: 0,
                    piece: None,
                };
                size.area().into()
            ],
            hand_black: Hand::new(size),
            hand_white: Hand::new(size),
            size,
            pending: Pending::None,
        }
    }

    pub fn at(&self, i: usize) -> Option<Player> {
        self.board[i].piece
    }

    pub fn current_hand(&self) -> &Hand {
        self.hand(self.player)
    }

    pub(crate) fn hand(&self, color: Player) -> &Hand {
        match color {
            Player::Black => &self.hand_black,
            Player::White => &self.hand_white,
        }
    }

    pub(crate) fn deplete(&mut self, piece: Piece) {
        match self.player {
            Player::Black => match piece {
                Piece::Sarsen => self.hand_black.sarsens -= 1,
                Piece::Lintel(_) => self.hand_black.lintels -= 1,
            },
            Player::White => match piece {
                Piece::Sarsen => self.hand_white.sarsens -= 1,
                Piece::Lintel(_) => self.hand_white.lintels -= 1,
            },
        }
    }

    /// Whether `color` could legally place a sarsen at `i`, ignoring hand
    /// count (callers check that separately). Depends only on `i`'s current
    /// occupant, not height -- a sarsen can stack on top of any height, as
    /// long as the topmost piece there (if any) is already `color`'s.
    /// Single source of truth shared by `moves()` (ground truth, `self.player`)
    /// and `MoveCache` (incremental, arbitrary color) so the two can't drift
    /// apart.
    pub(crate) fn sarsen_legal_at(&self, i: usize, color: Player) -> bool {
        match self.at(i) {
            None => true,
            Some(p) => p == color,
        }
    }

    /// Whether `color` could legally place a lintel of `orientation` anchored
    /// at `i`, ignoring hand count (callers check that separately): the
    /// anchor's own 3 touched cells (`{i, i+d, i+2d}` for the orientation's
    /// delta `d`) must share `h[0] == h[2]` with `h[1] <= h[0]`, and exactly 2
    /// of the 3 must already be `color`. Returns the touched cells on
    /// success (`None` covers both "out of bounds" and "not legal") --
    /// callers that just want a bool can `.is_some()` it. Single source of
    /// truth shared by `moves()`, `lintel_candidates_for` and `MoveCache`;
    /// see `sarsen_legal_at`.
    pub(crate) fn lintel_legal_at(
        &self,
        i: usize,
        orientation: Orientation,
        color: Player,
    ) -> Option<[usize; 3]> {
        let (dx, dy) = orientation.delta();
        let Pos(x, y) = Pos::from(i, self.size);
        let c = [
            Pos(x, y),
            Pos(x + dx, y + dy),
            Pos(x + dx + dx, y + dy + dy),
        ];
        if c[2].0 >= self.size.w || c[2].1 >= self.size.h {
            return None;
        }
        let cells = c.map(|p| p.index(self.size.w));
        let h = cells.map(|i| self.board[i].height);
        if h[0] != h[2] || h[1] > h[0] {
            return None;
        }
        let (Some(p0), Some(p2)) = (self.at(cells[0]), self.at(cells[2])) else {
            return None;
        };
        let mut count = 0;
        (p0 == color).then(|| count += 1);
        (p2 == color).then(|| count += 1);
        if let Some(p1) = self.at(cells[1]) {
            if p1 == color && h[1] == h[0] {
                count += 1;
            }
        }
        (count == 2).then_some(cells)
    }

    /// Enumerate every whole-turn placement legal for the current player, as
    /// `PlacedPiece`s. `self.player` drives legality, and hand counts are
    /// checked here (the honest, from-scratch ground truth that `MoveCache`
    ///-backed `generate_actions` is tested against).
    pub fn moves(&self, moves: &mut Vec<PlacedPiece>) {
        let hand = self.current_hand();
        for i in 0..self.size.area() as usize {
            if hand.sarsens > 0 && self.sarsen_legal_at(i, self.player) {
                moves.push(PlacedPiece(Piece::Sarsen, i as u8));
            }
            if hand.lintels > 0 {
                for orientation in [Orientation::Horizontal, Orientation::Vertical] {
                    if self.lintel_legal_at(i, orientation, self.player).is_some() {
                        moves.push(PlacedPiece(Piece::Lintel(orientation), i as u8));
                    }
                }
            }
        }
    }

    /// Candidate lintel placements available to `color`, alongside their
    /// touched cells -- same legality as `lintel_legal_at`, generalized to an
    /// arbitrary color instead of `self.player` and ignoring `color`'s hand
    /// count (callers check that separately). Used by the playout heuristic
    /// to reason about the *opponent's* candidate moves, not just the
    /// mover's.
    pub(crate) fn lintel_candidates_for(&self, color: Player) -> Vec<(PlacedPiece, [usize; 3])> {
        let mut out = Vec::new();
        for i in 0..self.size.area() as usize {
            for orientation in [Orientation::Horizontal, Orientation::Vertical] {
                if let Some(cells) = self.lintel_legal_at(i, orientation, color) {
                    out.push((PlacedPiece(Piece::Lintel(orientation), i as u8), cells));
                }
            }
        }
        out
    }

    /// Board cell indices touched by `m` -- 1 for a sarsen, 3 for a lintel.
    /// Depends only on `size`/`m`, not on current board contents, so it's
    /// safe to call before *or* after `apply` mutates the board (used by
    /// both `apply` itself and the incremental Zobrist update in
    /// `game::apply_turn`, which needs the same cells' pre- and post-move
    /// values). Unused slots beyond the returned length are filled with `0`,
    /// always a valid index.
    pub(crate) fn move_cells(&self, m: PlacedPiece) -> ([usize; 3], usize) {
        match m.0 {
            Piece::Sarsen => ([m.1 as usize, 0, 0], 1),
            Piece::Lintel(orientation) => {
                let (dx, dy) = orientation.delta();
                let Pos(x, y) = Pos::from(m.1 as usize, self.size);
                let c = [
                    Pos(x, y),
                    Pos(x + dx, y + dy),
                    Pos(x + dx + dx, y + dy + dy),
                ];
                (c.map(|p| Pos::index(p, self.size.w)), 3)
            }
        }
    }

    /// Apply a whole-turn placement to the board: deplete `mover`'s hand,
    /// write the piece, and flip the turn. Does *not* touch `pending` (the
    /// caller, `game::apply_turn`, manages the pending hash transition and
    /// resets it to `None`) -- this only mutates the shared board half of the
    /// state.
    pub fn apply(&mut self, m: PlacedPiece) {
        self.deplete(m.0);
        let (cells, n) = self.move_cells(m);
        match m.0 {
            Piece::Sarsen => {
                let i = cells[0];
                let sq = &self.board[i];
                self.board[i] = Square {
                    height: sq.height + 1,
                    piece: Some(self.player),
                }
            }
            Piece::Lintel(_) => {
                let h = self.board[cells[0]].height + 1;
                cells[..n].iter().for_each(|&i| {
                    self.board[i] = Square {
                        height: h,
                        piece: Some(self.player),
                    }
                })
            }
        }
        self.player.next();
    }

    fn get_adjacent(&self, pos: Pos, seen: &mut HashSet<usize>, color: Player) -> Vec<usize> {
        pos.adjacent(self.size)
            .map(|x| Pos::index(x, self.size.w))
            .filter(|x| self.board[*x].matches(color))
            // `insert` returns `true` only for a value not already present, so this both
            // filters out already-seen cells and marks newly-enqueued ones as seen.
            .filter(|x| seen.insert(*x))
            .collect()
    }

    fn bfs(
        &self,
        start: &Pos,
        goal: &HashSet<usize>,
        seen: &mut HashSet<usize>,
        color: Player,
    ) -> bool {
        let start_idx = start.index(self.size.w);
        if seen.contains(&start_idx) || !self.board[start_idx].matches(color) {
            return false;
        }
        seen.insert(start_idx);

        let mut frontier = VecDeque::from(vec![start_idx]);

        while let Some(idx) = frontier.pop_front() {
            if goal.contains(&idx) {
                return true;
            }

            frontier.extend(self.get_adjacent(Pos::from(idx, self.size), seen, color));
        }
        false
    }

    pub fn check_connection(&self, start: Vec<Pos>, end: Vec<Pos>, color: Player) -> bool {
        let goal = HashSet::from(
            end.into_iter()
                .map(|x| Pos::index(x, self.size.w))
                .collect(),
        );
        let mut seen = HashSet::default();
        start
            .iter()
            .any(|pos| self.bfs(pos, &goal, &mut seen, color))
    }

    pub fn connection(&self) -> Option<Player> {
        let (top, bottom): (Vec<Pos>, Vec<Pos>) = (0..self.size.w)
            .map(|x| (Pos(x, 0), Pos(x, self.size.h - 1)))
            .unzip();
        if self.check_connection(top, bottom, Player::Black) {
            return Some(Player::Black);
        }

        let (left, right): (Vec<Pos>, Vec<Pos>) = (0..self.size.h)
            .map(|y| (Pos(0, y), Pos(self.size.w - 1, y)))
            .unzip();
        if self.check_connection(left, right, Player::White) {
            return Some(Player::White);
        }

        None
    }

    /// Shortest border-to-border path `color` still needs to build, counted
    /// in cells that aren't already `color`'s. Used only as a heuristic for
    /// non-terminal (depth-cutoff) playouts -- see `game::compute_utilities`
    /// -- so it deliberately approximates: it charges a flat cost of one per
    /// cell regardless of piece type (a lintel covers 3 cells per hand item,
    /// a sarsen covers one) and ignores height/legality entirely. A cell
    /// already owned by the *opponent* still costs only one, not infinity,
    /// since a lintel's legality only requires 2 of its 3 touched cells to
    /// already be the mover's color (`moves()` above), so the third can
    /// repaint an opponent's cell -- there's no such thing as a permanently
    /// blocked cell here.
    ///
    /// 0-1 BFS (a plain BFS `VecDeque`, front-pushing 0-cost relaxations and
    /// back-pushing 1-cost ones) rather than Dijkstra, since every edge cost
    /// is 0 or 1. Every cell has a finite cost (no impassable cells), so on
    /// a non-empty board this always finds a path -- `unwrap_or(u32::MAX)`
    /// is unreachable in practice, just a safe default.
    pub(crate) fn connect_distance(&self, color: Player) -> u32 {
        let cost = |i: usize| -> u32 {
            if self.board[i].matches(color) {
                0
            } else {
                1
            }
        };

        let area = self.size.area() as usize;
        let mut dist = vec![u32::MAX; area];
        let mut done = vec![false; area];
        let mut deque: VecDeque<usize> = VecDeque::new();

        let starts: Vec<Pos> = match color {
            Player::Black => (0..self.size.w).map(|x| Pos(x, 0)).collect(),
            Player::White => (0..self.size.h).map(|y| Pos(0, y)).collect(),
        };
        for pos in starts {
            let i = pos.index(self.size.w);
            let c = cost(i);
            if c < dist[i] {
                dist[i] = c;
                if c == 0 {
                    deque.push_front(i);
                } else {
                    deque.push_back(i);
                }
            }
        }

        while let Some(i) = deque.pop_front() {
            if done[i] {
                continue;
            }
            done[i] = true;
            let d = dist[i];
            for adj in Pos::from(i, self.size).adjacent(self.size) {
                let j = adj.index(self.size.w);
                if done[j] {
                    continue;
                }
                let step = cost(j);
                let nd = d + step;
                if nd < dist[j] {
                    dist[j] = nd;
                    if step == 0 {
                        deque.push_front(j);
                    } else {
                        deque.push_back(j);
                    }
                }
            }
        }

        let goals: Vec<Pos> = match color {
            Player::Black => (0..self.size.w).map(|x| Pos(x, self.size.h - 1)).collect(),
            Player::White => (0..self.size.h).map(|y| Pos(self.size.w - 1, y)).collect(),
        };
        goals
            .into_iter()
            .map(|pos| dist[pos.index(self.size.w)])
            .min()
            .unwrap_or(u32::MAX)
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let color_map = generate_map(self.size, |i| match self.board[i].piece {
            None => " .".into(),
            Some(Player::Black) => " X".into(),
            Some(Player::White) => " O".into(),
        });
        let height_map = generate_map(self.size, |i| match self.board[i].height {
            0 => " .".into(),
            n => format!(" {:x}", n),
        });

        // Combine color_map and height_map side by side
        writeln!(f)?;
        let color_lines: Vec<&str> = color_map.split('\n').collect();
        let height_lines: Vec<&str> = height_map.split('\n').collect();
        for (color_line, height_line) in color_lines.iter().zip(height_lines.iter()) {
            writeln!(f, "{}   {}", color_line, height_line,)?;
        }

        Ok(())
    }
}

fn generate_map<F>(size: Size, mut func: F) -> String
where
    F: FnMut(usize) -> String,
{
    let mut map = Vec::new();

    let column_labels = |map: &mut Vec<String>| {
        for c in ('A'..).take(size.w as usize) {
            map.push(format!(" {}", c));
        }
    };

    // Generate map
    map.push("   ".to_string());
    column_labels(&mut map);
    let mut row = size.h as usize;
    map.push(format!("   \n{:>3}", row));
    for i in 0..size.area() as usize {
        let c = func(i);
        map.push(c);
        if ((i + 1) as u8).is_multiple_of(size.w) {
            map.push(format!(" {}", row));
            if row < 10 {
                map.push(" ".into());
            }
            row -= 1;
            if row != 0 {
                map.push(format!("\n{:>3}", row));
            }
        }
    }
    map.push("\n   ".into());
    column_labels(&mut map);
    map.push("   ".into());
    map.join("")
}